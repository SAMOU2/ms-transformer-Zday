"""
============================================================
 01_data_preparation.py — Préparation multimodale CICIDS2017 (v3.12)
============================================================
Version « streaming » : conçue pour tenir dans 2 Go de RAM.

Pourquoi : charger les 8 parquet d'un coup demande ~2,5 Go (2,8 M lignes × 79
colonnes en float64) et fait tomber les machines modestes. Ici on ne garde
jamais plus d'un fichier en mémoire, et on n'échantillonne que ce dont on a
besoin (≈85 k lignes au lieu de 2,8 M).

Deux passes :
  1. SCAN   : lecture de la seule colonne Label -> comptage par fichier
  2. ÉCHANT : relecture ciblée, tirage des quotas par (fichier, classe)

Garanties de non-fuite :
  • BENIGN train et BENIGN test sont tirés de façon DISJOINTE (aucun flux
    partagé entre les deux) — CICIDS2017 contient énormément de doublons,
    c'est la première cause d'AUROC artificiellement parfait.
  • Scaler fitté sur BENIGN train uniquement.
  • Les classes Zero-Day n'apparaissent JAMAIS dans l'échantillon d'entraînement.

Usage :
    python 01_data_preparation.py
    python 01_data_preparation.py --synthetic     # pipeline de test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
OUT_DIR = "data/processed"
MODEL_DIR = "models"

ZERO_DAY_ATTACKS = ["PortScan", "Web Attack", "Bot"]
VERSION = "3.12"

BENIGN_TRAIN = 40_000
BENIGN_TEST = 25_000
ATTACK_QUOTA_TRAIN = 5_000     # par famille d'attaque connue
ZERO_DAY_QUOTA_TEST = 20_000   # par famille Zero-Day
SEED = 42

SYSCALL_VOCAB = ["open", "read", "close", "socket", "connect", "send", "recv", "mmap",
                 "write", "fsync", "flood", "hulk", "scan", "auth", "fail", "bruteforce",
                 "execve", "chmod", "http", "injection", "irc", "beacon", "ptrace",
                 "exfiltration", "unknown"]

EDR_PATTERNS = [
    ("BENIGN", "open-read-close"),
    ("BENIGN", "socket-connect-send-recv"),
    ("BENIGN", "open-mmap-write-fsync"),
    ("DDOS", "socket-connect-flood-flood-flood"),
    ("DOS", "socket-send-flood-hulk-no-delay"),
    ("PORTSCAN", "socket-connect-close-scan-scan"),
    ("PATATOR", "open-auth-fail-bruteforce-fail"),
    ("WEB", "open-execve-chmod-socket-http-injection"),
    ("BOT", "socket-connect-irc-beacon-keepalive"),
    ("INFILTRATION", "open-ptrace-execve-socket-exfiltration"),
    ("HEARTBLEED", "socket-connect-read-recv-unknown"),
]


def normalize_label(s: str) -> str:
    """Les CSV CICIDS2017 contiennent 'Web Attack � XSS' (mojibake latin-1)."""
    s = str(s).replace("\ufffd", "-").replace("–", "-").replace("—", "-")
    return " ".join(s.replace("\xa0", " ").split()).strip()


def canon_label(s: str) -> str:
    """
    Ramène tous les libellés bénins à la forme canonique « BENIGN ».

    Les jeux CICIDS2017 circulent avec plusieurs orthographes : « BENIGN »,
    « Benign », « benign ». Le reste du code compare à la chaîne exacte
    « BENIGN » : sans canonisation, un fichier libellé « Benign » voyait sa
    classe bénigne traitée comme une ATTAQUE (quota de 5 000, placement dans
    le train) et les réservoirs bénins restaient vides.
    """
    s = normalize_label(s)
    return "BENIGN" if "BENIGN" in s.upper() else s


def edr_sequence(label: str) -> str:
    up = label.upper()
    if "BENIGN" in up:
        return "open-read-close"
    for key, seq in EDR_PATTERNS:
        if key != "BENIGN" and key in up:
            return seq
    if "BRUTE" in up or "FTP" in up or "SSH" in up:
        return "open-auth-fail-bruteforce-fail"
    return "open-read-close-unknown"


def proxy_edr(X_net: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Branche EDR SIMULÉE À PARTIR DU TRAFIC, jamais du label.

    Pourquoi ce changement : la version précédente générait la séquence système
    à partir du nom de la classe (`if "BOT" in label -> irc-beacon`). Le modèle
    pouvait donc lire la réponse dans les features EDR : d'où un AUROC de 1.0000
    qui ne mesurait rien. L'ablation « sans branche EDR » est tombée à 0.7285,
    ce qui a révélé la fuite.

    Ici la séquence est une fonction déterministe (plus un bruit contrôlé) des
    features réseau observables : durée, volumes, tailles de paquets, port.
    L'information EDR est donc une VUE du trafic, pas une étiquette déguisée.
    C'est la définition d'une vue augmentée en apprentissage contrastif.

    Conséquence assumée à écrire dans le mémoire : sans collecte EDR réelle,
    cette branche n'apporte pas d'information nouvelle — elle teste la capacité
    du modèle à fusionner deux vues bruitées d'un même flux.
    """
    rng = np.random.default_rng(seed)
    n, d = X_net.shape
    X = np.nan_to_num(X_net.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    idx = [0, 1, 2, 3, 4, 5] if d >= 6 else list(range(d))
    cols = []
    for j in idx:
        c = X[:, j]
        lo, hi = np.percentile(c, 1), np.percentile(c, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            cols.append(np.zeros(n))
        else:
            cols.append(np.clip((c - lo) / (hi - lo), 0, 1))

    out = np.zeros((n, len(SYSCALL_VOCAB)), dtype=np.float32)
    for j, c in enumerate(cols):
        # quantifie en 5 classes -> contributions réparties sur le vocabulaire
        bins = np.minimum((c * 5).astype(int), 4)
        for b in range(5):
            mask = bins == b
            if not mask.any():
                continue
            out[mask, (j * 3 + b * 2) % len(SYSCALL_VOCAB)] += 1.0
            out[mask, (j * 5 + b) % len(SYSCALL_VOCAB)] += 0.5
    out += rng.normal(0, 0.08, out.shape).astype(np.float32)
    return np.clip(out, 0, None)


def verifier_absence_de_fuite(X: np.ndarray, y_bin: np.ndarray, noms=None) -> list:
    """
    Garde-fou exécuté AVANT l'écriture des fichiers.

    Une colonne qui sépare presque parfaitement les classes à elle seule ne
    décrit pas le trafic : elle décrit l'étiquette. C'est exactement ce qui
    s'est produit avec la branche EDR, constante sur 100 % des flux bénins et
    non nulle sur 100 % des flux d'attaque (AUROC 1.0000 sans apprentissage).

    On refuse d'écrire un jeu qui contient une telle colonne.
    """
    from sklearn.metrics import roc_auc_score

    alertes = []
    for j in range(X.shape[1]):
        col = X[:, j]
        # Une colonne constante ou non finie ne peut RIEN fuiter : elle ne
        # porte aucune information. Elle doit être RETIRÉE, pas signalée
        # comme triche. (Bug corrigé : elle faisait échouer tout le pipeline
        # et expliquait un n_net à 77 au lieu de 78.)
        if not np.isfinite(col).all() or col.std() < 1e-9:
            continue
        a = float(roc_auc_score(y_bin, col))
        a = max(a, 1.0 - a)
        if a >= 0.99:
            nom = noms[j] if noms and j < len(noms) else f"col_{j}"
            alertes.append((j, nom, round(a, 4)))
    return alertes


def encode_syscall(seq: str) -> np.ndarray:
    return np.array([seq.count(w) for w in SYSCALL_VOCAB], dtype=np.float32)


# Correspondance nom de fichier -> classe.
# Utile pour les jeux "un fichier par classe" (ex. *-no-metadata.parquet) qui
# n'ont PAS de colonne Label : la classe est dans le nom du fichier.
# L'ordre compte : "ddos" doit être testé avant "dos", "benign" avant tout.
FILE_LABELS = [
    ("benign", "BENIGN"),
    ("botnet", "Bot"),
    ("bot", "Bot"),
    ("ddos", "DDoS"),
    ("dos", "DoS"),
    ("bruteforce", "Brute Force"),
    ("brute", "Brute Force"),
    ("infilteration", "Infiltration"),
    ("infiltration", "Infiltration"),
    ("portscan", "PortScan"),
    ("webattack", "Web Attack"),
    ("heartbleed", "Heartbleed"),
]


def label_from_filename(path: str):
    """Déduit la classe depuis le nom du fichier. None si non reconnu."""
    name = os.path.basename(path).lower()
    for key, lab in FILE_LABELS:
        if key in name:
            return lab
    return None


def dataset_mode(files):
    """Retourne ('column', None) si les fichiers ont une colonne Label,
    ('filename', {f: label}) si la classe est dans le nom, sinon lève une erreur."""
    import pyarrow.parquet as pq
    cols = [f.name for f in pq.read_schema(files[0])]
    if "Label" in cols:
        return "column", None
    mapping = {}
    for f in files:
        lab = label_from_filename(f)
        if lab is None:
            raise SystemExit(
                f"❌ Impossible de déduire la classe de « {os.path.basename(f)} »\n"
                f"   Aucune colonne 'Label' et le nom ne contient ni benign, botnet, ddos,\n"
                f"   dos, bruteforce, infiltration, portscan ni webattack.\n"
                f"   Renomme le fichier ou ajoute la classe dans le nom.")
        mapping[f] = lab
    return "filename", mapping


def is_zero_day(label: str) -> bool:
    return any(z.lower() in label.lower() for z in ZERO_DAY_ATTACKS)


# ------------------------------------------------------------------
# Passe 1 : comptage
# ------------------------------------------------------------------
def scan(files, mode="column", mapping=None):
    import pyarrow.parquet as pq
    if mode == "column":
        print("=== PASSE 1 : comptage (colonne Label) ===")
    else:
        print("=== PASSE 1 : comptage (classe déduite du nom de fichier) ===")
    counts = {}
    for f in files:
        n_rows = pq.ParquetFile(f).metadata.num_rows
        if mode == "column":
            lbl = pq.read_table(f, columns=["Label"]).column("Label").to_pandas()
            brut = lbl.map(normalize_label)
            # signale les libellés dont la casse n'est pas canonique
            variantes = {}
            for b, k in zip(brut, brut.map(canon_label)):
                if b != k:
                    variantes[b] = variantes.get(b, 0) + 1
            for b, n in variantes.items():
                print(f"      ↳ libellé « {b} » ramené à « {canon_label(b)} » ({n:,} lignes)")
            c = brut.map(canon_label).value_counts().to_dict()
        else:
            c = {mapping[f]: n_rows}
        counts[f] = c
        print(f"  {os.path.basename(f)[:46]:<46} " + ", ".join(f"{k}={v:,}" for k, v in c.items()))
    return counts


# ------------------------------------------------------------------
# Plan d'échantillonnage (quotas par fichier × classe)
# ------------------------------------------------------------------
def build_plan(counts):
    """Répartit les quotas globaux entre fichiers, proportionnellement."""
    total_ben = sum(c.get("BENIGN", 0) for c in counts.values())
    attacks = {}
    for c in counts.values():
        for k, v in c.items():
            if k != "BENIGN":
                attacks[k] = attacks.get(k, 0) + v

    plan = {}
    for f, c in counts.items():
        p = {}
        ben = c.get("BENIGN", 0)
        if ben:
            share = ben / total_ben
            p["BENIGN_train"] = int(round(BENIGN_TRAIN * share))
            p["BENIGN_test"] = int(round(BENIGN_TEST * share))
        for k, v in c.items():
            if k == "BENIGN":
                continue
            if is_zero_day(k):
                quota = int(round(ZERO_DAY_QUOTA_TEST * v / attacks[k]))
                p[f"{k}|test"] = min(quota, v)
            else:
                quota = int(round(ATTACK_QUOTA_TRAIN * v / attacks[k]))
                p[f"{k}|train"] = min(quota, v)
        plan[f] = p
    return plan, attacks


# ------------------------------------------------------------------
# Passe 2 : échantillonnage par réservoir (mémoire constante)
# ------------------------------------------------------------------
def sample(files, plan, num_cols, mode="column", mapping=None, counts=None):
    """
    Lecture par batchs pyarrow + réservoir de Vitter.

    Un réservoir de taille k garantit un échantillon uniforme sans remise sur
    un flux de longueur inconnue, avec une mémoire bornée à k lignes. C'est ce
    qui permet de traiter un fichier de 692 000 lignes dans 2 Go de RAM.
    """
    print("\n=== PASSE 2 : échantillonnage par réservoir ===")
    import pyarrow.parquet as pq
    rng = np.random.default_rng(SEED)

    # quotas par (fichier, clé)  ->  tailles de réservoir
    quotas = {}   # (file, key) -> k
    for f, p in plan.items():
        for k, q in p.items():
            if q > 0:
                quotas[(f, k)] = int(q)
    res = {key: np.zeros((k, len(num_cols)), dtype=np.float32) for key, k in quotas.items()}
    seen = {key: 0 for key in quotas}
    accepted = set()          # empreintes des lignes déjà retenues (train ET test)
    ben_counter = {}          # répartition régulière des bénins uniques

    def offer(key, row_m, h):
        """
        Propose une ligne à un réservoir.

        Refuse la ligne si son empreinte a déjà été retenue ailleurs : c'est ce
        qui garantit qu'un même flux ne peut pas se retrouver à la fois dans le
        train et dans le test. La déduplication a posteriori ne suffisait pas —
        elle supprimait les bénins du test APRÈS que le train les ait absorbés,
        laissant un test sans BENIGN et donc un AUROC incalculable.
        """
        tkey = (f, key)                  # les réservoirs sont indexés par (fichier, clé)
        k = quotas.get(tkey, 0)
        if k <= 0 or h in accepted:
            return False
        n = seen[tkey]
        if n < k:
            res[tkey][n] = row_m
        else:
            j = rng.integers(0, n + 1)
            if j < k:
                res[tkey][j] = row_m
            else:
                return False
        accepted.add(h)
        seen[tkey] = n + 1
        return True

    for f in files:
        keys_here = {k.split("|")[0] if "|" in k else "BENIGN": k
                     for (ff, k) in quotas if ff == f}
        if not keys_here:
            continue
        pf = pq.ParquetFile(f)
        n_rows = 0
        read_cols = num_cols + (["Label"] if mode == "column" else [])
        for batch in pf.iter_batches(batch_size=50_000, columns=read_cols):
            if mode == "column":
                lab = pd.Series(batch.column(len(num_cols)).to_pandas()).map(canon_label).to_numpy()
            else:
                lab = np.array([mapping[f]] * batch.num_rows)
            # matrice float32 construite colonne par colonne (pas de copie float64)
            M = np.empty((batch.num_rows, len(num_cols)), dtype=np.float32)
            for j in range(len(num_cols)):
                col = batch.column(j).to_numpy(zero_copy_only=False)
                M[:, j] = np.nan_to_num(col.astype(np.float64, copy=False),
                                        nan=0.0, posinf=0.0, neginf=0.0)
            n_rows += batch.num_rows

            for base_label, key in keys_here.items():
                if base_label == "BENIGN":
                    # Répartition RÉGULIÈRE (et non aléatoire) des bénins uniques
                    # entre train et test : garantit que les deux réservoirs
                    # reçoivent des lignes même si le fichier est très redondant.
                    k_tr = quotas.get((f, "BENIGN_train"), 0)
                    k_te = quotas.get((f, "BENIGN_test"), 0)
                    tot = k_tr + k_te
                    if tot <= 0:
                        continue
                    idxs = np.where(lab == "BENIGN")[0]
                    # Ratio de répartition train/test, borné à [0.2, 0.8].
                    # Sans cette borne, un quota test faible face à un quota train
                    # énorme donne au test une fraction dérisoire des lignes
                    # uniques (ex. 50/40050 -> 0,37 ligne) et le test se retrouve
                    # vide dès que le fichier est peu varié.
                    if k_te <= 0:
                        ratio_tr = 1.0
                    elif k_tr <= 0:
                        ratio_tr = 0.0
                    else:
                        ratio_tr = min(0.8, max(0.2, k_tr / tot))
                    for i in idxs:
                        h = hash(M[i].tobytes())
                        if h in accepted:
                            continue
                        ntr = ben_counter.get((f, "train"), 0)
                        nte = ben_counter.get((f, "test"), 0)
                        ntot = ntr + nte
                        key = ("BENIGN_train"
                               if ntr < round((ntot + 1) * ratio_tr) else "BENIGN_test")
                        if offer(key, M[i], h):
                            if key == "BENIGN_train":
                                ben_counter[(f, "train")] = ntr + 1
                            else:
                                ben_counter[(f, "test")] = nte + 1
                else:
                    idxs = np.where(lab == base_label)[0]
                    for i in idxs:
                        offer(key, M[i], hash(M[i].tobytes()))
            del M, lab, batch
        print(f"  {os.path.basename(f)[:46]:<46} {n_rows:>9,} lignes lues")

    Xtr, ytr, Xte, yte = [], [], [], []
    for (f, key), k in quotas.items():
        n_fill = min(seen[(f, key)], k)
        if n_fill == 0:
            continue
        block = res[(f, key)][:n_fill]
        if key == "BENIGN_train":
            Xtr.append(block); ytr += ["BENIGN"] * n_fill
        elif key == "BENIGN_test":
            Xte.append(block); yte += ["BENIGN"] * n_fill
        else:
            label, split = key.rsplit("|", 1)
            if split == "train":
                Xtr.append(block); ytr += [label] * n_fill
            else:
                Xte.append(block); yte += [label] * n_fill
    Xtr_a, ytr_a = np.vstack(Xtr), np.array(ytr)
    Xte_a, yte_a = np.vstack(Xte), np.array(yte)

    # ------------------------------------------------------------------
    # FILET DE SÉCURITÉ
    # Si le réservoir de test n'a reçu aucun bénin (quota arrondi à zéro,
    # fichier très redondant, répartition défavorable...), on transfère 20 %
    # des bénins du train vers le test. Sans bénin dans le test, l'AUROC et
    # le FPR sont incalculables : mieux vaut un train légèrement plus petit
    # qu'une évaluation sans référence.
    # ------------------------------------------------------------------
    ben_tr = np.array(["BENIGN" in str(l).upper() for l in ytr_a])
    ben_te = np.array(["BENIGN" in str(l).upper() for l in yte_a])
    if ben_te.sum() == 0 and ben_tr.sum() >= 2:
        n_move = max(1, int(round(0.20 * ben_tr.sum())))
        pos = np.where(ben_tr)[0]
        mv = pos[:n_move]
        # ATTENTION : on conserve TOUTES les lignes sauf celles déplacées.
        # Écrire `keep = pos[n_move:]` ne gardait que les bénins restants et
        # supprimait silencieusement toutes les attaques du jeu d'entraînement.
        keep = np.setdiff1d(np.arange(len(ytr_a)), mv)
        Xte_a = np.vstack([Xte_a, Xtr_a[mv]])
        yte_a = np.concatenate([yte_a, np.array(["BENIGN"] * len(mv))])
        Xtr_a, ytr_a = Xtr_a[keep], ytr_a[keep]
        print("\n" + "~" * 70)
        print("  ⚠️  FILET DE SÉCURITÉ ACTIVÉ : le réservoir de test était vide")
        print(f"     {n_move:,} flux bénins déplacés du train vers le test.")
        print(f"     Train : {ben_tr.sum() - n_move:,} bénins · Test : {n_move:,} bénins")
        print("     (cela signifie que la répartition par quotas n'a pas fonctionné :")
        print("      signale-le, c'est un cas limite à documenter)")
        print("~" * 70)

    return (Xtr_a, ytr_a), (Xte_a, yte_a)


# ------------------------------------------------------------------
def synthetic():
    rng = np.random.default_rng(SEED)
    n_net = 78
    classes = ["BENIGN", "DDoS", "DoS Hulk", "FTP-Patator", "SSH-Patator",
               "PortScan", "Web Attack - XSS", "Web Attack - Sql Injection", "Bot", "Infiltration"]
    shift = {"BENIGN": 0.0, "DDoS": 1.8, "DoS Hulk": 1.5, "FTP-Patator": 1.1,
             "SSH-Patator": 1.0, "PortScan": 1.3, "Web Attack - XSS": 0.8,
             "Web Attack - Sql Injection": 0.9, "Bot": 0.45, "Infiltration": 0.6}
    Xs, ys = [], []
    for c in classes:
        n = 5000 if c == "BENIGN" else 900
        X = rng.normal(0, 1, (n, n_net)).astype(np.float32)
        X[:, :12] += shift[c] * 2.2
        X[:, 12:30] += shift[c] * 1.1
        Xs.append(X); ys += [c] * n
    X = np.vstack(Xs); y = np.array(ys)
    cols = [f"feat_{i}" for i in range(n_net)]
    ben = np.where(y == "BENIGN")[0]
    rng.shuffle(ben)
    tr_i = np.where(y != "BENIGN")[0].tolist() + ben[:3500].tolist()
    te_i = ben[3500:].tolist() + [i for i in np.where(y != "BENIGN")[0] if is_zero_day(y[i])]
    return (X[tr_i], y[tr_i]), (X[te_i], y[te_i]), cols


def main():
    global BENIGN_TRAIN, BENIGN_TEST
    print(f"01_data_preparation.py — version {VERSION}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--drop-constant", action="store_true",
                    help="retirer les colonnes constantes (format instable d'une "
                         "exécution à l'autre — déconseillé)")
    ap.add_argument("--benign-train", type=int, default=None,
                    help="quota de flux BÉNIN pour le train (défaut %d)" % BENIGN_TRAIN)
    ap.add_argument("--benign-test", type=int, default=None,
                    help="quota de flux BÉNIN pour le test (défaut %d)" % BENIGN_TEST)
    ap.add_argument("--edr-mode", choices=["proxy", "label", "off"], default="proxy",
                    help="proxy = dérivé du trafic (défaut, honnête) · label = fuite (archives)")
    args = ap.parse_args()

    # Ajustement des quotas bénins (lus par build_plan via les globales)
    if args.benign_train is not None:
        BENIGN_TRAIN = int(args.benign_train)
    if args.benign_test is not None:
        BENIGN_TEST = int(args.benign_test)
    print(f"Quotas BENIGN : train={BENIGN_TRAIN:,} · test={BENIGN_TEST:,}")
    np.random.seed(SEED)

    if args.synthetic:
        print("⚠️  MODE SYNTHÉTIQUE\n")
        (Xtr, ytr), (Xte, yte), cols = synthetic()
        synthetic_flag = True
    else:
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
        if not files:
            raise SystemExit(f"❌ aucun .parquet dans {DATA_DIR}/ (utilise --synthetic pour tester)")
        import pyarrow.parquet as pq
        mode, mapping = dataset_mode(files)
        if mode == "column":
            print("Schéma détecté : colonne 'Label' présente ✅")
        else:
            print("Schéma détecté : pas de colonne 'Label' → classe déduite du nom de fichier ✅")
            for f, lab in mapping.items():
                print(f"   {os.path.basename(f)[:52]:<54} → {lab}")

        cols = [f.name for f in pq.read_schema(files[0])]
        num_cols = [c for c in cols if c != "Label"]
        counts = scan(files, mode, mapping)
        plan, attacks = build_plan(counts)
        print("\nClasses connues (entraînement) :",
              ", ".join(k for k in attacks if not is_zero_day(k)))
        print("Zero-Day (jamais vus à l'entraînement) :",
              ", ".join(k for k in attacks if is_zero_day(k)))
        (Xtr, ytr), (Xte, yte) = sample(files, plan, num_cols, mode, mapping, counts)
        cols = num_cols
        synthetic_flag = False
        # contrôle de non-fuite
        leak = [v for v in set(k for c in counts.values() for k in c)
                if is_zero_day(v) and v in set(ytr)]
        print(f"\nContrôle de non-fuite : {len(leak)} classe(s) Zero-Day dans le train"
              f" {'✅' if not leak else '❌ ' + str(leak)}")

    # --- Déduplication sur les features RÉSEAU (avant la branche EDR) ---
    # Indispensable : CICIDS2017 contient des milliers de flux strictement
    # identiques. S'ils se retrouvent à la fois en train et en test, le modèle
    # les reconnaît par mémorisation et l'AUROC ne mesure plus rien.
    print("\n=== DÉDUPLICATION (features réseau, anti-fuite train/test) ===")
    n0 = len(Xtr)
    Xtr_u, ui = np.unique(Xtr, axis=0, return_index=True)
    ytr = ytr[ui]
    print(f"Train : {n0:,} → {len(Xtr_u):,} lignes réseau uniques")
    keys_tr = {r.tobytes() for r in Xtr_u}
    keep = np.array([r.tobytes() not in keys_tr for r in Xte])
    n1 = len(Xte)
    Xte, yte = Xte[keep], yte[keep]
    Xtr = Xtr_u
    print(f"Test  : {n1:,} → {len(Xte):,} ({n1-len(Xte):,} flux identiques au train retirés)")

    # --- Branche EDR ---
    if args.edr_mode == "proxy":
        print("\n=== BRANCHE EDR (proxy dérivé du trafic — SANS fuite de label) ===")
        S_tr = proxy_edr(Xtr, seed=SEED)
        S_te = proxy_edr(Xte, seed=SEED + 1)
        r_corr = np.corrcoef(S_tr[:, 0], ytr == "BENIGN")[0, 1]
        print(f"  25 dims · corrélation moyenne avec le label ≈ {abs(r_corr):.2f} (0 = aucune fuite)")
        print("  ⚠️  Sans collecte EDR réelle, cette branche est une vue augmentée du réseau.")
    else:
        print("\n=== BRANCHE EDR (simulation PAR LABEL — fuite assumée, non publiable) ===")
        S_tr = np.vstack([encode_syscall(edr_sequence(l)) for l in ytr])
        S_te = np.vstack([encode_syscall(edr_sequence(l)) for l in yte])
        print("  ❌ Cette branche encode le label : l'AUROC obtenu est un artefact.")

    if args.edr_mode == "off":
        S_tr = np.zeros((len(Xtr), len(SYSCALL_VOCAB)), dtype=np.float32)
        S_te = np.zeros((len(Xte), len(SYSCALL_VOCAB)), dtype=np.float32)
        print("\n=== BRANCHE EDR DÉSACTIVÉE (zéros) ===")

    X_train = np.hstack([Xtr, S_tr]).astype(np.float32)
    X_test = np.hstack([Xte, S_te]).astype(np.float32)
    y_train, y_test = np.asarray(ytr), np.asarray(yte)

    # --- Colonnes mortes -----------------------------------------------------
    # Une colonne constante (ou non finie) sur le train n'apprend rien et peut
    # être le résidu d'une capture incomplète. On la retire des DEUX jeux avec
    # le MÊME masque : sinon train et test ne décrivent plus les mêmes
    # grandeurs, et n_net ne correspond plus à la réalité du tenseur.
    # Par DÉFAUT on les CONSERVE. Les retirer rend la largeur du tenseur
    # dépendante du jeu produit : deux exécutions donnaient 102 puis 92
    # colonnes, et un modèle entraîné sur l'une refusait l'autre. Pour un
    # travail qu'on doit pouvoir rejouer des mois plus tard, la stabilité du
    # format passe avant le gain d'une dizaine de colonnes mortes.
    # `--drop-constant` rétablit la suppression si on la souhaite.
    _mortes = [j for j in range(X_train.shape[1])
               if (not np.isfinite(X_train[:, j]).all())
               or float(np.nanstd(X_train[:, j])) < 1e-9]
    if _mortes and args.drop_constant:
        _garde = np.array([j for j in range(X_train.shape[1])
                           if j not in set(_mortes)])
        X_train, X_test = X_train[:, _garde], X_test[:, _garde]
        print(f"\n🧹 {len(_mortes)} colonne(s) morte(s) retirée(s) "
              f"(indices {_mortes}) → {X_train.shape[1]} colonnes")
    elif _mortes:
        print(f"\nℹ️  {len(_mortes)} colonne(s) constante(s) conservée(s) "
              f"(format stable, {X_train.shape[1]} colonnes). "
              f"Utiliser --drop-constant pour les retirer.")

    n_sys = S_tr.shape[1]
    n_net = X_train.shape[1] - n_sys
    print(f"\nRéseau {n_net}D + EDR {n_sys}D = {n_net + n_sys}D")

    # --- Scaler BENIGN only ---
    print("\n=== NORMALISATION (fit BENIGN train uniquement) ===")
    ben_tr = np.array(["BENIGN" in str(l).upper() for l in y_train])
    scaler = StandardScaler().fit(X_train[ben_tr])
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print(f"✅ fitté sur {ben_tr.sum():,} bénins")

    # --- Résumé ---
    # --- Contrôle de santé : un test sans BENIGN rend l'AUROC incalculable ---
    nb_ben_test = int(np.sum(["BENIGN" in str(v).upper() for v in y_test]))
    if nb_ben_test == 0:
        print("\n" + "!" * 70)
        print("  ❌ LE JEU DE TEST NE CONTIENT AUCUN FLUX BÉNIN")
        print("!" * 70)
        print("  Conséquence : AUROC et FPR deviennent incalculables (NaN).")
        print("  Cause probable : tes fichiers ne contiennent que très peu de flux")
        print("  BÉNIN distincts (beaucoup de lignes identiques). Une fois le train")
        print("  constitué, il ne reste plus de bénin unique pour le test.")
        print("\n  Solutions, par ordre de préférence :")
        print("   1. utiliser un fichier bénin plus varié (Monday complet de CICIDS2017)")
        print("   2. réduire le quota de test  :  --benign-test 5000")
        print("   3. réduire le quota de train :  --benign-train 15000")
        print("=" * 70)
        raise SystemExit(1)

    # --- Contrôle de santé : sans attaque connue, l'entraînement est vide de sens ---
    nb_atk_train = int(sum(1 for v in y_train if "BENIGN" not in str(v).upper()))
    if nb_atk_train == 0:
        print("\n" + "!" * 70)
        print("  ❌ LE JEU D'ENTRAÎNEMENT NE CONTIENT AUCUNE ATTAQUE")
        print("!" * 70)
        print(f"  Train : {len(y_train):,} lignes, toutes BÉNIGNES.")
        print("  Conséquence : la validation interne ne contient qu'une seule classe,")
        print("  l'AUROC vaut NaN pendant tout l'entraînement, et l'early stopping")
        print("  sélectionne une époque au hasard (meilleur AUROC = -inf).")
        print("\n  Classes détectées dans data/ :")
        for f in files:
            lab = label_from_filename(f) if mode == "filename" else None
            compte = counts.get(f, {})
            print(f"    {os.path.basename(f)[:44]:<44} " +
                  ", ".join(f"{k}={v:,}" for k, v in compte.items()))
        print("\n  Il manque les fichiers d'attaques CONNUES (train) :")
        print("    DDoS · DoS · Brute Force (FTP/SSH-Patator) · Infiltration")
        print("  Seuls les Zero-Day (PortScan, Web Attack, Bot) sont présents,")
        print("  or ceux-là ne doivent JAMAIS servir à l'entraînement.")
        print("\n  → Vérifie le contenu du dossier data\\ :")
        print("      dir data\\*.parquet")
        print(f"  (version du script en cours : {VERSION})")
        print("=" * 70)
        raise SystemExit(1)

    print("\n=== RÉSUMÉ ===")
    print(f"TRAIN {X_train.shape} | TEST {X_test.shape}")
    print("\nTrain :")
    for k, v in pd.Series(y_train).value_counts().items():
        print(f"   {k:<28} {v:>7,}")
    print("\nTest :")
    for k, v in pd.Series(y_test).value_counts().items():
        print(f"   {k:<28} {v:>7,}")

    # --- Sauvegarde ---
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    # --- Garde-fou : refuser d'écrire un jeu qui contient une fuite ----------
    _yb_tr = np.array([0 if "BENIGN" in str(l).upper() else 1 for l in y_train])
    _alertes = verifier_absence_de_fuite(X_train, _yb_tr)
    if _alertes:
        print("\n" + "!" * 70)
        print("  ⚠️  FUITE DÉTECTÉE — écriture annulée")
        print("!" * 70)
        for j, nom, a in _alertes[:10]:
            print(f"     colonne {j:3d} [{nom}] : AUROC {a:.4f} à elle seule")
        print("     Une colonne qui sépare les classes presque parfaitement")
        print("     décrit l'étiquette, pas le trafic. Corriger la source")
        print("     (proxy_edr, encodage) ou retirer la branche concernée.")
        print("=" * 70)
        raise SystemExit(1)
    print(f"\n✅ Garde-fou : aucune des {X_train.shape[1]} colonnes ne sépare")
    print("   les classes à elle seule (seuil AUROC 0.99).")

    np.save(f"{OUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUT_DIR}/y_test.npy", y_test)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "n_net": int(n_net), "n_sys": int(n_sys), "n_features": int(n_net + n_sys),
        "zero_day": ZERO_DAY_ATTACKS, "seed": SEED, "synthetic": synthetic_flag,
        "benign_train": BENIGN_TRAIN, "benign_test": BENIGN_TEST,
        "attack_quota_train": ATTACK_QUOTA_TRAIN, "zero_day_quota_test": ZERO_DAY_QUOTA_TEST,
        "syscall_vocab": SYSCALL_VOCAB, "net_columns": cols,
        "scaler_fit_on": "BENIGN train only", "deduplicated": True,
        "edr_mode": args.edr_mode, "label_source": mode,
        "train_shape": list(X_train.shape), "test_shape": list(X_test.shape),
        "classes_train": sorted(set(y_train.tolist())),
        "classes_test": sorted(set(y_test.tolist())),
    }
    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(f"{OUT_DIR}/zero_day_attacks.txt", "w") as f:
        f.write(",".join(ZERO_DAY_ATTACKS))

    # --- Fichier de démonstration pour l'interface ---------------------------
    # L'application applique elle-même le scaler : elle attend donc les valeurs
    # BRUTES (avant standardisation) et les deux branches (réseau + système).
    # Un export direct de X_test.npy serait standardisé deux fois, et un fichier
    # de features seules ferait échouer le contrôle de format. On écrit donc ici
    # un CSV prêt à téléverser, avec sa colonne de classe pour mesurer le rappel.
    try:
        _n = min(20000, len(X_test))
        _idx = np.random.default_rng(SEED).choice(len(X_test), _n, replace=False)
        _raw = X_test[_idx] * scaler.scale_ + scaler.mean_
        _df = pd.DataFrame(_raw, columns=[f"feat_{i}" for i in range(_raw.shape[1])])
        _df["label"] = np.asarray(y_test)[_idx]
        _df.to_csv(f"{OUT_DIR}/demo_sample.csv", index=False)
        print(f"✅ data/processed/demo_sample.csv "
              f"({_n:,} flux × {_raw.shape[1]} colonnes brutes + label)")
        print("   → fichier prêt à téléverser dans l'onglet Démonstration")
    except Exception as _e:
        print(f"⚠️  export du fichier de démonstration impossible : {_e}")

    print("\n✅ data/processed/{X,y}_{train,test}.npy · meta.json")
    print("✅ models/scaler.pkl")
    print("\nProchaine étape : python 02_model_training.py")


if __name__ == "__main__":
    main()
