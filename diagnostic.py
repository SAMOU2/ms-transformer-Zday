"""
============================================================
 05_diagnostic.py — Pourquoi j'ai des NaN ?
============================================================
Vérifie en 10 secondes que les données et le modèle sont exploitables.
Chaque contrôle affiche ✅ (ok) ou ❌ (problème) avec la correction.

Usage :
    python 05_diagnostic.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

OK, KO, WARN = "✅", "❌", "⚠️ "
problems = []


def head(t):
    print(f"\n{'=' * 62}\n  {t}\n{'=' * 62}")


def main():
    print("=" * 62)
    print("  DIAGNOSTIC — pipeline MS-Transformer")
    print("=" * 62)

    # ---------- 1. Fichiers ----------
    head("1. FICHIERS ATTENDUS")
    required = {
        "data/processed/X_train.npy": "données d'entraînement",
        "data/processed/y_train.npy": "labels d'entraînement",
        "data/processed/X_test.npy": "données de test",
        "data/processed/y_test.npy": "labels de test",
        "data/processed/meta.json": "métadonnées (n_net, n_sys, edr_mode)",
        "models/scaler.pkl": "normalisateur",
        "models/ms_encoder.pt": "poids du modèle",
        "models/centroid.pt": "centroïde bénin",
        "models/model_config.json": "configuration du modèle",
    }
    for path, desc in required.items():
        if os.path.exists(path):
            print(f"  {OK} {path:<34} {desc}")
        else:
            print(f"  {KO} {path:<34} {desc} — MANQUANT")
            problems.append(f"{path} absent → relance l'étape correspondante")

    if problems:
        raise SystemExit("\n  Arrêt : fichiers manquants (voir ci-dessus).")

    # ---------- 2. meta.json ----------
    head("2. MÉTADONNÉES")
    meta = json.load(open("data/processed/meta.json", encoding="utf-8"))
    n_net, n_sys = meta.get("n_net"), meta.get("n_sys")
    print(f"  n_net={n_net}  n_sys={n_sys}  total={meta.get('n_features')}")
    edr = meta.get("edr_mode", "proxy (défaut)")
    if edr == "label":
        print(f"  {KO} edr_mode='label' → la branche EDR encode le label : FUITE DE DONNÉES")
        problems.append("relance `python 01_data_preparation.py` (sans --edr-mode label)")
    else:
        print(f"  {OK} edr_mode='{edr}' (pas de fuite de label)")
    if meta.get("synthetic"):
        print(f"  {KO} données SYNTHÉTIQUES → les chiffres ne valent rien")
        problems.append("relance la préparation sur les vrais .parquet CICIDS2017")
    else:
        print(f"  {OK} données réelles")

    # ---------- 3. Données ----------
    head("3. CONTENU DES JEUX DE DONNÉES")
    X_train = np.load("data/processed/X_train.npy", mmap_mode="r")
    X_test = np.load("data/processed/X_test.npy", mmap_mode="r")
    y_train = np.load("data/processed/y_train.npy", allow_pickle=True)
    y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

    print(f"  X_train {X_train.shape} | X_test {X_test.shape}")
    if X_train.shape[1] != (n_net or 0) + (n_sys or 0):
        problems.append("X_train.shape[1] incompatible avec meta.json → relance 01")
        print(f"  {KO} dimension incohérente avec meta.json")

    for name, X in (("train", X_train), ("test", X_test)):
        sub = np.asarray(X[:2000], dtype=np.float32)
        nan = int(np.isnan(sub).sum())
        inf = int(np.isinf(sub).sum())
        print(f"  {name:<5} NaN={nan} inf={inf} " + (OK if nan == 0 and inf == 0 else KO))
        if nan or inf:
            problems.append(f"X_{name} contient NaN/inf")

    # --- LE contrôle qui explique les NaN de l'ablation ---
    def dist(y, title):
        s = {}
        for v in y:
            s[str(v)] = s.get(str(v), 0) + 1
        print(f"\n  {title} ({len(y):,}) :")
        for k, v in sorted(s.items(), key=lambda x: -x[1]):
            print(f"     {k:<34} {v:>7,}")
        return s

    d_tr = dist(y_train, "TRAIN")
    d_te = dist(y_test, "TEST")

    ben_test = sum(v for k, v in d_te.items() if "BENIGN" in k.upper())
    ben_train = sum(v for k, v in d_tr.items() if "BENIGN" in k.upper())

    print()
    if ben_test == 0:
        print(f"  {KO} AUCUN BENIGN dans le test  ← C'EST LA CAUSE DE TES NaN")
        print("     AUROC et FPR deviennent incalculables : il n'y a pas de classe")
        print("     de référence pour mesurer les faux positifs.")
        problems.append("y_test sans BENIGN → relance `python 01_data_preparation.py`")
    else:
        print(f"  {OK} BENIGN dans le test : {ben_test:,}")

    if ben_train == 0:
        print(f"  {KO} AUCUN BENIGN dans le train → centroïde impossible")
        problems.append("y_train sans BENIGN → relance 01")
    else:
        print(f"  {OK} BENIGN dans le train : {ben_train:,}")

    nb_classes = len([k for k in d_te if "BENIGN" not in k.upper()])
    if nb_classes == 0:
        print(f"  {KO} aucune attaque dans le test")
        problems.append("y_test sans attaque → relance 01")
    else:
        print(f"  {OK} {nb_classes} classe(s) d'attaque dans le test")

    # ---------- 4. Modèle ----------
    head("4. MODÈLE ET INFÉRENCE")
    try:
        import torch

        from model_ms import TokenizedMSEncoder, anomaly_score

        cfg = json.load(open("models/model_config.json", encoding="utf-8"))
        m = TokenizedMSEncoder(n_net=cfg["n_net"], n_sys=cfg["n_sys"],
                               embed_dim=cfg.get("embed_dim", 32), d=cfg.get("d", 64),
                               n_tok_net=cfg.get("n_tok_net", 4),
                               n_tok_sys=cfg.get("n_tok_sys", 5),
                               n_layers=cfg.get("n_layers", 2))
        m.load_state_dict(torch.load("models/ms_encoder.pt", map_location="cpu"))
        m.eval()
        c = torch.load("models/centroid.pt", map_location="cpu")
        print(f"  {OK} poids chargés ({sum(p.numel() for p in m.parameters()):,} paramètres)")

        with torch.no_grad():
            xb = torch.from_numpy(np.array(X_test[:512], dtype=np.float32, copy=True))
            z = m(xb)["z"]
            sc = anomaly_score(z, c).numpy()
        print(f"  centroïde norme = {float(c.norm()):.4f} " + (OK if abs(float(c.norm()) - 1) < 0.01 else KO))
        if np.isnan(sc).any():
            print(f"  {KO} scores NaN → centroïde ou modèle corrompu")
            problems.append("réentraîne : python 02_model_training.py")
        else:
            print(f"  {OK} scores calculés | min={sc.min():.4f} méd={np.median(sc):.4f} max={sc.max():.4f}")
        if sc.max() - sc.min() < 1e-9:
            print(f"  {KO} tous les scores sont identiques → modèle dégénéré")
            problems.append("réentraîne : python 02_model_training.py")
    except Exception as e:
        print(f"  {KO} inférence impossible : {e}")
        problems.append(f"inférence : {e}")

    # ---------- 5. Résultats ----------
    # ---------- 4.5 Cohérence données <-> modèle ----------
    head("4bis. COHÉRENCE DONNÉES ↔ MODÈLE")
    try:
        import torch as _t
        cfg2 = json.load(open("models/model_config.json", encoding="utf-8"))
        saved = _t.load("models/ms_encoder.pt", map_location="cpu")
        key = [k for k in saved if "proj_net" in k or "net.0.weight" in k][0]
        w = saved[key].shape
        expected_in = cfg2["n_net"]
        got_in = None
        for k2, v in saved.items():
            if k2.endswith("proj_net.weight") or k2.endswith("net.0.weight"):
                got_in = v.shape[1] if "proj_net" in k2 else v.shape[1]
                break
        print(f"  modèle entraîné avec n_net={cfg2['n_net']} n_sys={cfg2['n_sys']}")
        print(f"  données actuelles  n_net={n_net} n_sys={n_sys}")
        if (cfg2["n_net"], cfg2["n_sys"]) != (n_net, n_sys):
            print(f"  {KO} LE MODÈLE ET LES DONNÉES NE VIENNENT PAS DU MÊME RUN")
            problems.append("réentraîne sur les données actuelles : py 02_model_training.py")
        else:
            print(f"  {OK} dimensions cohérentes")
    except Exception as e:
        print(f"  {WARN}contrôle impossible : {e}")

    # ---------- 5. Résultats ----------
    head("5. RÉSULTATS EXISTANTS")
    for f in ("results/metrics.json", "results/ablation.csv"):
        if os.path.exists(f):
            print(f"  {OK} {f}")
        else:
            print(f"  {WARN}{f} absent (normal si tu n'as pas lancé l'étape)")
    if os.path.exists("results/metrics.json"):
        g = json.load(open("results/metrics.json"))["global"]
        a = g.get("auroc")
        if a is None or (isinstance(a, float) and np.isnan(a)):
            print(f"  {KO} metrics.json contient un AUROC NaN (même problème g)")
        else:
            print(f"  AUROC enregistré : {a:.4f} | synthetic={g.get('synthetic')}")

    # ---------- Bilan ----------
    head("BILAN")
    if not problems:
        print(f"  {OK} Tout est cohérent. Tu peux lancer 03_evaluation.py puis 04_ablation.py.")
    else:
        print(f"  {KO} {len(problems)} problème(s) :\n")
        for i, p in enumerate(problems, 1):
            print(f"   {i}. {p}")
        print("\n  Ordre de correction :")
        print("     py 01_data_preparation.py")
        print("     py 02_model_training.py")
        print("     py diagnostic.py      # doit afficher « Tout est cohérent »")
        print("     py 03_evaluation.py")
        print("     py 04_ablation.py --variants full no_edr no_contrast mlp --epochs 10")
        sys.exit(1)


if __name__ == "__main__":
    main()
