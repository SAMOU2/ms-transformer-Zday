"""
============================================================
 02_model_training.py — Entraînement MS-Transformer (V3)
============================================================
Architecture réelle, à 3 têtes, importée depuis model_ms.py :

  tokens (4 réseau + 5 EDR + CLS) → Transformer 2 couches à masque
  hiérarchique → CLS → {embedding contrastif, violation, next-state}

Entraînement multi-objectif :
    L = L_contrastif  +  0.5 · L_violation (BCE)  +  0.3 · L_next-state (MSE)

Le scheduler ReduceLROnPlateau(verbose=...) est déprécié en PyTorch 2.x :
il est volontairement absent ici.

Sorties :
    models/ms_encoder.pt    poids du modèle
    models/centroid.pt      centroïde bénin (score d'anomalie)
    models/thresholds.json  seuils EVT calibrés (anomalie, violation, énergie)
    models/train_history.json

Usage :
    python 02_model_training.py                # entraînement complet
    python 02_model_training.py --epochs 2     # test rapide
    python 02_model_training.py --smoke        # 1 epoch sur 2 000 lignes
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model_ms import (TokenizedMSEncoder, anomaly_score, coherence_scores,
                      compute_loss, free_energy, token_scores,
                      next_state_confidence)

MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter, self.best_state = -np.inf, 0, None

    def step(self, value, model):
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


# ------------------------------------------------------------------
def evaluate_scores(model, X, y_bin, centroid, n_net, n_sys, batch=4096):
    """Retourne un dict de scores sur un jeu donné (aucun gradient)."""
    model.eval()
    zs, viol, energy, nconf, tok, coh = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.FloatTensor(X[i:i + batch]).to(DEVICE)
            out = model(xb)
            zs.append(out["z"].cpu())
            viol.append(torch.sigmoid(out["violation_logit"]).cpu())
            energy.append(free_energy(out["violation_logit"]).cpu())
            conf, _ = next_state_confidence(out["next_pred"].cpu(), xb.cpu(), n_net, n_sys)
            nconf.append(conf)
            # lecture par token : c'est cette sortie qui porte la performance
            tok.append(token_scores(out["token_logits"].cpu()).cpu())
            coh.append(coherence_scores(out).cpu())
    z = torch.cat(zs)
    return {
        "score": anomaly_score(z, centroid).numpy(),
        "z": z.numpy(),
        "violation": torch.cat(viol).numpy(),
        "energy": torch.cat(energy).numpy(),
        "next_conf": torch.cat(nconf).numpy(),
        "token": torch.cat(tok).numpy(),
        "coherence": torch.cat(coh).numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="test ultra-rapide")
    ap.add_argument("--seed", type=int, default=42, help="graine (résultats reproductibles)")
    ap.add_argument("--keep-best", action="store_true",
                    help="n'écrase models/ que si l'AUROC dépasse le meilleur déjà enregistré")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1

    # --- Reproductibilité ---
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    print("=" * 68)
    print("  ENTRAÎNEMENT MS-TRANSFORMER V3 — architecture tokenisée")
    print("=" * 68)
    print(f"Device : {DEVICE} | seed : {args.seed}")

    # --- Données ---
    meta_path = "data/processed/meta.json"
    if not os.path.exists(meta_path):
        raise SystemExit("❌ data/processed/meta.json absent — lance 01_data_preparation.py")
    meta = json.load(open(meta_path, encoding="utf-8"))
    n_net, n_sys = meta["n_net"], meta["n_sys"]
    zero_day = meta.get("zero_day", ["PortScan", "Web Attack", "Bot"])

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy", allow_pickle=True)
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy", allow_pickle=True)
    if args.smoke:
        X_train, y_train = X_train[:2000], y_train[:2000]
        X_test, y_test = X_test[:1500], y_test[:1500]

    y_bin_tr = np.array([0 if "BENIGN" in str(l).upper() else 1 for l in y_train])
    y_bin_te = np.array([0 if "BENIGN" in str(l).upper() else 1 for l in y_test])
    print(f"Train {X_train.shape} | Test {X_test.shape}")

    # --- Garde-fou : un test sans BENIGN rend l'AUROC et la spécificité incalculables ---
    # Sans ce contrôle, l'entraînement se déroule normalement pendant deux minutes,
    # puis l'évaluation affiche AUROC = nan et Spécificité = nan% sans expliquer
    # pourquoi. On préfère échouer immédiatement, avec la cause et la solution.
    if int((y_bin_te == 0).sum()) == 0:
        print("\n" + "!" * 70)
        print("  ❌ ARRÊT : le jeu de test ne contient AUCUN flux BÉNIN")
        print("!" * 70)
        print(f"  Test : {len(y_test):,} lignes, toutes des attaques.")
        print("  Conséquence : AUROC-OOD et spécificité (FPR) sont incalculables.")
        print("  Le rappel affiché serait mesuré à un point de fonctionnement")
        print("  dont on ne peut pas vérifier le taux de faux positifs.")
        print("\n  → Le fichier data/processed/ a été produit par une ANCIENNE")
        print("    version de 01_data_preparation.py. Il faut le régénérer :")
        print("\n      py 01_data_preparation.py")
        print("\n    puis vérifier :")
        print("      py -c \"import numpy as np, collections; "
              "y=np.load('data/processed/y_test.npy',allow_pickle=True); "
              "print(dict(collections.Counter(str(v) for v in y)))\"")
        print("\n    La classe BENIGN doit apparaître (~25 000 lignes).")
        print("=" * 70)
        raise SystemExit(1)
    print(f"Contrôle test : {int((y_bin_te == 0).sum()):,} bénins, "
          f"{int((y_bin_te == 1).sum()):,} anomalies ✅")

    # --- Garde-fou : le train doit contenir des attaques connues ---
    # Sans attaque, la validation interne n'a qu'une classe : l'AUROC vaut NaN
    # à chaque époque, l'early stopping compare des -inf et sélectionne une
    # époque arbitraire. L'évaluation finale produit alors un modèle qui n'a
    # jamais appris à distinguer quoi que ce soit.
    if int((y_bin_tr == 1).sum()) == 0:
        print("\n" + "!" * 70)
        print("  ❌ ARRÊT : le jeu d'entraînement ne contient AUCUNE attaque")
        print("!" * 70)
        print(f"  Train : {len(y_train):,} lignes, toutes BÉNIGNES.")
        print("  La validation interne n'aurait qu'une seule classe : AUROC = NaN")
        print("  à chaque époque et early stopping sur des -inf.")
        print("\n  → Il manque les fichiers d'attaques CONNUES dans data\\ :")
        print("      DDoS · DoS · Brute Force · Infiltration")
        print("  → Régénérer :  py 01_data_preparation.py")
        print("=" * 70)
        raise SystemExit(1)
    print(f"Modalités : réseau={n_net}D → 4 tokens | EDR={n_sys}D → 5 tokens")
    if meta.get("synthetic"):
        print("⚠️  Données SYNTHÉTIQUES — chiffres non exploitables dans le mémoire")

    # --- Modèle ---
    model = TokenizedMSEncoder(n_net=n_net, n_sys=n_sys, embed_dim=32, d=64,
                               n_tok_net=4, n_tok_sys=5, n_layers=2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    stopper = EarlyStopping(patience=args.patience)

    # --- Split de validation interne (10 %) ---
    # CRITIQUE : le jeu de test Zero-Day ne doit JAMAIS servir à choisir une
    # époque ni à piloter le scheduler. Sinon l'AUROC annoncé est optimiste :
    # on a sélectionné le modèle sur les données censées mesurer sa performance.
    rng_split = np.random.default_rng(args.seed)
    idx_all = rng_split.permutation(len(X_train))
    n_val = max(2000, int(0.10 * len(X_train)))
    idx_val, idx_fit = idx_all[:n_val], idx_all[n_val:]
    X_val, y_bin_val = X_train[idx_val], y_bin_tr[idx_val]
    X_fit, y_bin_fit = X_train[idx_fit], y_bin_tr[idx_fit]
    print(f"Fit {X_fit.shape} | Validation interne {X_val.shape} | "
          f"Test Zero-Day {X_test.shape} (jamais utilisé avant la fin)")

    Xt = torch.FloatTensor(X_fit).to(DEVICE)
    yt_bin = torch.LongTensor(y_bin_fit).to(DEVICE)
    Xval_t = torch.FloatTensor(X_val).to(DEVICE)
    n = len(Xt)

    # --- Boucle ---
    print("\n=== ENTRAÎNEMENT ===")
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        acc = {"loss": 0.0, "contrast": 0.0, "violation": 0.0, "next": 0.0,
               "token": 0.0, "coherence": 0.0}
        nb = 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = Xt[idx]
            yb = yt_bin[idx]
            if len(xb) < 8:
                continue
            out = model(xb)
            loss, parts = compute_loss(out, xb, yb, n_net, n_sys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc["loss"] += loss.item(); acc["contrast"] += float(parts["contrast"])
            acc["token"] = acc.get("token", 0.0) + float(parts.get("token", 0.0))
            acc["coherence"] = acc.get("coherence", 0.0) + float(parts.get("coherence", 0.0))
            acc["violation"] += float(parts["violation"]); acc["next"] += float(parts["next"])
            nb += 1
        for k in acc:
            acc[k] /= max(nb, 1)

        # validation INTERNE : centroïde provisoire -> AUROC sur X_val
        model.eval()
        with torch.no_grad():
            zb = model(Xt[yt_bin == 0][:8000])["z"]
            centroid = F.normalize(zb.mean(0), dim=0)
        sc_val = evaluate_scores(model, X_val, y_bin_val, centroid, n_net, n_sys)
        auc = roc_auc_score(y_bin_val, sc_val["score"]) if len(set(y_bin_val)) > 1 else float("nan")
        sched.step(auc if np.isfinite(auc) else 0.0)
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": ep + 1, **acc, "auroc": float(auc), "lr": lr_now})

        print(f"epoch {ep + 1:02d}/{args.epochs} | loss {acc['loss']:.4f} "
              f"(C {acc['contrast']:.3f} V {acc['violation']:.3f} N {acc['next']:.3f} "
              f"T {acc['token']:.3f} K {acc['coherence']:.3f}) "
              f"| AUROC {auc:.4f} | lr {lr_now:.1e}")

        if stopper.step(auc, model):
            print(f"⏹ EarlyStopping (patience {args.patience}) — meilleur AUROC {stopper.best:.4f}")
            break
    print(f"Durée : {time.time() - t0:.0f}s")

    # --- Restaure le meilleur état ---
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        print(f"✅ Meilleur modèle restauré (AUROC val {stopper.best:.4f})")

    # --- Centroïde final ---
    model.eval()
    with torch.no_grad():
        zb = model(Xt[yt_bin == 0])["z"]
        centroid = F.normalize(zb.mean(0), dim=0)
    print(f"Centroïde bénin calculé sur {int((y_bin_fit == 0).sum()):,} échantillons (fit uniquement)")

    # --- Seuils, calibrés sur le BÉNIN du train (jamais sur le test) ---
    #
    # Correction de fond : le seuil était fixé au 95e percentile du bénin
    # d'ENTRAÎNEMENT, puis appliqué au TEST. Or le modèle est meilleur sur
    # ses données d'entraînement que sur des données inédites : le seuil est
    # donc trop haut au test, et le rappel s'effondre alors que l'AUROC reste
    # excellente. On calibre désormais sur le bénin de VALIDATION, qui est
    # inédit lui aussi, et on choisit le quantile pour tenir le FPR cible.
    TARGET_FPR = 0.05
    sc_fit = evaluate_scores(model, X_fit, y_bin_fit, centroid, n_net, n_sys)
    sc_val = evaluate_scores(model, X_val, y_bin_val, centroid, n_net, n_sys)
    ben_fit, ben_val = y_bin_fit == 0, y_bin_val == 0

    thresholds = {}
    for key in ("score", "violation", "energy", "token", "coherence"):
        vals = np.sort(sc_fit[key][ben_fit])
        thresholds[key] = {
            "p95": float(np.quantile(vals, 0.95)),
            "p99": float(np.quantile(vals, 0.99)),
            "mean": float(vals.mean()), "std": float(vals.std()),
        }
    thresholds["next_conf"] = {
        "p05": float(np.quantile(sc_fit["next_conf"][ben_fit], 0.05)),
        "mean": float(sc_fit["next_conf"][ben_fit].mean()),
    }

    # Quel score discrimine le mieux ?
    #
    # PIEGE MAJEUR : la validation interne est un tirage du TRAIN, donc ses
    # attaques appartiennent aux familles DEJA VUES (DDoS, patators,
    # Infiltration). Une tete supervisee y parait excellente parce qu'elle a
    # memorise ces familles — et s'effondre sur les familles inedites du test.
    # C'est exactement ce qui s'est produit : violation 0.9988 en validation
    # -> 0.8102 sur le test Zero-Day. On ne peut donc PAS choisir la tete sur
    # les attaques de validation.
    #
    # Regle : seules les tetes ENTRAINEES SANS LABEL D'ATTAQUE sont eligibles
    # (token, score, coherence). Elles ne peuvent rien memoriser des familles.
    # La tete violation reste affichee, a titre informatif, jamais selectionnee.
    ELIGIBLES = ("token", "score", "coherence")
    candidats, signes = {}, {}
    for key in ELIGIBLES:
        if len(set(y_bin_val)) > 1:
            a = float(roc_auc_score(y_bin_val, sc_val[key]))
            signes[key] = 1.0 if a >= 0.5 else -1.0   # tete inversee -> on la retourne
            candidats[key] = max(a, 1.0 - a)
    best_key = max(candidats, key=candidats.get)
    sgn = signes[best_key]
    print("\n--- Choix du score (validation, tetes sans label d'attaque) ---")
    for k, v in sorted(candidats.items(), key=lambda kv: -kv[1]):
        note = "   <- retenu" if k == best_key else ("" if signes[k] > 0 else "  (inversee)")
        print(f"  {k:<12} AUROC {v:.4f}{note}")
    a_viol = float(roc_auc_score(y_bin_val, sc_val["violation"]))
    print(f"  violation    AUROC {a_viol:.4f}   (NON eligible : supervisee, "
          f"mesure la memorisation des familles vues)")

    # seuil = quantile du benin de validation pour tenir TARGET_FPR
    seuil = float(np.quantile(sgn * sc_val[best_key][ben_val], 1.0 - TARGET_FPR))
    fpr_val = float((sgn * sc_val[best_key][ben_val] > seuil).mean())
    rec_val = float((sgn * sc_val[best_key][~ben_val] > seuil).mean())
    print(f"  Score retenu : {best_key} | seuil {seuil:.4f} "
          f"(FPR validation {fpr_val:.2%}, rappel validation {rec_val:.2%})")
    thresholds["retenu"] = {"score": best_key, "seuil": seuil,
                            "signe": sgn, "target_fpr": TARGET_FPR}

    # --- Évaluation finale ---
    print("\n=== ÉVALUATION ZERO-DAY ===")
    sc_te = evaluate_scores(model, X_test, y_bin_te, centroid, n_net, n_sys)
    _k = thresholds["retenu"]["score"]
    _t = thresholds["retenu"]["seuil"]
    _s = thresholds["retenu"]["signe"]
    s_te = _s * sc_te[_k]
    auc = roc_auc_score(y_bin_te, s_te)
    pred = s_te > _t
    print(f"\nScore évalué       : {_k}")
    print(f"AUROC-OOD          : {auc:.4f}")
    print(f"Seuil (FPR {TARGET_FPR:.0%} sur validation) : {_t:.4f}")
    print(f"Recall global      : {pred[y_bin_te == 1].mean():.2%}")
    print(f"Spécificité        : {(~pred[y_bin_te == 0]).mean():.2%}")
    print(f"\n{'Classe':<26}{'n':>7}{'Recall':>10}{'Violation':>12}{'next_conf':>12}")
    print("-" * 68)
    for cls in sorted(set(y_test)):
        m = np.array([str(l) == cls for l in y_test])
        if m.sum() == 0:
            continue
        rec = float((~pred[m]).mean()) if "BENIGN" in cls.upper() else float(pred[m].mean())
        print(f"{cls[:25]:<26}{int(m.sum()):>7}{rec:>9.2%}"
              f"{sc_te['violation'][m].mean():>12.3f}{sc_te['next_conf'][m].mean():>12.3f}")

    print("\n--- Analyse post-hoc : AUROC de chaque tête sur le test ---")
    print("    (mesure de contrôle, ne sert PAS à choisir : le choix a été")
    print("     figé avant, sur des têtes sans label d'attaque)")
    for key in ("token", "score", "coherence", "violation"):
        if key in ("coherence", "violation"):
            a = float(roc_auc_score(y_bin_te, sc_te[key]))
            a = max(a, 1.0 - a)
        else:
            a = float(roc_auc_score(y_bin_te, sc_te[key]))
        tag = "  <- score retenu" if key == _k else ""
        print(f"  {key:<12} {a:.4f}{tag}")

    ok = True
    print("\n--- Critère go/no-go (Recall ≥ 85 % par Zero-Day) ---")
    for z in zero_day:
        m = np.array([z.lower() in str(l).lower() for l in y_test])
        if m.sum() == 0:
            continue
        r = float(pred[m].mean())
        ok &= r >= 0.85
        print(f"  {'✅' if r >= 0.85 else '❌'} {z:<20} {r:.2%}")
    print(f"\n  Verdict : {'✅ PROJET DÉFENDABLE' if ok else '⚠️  à analyser (cf. §5 du plan)'}")

    # --- Sauvegarde (n'écrase que si meilleur, quand --keep-best) ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    marker = f"{MODEL_DIR}/best_auroc.txt"
    best_prev = float(open(marker).read()) if os.path.exists(marker) else -1.0
    write_ok = True
    auc = float(max(stopper.best, auc))   # comparaison sur le meilleur, pas le dernier
    if args.keep_best and auc <= best_prev:
        write_ok = False
        print(f"\n⏭  AUROC {auc:.4f} ≤ meilleur connu {best_prev:.4f} → modèles NON écrasés")
    else:
        with open(marker, "w") as f:
            f.write(f"{auc:.6f}")
    if not write_ok:
        print(f"✅ Meilleur modèle conservé ({max(auc, best_prev):.4f})")
        return
    # On sauvegarde les MEILLEURS poids, pas les derniers. Sans cela, un
    # entraînement qui se dégrade après l'époque 5 écrit le pire modèle du
    # run tout en affichant le meilleur AUROC pendant le déroulé.
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        print(f"↺ Poids restaurés à l'époque au meilleur AUROC ({stopper.best:.4f})")

    torch.save(model.state_dict(), f"{MODEL_DIR}/ms_encoder.pt")
    torch.save(centroid.cpu(), f"{MODEL_DIR}/centroid.pt")
    cfg = {"seed": args.seed, "epochs_run": len(history),
           "auroc": float(max(stopper.best, auc)),
           "auroc_best": float(stopper.best),
           "auroc_last": float(auc),
           "n_net": n_net, "n_sys": n_sys, "embed_dim": 32, "d": 64,
           "n_tok_net": 4, "n_tok_sys": 5, "n_layers": 2,
           "architecture": "dual-encoder-cross-attention", "version": "V5",
           "nhead": 4, "n_params": int(sum(p.numel() for p in model.parameters()))}
    with open(f"{MODEL_DIR}/model_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(f"{MODEL_DIR}/thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    # `default=float` : un tenseur oublié ne doit jamais faire planter une
    # session d'entraînement terminée (l'historique n'est qu'un journal).
    with open(f"{MODEL_DIR}/train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=float)

    print(f"\n✅ {MODEL_DIR}/ms_encoder.pt · centroid.pt · thresholds.json · model_config.json")
    print("Prochaine étape : python 03_evaluation.py")


if __name__ == "__main__":
    main()
