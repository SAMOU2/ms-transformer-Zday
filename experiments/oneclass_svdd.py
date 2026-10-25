"""
============================================================
 experiments/oneclass_svdd.py — Deep SVDD + contrastif
============================================================
Objectif : détecter le Zero-Day SANS jamais voir d'attaque à l'entraînement.

Principe (Deep SVDD, Ruff et al. 2018) :
    le réseau apprend à projeter tout le trafic BÉNIN dans une sphère de
    rayon minimal autour d'un centre c fixe. Un flux anormal tombe dehors.

    L(x) = || φ(x; W) − c ||²            (+ contrainte contrastive)

Pourquoi l'essai SimCLR précédent a échoué (0.383) et ce qui change ici :
    SimCLR cherchait à DISPERSER les bénins sur l'hypersphère. Les anomalies,
    plus « moyennes », atterrissaient donc près du centre et le score de
    distance perdait son sens. SVDD fait l'inverse : il COMPACTE les bénins.
    On ajoute une composante contrastive uniquement pour éviter l'effondrement
    (toutes les sorties égales au centre).

Garde-fous anti-effondrement, tous indispensables :
    1. centre c figé (jamais appris) — sinon le modèle ramène c sur les données
    2. PAS de F.normalize sur l'embedding SVDD — la norme unité détruit le rayon
    3. biais de la dernière couche remis à zéro et gelé — un biais libre pousse
       toutes les sorties vers une constante
    4. weight decay — limite la dérive des poids
    5. suivi du rayon moyen à chaque époque : s'il tend vers 0, effondrement

Protocole d'évaluation (strict) :
    • entraînement  : BENIGN du split fit UNIQUEMENT
    • calibration   : BENIGN du split validation interne
    • sélection     : AUROC sur validation = bénins vs ATTAQUES CONNUES
                      (DDoS, DoS, Patator, Infiltration) — jamais le Zero-Day
    • rapport final : Zero-Day du test, touché une seule fois à la fin

Usage :
    python experiments/oneclass_svdd.py
    python experiments/oneclass_svdd.py --epochs 15 --lambda-contrast 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Import de model_ms quel que soit l'emplacement du script (racine OU experiments/)
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if os.path.exists(os.path.join(_p, "model_ms.py")):
        sys.path.insert(0, _p)
        break
from model_ms import TokenizedMSEncoder  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
def svdd_loss(raw, center, eps=1e-6):
    """Distance quadratique moyenne au centre. Plus grand = plus anormal."""
    d2 = ((raw - center.to(raw.device)) ** 2).sum(dim=1)
    return d2.mean()


def variance_term(raw, gamma=1.0):
    """
    Répulsif anti-effondrement (VICReg, Bardes et al. 2022).

    Sans ce terme, SVDD converge vers la solution triviale : tous les bénins
    projetés au même point (rayon → 0). Le terme force chaque dimension de
    l'embedding à conserver un écart-type minimal gamma, donc la géométrie
    de l'espace latent reste informative.
    """
    std = torch.sqrt(raw.var(dim=0) + 1e-4)
    return torch.relu(gamma - std).mean()


def contrastive_ntxent(z, temperature=0.1):
    """SimCLR sur paires augmentées : la seule positive est (i, i+B)."""
    b = z.shape[0] // 2
    if b < 2:
        return torch.zeros((), device=z.device)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask.fill_diagonal_(False)
    pos = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    exp = torch.exp(sim) * mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    return -log_prob[torch.arange(len(z)), pos].mean()


@torch.no_grad()
def embed_raw(model, X, batch=4096):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.array(X[i:i + batch], dtype=np.float32))
        out.append(model(xb.to(DEVICE))["raw"].cpu())
    return torch.cat(out)


@torch.no_grad()
def score(model, X, center, batch=4096):
    """Score d'anomalie = distance euclidienne au centre. Grand = anormal."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.array(X[i:i + batch], dtype=np.float32))
        raw = model(xb.to(DEVICE))["raw"].cpu()
        out.append(((raw - center) ** 2).sum(1).sqrt())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-contrast", type=float, default=0.3,
                    help="poids de la contrainte contrastive (0 = SVDD pur)")
    ap.add_argument("--warmup", type=int, default=1, help="époques avant gel du centre")
    ap.add_argument("--lambda-var", type=float, default=1.0,
                    help="poids du terme de variance (anti-effondrement)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("  DEEP SVDD + CONTRASTIF — entraînement sur BENIGN uniquement")
    print("=" * 70)

    meta = json.load(open("data/processed/meta.json", encoding="utf-8"))
    n_net, n_sys = meta["n_net"], meta["n_sys"]
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy", allow_pickle=True)
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy", allow_pickle=True)
    zero_day = meta.get("zero_day", ["PortScan", "Web Attack", "Bot"])

    is_ben = np.array(["BENIGN" in str(l).upper() for l in y_train])

    # --- Split interne : fit / validation (le TEST n'est pas touché) ---
    rng = np.random.default_rng(args.seed)
    idx_ben = rng.permutation(np.where(is_ben)[0])
    n_val = max(2000, int(0.15 * len(idx_ben)))
    i_val, i_fit = idx_ben[:n_val], idx_ben[n_val:]
    idx_atk = np.where(~is_ben)[0]

    X_fit = X_train[i_fit]
    X_val_ben = X_train[i_val]
    # anomalies de validation = attaques CONNUES (vues à l'entraînement supervisé
    # historique, mais jamais vues par le SVDD) -> sert uniquement à choisir l'époque
    atk_sel = rng.choice(idx_atk, size=min(6000, len(idx_atk)), replace=False)
    X_val_atk = X_train[atk_sel]

    print(f"Fit (bénin)            : {X_fit.shape}")
    print(f"Validation (bénin)     : {X_val_ben.shape}")
    print(f"Validation (attaques connues, n={len(atk_sel):,}) : {X_val_atk.shape}")
    print(f"Test Zero-Day          : {X_test.shape}  ← ouvert seulement à la fin")
    print(f"contrastif λ={args.lambda_contrast} | warmup centre={args.warmup} | seed={args.seed}\n")

    # --- Modèle ---
    model = TokenizedMSEncoder(n_net=n_net, n_sys=n_sys).to(DEVICE)
    # GARDE-FOU 3 : on neutralise le biais de la dernière couche
    with torch.no_grad():
        model.proj.bias.zero_()
    model.proj.bias.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)

    Xf = torch.from_numpy(X_fit.astype(np.float32))
    n = len(Xf)

    # --- Centre initial (figé après warmup) ---
    center = embed_raw(model, X_fit[:5000]).mean(0)
    center = center / (center.norm() + 1e-9) * 2.0     # rayon initial raisonnable
    frozen = False
    print(f"centre initial : norme={center.norm():.3f}")

    y_val = np.r_[np.zeros(len(X_val_ben)), np.ones(len(X_val_atk))]
    X_val_all = np.vstack([X_val_ben, X_val_atk])

    best = {"auroc": -1, "epoch": 0, "state": None}
    history = []

    print("\n=== ENTRAÎNEMENT ===")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)

        # warmup : centre = moyenne courante (puis gel définitif)
        if ep == args.warmup and not frozen:
            center = embed_raw(model, X_fit[:8000]).mean(0).detach()
            frozen = True
            print(f"  → centre GELÉ à l'époque {ep + 1} (norme={center.norm():.3f})")

        tot_s, tot_c, nb = 0.0, 0.0, 0
        for i in range(0, n, args.batch):
            xb = Xf[perm[i:i + args.batch]].to(DEVICE)
            if len(xb) < 8:
                continue
            out = model(xb)
            l_s = svdd_loss(out["raw"], center)

            l_c = torch.zeros((), device=DEVICE)
            if args.lambda_contrast > 0:
                out_aug = model(xb + torch.randn_like(xb) * 0.05)
                z = F.normalize(torch.cat([out["raw"], out_aug["raw"]]), dim=1)
                l_c = contrastive_ntxent(z)

            l_v = variance_term(torch.cat([out["raw"], (model(xb + torch.randn_like(xb) * 0.05))["raw"]])) \
                if args.lambda_var > 0 else torch.zeros((), device=DEVICE)

            loss = l_s + args.lambda_contrast * l_c + args.lambda_var * l_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_s += l_s.item(); tot_c += float(l_c); nb += 1

        # --- suivi du rayon (détection d'effondrement) ---
        raw_fit = embed_raw(model, X_fit[:4000])
        radius = ((raw_fit - center) ** 2).sum(1).sqrt().mean().item()
        std_mean = raw_fit.std(dim=0).mean().item()
        sc_val = score(model, X_val_all, center)
        auc = roc_auc_score(y_val, sc_val) if len(set(y_val)) > 1 else float("nan")

        flag = ""
        if radius < 0.02:
            flag = "  ⚠️ rayon très faible (effondrement partiel)"
        if auc > best["auroc"]:
            best = {"auroc": auc, "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "center": center.clone(), "radius": radius}
            flag += "  ✅ meilleur"

        history.append({"epoch": ep + 1, "svdd": tot_s / max(nb, 1),
                        "contrast": tot_c / max(nb, 1), "radius": radius, "std_mean": float(std_mean),
                        "auroc_val": auc})
        print(f"epoch {ep + 1:02d}/{args.epochs} | SVDD {tot_s/max(nb,1):.4f} "
              f"| contr {tot_c/max(nb,1):.3f} | rayon {radius:.3f} | "
              f"std {std_mean:.3f} | AUROC val {auc:.4f}{flag}")

    print(f"Durée : {time.time() - t0:.0f}s")

    if best["state"] is None:
        print("\n❌ Aucune époque valide (AUROC de validation non calculable).")
        print("   Vérifie que le train contient bien BENIGN + au moins une attaque connue.")
        sys.exit(1)

    # --- Restauration du meilleur état ---
    model.load_state_dict(best["state"])
    center = best["center"]
    print(f"\n✅ Meilleure époque : {best['epoch']} | AUROC val {best['auroc']:.4f} "
          f"| rayon {best['radius']:.3f}")

    # --- ÉVALUATION FINALE (première fois que le test est touché) ---
    print("\n" + "=" * 70)
    print("  ÉVALUATION FINALE — ZERO-DAY")
    print("=" * 70)
    y_te_bin = np.array(["BENIGN" not in str(l).upper() for l in y_test]).astype(int)
    sc_te = score(model, X_test, center)
    auc_te = roc_auc_score(y_te_bin, sc_te)
    print(f"AUROC Zero-Day : {auc_te:.4f}")

    ben_scores = sc_te[y_te_bin == 0]
    thr95 = float(np.quantile(ben_scores, 0.95))
    pred = sc_te > thr95
    print(f"Seuil (p95 bénin) : {thr95:.4f} | FPR {pred[y_te_bin==0].mean():.2%} "
          f"| Recall {pred[y_te_bin==1].mean():.2%}")

    print(f"\n{'Classe':<28}{'n':>7}{'Recall':>10}{'AUROC':>9}")
    print("-" * 56)
    per_class = {}
    for cls in sorted(set(str(v) for v in y_test)):
        m = np.array([str(l) == cls for l in y_test])
        if m.sum() == 0:
            continue
        if "BENIGN" in cls.upper():
            print(f"{cls[:27]:<28}{m.sum():>7}{pred[m].mean():>9.2%}{'—':>9}  (taux de FP)")
            per_class[cls] = {"n": int(m.sum()), "recall": float(pred[m].mean()), "auroc": None}
            continue
        yb = np.r_[np.zeros((y_te_bin == 0).sum()), np.ones(m.sum())]
        sc = np.r_[sc_te[y_te_bin == 0], sc_te[m]]
        a = roc_auc_score(yb, sc)
        per_class[cls] = {"n": int(m.sum()), "recall": float(pred[m].mean()), "auroc": float(a)}
        print(f"{cls[:27]:<28}{m.sum():>7}{pred[m].mean():>9.2%}{a:>9.4f}")

    # --- Comparaison avec la référence supervisée ---
    print("\n" + "-" * 70)
    ref = None
    if os.path.exists("results/metrics.json"):
        ref = json.load(open("results/metrics.json"))["global"].get("auroc")
    print(f"  Référence (MS-Transformer supervisé, V3) : {ref if ref is None else round(ref,4)}")
    print(f"  One-class SVDD (ce run)                  : {auc_te:.4f}")
    if ref is not None:
        d = auc_te - ref
        print(f"  Écart                                    : {d:+.4f}"
              f"  {'✅ gain' if d > 0.005 else '⚠️ pas de gain' if d > -0.005 else '❌ dégradation'}")

    # --- Sauvegarde ---
    os.makedirs("results", exist_ok=True)
    out = {"method": "deep_svdd_contrastive", "epochs": args.epochs,
           "lambda_contrast": args.lambda_contrast, "seed": args.seed,
           "best_epoch": best["epoch"], "auroc_val_known": best["auroc"],
           "auroc_zero_day": float(auc_te), "threshold_p95": thr95,
           "radius_mean": best["radius"], "per_class": per_class,
           "history": history, "reference_supervised_auroc": ref}
    with open("results/oneclass_svdd.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    torch.save({"state": model.state_dict(), "center": center},
               "results/oneclass_svdd.pt")
    print("\n✅ results/oneclass_svdd.json · oneclass_svdd.pt")


if __name__ == "__main__":
    main()
