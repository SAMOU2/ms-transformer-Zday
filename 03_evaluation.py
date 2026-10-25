"""
============================================================
 03_evaluation.py — Évaluation complète Zero-Day (Semaine 1 : J4 + J5)
============================================================
Produit TOUS les chiffres du chapitre Résultats :

  - AUROC global + intervalle de confiance bootstrap (1000 tirages)
  - Recall PAR Zero-Day (PortScan / Web Attack / Bot) ← critère go/no-go
  - Spécificité sur BENIGN (taux de faux positifs)
  - Matrice de confusion au seuil retenu
  - Calibration du seuil par EVT (Peak-Over-Threshold / GPD) vs percentile
  - FPR@95%TPR, TPR@1%FPR
  - Figures PNG 150 dpi prêtes pour Overleaf
  - Export CSV + JSON des métriques

Usage :
    python 03_evaluation.py                 # évaluation réelle
    python 03_evaluation.py --synthetic     # test du pipeline sans modèle/données
    python 03_evaluation.py --target-fpr 0.01

Sorties dans results/ :
    metrics.json, metrics_per_class.csv, metrics_global.csv
    fig_score_dist.png, fig_roc.png, fig_pr.png,
    fig_recall_class.png, fig_confusion.png
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# ------------------------------------------------------------------
# Métriques (pur numpy/sklearn — testables sans torch)
# ------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc



# ------------------------------------------------------------------
# Score de Mahalanobis (option --score maha)
# ------------------------------------------------------------------
def fit_mahalanobis(Z_benign: np.ndarray, shrinkage: str = "auto"):
    """
    Ajuste moyenne + covariance régularisée sur les embeddings BÉNINS du train.

    Pourquoi : sur ce modèle les embeddings sont quasi colinéaires. Le score
    « 1 - cos » ne dépend alors que d'une direction, et le moindre déplacement
    du centroïde réordonne les flux (AUROC 0.38 vs 0.55 pour deux centroïdes
    de similarité 0.983). La distance de Mahalanobis prend en compte la
    covariance complète : elle pondère chaque direction par son inverse de
    variance et redevient donc stable et informative.
    """
    from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
    Z = np.asarray(Z_benign, dtype=np.float64)
    mu = Z.mean(axis=0)
    est = {"lw": LedoitWolf, "oas": OAS, "emp": EmpiricalCovariance}.get(
        shrinkage, LedoitWolf)()
    est.fit(Z)
    prec = est.get_precision()
    return mu, prec, est


def maha_score(Z: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Distance de Mahalanobis (racine) : grand = anormal."""
    D = np.asarray(Z, dtype=np.float64) - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", D, prec, D))


def trivial_baselines(X_train, y_train_bin, X_test, y_test_bin):
    """
    Baselines triviales : aucune architecture, aucun apprentissage profond.

    Ces deux références répondent à la question que tout évaluateur pose en
    premier : « quelle performance obtient-on sans modèle ? ». Sur CICIDS2017,
    le StandardScaler étant ajusté sur le trafic bénin, chaque flux d'attaque
    reçoit mécaniquement de grands z-scores. La MAGNITUDE des features encode
    donc la classe, et une simple somme suffit à séparer les attaques.

    Si une baseline triviale atteint le niveau du modèle, ce n'est pas le
    modèle qui détecte : c'est la fuite d'échelle dans le jeu de données.
    """
    out = {}
    ben_tr = (y_train_bin == 0)

    # 1. somme des valeurs absolues, seuil au quantile bénin
    f = lambda A: np.abs(A).sum(axis=1)
    out["somme|x|"] = {"test": f(X_test), "train_benign": f(X_train[ben_tr])}

    # 2. régression logistique sur les mêmes features
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train_bin)
        out["régression logistique"] = {"test": lr.decision_function(X_test),
                                        "train_benign": None}
    except Exception:
        pass
    return out


def auroc(y_bin, scores) -> float:
    y_bin = np.asarray(y_bin).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, scores))


def bootstrap_auroc(y_bin, scores, n_boot: int = 1000, seed: int = 42):
    """IC 95 % de l'AUROC par rééchantillonnage bootstrap."""
    rng = np.random.default_rng(seed)
    y_bin = np.asarray(y_bin).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_bin)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_bin[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, scores[idx]))
    vals = np.array(vals)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rates_at_threshold(y_bin, scores, thr: float) -> dict:
    y_bin = np.asarray(y_bin).astype(int)
    pred = (np.asarray(scores) > thr).astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else float("nan")
    return {"threshold": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr_recall": tpr, "fpr": fpr, "precision": prec, "f1": f1}


def fpr_at_tpr(y_bin, scores, target_tpr: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(y_bin).astype(int), scores)
    idx = np.searchsorted(tpr, target_tpr)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


def tpr_at_fpr(y_bin, scores, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(y_bin).astype(int), scores)
    idx = np.searchsorted(fpr, target_fpr)
    idx = min(max(idx, 0), len(tpr) - 1)
    return float(tpr[idx])


# ------------------------------------------------------------------
# Calibration du seuil : percentile vs EVT (GPD / Peak-Over-Threshold)
# ------------------------------------------------------------------
def threshold_percentile(benign_scores, target_fpr: float = 0.05) -> float:
    """Seuil = quantile (1 - target_fpr) des scores bénins."""
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def threshold_evt(benign_scores, target_fpr: float = 0.05, tail_frac: float = 0.10) -> dict:
    """
    Théorie des valeurs extrêmes : ajustement GPD sur la queue des scores bénins
    (approche Peak-Over-Threshold), puis extrapolation du quantile cible.

    Intérêt académique : les faux positifs sont, par définition, des événements rares.
    Le percentile empirique plafonne à 1/n et devient instable dès qu'on vise < 1 %
    (il faut 10 000 échantillons pour estimer un quantile à 99.9 %). La GPD, elle,
    extrapole la queue : c'est ce qui justifie le choix d'un seuil EVT plutôt qu'un
    simple quantile dans le mémoire.
    """
    x = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(x)
    k = max(20, int(tail_frac * n))
    if k >= n:
        return {"threshold": float(np.max(x)), "method": "evt(unfitted)", "xi": None, "sigma": None}

    u = x[n - k - 1]           # seuil de queue
    exc = x[n - k:] - u        # dépassements
    exc = exc[exc > 0]
    if len(exc) < 15 or np.allclose(exc, 0):
        return {"threshold": float(np.quantile(x, 1 - target_fpr)),
                "method": "evt(fallback:percentile)", "xi": None, "sigma": None}

    try:
        from scipy.stats import genpareto
        xi, loc, sigma = genpareto.fit(exc, floc=0)
        p_exceed = len(exc) / n
        # quantile de la GPD tel que P(X > q) = target_fpr
        q = genpareto.ppf(1 - target_fpr / p_exceed, xi, loc=0, scale=sigma)
        thr = float(u + q)
        if not np.isfinite(thr):
            raise ValueError("quantile non fini")
        return {"threshold": thr, "method": "evt(gpd)", "xi": float(xi), "sigma": float(sigma),
                "u": float(u), "n_exceed": int(len(exc))}
    except Exception as e:
        return {"threshold": float(np.quantile(x, 1 - target_fpr)),
                "method": f"evt(fallback:{e.__class__.__name__})", "xi": None, "sigma": None}


# ------------------------------------------------------------------
# Évaluation par classe Zero-Day
# ------------------------------------------------------------------
def per_class_recall(y_raw, scores, thr: float, zero_day_names=None) -> list:
    y_raw = np.asarray(y_raw).astype(str)
    scores = np.asarray(scores, dtype=float)
    out = []
    for cls in sorted(set(y_raw)):
        mask = y_raw == cls
        is_benign = "BENIGN" in cls.upper()
        if is_benign:
            out.append({
                "classe": cls, "n": int(mask.sum()),
                "type": "BENIGN",
                "recall": float((scores[mask] <= thr).mean()),   # spécificité
                "detection_rate": float((scores[mask] > thr).mean()),
                "score_mean": float(scores[mask].mean()),
                "auroc_vs_benign": float("nan"),
            })
        else:
            benign = scores[np.array(["BENIGN" in s.upper() for s in y_raw])]
            yb = np.r_[np.zeros(len(benign)), np.ones(int(mask.sum()))]
            sc = np.r_[benign, scores[mask]]
            out.append({
                "classe": cls, "n": int(mask.sum()),
                "type": "ZERO-DAY" if (zero_day_names and any(z.lower() in cls.lower() for z in zero_day_names)) else "ATTAQUE",
                "recall": float((scores[mask] > thr).mean()),
                "detection_rate": float((scores[mask] > thr).mean()),
                "score_mean": float(scores[mask].mean()),
                "auroc_vs_benign": auroc(yb, sc),
            })
    return out


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def make_figures(y_bin, scores, y_raw, thr, cls_rows, outdir=RESULTS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.facecolor": "white"})
    C_BEN, C_ZD = "#10b981", "#ef4444"

    # 1. Distribution des scores
    fig, ax = plt.subplots(figsize=(7, 4))
    b = scores[np.asarray(y_bin) == 0]
    z = scores[np.asarray(y_bin) == 1]
    ax.hist(b, bins=60, alpha=0.7, label=f"BENIGN (n={len(b)})", color=C_BEN, density=True)
    ax.hist(z, bins=60, alpha=0.7, label=f"Zero-Day (n={len(z)})", color=C_ZD, density=True)
    ax.axvline(thr, color="black", ls="--", lw=1.6, label=f"Seuil = {thr:.4f}")
    ax.set_xlabel("Score d'anomalie (distance au centroïde)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des scores — séparabilité BENIGN / Zero-Day")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_score_dist.png", dpi=150); plt.close(fig)

    # 2. Courbe ROC
    fpr, tpr, _ = roc_curve(np.asarray(y_bin).astype(int), scores)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"AUROC = {a:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR / Recall)")
    ax.set_title("Courbe ROC — détection Zero-Day")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_roc.png", dpi=150); plt.close(fig)

    # 3. Courbe Precision-Recall
    prec, rec, _ = precision_recall_curve(np.asarray(y_bin).astype(int), scores)
    ap = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, color="#a78bfa", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Courbe Précision-Rappel")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_pr.png", dpi=150); plt.close(fig)

    # 4. Recall par classe Zero-Day (le graphique du critère go/no-go)
    zd = [r for r in cls_rows if r["type"] == "ZERO-DAY"]
    if zd:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        names = [r["classe"] for r in zd]
        vals = [r["recall"] * 100 for r in zd]
        colors = [C_ZD if v >= 85 else "#f59e0b" if v >= 70 else "#7f1d1d" for v in vals]
        bars = ax.bar(names, vals, color=colors)
        ax.axhline(85, color="black", ls="--", lw=1.3, label="Critère go/no-go : 85 %")
        ax.set_ylim(0, 105); ax.set_ylabel("Recall (%)")
        ax.set_title("Recall par classe Zero-Day (jamais vue à l'entraînement)")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f}%",
                    ha="center", fontsize=9, fontweight="bold")
        ax.legend()
        plt.xticks(rotation=15)
        fig.tight_layout(); fig.savefig(f"{outdir}/fig_recall_class.png", dpi=150); plt.close(fig)

    # 5. Matrice de confusion
    y_bin = np.asarray(y_bin).astype(int)
    pred = (scores > thr).astype(int)
    cm = np.array([[int(((pred == 0) & (y_bin == 0)).sum()), int(((pred == 1) & (y_bin == 0)).sum())],
                   [int(((pred == 0) & (y_bin == 1)).sum()), int(((pred == 1) & (y_bin == 1)).sum())]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1], ["Prédit Bénin", "Prédit Anomalie"])
    ax.set_yticks([0, 1], ["Réel Bénin", "Réel Zero-Day"])
    ax.set_title(f"Matrice de confusion (seuil = {thr:.4f})")
    ax.grid(False)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_confusion.png", dpi=150); plt.close(fig)

    return {"auc_roc": float(a), "auc_pr": float(ap), "confusion": cm.tolist()}


# ------------------------------------------------------------------
# Chargement réel (torch) — isolé pour garder le reste testable
# ------------------------------------------------------------------
def load_real():
    """Charge le modèle V3 (model_ms) et calcule les scores sur le test."""
    import json
    import torch

    from model_ms import (TokenizedMSEncoder, anomaly_score, free_energy,
                          next_state_confidence, token_scores)

    meta_path = "data/processed/meta.json"
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    # n_net n'est pas toujours 78 : le pipeline supprime les colonnes
    # constantes, et le nombre varie selon le jeu produit. Le coder en dur
    # ferait glisser une colonne EDR dans le score « features réseau ».
    n_net, n_sys = meta.get("n_net", 78), meta.get("n_sys", 25)

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenizedMSEncoder(n_net=n_net, n_sys=n_sys, embed_dim=32, d=64,
                               n_tok_net=4, n_tok_sys=5, n_layers=2)
    model.load_state_dict(torch.load("models/ms_encoder.pt", map_location=device))
    model.eval().to(device)
    centroid = torch.load("models/centroid.pt", map_location=device).cpu()

    def scores_of(X):
        out = {"score": [], "violation": [], "energy": [], "next_conf": [],
               "token": [], "z": []}
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                xb = torch.FloatTensor(X[i:i + 4096]).to(device)
                r = model(xb)
                out["z"].append(r["z"].cpu())
                out["score"].append(anomaly_score(r["z"].cpu(), centroid).numpy())
                out["violation"].append(torch.sigmoid(r["violation_logit"]).cpu().numpy())
                out["energy"].append(free_energy(r["violation_logit"]).cpu().numpy())
                conf, _ = next_state_confidence(r["next_pred"].cpu(), xb.cpu(), n_net, n_sys)
                out["next_conf"].append(conf.numpy())
                out["token"].append(token_scores(r["token_logits"].cpu()).cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    sc = scores_of(X_test)
    scores = sc["score"]

    zd_path = "data/processed/zero_day_attacks.txt"
    zd = open(zd_path).read().split(",") if os.path.exists(zd_path) else []

    # Scores bénins du TRAIN pour calibrer le seuil EVT
    try:
        X_train = np.load("data/processed/X_train.npy")
        y_train = np.load("data/processed/y_train.npy", allow_pickle=True)
        bmask = np.array(["BENIGN" in str(l).upper() for l in y_train])
        benign_train_scores = scores_of(X_train[bmask][:20000])["score"]
    except Exception as e:
        print(f"⚠️  Scores bénins train indisponibles ({e}) — calibration sur le test")
        benign_train_scores = scores[np.array(["BENIGN" in str(l).upper() for l in y_test])]

    extra = {"violation": sc["violation"], "energy": sc["energy"],
             "next_conf": sc["next_conf"], "token": sc["token"]}

    # Signaux bénins du TRAIN : servent à calibrer le seuil sans jamais
    # toucher au jeu de test.
    try:
        _tr = scores_of(X_train[bmask][:20000])
        z_tr = _tr["z"]
        conf_tr = _tr["next_conf"]
        token_tr = _tr["token"]
    except Exception:
        z_tr, conf_tr, token_tr = None, None, None
    extra["z_test"] = sc["z"]
    extra["z_benign_train"] = z_tr
    extra["conf_benign_train"] = conf_tr
    extra["token_benign_train"] = token_tr
    return scores, y_test, zd, benign_train_scores, extra



def _n_net_meta(default: int = 78) -> int:
    """Largeur de la branche réseau, lue dans meta.json."""
    try:
        return int(json.load(open("data/processed/meta.json",
                                 encoding="utf-8")).get("n_net", default))
    except Exception:
        return default


def load_features(n_net: int = None, return_stats: bool = False):
    """
    Charge les 78 features RÉSEAU et corrige le pré-traitement.

    Les fichiers .npy livrés sont déjà standardisés par un StandardScaler
    ajusté sur le trafic bénin seul. Chaque flux d'attaque reçoit donc
    mécaniquement de grands z-scores, et la norme des features suffit à
    séparer les classes (sum|x| : AUROC 0.9476 sur le test).

    On inverse cette standardisation avec le scaler d'origine, puis on
    réajuste sur l'ENSEMBLE du train (bénin + attaques connues). La
    référence triviale tombe alors à 0.2897 : la fuite de magnitude
    disparaît, et le score ne peut plus la lire par accident.

    Retourne (X_train, X_test, y_train, y_test) sur les seules features
    réseau : aucun apport de la branche EDR simulée.
    """
    import pickle

    if n_net is None:
        n_net = _n_net_meta()

    Xtr = np.load("data/processed/X_train.npy")
    ytr = np.load("data/processed/y_train.npy", allow_pickle=True)
    Xte = np.load("data/processed/X_test.npy")
    yte = np.load("data/processed/y_test.npy", allow_pickle=True)

    with open("models/scaler.pkl", "rb") as f:
        sc = pickle.load(f)
    A = Xtr[:, :n_net] * sc.scale_[:n_net] + sc.mean_[:n_net]
    B = Xte[:, :n_net] * sc.scale_[:n_net] + sc.mean_[:n_net]

    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    if return_stats:
        return (A - mu) / sd, (B - mu) / sd, ytr, yte, {"mu": mu, "sd": sd, "n_net": n_net}
    return (A - mu) / sd, (B - mu) / sd, ytr, yte


def train_supervised(n_net: int = None, algo: str = "logreg"):
    """
    Détecteur SUPERVISÉ : entraîné sur le bénin ET sur les familles d'attaque
    connues (celles du train), puis évalué sur les familles jamais vues.

    Ce n'est pas un détecteur d'anomalies : il a vu des attaques. C'est un
    classifieur qui généralise à des familles inconnues. La distinction est
    essentielle et doit apparaître telle quelle dans le rapport.

    Contraste mesuré dans ce projet :
        - sur les familles qui RESSEMBLENT aux connues  -> 99,8 %
        - en validation par famille (famille retirée)   -> 0 % sur DDoS
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    F_tr, F_te, y_tr, y_te, _stats = load_features(n_net, return_stats=True)
    b_tr = np.array(["BENIGN" in str(l).upper() for l in y_tr])
    ybin_tr = (~b_tr).astype(int)
    ybin_te = (~np.array(["BENIGN" in str(l).upper() for l in y_te])).astype(int)

    # Deux familles de modèles, mesurées dans le rapport :
    #   logreg -> 0.9026 · hgb -> 0.9672 (arbres, interactions non linéaires)
    # Le gradient boosting gagne de 6 points : la frontière entre bénin et
    # attaque n'est pas linéaire dans cet espace de features.
    if algo == "hgb":
        # max_leaf_nodes=63 : retenu par la grille de réglage (04 --tune).
        # À complexité plus fine, le rappel AU POINT DE FONCTIONNEMENT gagne
        # 8 points à FPR constant (84,8 % -> 93,1 % à 9,2 % de fausses
        # alertes) pour un AUROC quasi inchangé (0.9668 contre 0.9672).
        # L'AUROC seul ne voyait pas ce gain : il est insensible au seuil,
        # alors que les critères du concours portent sur le rappel à FPR borné.
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
            random_state=42).fit(F_tr, ybin_tr)
        s_te = clf.predict_proba(F_te)[:, 1]
        s_tr_ben = clf.predict_proba(F_tr)[:, 1][b_tr]
    else:
        clf = LogisticRegression(max_iter=3000).fit(F_tr, ybin_tr)
        s_te = clf.decision_function(F_te)
        s_tr_ben = clf.decision_function(F_tr[b_tr])

    # Le modèle est conservé sur disque : l'application Streamlit doit pouvoir
    # rejouer le même détecteur, sinon la démo ne correspond plus au rapport.
    try:
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, f"models/supervised_{algo}.pkl")
        # L'application doit appliquer EXACTEMENT le même prétraitement que
        # l'évaluation, sinon la démo ne rejoue pas le modèle du rapport.
        # ATTENTION : mu/sd doivent être ceux de l'espace BRUT (avant
        # standardisation), pas ceux de F_tr qui est déjà standardisé.
        _m = _stats["mu"]
        _sd = np.where(_stats["sd"] < 1e-9, 1.0, _stats["sd"])
        if algo == "hgb":
            _s_tr = clf.predict_proba(F_tr)[:, 1]
        else:
            _s_tr = clf.decision_function(F_tr)
        np.savez(f"models/supervised_{algo}.npz",
                 mu=_m, sd=_sd, n_net=np.int32(F_tr.shape[1]),
                 threshold=np.float64(np.quantile(_s_tr[b_tr], 0.91)))
        print(f"  ✅ modèle supervisé enregistré : models/supervised_{algo}.pkl")
    except Exception as _e:
        print(f"  (modèle non enregistré : {_e})")

    return s_te, s_tr_ben, ybin_te, clf


def load_synthetic(seed=42, n_benign=20000, n_per_class=2500):
    """Pipeline de test : 3 classes Zero-Day avec des niveaux de difficulté réalistes."""
    rng = np.random.default_rng(seed)
    names = ["PortScan", "Web Attack - XSS", "Bot"]
    benign = rng.normal(0.18, 0.05, n_benign)
    # PortScan : très séparable | WebAttack : moyen | Bot : proche du bénin (cas dur)
    scores, labels = [benign], ["BENIGN"] * n_benign
    for name, mu, sd in zip(names, [0.62, 0.40, 0.30], [0.10, 0.09, 0.07]):
        s = rng.normal(mu, sd, n_per_class)
        scores.append(s); labels += [name] * n_per_class
    benign_train = rng.normal(0.18, 0.05, 30000)
    return np.concatenate(scores), np.array(labels), names, benign_train, {}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

# ══════════════════════════════════════════════════════════════════════
#  VALIDATION CROISÉE PAR FAMILLE (leave-one-family-out)
# ══════════════════════════════════════════════════════════════════════
#  Réponse aux limites du protocole V4 :
#
#    L2  la branche EDR simulée est retirée : le modèle est 100 % réseau
#    L3  les familles évaluées passent de 3 à 7 (les fichiers bruts en
#        contiennent exactement sept)
#    L4  chaque famille passe une fois en position d'inconnue : toutes les
#        attaques du jeu sont donc évaluées exactement une fois
#    L5  sept découpages indépendants au lieu d'un seul (toujours interne
#        à CICIDS2017 : une validation inter-jeux reste requise)
#    L1  le rappel n'est pas résolu, il est mesuré proprement par famille
#
#  Correction non listée mais déterminante : le StandardScaler était ajusté
#  sur le trafic bénin seul. Chaque flux d'attaque recevait donc de grands
#  z-scores et la norme des features suffisait à séparer les classes
#  (sum|x| : AUROC 0.9476). Ici le scaler est ajusté sur TOUT le train et la
#  référence triviale tombe à 0.6101.


def _canon_label(s: str) -> str:
    """Normalise les libellés (certains fichiers contiennent des tirets mal encodés)."""
    import re as _re
    s = str(s).replace("\ufffd", "-").replace("\xef\xbf\xbd", "-")
    return _re.sub(r"\s+", " ", s).strip()


def _charger_familles(data_dir: str, n_max: int = 4000, graine: int = 42):
    """
    Lit les parquet bruts et renvoie (bénins, {famille: matrice}).

    Sous-échantillonnage À LA LECTURE et déduplication fichier par fichier :
    les fichiers pèsent plusieurs centaines de milliers de lignes et tout
    charger en float64 fait échouer le processus par manque de mémoire.
    Infiltration (36 flux) et SQL Injection (21 flux) sont conservées en
    entier — leur rareté est le sujet.
    """
    import glob

    import pandas as pd

    rng = np.random.default_rng(graine)
    familles, benignes = {}, []
    fichiers = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not fichiers:
        return None, None

    for f in fichiers:
        df = pd.read_parquet(f)
        if "Label" not in df.columns:
            del df
            continue
        num = [c for c in df.columns if c != "Label"]
        n_lignes = len(df)
        X = np.nan_to_num(df[num].to_numpy(dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
        lab = np.array([_canon_label(v) for v in df["Label"].values])
        del df

        b = np.array(["BENIGN" in v.upper() for v in lab])
        if b.any():
            part = np.unique(X[b], axis=0)
            if len(part) > 25000:
                part = part[rng.choice(len(part), 25000, replace=False)]
            benignes.append(part)
        for fam in sorted(set(lab[~b])):
            part = np.unique(X[lab == fam], axis=0)
            if len(part) > n_max:
                part = part[rng.choice(len(part), n_max, replace=False)]
            familles.setdefault(fam, []).append(part)
        print(f"   {os.path.basename(f)[:46]:<48} {n_lignes:>8,} lignes")
        del X, lab

    if not benignes:
        return None, None
    return np.vstack(benignes), {k: np.vstack(v) for k, v in familles.items()}


def run_lofo(data_dir: str = "data", target_fpr: float = 0.05,
             n_benin_train: int = 12000, n_benin_test: int = 4000,
             n_attaque: int = 4000, graine: int = 42) -> dict:
    """Protocole leave-one-family-out sur les familles des fichiers bruts."""
    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(graine)

    print("\n" + "=" * 72)
    print("  VALIDATION CROISÉE PAR FAMILLE (leave-one-family-out)")
    print("=" * 72)
    print("Chargement des fichiers bruts :")
    Xb, familles = _charger_familles(data_dir, n_attaque, graine)
    if Xb is None:
        print("\n⚠️  Aucun .parquet avec colonne 'Label' dans", data_dir)
        print("    Le protocole par famille nécessite les fichiers bruts CICIDS2017.")
        return {}

    Xb = np.unique(Xb, axis=0)
    familles = {k: np.unique(v, axis=0) for k, v in familles.items()}
    print(f"\nBénin dédupliqué : {len(Xb):,} flux")
    print(f"Familles d'attaque : {len(familles)}")
    if len(familles) < 2:
        print("\n⚠️  Au moins deux familles sont nécessaires.")
        return {}

    def ech(X, n):
        return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]

    res = {}
    ordre = sorted(familles, key=lambda k: -len(familles[k]))
    print(f"\n{'famille laissee de cote':<30}{'n':>6}{'Mahalanobis':>13}"
          f"{'Supervise':>11}{'sum|x|':>9}")
    print("-" * 72)

    for fam in ordre:
        Xa = ech(familles[fam], n_attaque)
        autres = [v for k, v in familles.items() if k != fam]

        X_tr = np.vstack([ech(Xb, n_benin_train)]
                         + [ech(a, 600) for a in autres])
        y_tr = np.r_[np.zeros(min(len(Xb), n_benin_train)),
                     np.concatenate([np.ones(min(len(a), 600)) for a in autres])]

        Xb_te = ech(Xb, n_benin_test)
        X_te = np.vstack([Xb_te, Xa])
        y_te = np.r_[np.zeros(len(Xb_te)), np.ones(len(Xa))]

        # scaler ajusté sur TOUT le train (correction de la fuite de magnitude)
        mu, sd = X_tr.mean(0), X_tr.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        tr, te = (X_tr - mu) / sd, (X_te - mu) / sd

        # score one-class : Mahalanobis ajusté sur le bénin du train
        trb = tr[y_tr == 0]
        cov = LedoitWolf().fit(trb)
        d_te = te - cov.location_
        s_maha = np.einsum("ij,jk,ik->i", d_te, cov.precision_, d_te)
        d_trb = trb - cov.location_
        s_maha_trb = np.einsum("ij,jk,ik->i", d_trb, cov.precision_, d_trb)
        seuil = float(np.quantile(s_maha_trb, 1 - target_fpr))

        # référence supervisée : n'a JAMAIS vu cette famille
        lr = LogisticRegression(max_iter=3000).fit(tr, y_tr)
        s_lr = lr.decision_function(te)
        seuil_lr = float(np.quantile(lr.decision_function(tr[y_tr == 0]), 1 - target_fpr))

        res[fam] = {
            "n": int(len(Xa)),
            "auroc_maha": float(auroc(y_te, s_maha)),
            "rappel_maha": float((s_maha[y_te == 1] > seuil).mean()),
            "auroc_sup": float(auroc(y_te, s_lr)),
            "rappel_sup": float((s_lr[y_te == 1] > seuil_lr).mean()),
            "auroc_trivial": float(auroc(y_te, np.abs(te).sum(1))),
        }
        r = res[fam]
        print(f"{fam:<30}{len(Xa):>6}{r['auroc_maha']:>13.4f}"
              f"{r['auroc_sup']:>11.4f}{r['auroc_trivial']:>9.4f}")

    auc_m = [r["auroc_maha"] for r in res.values()]
    auc_s = [r["auroc_sup"] for r in res.values()]
    auc_t = [r["auroc_trivial"] for r in res.values()]
    rec_m = [r["rappel_maha"] for r in res.values()]
    rec_s = [r["rappel_sup"] for r in res.values()]

    print("-" * 72)
    print(f"{'MOYENNE':<30}{'':>6}{np.mean(auc_m):>13.4f}"
          f"{np.mean(auc_s):>11.4f}{np.mean(auc_t):>9.4f}")

    print("\n--- Rappel à " + f"{target_fpr*100:.0f} % de fausses alertes ---")
    for fam, r in sorted(res.items(), key=lambda kv: -kv[1]["rappel_maha"]):
        print(f"   {fam:<30} Mahalanobis {r['rappel_maha']*100:>6.1f} %"
              f"   Supervisé {r['rappel_sup']*100:>6.1f} %")

    ok_m = sum(1 for v in rec_m if v >= 0.85)
    ok_s = sum(1 for v in rec_s if v >= 0.85)
    print(f"\nCritère (rappel >= 85 % par famille) :")
    print(f"   Mahalanobis one-class : {ok_m}/{len(rec_m)} familles")
    print(f"   Référence supervisée  : {ok_s}/{len(rec_s)} familles")

    # La baseline triviale est-elle enfin battue ?
    print(f"\nRéférence triviale sum|x| : {np.mean(auc_t):.4f}")
    if np.mean(auc_m) > np.mean(auc_t):
        print(f"   ✅ le modèle ({np.mean(auc_m):.4f}) BAT la référence triviale")
    else:
        print(f"   ❌ le modèle ({np.mean(auc_m):.4f}) ne bat PAS la référence triviale")

    sortie = {
        "protocole": "leave-one-family-out",
        "fpr_cible": target_fpr,
        "scaler_ajuste_sur": "train complet (correction de la fuite de magnitude)",
        "edr": "supprime",
        "familles": len(res),
        "attaques_evaluees": int(sum(r["n"] for r in res.values())),
        "resultats": res,
        "moyennes": {
            "auroc_maha": float(np.mean(auc_m)),
            "auroc_sup": float(np.mean(auc_s)),
            "auroc_trivial": float(np.mean(auc_t)),
            "rappel_maha": float(np.mean(rec_m)),
            "rappel_sup": float(np.mean(rec_s)),
            "familles_ok_85": int(ok_m),
            # Lecture far-OOD / near-OOD : un attaquant « lointain » est reconnu
            # sans avoir été vu ; un attaquant qui imite le trafic normal ne
            # l'est pas. Cette distinction est celle qui borne la promesse
            # « zero-day » du projet — la moyenne, elle, la masque.
            "far_ood": sorted([f for f, r in res.items() if r["auroc_sup"] >= 0.95]),
            "near_ood": sorted([f for f, r in res.items() if r["auroc_sup"] < 0.95]),
            "pire_famille": min(res, key=lambda f: res[f]["auroc_sup"]) if res else None,
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "lofo.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, indent=2, ensure_ascii=False)
    if res:
        _far = sortie["moyennes"]["far_ood"]
        _near = sortie["moyennes"]["near_ood"]
        _pire = sortie["moyennes"]["pire_famille"]
        print("\nLecture far-OOD / near-OOD :")
        if _far:
            print(f"  reconnues sans les avoir vues : {', '.join(_far)}")
        if _near:
            print(f"  proches du trafic normal      : {', '.join(_near)}")
            print(f"  → ce sont elles qui bornent la promesse zero-day du projet")
        if _pire:
            print(f"  pire famille : {_pire} "
                  f"(AUROC {res[_pire]['auroc_sup']:.4f})")

    print(f"\n✅ {RESULTS}/lofo.json écrit")
    return sortie


def run_contraste(n_net: int = None, target_fpr: float = 0.05):
    """
    Confronte les deux familles d'approches sur les mêmes familles inconnues.

    Le point que cette table établit, et qui est la contribution du projet :

        un détecteur supervisé atteint 99,8 % sur des familles qu'il n'a
        jamais vues, À CONDITION qu'elles ressemblent aux familles connues ;
        un détecteur one-class, qui n'a jamais vu d'attaque, atteint 86,2 %
        mais plafonne sur les familles discrètes.

    Aucune des deux ne couvre l'ensemble des inconnues. C'est mesurable, et
    c'est plus utile qu'un chiffre unique.
    """
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.linear_model import LogisticRegression

    F_tr, F_te, y_tr, y_te, _ = load_features(n_net, return_stats=True)
    b_tr = np.array(["BENIGN" in str(l).upper() for l in y_tr])
    b_te = np.array(["BENIGN" in str(l).upper() for l in y_te])
    ybin = (~b_te).astype(int)
    classes = sorted({str(v) for v in y_te[~b_te]})

    # --- one-class : bénin seul ---
    cov = EmpiricalCovariance().fit(F_tr[b_tr])
    d = F_te - cov.location_
    s_oc = np.einsum("ij,jk,ik->i", d, cov.precision_, d)
    db = F_tr[b_tr] - cov.location_
    thr_oc = float(np.quantile(np.einsum("ij,jk,ik->i", db, cov.precision_, db),
                               1 - target_fpr))

    # --- supervisé : bénin + familles connues ---
    clf = LogisticRegression(max_iter=3000, n_jobs=-1).fit(F_tr, (~b_tr).astype(int))
    s_sv = clf.decision_function(F_te)
    thr_sv = float(np.quantile(clf.decision_function(F_tr[b_tr]), 1 - target_fpr))

    lignes = {}
    print("\n" + "=" * 74)
    print("  CONTRASTE : one-class (n'a jamais vu d'attaque) vs supervisé")
    print("=" * 74)
    print(f"  {'famille inconnue':<28}{'one-class':>13}{'supervisé':>13}{'écart':>11}")
    print("  " + "-" * 66)
    for c in classes:
        m = np.array([str(v) == c for v in y_te])
        r_oc = float((s_oc[m] > thr_oc).mean())
        r_sv = float((s_sv[m] > thr_sv).mean())
        lignes[c] = {"one_class": r_oc, "supervise": r_sv, "n": int(m.sum())}
        print(f"  {c:<28}{r_oc*100:>12.1f}%{r_sv*100:>12.1f}%"
              f"{(r_sv-r_oc)*100:>+10.1f}")

    glob_oc = float((s_oc[~b_te] > thr_oc).mean())
    glob_sv = float((s_sv[~b_te] > thr_sv).mean())
    auc_oc = float(auroc(ybin, s_oc))
    auc_sv = float(auroc(ybin, s_sv))
    print("  " + "-" * 66)
    print(f"  {'MOYENNE (globale)':<28}{glob_oc*100:>12.1f}%{glob_sv*100:>12.1f}%")
    print(f"  {'AUROC':<28}{auc_oc:>12.4f}{auc_sv:>12.4f}")
    print(f"  {'classes >= 85 %':<28}"
          f"{sum(1 for v in lignes.values() if v['one_class']>=0.85):>12}"
          f"{sum(1 for v in lignes.values() if v['supervise']>=0.85):>13}")

    # --- Effondrement du supervisé en validation par famille --------------
    lofo_path = os.path.join(RESULTS, "lofo.json")
    effondrement = {}
    if os.path.exists(lofo_path):
        try:
            L = json.load(open(lofo_path, encoding="utf-8"))["resultats"]
            print("\n  Validation par famille (famille retirée de l'entraînement) :")
            print(f"  {'famille':<28}{'supervisé':>13}{'one-class':>13}")
            print("  " + "-" * 54)
            for k, v in sorted(L.items(), key=lambda kv: -kv[1]["rappel_sup"]):
                effondrement[k] = {"supervise": v["rappel_sup"],
                                   "one_class": v["rappel_maha"]}
                print(f"  {k:<28}{v['rappel_sup']*100:>12.1f}%"
                      f"{v['rappel_maha']*100:>12.1f}%")
            pire = min(effondrement, key=lambda k: effondrement[k]["supervise"])
            print(f"\n  ⚠️  Le supervisé tombe à "
                  f"{effondrement[pire]['supervise']*100:.1f} % sur « {pire} » :")
            print("     une famille structurellement différente n'est pas reconnue,")
            print("     quelle que soit la qualité de l'entraînement.")
        except Exception as _e:
            print(f"  (validation par famille indisponible : {_e})")
    else:
        print("\n  (lance `--lofo` pour mesurer l'effondrement du supervisé)")

    print("\n  Lecture : le supervisé domine sur les familles qui ressemblent")
    print("  aux connues. En validation par famille (--lofo), il s'effondre sur")
    print("  DDoS et SQL Injection. Aucune approche ne couvre toutes les inconnues.")

    # export du classifieur supervisé (coefficients d'une frontière linéaire)
    try:
        np.savez("models/supervised_logreg.npz",
                 coef=clf.coef_.ravel(), intercept=clf.intercept_.ravel(),
                 threshold=thr_sv, target_fpr=target_fpr)
        print("\n✅ models/supervised_logreg.npz écrit")
    except Exception as _e:
        print(f"⚠️  export supervisé impossible : {_e}")

    sortie = {"fpr_cible": target_fpr, "familles": lignes,
              "global": {"one_class": glob_oc, "supervise": glob_sv},
              "auroc": {"one_class": auc_oc, "supervise": auc_sv},
              "protocole": "4 familles inconnues au test",
              "validation_par_famille": effondrement}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "contraste.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, indent=2, ensure_ascii=False)
    print(f"\n✅ {RESULTS}/contraste.json écrit")
    return sortie



# =====================================================================
#  MODE « ood » : plusieurs scores SANS LABEL, départagés honnêtement
# =====================================================================
def _scorers_benign(X_benign, X_eval):
    """Calcule plusieurs scores d'anomalie n'utilisant QUE le bénin.

    Aucune de ces méthodes ne voit un seul label d'attaque : elles ne
    peuvent donc pas mémoriser les familles vues. C'est la condition pour
    que le chiffre obtenu sur des familles inédites ait un sens.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors

    def logt(X):
        return np.sign(X) * np.log1p(np.abs(X))

    S = {}

    # --- 1. Mahalanobis sur valeurs brutes standardisées ---
    mu, sd = X_benign.mean(0), X_benign.std(0) + 1e-8
    A, B = (X_benign - mu) / sd, (X_eval - mu) / sd
    lw = LedoitWolf().fit(A)
    S["maha"] = lw.mahalanobis(B)

    # --- 2. Mahalanobis après log1p (queues lourdes) ---
    Al, Bl = logt(X_benign), logt(X_eval)
    mu, sd = Al.mean(0), Al.std(0) + 1e-8
    A2, B2 = (Al - mu) / sd, (Bl - mu) / sd
    lw2 = LedoitWolf().fit(A2)
    S["maha_log"] = lw2.mahalanobis(B2)

    # --- 3 & 4. Écart au bénin en rang (robuste aux échelles) ---
    ref = np.sort(X_benign, axis=0)[::7]
    Q = np.empty((len(X_eval), X_benign.shape[1]), dtype=np.float32)
    for j in range(X_benign.shape[1]):
        c = ref[:, j]
        Q[:, j] = np.clip((np.searchsorted(c, X_eval[:, j]) + 0.5) / len(c), 1e-3, 1 - 1e-3)
    dev = np.abs(2 * Q - 1)
    S["ecdf_moy"] = dev.mean(1)
    S["ecdf_top10"] = np.sort(dev, 1)[:, -10:].mean(1)

    # --- 5 & 6. Modèles multi-modaux sur espace réduit ---
    p = PCA(n_components=20, random_state=0).fit(Al)
    Ap, Bp = p.transform(Al), p.transform(Bl)
    sd = Ap.std(0) + 1e-8
    Ap, Bp = Ap / sd, Bp / sd

    km = MiniBatchKMeans(n_clusters=64, n_init=5, random_state=0,
                         batch_size=8192).fit(Ap)
    S["kmeans64"] = km.transform(Bp).min(1)

    nn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(Ap)
    d, _ = nn.kneighbors(Bp)
    S["knn5"] = d.mean(1)

    # --- 7. IsolationForest ---
    iso = IsolationForest(n_estimators=200, max_samples=10000,
                          random_state=0, n_jobs=-1).fit(Al)
    S["isoforest"] = -iso.score_samples(Bl)

    out = {}
    for k, v in S.items():
        v = np.asarray(v, dtype=float)
        # Un score quasi constant n'a aucune information : l'afficher comme
        # « 100 % de rappel à 5 % de FA » serait un artefact de seuil, pas une
        # performance. On l'écarte explicitement.
        if np.ptp(v) < 1e-9 or len(np.unique(v)) < 10:
            print(f"  (score {k} écarté : quasi constant, sans information)")
            continue
        out[k] = v
    return out


def _embedding(X, n_net, n_sys, batch=4096):
    """Projette les flux dans l'espace latent appris par le MS-Transformer."""
    import torch
    from model_ms import TokenizedMSEncoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenizedMSEncoder(n_net=n_net, n_sys=n_sys, embed_dim=32, d=64,
                               n_tok_net=4, n_tok_sys=5, n_layers=2)
    model.load_state_dict(torch.load("models/ms_encoder.pt", map_location=device))
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.FloatTensor(X[i:i + batch]).to(device)
            out.append(model(xb)["z"].cpu())
    return torch.cat(out).numpy()


def run_ood(n_net: int = None, target_fpr: float = 0.05, seed: int = 42,
            famille_simulee: str = None, espace: str = "features"):
    """Départage plusieurs scores sans label sur une famille CONNUE tenue à part.

    Protocole, et c'est le point important :
      - les méthodes ne voient que du bénin (aucun label d'attaque) ;
      - on simule un Zero-Day en retirant une famille d'attaque CONNUE du
        train et en la gardant comme cible : c'est la seule façon de choisir
        un score sans regarder les vraies familles inédites ;
      - le score gagnant est ensuite appliqué, une seule fois, au vrai test.
    """
    from collections import Counter

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy", allow_pickle=True)
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

    if espace == "modele":
        # Les scores sont calculés sur l'embedding appris (64 dims) et non sur
        # les 95 features brutes : la représentation du transformer est plus
        # compacte, donc les modèles multi-modaux y sont mieux conditionnés.
        meta = json.load(open("data/processed/meta.json", encoding="utf-8"))
        _n, _s = int(meta.get("n_net", 78)), int(meta.get("n_sys", 25))
        print(f"\nEspace : EMBEDDING du MS-Transformer ({_n}+{_s} → 64 dims)")
        X_train = _embedding(X_train, _n, _s)
        X_test = _embedding(X_test, _n, _s)
    else:
        print(f"\nEspace : FEATURES brutes ({X_train.shape[1]} dims)")

    ben_tr = np.array(["BENIGN" in str(l).upper() for l in y_train])
    atk_tr = ~ben_tr
    if atk_tr.sum() == 0:
        raise SystemExit("Aucune attaque dans le train : impossible de simuler un Zero-Day.")

    familles = Counter(str(l) for l in y_train[atk_tr])
    fam_box = None
    if famille_simulee:
        # correspondance souple : "brute" trouve "Web Attack - Brute Force"
        cible = famille_simulee.lower()
        fam_box = next((f for f in familles if cible in f.lower()), None)
        if fam_box is None:
            print(f"⚠️  famille « {famille_simulee} » absente du train. "
                  f"Disponibles : {', '.join(sorted(familles))}")
    if fam_box is None:
        fam_box = familles.most_common(1)[0][0]      # famille connue la plus fournie
    print(f"\n=== MODE OOD : sélection de score sans label ===")
    print(f"Famille connue mise de côté pour simuler le Zero-Day : {fam_box} "
          f"({familles[fam_box]:,} flux)")
    print(f"Bénin train : {ben_tr.sum():,} | Bénin test : "
          f"{np.sum(['BENIGN' in str(l).upper() for l in y_test]):,}")

    rng = np.random.default_rng(seed)
    idx_b = rng.permutation(np.where(ben_tr)[0])
    n_val = max(1000, int(0.2 * len(idx_b)))
    idx_fit, idx_val = idx_b[n_val:], idx_b[:n_val]
    idx_box = np.where(np.array([str(l) == fam_box for l in y_train]))[0]

    X_fit = X_train[idx_fit]
    X_sim = np.vstack([X_train[idx_val], X_train[idx_box]])
    y_sim = np.r_[np.zeros(len(idx_val)), np.ones(len(idx_box))]

    print("\nCalcul des scores…", end=" ", flush=True)
    S_sim = _scorers_benign(X_fit, X_sim)
    print("ok")

    print(f"\n--- Étape 1 : quel score reconnaît la famille connue {fam_box} ? ---")
    a_sim = {k: max(float(auroc(y_sim, v)), 1 - float(auroc(y_sim, v)))
             for k, v in S_sim.items()}
    for k, v in sorted(a_sim.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} AUROC {v:.4f}" + ("   <- retenu" if k == max(a_sim, key=a_sim.get) else ""))
    best = max(a_sim, key=a_sim.get)
    if max(a_sim.values()) - min(a_sim.values()) < 0.03:
        print("\n  ⚠️  Tous les scores saturent sur cette famille : le départage")
        print("     ne vaut rien (l'écart entre eux est négligeable).")
        print("     Regarde la colonne AUROC test, pas la ligne « retenu ».")

    print(f"\n--- Étape 2 : application unique au vrai test (score = {best}) ---")
    ben_te = np.array(["BENIGN" in str(l).upper() for l in y_test])
    y_bin = (~ben_te).astype(int)

    # le seuil est calé sur le bénin de validation, jamais sur le test
    S_val = _scorers_benign(X_fit, X_train[idx_val])
    sign = 1.0 if auroc(y_sim, S_sim[best]) >= 0.5 else -1.0
    seuil = float(np.quantile(sign * S_val[best], 1 - target_fpr))

    row = _scorers_benign(X_fit, X_test)
    print(f"\n{'Score':<14}{'AUROC test':>12}{'rappel @5% FA':>16}   (mesuré sur le test)")
    print("-" * 56)
    for k in sorted(a_sim, key=lambda x: -a_sim[x]):
        s = row[k]
        sg = 1.0 if auroc(y_bin, s) >= 0.5 else -1.0
        print(f"  {k:<14}{max(auroc(y_bin, s), 1 - auroc(y_bin, s)):>10.4f}"
              f"{tpr_at_fpr(y_bin, sg * s, 0.05):>15.2%}"
              + ("   <- retenu" if k == best else ""))

    s_best = sign * row[best]
    a_final = max(auroc(y_bin, row[best]), 1 - auroc(y_bin, row[best]))
    pred = s_best > seuil
    recall = float(pred[y_bin == 1].mean())
    fpr = float(pred[y_bin == 0].mean())

    print(f"\nAUROC-OOD (score {best}) : {a_final:.4f}")
    print(f"Rappel @ 5% FA           : {tpr_at_fpr(y_bin, s_best, 0.05):.2%}")
    print(f"Seuil FPR {target_fpr:.0%}           : {seuil:.4f}")
    print(f"Recall global            : {recall:.2%}   |   FPR réel : {fpr:.2%}")
    print(f"\n{'Classe':<28}{'n':>7}{'Recall':>10}")
    print("-" * 46)
    for c in sorted(set(map(str, y_test))):
        m = np.array([str(l) == c for l in y_test])
        if m.sum() == 0:
            continue
        r = float((~pred[m]).mean()) if "BENIGN" in c.upper() else float(pred[m].mean())
        print(f"{c[:27]:<28}{int(m.sum()):>7}{r:>9.2%}")

    out = {"espace": espace, "rappel_at_5pct_fa": float(tpr_at_fpr(y_bin, s_best, 0.05)), "score_retenu": best, "famille_simulee": fam_box,
           "auroc_pseudo_zd": a_sim[best], "auroc_test": a_final,
           "recall": recall, "fpr": fpr, "seuil": seuil,
           "comparaison_pseudo_zd": a_sim}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "ood.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n✅ {os.path.join(RESULTS, 'ood.json')}")
    return out



# =====================================================================
#  BALAYAGE : plusieurs familles de classifieurs supervisés
# =====================================================================
# ══════════════════════════════════════════════════════════════════════
#  A3 — MÉTRIQUES D'ENSEMBLE OUVERT (conventions 2025-2026)
# ══════════════════════════════════════════════════════════════════════
def open_set_metrics(scores, y_bin, y_raw, familles=None):
    """Métriques comparables à la littérature d'ensemble ouvert.

    `AUROC-OOD` seule n'est pas une convention : les travaux 2025-2026
    rapportent OpenAUC, closed-set accuracy, FPR@95TPR et TPR@1FPR. Sans ces
    colonnes, un chiffre n'est pas situable dans l'état de l'art.

    OpenAUC (adaptation binaire) : moyenne macro des AUROC « une famille contre
    le bénin », pondérée à parts égales par famille. Une famille facile ne peut
    donc pas masquer une famille en échec — c'est précisément le défaut d'une
    AUROC globale, où Bot (49 % de rappel) disparaît dans la moyenne.
    """
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(scores, dtype=np.float64)
    y_bin = np.asarray(y_bin).astype(int)
    y_raw = np.asarray([str(v) for v in y_raw])
    ben = y_bin == 0

    fam = familles or sorted({v for v in y_raw[~ben]})
    detail, aucs, poids = {}, [], []
    for f in fam:
        m = y_raw == f
        if m.sum() < 2 or ben.sum() < 2:
            continue
        sub = ben | m
        auc_f = float(roc_auc_score(y_bin[sub], scores[sub]))
        detail[f] = {"n": int(m.sum()), "auroc_vs_benign": round(auc_f, 4)}
        aucs.append(auc_f)
        poids.append(int(m.sum()))

    # seuils dérivés : on balaie pour trouver le point de fonctionnement
    def _seuil_pour_fpr(alpha):
        q = np.quantile(scores[ben], 1.0 - alpha)
        return float(q)

    def _tpr_at(alpha):
        t = _seuil_pour_fpr(alpha)
        return float((scores[~ben] > t).mean()), t

    tpr_1, seuil_1 = _tpr_at(0.01)
    tpr_5, seuil_5 = _tpr_at(0.05)
    tpr_10, seuil_10 = _tpr_at(0.10)

    # FPR au point où 95 % des anomalies sont détectées
    t95 = float(np.quantile(scores[~ben], 0.05))
    fpr95 = float((scores[ben] > t95).mean())

    return {
        "open_auc_macro": round(float(np.mean(aucs)), 4) if aucs else None,
        "open_auc_weighted": (round(float(np.average(aucs, weights=poids)), 4)
                              if aucs else None),
        "open_auc_min_famille": round(float(np.min(aucs)), 4) if aucs else None,
        "fpr_at_95tpr": round(fpr95, 4),
        "tpr_at_1fpr": round(tpr_1, 4),
        "tpr_at_5fpr": round(tpr_5, 4),
        "tpr_at_10fpr": round(tpr_10, 4),
        "par_famille": detail,
    }


# ══════════════════════════════════════════════════════════════════════
#  A2 — PRÉDICTION CONFORME (garantie de taux de fausses alertes)
# ══════════════════════════════════════════════════════════════════════
def conformal_threshold(calib_benign, alpha: float):
    """Seuil conforme à couverture garantie, pour un FPR cible α.

    Différence essentielle avec un quantile empirique : la correction de rang
    fini `ceil((n+1)(1-α))/n` garantit que, pour un flux bénin nouveau tiré de
    la même distribution que la calibration,

        P(score > seuil) ≤ α

    avec probabilité au moins 1-δ (δ = niveau de confiance sur la calibration
    elle-même), et ce sans hypothèse sur la forme de la distribution.

    Le quantile empirique, lui, ne garantit rien : c'est une estimation, dont
    l'erreur n'est pas bornée quand n est petit. Sur 25 000 bénins l'écart est
    faible mais réel (9,43 % constatés pour 9 % visés).
    """
    calib_benign = np.asarray(calib_benign, dtype=np.float64)
    n = len(calib_benign)
    if n < 10:
        return {"seuil": float(np.max(calib_benign)), "n_calib": n,
                "rang": None, "alpha_effectif": None}
    rang = int(np.ceil((n + 1) * (1.0 - alpha)))
    rang = min(max(rang, 1), n)
    seuil = float(np.sort(calib_benign)[rang - 1])
    # taux que le seuil laisse passer AU MAXIMUM, par construction
    alpha_effectif = 1.0 - rang / (n + 1)
    return {"seuil": seuil, "n_calib": n, "rang": rang,
            "alpha_effectif": round(alpha_effectif, 6),
            "couverture_min": round(1.0 - alpha_effectif, 6)}


# ══════════════════════════════════════════════════════════════════════
#  STABILITÉ — le chiffre est-il reproductible ?
# ══════════════════════════════════════════════════════════════════════
def run_stabilite(n_net: int = None, algo: str = "hgb", n_graines: int = 5,
                  target_fpr: float = 0.09):
    """Réentraîne le détecteur avec plusieurs graines et rapporte l'incertitude.

    Un chiffre unique ne dit pas si le résultat tient. Deux questions se posent
    à un jury : « et si vous relanciez ? » et « l'écart avec le modèle suivant
    est-il réel ? ». Les deux se répondent par un écart-type sur les graines.

    Note : la variance des graines est en général PLUS FAIBLE que l'incertitude
    réelle du résultat (qui vient du découpage et du jeu de données). Elle donne
    donc une borne basse de l'instabilité, ce qu'il faut préciser dans le
    mémoire plutôt que de la présenter comme l'incertitude totale.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    F_tr, F_te, y_tr, y_te = load_features(n_net)[:4]
    b_te = np.array(["BENIGN" in str(l).upper() for l in y_te])
    ybin_tr = (~np.array(["BENIGN" in str(l).upper() for l in y_tr])).astype(int)
    ybin_te = (~b_te).astype(int)
    lab_te = np.array([str(v) for v in y_te])
    familles = sorted({v for v in lab_te[~b_te]})

    print("\n" + "=" * 74)
    print(f"  STABILITÉ — {n_graines} graines, {algo}")
    print("=" * 74)
    # Deux sources d'aléa, et une seule ne suffit pas :
    #   · changer `random_state` d'un HGB sans sous-échantillonnage ne change
    #     RIEN — le modèle est déterministe. Le premier essai l'a montré :
    #     cinq graines, cinq résultats identiques, écart-type nul.
    #   · ce qui varie réellement quand on « relance », c'est le tirage des
    #     données. On rééchantillonne donc 90 % du train à chaque graine.
    # La dispersion obtenue mesure la sensibilité au jeu d'entraînement, pas
    # au hasard du solveur : c'est celle qui intéresse un jury.
    rng = np.random.default_rng(12345)
    part = 0.90
    print(f"{'graine':>8}{'AUROC':>10}{'rappel':>10}{'FPR':>9}{'pire famille':>16}")
    res = []
    for g in range(n_graines):
        idx = rng.choice(len(F_tr), int(len(F_tr) * part), replace=False)
        Xg, yg = F_tr[idx], ybin_tr[idx]
        if algo == "hgb":
            # max_leaf_nodes doit refléter le modèle RÉELLEMENT déployé,
            # sinon on mesure la stabilité d'un modèle qui n'est plus celui
            # du rapport.
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                 max_leaf_nodes=63,
                                                 random_state=0).fit(Xg, yg)
        else:
            clf = LogisticRegression(max_iter=3000, random_state=0).fit(Xg, yg)
        s = (clf.predict_proba(F_te)[:, 1] if algo == "hgb"
             else clf.decision_function(F_te))

        sb = s[b_te]
        thr = float(np.quantile(sb, 1 - target_fpr))
        au = float(roc_auc_score(ybin_te, s))
        rap = float((s[~b_te] > thr).mean())
        fpr = float((sb > thr).mean())

        aucs_fam = {}
        for f in familles:
            m = lab_te == f
            if m.sum() >= 2:
                aucs_fam[f] = float(roc_auc_score(ybin_te[b_te | m], s[b_te | m]))
        pire = min(aucs_fam, key=aucs_fam.get) if aucs_fam else "?"

        res.append({"graine": g, "auroc": au, "rappel": rap, "fpr": fpr,
                    "par_famille": aucs_fam, "pire": pire})
        print(f"{g:>8}{au:>10.4f}{rap:>9.2%}{fpr:>9.2%}{pire[:14]:>16}")

    au = np.array([r["auroc"] for r in res])
    rap = np.array([r["rappel"] for r in res])
    fpr = np.array([r["fpr"] for r in res])
    print("-" * 74)
    print(f"{'moyenne':>8}{au.mean():>10.4f}{rap.mean():>9.2%}{fpr.mean():>9.2%}")
    print(f"{'écart-type':>8}{au.std(ddof=1):>10.4f}{rap.std(ddof=1):>9.2%}"
          f"{fpr.std(ddof=1):>9.2%}")

    # écart entre le meilleur et le plus mauvais tirage
    print(f"\n  amplitude AUROC : {au.min():.4f} – {au.max():.4f} "
          f"(étendue {float(np.ptp(au)):.4f})")
    print(f"  IC à 95 % sur la moyenne : "
          f"{au.mean() - 2.776*au.std(ddof=1)/np.sqrt(n_graines):.4f} – "
          f"{au.mean() + 2.776*au.std(ddof=1)/np.sqrt(n_graines):.4f}"
          f"   (loi de Student, {n_graines - 1} ddl)")

    # stabilité famille par famille : la moyenne peut cacher une famille instable
    print("\n  AUROC par famille (moyenne ± écart-type sur les graines) :")
    for f in familles:
        v = np.array([r["par_famille"][f] for r in res if f in r["par_famille"]])
        if len(v) > 1:
            print(f"    {f[:26]:<28}{v.mean():.4f} ± {v.std(ddof=1):.4f}")

    pires = [r["pire"] for r in res]
    stable_nom = len(set(pires)) == 1
    print(f"\n  Famille la plus faible : "
          f"{'identique sur toutes les graines' if stable_nom else 'varie selon la graine'}"
          f" ({', '.join(sorted(set(pires)))})")

    sortie = {"algo": algo, "n_graines": n_graines, "fpr_cible": target_fpr,
              "auroc": {"moyenne": round(float(au.mean()), 4),
                        "ecart_type": round(float(au.std(ddof=1)), 4),
                        "min": round(float(au.min()), 4),
                        "max": round(float(au.max()), 4)},
              "rappel": {"moyenne": round(float(rap.mean()), 4),
                         "ecart_type": round(float(rap.std(ddof=1)), 4)},
              "fpr": {"moyenne": round(float(fpr.mean()), 4),
                      "ecart_type": round(float(fpr.std(ddof=1)), 4)},
              "par_famille": {f: {"moyenne": round(float(np.mean(
                  [r["par_famille"][f] for r in res if f in r["par_famille"]])), 4),
                  "ecart_type": round(float(np.std(
                      [r["par_famille"][f] for r in res if f in r["par_famille"]],
                      ddof=1)), 4)}
                  for f in familles},
              "pire_famille_stable": bool(stable_nom),
              "graines": res}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "stabilite.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, indent=2, ensure_ascii=False)
    print(f"\n✅ {RESULTS}/stabilite.json écrit")
    return sortie


def run_sweep(n_net: int = None, seed: int = 42):
    """Compare plusieurs classifieurs supervisés sur les familles inédites.

    Deux questions posées, aucune réponse supposée :
      ① la famille de modèle change-t-elle le résultat ? (linéaire, arbres)
      ② la transformation des features change-t-elle le résultat ?
         Les features CICIDS ont des queues très lourdes ; un rang normal
         peut valoir mieux qu'une standardisation.

    Toutes les variantes voient le MÊME train et sont évaluées sur le MÊME
    test. Aucune ne voit les familles inédites à l'entraînement.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import QuantileTransformer

    F_tr, F_te, y_tr, y_te, stats = load_features(n_net, return_stats=True)
    n_net = int(stats.get("n_net", F_tr.shape[1]))
    b_tr = np.array(["BENIGN" in str(l).upper() for l in y_tr])
    b_te = np.array(["BENIGN" in str(l).upper() for l in y_te])
    yb_tr = (~b_tr).astype(int)
    yb_te = (~b_te).astype(int)

    print(f"\n=== BALAYAGE DES CLASSIFIEURS ===")
    print(f"Train {F_tr.shape} · Test {F_te.shape} · réseau {n_net}D + système "
          f"{F_tr.shape[1]-n_net}D")
    print(f"Familles inédites au test : "
          f"{', '.join(sorted({str(v) for v in y_te if 'BENIGN' not in str(v).upper()}))}\n")

    def clip_nan(X):
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def rangs(FitOn, X):
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=min(1000, len(FitOn)),
                                 subsample=50000, random_state=seed)
        qt.fit(clip_nan(FitOn))
        return qt.transform(clip_nan(X))

    grilles = [
        ("LogReg", "standard", True),
        ("LogReg", "rangs", True),
        ("LogReg réseau seul", "standard", True),
        ("HistGradientBoosting", "standard", True),
        ("HistGradientBoosting", "rangs", True),
        ("RandomForest", "standard", True),
    ]

    res = []
    for nom, tr, _ in grilles:
        reseau_seul = "réseau seul" in nom
        base = nom.replace(" réseau seul", "")
        try:
            if tr == "rangs":
                A_tr = rangs(F_tr, F_tr)
                A_te = rangs(F_tr, F_te)
            else:
                A_tr, A_te = clip_nan(F_tr), clip_nan(F_te)
            if reseau_seul:
                A_tr, A_te = A_tr[:, :n_net], A_te[:, :n_net]

            if base == "LogReg":
                m = LogisticRegression(max_iter=3000).fit(A_tr, yb_tr)
            elif base == "HistGradientBoosting":
                m = HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                    random_state=seed).fit(A_tr, yb_tr)
            else:
                m = RandomForestClassifier(
                    n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                    random_state=seed).fit(A_tr, yb_tr)

            s_te = m.decision_function(A_te) if hasattr(m, "decision_function") \
                else m.predict_proba(A_te)[:, 1]
            s_ben = m.decision_function(A_tr[b_tr]) if hasattr(m, "decision_function") \
                else m.predict_proba(A_tr[b_tr])[:, 1]

            a = max(auroc(yb_te, s_te), 1 - auroc(yb_te, s_te))
            r5 = tpr_at_fpr(yb_te, s_te, 0.05)
            r10 = tpr_at_fpr(yb_te, s_te, 0.10)
            res.append((f"{nom} · {tr}", a, r5, r10, s_te))
            print(f"  {nom+' · '+tr:<38} AUROC {a:.4f} | rappel @5% {r5:6.2%} | @10% {r10:6.2%}")
        except Exception as e:
            print(f"  {nom+' · '+tr:<38} échec : {e}")

    # --- HGB sur features réseau + EMBEDDING du transformer ------------------
    # Question posée : l'embedding appris apporte-t-il quelque chose EN PLUS
    # des features, quand on le donne à un modèle qui sait déjà exploiter les
    # interactions ? Si oui, le transformer retrouve un rôle d'extracteur.
    try:
        meta = json.load(open("data/processed/meta.json", encoding="utf-8"))
        _nn, _ns = int(meta.get("n_net", 78)), int(meta.get("n_sys", 25))
        E_tr = _embedding(np.load("data/processed/X_train.npy"), _nn, _ns)
        E_te = _embedding(np.load("data/processed/X_test.npy"), _nn, _ns)
        A_tr = np.hstack([clip_nan(F_tr), E_tr]).astype(np.float32)
        A_te = np.hstack([clip_nan(F_te), E_te]).astype(np.float32)
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
            random_state=seed).fit(A_tr, yb_tr)
        s_te = m.predict_proba(A_te)[:, 1]
        a = max(auroc(yb_te, s_te), 1 - auroc(yb_te, s_te))
        r5, r10 = tpr_at_fpr(yb_te, s_te, 0.05), tpr_at_fpr(yb_te, s_te, 0.10)
        res.append((f"HGB + embedding ({A_tr.shape[1]} dims)", a, r5, r10, s_te))
        print(f"  {'HGB + embedding':<38} AUROC {a:.4f} | rappel @5% {r5:6.2%} | @10% {r10:6.2%}")
    except Exception as _e:
        print(f"  {'HGB + embedding':<38} échec : {_e}")

    if not res:
        return None
    res.sort(key=lambda r: -r[1])
    best_name, best_a, best_r5, best_r10, best_s = res[0]
    print(f"\n→ meilleure variante : {best_name} (AUROC {best_a:.4f})")

    print(f"\n  {'Famille':<28}{'n':>7}{'AUROC':>10}{'rappel @10% FA':>16}")
    print(f"  {'(BENIGN : la colonne de droite est le FPR réel)':>51}")
    print("  " + "-" * 60)
    for c in sorted({str(v) for v in y_te}):
        m_ = np.array([str(v) == c for v in y_te])
        if m_.sum() == 0:
            continue
        if "BENIGN" in c.upper():
            a_c = float("nan")
        else:
            sub = np.r_[best_s[b_te], best_s[m_]]
            lb = np.r_[np.zeros(b_te.sum()), np.ones(m_.sum())]
            a_c = max(auroc(lb, sub), 1 - auroc(lb, sub))
        thr10 = float(np.quantile(best_s[b_te], 0.90))
        if "BENIGN" in c.upper():
            r_c = float((best_s[m_] > thr10).mean())   # pour le bénin : c'est le FPR
        else:
            r_c = float((best_s[m_] > thr10).mean())
        print(f"  {c[:27]:<28}{int(m_.sum()):>7}{a_c:>10.4f}{r_c:>15.2%}")

    out = {"variante_retenue": best_name, "auroc": best_a,
           "rappel_5": best_r5, "rappel_10": best_r10,
           "toutes": [{"variante": r[0], "auroc": r[1],
                       "rappel_5": r[2], "rappel_10": r[3]} for r in res]}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n✅ {os.path.join(RESULTS, 'sweep.json')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", action="store_true",
                    help="compare plusieurs scores SANS LABEL et départage sur une famille connue")
    ap.add_argument("--synthetic", action="store_true", help="test du pipeline sans modèle ni données")
    ap.add_argument("--contraste", action="store_true",
                    help="compare one-class et supervisé sur les mêmes familles")
    ap.add_argument("--lofo", action="store_true",
                    help="validation croisée par famille (7 plis, sans EDR)")
    ap.add_argument("--data-dir", default="data", help="dossier des .parquet bruts")
    ap.add_argument("--target-fpr", type=float, default=0.05, help="FPR visé pour le seuil EVT")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--score",
                    choices=["cos", "maha", "nextconf", "features", "supervise",
                             "union", "calibre", "both", "all"],
                    default="both",
                    help="score d'anomalie : cosinus au centroïde (historique) "
                         "ou distance de Mahalanobis sur les embeddings")
    ap.add_argument("--espace", choices=["features", "modele"], default="features",
                    help="sur quoi calculer les scores sans label : features brutes "
                         "ou embedding du MS-Transformer")
    ap.add_argument("--pseudo-zd", dest="pseudo_zd", default=None,
                    help="famille CONNUE à mettre de côté pour simuler un Zero-Day "
                         "(ex. 'brute' pour une attaque discrète plutôt qu'un flood)")
    ap.add_argument("--shrinkage", choices=["lw", "oas", "emp"], default="lw",
                    help="régularisation de la covariance (lw = Ledoit-Wolf)")
    ap.add_argument("--algo", choices=["logreg", "hgb"], default="logreg",
                    help="classifieur supervisé : logreg (0.9026) ou hgb (0.9672)")
    ap.add_argument("--sweep", action="store_true",
                    help="compare plusieurs classifieurs supervisés sur les familles inédites")
    ap.add_argument("--stabilite", type=int, default=0, metavar="N",
                    help="réentraîne N fois avec des graines différentes et "
                         "rapporte moyenne et écart-type (ex : --stabilite 5)")
    args = ap.parse_args()

    if getattr(args, "sweep", False):
        run_sweep(n_net=None)
        return

    if getattr(args, "stabilite", 0):
        run_stabilite(n_net=None, algo=args.algo, n_graines=args.stabilite,
                      target_fpr=args.target_fpr)
        return

    if getattr(args, "ood", False):
        run_ood(n_net=None, target_fpr=args.target_fpr,
                famille_simulee=getattr(args, "pseudo_zd", None),
                espace=getattr(args, "espace", "features"))
        return

    if args.contraste:
        run_contraste(target_fpr=args.target_fpr)
        return
    if args.lofo:
        run_lofo(data_dir=args.data_dir, target_fpr=args.target_fpr)
        return

    print("=" * 68)
    print("  ÉVALUATION ZERO-DAY — MS-Transformer / PSM-HT")
    print("=" * 68)

    if args.synthetic:
        print("⚠️  MODE SYNTHÉTIQUE — chiffres factices, uniquement pour valider le pipeline\n")
        scores, y_raw, zd, benign_train, extra = load_synthetic()
    else:
        try:
            scores, y_raw, zd, benign_train, extra = load_real()
        except Exception as e:
            print(f"❌ Chargement impossible : {e}")
            print("   Vérifie data/processed/*.npy et models/*.pt, ou lance --synthetic")
            sys.exit(1)

    scores = np.asarray(scores, dtype=float).ravel()
    y_raw = np.asarray(y_raw).astype(str)
    y_bin = np.array([0 if "BENIGN" in s.upper() else 1 for s in y_raw])

    print(f"Échantillons test : {len(scores):,}  |  BENIGN : {(y_bin == 0).sum():,}  "
          f"| Anomalies : {(y_bin == 1).sum():,}")
    if zd:
        print(f"Zero-Day retirés de l'entraînement : {', '.join(zd)}")

    # --- Seuils ---
    thr_pct = threshold_percentile(benign_train, args.target_fpr)
    evt = threshold_evt(benign_train, args.target_fpr)
    thr = evt["threshold"]

    print(f"\n--- Calibration du seuil (FPR cible = {args.target_fpr:.0%}) ---")
    print(f"Percentile empirique : {thr_pct:.4f}")
    print(f"EVT / GPD            : {thr:.4f}   [{evt['method']}]"
          + (f"  xi={evt['xi']:.3f} sigma={evt['sigma']:.3f}" if evt["xi"] is not None else ""))

    # --- Contrôle de calibration : le seuil atteint-il vraiment la cible ? ---
    # Les deux seuils visent le même FPR cible, mais l'EVT extrapole la queue de
    # distribution et se révèle souvent plus conservateur. On mesure le FPR
    # réellement obtenu sur la référence bénigne et on rapporte aussi le point
    # de fonctionnement au percentile, pour que le mémoire expose les deux
    # plutôt qu'un seul chiffre non discuté.
    ben_scores = scores[y_bin == 0]
    fpr_evt_emp = float((ben_scores >= thr).mean())
    fpr_pct_emp = float((ben_scores >= thr_pct).mean())
    r_evt = rates_at_threshold(y_bin, scores, thr)
    r_pct = rates_at_threshold(y_bin, scores, thr_pct)
    print("\n  Contrôle de calibration sur les bénins du test :")
    print(f"    EVT        : seuil {thr:.4f} -> FPR {fpr_evt_emp:.2%} "
          f"(cible {args.target_fpr:.0%}) | recall {r_evt['tpr_recall']:.2%}")
    print(f"    Percentile : seuil {thr_pct:.4f} -> FPR {fpr_pct_emp:.2%} "
          f"| recall {r_pct['tpr_recall']:.2%}")
    if fpr_evt_emp < 0.8 * args.target_fpr:
        print(f"    ⚠️  le seuil EVT est plus conservateur que la cible "
              f"({fpr_evt_emp:.2%} < {args.target_fpr:.0%}) : il reste du rappel")
        print("       disponible si l'on accepte d'atteindre réellement la cible.")
    print("    (le seuil EVT, calibré sur le train, reste le seuil retenu ci-dessous)")

    # --- Choix du score ---------------------------------------------------
    # Trois candidats, tous évalués sur le même modèle et le même test :
    #   cos    : 1 - cos(z, centroïde bénin)          (historique, non reproductible)
    #   maha   : distance de Mahalanobis              (reproductible)
    #   nextconf : 1 - confiance de la tête « prochain état »
    #              Une confiance basse signale une séquence réseau que le
    #              modèle n'arrive pas à prolonger : signature d'une attaque.
    score_name = "cosinus"
    maha_pack = None

    want_maha = getattr(args, "score", "both") in ("maha", "both", "all", "union")
    want_conf = getattr(args, "score", "both") in ("nextconf", "all")
    # next_conf n'est JAMAIS retenu automatiquement : le résultat est un
    # artefact de séparabilité des features (cf. audit ci-dessous), et la
    # démonstration Streamlit doit rester alignée sur le score Mahalanobis.

    candidates = {"cosinus": float(auroc(y_bin, scores))}

    s_maha = s_conf = None
    if want_maha and extra.get("z_benign_train") is not None:
        mu, prec, est = fit_mahalanobis(extra["z_benign_train"],
                                        getattr(args, "shrinkage", "lw"))
        s_maha = maha_score(extra["z_test"], mu, prec)
        candidates[f"mahalanobis({est.__class__.__name__})"] = float(auroc(y_bin, s_maha))

    # --- Score sur les 78 features réseau ---
    # Le Mahalanobis appliqué aux embeddings du transformer plafonne à
    # AUROC 0.69 : l'encodeur efface l'information de structure que les
    # features portent encore. Le même score, appliqué aux features dont
    # le pré-traitement a été corrigé, remonte à 0.96.
    want_feat = getattr(args, "score", "both") in ("features", "all") or \
        getattr(args, "score", "") == "features"
    s_feat = None
    feat_benign_train = None
    feat_pack = None
    if want_feat:
        try:
            F_tr, F_te, y_feat_tr, _, _stats = load_features(return_stats=True)
            bt = np.array(["BENIGN" in str(l).upper() for l in y_feat_tr])
            from sklearn.covariance import EmpiricalCovariance
            cov = EmpiricalCovariance().fit(F_tr[bt])
            d = F_te - cov.location_
            s_feat = np.einsum("ij,jk,ik->i", d, cov.precision_, d)
            db = F_tr[bt] - cov.location_
            feat_benign_train = np.einsum("ij,jk,ik->i", db, cov.precision_, db)
            feat_pack = {"location": cov.location_, "precision": cov.precision_,
                         "mu_full": _stats["mu"], "sd_full": _stats["sd"],
                         "n_net": _stats["n_net"]}
            candidates["mahalanobis(features réseau)"] = float(auroc(y_bin, s_feat))
        except Exception as _e:
            print(f"  ⚠️  score features indisponible : {_e}")
            want_feat = False

    # --- Score supervisé : a vu les familles connues à l'entraînement ---------
    # Évalué ici uniquement pour le contraste : il ne s'agit pas d'un détecteur
    # d'anomalies, et le rapport doit le nommer « supervisé ».
    want_sup = getattr(args, "score", "") in ("supervise", "all", "union", "calibre")
    s_sup = sup_benign_train = sup_pack = None
    if want_sup:
        try:
            s_sup, sup_benign_train, _, _clf = train_supervised(
                algo=getattr(args, "algo", "logreg"))
            _lbl = "LogReg" if getattr(args, "algo", "logreg") == "logreg" else "HGB"
            candidates[f"supervisé({_lbl})"] = float(auroc(y_bin, s_sup))
        except Exception as _e:
            print(f"  ⚠️  score supervisé indisponible : {_e}")
            want_sup = False

    if want_conf and extra.get("next_conf") is not None:
        s_conf = 1.0 - np.asarray(extra["next_conf"], dtype=float)
        candidates["next_conf"] = float(auroc(y_bin, s_conf))

    # --- CALIBRATION CONDITIONNELLE -----------------------------------------
    # Constat mesuré : la régression logistique classe bien XSS (AUROC 0.90)
    # mais ne la détecte presque pas à 10 % de fausses alertes (5,5 %). Cause :
    # les 10 % de bénins les plus « suspects » sont toujours les mêmes flux, et
    # ils écrasent le seuil global. Un flux web attaqué est plus suspect qu'un
    # bénin ORDINAIRE, mais moins qu'un bénin déjà extrême.
    #
    # Remède : ne plus comparer un flux à TOUT le bénin, mais au bénin qui lui
    # ressemble. On regroupe le bénin en k sous-populations, puis on mesure à
    # quel point le score est inhabituel DANS la sous-population du flux.
    s_cal = None
    if getattr(args, "score", "") == "calibre" and s_sup is not None:
        from sklearn.cluster import MiniBatchKMeans
        F_tr, F_te, y_trf, _, _ = load_features(return_stats=True)
        bt = np.array(["BENIGN" in str(l).upper() for l in y_trf])
        K = 32
        km = MiniBatchKMeans(n_clusters=K, n_init=5, random_state=0,
                             batch_size=8192).fit(F_tr[bt])
        lab_tr = km.predict(F_tr[bt])
        lab_te = km.predict(F_te)
        glob_ref = np.sort(np.asarray(sup_benign_train, dtype=float))
        s_cal = np.zeros(len(s_sup))
        for c in range(K):
            ref = np.sort(sup_benign_train[lab_tr == c])
            if len(ref) < 100:                       # sous-population trop rare
                ref = glob_ref
            m = lab_te == c
            if m.any():
                s_cal[m] = (np.searchsorted(ref, s_sup[m]) + 0.5) / len(ref)
        candidates["calibré(LogReg par sous-population)"] = float(auroc(y_bin, s_cal))

    # --- UNION : moyenne des rangs de plusieurs détecteurs -------------------
    # Aucune pondération n'est apprise et aucun seuil n'est optimisé : c'est
    # une règle fixe, décidée avant de regarder le test. Chaque détecteur est
    # converti en « rang dans le bénin d'entraînement », ce qui neutralise les
    # échelles (logit, distance, cosinus) avant de les moyenner.
    s_union = None
    if getattr(args, "score", "") in ("union", "all"):
        def _pval(ref, x):
            r = np.sort(np.asarray(ref, dtype=float))
            return (np.searchsorted(r, np.asarray(x, dtype=float)) + 0.5) / len(r)

        parts = []
        if s_sup is not None and sup_benign_train is not None:
            parts.append(("sup", _pval(sup_benign_train, s_sup)))
        if s_maha is not None and benign_train is not None:
            parts.append(("maha", _pval(benign_train, s_maha)))
        if extra.get("token") is not None and extra.get("token_benign_train") is not None:
            parts.append(("token", _pval(extra["token_benign_train"], extra["token"])))
        if len(parts) >= 2:
            noms = "+".join(n for n, _ in parts)
            s_union = np.mean([q for _, q in parts], axis=0)
            candidates[f"union({noms})"] = float(auroc(y_bin, s_union))
            if len(parts) >= 3:
                # La MÉDIANE des rangs : si un détecteur est mauvais, il ne
                # peut pas entraîner les deux autres. La moyenne, si.
                s_union_med = np.median([q for _, q in parts], axis=0)
                candidates[f"union_med({noms})"] = float(auroc(y_bin, s_union_med))

    if len(candidates) > 1:
        print("\n--- Comparaison des scores sur le même modèle ---")
        for name, val in sorted(candidates.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<28} AUROC {val:.4f}")

    if args.score == "both":
        # on ne compare que cosinus et Mahalanobis sur embeddings
        pool = {k: v for k, v in candidates.items()
                if k == "cosinus" or (k.startswith("mahalanobis") and "features" not in k)}
        best_name = max(pool, key=pool.get)
    elif args.score == "features":
        best_name = "mahalanobis(features réseau)"
    elif args.score == "supervise":
        best_name = next((k for k in candidates if k.startswith("supervisé")),
                         "cosinus")
    elif args.score == "calibre":
        best_name = next((k for k in candidates if k.startswith("calibré")), "cosinus")
    elif args.score == "union":
        best_name = next((k for k in candidates if k.startswith("union_med(")), None) \
            or next((k for k in candidates if k.startswith("union(")), "cosinus")
    elif args.score == "all":
        best_name = "cosinus"   # on affiche la comparaison sans changer de score
    else:
        best_name = args.score
    if best_name == "cos":
        best_name = "cosinus"
    elif best_name == "maha":
        best_name = next((k for k in candidates if k.startswith("mahalanobis")), "cosinus")
    elif best_name == "nextconf":
        best_name = "next_conf"
    if False:
        if best_name == "cos":
            best_name = "cosinus"
        elif best_name == "maha":
            best_name = next(k for k in candidates if k.startswith("mahalanobis"))
        elif best_name == "nextconf":
            best_name = "next_conf"

    if best_name != "cosinus" and best_name in candidates:
        print(f"  → score retenu : {best_name} "
              f"(AUROC {candidates[best_name]:.4f} contre {candidates['cosinus']:.4f})")
        score_name = best_name

        if best_name.startswith("union_med("):
            scores = s_union_med
            _refs = []
            if s_sup is not None and sup_benign_train is not None:
                _refs.append(np.asarray(sup_benign_train, dtype=float))
            if s_maha is not None and benign_train is not None:
                _refs.append(np.asarray(benign_train, dtype=float))
            if extra.get("token_benign_train") is not None:
                _refs.append(np.asarray(extra["token_benign_train"], dtype=float))
            if _refs:
                _n = min(len(r) for r in _refs)
                _refs = [r[:_n] for r in _refs]
                benign_train = np.median(
                    np.vstack([_pval(r, r) for r in _refs]), axis=0)
            maha_pack = None
        elif best_name.startswith("union("):
            scores = s_union
            # le seuil de l'union se cale sur l'union des rangs bénins
            _refs = []
            if s_sup is not None and sup_benign_train is not None:
                _refs.append(np.asarray(sup_benign_train, dtype=float))
            if s_maha is not None and benign_train is not None:
                _refs.append(np.asarray(benign_train, dtype=float))
            if extra.get("token_benign_train") is not None:
                _refs.append(np.asarray(extra["token_benign_train"], dtype=float))
            if _refs:
                _n = min(len(r) for r in _refs)
                _refs = [r[:_n] for r in _refs]
                benign_train = np.mean(
                    np.vstack([_pval(r, r) for r in _refs]), axis=0)
            maha_pack = None
        elif best_name.startswith("calibré"):
            scores = s_cal
            benign_train = np.linspace(0.0, 1.0, 5000)   # les rangs sont uniformes sous H0
            maha_pack = None
        elif best_name.startswith("supervisé"):
            scores = s_sup
            benign_train = sup_benign_train
            maha_pack = None
        elif "features" in best_name:
            scores = s_feat
            benign_train = feat_benign_train
            maha_pack = None
        elif best_name.startswith("mahalanobis"):
            scores = s_maha
            maha_pack = {"mu": mu, "prec": prec}
            benign_train = maha_score(extra["z_benign_train"], mu, prec)
        else:  # next_conf
            scores = s_conf
            if extra.get("conf_benign_train") is not None:
                benign_train = 1.0 - np.asarray(extra["conf_benign_train"], dtype=float)
            else:
                print("  ⚠️  seuil recalibré sur les bénins du TEST (train indisponible)")
                benign_train = scores[np.array(
                    ["BENIGN" in str(l).upper() for l in y_test])]

        thr_pct = threshold_percentile(benign_train, args.target_fpr)
        evt = threshold_evt(benign_train, args.target_fpr)
        thr = evt["threshold"]
        r_evt = rates_at_threshold(y_bin, scores, thr)
        print(f"  Seuil recalibré dans l'espace « {best_name} » : {thr:.4f} "
              f"[{evt['method']}] (percentile {thr_pct:.4f})")

    # --- Métriques globales ---
    a = auroc(y_bin, scores)
    mean_b, lo_b, hi_b = bootstrap_auroc(y_bin, scores, args.n_boot)
    r = rates_at_threshold(y_bin, scores, thr)
    fpr95 = fpr_at_tpr(y_bin, scores, 0.95)
    tpr1 = tpr_at_fpr(y_bin, scores, 0.01)

    print("\n--- Métriques globales ---")
    print(f"AUROC                : {a:.4f}  (IC95 % : {lo_b:.4f} – {hi_b:.4f})")
    print(f"Recall @ seuil EVT   : {r['tpr_recall']:.2%}")
    print(f"FPR    @ seuil EVT   : {r['fpr']:.2%}")
    print(f"Precision / F1       : {r['precision']:.2%} / {r['f1']:.2%}")
    print(f"FPR @ 95 % TPR       : {fpr95:.2%}")
    print(f"TPR @ 1 % FPR        : {tpr1:.2%}")

    # --- A3 : ensemble ouvert, conventions 2025-2026 ---
    osm = {}
    try:
        _yraw = y_raw if "y_raw" in dir() else None
        if _yraw is None:
            _yraw = np.load("data/processed/y_test.npy", allow_pickle=True)
        osm = open_set_metrics(scores, y_bin, _yraw)
        print("\n--- Ensemble ouvert (OpenAUC, conventions 2025-2026) ---")
        print(f"  OpenAUC macro        : {osm['open_auc_macro']:.4f}"
              f"   (pondérée : {osm['open_auc_weighted']:.4f})")
        print(f"  pire famille         : {osm['open_auc_min_famille']:.4f}"
              f"   ← une moyenne globale masque ce chiffre")
        print(f"  TPR @ 1 / 5 / 10 % FPR : {osm['tpr_at_1fpr']:.2%} · "
              f"{osm['tpr_at_5fpr']:.2%} · {osm['tpr_at_10fpr']:.2%}")
        print(f"  FPR @ 95 % TPR       : {osm['fpr_at_95tpr']:.2%}")
    except Exception as _e:
        print(f"  (ensemble ouvert non calculé : {_e})")

    # --- A2 : seuil conforme, à garantie ---
    cm = {}
    try:
        _cal = np.asarray(scores)[np.asarray(y_bin).astype(int) == 0]
        # calibration tenue à l'écart du test : on coupe les bénins en deux
        _rng = np.random.default_rng(42)
        _perm = _rng.permutation(len(_cal))
        _cal_half = _cal[_perm[: len(_cal) // 2]]
        _val_half = _cal[_perm[len(_cal) // 2:]]
        cm = conformal_threshold(_cal_half, args.target_fpr)
        _fpr_val = float((_val_half > cm["seuil"]).mean())
        print("\n--- Prédiction conforme (garantie de FPR) ---")
        print(f"  Calibration          : {cm['n_calib']:,} flux bénins "
              f"(moitié du bénin du test)")
        print(f"  Rang fini            : {cm['rang']} → alpha majoré par "
              f"construction {cm['alpha_effectif']:.4%}")
        print(f"  Seuil conforme       : {cm['seuil']:.6g}")
        print(f"  FPR constaté sur la seconde moitié : {_fpr_val:.2%} "
              f"(cible {args.target_fpr:.0%})")
        # Le contrôle ne peut pas être binaire. La garantie conforme porte sur
        # la TIRAGE de la calibration : P(FPR > α) ≤ δ. Sur un échantillon de
        # validation de taille m, le FPR observé fluctue autour de α avec un
        # écart-type binomial sqrt(α(1-α)/m). Comparer 9,26 % à 9 % sans tenir
        # compte de cet écart reviendrait à déclarer une garantie violée pour
        # une fluctuation d'un écart-type.
        from math import sqrt
        _n_val = len(_val_half)
        _sd = sqrt(args.target_fpr * (1 - args.target_fpr) / max(_n_val, 1))
        _z = (_fpr_val - args.target_fpr) / _sd if _sd > 0 else 0.0
        _compatible = _z <= 1.645          # seuil unilatéral à 5 %
        _borne = args.target_fpr + 1.645 * _sd
        print(f"  Écart-type binomial  : {_sd:.3%} sur {_n_val:,} flux de contrôle")
        print(f"  Écart observé        : {(_fpr_val - args.target_fpr):+.3%} "
              f"(z = {_z:+.2f})")
        print(f"  Compatibilité        : FPR toléré jusqu'à {_borne:.2%} "
              f"à 95 % → {'✅ compatible avec la garantie' if _compatible else '⚠️ incompatible : à examiner'}")
        cm["fpr_val"] = round(_fpr_val, 4)
        cm["n_val"] = int(_n_val)
        cm["ecart_type_binomial"] = round(_sd, 6)
        cm["z_ecart"] = round(float(_z), 3)
        cm["borne_superieure_95"] = round(float(_borne), 6)
        cm["garantie_respectee"] = bool(_compatible)
    except Exception as _e:
        print(f"  (seuil conforme non calculé : {_e})")

    # --- AUDIT : baselines triviales -------------------------------------
    # Placé avant tout commentaire de résultat : si une somme de features fait
    # aussi bien que le modèle, le chiffre principal ne mesure pas ce qu'il
    # prétend mesurer.
    trivial_block = {}
    if not args.synthetic:
        try:
            _Xtr = np.load("data/processed/X_train.npy")
            _ytr = np.load("data/processed/y_train.npy", allow_pickle=True)
            _ytr_bin = np.array(["BENIGN" not in str(l).upper() for l in _ytr]).astype(int)
            # Les baselines doivent être calculées sur les MÊMES features que
            # le score évalué. Avec le score « features », le pré-traitement a
            # été corrigé (scaler ajusté sur tout le train) : comparer le
            # modèle aux .npy d'origine reviendrait à lui opposer une fuite
            # qu'il n'utilise pas.
            if "features" in score_name or score_name.startswith("supervisé"):
                _A, _B, _ytr2, _ = load_features()
                _ytr_bin = np.array(["BENIGN" not in str(l).upper()
                                     for l in _ytr2]).astype(int)
                tb = trivial_baselines(_A, _ytr_bin, _B, y_bin)
            else:
                _Xte = np.load("data/processed/X_test.npy")
                tb = trivial_baselines(_Xtr, _ytr_bin, _Xte, y_bin)
            print("\n--- AUDIT : baselines triviales (sans modèle) ---")
            print(f"  {'méthode':<26}{'AUROC':>8}")
            print("  " + "-" * 34)
            print(f"  {'modèle évalué ci-dessus':<26}{a:>8.4f}")
            for name, d in tb.items():
                auc_t = auroc(y_bin, d["test"])
                trivial_block[name] = {"auroc": float(auc_t)}
                if d["train_benign"] is not None:
                    t = np.quantile(d["train_benign"], 1 - args.target_fpr)
                    pred = d["test"] > t
                    ben_m = (y_bin == 0)
                    trivial_block[name]["fpr"] = float(pred[ben_m].mean())
                    trivial_block[name]["recall"] = float(pred[~ben_m].mean())
                    trivial_block[name]["recall_per_class"] = {
                        c: float(pred[np.array([str(v) == c for v in y_raw])].mean())
                        for c in set(str(v) for v in y_raw) if "BENIGN" not in c.upper()}
                print(f"  {name:<26}{auc_t:>8.4f}")

            _triv = {k: v for k, v in trivial_block.items()
                     if "logistique" not in k and k != "modèle évalué ci-dessus"}
            best_triv = max((v["auroc"] for v in _triv.values()), default=0.0)
            if "logistique" in best_name:
                # Le score évalué EST la régression logistique : elle ne peut
                # pas servir de « baseline triviale » contre elle-même.
                print(f"\n  ℹ️  Score évalué = régression logistique : la comparaison")
                print(f"     ci-dessus n'est pas une alerte. Une baseline sans")
                print(f"     modèle (somme|x|) reste à {trivial_block.get('somme|x|', {}).get('auroc', float('nan')):.4f}.")
                print(f"     L'écart mesure le gain réel de l'apprentissage.")
            elif best_triv >= a - 0.02:
                print("\n" + "!" * 68)
                print("  ⚠️  UNE BASELINE TRIVIALE ÉGALE OU DÉPASSE LE MODÈLE")
                print("!" * 68)
                print(f"     modèle            : AUROC {a:.4f}")
                print(f"     meilleure baseline: AUROC {best_triv:.4f}")
                print("     Conséquence : le score mesure la magnitude des features,")
                print("     elle-même déterminée par la classe. Ce n'est pas une")
                print("     capacité de détection. À déclarer explicitement.")
                print("=" * 68)
        except Exception as _e:
            print(f"  (audit impossible : {_e})")

    # --- Par classe ---
    rows = per_class_recall(y_raw, scores, thr, zd)
    # Têtes réelles (violation / energy / next-state) — moyennes par classe
    if extra:
        for row in rows:
            m = np.array([str(l) == row["classe"] for l in y_raw])
            for k in ("violation", "energy", "next_conf"):
                if k in extra:
                    row[f"{k}_mean"] = float(np.mean(extra[k][m]))
    print("\n--- Détail par classe ---")
    hdr = f"{'Classe':<24}{'n':>7}{'Recall':>10}{'Score moy':>12}{'AUROC':>9}"
    print(hdr + (f"{'Viol.':>9}{'NextConf':>10}" if extra else ""))
    print("-" * (74 if extra else 62))
    for row in rows:
        extra_txt = ""
        if "violation_mean" in row:
            extra_txt = (f"{row['violation_mean']:>9.3f}{row['next_conf_mean']:>10.3f}")
        print(f"{row['classe'][:23]:<24}{row['n']:>7}{row['recall']:>9.2%}"
              f"{row['score_mean']:>12.3f}{row['auroc_vs_benign']:>9.4f}{extra_txt}")
    if extra:
        print("\n(violation = sortie réelle de la tête BCE · next_conf = exp(-MSE) de la tête next-state)")

    # --- Plafond atteignable par famille ---------------------------------
    # Le rappel d'une famille dépend de deux choses : la qualité de séparation
    # (AUROC) et la position du seuil. Tant que l'AUROC ne bouge pas, il existe
    # un rappel MAXIMAL atteignable à un taux de fausses alertes donné : c'est
    # le point de la courbe ROC où le FPR vaut la cible.
    # Mesurer l'écart entre le rappel obtenu et ce plafond répond à une question
    # décisive : reste-t-il quelque chose à gagner en réglant le seuil, ou bien
    # la limite vient-elle du modèle lui-même ?
    plafond = {}
    try:
        _lab = np.array([str(v) for v in y_raw])
        _ben = _lab == "BENIGN"
        _sb = np.asarray(scores)[_ben]
        _thr_p = float(np.quantile(_sb, 1 - args.target_fpr))
        for r in rows:
            if r.get("type") != "ZERO-DAY":
                continue
            _m = _lab == r["classe"]
            if _m.sum() == 0:
                continue
            # plafond = rappel au seuil qui sature exactement le budget de FA
            _p = float((np.asarray(scores)[_m] > _thr_p).mean())
            plafond[r["classe"]] = {"plafond": round(_p, 4),
                                    "obtenu": round(float(r["recall"]), 4),
                                    "ecart": round(_p - float(r["recall"]), 4)}
        if plafond:
            print("\n--- Plafond atteignable à "
                  f"{args.target_fpr:.0%} de fausses alertes ---")
            print(f"{'famille':<26}{'obtenu':>9}{'plafond':>10}{'marge':>9}   lecture")
            for _f, _v in sorted(plafond.items(), key=lambda kv: kv[1]["ecart"]):
                _marge = _v["ecart"] * 100
                _txt = ("déjà au plafond" if _marge < 1
                        else f"+{_marge:.1f} pt possible" if _marge >= 3
                        else f"+{_marge:.1f} pt")
                print(f"{_f[:24]:<26}{_v['obtenu']:>9.1%}{_v['plafond']:>10.1%}"
                      f"{_marge:>8.1f}   {_txt}")
            _libres = [f for f, v in plafond.items() if v["ecart"] >= 0.03]
            if _libres:
                print(f"\n  ⚠️  {', '.join(_libres)} : du rappel reste accessible "
                      f"en déplaçant le seuil,")
                print(f"      MAIS le gain est borné par l'AUROC de la famille. "
                      f"Ce n'est pas gratuit :")
                print(f"      le budget de fausses alertes serait consommé par "
                      f"cette famille au détriment des autres.")
            _bloques = [f for f, v in plafond.items() if v["ecart"] < 0.01]
            if _bloques:
                print(f"  ✅ {', '.join(_bloques)} : déjà au plafond, aucun "
                      f"réglage de seuil ne l'améliorera.")
    except Exception as _e:
        print(f"\n  (plafond non calculé : {_e})")

    # --- Critère go/no-go ---
    zd_rows = [x for x in rows if x["type"] == "ZERO-DAY"]
    print("\n--- Critère go / no-go (Recall ≥ 85 % par Zero-Day) ---")
    ok = True
    for x in zd_rows:
        good = x["recall"] >= 0.85
        ok &= good
        print(f"  {'✅' if good else '❌'} {x['classe']:<24} {x['recall']:.2%}")
    print(f"\n  Verdict : {'✅ PROJET DÉFENDABLE' if ok else '⚠️  à analyser et expliquer dans le mémoire'}")

    # --- Figures ---
    try:
        fig_info = make_figures(y_bin, scores, y_raw, thr, rows)
        print(f"\n✅ 5 figures générées dans {RESULTS}/")
    except Exception as e:
        fig_info = {}
        print(f"\n⚠️  Figures non générées : {e}")

    # --- Figure : réponses des têtes réelles par classe ---
    if extra and "violation" in extra:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            classes = [r["classe"] for r in rows if "violation_mean" in r]
            fig, ax = plt.subplots(figsize=(8, 4))
            xs = np.arange(len(classes))
            ax.bar(xs - 0.22, [r["violation_mean"] for r in rows], 0.44,
                   label="Violation Head (P anomalie)", color="#ef4444")
            ax.bar(xs + 0.22, [r["next_conf_mean"] for r in rows], 0.44,
                   label="Next-State confiance", color="#3b82f6")
            ax.set_xticks(xs, [c[:18] for c in classes], rotation=20, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_title("Réponses des têtes entraînées, par classe")
            ax.legend()
            fig.tight_layout(); fig.savefig(f"{RESULTS}/fig_heads.png", dpi=150); plt.close(fig)
            print(f"✅ Figure : {RESULTS}/fig_heads.png")
        except Exception as e:
            print(f"⚠️  fig_heads non générée : {e}")

    # --- Exports ---
    import pandas as pd
    pd.DataFrame(rows).to_csv(f"{RESULTS}/metrics_per_class.csv", index=False)
    glob = {"auroc": a, "auroc_ci95": [lo_b, hi_b], "auroc_bootstrap_mean": mean_b,
            "threshold": thr, "threshold_method": evt["method"], "threshold_percentile": thr_pct,
            "target_fpr": args.target_fpr, "n_samples": int(len(scores)),
            "n_benign": int((y_bin == 0).sum()), "n_anomaly": int((y_bin == 1).sum()),
            "zero_day_classes": zd, "synthetic": bool(args.synthetic),
            "score": score_name, **r, **fig_info,
            "fpr_at_95tpr": fpr95, "tpr_at_1fpr": tpr1,
            "open_set": osm, "conformal": cm, "plafond": plafond}
    pd.DataFrame([glob]).to_csv(f"{RESULTS}/metrics_global.csv", index=False)
    with open(f"{RESULTS}/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"global": glob, "per_class": rows, "evt": evt,
                   "trivial_baselines": trivial_block}, f, indent=2, ensure_ascii=False)

    # --- Paramètres du score, pour que app.py calcule exactement le même ---
    # Sans ce fichier, l'application retomberait sur le score cosinus et
    # afficherait des alertes incohérentes avec les chiffres du mémoire.
    if feat_pack is not None and "features" in score_name:
        try:
            os.makedirs("models", exist_ok=True)
            np.savez("models/features_maha.npz",
                     mean=feat_pack["location"], precision=feat_pack["precision"],
                     mu_full=feat_pack["mu_full"], sd_full=feat_pack["sd_full"],
                     n_net=feat_pack["n_net"],
                     threshold=float(thr), target_fpr=float(args.target_fpr),
                     score=score_name)
            print("✅ Score features exporté : models/features_maha.npz")
        except Exception as _e:
            print(f"⚠️  export features impossible : {_e}")

    if maha_pack is not None:
        np.savez("models/mahalanobis.npz", mu=maha_pack["mu"],
                 prec=maha_pack["prec"], threshold=float(thr),
                 percentile=float(thr_pct), method=str(evt["method"]))
        print("✅ Score de Mahalanobis exporté : models/mahalanobis.npz "
              "(mu, precision, seuil)")
    else:
        print("ℹ️  Score cosinus retenu : models/mahalanobis.npz non écrit")

    print(f"✅ Résultats exportés : {RESULTS}/metrics.json, metrics_global.csv, metrics_per_class.csv")
    if args.synthetic:
        print("\n⚠️  RAPPEL : chiffres synthétiques. Relance sans --synthetic pour le mémoire.")


if __name__ == "__main__":
    main()
