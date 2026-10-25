"""
MS-Transformer — Encodeur multimodal à fusion par cross-attention.

ARCHITECTURE
────────────
Version 4 (celle du dépôt jusqu'ici) : un seul Transformer partagé, masquage
hiérarchique, puis un unique embedding [CLS] de 32 dimensions. Mesures faites
sur ce modèle :

    score sur l'embedding [CLS]         AUROC 0.6922   (13,1 % de détection)
    score sur les tokens réseau         AUROC 0.9604   (81,5 % de détection)

L'écart ne vient pas de l'encodeur mais du pooling : la projection finale
écrase 9 × 64 dimensions en 32.

Version 5 (ce fichier) : deux encodeurs INDÉPENDANTS, une fusion croisée
bidirectionnelle, et une lecture PAR TOKEN au lieu d'un vecteur global.
C'est la Phase 3 du plan de développement — CrossAttention(Q=Trafic,
K=V=Système) puis Self-Attention finale — ramenée à l'échelle d'un poste.

    réseau (78) ──► Encodeur réseau (2 couches) ──┐
                                                   ├──► Cross-Attention ──┐
    système (25) ─► Encodeur système (2 couches) ──┘      bidirectionnelle │
                                                                           ▼
                                                          Fusion (1 couche)
                                                                           │
                       ┌───────────────────────┬───────────────────────────┤
                       ▼                       ▼                           ▼
              lecture par token        tête de cohérence          tête next-state
              (score d'anomalie)       (les deux vues               (prédiction)
                                        concordent-elles ?)

Ce que la cross-attention apporte, et que la concaténation ne peut pas :
elle mesure si une vue permet de RECONSTRUIRE l'autre. Un balayage de ports
est massif côté réseau et invisible côté système ; une exfiltration est
l'inverse. L'incohérence entre les deux vues devient un signal, alors qu'un
simple empilement de tokens ne peut que constater leur présence.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TokenizedMSEncoder", "MSFusion", "anomaly_score", "free_energy",
           "next_state_confidence", "supcon_loss", "compute_loss",
           "token_scores", "coherence_scores"]


# ══════════════════════════════════════════════════════════════════════
#  BLOCS
# ══════════════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * mult, d))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Self-attention pré-normée, propre à une modalité."""

    def __init__(self, d: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = FeedForward(d, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class CrossBlock(nn.Module):
    """
    Cross-attention bidirectionnelle.

    `q` interroge `kv` : « ce que je vois me permet-il d'expliquer ce que
    l'autre modalité rapporte ? » La sortie est la somme des deux sens,
    ce qui autorise l'asymétrie — un port scan est visible côté réseau
    mais pas côté système, et inversement pour une exfiltration.
    """

    def __init__(self, d: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.nq1, self.nkv1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn1 = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                           batch_first=True)
        self.nq2, self.nkv2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn2 = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                           batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = FeedForward(d, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, net, sys):
        # réseau interroge le système
        a1, w1 = self.attn1(self.nq1(net), self.nkv1(sys), self.nkv1(sys),
                            need_weights=True, average_attn_weights=True)
        # système interroge le réseau
        a2, w2 = self.attn2(self.nq2(sys), self.nkv2(net), self.nkv2(net),
                            need_weights=True, average_attn_weights=True)
        net = net + self.drop(a1)
        sys = sys + self.drop(a2)
        fused = torch.cat([net, sys], dim=1)
        fused = self.norm(fused + self.drop(self.ff(fused)))
        return fused, net, sys, (w1, w2)


# ══════════════════════════════════════════════════════════════════════
#  MODÈLE
# ══════════════════════════════════════════════════════════════════════
class TokenizedMSEncoder(nn.Module):
    """
    MS-Transformer multimodal.

    L'interface publique est conservée à l'identique pour que
    `02_model_training.py`, `03_evaluation.py` et `app.py` continuent de
    fonctionner : la sortie contient toujours les clés `z`, `raw`,
    `cls_feat`, `violation_logit`, `next_pred`, plus `tokens`.
    """

    def __init__(self, n_net: int, n_sys: int, embed_dim: int = 32, d: int = 64,
                 n_tok_net: int = 4, n_tok_sys: int = 5, n_layers: int = 2,
                 nhead: int = 4, dropout: float = 0.1, **_ignored):
        super().__init__()
        self.n_net, self.n_sys = n_net, n_sys
        self.n_tok_net, self.n_tok_sys = n_tok_net, n_tok_sys
        self.chunk_net = int(math.ceil(n_net / n_tok_net))
        self.chunk_sys = int(math.ceil(max(n_sys, 1) / n_tok_sys))
        self.n_tok = n_tok_net + n_tok_sys
        self.embed_dim, self.d = embed_dim, d

        # --- projections d'entrée, une par modalité ---
        self.proj_net = nn.Sequential(
            nn.Linear(self.chunk_net, d), nn.LayerNorm(d), nn.GELU())
        self.proj_sys = nn.Sequential(
            nn.Linear(self.chunk_sys, d), nn.LayerNorm(d), nn.GELU())

        self.pos_net = nn.Parameter(torch.randn(1, n_tok_net, d) * 0.02)
        self.pos_sys = nn.Parameter(torch.randn(1, n_tok_sys, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # --- deux encodeurs indépendants ---
        self.enc_net = nn.ModuleList(
            [EncoderBlock(d, nhead, dropout) for _ in range(n_layers)])
        self.enc_sys = nn.ModuleList(
            [EncoderBlock(d, nhead, dropout) for _ in range(n_layers)])

        # --- fusion croisée ---
        self.cross = CrossBlock(d, nhead, dropout)
        self.fusion_norm = nn.LayerNorm(d)
        self.merge = nn.Linear(d, d)

        # --- têtes de lecture ---
        self.proj = nn.Linear(d, embed_dim)              # embedding contrastif
        self.head_violation = nn.Linear(d, 1)            # P(anomalie)
        self.head_next = nn.Linear(d, n_sys)             # prédit le profil système
        # tête de cohérence : les deux modalités concordent-elles ?
        self.head_coherence = nn.Linear(2 * d, 1)
        # lecture par token : évite le goulot du pooling
        self.token_score = nn.Linear(d, 1)

    # -- tokenisation -------------------------------------------------
    def tokens(self, x):
        net = x[:, : self.n_net]
        pad = self.n_tok_net * self.chunk_net - self.n_net
        if pad > 0:
            net = F.pad(net, (0, pad))
        tn = self.proj_net(net.view(-1, self.n_tok_net, self.chunk_net))

        sysf = x[:, self.n_net: self.n_net + self.n_sys]
        p = self.n_tok_sys * self.chunk_sys - sysf.shape[1]
        if p > 0:
            sysf = F.pad(sysf, (0, p))
        ts = self.proj_sys(sysf.view(-1, self.n_tok_sys, self.chunk_sys))
        return tn + self.pos_net, ts + self.pos_sys

    # -- passe avant --------------------------------------------------
    def forward(self, x, mask: bool = True, return_tokens: bool = False):
        tn, ts = self.tokens(x)                       # (B, Tn, d) (B, Ts, d)

        for blk in self.enc_net:
            tn = blk(tn)
        for blk in self.enc_sys:
            ts = blk(ts)

        fused, net_c, sys_c, (w1, w2) = self.cross(tn, ts)

        # vecteur global obtenu par moyenne des tokens fusionnés,
        # avec le CLS ajouté en tête pour conserver l'API historique.
        pooled = fused.mean(dim=1, keepdim=True)          # (B, 1, d)
        cls_in = torch.cat([self.cls.expand(x.shape[0], -1, -1), pooled], dim=1)
        cls = self.fusion_norm(cls_in.mean(dim=1))        # (B, d)

        raw = self.proj(cls)                              # embedding non normalisé
        z = F.normalize(raw, dim=1)                       # hypersphérique

        # vecteur de cohérence : écart entre les deux vues contextualisées
        net_pool = net_c.mean(dim=1)
        sys_pool = sys_c.mean(dim=1)
        prof = torch.cat([net_pool, sys_pool], dim=1)

        out = {
            "z": z,
            "raw": raw,
            "cls_feat": cls,
            "violation_logit": self.head_violation(cls).squeeze(1),
            "next_pred": self.head_next(cls),
            "coherence_logit": self.head_coherence(prof).squeeze(1),
            # score PAR TOKEN : c'est ici que la performance se joue
            "token_logits": self.token_score(fused).squeeze(-1),   # (B, n_tok)
            "tokens": fused,
            "attn_net_to_sys": w1,
            "attn_sys_to_net": w2,
        }
        return out


# ══════════════════════════════════════════════════════════════════════
#  SCORES DÉRIVÉS
# ══════════════════════════════════════════════════════════════════════
def anomaly_score(z: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """1 − cosinus au centroïde bénin. Historique, conservé pour comparaison."""
    return 1.0 - F.cosine_similarity(z, centroid.view(1, -1), dim=1)


def free_energy(violation_logit: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Énergie libre d'une sortie BCE : −log Σ exp(logits/T), stable en log."""
    l = violation_logit / temperature
    return -(torch.logsumexp(torch.stack([torch.zeros_like(l), l], dim=1), dim=1))


def next_state_confidence(next_pred: torch.Tensor, x: torch.Tensor,
                         n_net: int, n_sys: int):
    """Confiance de la prédiction de l'état système suivant, exp(−MSE)."""
    target = x[:, n_net: n_net + n_sys]
    mse = F.mse_loss(next_pred, target, reduction="none").mean(dim=1)
    return torch.exp(-mse), mse


def token_scores(logits: torch.Tensor, reduce: str = "max") -> torch.Tensor:
    """
    Score d'anomalie agrégé depuis les logits par token.

    `max` retient le token le plus anormal. Mesuré sur le modèle v4 :
    max sur tokens → AUROC 0.9604, contre 0.6922 pour l'embedding CLS.
    """
    if reduce == "max":
        return logits.max(dim=1).values
    if reduce == "mean":
        return logits.mean(dim=1)
    k = min(2, logits.shape[1])
    return logits.topk(k, dim=1).values.mean(dim=1)


def coherence_scores(out: dict) -> torch.Tensor:
    """
    Incohérence entre les deux modalités.

    Deux mesures combinées : le logit de la tête de cohérence, et le
    désaccord angulaire entre les vues contextualisées. Plus la valeur
    est haute, plus les deux capteurs racontent des histoires différentes.
    """
    tokens = out["tokens"]
    nt = tokens.shape[1] // 2
    n_view, s_view = tokens[:, :nt], tokens[:, nt:]
    a = F.normalize(n_view.mean(dim=1), dim=1)
    b = F.normalize(s_view.mean(dim=1), dim=1)
    angular = 1.0 - F.cosine_similarity(a, b, dim=1)
    return angular + 0.1 * torch.sigmoid(out["coherence_logit"])


def supcon_loss(z: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """Perte contrastive supervisée (SupCon), implémentation standard."""
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(dim=1).clamp(min=1)
    return -((log_prob * pos).sum(dim=1) / pos_cnt).mean()


def compute_loss(out, x, y_bin, n_net, n_sys,
                 w_contrast: float = 1.0, w_violation: float = 0.5,
                 w_next: float = 0.3, w_token: float = 0.7,
                 w_coherence: float = 0.4):
    """
    Perte totale. `w_token` et `w_coherence` sont nouveaux : ils entraînent
    les deux têtes de lecture qui portent la performance, au lieu de laisser
    toute l'information se perdre dans le vecteur global.
    """
    losses = {}

    # 1. contrastif sur les embeddings
    losses["contrast"] = supcon_loss(out["z"], y_bin) if len(y_bin.unique()) > 1 \
        else torch.tensor(0.0, device=x.device)

    # 2. violation : le flux est-il anormal ?
    losses["violation"] = F.binary_cross_entropy_with_logits(
        out["violation_logit"], y_bin.float())

    # 3. next-state : prédire le profil système depuis le réseau
    target = x[:, n_net: n_net + n_sys]
    losses["next"] = F.mse_loss(out["next_pred"], target)

    # 4. cohérence : les deux vues concordent-elles pour un flux bénin ?
    coh_target = 1.0 - y_bin.float()
    losses["coherence"] = F.binary_cross_entropy_with_logits(
        out["coherence_logit"], coh_target)

    # 5. lecture par token : chaque token porte un verdict
    tl = out["token_logits"]
    tgt = y_bin.float().unsqueeze(1).expand_as(tl)
    losses["token"] = F.binary_cross_entropy_with_logits(tl, tgt)

    total = (w_contrast * losses["contrast"] + w_violation * losses["violation"]
             + w_next * losses["next"] + w_token * losses["token"]
             + w_coherence * losses["coherence"])
    losses["total"] = total
    return total, losses
