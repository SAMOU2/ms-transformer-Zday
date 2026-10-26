"""
Construit `preview_parcours.html` : une maquette STATIQUE du nouvel écran.

Pourquoi ce fichier : l'application demande torch, streamlit et les données
préparées. Pour juger un dessin, on n'a besoin d'aucun des trois. Ce script
extrait de app.py le bloc CSS, les trois dessins SVG et les chaînes de texte
réellement utilisés, puis les assemble dans une page autonome.

Ce n'est pas une seconde interface : c'est une photo de l'interface.
Le jour où app.py change, on relance ce script et la photo se refait.
"""

import html as _html
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()

# ---- 1. le CSS réel -------------------------------------------------------
css = src[src.index("<style>\n", src.index("#  STYLE")) + len("<style>\n"):
          src.index("</style>\n\"\"\", unsafe_allow_html=True)")]

# ---- 2. les dessins et les liens : on exécute le vrai code ---------------
h0 = src.index("def mascotte_svg(")
h1 = src.index("DUO_POS = [3, 4, 3, 2, 2, 3, 4]")
_ns = {"_html": _html}
exec(src[h0:h1], _ns)          # mascotte_svg, coffre_svg, lien_svg
mascotte_svg = _ns["mascotte_svg"]
coffre_svg = _ns["coffre_svg"]
lien_svg = _ns["lien_svg"]


def esc(s):
    return _html.escape(s)


# ---- 3. les composants, écrits comme Streamlit les rendrait --------------
def node(etat, num, nom, statut, x):
    marque = {"done": "✓", "cur": str(num), "open": str(num), "lock": "🔒"}[etat]
    pill = ('<span class="duo-now-pill">● étape en cours</span>'
            if etat == "cur" else
            '<span class="duo-lock-pill">🔒 verrouillée</span>'
            if etat == "lock" else "")
    cells = []
    for c in range(1, 6):
        inner = ""
        if c == x:
            inner = (f'<div class="stButton st-key-pcb_{etat}_{num}">'
                     f'<button title="{esc(nom)} — {esc(statut)}">{marque}</button></div>'
                     f'<div class="duo-lab"><span class="duo-k">Étape {num}</span>'
                     f'<b>{esc(nom)}</b><span class="duo-sub">{esc(statut)}</span>'
                     f'{pill}</div>')
        cells.append(f'<div data-testid="stColumn">{inner}</div>')
    return (f'<div class="st-key-duo_row_{num}">'
            '<div data-testid="stHorizontalBlock" class="hrow">'
            + "".join(cells) + "</div></div>")


def jalon(ouvert, libelle):
    return ('<div class="jalonrow">'
            f'<div class="duo-jalon{" on" if ouvert else ""}">{coffre_svg(ouvert)}'
            f'<span class="jl">{esc(libelle)}</span></div></div>')


X = {1: 10.0, 2: 30.0, 3: 50.0, 4: 70.0, 5: 90.0}
POS = [3, 4, 3, 2, 2, 3, 4]

ETAPES = [
    ("Ouvrir un trafic", "étape faite", "done"),
    ("Lancer l'analyse", "étape faite", "done"),
    ("Lire le résultat", "étape en cours", "cur"),
    ("Suivre une décision", "étape à venir", "open"),
    ("Comparer les approches", "étape à venir", "open"),
    ("Exporter les preuves", "ouverte", "open"),
    ("Refermer le rapport", "verrouillée · après l'étape 6", "lock"),
]

path = []
prev = None
for i, (nom, statut, etat) in enumerate(ETAPES, start=1):
    x = X[POS[i - 1]]
    if prev is not None:
        teinte = "done" if etat == "done" else "cur" if etat in ("cur", "open") else ""
        path.append(lien_svg(prev, x, teinte))
    path.append(node(etat, i, nom, statut, POS[i - 1]))
    prev = x
    if i in (3, 6):
        ouverts = {3: True, 6: False}[i]
        path.append(lien_svg(prev, 50.0, "done" if ouverts else ""))
        path.append(jalon(ouverts, "Halte · une analyse existe" if ouverts
                          else "Halte · rien encore"))
        prev = 50.0
PATH = "".join(path)

KPI = "".join(
    f'<div class="kpi"><div class="kpi-l">{k}</div><div class="kpi-v">{v}</div>'
    f'<div class="kpi-s">{s}</div></div>'
    for k, v, s in (("flux du lot", "200", "200 flux lus"),
                    ("alertes", "23", "177 sous le seuil · 11.5 % détectés"),
                    ("verdict", "MENACE", "des flux dépassent le seuil")))

TABLE = """<div class="tbl-wrap"><table class="tbl"><caption>Résultat sur le lot analysé</caption>
<thead><tr><th scope="col">Détecteur</th><th scope="col">Alertes</th><th scope="col">Rappel mesuré</th></tr></thead>
<tbody>
<tr><th scope="row">MS-Transformer</th><td>37 / 200</td><td>71.2 %</td></tr>
<tr><th scope="row">Mahalanobis · 77 features</th><td>24 / 200</td><td>93.4 %</td></tr>
<tr><th scope="row">Gradient boosting</th><td>23 / 200</td><td>92.8 %</td></tr>
</tbody></table></div>"""

PAGE = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aperçu · parcours</title>
<style>
body {{ margin: 0; background: #ffffff; display: flex; }}
.stApp {{ padding: 0; }}
.mock-side {{ flex: 0 0 268px; padding: 22px 18px; box-sizing: border-box;
  border-right: 2px solid #e5e5e5; background: #f7f7f7; align-self: stretch; }}
.mock-main {{ flex: 1 1 auto; min-width: 0; padding: 26px 26px 60px;
  max-width: 1180px; }}
.mock-side h2 {{ font-size: 1.05rem !important; margin: 0 0 4px; }}
.mock-side .cap {{ font-size: .78rem; color: #6f6f6f; line-height: 1.45;
  margin: 0 0 12px; }}
.mock-side .lab {{ font-size: .82rem; font-weight: 700; color: #3c3c3c;
  margin: 14px 0 6px; }}
.radios {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.radios label {{ font-size: .8rem; font-weight: 700; color: #3c3c3c; }}
.slider {{ position: relative; height: 8px; border-radius: 999px;
  background: #e5e5e5; margin: 10px 0 4px; }}
.slider span {{ display: block; height: 100%; border-radius: 999px;
  background: #58cc02; }}
.slider i {{ position: absolute; top: -5px; width: 18px; height: 18px;
  border-radius: 50%; background: #58cc02; border: 2px solid #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,.25); }}
.apply {{ background: #3d8a00; border: 2px solid #3a7f02;
  border-bottom-width: 4px; border-radius: 12px; color: #fff; font-weight: 800;
  text-align: center; padding: 9px; font-size: .85rem; margin: 12px 0; }}
.hrow {{ display: flex; gap: 4px; align-items: flex-start; }}
.hrow > [data-testid="stColumn"] {{ flex: 1 1 0; min-width: 0;
  display: flex; flex-direction: column; align-items: center; }}
.split {{ display: flex; gap: 26px; align-items: flex-start; }}
.split > .l {{ flex: 1.35; min-width: 0; }}
.split > .r {{ flex: 2.5; min-width: 0; }}
.card {{ background: #fff; border: 2px solid #e5e5e5; border-radius: 20px;
         padding: 16px 10px 24px; }}
.jalonrow {{ display: flex; justify-content: center; }}
.kpis {{ display: flex; gap: 12px; margin-top: 14px; }}
.kpis > * {{ flex: 1; }}
.stButton button {{ border-radius: 12px; border: 2px solid #e5e5e5;
  border-bottom-width: 4px; background: #fff; color: #3c3c3c; font-weight: 800;
  padding: 9px 16px; cursor: pointer;
  font-family: "Segoe UI", system-ui, sans-serif; }}
.mockbtn {{ border-radius: 12px; border: 2px solid #e5e5e5; border-bottom-width: 4px;
  background: #fff; color: #3c3c3c; font-weight: 800; padding: 10px 16px;
  text-align: center; font-size: 0.85rem; letter-spacing: .04em; }}
.mockbtn.primary {{ background: #3d8a00; border-color: #3a7f02; color: #fff; }}
.unit-inner {{ display: flex; align-items: center; gap: 18px; }}
.unit-inner .unit {{ flex: 1; }}
.guide {{ background: #fff; border: 2px solid #fff; border-bottom: 4px solid #cfe8b8;
  border-radius: 12px; color: #16350a; font-weight: 800; padding: 8px 16px;
  font-size: .8rem; white-space: nowrap; }}
.leftnav {{ display: flex; gap: 8px; margin: 10px 0 14px; }}
.leftnav > * {{ flex: 1; }}
{css}
</style>
</head>
<body class="stApp">
  <aside data-testid="stSidebar" class="mock-side">
    <h2>Réglages</h2>
    <p class="cap">Les trois réglages s'appliquent au clic sur Appliquer.</p>
    <p class="lab">Langue</p>
    <div role="radiogroup" class="radios">
      <label><input type="radio" checked hidden>fr</label>
      <label><input type="radio" hidden>en</label>
    </div>
    <p class="lab">Détecteur</p>
    <div role="radiogroup" class="radios">
      <label><input type="radio" hidden>MS-Transformer</label>
      <label><input type="radio" checked hidden>Gradient boosting</label>
    </div>
    <p class="lab">Budget de fausses alertes</p>
    <div class="slider"><span style="width:24%"></span><i style="left:24%"></i></div>
    <p class="cap">0,5 % — 20,0 % · 5,0 %</p>
    <div class="apply">Appliquer</div>
    <p class="cap">Application hors ligne : rien ne quitte la machine.</p>
  </aside>
  <main class="mock-main">

  <div class="eyebrow">Mémoire · M2 cybersécurité</div>
  <h1 class="proj-title">Détection d'intrusions Zero-Day</h1>
  <p class="proj-sub">Gradient boosting sur les 77 colonnes réseau · familles jamais
  vues à l'entraînement</p>
  <p><span class="badge badge-ok"><span aria-hidden="true">✓</span> hors ligne</span>
     <span class="badge badge-warn"><span aria-hidden="true">⚠</span> résultat négatif conservé</span>
     <span class="badge badge-info"><span aria-hidden="true">≡</span> auditable</span></p>

  <div style="height:18px"></div>

  <div class="unit-head st-key-duo_unit_2">
    <div data-testid="stHorizontalBlock" class="unit-inner">
    <div class="unit">
      <div class="unit-k">Unité 2</div>
      <div class="unit-t">Analyser un trafic réseau</div>
      <div class="unit-s">Chaque étape s'ouvre quand la précédente a produit
      un résultat réel — jamais par simple clic.</div>
      <div class="unit-p"><span class="unit-pl">Progression</span>
        <span class="unit-dots"><span class="u-dot u-on"></span><span class="u-dot u-on"></span>
        <span class="u-dot u-on"></span><span class="u-dot"></span><span class="u-dot"></span>
        <span class="u-dot"></span><span class="u-dot"></span></span>
        <span class="unit-pn">3 étapes sur 7 · état réel des données</span></div>
    </div>
    <div class="guide">Guide</div>
    </div>
  </div>

  <div style="height:22px"></div>

  <div class="split">
    <div class="l"><div class="card">
      <div class="duo-mascotte">{mascotte_svg(84)}
        <div class="duo-bubble">Analyse faite : <b>23</b> alertes sur <b>200</b> flux,
        seuil 0.42. Monte ou descends le long du chemin pour relire chaque étape.</div>
      </div>
      <p class="pc-path-h">Parcours</p>
      <p class="pc-path-l">Un nœud par étape. Un cadenas dit ce qui manque.</p>
      <div class="pc-prog"><div class="pc-prog-l">Progression · 3/7</div>
        <div class="pc-prog-b"><span style="width:43%"></span></div></div>
      {PATH}
      <div class="pc-legend"><span><b aria-hidden="true">✓</b> étape faite</span>
        <span><b aria-hidden="true">●</b> étape en cours</span>
        <span><b aria-hidden="true">🔒</b> verrouillée</span>
        <span class="pc-legend-n">L'état n'est jamais porté par la couleur seule.</span></div>
      <div class="u-lock"><div class="u-lock-h">Ce que fait un cadenas</div>
        <div class="u-lock-b">Un nœud verrouillé reste cliquable : il ouvre un encart
        qui dit ce qui manque, et propose d'aller le chercher. Rien ne refuse en silence.</div>
        <div class="u-lock-g">La progression compte les étapes que les données autorisent.</div></div>
    </div></div>

    <div class="r">
      <div class="pc-panel-h">Étape 3 · en cours</div>
      <h3>Lire le résultat</h3>
      <p class="pc-do">Ce que le détecteur a signalé sur ce lot, et ce que ces
      nombres veulent dire — avec la réserve qui va avec.</p>
      <div class="leftnav"><span class="mockbtn">◀ Précédent</span>
        <span class="mockbtn">Étape suivante ▶</span></div>
      <div class="kpis">{KPI}</div>
      <div style="height:14px"></div>
      {TABLE}
      <div style="display:flex;gap:14px;margin-top:6px">
        <div class="panel" style="flex:1;border-left:5px solid #2c8000">
          <div style="font-weight:800;color:#2c8000;margin-bottom:6px">Ce que ce résultat veut dire</div>
          <div style="font-size:.9rem;color:#4b4b4b">Sur un trafic déjà noté, le
          modèle retrouve la majorité des flux intrusifs sans noyer l'analyste.</div>
        </div>
        <div class="panel" style="flex:1;border-left:5px solid #c62828">
          <div style="font-weight:800;color:#c62828;margin-bottom:6px">Ce qu'il ne dit pas</div>
          <div style="font-size:.9rem;color:#4b4b4b">Il ne prouve pas qu'une alerte
          soit une attaque : c'est une invitation à regarder.</div>
        </div>
      </div>
      <div style="display:flex;gap:12px;margin-top:16px">
        <span class="mockbtn primary" style="flex:1.4">Débloquer l'étape 4 · Suivre une décision</span>
        <span class="mockbtn" style="flex:1">Refaire l'analyse</span>
      </div>
      <div class="u-mem"><div class="u-mem-h">Ce qui est mémorisé entre les étapes</div>
        <ul><li>le lot analysé et ses scores ;</li>
        <li>le budget de fausses alertes ;</li>
        <li>la langue et le détecteur choisis ;</li>
        <li>le retour en arrière reste possible : rien n'est consommé.</li></ul></div>
    </div>
  </div>

  <div style="height:26px"></div>
  <div class="note note-warn"><b>Bandeau de statut.</b> Un chiffre sans sa limite
  se lit comme une promesse : chaque mesure de cette page porte l'effectif qui la
  soutient.</div>

  <div class="foot"><b>Source</b> CICIDS2017 · <b>modèle</b>
  <code>ms_encoder.pt</code> · <b>score</b> <code>gradient boosting · 77 features</code></div>
  </main>
</body>
</html>
"""

open(os.path.join(ROOT, "preview_parcours.html"), "w", encoding="utf-8").write(PAGE)
print("preview_parcours.html écrit —", len(PAGE), "octets")
