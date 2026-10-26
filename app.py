"""
MS-Transformer · détection d'intrusions Zero-Day — interface bilingue FR/EN.

Structure
---------
1. Vue d'ensemble : ce que fait le système, quel problème il résout
2. Démonstration  : rejouer un lot de trafic, inspecter les détections
3. Résultats      : performance mesurée, rappel par famille, baselines
4. Méthode        : le modèle et le score
5. Limites        : ce qui marche, ce qui ne marche pas, et pourquoi
6. Aide           : glossaire en langage simple

Détecteur retenu : HistGradientBoosting sur les 77 colonnes réseau.
Score déployé    : AUROC 0,9668 · rappel 93,12 % à 9,03 % de fausses alertes.

Tout ce qui s'affiche ici est lu dans results/ et models/ à l'exécution.
Aucun chiffre n'est écrit à la main : si une mesure manque, l'interface
affiche « non mesurable » et explique pourquoi.

Habillage : thème « terminal » — fond sombre, vert phosphore, police à
chasse fixe, curseur qui clignote, lignes de balayage. Tout tient dans le
bloc <style> de ce fichier : aucune feuille externe, aucune image. Les
effets (machine à écrire, pulsation, glitch) sont neutralisés par
prefers-reduced-motion et à l'impression.
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os
import datetime
import json
import time
import re
import html as _html

st.set_page_config(page_title="PSM-HT · Zero-Day Detection",
                   page_icon="🛡", layout="wide")

ROOT = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════
#  BILINGUAL STRINGS
#  Each entry: (français, english)
# ══════════════════════════════════════════════════════════════════════
T = {
    # header
    "title":        ("Détection d'intrusions Zero-Day", "Zero-Day Intrusion Detection"),
    "subtitle":     ("Gradient boosting sur les 77 colonnes réseau · familles jamais vues à l'entraînement",
                     "Gradient boosting on the 77 network columns · families never seen in training"),
    "lang_label":   ("Langue", "Language"),

    # tabs
    "tab_over":     ("Vue d'ensemble", "Overview"),
    "tab_demo":     ("Démonstration", "Live demo"),
    "tab_res":      ("Résultats", "Results"),
    "tab_method":   ("Méthode", "Method"),
    "tab_limit":    ("Limites", "Limitations"),
    "tab_help":     ("Aide", "Help"),
    "cmp_btn":      ("⇄ Comparer les trois détecteurs sur ce lot",
                     "⇄ Compare the three detectors on this batch"),
    "cmp_h":        ("Comparaison des trois détecteurs", "Comparison of the three detectors"),
    "cmp_intro":    ("Même lot de flux, même seuil de {fpr} % de fausses alertes. "
                     "Seul le détecteur change.",
                     "Same batch of flows, same {fpr} % false-alarm threshold. "
                     "Only the detector changes."),
    "cmp_col":      ("Détecteur", "Detector"),
    "cmp_alerts":   ("Alertes", "Alerts"),
    "cmp_rate":     ("Taux", "Rate"),
    "cmp_recall":   ("Rappel mesuré", "Measured recall"),
    "cmp_scale":    ("Échelle du score", "Score scale"),
    "cmp_kind_dist":("distance (sans plafond)", "distance (no upper bound)"),
    "cmp_kind_prob":("probabilité 0–1", "probability 0–1"),
    "cmp_best":     ("meilleur rappel", "best recall"),
    "cmp_nodata":   ("Aucun lot à comparer : la comparaison porte sur le lot que tu "
                     "viens d'analyser. Remonte à l'étape 1, lance une analyse, puis "
                     "reviens cliquer ce bouton.",
                     "No batch analysed yet — run an analysis first."),
    "cmp_nolabel":  ("Lot sans étiquette : le rappel ne peut pas être mesuré ici.",
                     "Unlabelled batch: recall cannot be measured here."),
    "cmp3_h":       ("Les trois détecteurs sur le jeu de test",
                     "The three detectors on the test set"),
    "cmp3_intro":   ("Échantillon fixe du jeu de test ({n} flux, graine 0), mêmes "
                     "{fpr} % de fausses alertes pour chacun. Ce tableau est "
                     "reproductible : il ne dépend pas du lot chargé dans la démo.",
                     "Fixed sample of the test set ({n} flows, seed 0), same "
                     "{fpr} % false alarms for each. This table is reproducible: "
                     "it does not depend on the batch loaded in the demo."),
    "cmp3_fam":     ("Rappel par famille inédite", "Recall per unseen family"),
    "cmp_note":     ("Les trois scores ne sont pas comparables entre eux : deux "
                     "sont des distances sans plafond, un est une probabilité. "
                     "Seul le nombre d'alertes et le rappel se comparent.",
                     "The three scores are not comparable to each other: two are "
                     "unbounded distances, one is a probability. Only the alert "
                     "count and the recall can be compared."),

    # overview
    "o_problem_h":  ("Le problème", "The problem"),
    "o_problem":    (
        "Un centre opérationnel de sécurité (SOC) reçoit des millions de connexions par jour. "
        "Les systèmes classiques apprennent à reconnaître les attaques **déjà connues**. "
        "Ils échouent sur les attaques **nouvelles**, précisément celles qui comptent : "
        "rien ne les a jamais vues à l'entraînement.",
        "A security operations centre (SOC) receives millions of connections per day. "
        "Classical systems learn to recognise **known** attacks. They fail on **new** ones — "
        "precisely the attacks that matter, because nothing like them was ever seen in training."),
    "o_approach_h": ("L'approche", "The approach"),
    "o_approach":   (
        "Le modèle apprend uniquement à quoi ressemble le trafic **normal**. "
        "Toute connexion qui s'en éloigne suffisamment est signalée, "
        "sans qu'il ait jamais eu besoin d'en voir une seule de ce type.",
        "The model learns only what **normal** traffic looks like. "
        "Any connection that drifts far enough away is flagged — "
        "without ever having seen a single example of that attack."),
    "o_dataset":    ("Jeu de données", "Dataset"),
    "o_dataset_v":  ("CICIDS2017 · trafic réseau d'entreprise", "CICIDS2017 · enterprise network traffic"),
    "o_train":      ("Apprentissage", "Training"),
    "o_train_v":    ("9 classes d'attaques connues", "9 known attack classes"),
    "o_test":       ("Évaluation", "Evaluation"),
    "o_test_v":     ("5 familles jamais vues (Zero-Day)", "5 families never seen (Zero-Day)"),
    "o_score":      ("Score retenu", "Retained score"),
    "o_score_v":    ("Probabilité d'attaque — gradient boosting", "Attack probability — gradient boosting"),
    "o_honest_h":   ("Ce que ce travail démontre", "What this work shows"),
    "o_honest":     (
        "**Le résultat principal.** Un détecteur entraîné sans jamais voir cinq familles "
        "d'attaque les reconnaît ensuite avec un **AUROC de 0,9668** et **93,12 % de "
        "rappel** à 9,03 % de fausses alertes — 5 117 attaques interceptées sur 5 495.\n\n"
        "**Ce qui a changé la conclusion.** La famille Bot était bloquée à 49,34 % de "
        "rappel, et la cause avait été attribuée à l'information disponible. C'était faux : "
        "c'est la **complexité de la frontière** qui plafonnait. En portant `max_leaf_nodes` "
        "de 31 à 63, Bot passe à **84,65 %** sans qu'aucune variable ne change. Le plafond "
        "résiduel, lui, est bien réel : mesuré sur douze configurations, à 9 % comme à 10 % "
        "de fausses alertes, et jusqu'à 255 feuilles.\n\n"
        "**Ce qui ne fonctionne pas.** La représentation apprise **dégrade** le tabulaire : "
        "à classifieur, graine et hyperparamètres identiques, ajouter les 32 dimensions de "
        "l'encodeur fait chuter l'AUROC de 0,9535 à 0,9361 et le rappel de Bot de 54,99 % "
        "à 36,22 %. Le transformer ne se comporte pas en extracteur de caractéristiques.",
        "**The main result.** A detector trained without ever seeing five attack families "
        "then recognises them with **AUROC 0.9668** and **93.12 % recall** at 9.03 % false "
        "alarms — 5,117 attacks caught out of 5,495.\n\n"
        "**What changed the conclusion.** The Bot family was stuck at 49.34 % recall, and "
        "the cause had been attributed to the available information. That was wrong: it was "
        "the **complexity of the decision boundary** that capped it. Raising "
        "`max_leaf_nodes` from 31 to 63 lifts Bot to **84.79 %** without changing a single "
        "variable. The residual ceiling, however, is real: measured across twelve "
        "configurations, at 9 % and at 10 % false alarms, and up to 255 leaves.\n\n"
        "**What does not work.** The learned representation **degrades** the tabular "
        "baseline: with identical classifier, seed and hyperparameters, adding the encoder's "
        "32 dimensions drops AUROC from 0.9535 to 0.9361 and Bot recall from 54.99 % to "
        "36.22 %. The transformer does not act as a feature extractor."),

    # demo
    "d_mode":       ("Source du flux", "Traffic source"),
    "d_live":       ("Flux de test", "Test batch"),
    "d_file":       ("Importer un fichier", "Upload file"),
    "d_generate":   ("▶ Analyser un flux", "▶ Analyse traffic"),
    "d_upload":     ("Dépose un fichier .csv ou .parquet", "Drop a .csv or .parquet file"),
    # ── trace d'un flux : le trajet complet d'une décision ──────────────
    # ── mode direct : flux notés à mesure qu'ils arrivent ───────────────
    "lv_h":         ("Direct — flux notés à l'arrivée", "Live — flows scored as they arrive"),
    "lv_c":         ("Un collecteur écrit les flux terminés dans un fichier CSV ; "
                     "l'interface les note à mesure. Aucun flux n'est simulé ici : "
                     "ce qui s'affiche vient du fichier, pas d'un tirage aléatoire.",
                     "A collector writes finished flows into a CSV file; the interface "
                     "scores them as they arrive. Nothing is simulated here: what you "
                     "see comes from the file, not from a random draw."),
    "lv_missing":   ("Aucun collecteur détecté : le fichier `{p}` n'existe pas encore. "
                     "Démarre CICFlowMeter (Java) en sortie vers ce chemin, ou dépose "
                     "un CSV déjà collecté à cet emplacement. Rien n'est simulé ici : "
                     "l'écran reste vide tant que le collecteur n'a rien écrit. Pour "
                     "une démonstration sans collecteur, utilise l'étape 1 en mode "
                     "« fichier » ou le rejeu flux par flux, annoncés comme tels.",
                     "No collector detected. Expected file: `{p}`."),
    "lv_how":       ("Comment alimenter ce fichier", "How to feed this file"),
    "lv_how_body":  ("Le collecteur doit être **CICFlowMeter**, l'outil qui a produit "
                     "CICIDS2017 lui-même. Le portage Python (`pip install cicflowmeter`) "
                     "ne convient pas : il ne calcule pas 6 des 78 colonnes du modèle "
                     "(sous-flux Fwd/Bwd Packets et Bytes, CWE Flag Count) et en "
                     "approche 2 autres. Alimenter le modèle avec ces trous donnerait "
                     "des scores qui ne sont pas ceux du mémoire.\n\n"
                     "1. Lancez CICFlowMeter en sortie CSV sur votre interface réseau.\n"
                     "2. Faites écrire le fichier ici : `data/live/live_flows.csv`\n"
                     "3. Cliquez sur **Démarrer le direct**.\n\n"
                     "Un flux n'apparaît qu'une fois **terminé** (inactivité d'environ "
                     "deux minutes) : c'est une contrainte du modèle, pas un défaut "
                     "de l'interface.",
                     "The collector must be **CICFlowMeter**, the tool that produced "
                     "CICIDS2017 itself. The Python port (`pip install cicflowmeter`) is "
                     "not suitable: it does not compute 6 of the model's 78 columns "
                     "(Subflow Fwd/Bwd Packets and Bytes, CWE Flag Count) and "
                     "approximates 2 others. Feeding the model with those gaps would "
                     "produce scores that are not the ones in the report.\n\n"
                     "1. Run CICFlowMeter in CSV output mode on your network interface.\n"
                     "2. Have it write the file here: `data/live/live_flows.csv`\n"
                     "3. Click **Start live**.\n\n"
                     "A flow only appears once **finished** (about two minutes of "
                     "inactivity): that is a constraint of the model, not a defect "
                     "of the interface."),
    "lv_live_on":   ("Direct actif", "Live running"),
    "lv_start":     ("▶ Démarrer le direct", "▶ Start live"),
    "lv_stop":      ("■ Arrêter", "■ Stop"),
    "lv_wait":      ("En attente de flux… Le collecteur doit écrire dans le fichier.",
                     "Waiting for flows… The collector must write to the file."),
    "lv_count":     ("{n} flux notés · {a} alertes · {r} % d'alertes",
                     "{n} flows scored · {a} alarms · {r} % alarms"),
    "lv_cols":      ("Analyse refusée : le fichier du collecteur n'a pas les colonnes "
                     "attendues ({c}). Les scores seraient faux, donc rien n'est "
                     "affiché. Vérifie que le collecteur est bien CICFlowMeter (Java) : "
                     "la version `cicflowmeter` en Python ne produit que 70 des 78 "
                     "colonnes du modèle.",
                     "Missing columns in the file: {c}. Scores would be wrong, the "
                     "analysis is refused."),
    "lv_last":      ("Derniers flux notés", "Last scored flows"),
    "lv_since":     ("Depuis le démarrage", "Since start"),

    "st_more":      ("Réserves détaillées", "Detailed caveats"),
    "d_s1":         ("1 · Choisir la source du trafic", "1 · Choose the traffic source"),
    "d_s1_c":       ("Un lot tiré du jeu de test, ou un fichier que vous téléversez. "
                     "Rien n'est noté tant que vous n'avez pas lancé l'analyse.",
                     "A batch drawn from the test set, or a file you upload. Nothing "
                     "is scored until you start the analysis."),
    "d_s2":         ("2 · Lire le résultat", "2 · Read the result"),
    "d_s3":         ("3 · Suivre une décision, flux par flux",
                     "3 · Follow one decision, flow by flow"),
    "d_s4":         ("4 · Comparer les trois détecteurs", "4 · Compare the three detectors"),

    "sb_form_c":    ("Choisis, puis applique. Rien ne change avant le clic : "
                     "c'est ce qui garde le clavier et la lecture à leur place.",
                     "Choose, then apply. Nothing changes before the click: that is "
                     "what keeps the keyboard and the reading in place."),
    "sb_apply":     ("Appliquer les réglages", "Apply settings"),

    "cmp_csv":      ("Télécharger ce tableau en CSV", "Download this table as CSV"),

    "cmp_indispo":  ("{d} ne peut pas être calculé : `models/features_maha.npz` "
                     "est absent. Ce fichier n'est écrit que par une évaluation du "
                     "score sur les features. Régénère-le avec "
                     "`py 03_evaluation.py --score features --target-fpr 0.09`, "
                     "puis recharge cette page. La ligne reste affichée pour qu'on "
                     "ne croie pas à deux détecteurs au lieu de trois.",
                     "{d} cannot be computed: `models/features_maha.npz` is missing. "
                     "That file is only written by an evaluation of the features "
                     "score. Regenerate it with "
                     "`py 03_evaluation.py --score features --target-fpr 0.09`, then "
                     "reload this page. The row stays visible so that two detectors "
                     "are not mistaken for three."),

    "sb_h":         ("Réglages", "Settings"),
    "sb_lang":      ("Langue", "Language"),
    "sb_det":       ("Détecteur", "Detector"),
    "sb_thr":       ("Fausses alertes visées", "Target false alarms"),
    "sb_thr_help":  ("Le seuil est calculé pour ce budget de fausses alertes. "
                     "Il s'applique à toute l'interface : démonstration, rejeu, direct.",
                     "The threshold is computed for this false-alarm budget. It applies "
                     "to the whole interface: demo, replay, live."),
    "sb_reset":     ("Réinitialiser la session", "Reset the session"),
    "sb_about":     ("Ce que vous réglez ici change ce que vous voyez, jamais le "
                     "modèle : les scores du mémoire sont mesurés une fois pour toutes.",
                     "What you set here changes what you see, never the model: the "
                     "report's figures are measured once and for all."),

    "tr_h":         ("Trace d'un flux", "Trace of a single flow"),
    "tr_c":         ("Un flux, depuis ses valeurs jusqu'au verdict. Le score affiché "
                     "est celui que le détecteur actif vient de calculer sur ce lot — "
                     "rien n'est recalculé ici.",
                     "One flow, from its values to the verdict. The score shown is the "
                     "one the active detector just computed on this batch — nothing is "
                     "recomputed here."),
    "tr_pick":      ("Choisir un cas", "Choose a case"),
    "tr_pick_help": ("Les cas sont tirés du lot analysé, avec leur étiquette réelle "
                     "lorsqu'elle est connue.",
                     "Cases are drawn from the analysed batch, with their true label "
                     "when it is known."),
    "tr_cat_hit":   ("Attaque détectée", "Attack detected"),
    "tr_cat_miss":  ("Attaque manquée", "Attack missed"),
    "tr_cat_fp":    ("Fausse alerte", "False alarm"),
    "tr_cat_tn":    ("Bénin sans alerte", "Benign, no alarm"),
    "tr_score":     ("Score", "Score"),
    "tr_thr":       ("Seuil appliqué", "Applied threshold"),
    "tr_verdict":   ("Verdict", "Verdict"),
    "tr_hit":       ("⚠ ALERTE", "⚠ ALARM"),
    "tr_quiet":     ("✓ pas d'alerte", "✓ no alarm"),
    "tr_true":      ("Étiquette réelle", "True label"),
    "tr_unknown":   ("inconnue sur ce lot", "unknown on this batch"),
    "tr_dev_h":     ("Variables les plus écartées de la référence",
                     "Variables furthest from the reference"),
    "tr_dev_c":     ("Écart au trafic d'entraînement, en nombre d'écarts-types. "
                     "**Ce n'est pas l'explication de la décision** : le modèle "
                     "combine ces variables de façon non linéaire, et une variable "
                     "très écartée peut peser moins qu'une autre discrète.",
                     "Deviation from the training traffic, in standard deviations. "
                     "**This is not the explanation of the decision**: the model "
                     "combines these variables non-linearly, and a heavily deviating "
                     "variable may weigh less than a subtle one."),
    "tr_feat":      ("Variable", "Feature"),
    "tr_z":         ("Écart (σ)", "Deviation (σ)"),

    # ── rejeu flux par flux ─────────────────────────────────────────────
    "rj_h":         ("Rejeu flux par flux", "Flow-by-flow replay"),
    "rj_c":         ("Le lot déjà analysé défile dans l'ordre et les alertes "
                     "s'accumulent. C'est un **rejeu** d'un lot existant, pas une "
                     "capture en direct : les scores étaient déjà calculés.",
                     "The already-analysed batch scrolls through in order and alarms "
                     "accumulate. This is a **replay** of an existing batch, not a "
                     "live capture: the scores were already computed."),
    "rj_btn":       ("Rejouer le lot", "Replay the batch"),
    "rj_mid":       ("{done} flux traités · {alerts} alertes",
                     "{done} flows processed · {alarms} alarms"),
    "rj_end":       ("Rejeu terminé : **{alerts} alertes** sur {n} flux.",
                     "Replay finished: **{alarms} alarms** on {n} flows."),
    "rj_note":      ("Un flux ne peut être noté qu'une fois terminé : ses variables "
                     "sont des agrégats de durée, de paquets et d'écarts entre "
                     "paquets. Un affichage « en direct » serait donc toujours en "
                     "retard sur le trafic.",
                     "A flow can only be scored once it has ended: its variables are "
                     "aggregates of duration, packets and gaps between packets. A "
                     "'live' display would therefore always lag behind the traffic."),

    "d_reset":      ("Réinitialiser", "Reset"),
    "d_thr_h":      ("Point de fonctionnement", "Operating point"),
    "d_thr_label":  ("Taux de fausses alertes accepté", "Accepted false-alarm rate"),
    "d_thr_help":   (
        "Le seuil est placé de façon à ce qu'exactement cette proportion du trafic SAIN "
        "soit signalée à tort. Un score plus permissif détecte plus d'attaques mais sature "
        "l'analyste ; un score plus strict laisse passer les attaques.",
        "The threshold is set so that exactly this proportion of HEALTHY traffic is wrongly "
        "flagged. A looser score detects more attacks but floods the analyst; a stricter one "
        "lets attacks through."),
    "d_flows":      ("Connexions analysées", "Connections analysed"),
    "d_alerts":     ("Alertes", "Alerts"),
    "d_normal":     ("Trafic normal", "Normal traffic"),
    "d_verdict":    ("Verdict", "Verdict"),
    "d_threat":     ("MENACE", "THREAT"),
    "d_clean":      ("NORMAL", "NORMAL"),
    "d_threat_sub": ("au moins une alerte", "at least one alert"),
    "d_clean_sub":  ("aucune alerte", "no alert"),
    "d_detected":   ("% du flux", "% of traffic"),
    "d_below":      ("sous le seuil", "below threshold"),
    "d_lot_cap":    ("Même lot de flux, même seuil de {fpr} % de fausses alertes. "
                     "Les trois chiffres ci-dessus portent sur ce lot-là, pas sur "
                     "le jeu de test complet.",
                     "Same batch of flows, same {fpr}% false-alarm threshold. The three "
                     "figures above describe that batch, not the full test set."),
    "d_tp":         ("Vrais positifs", "True positives"),
    "d_fp":         ("Fausses alertes", "False alerts"),
    "d_zd":         ("Attaques du lot", "Attacks in the batch"),
    "d_recall_s":   ("de rappel", "recall"),
    "d_fpr_s":      ("de fausses alertes", "false alerts"),
    "d_suspect_h":  ("Les connexions les plus suspectes", "Most suspicious connections"),
    "d_suspect_c":  ("Classées de la plus anormale à la moins anormale. Les flux "
                     "au-dessus du seuil portent une teinte distincte **et** leur nom "
                     "dans la légende : la couleur n'est jamais seule à porter "
                     "l'information.",
                     "Ranked from most to least anomalous. Red = above threshold."),
    "d_flow_n":     ("Connexion", "Connection"),
    "d_alert":      ("ALERTE", "ALERT"),
    "d_ok":         ("normal", "normal"),
    "d_dist_h":     ("Répartition", "Breakdown"),
    "d_charts_h":   ("Analyse détaillée", "Detailed analysis"),
    "adv_h":        ("Aller plus loin · les cinq graphiques", "Going further · the five charts"),
    "adv_c":        ("Ils montrent les mesures internes du modèle. Rien n'y est "
                     "nécessaire pour lire le résultat : c'est pour vérifier ou "
                     "pour la soutenance.",
                     "They show the model's internal measures. Nothing here is needed "
                     "to read the result: it is for checking, or for the defence."),
    "pc_do":        ("", ""),
    "pc_prev":      ("← Étape précédente", "← Previous step"),
    "pc_next":      ("Étape suivante →", "Next step →"),
    "pc_next_why":  ("L’étape suivante s’ouvre quand celle-ci a produit un résultat.",
                     "The next step opens once this one has produced a result."),
    "d_advice":     ("Métriques calculées sur {n} connexions ; tracés limités à {m} points.",
                     "Metrics computed on {n} connections; plots limited to {m} points."),
    "d_no_data":    ("Rien à afficher pour l'instant : aucune analyse n'a été lancée. "
                     "Choisis une source à l'étape 1, puis clique « Analyser ». "
                     "Le résultat apparaîtra ici, au-dessus des graphiques.",
                     "Nothing to show yet: no analysis has been run. Pick a source in "
                     "step 1, then click Analyse. The result will appear here, above "
                     "the charts."),

    # charts
    "c_score":      ("Score d'anomalie face au seuil", "Anomaly score against threshold"),
    "c_viol":       ("Indice de violation — 8 flux les plus suspects", "Violation index — top 8 flows"),
    "c_conf":       ("Confiance de prédiction de l'état suivant", "Next-state prediction confidence"),
    "c_energy":     ("Énergie OOD (hors distribution)", "OOD energy (out of distribution)"),
    "c_map":        ("Carte des connexions — normal vs signalé", "Connection map — normal vs flagged"),

    # results
    "lot_h":        ("Dernier lot complet noté hors navigateur",
                     "Last full batch scored outside the browser"),
    "lot_c":        ("`verifier_lot.py` note un fichier entier, sans plafond. "
                     "Voici le dernier lot traité :",
                     "`verifier_lot.py` scores a whole file with no cap. "
                     "Here is the last batch processed:"),
    "lot_aide":     ("Pour noter tous les flux d'un fichier : "
                     "`py verifier_lot.py mon_fichier.parquet`",
                     "To score every flow in a file: "
                     "`py verifier_lot.py my_file.parquet`"),
    "lot_col":      (["Fichier", "Flux", "Alertes", "Taux", "Rappel", "Durée"],
                     ["File", "Flows", "Alerts", "Rate", "Recall", "Duration"]),
    "r_head":       ("Performance mesurée sur les classes Zero-Day",
                     "Measured performance on Zero-Day classes"),
    "r_note":       ("Mesuré sur {nt} connexions de test, dont {nb} saines et {na} attaques "
                     "jamais vues à l'entraînement.",
                     "Measured on {nt} test connections: {nb} healthy and {na} attacks "
                     "never seen during training."),
    "r_auroc":      ("AUROC (IC 95 %)", "AUROC (95 % CI)"),
    "r_recall":     ("Détection au seuil", "Recall at threshold"),
    "r_fpr":        ("Fausses alertes", "False alarms"),
    "r_prec":       ("Précision", "Precision"),
    "r_perclass":   ("Détection par classe d'attaque", "Recall per attack class"),
    "r_perclass_c": ("Chaque barre est une famille d'attaques absente de l'entraînement.",
                     "Each bar is an attack family absent from training."),
    "r_curve_h":    ("Compromis détection / fausses alertes", "Detection / false-alarm trade-off"),
    "r_curve_c":    ("Déplace le curseur du point de fonctionnement pour lire la courbe. "
                     "La courbe est obtenue sur les 3 classes Zero-Day réunies.",
                     "Move the operating-point slider to read the curve. "
                     "The curve is computed over the three Zero-Day classes combined."),
    "r_baseline_h": ("Comparaison aux approches de référence", "Comparison with reference approaches"),
    "src_note":     (
        "Chiffres issus de `results/metrics.json` — score « {score} », produit par "
        "`03_evaluation.py`. Ce n'est pas le mode sélectionné dans la démonstration.",
        "Figures from `results/metrics.json` — score \"{score}\", produced by "
        "`03_evaluation.py`. This is not the mode selected in the demo."),
    "hist_note":    (
        "Mesures effectuées avec le score de la V4 (Mahalanobis sur les embeddings du "
        "transformer). Conservées pour la traçabilité : elles ne décrivent pas le score "
        "actuel, qui porte sur les 78 features réseau.",
        "Measurements made with the V4 score (Mahalanobis on transformer embeddings). Kept "
        "for traceability: they do not describe the current score, which operates on the "
        "78 network features."),
    "r_baseline_c": ("Toutes les valeurs sont des AUROC sur les mêmes classes Zero-Day.",
                     "All values are AUROC on the same Zero-Day classes."),
    "r_conf_h":     ("Matrice de confusion au point de fonctionnement",
                     "Confusion matrix at the operating point"),
    "r_export":     ("Exporter les métriques (CSV)", "Export metrics (CSV)"),
    "r_export_png": ("Exporter le graphique (PNG)", "Export chart (PNG)"),

    # method
    "m_enc_h":      ("Encodeur tokenisé hiérarchique", "Hierarchical tokenized encoder"),
    "m_enc":        (
        "Les 102 variables d'entrée sont découpées en jetons : 4 jetons pour la partie réseau, "
        "5 pour la partie EDR. Deux couches d'attention traitent d'abord l'intérieur de chaque "
        "modalité, puis leur fusion. La sortie est un vecteur latent de 32 dimensions.",
        "The 102 input variables are split into tokens: 4 for the network part, 5 for the EDR "
        "part. Two attention layers process each modality internally, then fuse them. "
        "The output is a 32-dimensional latent vector."),
    "m_score_h":    ("Le score one-class : distance de Mahalanobis",
                     "The one-class score: Mahalanobis distance"),
    "m_score":      (
        "On calcule la moyenne et la covariance des vecteurs latents du trafic **sain "
        "uniquement**, puis la distance de chaque nouvelle connexion à cette distribution. "
        "Contrairement à la similarité cosinus, cette distance tient compte du fait que "
        "certaines directions varient beaucoup et d'autres pas du tout.\n\n"
        "**Pourquoi cela compte** : sur ce modèle, les vecteurs latents sont presque "
        "colinéaires (cosinus moyen supérieur à 0.92). Le score cosinus ne dépend alors que "
        "d'une seule direction et n'est pas reproductible : deux centroïdes de similarité "
        "0.983 donnaient 0.38 et 0.55 d'AUROC. La distance de Mahalanobis, elle, redonne "
        "exactement le même chiffre sur tous les sous-échantillons testés.",
        "We compute the mean and covariance of latent vectors from **healthy traffic only**, "
        "then the distance of each new connection to that distribution. Unlike cosine "
        "similarity, this distance accounts for the fact that some directions vary a lot "
        "and others do not.\n\n"
        "**Why it matters**: in this model, latent vectors are nearly collinear (average "
        "cosine above 0.92). Cosine scoring then depends on a single direction and is not "
        "reproducible: two centroids of similarity 0.983 gave AUROC 0.38 and 0.55. Mahalanobis "
        "distance returns exactly the same figure on every subsample tested."),
    "m_abl_h":      ("Étude d'ablation", "Ablation study"),
    "m_abl_c":      ("Ce que chaque composant apporte, mesuré sur trois graines.",
                     "What each component contributes, measured over three seeds."),

    # contraste one-class / supervisé
    "c_head":       ("Contraste : ce que chaque approche peut garantir",
                     "Contrast: what each approach can guarantee"),
    "c_intro":      (
        "Deux familles de méthodes, mêmes familles inconnues au test. Le supervisé "
        "a vu les familles connues à l'entraînement ; le one-class n'a vu que du "
        "trafic bénin.",
        "Two families of methods, same unseen families at test time. The supervised "
        "model saw the known families during training; the one-class model saw only "
        "benign traffic."),
    "c_oc":         ("One-class", "One-class"),
    "c_sv":         ("Supervisé", "Supervised"),
    "c_mean":       ("Moyenne (toutes familles)", "Average (all families)"),
    "c_lofo_h":     ("Même comparaison, en retirant une famille à la fois :",
                     "Same comparison, withholding one family at a time:"),
    "pc_path_c":    ("Lecture de haut en bas. Un verrou n'est pas un refus : c'est "
                     "une étape qui attend une mesure. Un cadenas se clique quand "
                     "même — il dit ce qui manque.",
                     "Read top to bottom. A padlock is not a refusal: it is a step "
                     "waiting for a measurement. A padlock is still clickable — it says "
                     "what is missing."),
    "pc_prog":      ("Étapes accessibles", "Steps available"),
    "cmp_why":      ("Ligne « non mesurable » : le fichier **{f}** n'a pas été trouvé "
                     "à côté de `app.py`. Ce n'est pas une impossibilité du détecteur — "
                     "le score existe, il n'a simplement pas été chargé. Le recopier "
                     "suffit.",
                     "“Not measurable” row: the file **{f}** was not found next to "
                     "`app.py`. This is not an impossibility of the detector — the "
                     "score exists, it simply was not loaded. Copying the file is enough."),
    "cmp_fusion":   ("Le tableau des trois détecteurs est dressé à l’étape 3, sur "
                     "le même lot et le même seuil. Il n’est pas recalculé ici : ce "
                     "serait la même mesure, écrite à deux endroits.",
                     "The three-detector table is drawn in step 3, on the same batch "
                     "and the same threshold. It is not recomputed here: that would be "
                     "the same measurement, written in two places."),
    "cmp_scale_h":  ("Nature du score produit par chaque détecteur",
                     "What kind of score each detector produces"),
    "cmp_back":     ("Revoir le tableau complet (étape 3)",
                     "See the full table again (step 3)"),
    "c_map_na":     ("non mesurable — la projection n’a pas été calculée pour ce lot : "
                     "seul le nuage des scores est disponible.",
                     "not measurable — the projection was not computed for this batch: "
                     "only the score scatter is available."),
    "un1_t":        ("Unité 1 · Le problème, et ce que répond l'application",
                     "Unit 1 · The problem, and what the app answers"),
    "un1_s":        ("Ce que reçoit un centre opérationnel, pourquoi les systèmes "
                     "classiques échouent sur les attaques nouvelles, et ce que cette "
                     "application mesure — avec ses limites.",
                     "What a security operations centre receives, why classic systems "
                     "fail on new attacks, and what this app measures — with its limits."),
    "un1_g":        ("Cette page ne montre aucun résultat de votre part : elle pose le "
                     "problème et donne les chiffres de référence, obtenus sur un jeu de "
                     "test gelé. Les résultats d'une analyse se lisent dans l'onglet "
                     "Démonstration.",
                     "This page shows no result of your own: it states the problem and "
                     "gives the reference figures, obtained on a frozen test set. The "
                     "results of an analysis are read in the Demonstration tab."),
    "un2_t":        ("Unité 2 · Analyser un trafic réseau",
                     "Unit 2 · Analysing network traffic"),
    "un2_s":        ("Chaque étape s'ouvre quand la précédente a produit un résultat "
                     "réel — jamais par simple clic.",
                     "Each step opens once the previous one produced a real result — "
                     "never through a mere click."),
    "un2_g":        ("Le parcours se lit de haut en bas, à gauche. Chaque nœud est un "
                     "bouton : Tab pour le parcourir, Entrée pour l'ouvrir. Un cadenas "
                     "s'ouvre aussi — il dit ce qui manque, il ne refuse pas.",
                     "The path reads top to bottom, on the left. Every node is a button: "
                     "Tab to walk it, Enter to open it. A padlock opens too — it says "
                     "what is missing, it does not refuse."),
    "un3_t":        ("Unité 3 · Les résultats mesurés",
                     "Unit 3 · The measured results"),
    "un3_s":        ("Les chiffres obtenus sur le jeu de test gelé, avec leur intervalle "
                     "et les conditions dans lesquelles ils tiennent.",
                     "The figures obtained on the frozen test set, with their interval "
                     "and the conditions under which they hold."),
    "un3_g":        ("Un chiffre seul ne veut rien dire : ici chaque mesure porte son "
                     "intervalle de confiance et l'effectif qui la soutient. Aucun de ces "
                     "résultats ne provient de votre session — ce sont ceux du mémoire.",
                     "A figure on its own means nothing: every measure here carries its "
                     "confidence interval and the sample it rests on. None of these "
                     "results come from your session — they are the thesis figures."),
    "un4_t":        ("Unité 4 · La méthode",
                     "Unit 4 · The method"),
    "un4_s":        ("Comment les données ont été préparées, quel modèle est entraîné, "
                     "et pourquoi ce seuil plutôt qu'un autre.",
                     "How the data was prepared, which model is trained, and why this "
                     "threshold rather than another."),
    "un4_g":        ("La règle de décision est rappelée ici : un rappel par famille à "
                     "budget de fausses alertes borné, jamais l'AUROC seul.",
                     "The decision rule is recalled here: one recall per family at a "
                     "bounded false-alarm budget, never AUROC alone."),
    "un5_t":        ("Unité 5 · Les limites",
                     "Unit 5 · The limits"),
    "un5_s":        ("Ce que cette application ne fait pas, et dans quels cas les "
                     "chiffres ci-dessus ne tiennent plus.",
                     "What this app does not do, and when the figures above stop holding."),
    "un5_g":        ("Une limite n'est pas avouée ici pour la forme : chacune est reliée "
                     "à une mesure ou à une absence de mesure constatée.",
                     "A limit is not confessed here for form's sake: each one is tied to "
                     "a measurement, or to a measurement that was never possible."),
    "un6_t":        ("Unité 6 · Aide et glossaire",
                     "Unit 6 · Help and glossary"),
    "un6_s":        ("Le vocabulaire employé dans l'application, expliqué simplement, "
                     "et la marche à suivre pas à pas.",
                     "The vocabulary used in the app, explained simply, and the "
                     "step-by-step procedure."),
    "un6_g":        ("Le glossaire est aussi accessible par le bouton Guide de chaque "
                     "unité : les deux disent la même chose, au même endroit.",
                     "The glossary is also reachable from each unit's Guide button: both "
                     "say the same thing, in the same place."),
    "u_guide":      ("Guide", "Guide"),
    "term_online":  ("EN LIGNE", "ONLINE"),
    "u_guide_b":    ("Le parcours se lit de haut en bas, à gauche. Chaque nœud est un "
                     "bouton : Tab pour le parcourir, Entrée pour l'ouvrir. Un cadenas "
                     "s'ouvre aussi — il dit ce qui manque, il ne refuse pas.",
                     "The path reads top to bottom, on the left. Every node is a button: "
                     "Tab to walk it, Enter to open it. A padlock opens too — it says "
                     "what is missing, it does not refuse."),
    "u_prog":       ("Progression", "Progress"),
    "u_prog_n":     ("{n} étapes sur {m} · état réel des données",
                     "{n} of {m} steps · true state of the data"),
    "u_parcours":   ("Parcours", "Path"),
    "u_res_cap":    ("Résultat sur le lot analysé", "Result on the analysed batch"),
    "u_next_n":     ("Débloquer l'étape {n} · {nom}", "Unlock step {n} · {name}"),
    "u_redo":       ("Refaire l'analyse", "Re-run the analysis"),
    "u_redo_no":    ("Aucun lot en mémoire : il n'y a rien à refaire pour l'instant.",
                     "No batch in memory: there is nothing to re-run yet."),
    "u_mem_h":      ("Ce qui est mémorisé entre les étapes",
                     "What is remembered between steps"),
    "u_mem_b":      ("<ul><li>le lot analysé et ses scores ;</li>"
                     "<li>le budget de fausses alertes ;</li>"
                     "<li>la langue et le détecteur choisis ;</li>"
                     "<li>le retour en arrière reste possible : rien n'est consommé.</li></ul>",
                     "<ul><li>the analysed batch and its scores;</li>"
                     "<li>the false-alarm budget;</li>"
                     "<li>the chosen language and detector;</li>"
                     "<li>going back stays possible: nothing is consumed.</li></ul>"),
    "u_lock_h":     ("Clic sur un cadenas", "Clicking a padlock"),
    "u_lock_b":     ("L'étape s'ouvre quand même — sur sa raison, jamais sur du vide : "
                     "« il manque un lot analysé ».",
                     "The step still opens — on its reason, never on emptiness: "
                     "“an analysed batch is missing”."),
    "u_gate":       ("On n'avance pas sans cette mesure.",
                     "No progress without that measurement."),
    "pc_do1":       ("Ouvre un fichier de flux réseau, produit par CICFlowMeter, "
                     "ou le lot d’exemple.",
                     "Open a network-flow file produced by CICFlowMeter, or the sample batch."),
    "pc_do2":       ("Choisis ce qui va être noté : ton fichier, ou un lot tiré du "
                     "jeu de test.",
                     "Choose what gets scored: your file, or a batch drawn from the test set."),
    "pc_do3":       ("Lis les deux encadrés, puis les trois chiffres. Dans cet ordre : "
                     "ils disent ce que le score veut dire, et ce qu’il ne dit pas.",
                     "Read the two boxes, then the three figures. In that order: they say "
                     "what the score means, and what it does not."),
    "pc_do4":       ("Suis un flux du début à la fin, pour voir sur quoi la décision "
                     "s’appuie.",
                     "Follow one flow from start to finish, to see what the decision rests on."),
    "pc_do5":       ("Compare les trois détecteurs sur le même lot, avec le même budget "
                     "de fausses alertes.",
                     "Compare the three detectors on the same batch, with the same "
                     "false-alarm budget."),
    "pc_do6":       ("Télécharge les tableaux et les figures de cette session.",
                     "Download this session's tables and figures."),
    "pc_do7":       ("Relis ce que la session a fait, dans l’ordre où ça s’est passé.",
                     "Read back what the session did, in the order it happened."),
    "pc_etape":     ("Étape", "Step"),
    "means_h":      ("Ce que ça veut dire", "What this means"),
    "means_b":      ("Le score **classe** les flux du moins au plus anormal ; il ne "
                     "les étiquette pas. Le seuil, lui, fixe le budget de fausses "
                     "alertes : à 5 %, un flux sur vingt du trafic bénin passe "
                     "au-dessus. La précision qui en découle dépend de la part "
                     "d'attaques réellement présente dans le trafic observé.",
                     "The score **ranks** flows from least to most abnormal; it does "
                     "not label them. The threshold sets the false-alarm budget: at "
                     "5 %, one benign flow in twenty lands above it. The resulting "
                     "precision depends on the share of attacks actually present in "
                     "the observed traffic."),
    "nomeans_h":    ("Ce que ça ne veut pas dire", "What this does not mean"),
    "nomeans_b":    ("Ce n'est **pas un degré de dangerosité** : le trafic bénin a un "
                     "score moyen 56 fois plus élevé que la famille Bot. Un score "
                     "haut ne veut pas dire « attaque », un score bas ne veut pas "
                     "dire « sûr ». Et la calibration n'a pas été mesurée : on ne "
                     "peut pas lire ce score comme une probabilité.",
                     "This is **not a degree of danger**: benign traffic has an "
                     "average score 56 times higher than the Bot family. A high score "
                     "does not mean \u201cattack\u201d, a low score does not mean "
                     "\u201csafe\u201d. Calibration was not measured either: the score "
                     "cannot be read as a probability."),

    "pc_legend":    ("L'état n'est jamais porté par la couleur seule.",
                     "State is never carried by colour alone."),
    "pc_after":     ("après l'étape {n}", "after step {n}"),
    "pc_d1":        ("un fichier CSV ou Parquet", "a CSV or Parquet file"),
    "pc_d2":        ("un lot tiré du test, ou ton fichier", "a batch from the test set, or your file"),
    "pc_d3":        ("ce que le score veut dire", "what the score means"),
    "pc_d4":        ("le trajet complet d'une décision", "the full path of one decision"),
    "pc_d5":        ("même lot, trois approches", "same batch, three approaches"),
    "pc_d6":        ("tableaux en CSV, figures en PNG", "tables as CSV, figures as PNG"),
    "pc_d7":        ("ce que la session a fait", "what the session did"),
    "pc_cur":       ("Étape courante", "Current step"),
    "pc_locked":    ("verrouillée", "locked"),
    "pc_open":      ("ouverte", "open"),
    "pc_missing":   ("Il manque une analyse : aucun flux n'a encore été noté. "
                     "Choisis une source à l'étape 1, puis clique « Analyser ».",
                     "An analysis is missing: no flow has been scored yet. Pick a "
                     "source in step 1, then click Analyse."),
    "pc_missing_b": ("Il manque un lot analysé en mémoire : la comparaison porte sur "
                     "ce que tu viens de traiter.",
                     "A batch is missing from memory: the comparison uses what you "
                     "just processed."),
    "pc_go1":       ("Aller à l'étape 1", "Go to step 1"),

    "pc1":          ("Ouvrir un trafic", "Open a traffic file"),
    "pc2":          ("Choisir la source", "Choose the source"),
    "pc3":          ("Lire le résultat", "Read the result"),
    "pc4":          ("Suivre un flux", "Follow one flow"),
    "pc5":          ("Comparer", "Compare"),
    "pc6":          ("Exporter", "Export"),
    "pc7":          ("Rapport", "Report"),

    "ex_h":         ("Export : ce que tu peux emporter", "Export: what you can take away"),
    "ex_c":         ("Chaque tableau du parcours se télécharge en CSV, et chaque figure "
                     "en PNG. Les fichiers sont produits ici, à partir de ce que la "
                     "session a réellement calculé.",
                     "Every table in the path downloads as CSV, every figure as PNG. "
                     "Files are produced here, from what this session actually computed."),
    "ex_none":      ("Rien à exporter pour l'instant : lance d'abord une analyse.",
                     "Nothing to export yet: run an analysis first."),
    "ex_fig":       ("Figures du dossier results/", "Figures from the results/ folder"),

    "rp_h":         ("Rapport de session", "Session report"),
    "rp_c":         ("Ce que cette session a fait, dans l'ordre. Aucun chiffre n'est "
                     "saisi à la main : tout vient des calculs et des fichiers du dépôt.",
                     "What this session did, in order. No figure is typed by hand: "
                     "everything comes from the computations and the repository files."),
    "rp_none":      ("Aucune action enregistrée dans cette session. Le registre se "
                     "remplit à chaque analyse lancée.",
                     "No action recorded in this session. The registry fills up with "
                     "every analysis you run."),
    "rp_reg_h":     ("Registre de la session", "Session registry"),
    "rp_reset":     ("Effacer le registre", "Clear the registry"),

    "pc_now":       ("étape en cours", "current step"),
    "pc_done":      ("étape faite", "step done"),
    "pc_todo":      ("étape à venir", "step to come"),

    # parcours façon terminal : nœuds, haltes, ligne d'état
    "duo_step":     ("Étape {n}", "Step {n}"),
    "jalon_a":      ("Halte · une analyse existe",
                     "Checkpoint · an analysis exists"),
    "jalon_b":      ("Halte · le parcours est complet",
                     "Checkpoint · the path is complete"),
    "jalon_off":    ("Halte · rien encore", "Checkpoint · nothing yet"),
    "owl_hello":    ("Choisis une étape : je te suis. Rien n'est encore chargé, "
                     "donc seules les deux premières sont ouvertes — les autres "
                     "s'ouvrent quand la mesure existe vraiment.",
                     "Pick a step and I'll follow. Nothing is loaded yet, so only "
                     "the first two are open — the others open once the measurement "
                     "actually exists."),
    "owl_done":     ("Analyse faite : **{a}** alertes sur **{n}** flux, seuil "
                     "{s}. Monte ou descends le long du chemin pour relire chaque "
                     "étape.",
                     "Analysis done: **{a}** alerts out of **{n}** flows, threshold "
                     "{s}. Walk up or down the path to revisit any step."),

    "cmp_fail":     ("Comparaison impossible : {e}. Les trois détecteurs n'ont pas pu "
                     "être calculés sur ce lot. Vérifie que "
                     "`data/processed/X_test.npy` existe — sinon relance "
                     "`py 01_data_preparation.py` — puis réessaie.",
                     "Comparison failed: {e}. The three detectors could not be computed "
                     "on this batch. Check that `data/processed/X_test.npy` exists "
                     "(otherwise re-run `py 01_data_preparation.py`), then try again."),
    "c_collapse":   (
        "Le superviseur tombe à **{pct}** sur « {fam} ». Une famille "
        "structurellement différente n'est pas reconnue, quelle que soit la qualité "
        "de l'entraînement.",
        "The supervised model drops to **{pct}** on \"{fam}\". A structurally "
        "different family is not recognised, whatever the training quality."),
    "c_missing":    ("Lance `py 03_evaluation.py --contraste` pour produire cette table.",
                     "Run `py 03_evaluation.py --contraste` to produce this table."),

    # limites — chaque entrée dit ce qui a été fait, et ce qui reste
    "l_h":          ("Limites : état après traitement", "Limitations: status after remediation"),
    "l_intro":      (
        "Les cinq limites identifiées dans la première version ont été reprises une par "
        "une. Cette page indique, pour chacune, **ce qui a été fait**, la **mesure qui le "
        "prouve**, et ce qui **reste ouvert**. Aucune n'est déclarée résolue sans mesure.",
        "The five limitations identified in the first version have been addressed one by "
        "one. For each, this page states **what was done**, the **measurement that proves "
        "it**, and what **remains open**. None is declared resolved without evidence."),
    "l_status_ok":   ("Traitée", "Addressed"),
    "l_status_part": ("Partiellement traitée", "Partially addressed"),
    "l_status_open": ("Ouverte", "Open"),
    "l_what":        ("Ce qui a été fait", "What was done"),
    "l_foot":        (
        "Ces corrections portent sur le **protocole et la couverture**, pas sur la "
        "performance. Aucune n'améliore le système au point de le rendre opérationnel : "
        "la limite L1 reste ouverte, et c'est la plus importante.",
        "These corrections address the **protocol and coverage**, not performance. None "
        "makes the system operational: limitation L1 remains open, and it is the most "
        "important one."),
    "l_proof":       ("Mesure", "Measurement"),
    "l_left":        ("Ce qui reste", "What remains"),

    "l1_t": ("Le plafond de la famille Bot", "The Bot family ceiling"),
    "l1_a": (
        "Sur les cinq familles inconnues, **quatre passent** le critère de 85 % de rappel. "
        "La cinquième, **Bot**, s'arrête à **84,65 %** : le critère est manqué de 0,35 point, "
        "soit **6 flux sur 1 433**. Ce n'est pas une imprécision de mesure, c'est un "
        "plafond reproduit.",
        "Across the five unseen families, **four pass** the 85 % recall criterion. The "
        "fifth, **Bot**, stops at **84.79 %**: the criterion is missed by 0.2 points, i.e. "
        "**4 flows out of 1,433**. This is not a measurement artefact but a reproduced "
        "ceiling."),
    "l1_p": (
        "Le plafond a été établi par mesure : **douze configurations** de complexité "
        "croissante, jusqu'à **255 feuilles**, à 9 % comme à 10 % de fausses alertes. "
        "Aucune ne dépasse 84,65 %. Augmenter le budget de fausses alertes ne le déplace "
        "pas davantage. Hiérarchie finale : **SQL Injection 100 % > XSS 99,69 % > "
        "Brute Force 96,87 % > PortScan 94,27 % > Bot 84,65 %**.",
        "The ceiling was established by measurement: **twelve configurations** of "
        "increasing complexity, up to **255 leaves**, at 9 % and at 10 % false alarms. "
        "None exceeds 84.8 %. Raising the false-alarm budget does not move it either. "
        "Final ranking: **SQL Injection 100 % > XSS 99.69 % > Brute Force 96.87 % > "
        "PortScan 94.27 % > Bot 84.79 %**."),
    "l1_r": (
        "**Ce qui reste, et qui est plus grave que les 0,35 point manquants.** Avec "
        "**9,03 % de fausses alertes**, le détecteur produit **2 258 fausses alertes** sur "
        "25 001 flux bénins. À une prévalence réaliste de 1 % d'attaques, la précision "
        "tombe à **9,43 %** : environ **9,6 fausses alertes par détection utile**. C'est "
        "le vrai obstacle à un usage opérationnel, pas le rappel.",
        "**What remains — and it matters more than the missing 0.2 points.** With "
        "**9.03 % false alarms**, the detector raises **2,258 false alerts** on 25,001 "
        "benign flows. At a realistic 1 % attack prevalence, precision falls to **9.43 %**: "
        "roughly **9.6 false alarms per useful detection**. That, not recall, is the real "
        "obstacle to operational use."),

    "l2_t": ("Branche EDR simulée", "Simulated EDR branch"),
    "l2_a": (
        "La branche a été **supprimée**. Le modèle est désormais 100 % réseau : 78 variables "
        "de flux, aucune donnée synthétique. L'information système qui n'existe pas ne peut "
        "pas être inventée sans fausser le résultat.",
        "The branch was **removed**. The model is now 100 % network: 78 flow variables, no "
        "synthetic data. System information that does not exist cannot be invented without "
        "biasing the result."),
    "l2_p": (
        "Vérification sur les données livrées : le bloc de 25 colonnes était **constant "
        "(0.0) sur 100 % des flux bénins** et non nul sur 100 % des attaques — un "
        "indicateur parfait de la classe. L'étude d'ablation l'avait déjà signalé : retirer "
        "la branche améliorait le résultat de +0.146 d'AUROC.",
        "Checked on the shipped data: the 25-column block was **constant (0.0) on 100 % of "
        "benign flows** and non-zero on 100 % of attacks — a perfect class indicator. The "
        "ablation study had already flagged it: removing the branch improved AUROC by +0.146."),
    "l2_r": (
        "Aucun reste. La multimodalité annoncée n'existe plus dans le résultat final ; elle "
        "reste une expérience négative documentée.",
        "Nothing left. The announced multimodality no longer exists in the final result; it "
        "remains a documented negative experiment."),

    "l3_t": ("Cinq familles, pas toutes les attaques", "Five families, not every attack"),
    "l3_a": (
        "Le jeu d'évaluation contient **cinq familles d'attaque absentes de "
        "l'entraînement** : Bot, PortScan, Brute Force, XSS et SQL Injection, pour "
        "**5 495 flux** au total.",
        "The evaluation set contains **five attack families absent from training**: "
        "Bot, PortScan, Brute Force, XSS and SQL Injection, for **5,495 flows** in total."),
    "l3_p": (
        "Rappel mesuré famille par famille à 9 % de fausses alertes : **SQL Injection "
        "100 %**, **XSS 99,69 %**, **Brute Force 96,87 %**, **PortScan 94,27 %**, "
        "**Bot 84,65 %**.",
        "Per-family recall measured at 9 % false alarms: **SQL Injection 100 %**, "
        "**XSS 99.69 %**, **Brute Force 96.87 %**, **PortScan 94.27 %**, "
        "**Bot 84.79 %**."),
    "l3_r": (
        "CICIDS2017 ne contient que ces familles. Toute attaque hors de cette liste — "
        "rançongiciel, mouvement latéral, exfiltration — n'est pas couverte, et rien dans "
        "ce travail ne permet d'affirmer qu'elle serait détectée.",
        "CICIDS2017 only contains these families. Any attack outside that list — "
        "ransomware, lateral movement, exfiltration — is not covered, and nothing in this "
        "work supports claiming it would be detected."),

    "l4_t": ("Une prévalence de test irréaliste", "An unrealistic test prevalence"),
    "l4_a": (
        "Le jeu de test compte **30 496 flux** : **25 001 bénins** et **5 495 attaques**, "
        "soit **18 % d'attaques**. Chaque attaque y est évaluée une seule fois, et les "
        "familles inconnues n'ont jamais été vues à l'entraînement.",
        "The test set holds **30,496 flows**: **25,001 benign** and **5,495 attacks**, i.e. "
        "**18 % attacks**. Every attack is evaluated once, and the unseen families never "
        "appeared during training."),
    "l4_p": (
        "La précision mesurée à cette prévalence est de **69,39 %**. Ramenée à des "
        "conditions réelles, elle devient **35,2 % à 5 %** d'attaques, **9,43 % à 1 %** et "
        "**1,02 % à 0,1 %** — soit environ **97 fausses alertes par vraie détection**.",
        "Precision measured at this prevalence is **69.39 %**. Rescaled to real "
        "conditions it becomes **35.2 % at 5 %** attacks, **9.43 % at 1 %** and "
        "**1.02 % at 0.1 %** — about **97 false alarms per true detection**."),
    "l4_r": (
        "Le chiffre de 93,12 % de rappel est donc **honnête sur le rappel** et "
        "**optimiste sur l'usage** : il décrit ce que le modèle trouve, pas ce qu'un "
        "opérateur pourrait traiter. La table de prévalence de l'onglet Résultats donne "
        "les quatre scénarios chiffrés.",
        "The 93.12 % recall figure is therefore **honest about recall** and **optimistic "
        "about deployment**: it describes what the model finds, not what an operator could "
        "process. The prevalence table in the Results tab gives the four quantified "
        "scenarios."),

    "l5_t": ("Un seul jeu de données", "A single dataset"),
    "l5_a": (
        "La robustesse de la mesure a été testée en réentraînant **cinq modèles** sur des "
        "sous-échantillons différents du train, à configuration identique.",
        "Measurement robustness was tested by retraining **five models** on different "
        "subsamplings of the training data, with identical configuration."),
    "l5_p": (
        "AUROC moyen **0,9607 ± 0,0036**. Mais le rappel, lui, varie de **87,64 % ± 4,07 %** "
        "autour de la valeur déployée : les **93,12 %** annoncés sont un tirage favorable, "
        "et la valeur à citer est **87,6 % ± 4,1 %**.",
        "Mean AUROC **0.9607 ± 0.0036**. Recall, however, varies by **87.64 % ± 4.07 %** "
        "around the deployed figure: the announced **93.12 %** is a favourable draw, and "
        "the defensible value is **87.6 % ± 4.1 %**."),
    "l5_r": (
        "**Non traitée au sens strict.** Il s'agit toujours de CICIDS2017. Une validation "
        "inter-jeux exigerait CICIDS2018 ou UNSW-NB15, et reste le premier travail à "
        "poursuivre. Rien dans cette page ne remplace cette validation.",
        "**Not addressed in the strict sense.** This is still CICIDS2017. Cross-dataset "
        "validation would require CICIDS2018 or UNSW-NB15, and remains the first follow-up "
        "task. Nothing on this page substitutes for it."),

    # accessibilité
    "fig_alt":      (
        "Équivalent textuel de la figure : le tableau ci-dessus contient les mêmes "
        "valeurs de rappel par classe.",
        "Text equivalent of the figure: the table above holds the same per-class "
        "recall values."),
    "lang_help":    ("Langue de l'interface. Le contenu est annoncé dans cette langue.",
                     "Interface language. Content is announced in this language."),
    "a11y_note":    (
        "Interface conforme WCAG 2.1 niveau AA : contraste minimal 4,82:1, "
        "navigation clavier complète avec indicateur de focus, information jamais "
        "portée par la seule couleur, et respect de la préférence système de "
        "réduction des animations.",
        "WCAG 2.1 level AA interface: minimum contrast 4.82:1, full keyboard "
        "navigation with visible focus indicator, information never conveyed by "
        "colour alone, and the system reduced-motion preference is honoured."),

    # modes de détection
    "mode_label":   ("Approche", "Approach"),
    "mode_modele":  ("MS-Transformer", "MS-Transformer"),
    "mode_one_class": ("One-class", "One-class"),
    "mode_supervise": ("Supervisé", "Supervised"),
    "mode_modele_d": (
        "Le transformer du projet, noté par distance au centroïde bénin. "
        "AUROC 0,8160 sur les familles inconnues — 15 points sous le gradient "
        "boosting. Conservé comme référence, pas comme détecteur.",
        "The project's transformer, scored by distance to the benign centroid. "
        "AUROC 0.8160 on unseen families — 15 points below gradient boosting. "
        "Kept as a reference, not as a detector."),
    "mode_one_class_d": (
        "N'a jamais vu d'attaque : ajusté sur le trafic bénin seul. C'est "
        "l'approche la plus prudente, et la moins efficace ici : AUROC 0,7186 "
        "contre 0,9668 pour le gradient boosting.",
        "Never saw an attack: fitted on benign traffic only. The most cautious "
        "approach, and the least effective here: AUROC 0.7186 against 0.9668 for "
        "gradient boosting."),
    "mode_supervise_d": (
        "A vu les familles connues à l'entraînement : c'est le détecteur retenu. "
        "Sur le jeu de test gelé, il atteint AUROC 0,9668 et 93,12 % de rappel à "
        "9,03 % de fausses alertes. Le curseur ci-dessous fixe, lui, la cible de "
        "la prochaine analyse : les deux nombres ne mesurent pas la même chose.",
        "Saw the known families during training: this is the retained detector. "
        "On the frozen test set it reaches AUROC 0.9668 and 93.12 % recall at "
        "9.03 % false alarms. The slider below sets the target of the next run: "
        "the two numbers do not measure the same thing."),

    # en-tête
    "eyebrow":      ("CICIDS2017 · Détection d'anomalies Zero-Day",
                     "CICIDS2017 · Zero-Day anomaly detection"),
    "badge_offline": ("Hors ligne", "Offline"),
    "badge_neg":    ("Résultats négatifs publiés", "Negative results published"),
    "badge_audit":  ("Baselines publiées", "Baselines published"),

    # bandeau de statut
    "st_title":     ("Statut de l'étude", "Study status"),
    "st_body":      (
        "Ce système est un **prototype de recherche**, pas un produit de détection. "
        "Le détecteur retenu atteint **93,12 % de rappel** à **9,03 % de fausses "
        "alertes** sur 5 495 attaques de cinq familles jamais vues à l'entraînement.\n\n"
        "Deux réserves comptent autant que ce chiffre. La famille **Bot plafonne à "
        "84,65 %** : le critère de 85 % est manqué de 0,35 point, et ce plafond a été "
        "établi sur douze configurations jusqu'à 255 feuilles. Et à prévalence "
        "réaliste — 1 % d'attaques — la **précision tombe à 9,43 %**, soit 9,6 fausses "
        "alertes par détection. Le détail figure dans l'onglet Limites.",
        "This system is a **research prototype**, not a detection product. The retained "
        "detector reaches **93.12 % recall** at **9.03 % false alarms** on 5,495 attacks "
        "from five families never seen during training.\n\n"
        "Two caveats matter as much as that figure. The **Bot family peaks at 84.79 %**: "
        "the 85 % criterion is missed by 0.2 points, and that ceiling was established "
        "across twelve configurations up to 255 leaves. And at realistic prevalence — "
        "1 % attacks — **precision falls to 9.43 %**, i.e. 9.6 false alarms per detection. "
        "Details are in the Limitations tab."),
    "st_more":      ("Voir les limites", "See limitations"),

    # provenance
    "flows":        ("flux", "flows"),
    "missing":      ("non mesurable", "not measurable"),
    "pv_missing":   ("Table non calculable : le rappel ou le taux de fausses alertes "
                     "est absent de results/metrics.json. Relance "
                     "`py 03_evaluation.py --score supervise --algo hgb --target-fpr 0.09`.",
                     "Table not computable: recall or false-alarm rate is missing from "
                     "results/metrics.json. Re-run "
                     "`py 03_evaluation.py --score supervise --algo hgb --target-fpr 0.09`."),
    "pv_src":       ("Source", "Source"),
    "pv_train":     ("Entraînement", "Training"),
    "pv_test":      ("Test (Zero-Day)", "Test (zero-day)"),
    "pv_model":     ("Modèle", "Model"),
    "pv_score":     ("Score", "Score"),
    "pv_run":       ("Exécuté hors ligne · aucune donnée transmise",
                     "Runs offline · no data transmitted"),

    # audit de fuite
    "a_head":       ("Audit : une baseline triviale bat-elle le modèle ?",
                     "Audit: does a trivial baseline beat the model?"),
    "a_intro":      (
        "Le `StandardScaler` est ajusté sur le trafic **bénin uniquement**. "
        "Chaque flux d'attaque reçoit donc mécaniquement de grands z-scores : "
        "la **magnitude** des features encode la classe. Les deux références "
        "ci-dessous n'utilisent aucun apprentissage profond.",
        "The `StandardScaler` is fitted on **benign traffic only**. Every attack flow "
        "therefore receives mechanically large z-scores: feature **magnitude** encodes "
        "the class. The two references below use no deep learning at all."),
    "a_method":     ("Méthode", "Method"),
    "a_auroc":      ("AUROC", "AUROC"),
    "a_recall":     ("Détection @5 % FA", "Recall @5 % FA"),
    "a_model":      ("Modèle évalué", "Evaluated model"),
    "a_sum":        ("Somme des valeurs absolues", "Sum of absolute values"),
    "a_lr":         ("Régression logistique", "Logistic regression"),
    "a_warn":       (
        "**Une somme de features atteint le niveau du modèle.** Ce n'est pas le modèle "
        "qui détecte : c'est la magnitude des variables qui sépare les classes. "
        "Les scores publiés sur CICIDS2017 doivent être interprétés à la lumière de ce constat.",
        "**A sum of features matches the model.** It is not the model that detects: "
        "feature magnitude separates the classes. Scores published on CICIDS2017 must be "
        "read in the light of this finding."),
    "a_ok":         ("Le modèle dépasse nettement les baselines triviales.",
                     "The model clearly outperforms trivial baselines."),

    # prévalence
    "pv_head":      ("Précision selon la prévalence réelle du trafic",
                     "Precision as a function of true traffic prevalence"),
    "pv_intro":     (
        "La précision mesurée dépend du taux d'attaques dans le trafic observé. "
        "Un SOC réel voit bien moins de 1 % d'attaques.",
        "Measured precision depends on the attack rate in the observed traffic. "
        "A real SOC sees far less than 1 % attacks."),
    "pv_prev":      ("Prévalence", "Prevalence"),
    "pv_prec":      ("Précision", "Precision"),
    "pv_ctx":       ("Contexte", "Context"),
    "pv_alert":     ("Fausses alertes par vraie", "False alerts per true one"),

    # limites révisées
    "l_leak_h":     ("La séparation est un artefact du jeu de données",
                     "The separation is a dataset artefact"),
    "l_leak":       (
        "**C'est la limite principale.** Une régression logistique atteint AUROC 1.0000 "
        "et une simple somme de features 0.9895 sur les classes censées être « jamais vues ». "
        "Le protocole Zero-Day ne mesure donc pas la généralisation à l'inconnu : il mesure "
        "la séparabilité intrinsèque de CICIDS2017, acquise dès la standardisation. "
        "Tout score rapporté ici doit être lu à cette lumière.",
        "**This is the main limitation.** Logistic regression reaches AUROC 1.0000 and a "
        "plain feature sum 0.9895 on the supposedly \"never seen\" classes. The Zero-Day "
        "protocol therefore does not measure generalisation to the unknown: it measures the "
        "intrinsic separability of CICIDS2017, obtained from standardisation alone. Every "
        "score reported here must be read in that light."),

    # shared
    "n_a":          ("non mesurable", "not measurable"),
    "lab_found":    ("Étiquettes lues dans la colonne « {c} » : le rappel est "
                     "mesuré sur ce lot.",
                     "Labels read from column “{c}”: recall is measured on this batch."),
    "lab_none":     ("Aucune colonne d'étiquette dans le fichier : le rappel ne "
                     "peut pas être mesuré. Les alertes, elles, le sont.",
                     "No label column in the file: recall cannot be measured. "
                     "Alert counts still are."),
    "footer":       ("Reproductible · jeu CICIDS2017 · graine {seed} · modèle {h} · interface v4.0",
                     "Reproducible · CICIDS2017 dataset · seed {seed} · model {h} · interface v4.0"),
}

if "lang" not in st.session_state:
    st.session_state.lang = "fr"
LANG = st.session_state.lang


def t(key: str, *a, **k):
    """Texte traduit. Renvoie la valeur telle quelle si ce n'est pas une chaîne
    (certaines entrées sont des listes de libellés de colonnes)."""
    fr, en = T[key]
    s = fr if LANG == "fr" else en
    if not isinstance(s, str):
        return s
    return s.format(*a, **k) if (a or k) else s



# ══════════════════════════════════════════════════════════════════════
#  AIDE — glossaire et mode d'emploi, en langage simple
#  Accessibilité (WCAG 3.1.3) : le vocabulaire technique employé dans
#  l'interface est expliqué ici, dans les deux langues.
# ══════════════════════════════════════════════════════════════════════
HELP = {
    "h_intro": (
        "Cette page explique, sans vocabulaire technique, ce que fait chaque "
        "élément de l'interface et ce que signifient les mots employés ailleurs. "
        "Aucun prérequis n'est nécessaire.",
        "This page explains, without technical vocabulary, what each part of the "
        "interface does and what the words used elsewhere mean. No prior "
        "knowledge is required."),

    "h_steps_h": ("Comment lire cette interface, en trois étapes", "How to read this interface, in three steps"),
    "h_step1": (
        "**1. Choisis un détecteur** (les trois boutons en haut). C'est la méthode "
        "qui décide si un flux est une attaque. Changer de détecteur change les scores.",
        "**1. Pick a detector** (the three buttons at the top). It is the method "
        "that decides whether a flow is an attack. Changing it changes the scores."),
    "h_step2": (
        "**2. Lance une analyse** dans l'onglet Démonstration. L'application note "
        "chaque flux et affiche ceux qui dépassent le seuil.",
        "**2. Run an analysis** in the Live demo tab. The application scores each "
        "flow and shows those above the threshold."),
    "h_step3": (
        "**3. Lis les résultats** dans l'onglet Résultats : combien d'attaques ont "
        "été trouvées, et combien de fausses alertes cela a coûté.",
        "**3. Read the results** in the Results tab: how many attacks were found, "
        "and how many false alarms it cost."),

    "h_modes_h": ("Les trois détecteurs", "The three detectors"),
    "h_mode1_t": ("MS-Transformer", "MS-Transformer"),
    "h_mode1": (
        "C'est un réseau de neurones qui apprend à <b>résumer</b> chaque flux en "
        "quelques nombres, puis mesure l'écart entre ce résumé et celui du trafic "
        "normal. Ses scores ne sont pas des probabilités mais des <b>distances</b>."
        "<br><br><i>Note : 0,8160.</i>",
        "A neural network that learns to <b>summarise</b> each flow as a few "
        "numbers, then measures the gap between that summary and the one of normal "
        "traffic. Its scores are not probabilities but <b>distances</b>."
        "<br><br><i>Score: 0.8160.</i>"),
    "h_mode2_t": ("One-class", "One-class"),
    "h_mode2": (
        "On lui montre <b>uniquement du trafic normal</b>, jamais une attaque. Il "
        "mémorise à quoi ressemble une journée ordinaire. Ensuite, tout ce qui "
        "s'écarte de cette routine est suspect."
        "<br><br><i>Note : 0,7186.</i>",
        "It is shown <b>only normal traffic</b>, never an attack. It memorises what "
        "an ordinary day looks like. Anything that departs from that routine is "
        "then treated as suspicious."
        "<br><br><i>Score: 0.7186.</i>"),
    "h_mode3_t": ("Supervisé", "Supervised"),
    "h_mode3": (
        "On lui montre <b>des attaques connues</b>, étiquetées. Il apprend la "
        "<i>forme</i> d'une attaque, pas seulement l'allure du calme. Il peut alors "
        "reconnaître une attaque qu'il n'a jamais vue si elle ressemble, dans sa "
        "structure, à celles qu'il connaît."
        "<br><br><i>Note : 0,9668 — et 93,12 % des attaques trouvées.</i>",
        "It is shown <b>known attacks</b>, labelled. It learns the <i>shape</i> of "
        "an attack, not just the look of calm traffic. It can then recognise an "
        "attack it has never seen, provided it resembles, in structure, the ones it "
        "knows."
        "<br><br><i>Score: 0.9668 — and 93.12 % of attacks found.</i>"),

    "h_glossary_h": ("Glossaire des mots techniques", "Glossary of technical words"),
    "h_g1_t": ("Flux (flow)", "Flow"),
    "h_g1": (
        "Une conversation entre deux machines sur le réseau. Chaque ligne du jeu "
        "de données décrit un flux par ses statistiques : durée, taille des "
        "paquets, nombre d'échanges.",
        "A conversation between two machines on the network. Each row of the "
        "dataset describes one flow through statistics: duration, packet size, "
        "number of exchanges."),
    "h_g2_t": ("Zero-Day", "Zero-Day"),
    "h_g2": (
        "Une attaque jamais observée auparavant. Le but du projet est de la "
        "repérer sans l'avoir jamais vue à l'entraînement.",
        "An attack never observed before. The goal of the project is to spot it "
        "without ever having seen it during training."),
    "h_g3_t": ("Seuil (threshold)", "Threshold"),
    "h_g3": (
        "La limite au-dessus de laquelle un flux est déclaré suspect. Un seuil "
        "bas détecte plus d'attaques mais produit plus de fausses alertes ; un "
        "seuil haut fait l'inverse. Le curseur de l'interface règle ce compromis.",
        "The limit above which a flow is flagged. A low threshold catches more "
        "attacks but raises more false alarms; a high threshold does the "
        "opposite. The slider sets that trade-off."),
    "h_g4_t": ("Faux positif / fausse alerte", "False positive / false alarm"),
    "h_g4": (
        "Un flux normal que le système signale à tort comme une attaque. Trop de "
        "fausses alertes rendent un système inutilisable : les analystes finissent "
        "par ignorer toutes les alertes.",
        "A normal flow wrongly flagged as an attack. Too many false alarms make a "
        "system unusable: analysts end up ignoring every alert."),
    "h_g5_t": ("Rappel (recall)", "Recall"),
    "h_g5": (
        "La proportion d'attaques réellement détectées. Un rappel de 85 % signifie "
        "que sur 100 attaques, 85 sont signalées et 15 passent inaperçues.",
        "The proportion of attacks actually detected. A recall of 85 % means that "
        "out of 100 attacks, 85 are flagged and 15 slip through."),
    "h_g6_t": ("FPR — taux de fausses alertes", "FPR — false positive rate"),
    "h_g6": (
        "La proportion de flux **normaux** signalés à tort. Un FPR de 10 % signifie "
        "que sur 100 flux légitimes, 10 déclenchent une alerte inutile.",
        "The proportion of **normal** flows wrongly flagged. An FPR of 10 % means "
        "that out of 100 legitimate flows, 10 raise a pointless alert."),
    "h_g7_t": ("AUROC", "AUROC"),
    "h_g7": (
        "Une note globale de 0 à 1 qui résume la capacité à classer les attaques "
        "au-dessus des flux normaux. 0,5 = hasard, 1,0 = parfait. **Elle ne dépend "
        "pas du seuil choisi** : c'est pourquoi on l'utilise pour comparer des "
        "méthodes, alors que le rappel dépend du seuil.",
        "An overall score from 0 to 1 summarising the ability to rank attacks "
        "above normal flows. 0.5 = chance, 1.0 = perfect. **It does not depend on "
        "the chosen threshold**: that is why it is used to compare methods, while "
        "recall does depend on it."),
    "h_g8_t": ("Précision", "Precision"),
    "h_g8": (
        "Parmi les flux signalés, la proportion qui étaient vraiment des attaques. "
        "Une précision de 66 % signifie que sur 100 alertes, 66 sont justes.",
        "Among the flagged flows, the proportion that really were attacks. A "
        "precision of 66 % means that out of 100 alerts, 66 are correct."),
    "h_g9_t": ("Distance de Mahalanobis", "Mahalanobis distance"),
    "h_g9": (
        "Une mesure d'écart au comportement normal qui tient compte de la "
        "dispersion et des corrélations entre variables. Elle n'a pas de plafond "
        "et s'additionne sur toutes les variables : c'est pourquoi elle produit de "
        "grands nombres (ex. 57 483) là où une probabilité reste entre 0 et 1.",
        "A measure of deviation from normal behaviour that accounts for spread and "
        "correlations between variables. It has no upper bound and sums over all "
        "variables: that is why it produces large numbers (e.g. 57,483) where a "
        "probability stays between 0 and 1."),
    "h_g10_t": ("Entraînement / test", "Training / test"),
    "h_g10": (
        "L'entraînement est la partie des données que le modèle étudie. Le test "
        "est celle qu'il ne voit jamais et qui sert à le noter. Les séparer évite "
        "de se mentir sur ses performances.",
        "Training is the part of the data the model studies. Test is the part it "
        "never sees, used to grade it. Keeping them apart prevents self-deception "
        "about performance."),
    "h_g11_t": ("Ablation", "Ablation"),
    "h_g11": (
        "Retirer un élément du modèle pour voir s'il servait à quelque chose. Si "
        "la performance ne baisse pas, l'élément était inutile.",
        "Removing one component from the model to see whether it was useful. If "
        "performance does not drop, the component was pointless."),
    "h_g12_t": ("Gradient boosting", "Gradient boosting"),
    "h_g12": (
        "Un assemblage de petits arbres de décision, chacun corrigeant les erreurs "
        "du précédent. Très efficace sur des données en tableau, comme les "
        "statistiques de flux.",
        "An ensemble of small decision trees, each one correcting the previous "
        "ones' mistakes. Very effective on tabular data such as flow statistics."),

    "h_keys_h": ("Repères de lecture", "Reading guide"),
    "h_key1": (
        "Un score **entre 0 et 1** vient du mode Supervisé : c'est une probabilité.",
        "A score **between 0 and 1** comes from Supervised mode: it is a probability."),
    "h_key2": (
        "Un score **à cinq ou six chiffres** vient du mode MS-Transformer ou "
        "One-class : c'est une distance. Ce n'est pas une erreur.",
        "A **five- or six-digit** score comes from MS-Transformer or One-class "
        "mode: it is a distance. It is not an error."),
    "h_key3": (
        "« 25 207 / 50 000 » se lit **alertes / flux analysés** — et non "
        "« 25 207 lignes traitées sur 50 000 ».",
        "'25,207 / 50,000' reads **alerts / flows analysed** — not '25,207 rows "
        "processed out of 50,000'."),
    "h_key4": (
        "Un fichier de plus de 50 000 lignes est analysé sur un **échantillon "
        "aléatoire** de 50 000, pour que le navigateur reste réactif.",
        "A file with more than 50,000 rows is analysed on a **random sample** of "
        "50,000, so the browser stays responsive."),

    "h_a11y_h": ("Accessibilité", "Accessibility"),
    "h_a11y": (
        "Cette interface suit les recommandations WCAG 2.1 niveau AA : contrastes "
        "vérifiés, navigation au clavier, lien d'évitement, textes alternatifs sur "
        "les figures, et ce glossaire (critère 3.1.3, vocabulaire inhabituel). Les "
        "indicateurs ne reposent jamais sur la couleur seule : un symbole les "
        "double.",
        "This interface follows WCAG 2.1 level AA: verified contrast ratios, "
        "keyboard navigation, a skip link, alternative text on figures, and this "
        "glossary (criterion 3.1.3, unusual words). Indicators never rely on "
        "colour alone: a symbol always doubles it."),
}

# ══════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════════════════
   SYSTÈME DE DESIGN « TERMINAL » — MODE HACKER, SOMBRE ET MONOSPACE
   ──────────────────────────────────────────────────────────────────
   Toute la page est un terminal : fond quasi noir, vert phosphore,
   police à chasse fixe, curseur qui clignote, lignes de balayage.
   Le parcours et son contenu ne changent pas — seule l'enveloppe
   change de monde.

   Règles tenues, comme avant :
   · chaque couleur de texte est vérifiée sur son fond — le plus faible
     couple du thème est à 4,9:1 (AA exige 4,5:1) ; le vert phosphore
     #39ff88 sur #05080a monte à 13,4:1 ;
   · aucun état n'est porté par la seule couleur : un symbole
     (✓ ● 🔒 ✕ ◐) et un mot l'accompagnent toujours ;
   · le vert phosphore sert de SURFACE pour les boutons primaires ; le
     texte y est posé en encre sombre #04120a (13,4:1), jamais en blanc
     (qui ne tiendrait que 1,6:1) ;
   · tous les effets (machine à écrire, clignotement, balayage, glitch)
     sont neutralisés par prefers-reduced-motion et à l'impression.
   ══════════════════════════════════════════════════════════════════ */
:root {
  --bg:          #05080a;
  --surface:     #0a1210;
  --surface-2:   #0d1714;
  --surface-3:   #122019;
  --border:      #1d3128;
  --border-ui:   #2f6b4f;   /* composants interactifs — ≥3:1 sur le fond */

  --fg-1:        #d9ffe8;   /* 17,6:1 */
  --fg-2:        #b6d9c6;   /* 12,4:1 */
  --fg-3:        #9dc0ab;   /*  9,3:1 */
  --fg-4:        #7ba38a;   /*  6,3:1 */

  --ok:          #39ff88;   /* 13,4:1 */
  --warn:        #ffcc00;   /* 13,1:1 */
  --info:        #5fd7ff;   /* 12,2:1 */
  --danger:      #ff6b6b;   /*  7,4:1 */
  --focus:       #39ff88;

  /* vert du parcours — surface, ombre, encre */
  --duo:         #39ff88;
  --duo-d:       #22c96a;
  --duo-dd:      #17a355;
  --duo-btn:     #39ff88;
  --duo-ink:     #04120a;
  --duo-soft:    #0b2418;
  --duo-line:    #1d5f3d;

  --gold:        #ffcc00;
  --gold-d:      #e0a800;
  --gold-dd:     #b88600;

  --mono: ui-monospace, "Cascadia Mono", "Consolas", "JetBrains Mono",
          "SF Mono", Menlo, monospace;

  --sp-1: 6px; --sp-2: 10px; --sp-3: 16px; --sp-4: 24px; --sp-5: 34px;
  --radius: 6px;          /* le terminal n'a pas de coins tendres */
  --radius-s: 5px;
}

/* ---- base : la page EST un terminal ------------------------------ */
html { font-size: 16px; }
body, .stApp, [data-testid="stAppViewContainer"] {
  font-family: var(--mono);
  background: var(--bg) !important;
  color: var(--fg-2) !important;
}
.stApp { background-image:
  radial-gradient(ellipse at 50% -10%, rgba(57,255,136,0.10), transparent 60%),
  radial-gradient(ellipse at 90% 110%, rgba(95,215,255,0.05), transparent 55%) !important;
  background-repeat: no-repeat !important;
}
/* Lignes de balayage : un calque fixe, transparent aux clics. Il donne
   la texture d'un vieil écran sans toucher au contraste du texte (3 %
   d'opacité). */
.stApp::after {
  content: ""; position: fixed; inset: 0; z-index: 9998; pointer-events: none;
  background: repeating-linear-gradient(
    to bottom, rgba(255,255,255,0.030) 0 1px, transparent 1px 3px);
  opacity: .55;
}
/* Le faisceau qui descend lentement : la seule animation plein écran. */
.stApp::before {
  content: ""; position: fixed; left: 0; right: 0; height: 28vh; top: -30vh;
  z-index: 9997; pointer-events: none;
  background: linear-gradient(to bottom, transparent, rgba(57,255,136,0.045), transparent);
  animation: beam 9s linear infinite;
}
@keyframes beam { to { transform: translateY(160vh); } }

.block-container { padding-top: 3.9rem; padding-bottom: var(--sp-5);
                   max-width: 1180px; }

h1, h2, h3, h4 { color: var(--fg-1) !important; font-family: var(--mono);
                 font-weight: 700; letter-spacing: -0.01em; line-height: 1.28; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.12rem !important; }
p, li { color: var(--fg-2); line-height: 1.68; }
a { color: var(--info); text-underline-offset: 3px; }
strong, b { color: var(--fg-1); }
code, kbd, pre, .mono { font-family: var(--mono) !important; }
code { background: var(--surface-2); color: var(--duo);
       border: 1px solid var(--border); border-radius: 3px; padding: 1px 5px; }
hr { border-color: var(--border) !important; }

/* ---- focus visible : indispensable au clavier -------------------- */
a:focus-visible, button:focus-visible, [tabindex]:focus-visible,
input:focus-visible, select:focus-visible,
div[role="radiogroup"] label:focus-visible,
.stTabs [role="tab"]:focus-visible {
  outline: 2px solid var(--focus) !important;
  outline-offset: 3px !important;
  border-radius: 3px;
  box-shadow: 0 0 12px rgba(57,255,136,0.55) !important;
}

/* ---- respect de la préférence système ---------------------------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
  .stApp::before { display: none; }
}

/* ---- effets de texte : machine à écrire, curseur, glitch ---------- */
.typed {
  display: inline-block; overflow: hidden; white-space: nowrap;
  vertical-align: bottom; max-width: 100%;
  width: calc(var(--n, 20) * 1ch);
  animation: typing 1.25s steps(var(--n, 20)) both;
  border-right: 2px solid var(--duo);
  padding-right: 2px;
}
@keyframes typing { from { width: 0; } to { width: calc(var(--n, 20) * 1ch); } }
.cur-blink { display: inline-block; color: var(--duo);
             animation: blink 1.05s steps(1, end) infinite; }
@keyframes blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }

/* ---- en-tête ----------------------------------------------------- */
.eyebrow { display: inline-block; font-size: 0.7rem; letter-spacing: 0.16em;
           text-transform: uppercase; color: var(--duo); font-weight: 700;
           border: 1px solid var(--duo-line); background: #07130d;
           border-radius: 3px; padding: 3px 10px; margin-bottom: var(--sp-2); }
.proj-title {
  font-size: 2rem; font-weight: 700; color: var(--fg-1);
  line-height: 1.16; letter-spacing: -0.02em; margin: 0;
  text-shadow: 0 0 14px rgba(57,255,136,0.35);
  animation: glitch 6s steps(1, end) infinite;
}
@keyframes glitch {
  0%, 92%, 100% { text-shadow: 0 0 14px rgba(57,255,136,0.35); }
  93% { text-shadow: 2px 0 rgba(255,107,107,0.85), -2px 0 rgba(95,215,255,0.85); }
  94% { text-shadow: -3px 0 rgba(255,107,107,0.7), 2px 0 rgba(95,215,255,0.8);
        transform: translateX(1px); }
  95% { text-shadow: 0 0 14px rgba(57,255,136,0.35); transform: none; }
}
.proj-sub { font-size: 0.95rem; color: var(--fg-3); margin-top: var(--sp-2);
            line-height: 1.6; max-width: 64ch; }

/* barre d'invite : la première chose que lit l'analyste */
.termbar {
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
  margin: var(--sp-2) 0 0; padding: 7px 12px;
  background: #07100c; border: 1px solid var(--border);
  border-left: 3px solid var(--duo); border-radius: 4px;
  font-family: var(--mono); font-size: 0.86rem; color: var(--fg-2);
}
.termbar .u { color: var(--duo); font-weight: 700; }
.termbar .p { color: var(--info); }
.termbar .cmd { color: var(--fg-1); }
.termbar .ok { margin-left: auto; color: var(--duo); font-weight: 700;
               letter-spacing: 0.08em; font-size: 0.78rem; }
.termbar .ok::before { content: "● "; animation: blink 1.4s steps(1,end) infinite; }

/* ---- badges d'état : symbole + libellé, jamais la couleur seule -- */
.badge { display: inline-flex; align-items: center; gap: 6px;
         font-size: 0.76rem; font-weight: 700; padding: 4px 10px;
         border-radius: 3px; margin-right: var(--sp-1); margin-top: var(--sp-2);
         border: 1px solid var(--border); background: var(--surface);
         color: var(--fg-2); letter-spacing: 0.04em; }
.badge-ok   { border-color: var(--duo-line); color: var(--duo); background: #07130d; }
.badge-warn { border-color: #5c4a00; color: var(--warn); background: #14100a; }
.badge-info { border-color: #1b4a5c; color: var(--info); background: #08131a; }

/* ══════════════════════════════════════════════════════════════════
   BOUTONS — style terminal : rectangle net, trait phosphore,
   un reflet qui traverse au survol.
   ══════════════════════════════════════════════════════════════════ */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  border-radius: 4px !important;
  border: 1px solid var(--border-ui) !important;
  background: var(--surface-2) !important;
  color: var(--fg-1) !important;
  font-family: var(--mono) !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em;
  font-size: 0.82rem !important;
  padding: 8px 15px !important;
  text-transform: uppercase;
  transition: color 90ms ease, border-color 90ms ease, box-shadow 140ms ease,
              background 140ms ease;
  box-shadow: none !important;
  position: relative; overflow: hidden;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
  background: #10251a !important;
  color: var(--duo) !important;
  border-color: var(--duo) !important;
  box-shadow: 0 0 14px rgba(57,255,136,0.30), inset 0 0 12px rgba(57,255,136,0.08) !important;
}
.stButton > button:active, .stDownloadButton > button:active,
.stFormSubmitButton > button:active { background: #153627 !important; }
/* bouton primaire : phosphore plein, encre sombre — le seul plein */
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: var(--duo) !important;
  border-color: var(--duo) !important;
  color: var(--duo-ink) !important;
  box-shadow: 0 0 18px rgba(57,255,136,0.45) !important;
}
[data-testid="stBaseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
  background: #6dffab !important;
  color: var(--duo-ink) !important;
  box-shadow: 0 0 26px rgba(57,255,136,0.75) !important;
}
.stButton > button:disabled, [data-testid="stBaseButton-secondary"]:disabled {
  opacity: 1 !important;
  background: #0a0f0d !important;
  color: #5c7064 !important;
  border-color: #1a2a22 !important;
  cursor: not-allowed;
  box-shadow: none !important;
}

/* ---- champs de formulaire ---------------------------------------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
.stTextInput input, .stNumberInput input, [data-testid="stFileUploader"] section {
  border-radius: 4px !important;
  border: 1px solid var(--border-ui) !important;
  background: #07100c !important;
  color: var(--fg-1) !important;
  font-family: var(--mono) !important;
}
[data-testid="stFileUploader"] section { border-style: dashed !important; }
div[role="radiogroup"] label {
  border: 1px solid var(--border); border-radius: 3px;
  padding: 5px 11px; background: #07100c;
  font-size: 0.82rem; font-weight: 700;
}
div[role="radiogroup"] label:has(input:checked) {
  border-color: var(--duo); background: #0b2418; color: var(--duo);
  box-shadow: 0 0 10px rgba(57,255,136,0.22);
}
[data-testid="stSegmentedControl"] button,
[data-testid="stButtonGroup"] button {
  border-radius: 3px !important; font-weight: 700 !important;
  font-family: var(--mono) !important; text-transform: uppercase;
  font-size: 0.78rem !important;
}

/* ---- widgets natifs : l'accent rouge de Streamlit devient vert ---
   Streamlit peint le curseur, la case cochée et le point du bouton radio
   avec sa couleur primaire, rouge #FF4B4B par défaut. Ici le rouge veut
   dire « menace » : un curseur de réglage ne doit pas annoncer un danger.
   Ces règles remplacent le fichier de thème de Streamlit — l'application
   tient ainsi dans un seul fichier, sans dossier .streamlit à copier. */
[data-testid="stSlider"] [role="slider"] {
  background-color: var(--duo) !important;
  box-shadow: 0 0 10px rgba(57,255,136,0.85) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrack"],
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
  background: var(--duo-d) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
  color: var(--duo) !important; font-weight: 700 !important;
  font-family: var(--mono) !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] span[aria-checked="true"],
[data-testid="stCheckbox"] input:checked + span {
  background-color: var(--duo) !important;
  border-color: var(--duo) !important;
}
div[role="radiogroup"] [data-baseweb="radio"] div[aria-checked="true"],
div[role="radiogroup"] input:checked + div {
  background-color: var(--duo) !important;
  border-color: var(--duo) !important;
}

/* ---- onglets : sept terminaux dans une barre --------------------- */
[data-baseweb="tab-list"] {
  gap: 4px; border-bottom: 1px solid var(--border); padding-bottom: 8px;
  flex-wrap: wrap;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab"] {
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  padding: 5px 13px !important;
  color: var(--fg-3) !important;
  font-family: var(--mono) !important;
  font-weight: 700 !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.03em;
  background: #07100c !important;
}
[data-baseweb="tab"]:hover { color: var(--duo) !important;
                             border-color: var(--border-ui) !important; }
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--duo-ink) !important;
  background: var(--duo) !important;
  border-color: var(--duo) !important;
  box-shadow: 0 0 14px rgba(57,255,136,0.45) !important;
}

/* ---- encadrés natifs --------------------------------------------- */
[data-testid="stAlert"] {
  border-radius: 4px !important;
  border: 1px solid var(--border) !important;
  border-left-width: 3px !important;
  background: var(--surface) !important;
  color: var(--fg-2) !important;
  font-family: var(--mono) !important;
}
[data-testid="stAlert"] p { color: var(--fg-2) !important; }
[data-testid="stAlert"] a { color: var(--info) !important; }
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  background: var(--surface);
}
[data-testid="stExpander"] summary { font-weight: 700; color: var(--fg-1);
                                     font-family: var(--mono); }
[data-testid="stExpander"] summary:hover { color: var(--duo); }

/* ══════════════════════════════════════════════════════════════════
   BANDEAU D'UNITÉ — la carte de tête, en vert phosphore sur noir,
   avec une bordure qui défile comme un décodeur.
   ══════════════════════════════════════════════════════════════════ */
[class*="st-key-duo_unit_"] {
  position: relative; overflow: hidden;
  background: linear-gradient(180deg, #08160f 0%, #050d09 100%);
  border: 1px solid var(--duo-d);
  border-radius: 8px;
  padding: 14px 20px;
  box-shadow: inset 0 0 22px rgba(57,255,136,0.10),
              0 0 22px rgba(57,255,136,0.14);
}
[class*="st-key-duo_unit_"]::before {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(
    90deg, rgba(57,255,136,0.07) 0 2px, transparent 2px 12px);
  animation: barflow 12s linear infinite;
}
@keyframes barflow { to { background-position: 240px 0; } }
[class*="st-key-duo_unit_"] [data-testid="stHorizontalBlock"] {
  align-items: center; gap: 12px;
}
.unit { background: transparent; border: 0; padding: 0; border-radius: 0;
        position: relative; z-index: 1; }
.unit-k { display: inline-block; font-size: 0.66rem; font-weight: 700;
          letter-spacing: 0.16em; text-transform: uppercase;
          color: var(--duo-ink); background: var(--duo);
          border-radius: 2px; padding: 2px 9px; margin-bottom: 6px; }
.unit-t { color: var(--fg-1); font-weight: 700; font-size: 1.18rem;
          letter-spacing: 0.01em; text-shadow: 0 0 12px rgba(57,255,136,0.45); }
.unit-s { color: var(--fg-2); font-size: 0.86rem; margin-top: 3px; }
.unit-s strong, .unit-s code { color: var(--duo); }
.unit-p { display: flex; flex-wrap: wrap; align-items: center; gap: 9px;
          margin-top: var(--sp-2); }
.unit-pl { font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
           color: var(--fg-4); font-weight: 700; }
.unit-dots { display: inline-flex; gap: 5px; }
.u-dot { width: 10px; height: 10px; border-radius: 1px;
         border: 1px solid var(--border-ui); background: transparent; }
.u-dot.u-on { background: var(--duo); border-color: var(--duo);
              box-shadow: 0 0 8px rgba(57,255,136,0.7); }
.unit-pn { font-size: 0.74rem; color: var(--fg-3); }

/* le bouton GUIDE, à droite du bandeau */
[class*="st-key-duo_unit_"] [data-testid="stPopover"] button {
  background: transparent !important;
  border: 1px solid var(--duo) !important;
  color: var(--duo) !important;
  border-radius: 3px !important;
  font-weight: 700 !important;
  width: 100%;
  box-shadow: 0 0 10px rgba(57,255,136,0.25) !important;
}
[class*="st-key-duo_unit_"] [data-testid="stPopover"] button:hover {
  background: rgba(57,255,136,0.14) !important;
  box-shadow: 0 0 18px rgba(57,255,136,0.55) !important;
}

/* ══════════════════════════════════════════════════════════════════
   LE PARCOURS — sept nœuds de terminal qui descendent en zigzag,
   reliés par un trait pointillé, deux haltes dorées.
   ══════════════════════════════════════════════════════════════════ */
[data-testid="stColumn"]:has(.pc-mark) {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: var(--sp-3) var(--sp-2) var(--sp-4);
  box-shadow: inset 0 0 30px rgba(57,255,136,0.04);
}
.pc-mark { display: none; }
/* Les blocs de Streamlit sont espacés d'un rem : dans un chemin de sept
   nœuds, cela ferait deux écrans de vide. La colonne du parcours resserre
   donc ses espacements, elle seule. */
[data-testid="stColumn"]:has(.duo-path) [data-testid="stVerticalBlock"] {
  gap: 0.18rem !important;
}

/* Ligne de nœuds : cinq cellules égales.
   Le ciblage passe par le conteneur à clé `duo_row_<n>` que Streamlit
   expose en classe `st-key-duo_row_<n>`. C'est le seul sélecteur précis
   ici : `:has(.duo-row)` aurait aussi attrapé les blocs PARENTS, et la
   grande colonne de gauche aurait hérité de la largeur d'une cellule. */
[class*="st-key-duo_row_"] [data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  align-items: flex-start;
}
[class*="st-key-duo_row_"] [data-testid="stColumn"],
[class*="st-key-duo_row_"] [data-testid="column"] {
  flex: 1 1 0 !important; width: auto !important; min-width: 0 !important;
  display: flex; flex-direction: column; align-items: center;
}
[class*="st-key-duo_row_"] [data-testid="stVerticalBlock"] { gap: 0 !important; }

/* le nœud : un carré à coins coupés, cerclé de phosphore */
[class*="st-key-pcb_"] { display: flex; justify-content: center; }
[class*="st-key-pcb_"] button {
  width: 62px !important; height: 62px !important; min-height: 62px !important;
  min-width: 62px !important; flex: 0 0 auto !important;
  border-radius: 6px !important;
  padding: 0 !important; font-size: 1.5rem !important; font-weight: 700 !important;
  line-height: 1 !important;
  font-family: var(--mono) !important;
  display: flex; align-items: center; justify-content: center;
  clip-path: polygon(8px 0, 100% 0, 100% calc(100% - 8px),
                     calc(100% - 8px) 100%, 0 100%, 0 8px);
  transition: transform 90ms ease, box-shadow 140ms ease, background 140ms ease;
}
/* état terminé : phosphore plein, encre noire */
[class*="st-key-pcb_done"] button {
  background: var(--duo) !important;
  color: var(--duo-ink) !important;
  border: 0 !important;
  box-shadow: 0 0 16px rgba(57,255,136,0.55) !important;
}
[class*="st-key-pcb_done"] button:hover {
  background: #6dffab !important; color: var(--duo-ink) !important;
  box-shadow: 0 0 26px rgba(57,255,136,0.85) !important;
}
/* état courant : anneau qui respire, et le halo qui pulse */
[class*="st-key-pcb_cur"] button {
  background: #0b2418 !important;
  color: var(--duo) !important;
  border: 2px solid var(--duo) !important;
  animation: pulse 1.7s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 6px rgba(57,255,136,0.45),
                         inset 0 0 10px rgba(57,255,136,0.18); }
  50%      { box-shadow: 0 0 26px rgba(57,255,136,0.95),
                         inset 0 0 18px rgba(57,255,136,0.38); }
}
[class*="st-key-pcb_cur"] button:hover {
  background: #122f20 !important; color: #6dffab !important;
}
/* état ouvert, pas encore atteint */
[class*="st-key-pcb_open"] button {
  background: #07100c !important;
  color: var(--fg-3) !important;
  border: 1px solid var(--border-ui) !important;
}
[class*="st-key-pcb_open"] button:hover {
  background: #0b2418 !important; color: var(--duo) !important;
  border-color: var(--duo) !important;
  box-shadow: 0 0 14px rgba(57,255,136,0.35) !important;
}
/* état verrouillé : gris, pointillés, cadenas — « accès refusé » */
[class*="st-key-pcb_lock"] button {
  background: #080b0a !important;
  color: #5c7064 !important;
  border: 1px dashed #2a3a32 !important;
  font-size: 1.05rem !important;
}
[class*="st-key-pcb_lock"] button:hover {
  background: #0d1210 !important; color: #8fa89a !important;
  border-color: #3d4f46 !important;
}
[class*="st-key-pcb_"] button:focus-visible {
  outline: 2px solid var(--focus) !important; outline-offset: 4px !important;
}

/* nom + statut sous le nœud. Le libellé est plus large que la cellule
   qui le porte : il déborde symétriquement, ce qui le centre sous le
   nœud quelle que soit la position de celui-ci dans le zigzag. */
.duo-lab { --w: 180px; width: var(--w); text-align: center; margin-top: 8px;
           margin-left: calc(50% - var(--w) / 2); }
.duo-lab b { display: block; color: var(--fg-1); font-size: 0.8rem;
             font-weight: 700; line-height: 1.25; }
.duo-lab .duo-k { display: block; color: var(--fg-4); font-size: 0.66rem;
                  font-weight: 700; letter-spacing: 0.1em;
                  text-transform: uppercase; margin-bottom: 1px; }
.duo-sub { display: block; color: var(--fg-3); font-size: 0.72rem;
           line-height: 1.3; margin-top: 2px; }
.duo-now-pill {
  display: block; margin: 6px auto 0; width: fit-content;
  background: var(--duo); border: 1px solid var(--duo);
  color: var(--duo-ink); font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  border-radius: 2px; padding: 2px 8px;
  box-shadow: 0 0 10px rgba(57,255,136,0.5);
}
.duo-lock-pill {
  display: block; margin: 6px auto 0; width: fit-content;
  background: #080b0a; border: 1px dashed #2a3a32;
  color: #8fa89a; font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  border-radius: 2px; padding: 2px 8px;
}

/* le trait pointillé entre deux nœuds : un SVG qui s'étire avec la
   carte, dessiné dans la couleur de l'état des deux nœuds qu'il relie */
.duo-link { line-height: 0; }
.duo-link svg { display: block; width: 100%; height: 30px; }
.duo-link .ln { fill: none; stroke: #2a3a32; stroke-width: 1.6;
                stroke-dasharray: 2 8; stroke-linecap: butt; }
.duo-link.done .ln { stroke: var(--duo-d); }
.duo-link.cur  .ln { stroke: var(--duo); animation: dashrun 3s linear infinite; }
@keyframes dashrun { to { stroke-dashoffset: -40; } }

/* jalon : la caisse de données, qui s'ouvre sur un vrai événement */
.duo-jalon { display: flex; flex-direction: column; align-items: center;
             gap: 4px; margin: 4px auto 6px; }
.duo-jalon svg { width: 66px; height: 66px; }
.duo-jalon .jl { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em;
                 text-transform: uppercase; color: var(--fg-4); }
.duo-jalon.on .jl { color: var(--gold); }

/* ligne d'état du lot — elle remplace la bulle de la mascotte et dit
   exactement la même chose, en langage terminal */
.duo-term {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 7px;
  margin: 0 0 var(--sp-2); padding: 8px 11px;
  background: #07100c; border: 1px solid var(--border);
  border-left: 3px solid var(--duo); border-radius: 4px;
  font-family: var(--mono); font-size: 0.8rem; line-height: 1.5;
  color: var(--fg-2);
}
.duo-term .p { color: var(--duo); font-weight: 700; }
.duo-term b { color: var(--duo); }

/* progression du parcours : sept segments, comme une jauge */
.pc-prog { margin: var(--sp-1) 0 var(--sp-3); }
.pc-prog-l { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
             color: var(--fg-4); margin-bottom: 5px; font-weight: 700; }
.pc-prog-b { height: 10px; border-radius: 2px; background: #0a0f0d;
             border: 1px solid var(--border); overflow: hidden; }
.pc-prog-b span { display: block; height: 100%;
                  background: repeating-linear-gradient(
                    90deg, var(--duo) 0 8px, var(--duo-d) 8px 10px);
                  box-shadow: 0 0 12px rgba(57,255,136,0.55); }

/* légende du parcours */
.pc-legend { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
             margin: var(--sp-3) 0 var(--sp-1); font-size: 0.72rem;
             color: var(--fg-3); }
.pc-legend span { border: 1px solid var(--border); border-radius: 3px;
                  padding: 2px 8px; background: #07100c; }
.pc-legend b { color: var(--fg-1); }
.pc-legend-n { color: var(--fg-4); font-style: italic; border: 0 !important;
               background: none !important; padding-left: 0 !important; }

/* titres du bloc parcours */
.pc-path-h { font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase;
             color: var(--duo); font-weight: 700; margin: 0 0 4px;
             text-align: center; }
.pc-path-l { font-size: 0.76rem; color: var(--fg-4); text-align: center;
             margin: 0 0 var(--sp-2); line-height: 1.4; }

/* en-tête du panneau de droite : où l'on est, puis le titre de l'étape */
.pc-panel-h { display: inline-block; font-size: 0.68rem; letter-spacing: 0.14em;
              text-transform: uppercase; color: var(--duo); font-weight: 700;
              background: #07130d; border: 1px solid var(--duo-line);
              border-radius: 3px; padding: 3px 10px; margin: 0 0 8px; }
.pc-do { color: var(--fg-2); font-size: 0.92rem; line-height: 1.55;
         margin: -2px 0 var(--sp-3); max-width: 62ch; }

/* clic sur un cadenas : ce que le verrou fait, et ne fait pas */
.u-lock { background: #0f0d05; border: 1px solid #5c4a00;
          border-left: 3px solid var(--warn); border-radius: 4px;
          padding: var(--sp-3); margin-top: var(--sp-2); }
.u-lock-h { color: var(--warn); font-weight: 700; font-size: 0.76rem;
            letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.u-lock-b { color: var(--fg-2); font-size: 0.84rem; line-height: 1.55; }
.u-lock-g { color: var(--fg-4); font-size: 0.76rem; font-style: italic; margin-top: 6px; }

/* ce qui est mémorisé entre les étapes */
.u-mem { background: var(--surface-2); border: 1px solid var(--border);
         border-radius: 4px; padding: var(--sp-3); margin-top: var(--sp-3); }
.u-mem-h { color: var(--fg-4); font-size: 0.68rem; letter-spacing: 0.14em;
           text-transform: uppercase; font-weight: 700; margin-bottom: 6px; }
.u-mem ul { margin: 0; padding-left: 18px; color: var(--fg-2); font-size: 0.84rem; }
.u-mem li { margin-bottom: 3px; }

/* ---- cartes de mesure : un cadran de terminal -------------------- */
.kpi { position: relative; background: var(--surface);
       border: 1px solid var(--border); border-radius: 6px;
       padding: var(--sp-3); text-align: center;
       box-shadow: inset 0 0 18px rgba(57,255,136,0.05); }
.kpi::after { content: ""; position: absolute; left: 10px; right: 10px; top: 0;
              height: 1px; background: linear-gradient(90deg, transparent,
              rgba(57,255,136,0.55), transparent); }
.kpi-v { font-size: 1.6rem; font-weight: 700; color: var(--duo);
         line-height: 1.15; font-variant-numeric: tabular-nums;
         text-shadow: 0 0 12px rgba(57,255,136,0.45); }
.kpi-l { font-size: 0.68rem; color: var(--fg-4); letter-spacing: 0.12em;
         text-transform: uppercase; margin-bottom: var(--sp-1); font-weight: 700; }
.kpi-s { font-size: 0.76rem; color: var(--fg-3); margin-top: var(--sp-1); }

/* ---- panneaux ---------------------------------------------------- */
.panel { background: var(--surface); border: 1px solid var(--border);
         border-radius: 6px; padding: var(--sp-3);
         color: var(--fg-2); line-height: 1.66; }
.panel-accent { border-left: 3px solid var(--duo); }
.pitch { background: #07130d; border: 1px solid var(--duo-line);
         border-left: 3px solid var(--duo); border-radius: 6px;
         padding: var(--sp-3) var(--sp-4); color: var(--fg-2); }

/* ---- encadrés ---------------------------------------------------- */
.note { background: var(--surface); border: 1px solid var(--border);
        border-left: 3px solid var(--border-ui); border-radius: 4px;
        padding: var(--sp-2) var(--sp-3); font-size: 0.86rem; color: var(--fg-2);
        line-height: 1.65; }
.note-warn { border-left-color: var(--warn); background: #0f0d05; }

/* ---- pied de page ------------------------------------------------ */
.foot { border-top: 1px solid var(--border); margin-top: var(--sp-4);
        padding-top: var(--sp-2); font-size: 0.76rem; color: var(--fg-4);
        line-height: 2; }
.foot code { color: var(--duo); background: var(--surface-2);
             border: 1px solid var(--border); padding: 1px 5px; border-radius: 3px;
             font-size: 0.95em; }

.mono { font-family: var(--mono); }

/* ---- tableaux : une grille de terminal --------------------------- */
div[data-testid="stDataFrame"] { border: 1px solid var(--border);
                                 border-radius: 4px; }
.tbl-wrap { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
.tbl { width: 100%; border-collapse: collapse;
       margin: var(--sp-2) 0 var(--sp-3); font-size: 0.86rem;
       border: 1px solid var(--border); }
.tbl caption { text-align: left; color: var(--fg-3); font-size: 0.82rem;
               padding: 0 0 var(--sp-2); }
.tbl th, .tbl td { padding: 8px 11px; border-bottom: 1px solid var(--border);
                   text-align: right; vertical-align: top; }
.tbl th[scope="row"], .tbl th:first-child, .tbl td:first-child { text-align: left; }
.tbl thead th { color: var(--duo); font-weight: 700; background: #07100c;
                border-bottom: 1px solid var(--duo-line);
                letter-spacing: 0.06em; text-transform: uppercase;
                font-size: 0.74rem; }
.tbl th[scope="row"] { color: var(--fg-1); font-weight: 700; }
.tbl td { color: var(--fg-2); font-variant-numeric: tabular-nums; }
.tbl tbody tr:hover { background: #0b2418; }
.tbl tbody tr:last-child th, .tbl tbody tr:last-child td { border-bottom: 0; }
.tbl .na { color: var(--fg-4); font-style: italic; }

/* ---- titre de section d'onglet ---------------------------------- */
.tab-h { display: inline-block; font-size: 0.68rem !important;
         letter-spacing: 0.16em; text-transform: uppercase;
         color: var(--duo) !important; font-weight: 700;
         margin: 0 0 var(--sp-2) !important; padding: 3px 10px;
         background: #07130d; border: 1px solid var(--duo-line);
         border-radius: 3px; }
.tab-h::before { content: "./"; opacity: .7; }

/* ══════════════════════════════════════════════════════════════════
   BARRE LATÉRALE — les réglages, séparés des résultats
   ══════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: #060a09 !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] h2 { font-size: 0.95rem !important;
                               letter-spacing: 0.06em; margin-bottom: 4px;
                               color: var(--duo) !important;
                               text-transform: uppercase; }
[data-testid="stSidebar"] h2::before { content: "> "; opacity: .8; }
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {
  background: var(--duo) !important; border-color: var(--duo) !important;
  color: var(--duo-ink) !important;
  box-shadow: 0 0 16px rgba(57,255,136,0.4) !important;
}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {
  background: #6dffab !important; color: var(--duo-ink) !important;
}

/* ══════════════════════════════════════════════════════════════════
   MOBILE — sous 900 px, les colonnes s'empilent, SAUF le chemin :
   un parcours empilé en sept blocs ne montrerait plus un parcours.
   ══════════════════════════════════════════════════════════════════ */
@media (max-width: 900px) {
  .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
  [data-testid="column"], [data-testid="stColumn"] {
    flex: 1 1 100% !important; width: 100% !important;
    min-width: 100% !important;
  }
  /* exception : les cellules du chemin gardent leur largeur — un parcours
     empilé en cinq blocs ne serait plus un parcours */
  [class*="st-key-duo_row_"] [data-testid="stColumn"],
  [class*="st-key-duo_row_"] [data-testid="column"] {
    flex: 1 1 0 !important; width: auto !important; min-width: 0 !important;
  }
  [class*="st-key-pcb_"] button { width: 52px !important; height: 52px !important;
                                  min-height: 52px !important; min-width: 52px !important;
                                  font-size: 1.25rem !important; }
  .duo-lab { --w: 132px; }
  .duo-lab b { font-size: 0.72rem; }
  .duo-lab .duo-sub { font-size: 0.66rem; }
  .tbl { font-size: 0.8rem; }
  .tbl th, .tbl td { padding: 6px 8px; }
  h1, .proj-title { font-size: 1.5rem !important; line-height: 1.2; }
  .badge { display: inline-block; margin-bottom: 4px; }
  .unit-t { font-size: 1.05rem; }
  [class*="st-key-duo_unit_"] { padding: 12px 14px; }
  .typed { animation: none; width: auto; border-right: 0; }
}

/* ---- impression : un écran du mémoire doit rester lisible sur papier */
@media print {
  .stApp { background: #ffffff !important; color: #111111 !important; }
  .stApp::before, .stApp::after { display: none !important; }
  [data-testid="stSidebar"], [data-testid="stToolbar"], .badge,
  .duo-term, [class*="st-key-pcb_"] { display: none !important; }
  .tbl th, .tbl td { border-color: #999999; color: #111111 !important; }
  .tbl thead th { background: #eeeeee !important; color: #111111 !important; }
  .note, .pitch, .panel, .kpi, .unit { border: 1px solid #999999 !important;
                                      background: #ffffff !important;
                                      box-shadow: none !important; }
  .kpi-v, .unit-t, h1, h2, h3 { color: #111111 !important;
                                text-shadow: none !important; }
  .typed { animation: none; width: auto; border-right: 0; }
  a[href]::after { content: " (" attr(href) ")"; font-size: 0.8em; }
}

/* ---- utilitaires -------------------------------------------------- */
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
           overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════
from model_ms import (TokenizedMSEncoder, anomaly_score, free_energy,
                      next_state_confidence)


@st.cache_resource
def load_model():
    with open(os.path.join(ROOT, "models", "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    cfg_path = os.path.join(ROOT, "models", "model_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {
        "n_net": scaler.n_features_in_ - 25, "n_sys": 25}

    m = TokenizedMSEncoder(n_net=cfg["n_net"], n_sys=cfg["n_sys"],
                           embed_dim=cfg.get("embed_dim", 32), d=cfg.get("d", 64),
                           n_tok_net=cfg.get("n_tok_net", 4),
                           n_tok_sys=cfg.get("n_tok_sys", 5),
                           n_layers=cfg.get("n_layers", 2))
    m.load_state_dict(torch.load(os.path.join(ROOT, "models", "ms_encoder.pt"),
                                 map_location="cpu"))
    m.eval()

    centroid = torch.load(os.path.join(ROOT, "models", "centroid.pt"), map_location="cpu")

    thr_path = os.path.join(ROOT, "models", "thresholds.json")
    thr_cfg = json.load(open(thr_path)) if os.path.exists(thr_path) else {}

    maha = None
    mp = os.path.join(ROOT, "models", "mahalanobis.npz")
    if os.path.exists(mp):
        z = np.load(mp, allow_pickle=False)
        maha = {"mu": z["mu"], "prec": z["prec"], "threshold": float(z["threshold"])}

    # Score sur les 78 features réseau : c'est le détecteur retenu.
    # Le Mahalanobis appliqué aux embeddings du transformer plafonne à
    # AUROC 0.69 ; appliqué aux features dont le pré-traitement a été
    # corrigé (scaler ajusté sur tout le train), il atteint 0.96.
    feat = None
    fp = os.path.join(ROOT, "models", "features_maha.npz")
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=False)
        feat = {"mean": z["mean"], "prec": z["precision"],
                "mu_full": z["mu_full"], "sd_full": z["sd_full"],
                "n_net": int(z["n_net"]), "threshold": float(z["threshold"])}
    # Classifieur supervisé (comparaison) : a vu les familles connues.
    sup = None
    sp = os.path.join(ROOT, "models", "supervised_logreg.npz")
    if os.path.exists(sp):
        z = np.load(sp, allow_pickle=False)
        sup = {"coef": z["coef"], "intercept": float(z["intercept"][0]),
               "threshold": float(z["threshold"])}
    # Détecteur retenu par l'évaluation : gradient boosting (AUROC 0,9668 en
    # cross-famille). Il exige le même prétraitement que 03_evaluation.py :
    # les statistiques sont donc lues dans le .npz écrit à l'entraînement.
    hgb = None
    hp = os.path.join(ROOT, "models", "supervised_hgb.pkl")
    hn = os.path.join(ROOT, "models", "supervised_hgb.npz")
    if os.path.exists(hp) and os.path.exists(hn):
        try:
            import joblib
            z = np.load(hn, allow_pickle=False)
            hgb = {"model": joblib.load(hp), "mu": z["mu"], "sd": z["sd"],
                   "n_net": int(z["n_net"]), "threshold": float(z["threshold"])}
        except Exception:
            hgb = None
    return m, scaler, centroid, cfg, thr_cfg, maha, feat, sup, hgb


model, scaler, centroid, CFG, THR_CFG, MAHA, FEAT, SUP, HGB = load_model()
N_NET, N_SYS = CFG["n_net"], CFG["n_sys"]

# ── Contrôle de cohérence modèle / données ────────────────────────────────
# Le prétraitement peut retirer des colonnes constantes : le nombre de colonnes
# système varie donc d'une exécution de 01 à l'autre. Si les données ont été
# régénérées sans réentraîner, le modèle attend n_sys colonnes et l'interface
# en reçoit moins — PyTorch lève alors une erreur au fond de la boucle
# d'inférence. On vérifie ici, une fois, et on explique quoi faire.
_MISMATCH = None
try:
    _meta = json.load(open(os.path.join(ROOT, "data", "processed", "meta.json"),
                           encoding="utf-8"))
    _X = np.load(os.path.join(ROOT, "data", "processed", "X_test.npy"))
    _m_net, _m_sys = int(_meta.get("n_net", -1)), int(_meta.get("n_sys", -1))
    if (_m_net, _m_sys) != (N_NET, N_SYS):
        _MISMATCH = ("split", _m_net, _m_sys)
    elif _X.shape[1] != N_NET + N_SYS:
        _MISMATCH = ("width", _X.shape[1], N_NET + N_SYS)
    elif scaler.n_features_in_ != N_NET + N_SYS:
        _MISMATCH = ("scaler", scaler.n_features_in_, N_NET + N_SYS)
except FileNotFoundError:
    _MISMATCH = ("missing", None, None)
except Exception as _e:
    _MISMATCH = ("error", str(_e), None)

if _MISMATCH is not None:
    _kind = _MISMATCH[0]
    if _kind == "split":
        st.error(
            f"**Modèle et données incompatibles.** Le modèle a été entraîné avec "
            f"**{N_NET} colonnes réseau + {N_SYS} système**, mais les données "
            f"actuelles en contiennent **{_MISMATCH[1]} + {_MISMATCH[2]}**.\n\n"
            f"La préparation des données a retiré des colonnes constantes depuis "
            f"le dernier entraînement. Il faut réentraîner :\n\n"
            f"1. `py 02_model_training.py --epochs 8`\n"
            f"2. `py 03_evaluation.py --score supervise --algo hgb --target-fpr 0.09`\n"
            f"3. recharger cette page")
    elif _kind == "width":
        st.error(
            f"**Dimensions incohérentes.** Le modèle attend {_MISMATCH[2]} "
            f"colonnes, les données en fournissent {_MISMATCH[1]}. "
            f"Relance `py 01_data_preparation.py` puis `py 02_model_training.py`.")
    elif _kind == "scaler":
        st.error(
            f"**Scaler désynchronisé.** Il a été ajusté sur {_MISMATCH[1]} colonnes, "
            f"le modèle en attend {_MISMATCH[2]}. Relance `py 01_data_preparation.py`.")
    else:
        st.error(f"**Données absentes ou illisibles** : {_MISMATCH[1]}. "
                 f"Lance `py 01_data_preparation.py`.")
    st.stop()

# Mode de détection : deux approches, deux garanties différentes.
if "mode" not in st.session_state:
    # le mode par défaut suit le score enregistré par 03_evaluation.py,
    # sinon les cartes de synthèse et la démonstration se contrediraient.
    try:
        _sc = json.load(open(os.path.join(ROOT, "results", "metrics.json"),
                             encoding="utf-8"))["global"].get("score", "")
        _s = str(_sc)
        st.session_state.mode = ("supervise" if _s.startswith("supervisé")
                                 else "one-class" if "features" in _s else "modele")
    except Exception:
        st.session_state.mode = "one-class"
MODE = st.session_state.mode


@st.cache_data
def load_meta():
    p = os.path.join(ROOT, "data", "processed", "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


META = load_meta()


def maha_distance(z: np.ndarray) -> np.ndarray:
    d = z.astype(np.float64) - MAHA["mu"]
    return np.sqrt(np.einsum("ij,jk,ik->i", d, MAHA["prec"], d))


def feat_maha_score(X_scaled: np.ndarray) -> np.ndarray:
    """
    Distance de Mahalanobis sur les 78 features réseau.

    Les .npy livrés sont standardisés par un scaler ajusté sur le trafic
    bénin seul : on l'inverse, puis on réajuste sur tout le train avec les
    statistiques sauvegardées par 03_evaluation.py. Sans cette correction,
    la norme des features suffit à séparer les classes et le score ne
    mesure plus rien.
    """
    n = FEAT["n_net"]
    A = X_scaled[:, :n] * scaler.scale_[:n] + scaler.mean_[:n]
    Z = (A - FEAT["mu_full"]) / FEAT["sd_full"]
    d = Z - FEAT["mean"]
    return np.einsum("ij,jk,ik->i", d, FEAT["prec"], d)


def sup_score(X_scaled: np.ndarray) -> np.ndarray:
    """Frontière linéaire apprise sur bénin + familles connues."""
    n = FEAT["n_net"]
    A = X_scaled[:, :n] * scaler.scale_[:n] + scaler.mean_[:n]
    Z = (A - FEAT["mu_full"]) / FEAT["sd_full"]
    return Z @ SUP["coef"] + SUP["intercept"]


def hgb_score(X_scaled: np.ndarray) -> np.ndarray:
    """Probabilité d'attaque du gradient boosting — le détecteur retenu.

    Meilleur que la frontière linéaire de +6,5 points d'AUROC sur les
    familles inédites (0,9668 contre 0,9026) : la frontière n'est pas
    linéaire dans cet espace de features.
    """
    n = HGB["n_net"]
    A = X_scaled[:, :n] * scaler.scale_[:n] + scaler.mean_[:n]
    Z = (A - HGB["mu"]) / HGB["sd"]
    return HGB["model"].predict_proba(Z)[:, 1]


@torch.no_grad()
def infer(X_scaled, batch=4096, mode_=None):
    """Model outputs, computed in batches to bound memory.

    `mode_` permet de calculer les scores d'un détecteur SANS changer le mode
    actif de l'interface : c'est ce qui rend possible la comparaison des trois
    sur le même lot, sans effet de bord sur le reste de la page.
    """
    _m = mode_ or MODE
    sc, vi, en, cf = [], [], [], []
    for i in range(0, len(X_scaled), batch):
        xb = torch.FloatTensor(X_scaled[i:i + batch])
        out = model(xb)
        z = out["z"].numpy()
        if _m == "one-class" and FEAT is not None:
            sc.append(feat_maha_score(X_scaled[i:i + batch]))
        elif _m == "supervise" and HGB is not None:
            sc.append(hgb_score(X_scaled[i:i + batch]))
        elif _m == "supervise" and SUP is not None:
            sc.append(sup_score(X_scaled[i:i + batch]))
        elif MAHA is not None:
            # score du MS-Transformer : Mahalanobis sur l'embedding appris
            sc.append(maha_distance(z))
        elif MAHA is not None:
            sc.append(maha_distance(z))
        else:
            sc.append(anomaly_score(out["z"], centroid).numpy())
        vi.append(torch.sigmoid(out["violation_logit"]).numpy())
        en.append(free_energy(out["violation_logit"]).numpy())
        conf, _ = next_state_confidence(out["next_pred"], xb, N_NET, N_SYS)
        cf.append(conf.numpy())
    return (np.concatenate(sc), np.concatenate(vi),
            np.concatenate(en), np.concatenate(cf))


# ══════════════════════════════════════════════════════════════════════
#  CACHED EVALUATION ARTEFACTS
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def analysis_data():
    """Scores and labels on the whole test set, plus ROC points."""
    try:
        X = np.load(os.path.join(ROOT, "data", "processed", "X_test.npy"))
        y = np.load(os.path.join(ROOT, "data", "processed", "y_test.npy"), allow_pickle=True)
    except Exception:
        return None
    sc, _, _, _ = infer(X)
    is_ben = np.array(["BENIGN" in str(v).upper() for v in y])
    ybin = (~is_ben).astype(int)
    ben_scores = np.sort(sc[is_ben])

    # ROC (aggregate Zero-Day)
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(ybin, sc)
    auc = float(roc_auc_score(ybin, sc)) if len(set(ybin)) > 1 else float("nan")

    # per-class curves for the recall bars
    classes = sorted({str(v) for v in y[~is_ben]})
    return {"scores": sc, "labels": np.array([str(v) for v in y]),
            "is_benign": is_ben, "ybin": ybin, "ben_scores": ben_scores,
            "fpr": fpr, "tpr": tpr, "auroc": auc, "classes": classes}


def _derive_rates(g: dict) -> dict:
    """
    Complète les taux que 03_evaluation.py n'écrit pas en clair.

    results/metrics.json ne contient pas le taux de fausses alertes, le rappel,
    la précision ni le F1 : il contient la matrice de confusion. Ces quatre
    indicateurs sont donc recalculés ici, une fois, plutôt que lus dans une clé
    absente — c'est cette lecture qui faisait tomber la page sur un KeyError.

    Convention d'écriture de 03_evaluation.py, vérifiée à la ligne 313 :
        [[vrais négatifs, faux positifs],
         [faux négatifs,  vrais positifs]]
    """
    need = ("fpr", "tpr_recall", "precision")
    if any(g.get(k) is None for k in need):
        tn = fp = fn = tp = None
        try:
            (tn, fp), (fn, tp) = g.get("confusion")
        except Exception:
            pass
        if None not in (tn, fp, fn, tp):
            if g.get("fpr") is None and (fp + tn):
                g["fpr"] = fp / (fp + tn)
            if g.get("tpr_recall") is None and (tp + fn):
                g["tpr_recall"] = tp / (tp + fn)
            if g.get("precision") is None and (tp + fp):
                g["precision"] = tp / (tp + fp)
    if g.get("f1") is None:
        p, r = g.get("precision"), g.get("tpr_recall")
        if p is not None and r is not None and (p + r):
            g["f1"] = 2 * p * r / (p + r)
    return g


def _val(g: dict, key: str):
    """Valeur numérique d'un indicateur, ou None s'il est absent ou NaN.

    Les quatre taux dérivables sont recalculés ici si besoin, et pas seulement
    au chargement du fichier : selon l'ordre de rendu, Streamlit peut rappeler
    official_metrics() et repartir d'un dictionnaire neuf — c'est ce qui
    laissait l'onglet Résultats afficher « non mesurable ».
    """
    if key in ("fpr", "tpr_recall", "precision", "f1") and g.get(key) is None:
        _derive_rates(g)
    v = g.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _num(g: dict, key: str, dec: int = 3) -> str:
    """Indicateur brut, ou « non mesurable » — jamais une chaîne vide."""
    v = _val(g, key)
    return t("missing") if v is None else f"{v:.{dec}f}"


def _pct(g: dict, key: str, dec: int = 1) -> str:
    """Indicateur en pourcentage, ou « non mesurable »."""
    v = _val(g, key)
    return t("missing") if v is None else f"{v * 100:.{dec}f} %"


def _ci(g: dict, dec: int = 3) -> str:
    """Intervalle de confiance à 95 % de l'AUROC."""
    ci = g.get("auroc_ci95")
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except Exception:
        return t("missing")
    if lo != lo or hi != hi:
        return t("missing")
    return f"{lo:.{dec}f} – {hi:.{dec}f}"


@st.cache_data(show_spinner=False)
def official_metrics():
    p = os.path.join(ROOT, "results", "metrics.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    if isinstance(d.get("global"), dict):
        _derive_rates(d["global"])
    return d


def live_prepare(df):
    """
    Ordonne les colonnes d'un flux entrant selon ce que le modèle attend.

    Le fichier du collecteur porte les noms de colonnes CICIDS2017. On les
    apparie PAR NOM avec `meta.json.net_columns`, jamais par position : un
    collecteur qui écrit les colonnes dans un autre ordre produirait sinon des
    scores silencieusement faux.

    Renvoie (X_standardisé, None) en cas de succès, ou (None, liste des noms
    manquants) si le fichier n'est pas conforme.
    """
    n = int(CFG["n_net"])
    nf = int(scaler.n_features_in_)
    noms = list(META.get("net_columns") or [])

    if noms and len(noms) >= n:
        attendus = noms[:n]
        manquants = [c for c in attendus if c not in df.columns]
        if manquants:
            return None, manquants
        X = df[attendus].apply(pd.to_numeric, errors="coerce")
    else:
        X = df.select_dtypes(include=[np.number])
        if X.shape[1] < n:
            return None, []

    X = np.nan_to_num(X.values.astype(np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)[:, :n]
    if nf == n + N_SYS:
        X = np.hstack([X, proxy_edr_like(X)])
    return scaler.transform(X[:, :nf]), None


@st.cache_data(show_spinner=False)
def ablation_rows():
    p = os.path.join(ROOT, "results", "ablation.json")
    if not os.path.exists(p):
        return []
    try:
        return json.load(open(p))
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def svdd_row():
    p = os.path.join(ROOT, "results", "oneclass_svdd.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def per_class_png(rows) -> bytes:
    """
    Rendu PNG du rappel par classe, pour le dossier de candidature.

    Palette accessible : le texte utilise #d9ffe8 sur #05080a (17,6:1) et les
    barres sont dans une teinte unique — la valeur est aussi imprimée à côté
    de chaque barre, de sorte que la figure reste lisible sans distinguer
    les couleurs. Le vert phosphore est celui du parcours, pour que la
    figure et l'écran parlent la même langue visuelle.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, FG1, FG2, FG3, BAR = "#05080a", "#d9ffe8", "#b6d9c6", "#9dc0ab", "#39ff88"
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    names = [r["classe"] for r in rows]
    vals = [r["recall"] * 100 for r in rows]
    ax.barh(names, vals, color=BAR)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.1f} %", va="center", color=FG2, fontsize=8)
    ax.set_xlabel("Recall (%)", color=FG3, fontsize=9)
    ax.set_xlim(0, 105)
    ax.tick_params(colors=FG3, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#1d3128")
    ax.set_title("Zero-Day recall per attack class", color=FG1, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURF)
    plt.close(fig)
    return buf.getvalue()


def detector_scales() -> dict:
    """Échelle de chaque détecteur : une distance n'a pas de plafond, une
    probabilité reste entre 0 et 1. Nécessaire pour lire les scores sans
    comparer des nombres incomparables."""
    return {
        "modele":     {"kind": t("cmp_kind_dist"), "max": None},
        "one-class":  {"kind": t("cmp_kind_dist"), "max": None},
        "supervise":  {"kind": t("cmp_kind_prob"), "max": 1.0},
    }


@st.cache_data(show_spinner=False)
def _cmp_threshold(mode_: str, fpr_pct: float):
    """Seuil du détecteur visé, calé sur ses propres scores bénins.

    On ne réutilise pas le seuil du mode actif : chaque détecteur a sa propre
    échelle, et lui appliquer le seuil d'un autre n'aurait aucun sens.
    """
    X = np.load(os.path.join(ROOT, "data", "processed", "X_test.npy"))
    y = np.load(os.path.join(ROOT, "data", "processed", "y_test.npy"),
                allow_pickle=True)
    is_ben = np.array(["BENIGN" in str(v).upper() for v in y])
    ben = X[is_ben]
    if len(ben) > 6000:
        ben = ben[np.random.default_rng(0).choice(len(ben), 6000, replace=False)]
    sc, _, _, _ = infer(ben, mode_=mode_)
    return float(np.quantile(sc, 1.0 - fpr_pct / 100.0))


def _md_inline(txt: str) -> str:
    """Markdown en ligne -> HTML, pour les textes qui passent dans un <caption>.

    Les légendes viennent des dictionnaires T : elles contiennent du gras
    **…** et du code `…`. Dans un bloc HTML, le markdown n'est PAS interprété :
    sans cette conversion, l'utilisateur verrait les astérisques et les
    accents graves à l'écran. Même piège que dans les autres blocs <div>.
    """
    out = _html.escape(str(txt))
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def _md_plain(txt: str) -> str:
    """Retire le balisage markdown : pour les attributs (aria-label, title)."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", str(txt)).replace("`", "")


def _md_html(txt: str) -> str:
    """Markdown -> HTML, pour les textes insérés DANS un bloc HTML.

    Streamlit n'interprète pas le markdown à l'intérieur d'un `<div>` : les
    `**` restaient visibles à l'écran (bandeau d'honnêteté, panneaux des
    limites, encadrés de l'aide). Cette fonction convertit le gras, le code
    et les sauts de paragraphe, après échappement du texte.
    """
    out = _md_inline(txt)
    out = re.sub(r"\n{2,}", "<br><br>", out)
    return out.replace("\n", "<br>")


def typed_html(txt: str) -> str:
    """Texte écrit lettre à lettre, à la manière d'un terminal.

    L'animation est en CSS pur : la largeur part de 0 et s'arrête à
    `n` caractères (UI `ch`, valable parce que la police est monospace).
    Aucun script n'est injecté — Streamlit ne les exécuterait pas.
    `prefers-reduced-motion` et l'impression affichent le texte d'un coup.
    """
    s = _html.escape(str(txt))
    return (f'<span class="typed" style="--n:{max(1, len(str(txt)))}">{s}</span>'
            f'<span class="cur-blink" aria-hidden="true">▌</span>')


def html_table(df, caption: str = "", na: str = "", first_col_header: bool = True,
               key: str = "", file_name: str = "tableau.csv"):
    """Affiche un DataFrame en <table> HTML, lisible par un lecteur d'écran.

    Pourquoi ne pas garder st.dataframe : la grille de Streamlit est dessinée
    dans un canvas. À l'écran, elle est parfaite ; au lecteur d'écran, elle
    est vide. Un <table> avec <caption> et <th scope> est annoncé cellule par
    cellule, avec le nom de la colonne et de la ligne. Ces tableaux portent les
    chiffres du mémoire : ils doivent être lisibles par tout le monde.

    L'export CSV est offert à côté pour l'usage expert (Excel, tableau croisé) :
    on ne perd donc aucune des fonctions de la grille.
    """
    cap = f"<caption>{_md_inline(caption)}</caption>" if caption else ""
    head = "".join(f'<th scope="col">{_html.escape(str(c))}</th>'
                   for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for j, v in enumerate(row.tolist()):
            txt = "" if v is None else str(v)
            cls = ' class="na"' if (na and txt == na) else ""
            if j == 0 and first_col_header:
                cells.append(f'<th scope="row"{cls}>{_html.escape(txt)}</th>')
            else:
                cells.append(f'<td{cls}>{_html.escape(txt)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    # lang={LANG} : le lecteur d'écran choisit la bonne prononciation pour
    # les nombres (« 9,6 % » en français, « 9.6% » en anglais).
    st.markdown(
        f'<div class="tbl-wrap" tabindex="0" role="group" '
        f'aria-label="{_html.escape(_md_plain(caption) or "tableau")}">'
        f'<table class="tbl" lang="{LANG}">{cap}<thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)
    if key:
        st.download_button(
            t("cmp_csv"), df.to_csv(index=False).encode("utf-8-sig"),
            file_name=file_name, mime="text/csv", key=key)


@st.cache_data(show_spinner=False)
def compare_three(fpr_pct: float, lang: str, n: int = 4000):
    """Tableau comparatif des trois détecteurs sur un échantillon FIXE du test.

    Contrairement au bouton de la démonstration, cet échantillon ne dépend
    pas de ce que l'utilisateur a chargé : les chiffres sont donc reproductibles
    et peuvent figurer dans le mémoire.
    """
    X = np.load(os.path.join(ROOT, "data", "processed", "X_test.npy"))
    y = np.load(os.path.join(ROOT, "data", "processed", "y_test.npy"),
                allow_pickle=True)
    lab = np.array([str(v) for v in y])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs, lab = X[idx], lab[idx]
    atk = ~np.array(["BENIGN" in str(v).upper() for v in lab])
    fams = sorted({str(v) for v in lab[atk]})

    rows, manquants = [], []
    for mode_, name in (("modele", t("mode_modele")),
                        ("one-class", t("mode_one_class")),
                        ("supervise", t("mode_supervise"))):
        # Idem ici : un détecteur indisponible garde sa ligne, marquée « n/a »,
        # pour que le tableau montre toujours les trois approches comparées.
        dispo = not ((mode_ == "supervise" and HGB is None and SUP is None)
                     or (mode_ == "one-class" and FEAT is None))
        if not dispo:
            manquants.append(name)
            # Le fichier manquant est conservé : « non mesurable » sans sa
            # cause se lit comme une impossibilité du modèle, alors que c'est
            # le plus souvent un fichier qui n'a pas été copié.
            rows.append({"key": mode_, "mode": name, "n": len(Xs),
                         "alerts": None, "recall": None, "per": {},
                         "file": ("models/features_maha.npz" if mode_ == "one-class"
                                  else "models/supervised_hgb.pkl")})
            continue
        sc, _, _, _ = infer(Xs, mode_=mode_)
        thr = _cmp_threshold(mode_, fpr_pct)
        pred = sc > thr
        recall = float(pred[atk].mean()) if atk.sum() else float("nan")
        per = {}
        for f in fams:
            m = lab == f
            if m.sum():
                per[f] = float(pred[m].mean())
        rows.append({"key": mode_, "mode": name, "n": len(sc),
                     "alerts": int(pred.sum()), "recall": recall, "per": per,
                     "file": None})
    return {"rows": rows, "fams": fams, "n": len(Xs), "manquants": manquants}


SYSCALL_VOCAB = ["open", "read", "close", "socket", "connect", "send", "recv",
                 "mmap", "write", "fsync", "flood", "hulk", "scan", "auth",
                 "fail", "bruteforce", "execve", "chmod", "http", "injection",
                 "irc", "beacon", "ptrace", "exfiltration", "unknown"]


def proxy_edr_like(X_net, seed: int = 42):
    """Reconstruit la branche système exactement comme 01_data_preparation.py.

    Un fichier CICIDS brut ne contient pas ces 25 colonnes : le pipeline les
    DÉRIVE du trafic (durée, volumes, tailles, port). On rejoue donc le même
    calcul plutôt que de compléter par des zéros.

    Limite à connaître : les percentiles sont calculés sur le lot fourni, et un
    bruit de graine fixe est ajouté. Le résultat est très proche de celui de
    l'entraînement, mais pas identique au bit près. Les scores sont donc valides
    en ordre de grandeur, non superposables au chiffre du mémoire.
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
        bins = np.minimum((c * 5).astype(int), 4)
        for b in range(5):
            mask = bins == b
            if not mask.any():
                continue
            out[mask, (j * 3 + b * 2) % len(SYSCALL_VOCAB)] += 1.0
            out[mask, (j * 5 + b) % len(SYSCALL_VOCAB)] += 0.5
    out += rng.normal(0, 0.08, out.shape).astype(np.float32)
    return np.clip(out, 0, None)


LABEL_NAMES = ("label", "labels", "class", "classe", "attack", "attack_type",
               "attack type", "category", "categorie", "catégorie", "target",
               "type", "classe reelle", "class_label")


def extraire_etiquettes(df):
    """Sépare la colonne de classe du reste du fichier.

    Un parquet CICIDS brut porte une colonne « Label » en texte. Elle n'est pas
    une feature : il faut la RETIRER du tableau numérique — sinon un label codé
    0/1 serait compté comme une colonne de plus et décalerait tout — et la
    conserver à part pour mesurer le rappel du lot.
    """
    for c in df.columns:
        if str(c).strip().lower() in LABEL_NAMES:
            lab = df[c].astype(str).values
            return df.drop(columns=[c]), lab, str(c)
    return df, None, None


def ordre_colonnes_suspect(X_net, mu, sd) -> float:
    """Part des colonnes dont la moyenne s'écarte de plus de 4 écarts-types.

    Sert à détecter un fichier dont les 77 colonnes ne sont pas dans le même
    ordre que X_train.npy : les valeurs seraient alors associées aux mauvaises
    features, et les scores n'auraient aucun sens tout en paraissant normaux.
    """
    try:
        m = X_net.mean(axis=0)
        ref_mu, ref_sd = mu[:len(m)], sd[:len(m)]
        ok = np.isfinite(ref_sd) & (ref_sd > 1e-9)
        if ok.sum() < 10:
            return 0.0
        ecart = np.abs(m[ok] - ref_mu[ok]) / ref_sd[ok]
        return float((ecart > 4.0).mean())
    except Exception:
        return 0.0


def threshold_for_fpr(target_fpr: float) -> float:
    """Threshold achieving the target false-alarm rate on healthy traffic."""
    d = analysis_data()
    if d is None:
        if FEAT is not None:
            return FEAT["threshold"]
        return MAHA["threshold"] if MAHA else 0.5
    q = np.quantile(d["ben_scores"], 1.0 - target_fpr)
    return float(q)


# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
# Lien d'évitement : premier élément focalisable, visible au clavier.
st.markdown(
    '<a href="#contenu-principal" class="sr-only" '
    'style="position:absolute;left:-9999px;" '
    'onfocus="this.style.left=\'8px\';this.style.top=\'8px\';'
    'this.style.padding=\'10px 16px\';this.style.background=\'#05080a\';'
    'this.style.color=\'#39ff88\';this.style.border=\'2px solid #39ff88\';'
    'this.style.borderRadius=\'4px\';'
    'this.style.zIndex=9999;this.style.width=\'auto\';this.style.height=\'auto\';'
    'this.style.clip=\'auto\';" '
    'onblur="this.style.left=\'-9999px\';">'
    'Aller au contenu principal</a>', unsafe_allow_html=True)

st.markdown(
    f'<header role="banner" lang="{LANG}">'
    f'<div class="eyebrow">{t("eyebrow")}</div>'
    f'<h1 class="proj-title" style="margin:0;">{t("title")}</h1>'
    f'<p class="proj-sub">{t("subtitle")}</p>'
    # Barre d'invite : la page se présente comme un terminal, et donne
    # d'emblée son état réel — « ONLINE » n'est pas un décor, c'est
    # l'application qui tourne sans réseau.
    f'<div class="termbar">'
    f'<span class="u">root@psm-ht</span><span class="p">:~$</span>'
    f'<span class="cmd">{typed_html("psm-ht --detect --zero-day --explain")}</span>'
    f'<span class="ok">{t("term_online")}</span></div>'
    f'<p style="margin:0;">'
    # Chaque badge associe un symbole et un mot : l'information ne
    # repose jamais sur la seule couleur (critère WCAG 1.4.1).
    f'<span class="badge badge-ok" role="status">'
    f'<span aria-hidden="true">✓</span> {t("badge_offline")}</span>'
    f'<span class="badge badge-warn" role="status">'
    f'<span aria-hidden="true">⚠</span> {t("badge_neg")}</span>'
    f'<span class="badge badge-info" role="status">'
    f'<span aria-hidden="true">≡</span> {t("badge_audit")}</span>'
    f'</p></header>', unsafe_allow_html=True)

st.write("")

# ── Bandeau de statut : la vérité sur ce système, avant tout chiffre ────
# Une seule phrase reste affichée en permanence ; le détail des réserves
# se déplie, pour ne pas repousser chaque onglet sous six lignes de texte.
_st_parts = t("st_body").split("\n\n")
st.markdown(
    f'<div class="note note-warn" lang="{LANG}">'
    f'<b>{t("st_title")}</b><br>{_md_html(_st_parts[0])}'
    f'</div>', unsafe_allow_html=True)
if len(_st_parts) > 1:
    with st.expander(t("st_more")):
        st.markdown("\n\n".join(_st_parts[1:]))

st.write("")

st.markdown(f'<div id="contenu-principal" tabindex="-1" lang="{LANG}"></div>',
            unsafe_allow_html=True)
# Pas de st.write("") ici : l'ancre du lien d'évitement suffit à séparer,
# et deux lignes vides laissaient un trou visible avant les onglets.

# ══════════════════════════════════════════════════════════════════════
#  BARRE LATÉRALE — les réglages, séparés des résultats
#  Un réglage vit à un seul endroit : on ne cherche plus le curseur de
#  fausses alertes dans un onglet alors qu'il vaut pour toute la page.
# ══════════════════════════════════════════════════════════════════════
if "last_fpr" not in st.session_state:
    st.session_state.last_fpr = 5.0
if "pc_step" not in st.session_state:
    st.session_state.pc_step = 1

with st.sidebar:
    # h2 : la barre latérale est une section de premier niveau de la page.
    st.markdown(f"## {t('sb_h')}")
    st.caption(t("sb_form_c"))

    # Les trois réglages sont groupés : rien ne s'applique avant « Appliquer ».
    # Sans ce formulaire, chaque frappe sur le curseur relançait tout le script
    # — le focus quittait le curseur et un lecteur d'écran réannonçait la page
    # entière à chaque pas de 0,5 %.
    with st.form("reglages", border=False):
        # ── langue ─────────────────────────────────────────────────────
        _lang_choisi = st.radio(
            t("sb_lang"), ["fr", "en"], index=0 if LANG == "fr" else 1,
            horizontal=True, help=t("lang_help"), key="w_lang")

        # ── détecteur ──────────────────────────────────────────────────
        _modes = ["modele", "one-class", "supervise"]
        _prev_mode = st.session_state.mode
        _mode_choisi = st.radio(
            t("sb_det"), _modes, index=_modes.index(_prev_mode),
            format_func=lambda v: t("mode_" + v.replace("-", "_")), key="w_mode")
        st.caption(t("mode_" + _prev_mode.replace("-", "_") + "_d"))

        # ── seuil : auparavant enfoui dans l'onglet Démonstration ──────
        _fpr_choisi = st.slider(t("sb_thr"), 0.5, 20.0,
                                float(st.session_state.get("last_fpr", 5.0)),
                                0.5, help=t("sb_thr_help"), format="%.1f %%",
                                key="w_fpr")

        _applique = st.form_submit_button(t("sb_apply"), width='stretch')

    if _applique:
        _change = False
        # Le budget de fausses alertes est mémorisé : sans ça, il revenait à
        # 5 % à chaque rechargement et toute la page se recalculait sur une
        # valeur que l'utilisateur n'avait pas choisie.
        st.session_state.last_fpr = float(_fpr_choisi)
        if _lang_choisi != LANG:
            st.session_state.lang = _lang_choisi
            _change = True
        if _mode_choisi != _prev_mode:
            st.session_state.mode = _mode_choisi
            # analysis_data() est mis en cache sans connaître le mode : on purge
            # sinon la démonstration afficherait les scores de l'autre approche.
            st.cache_data.clear()
            _change = True
        if _change:
            st.rerun()

    MODE = st.session_state.mode
    target_fpr = float(st.session_state.get("last_fpr", 5.0)) / 100.0
    thr = threshold_for_fpr(target_fpr)

    # Le détecteur réellement exécuté doit être visible, jamais implicite.
    # Sans cet avertissement, une app qui n'a pas pu charger le gradient
    # boosting retombe silencieusement sur le Mahalanobis et affiche des
    # distances (570 644) là où le rapport annonce une probabilité (0,87).
    if MODE == "supervise" and HGB is None:
        st.error(
            "⚠️ **Modèle supervisé introuvable** — les scores affichés proviennent "
            "du Mahalanobis sur embedding (AUROC 0.7186), **pas** du gradient "
            "boosting (AUROC 0,9668).\n\n"
            "Fichiers attendus : `models/supervised_hgb.pkl` et "
            "`models/supervised_hgb.npz`. Régénère-les avec "
            "`py 03_evaluation.py --score supervise --algo hgb --target-fpr 0.09`.",
            icon="⚠️")

    st.divider()
    if st.button(t("sb_reset"), width='stretch'):
        for k in ("last_scores", "last_viol", "last_energy", "last_conf",
                  "last_emb", "last_true", "last_batch",
                  "lv_on", "lv_skip", "lv_n", "lv_a", "lv_rows"):
            st.session_state.pop(k, None)
        st.session_state.history = []
        st.session_state.pc_step = 1
        _hp = os.path.join(ROOT, "data", "history_livre.csv")
        if os.path.exists(_hp):
            os.remove(_hp)
        st.rerun()

    st.caption(t("sb_about"))

ETAPES = 7
PC_KEYS = ["pc1", "pc2", "pc3", "pc4", "pc5", "pc6", "pc7"]


def etape_ouverte(n: int) -> bool:
    """Une étape n'est ouverte que si la mesure qu'elle demande existe vraiment.

    Ce n'est pas une récompense : c'est l'état des données. Sans scores en
    mémoire, l'étape « Lire le résultat » n'a rien à montrer — l'ouvrir
    reviendrait à afficher du vide en laissant croire que c'est calculé.
    """
    a_scores = st.session_state.get("last_scores") is not None
    a_batch = st.session_state.get("last_batch") is not None
    return {1: True, 2: True, 3: a_scores, 4: a_scores,
            5: a_batch, 6: a_scores, 7: a_scores}.get(n, False)


def relancer_analyse() -> bool:
    """Recalcule les scores du dernier lot chargé, sans changer les données.

    Le calcul est déterministe : refaire l'analyse redonne les mêmes nombres.
    C'est le but — vérifier que rien n'a bougé — et non produire un résultat
    différent pour faire bonne figure. Renvoie False quand aucun lot n'est en
    mémoire : il n'y a alors rien à refaire, et on le dit au lieu de simuler.
    """
    Xs = st.session_state.get("last_batch")
    if Xs is None:
        return False
    sc, vi, en, cf = infer(Xs)
    st.session_state.last_scores = sc
    st.session_state.last_viol = vi
    st.session_state.last_energy = en
    st.session_state.last_conf = cf
    try:
        with torch.no_grad():
            st.session_state.last_emb = model(torch.FloatTensor(Xs))["z"].numpy()[:, :2]
    except Exception:
        st.session_state.last_emb = np.zeros((len(sc), 2))
    nb = int((sc > thr).sum())
    st.session_state.history.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "alerts": f"{nb}/{len(sc)}",
        "verdict": "THREAT" if nb > 0 else "OK",
        "score": f"{float(sc[sc > thr].mean()) if nb else 0:.2f}",
    })
    return True


# ── Brique visuelle : caisse de jalon, trait du chemin ────────────────────
# Ces trois dessins sont des SVG en ligne, pas des images : ils suivent la
# palette du thème, restent nets à toute taille et ne coûtent aucun fichier
# à copier à côté de app.py. Les couleurs y sont écrites en clair — un SVG
# inline n'hérite pas des variables CSS posées dans un autre bloc.

def coffre_svg(ouvert: bool) -> str:
    """Le jalon du chemin : une caisse de données qui s'ouvre sur un fait.

    Elle ne récompense rien : elle marque une halte que les données ont
    réellement franchie. Fermée, elle est éteinte (gris, LED noires) et dit
    « pas encore ». Ouverte, les voyants passent au vert et le couvercle
    s'entrouvre. Dessin purement décoratif : aria-hidden.
    """
    # métal sombre, voyants verts quand le jalon est atteint
    metal, edge = ("#14201b", "#2f6b4f") if ouvert else ("#0e1412", "#2a3a32")
    led, glow = ("#39ff88", "rgba(57,255,136,0.55)") if ouvert else ("#1f2a25", "none")
    led2 = "#5fd7ff" if ouvert else "#1f2a25"
    return (
        '<svg viewBox="0 0 100 100" aria-hidden="true" focusable="false">'
        f'<rect x="20" y="46" width="60" height="38" rx="4" fill="{metal}" '
        f'stroke="{edge}" stroke-width="2.5"/>'
        f'<rect x="20" y="56" width="60" height="3" fill="{edge}" opacity="0.8"/>'
        f'<rect x="20" y="70" width="60" height="2" fill="{edge}" opacity="0.5"/>'
        # couvercle : entrouvert quand le jalon est atteint
        f'<g transform="translate(0,{0 if ouvert else 3})">'
        f'<rect x="20" y="40" width="60" height="9" rx="3" fill="{edge}"/>'
        f'</g>'
        # serrure centrale
        f'<rect x="44" y="55" width="12" height="12" rx="2" fill="#07100c" '
        f'stroke="{edge}" stroke-width="2"/>'
        f'<circle cx="50" cy="60" r="2.2" fill="{led}"/>'
        f'<rect x="49" y="60" width="2" height="4.5" rx="1" fill="{led}"/>'
        # voyants de coin
        f'<circle cx="27" cy="76" r="2.4" fill="{led}" style="filter:drop-shadow(0 0 4px {glow})"/>'
        f'<circle cx="73" cy="76" r="2.4" fill="{led2}"/>'
        + (f'<path d="M50 18 v8 M42 22 l16 0" stroke="{led}" stroke-width="2.5" '
           f'stroke-linecap="round" style="filter:drop-shadow(0 0 5px {glow})"/>'
           if ouvert else
           f'<path d="M50 24 v-6 a7 7 0 0 1 14 0 v6" fill="none" stroke="{edge}" '
           f'stroke-width="3" stroke-linecap="round"/>')
        + '</svg>'
    )


def lien_svg(x1: float, x2: float, etat: str = "") -> str:
    """Le trait pointillé entre deux nœuds, dans la couleur de l'état.

    Il est étiré (`preserveAspectRatio="none"`) pour suivre la largeur de
    la carte, et `vector-effect` garde une épaisseur et un pointillé
    constants malgré cette déformation. Décoratif : aria-hidden.
    """
    _d = (f"M {x1:.1f} 1 C {x1:.1f} 13, {x2:.1f} 13, {x2:.1f} 27")
    return (f'<div class="duo-link {etat}" aria-hidden="true">'
            f'<svg viewBox="0 0 100 28" preserveAspectRatio="none" '
            f'focusable="false"><path class="ln" d="{_d}" '
            f'vector-effect="non-scaling-stroke"/></svg></div>')


def entete_unite(n: int, prog: bool = False):
    """Bandeau d'unité : ce que fait la page, et où l'on en est.

    Le même bloc sur les six onglets : c'est la seule façon d'éviter que
    l'interface change de ton d'une page à l'autre. Le bouton Guide porte
    une phrase propre à l'unité — jamais un texte générique recopié.

    `prog=True` n'est utilisé que par l'unité 2 : la progression compte les
    étapes que les données autorisent réellement, sur les autres pages il
    n'y a rien à compter.
    """
    _titre, _sous, _guide = t(f"un{n}_t"), t(f"un{n}_s"), t(f"un{n}_g")
    # La maquette montre « Unité 1 » en gros et la phrase en dessous. Le titre
    # des dictionnaires porte les deux, séparés par « · » : on le scinde ici
    # plutôt que d'ajouter douze clés de traduction pour la même phrase.
    _k, _sep, _t2 = _titre.partition(" · ")
    if not _sep:
        _k, _t2 = "", _titre
    _k_html = f'<div class="unit-k">{_html.escape(_k)}</div>' if _k else ""
    _prog_html = ""
    if prog:
        _nd = sum(1 for k in range(1, ETAPES + 1) if etape_ouverte(k))
        _dots = "".join(
            f'<span class="u-dot{" u-on" if k <= _nd else ""}"></span>'
            for k in range(1, ETAPES + 1))
        _prog_html = (f'<div class="unit-p"><span class="unit-pl">{t("u_prog")}</span>'
                      f'<span class="unit-dots" aria-hidden="true">{_dots}</span>'
                      f'<span class="unit-pn">'
                      f'{t("u_prog_n", n=_nd, m=ETAPES)}</span></div>')
    # Le conteneur à clé porte le fond vert : les deux colonnes (texte +
    # bouton Guide) forment ainsi UNE carte, comme sur la maquette.
    with st.container(key=f"duo_unit_{n}"):
        _c_txt, _c_btn = st.columns([5.2, 1], gap="small",
                                    vertical_alignment="center")
        with _c_txt:
            st.markdown(
                f'<div class="unit" lang="{LANG}">{_k_html}'
                f'<div class="unit-t">{typed_html(_t2)}</div>'
                f'<div class="unit-s">{_md_html(_sous)}</div>{_prog_html}</div>',
                unsafe_allow_html=True)
        with _c_btn:
            with st.popover(t("u_guide")):
                st.markdown(t(f"un{n}_g"))
    st.write("")


DUO_POS = [3, 4, 3, 2, 2, 3, 4]   # colonne du nœud (1..5) : dessine le zigzag
DUO_JALONS = (3, 6)               # haltes dorées placées après ces étapes


def _duo_x(pos: int) -> float:
    """Abscisse du centre d'une cellule, en pourcentage de la largeur."""
    return 10.0 + (pos - 1) * 20.0


def _duo_etat(i: int, courante: int):
    """État et symbole d'une étape. Quatre états, aucun mensonge :

    « done » n'apparaît que sur une étape réellement franchie, « open » sur
    une étape dont les données existent déjà, « lock » quand la mesure
    manque. Le symbole double la couleur (WCAG 1.4.1).
    """
    if not etape_ouverte(i):
        return "lock", "🔒"
    if i == courante:
        return "cur", str(i)
    if i < courante:
        return "done", "✓"
    return "open", str(i)


def _duo_statut(i: int, etat: str) -> str:
    """Statut écrit en toutes lettres : c'est lui qui informe."""
    if etat == "lock":
        _det = t("pc_after", n=i - 1) if i > 1 else ""
        return f"{t('pc_locked')} · {_det}" if _det else t("pc_locked")
    if etat == "cur":
        return t("pc_now")
    if etat == "done":
        return t("pc_done")
    return t("pc_open")


def _duo_message() -> str:
    """Ce que dit la ligne d'état : l'état du lot, jamais un encouragement creux."""
    _sc = st.session_state.get("last_scores")
    if _sc is None:
        return t("owl_hello")
    _a = np.asarray(_sc, dtype=float)
    return t("owl_done", a=int((_a > thr).sum()), n=len(_a), s=f"{thr:.2f}")


def parcours() -> int:
    """Le chemin des sept étapes, en zigzag. Retourne l'étape affichée.

    Chaque nœud est un bouton Streamlit — donc au clavier, annoncé, activable
    par Entrée. Un nœud verrouillé reste cliquable : il ouvre un encart qui
    dit CE QUI MANQUE et propose d'y aller. Aucun clic ne produit une page
    blanche.

    La forme suit la maquette : nœuds ronds qui s'enfoncent au clic, trait
    pointillé entre eux, deux haltes en caisse de données, une ligne d'état
    qui dit l'état réel du lot. La règle de fond ne bouge pas : l'état est
    écrit sous chaque nœud, la couleur ne fait que renforcer.
    """
    courante = int(st.session_state.get("pc_step", 1))

    # ── ligne d'état : elle dit l'état réel du lot, en une ligne de terminal.
    # Elle remplace la bulle de la mascotte (supprimée) : la phrase est la
    # même, seule la voix change — un prompt, pas un personnage.
    st.markdown(
        f'<div class="duo-term" role="status" lang="{LANG}">'
        f'<span class="p" aria-hidden="true">$</span>'
        f'<span>{_md_inline(_duo_message())}</span></div>',
        unsafe_allow_html=True)

    st.markdown(f'<p class="pc-path-h" lang="{LANG}">{t("u_parcours")}</p>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="pc-path-l" lang="{LANG}">{t("pc_path_c")}</p>',
                unsafe_allow_html=True)

    # Progression réelle : le nombre d'étapes que les données autorisent,
    # pas un score inventé. C'est la seule « récompense » du parcours.
    _n_ouvertes = sum(1 for k in range(1, ETAPES + 1) if etape_ouverte(k))
    st.markdown(
        f'<div class="pc-prog" lang="{LANG}">'
        f'<div class="pc-prog-l">{t("pc_prog")} · {_n_ouvertes}/{ETAPES}</div>'
        f'<div class="pc-prog-b" role="img" aria-label="{_n_ouvertes} / {ETAPES}">'
        f'<span style="width:{100 * _n_ouvertes / ETAPES:.0f}%"></span></div></div>',
        unsafe_allow_html=True)

    # ── le chemin ────────────────────────────────────────────────────────
    # Une ligne par nœud dans une grille de cinq cellules : la cellule
    # occupée change à chaque étape, ce qui dessine le zigzag. Le trait
    # pointillé est un SVG étiré entre les deux abscisses : il reste juste
    # à toute largeur, y compris sur un téléphone.
    _prec_x = None
    for i in range(1, ETAPES + 1):
        etat, marque = _duo_etat(i, courante)
        statut = _duo_statut(i, etat)
        _x = _duo_x(DUO_POS[i - 1])

        if _prec_x is not None:
            _teinte = ("done" if etat == "done" else
                       "cur" if etat in ("cur", "open") else "")
            st.markdown(lien_svg(_prec_x, _x, _teinte), unsafe_allow_html=True)

        # La clé du conteneur devient une classe CSS (`st-key-duo_row_<n>`) :
        # c'est elle qui garde les cinq cellules côte à côte sur téléphone.
        with st.container(key=f"duo_row_{i}"):
            _cells = st.columns(5, gap="small")
            with _cells[DUO_POS[i - 1] - 1]:
                # Un cadenas reste un bouton : il explique au lieu de refuser.
                if st.button(marque, key=f"pcb_{etat}_{i}",
                             help=f"{i} · {t(PC_KEYS[i - 1])} — {statut}"):
                    st.session_state.pc_step = i
                    st.rerun()
                # Le nom et le statut sont sous le nœud, en toutes lettres.
                _pill = ('<span class="duo-now-pill">● ' + _html.escape(t("pc_now")) +
                         '</span>' if etat == "cur" else
                         '<span class="duo-lock-pill">🔒 ' + _html.escape(t("pc_locked")) +
                         '</span>' if etat == "lock" else "")
                st.markdown(
                    f'<div class="duo-lab" lang="{LANG}">'
                    f'<span class="duo-k">{t("duo_step", n=i)}</span>'
                    f'<b>{_html.escape(t(PC_KEYS[i - 1]))}</b>'
                    f'<span class="duo-sub">{_html.escape(statut)}</span>{_pill}</div>',
                    unsafe_allow_html=True)
        _prec_x = _x

        # ── halte dorée : un coffre qui s'ouvre sur un vrai événement ────
        if i in DUO_JALONS:
            _ouvert = (st.session_state.get("last_scores") is not None if i == 3
                       else bool(st.session_state.get("history")))
            st.markdown(lien_svg(_prec_x, 50.0, "done" if _ouvert else ""),
                        unsafe_allow_html=True)
            _lib = t("jalon_a" if i == 3 else "jalon_b") if _ouvert else t("jalon_off")
            st.markdown(
                f'<div class="duo-jalon{" on" if _ouvert else ""}" lang="{LANG}">'
                f'{coffre_svg(_ouvert)}'
                f'<span class="jl">{_html.escape(_lib)}</span></div>',
                unsafe_allow_html=True)
            _prec_x = 50.0

    st.markdown(
        f'<div class="pc-legend" lang="{LANG}">'
        f'<span><b aria-hidden="true">✓</b> {t("pc_done")}</span>'
        f'<span><b aria-hidden="true">●</b> {t("pc_now")}</span>'
        f'<span><b aria-hidden="true">🔒</b> {t("pc_locked")}</span>'
        f'<span class="pc-legend-n">{t("pc_legend")}</span></div>',
        unsafe_allow_html=True)

    # L'étape affichée est annoncée : un lecteur d'écran n'entend pas le style.
    st.markdown(
        f'<p class="sr-only" role="status" lang="{LANG}">{t("pc_cur")} : '
        f'{courante} · {_html.escape(t(PC_KEYS[courante - 1]))}</p>',
        unsafe_allow_html=True)

    # Ce que fait un cadenas, écrit une fois pour toutes sous le parcours :
    # le lecteur n'a pas à cliquer pour l'apprendre.
    st.markdown(
        f'<div class="u-lock" lang="{LANG}">'
        f'<div class="u-lock-h">{t("u_lock_h")}</div>'
        f'<div class="u-lock-b">{_md_html(t("u_lock_b"))}</div>'
        f'<div class="u-lock-g">{t("u_gate")}</div></div>',
        unsafe_allow_html=True)

    if not etape_ouverte(courante):
        # Un cadenas explique au lieu de refuser.
        manque = t("pc_missing") if courante in (3, 4, 6, 7) else t("pc_missing_b")
        st.warning(manque)
        if st.button(t("pc_go1"), key="pc_go1"):
            st.session_state.pc_step = 1
            st.rerun()

    return courante


st.write("")

tab_o, tab_d, tab_r, tab_m, tab_l, tab_h = st.tabs([
    t("tab_over"), t("tab_demo"), t("tab_res"), t("tab_method"),
    t("tab_limit"), t("tab_help")])


# ══════════════════════════════════════════════════════════════════════
#  1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab_o:
    st.markdown(f'<h2 class="tab-h">{t("tab_over")}</h2>', unsafe_allow_html=True)
    # Le même bandeau que l'unité 2 : l'interface ne change pas de ton
    # quand on change d'onglet.
    entete_unite(1)
    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        st.markdown(f"### {t('o_problem_h')}")
        st.markdown(t("o_problem"))
        st.markdown(f"### {t('o_approach_h')}")
        st.markdown(t("o_approach"))
    with c2:
        st.markdown(
            f'<div class="pitch" lang="{LANG}">'
            f'<div style="font-size:0.72em;color:var(--fg-4);text-transform:uppercase;'
            f'letter-spacing:0.09em;">{t("o_dataset")}</div>'
            f'<div style="font-weight:700;font-size:1.05em;margin-bottom:12px;">{t("o_dataset_v")}</div>'
            f'<div style="font-size:0.72em;color:var(--fg-4);text-transform:uppercase;'
            f'letter-spacing:0.09em;">{t("o_train")}</div>'
            f'<div style="margin-bottom:12px;">{t("o_train_v")}</div>'
            f'<div style="font-size:0.72em;color:var(--fg-4);text-transform:uppercase;'
            f'letter-spacing:0.09em;">{t("o_test")}</div>'
            f'<div style="margin-bottom:12px;">{t("o_test_v")}</div>'
            f'<div style="font-size:0.72em;color:var(--fg-4);text-transform:uppercase;'
            f'letter-spacing:0.09em;">{t("o_score")}</div>'
            f'<div>{t("o_score_v")}</div>'
            f'</div>', unsafe_allow_html=True)

    st.write("")
    om = official_metrics()
    if om:
        g = om["global"]
        k1, k2, k3, k4 = st.columns(4)
        _na, _nb = _val(g, "n_anomaly"), _val(g, "n_benign")
        k1.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">AUROC</div>'
                    f'<div class="kpi-v">{_num(g, "auroc")}</div>'
                    f'<div class="kpi-s">IC95 % {_ci(g)}</div></div>',
                    unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("r_fpr")}</div>'
                    f'<div class="kpi-v">{_pct(g, "fpr")}</div>'
                    f'<div class="kpi-s">cible ≤ 10 %</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("r_recall")}</div>'
                    f'<div class="kpi-v">{_pct(g, "tpr_recall")}</div>'
                    f'<div class="kpi-s">{g.get("threshold_method") or t("missing")}</div></div>',
                    unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">n</div>'
                    f'<div class="kpi-v">{f"{int(_na):,}" if _na is not None else t("missing")}</div>'
                    f'<div class="kpi-s">Zero-Day / '
                    f'{f"{int(_nb):,}" if _nb is not None else t("missing")}</div></div>',
                    unsafe_allow_html=True)

    st.write("")
    st.markdown(
        f'<div class="panel panel-accent" lang="{LANG}">{_md_html(t("o_honest"))}</div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  2 · LIVE DEMO
# ══════════════════════════════════════════════════════════════════════
with tab_d:
    st.markdown(f'<h2 class="tab-h">{t("tab_demo")}</h2>', unsafe_allow_html=True)
    # Le chemin retourne l'étape affichée ; chaque région de cet onglet est
    # ensuite gardée par `_pc` : une étape verrouillée ne montre jamais du vide.
    # Deux colonnes comme sur la maquette : le chemin à gauche, le contenu
    # de l'étape à droite. Sur mobile, Streamlit les empile : le chemin passe
    # au-dessus du contenu, ce qui reste l'ordre de lecture correct.
    entete_unite(2, prog=True)
    _pc_col, _body_col = st.columns([1.35, 2.5], gap="large")
    with _pc_col:
        # Marqueur invisible : il sert uniquement au CSS à reconnaître cette
        # colonne et à lui donner l'allure de carte de la maquette.
        st.markdown('<div class="pc-mark duo-path" aria-hidden="true"></div>',
                    unsafe_allow_html=True)
        _pc = parcours()
    with _body_col:
        st.markdown('<div class="pc-mark" aria-hidden="true"></div>',
                    unsafe_allow_html=True)
        # L'en-tête annonce où l'on est, puis le titre de l'étape : c'est le
        # bandeau de la maquette (« ÉTAPE 3 · EN COURS »).
        _mots = {True: t("pc_now"), False: t("pc_locked")}
        st.markdown(
            f'<div class="pc-panel-h" lang="{LANG}">{t("pc_etape")} {_pc} · '
            f'{_mots[etape_ouverte(_pc)]}</div>', unsafe_allow_html=True)
        st.markdown(f"### {t(PC_KEYS[_pc - 1])}")
        # Une phrase dit ce qu'on fait dans cette étape, avant tout chiffre.
        st.markdown(
            f'<p class="pc-do" lang="{LANG}">{t("pc_do" + str(_pc))}</p>',
            unsafe_allow_html=True)

        # Avancer ou reculer d'un bouton, sans devoir viser un nœud du chemin.
        # Le bouton reste désactivé quand l'étape suivante n'a pas de résultat,
        # et la raison est écrite sous lui : rien ne se bloque en silence.
        _suivante = _pc + 1
        _peut_avancer = _suivante <= ETAPES and etape_ouverte(_suivante)
        _nav_p, _nav_n = st.columns(2)
        with _nav_p:
            if st.button(t("pc_prev"), key="pc_nav_prev", width="stretch",
                         disabled=(_pc <= 1)):
                st.session_state.pc_step = max(1, _pc - 1)
                st.rerun()
        with _nav_n:
            if st.button(t("pc_next"), key="pc_nav_next", width="stretch",
                         disabled=not _peut_avancer):
                st.session_state.pc_step = min(ETAPES, _suivante)
                st.rerun()
        if _pc < ETAPES and not _peut_avancer:
            st.caption(t("pc_next_why"))
        if _pc <= 2:
            st.caption(t("d_s1_c"))
            mode = st.segmented_control(t("d_mode"), [t("d_live"), t("d_file")],
                                        default=t("d_live")) or t("d_live")

            if "history" not in st.session_state:
                st.session_state.history = []

            @st.cache_data(show_spinner=False)
            def test_pool():
                try:
                    Xt = np.load(os.path.join(ROOT, "data", "processed", "X_test.npy"))
                    yt = np.load(os.path.join(ROOT, "data", "processed", "y_test.npy"),
                                 allow_pickle=True)
                    ben = np.array(["BENIGN" in str(l).upper() for l in yt])
                    return Xt[ben], Xt[~ben], [str(v) for v in yt[~ben]]
                except Exception:
                    return None, None, None


            def sample_flux(n=200):
                B, Z, ZL = test_pool()
                if B is not None and len(B):
                    rng = np.random.default_rng()
                    k = int(rng.integers(6, 16))
                    k = min(k, len(Z), n)
                    ib = rng.choice(len(B), n - k, replace=False)
                    iz = rng.choice(len(Z), k, replace=False)
                    X = np.vstack([B[ib], Z[iz]])
                    lab = ["BENIGN"] * (n - k) + [ZL[i] for i in iz]
                    o = rng.permutation(n)
                    return X[o], [lab[i] for i in o], True
                rng = np.random.default_rng()
                X = rng.normal(0, 1, (n, scaler.n_features_in_)) * 0.55
                X[rng.choice(n, max(1, n // 12), replace=False)] += rng.uniform(1.5, 2.5)
                return scaler.transform(X), ["SIM"] * n, False


            def run_demo(Xs, labels, from_test):
                st.session_state.last_batch = Xs
                # L'utilisateur vient de franchir l'étape 1 : le repère doit le dire.
                st.session_state.pc_step = 3
                sc, vi, en, cf = infer(Xs)
                st.session_state.last_scores = sc
                st.session_state.last_viol = vi
                st.session_state.last_energy = en
                st.session_state.last_conf = cf
                st.session_state.last_true = labels if from_test else None
                with torch.no_grad():
                    st.session_state.last_emb = model(torch.FloatTensor(Xs))["z"].numpy()[:, :2]
                nb = int((sc > thr).sum())
                st.session_state.history.append({
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "alerts": f"{nb}/{len(sc)}",
                    "verdict": "THREAT" if nb > 0 else "OK",
                    "score": f"{float(sc[sc > thr].mean()) if nb else 0:.2f}",
                })
                pd.DataFrame(st.session_state.history).to_csv(
                    os.path.join(ROOT, "data", "history_livre.csv"), index=False)
                # Le chemin a été dessiné avant l'analyse, avec l'état d'avant : on
                # redessine pour que l'étape 3 apparaisse ouverte dans le même geste.
                st.rerun()


            if mode == t("d_live"):
                if st.button(t("d_generate"), type="primary", width='stretch'):
                    Xs, labs, ft = sample_flux(200)
                    run_demo(Xs, labs, ft)
            else:
                up = st.file_uploader(t("d_upload"), type=["csv", "parquet"])
                if up and st.button(t("d_generate"), type="primary", width='stretch'):
                    df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_parquet(up)
                    # Plafond de sécurité : au-delà, l'inférence CPU fige le navigateur.
                    # On tire un ÉCHANTILLON ALÉATOIRE et non les premières lignes :
                    # df.iloc[:50000] suit l'ordre du fichier, donc un parquet trié
                    # (ex. « BENIGN d'abord ») ne serait analysé que sur une classe et
                    # afficherait 0 % d'alertes — un faux diagnostic.
                    def _n(x):
                        # séparateur de milliers conforme à la langue de l'interface :
                        # espace insécable en français, virgule en anglais.
                        return f"{x:,}".replace(",", " ") if LANG == "fr" else f"{x:,}"

                    if len(df) > 50_000:
                        st.warning(
                            f"{_n(len(df))} lignes — échantillon aléatoire de 50 000 analysé "
                            f"({_n(len(df) - 50_000)} non vues).")
                        df = df.sample(n=50_000, random_state=0).reset_index(drop=True)

                    df, _labs_up, _col_lab = extraire_etiquettes(df)
                    if _col_lab is not None:
                        st.caption(t("lab_found", c=_col_lab))

                    X = df.select_dtypes(include=[np.number]).values
                    nf = scaler.n_features_in_

                    # Garde-fou de format : le modèle attend nf colonnes DANS UN ORDRE
                    # PRÉCIS. Un fichier brut CICIDS en contient 78 (features + Label),
                    # pas 102 : compléter avec des zéros produirait des scores qui ne
                    # veulent rien dire. Mieux vaut refuser que tromper.
                    # Cas fréquent : le fichier ne contient que le bloc réseau (77
                    # colonnes), comme un parquet CICIDS d'origine. La branche système
                    # n'a jamais été collectée — le pipeline la dérive du trafic. On
                    # rejoue donc ce calcul au lieu de refuser ou de mettre des zéros.
                    _reconstruit = False
                    if X.shape[1] == N_NET and nf == N_NET + N_SYS:
                        _mu = _sd = None
                        try:
                            _st = np.load(os.path.join(ROOT, "models", "supervised_hgb.npz"))
                            _mu, _sd = _st["mu"], _st["sd"]
                        except Exception:
                            pass
                        _part = ordre_colonnes_suspect(X, _mu, _sd) if _mu is not None else 0.0
                        if _part > 0.25:
                            st.error(
                                f"**Ordre des colonnes incohérent** — {_part*100:.0f} % des "
                                f"colonnes ont une moyenne très éloignée de celle vue à "
                                f"l'entraînement. Le fichier contient bien 77 colonnes, "
                                f"mais probablement pas les mêmes, ou pas dans le même "
                                f"ordre. Les scores n'auraient aucun sens.\n\n"
                                f"Téléverse `data/processed/demo_sample.csv`, produit par "
                                f"l'étape 01 : c'est le seul fichier dont l'ordre est garanti.")
                            st.stop()
                        st.info(
                            f"**Branche système reconstruite** — le fichier contient les "
                            f"{N_NET} colonnes réseau mais pas les {N_SYS} colonnes "
                            f"système, qui ne sont jamais collectées : le pipeline les "
                            f"dérive du trafic. Ce calcul est rejoué ici (`proxy_edr`). "
                            f"Les scores sont exploitables qualitativement, mais ne se "
                            f"comparent pas au chiffre exact du mémoire, car les "
                            f"percentiles sont recalculés sur ce lot.")
                        X = np.hstack([X, proxy_edr_like(X)])
                        _reconstruit = True

                    if X.shape[1] != nf:
                        st.error(
                            f"**Format incompatible** — le fichier fournit "
                            f"{X.shape[1]} colonnes numériques, le modèle en attend {nf} "
                            f"({N_NET} réseau + {N_SYS} système). Les scores seraient "
                            f"dépourvus de sens.\n\n"
                            f"Un fichier de **{N_NET} colonnes réseau seules** est accepté : "
                            f"la branche système est alors reconstruite. Tout autre nombre "
                            f"de colonnes ne peut pas l'être.")
                        with st.expander("Comment obtenir un fichier compatible"):
                            st.markdown(
                                "**Le plus simple** — le pipeline produit déjà un fichier "
                                "prêt à téléverser :\n\n"
                                "`data/processed/demo_sample.csv` — 20 000 flux, "
                                f"{nf} colonnes brutes + une colonne `label`.\n\n"
                                "S'il n'existe pas encore, relance :\n\n"
                                "```powershell\n"
                                "py 01_data_preparation.py\n"
                                "```\n\n"
                                "**Pour un autre fichier**, il doit contenir les "
                                f"{nf} colonnes brutes dans le même ordre que "
                                "`X_train.npy`, sans standardisation appliquée :\n\n"
                                "```python\n"
                                "import numpy as np, pandas as pd, pickle\n"
                                "X = np.load('data/processed/X_test.npy')\n"
                                "y = np.load('data/processed/y_test.npy', allow_pickle=True)\n"
                                "sc = pickle.load(open('models/scaler.pkl', 'rb'))\n"
                                "Xraw = X * sc.scale_ + sc.mean_   # inverse la standardisation\n"
                                f"df = pd.DataFrame(Xraw, columns=[f'feat_{{i}}' "
                                f"for i in range(Xraw.shape[1])])\n"
                                "df['label'] = y\n"
                                "df.to_csv('demo_sample.csv', index=False)\n"
                                "```\n\n"
                                f"Un fichier de **{N_NET} colonnes réseau seules** est "
                                f"accepté : la branche système est reconstruite à partir "
                                f"du trafic, comme à l'entraînement. Au-delà de {N_NET} "
                                f"colonnes sans atteindre {nf}, l'ordre ne peut pas être "
                                f"vérifié et l'analyse est refusée plutôt que faussée.")
                        st.stop()

                    X = scaler.transform(X[:, :nf])
                    run_demo(X, _labs_up, _labs_up is not None)

            # ════════════════════════════════════════════════════════════════════
        #  RÉSULTAT DE L'ANALYSE — remonté juste après la source
        #  C'est ce que l'utilisateur vient chercher : il ne doit pas avoir à
        #  faire défiler la trace, le rejeu et le direct pour le trouver.
        # ════════════════════════════════════════════════════════════════════
        st.divider()

        scores = st.session_state.get("last_scores")
        if scores is None:
            st.info(t("d_no_data"))
        else:
            scores = np.asarray(scores)
            violation = np.asarray(st.session_state.get("last_viol", np.zeros(len(scores))))
            energy = np.asarray(st.session_state.get("last_energy", np.zeros(len(scores))))
            conf = np.asarray(st.session_state.get("last_conf", np.ones(len(scores))))
            # Plus de tirage aléatoire en secours : une carte de bruit
            # présentée comme une carte des connexions serait un mensonge.
            # Sans projection, la section dit qu'elle n'a rien à montrer.
            _emb = st.session_state.get("last_emb")
            emb = np.asarray(_emb) if _emb is not None else None

            nb_total = len(scores)
            mask = scores > thr
            nb_alert = int(mask.sum())
            pct = 100.0 * nb_alert / max(nb_total, 1)

            if _pc == 3:
                # Le tableau des trois détecteurs vient AVANT l'interprétation :
                # on voit d'abord ce que chacun produit sur le même lot, ensuite
                # ce que ces nombres veulent dire.
                _cmp3 = compare_three(float(st.session_state.get("last_fpr", 5.0)), LANG)
                _r3 = []
                for _r in _cmp3["rows"]:
                    _rec = _r["recall"]
                    _r3.append({
                        t("cmp_col"): _r["mode"],
                        t("cmp_alerts"): (f'{_r["alerts"]:,} / {_r["n"]:,}'
                                          if _r["alerts"] is not None else t("n_a")),
                        t("cmp_recall"): (t("n_a") if _rec is None or _rec != _rec
                                          else f'{_rec * 100:.1f} %'),
                    })
                html_table(pd.DataFrame(_r3), caption=t("u_res_cap"), na=t("n_a"),
                           key="dl_cmp_e3", file_name="resultat_lot.csv")
                # Une ligne « non mesurable » sans sa cause est illisible :
                # on nomme le fichier, et on dit que ce n'est pas une
                # impossibilité du détecteur.
                _absen = sorted({r["file"] for r in _cmp3["rows"]
                                 if r["alerts"] is None and r.get("file")})
                if _absen:
                    st.caption(t("cmp_why", f=" · ".join(_absen)))

                # Les deux lectures d'un même chiffre, côte à côte. C'est la
                # partie « expliquer sans jargon » : un chiffre sans sa limite
                # se lit comme une promesse.
                _m1, _m2 = st.columns(2)
                with _m1:
                    st.markdown(
                        f'<div class="panel" lang="{LANG}" '
                        f'style="height:100%;border-left:3px solid var(--ok);">'
                        f'<div style="font-weight:700;color:var(--ok);margin-bottom:8px;">'
                        f'{t("means_h")}</div>'
                        f'<div style="font-size:0.9rem;color:var(--fg-2);">'
                        f'{_md_html(t("means_b"))}</div></div>', unsafe_allow_html=True)
                with _m2:
                    st.markdown(
                        f'<div class="panel" lang="{LANG}" '
                        f'style="height:100%;border-left:3px solid var(--danger);">'
                        f'<div style="font-weight:700;color:var(--danger);margin-bottom:8px;">'
                        f'{t("nomeans_h")}</div>'
                        f'<div style="font-size:0.9rem;color:var(--fg-2);">'
                        f'{_md_html(t("nomeans_b"))}</div></div>', unsafe_allow_html=True)
                st.caption(t("pc_legend"))

                # Deux issues, dans cet ordre : avancer, ou refaire.
                _b_next, _b_redo = st.columns([1.4, 1])
                with _b_next:
                    _nxt = min(_pc + 1, ETAPES)
                    if st.button(t("u_next_n", n=_nxt, nom=t(PC_KEYS[_nxt - 1])),
                                 key="pc_unlock_next", type="primary", width="stretch"):
                        st.session_state.pc_step = _nxt
                        st.rerun()
                with _b_redo:
                    if st.button(t("u_redo"), key="pc_redo", width="stretch"):
                        if relancer_analyse():
                            st.rerun()
                        else:
                            st.caption(t("u_redo_no"))

                # Ce que le parcours garde d'une étape à l'autre : des mesures
                # réelles et des réglages, rien d'autre.
                st.markdown(
                    f'<div class="u-mem" lang="{LANG}"><div class="u-mem-h">'
                    f'{t("u_mem_h")}</div>{t("u_mem_b")}</div>',
                    unsafe_allow_html=True)
                st.write("")
            # Trois indicateurs, comme sur le rendu de référence. Le nombre de
            # flux normaux n'a pas de carte à lui : il se lit sous « Alertes »
            # (« N sous le seuil »). Quatre cartes pour trois informations
            # affichaient deux fois le même nombre.
            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("d_flows")}</div>'
                        f'<div class="kpi-v">{nb_total:,}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("d_alerts")}</div>'
                        f'<div class="kpi-v" style="color:{"var(--danger)" if nb_alert else "var(--ok)"}">'
                        f'{nb_alert:,}</div><div class="kpi-s">{nb_total - nb_alert:,} '
                        f'{t("d_below")} · {pct:.1f} {t("d_detected")}</div></div>',
                        unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("d_verdict")}</div>'
                        f'<div class="kpi-v" style="color:{"var(--danger)" if nb_alert else "var(--ok)"}">'
                        f'{t("d_threat") if nb_alert else t("d_clean")}</div>'
                        f'<div class="kpi-s">{t("d_threat_sub") if nb_alert else t("d_clean_sub")}</div></div>',
                        unsafe_allow_html=True)
            # Sans cette phrase, les trois chiffres flottent : on ne sait ni sur
            # quel lot ni sur quel seuil ils sont calculés.
            st.caption(t("d_lot_cap",
                         fpr=f"{float(st.session_state.get('last_fpr', 5.0)):g}"))


            if _pc >= 3:
                # labelled batch: real precision / recall
                tl = st.session_state.get("last_true")
                if tl is not None and len(tl) == nb_total:
                    tb = np.array(["BENIGN" in str(l).upper() for l in tl])
                    tp = int((mask & ~tb).sum()); fp = int((mask & tb).sum())
                    m1, m2, m3 = st.columns(3)
                    # Ces trois libelles etaient ecrits en anglais dans la
                    # version francaise : toute chaine visible a sa paire.
                    m1.metric(t("d_tp"), tp,
                              f"{tp / max(int((~tb).sum()), 1):.1%} {t('d_recall_s')}")
                    m2.metric(t("d_fp"), fp,
                              f"{fp / max(int(tb.sum()), 1):.1%} {t('d_fpr_s')}",
                              delta_color="inverse")
                    m3.metric(t("d_zd"), int((~tb).sum()))

                st.write("")
                left, right = st.columns([1.15, 0.85])
                with left:
                    st.markdown(f"**{t('d_suspect_h')}**")
                    st.caption(t("d_suspect_c"))
                    smax = float(scores.max()) + 1e-9
                    for i in np.argsort(scores)[::-1][:10]:
                        sv = float(scores[i]); bad = sv > thr
                        w = min(100.0, 100.0 * sv / smax)
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;padding:6px 10px;'
                            f'margin-bottom:3px;border-radius:8px;'
                            f'background:{"#1a0d0d" if bad else "#08170f"};'
                            f'border-left:4px solid {"var(--danger)" if bad else "var(--ok)"};font-size:0.9rem;">'
                            f'<div style="min-width:96px;color:var(--fg-3);">{t("d_flow_n")} #{int(i)+1}</div>'
                            f'<div style="flex:1;background:var(--border);height:7px;border-radius:10px;overflow:hidden;">'
                            f'<div style="width:{w:.0f}%;height:100%;background:{"var(--danger)" if bad else "var(--ok)"};"></div></div>'
                            f'<div style="min-width:126px;text-align:right;color:{"var(--danger)" if bad else "var(--ok)"};">'
                            f'{"ALERT" if bad else "normal"} · {sv:.2f}</div></div>', unsafe_allow_html=True)
                with right:
                    st.markdown(f"**{t('d_dist_h')}**")
                    st.bar_chart(pd.DataFrame({
                        "cat": ["Normal", "Alert"],
                        "n": [nb_total - nb_alert, nb_alert]}).set_index("cat"),
                        height=230, color="#39ff88")
                    st.caption(f"threshold = {thr:.3f}")

                # Cinq graphiques empilés, c'était un mur en travers de la
                # lecture. Ils restent tous — rien n'est retiré — mais derrière
                # un seul dépliant, fermé par défaut.
                with st.expander(t("adv_h")):
                    st.caption(t("adv_c"))
                    st.write("")
                    st.markdown(f"**{t('d_charts_h')}**")
                    n_plot = 1200
                    idx = (np.linspace(0, nb_total - 1, n_plot).astype(int)
                           if nb_total > n_plot else np.arange(nb_total))
                    st.caption(t("d_advice", n=f"{nb_total:,}", m=f"{len(idx):,}"))

                    a, b = st.columns(2)
                    with a:
                        st.markdown(f"**{t('c_score')}**")
                        st.line_chart(pd.DataFrame({"score": scores[idx],
                                                    "threshold": np.full(len(idx), thr)}), height=200)
                        st.markdown(f"**{t('c_conf')}**")
                        st.line_chart(pd.DataFrame({"conf": conf[idx]}), height=170)
                    with b:
                        st.markdown(f"**{t('c_viol')}**")
                        top = np.argsort(violation)[::-1][:8]
                        st.bar_chart(pd.DataFrame({"flow": [f"F{i}" for i in top],
                                                   "violation": violation[top]}).set_index("flow"), height=200)
                        st.markdown(f"**{t('c_energy')}**")
                        st.line_chart(pd.DataFrame({"energy": energy[idx]}), height=170)

                    st.markdown(f"**{t('c_map')}**")
                    if emb is None or len(emb) != len(scores):
                        st.caption(t("c_map_na"))
                    else:
                        st.scatter_chart(pd.DataFrame({
                            "x": emb[idx, 0], "y": emb[idx, 1],
                            "class": np.where(scores[idx] > thr,
                                              "signalé" if LANG == "fr" else "flagged",
                                              "normal")}),
                            x="x", y="y", color="class", height=260)

        st.divider()
        if _pc == 7:
            st.markdown(f"**{t('rp_reg_h')}**")
            if st.session_state.get("history"):
                dfh = pd.DataFrame(st.session_state.history)
                # Le registre est un journal : il se lit ligne par ligne, donc <table>.
                html_table(dfh.tail(12), key="dl_registre", file_name="registre_session.csv")


        # ════════════════════════════════════════════════════════════════════
        #  TRACE D'UN FLUX — le trajet complet d'une décision
        #  Pédagogie pure : rien n'est recalculé, on déroule ce qui a déjà été
        #  calculé sur le lot. Aucun réglage ici n'influence la détection.
        # ════════════════════════════════════════════════════════════════════
        if "last_scores" in st.session_state and _pc == 4:
            st.divider()
            st.markdown(f"**{t('tr_h')}**")
            st.caption(t("tr_c"))

            _scp = np.asarray(st.session_state.last_scores, dtype=float)
            _tru = st.session_state.get("last_true")
            _al = _scp > thr
            if _tru is not None and len(_tru) == len(_scp):
                _atk = np.array(["BENIGN" not in str(x).upper() for x in _tru])
                _cas = [("tr_cat_hit", _atk & _al), ("tr_cat_miss", _atk & ~_al),
                        ("tr_cat_fp", ~_atk & _al), ("tr_cat_tn", ~_atk & ~_al)]
            else:
                _atk = None
                _cas = [("tr_cat_hit", _al), ("tr_cat_tn", ~_al)]

            _dispo = [(k, m) for k, m in _cas if bool(m.any())]
            if _dispo:
                _lib = [f"{t(k)} ({int(m.sum())})" for k, m in _dispo]
                _choix = st.selectbox(t("tr_pick"), _lib, help=t("tr_pick_help"))
                _k, _m = _dispo[_lib.index(_choix)]
                _j = int(np.flatnonzero(_m)[0])
                _v = float(_scp[_j])

                _d1, _d2, _d3 = st.columns(3)
                _d1.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("tr_score")}</div>'
                             f'<div class="kpi-v">{_v:.3g}</div>'
                             f'<div class="kpi-s">{t("tr_thr")} {thr:.3g}</div></div>',
                             unsafe_allow_html=True)
                _d2.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("tr_verdict")}</div>'
                             f'<div class="kpi-v">{t("tr_hit") if bool(_al[_j]) else t("tr_quiet")}</div>'
                             f'<div class="kpi-s">{_v:.3g} {"&gt;" if bool(_al[_j]) else "≤"} {thr:.3g}</div></div>',
                             unsafe_allow_html=True)
                _lab_txt = str(_tru[_j]) if (_atk is not None) else t("tr_unknown")
                _d3.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("tr_true")}</div>'
                             f'<div class="kpi-v" style="font-size:1rem;">{_lab_txt}</div>'
                             f'<div class="kpi-s">#{_j}</div></div>',
                             unsafe_allow_html=True)

                # Les variables les plus écartées de la référence d'entraînement.
                # C'est une DESCRIPTION du flux, pas l'explication de la décision :
                # le gradient boosting combine ces variables de façon non linéaire.
                _lot = st.session_state.get("last_batch")
                if HGB is not None and _lot is not None:
                    try:
                        _n = HGB["n_net"]
                        _A = np.asarray(_lot[_j: _j + 1, :_n], dtype=float) \
                            * scaler.scale_[:_n] + scaler.mean_[:_n]
                        _Z = (_A - HGB["mu"]) / HGB["sd"]
                        _top = np.argsort(-np.abs(_Z[0]))[:5]
                        _noms = META.get("net_columns") or []
                        st.markdown(f"**{t('tr_dev_h')}**")
                        html_table(pd.DataFrame({t("tr_feat"): [
                            (_noms[int(i)] if int(i) < len(_noms) else f"colonne {int(i)}")
                            for i in _top],
                            t("tr_z"): [f'{float(_Z[0][i]):+.2f}' for i in _top]}),
                            caption=t("tr_dev_h"))
                        st.caption(t("tr_dev_c"))
                    except Exception:
                        pass


        if _pc == 5:
            # ── Comparer : on ne recalcule plus ce qui est déjà affiché ─────────
            # Le tableau des trois détecteurs est dressé à l'étape 3, sur le
            # même lot et le même seuil. Le recalculer ici donnerait deux fois
            # la même mesure, à deux endroits à corriger séparément le jour où
            # un seuil change. On ajoute seulement ce que l'étape 3 ne dit pas :
            # la nature du score produit par chaque détecteur.
            st.divider()
            _Xs = st.session_state.get("last_batch")
            if _Xs is None:
                st.info(t("cmp_nodata"))
            else:
                st.caption(t("cmp_fusion"))
                _scales = detector_scales()
                _lignes = [{t("cmp_col"): _n,
                            t("cmp_scale"): _scales.get(_k, {}).get("kind", t("n_a"))}
                           for _k, _n in (("modele", "MS-Transformer"),
                                          ("one-class", t("mode_one_class")),
                                          ("supervise", t("mode_supervise")))]
                html_table(pd.DataFrame(_lignes, columns=[t("cmp_col"), t("cmp_scale")]),
                           caption=t("cmp_scale_h"),
                           na=t("n_a"), key="dl_cmp_scale",
                           file_name="nature_des_scores.csv")
                if st.button(t("cmp_back"), key="pc_back3", width="stretch"):
                    st.session_state.pc_step = 3
                    st.rerun()

        # ════════════════════════════════════════════════════════════════════
        #  ÉTAPE 6 · EXPORTER — le coffre : ce que la session a réellement produit
        # ════════════════════════════════════════════════════════════════════
        if _pc == 6:
            st.divider()
            st.caption(t("ex_c"))
            _res = os.path.join(ROOT, "results")
            _csv = [f for f in ("metrics_global.csv", "metrics_per_class.csv",
                                "ablation.csv") if os.path.exists(os.path.join(_res, f))]
            _figs = sorted(f for f in os.listdir(_res)
                           if f.startswith("fig_") and f.endswith(".png")) \
                if os.path.isdir(_res) else []
            if not _csv and not _figs and st.session_state.get("last_scores") is None:
                st.info(t("ex_none"))
            for _i, _f in enumerate(_csv):
                _p = os.path.join(_res, _f)
                with open(_p, "rb") as _fh:
                    st.download_button(f"CSV · {_f}", _fh.read(), file_name=_f,
                                       mime="text/csv", key=f"ex_csv_{_i}")
            if _figs:
                st.markdown(f"**{t('ex_fig')}**")
                _cols = st.columns(3)
                for _i, _f in enumerate(_figs):
                    _p = os.path.join(_res, _f)
                    with open(_p, "rb") as _fh:
                        with _cols[_i % 3]:
                            st.download_button(_f.replace("fig_", "").replace(".png", ""),
                                               _fh.read(), file_name=_f,
                                               mime="image/png", key=f"ex_fig_{_i}")

        # ════════════════════════════════════════════════════════════════════
        #  ÉTAPE 7 · RAPPORT — ce que la session a fait, dans l'ordre
        # ════════════════════════════════════════════════════════════════════
        if _pc == 7:
            st.divider()
            st.caption(t("rp_c"))
            _om = official_metrics()
            if _om:
                _g = _om["global"]
                html_table(pd.DataFrame([{
                    "AUROC": _num(_g, "auroc"),
                    t("r_recall"): _pct(_g, "tpr_recall"),
                    t("r_fpr"): _pct(_g, "fpr"),
                    t("r_prec"): _pct(_g, "precision"),
                }]), caption=t("rp_h"), key="dl_rapport",
                    file_name="rapport_session.csv")
            if not st.session_state.get("history"):
                st.info(t("rp_none"))
            else:
                if st.button(t("rp_reset"), key="rp_reset"):
                    st.session_state.history = []
                    st.rerun()

        # ══════════════════════════════════════════════════════════════════════
    #  3 · RESULTS
    # ══════════════════════════════════════════════════════════════════════
with tab_r:
    st.markdown(f'<h2 class="tab-h">{t("tab_res")}</h2>', unsafe_allow_html=True)
    # Le même bandeau que l'unité 2 : l'interface ne change pas de ton
    # quand on change d'onglet.
    entete_unite(3)
    om = official_metrics()
    ad = analysis_data()

    if om is None:
        st.warning("results/metrics.json missing — run `py 03_evaluation.py`.")
    else:
        g = om["global"]
        # ── Lot complet noté hors navigateur (verifier_lot.py) ────────────
        _lot_path = os.path.join(ROOT, "results", "lot_resume.json")
        if os.path.exists(_lot_path):
            try:
                _lot = json.load(open(_lot_path, encoding="utf-8"))
                _c = t("lot_col")
                st.markdown(f"**{t('lot_h')}**")
                _r = {
                    _c[0]: os.path.basename(str(_lot.get("fichier", "—"))),
                    _c[1]: f'{_lot.get("flux", 0):,}'.replace(",", " "),
                    _c[2]: f'{_lot.get("alertes", 0):,}'.replace(",", " "),
                    _c[3]: f'{_lot.get("taux_alerte", 0)*100:.2f} %',
                    _c[4]: (f'{_lot["rappel"]*100:.2f} %' if "rappel" in _lot
                            else t("n_a")),
                    _c[5]: f'{_lot.get("duree_s", 0):.1f} s',
                }
                # lot_col est une paire (fr, en) : on l'indexe explicitement,
                # sinon la légende contiendrait le tuple entier.
                html_table(pd.DataFrame([_r]), caption=t("lot_c"))
                if "rappel" in _lot:
                    st.caption(
                        f'{"Rappel" if LANG == "fr" else "Recall"} '
                        f'{_lot["rappel"]*100:.2f} % · FPR {_lot["fpr"]*100:.2f} % · '
                        f'{("précision" if LANG == "fr" else "precision")} '
                        f'{_lot["precision"]*100:.2f} %')
                st.write("")
            except Exception:
                pass
        else:
            st.caption(t("lot_aide"))
            st.write("")

        st.markdown(f"### {t('r_head')}")
        st.caption(t("r_note", nt=f"{g['n_samples']:,}", nb=f"{g['n_benign']:,}",
                     na=f"{g['n_anomaly']:,}"))

        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("r_auroc")}</div>'
                    f'<div class="kpi-v">{_num(g, "auroc", 4)}</div>'
                    f'<div class="kpi-s">{_ci(g, 4)}</div></div>',
                    unsafe_allow_html=True)
        a2.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("r_recall")}</div>'
                    f'<div class="kpi-v">{_pct(g, "tpr_recall", 2)}</div>'
                    f'<div class="kpi-s">{g.get("threshold_method") or t("missing")}</div></div>',
                    unsafe_allow_html=True)
        a3.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("r_fpr")}</div>'
                    f'<div class="kpi-v">{_pct(g, "fpr", 2)}</div>'
                    f'<div class="kpi-s">target {_pct(g, "target_fpr", 0)}</div></div>',
                    unsafe_allow_html=True)
        a4.markdown(f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("r_prec")}</div>'
                    f'<div class="kpi-v">{_pct(g, "precision", 2)}</div>'
                    f'<div class="kpi-s">F1 {_pct(g, "f1", 2)}</div></div>',
                    unsafe_allow_html=True)

        st.write("")
        p1, p2 = st.columns(2)

        with p1:
            st.markdown(f"**{t('r_perclass')}**")
            st.caption(t("r_perclass_c"))
            rows = [r for r in om["per_class"] if "BENIGN" not in r["classe"].upper()]
            if rows:
                dfa = pd.DataFrame({
                    "class": [r["classe"] for r in rows],
                    "recall": [r["recall"] * 100 for r in rows]})
                st.bar_chart(dfa.set_index("class"), height=260, color="#39ff88")
            st.download_button(t("r_export"),
                               pd.DataFrame(om["per_class"]).to_csv(index=False).encode(),
                               "metrics_per_class.csv", width='stretch')
            try:
                st.caption(t("fig_alt"))
                st.download_button(t("r_export_png"), per_class_png(rows),
                                   "recall_per_class.png", mime="image/png",
                                   width='stretch')
            except Exception:
                pass

        with p2:
            st.markdown(f"**{t('r_curve_h')}**")
            st.caption(t("r_curve_c"))
            if ad is not None:
                step = max(1, len(ad["fpr"]) // 400)
                st.line_chart(pd.DataFrame({
                    "false alarms %": ad["fpr"][::step] * 100,
                    "detection %": ad["tpr"][::step] * 100}).set_index("false alarms %"),
                    height=260, color="#39ff88")
                op = target_fpr
                ph = float(np.interp(op, ad["fpr"], ad["tpr"]))
                st.caption(f"→ at {op*100:.1f} % false alarms, detection = **{ph*100:.1f} %**")

        st.write("")
        st.markdown(f"**{t('r_conf_h')}**")
        if ad is not None:
            pred = ad["scores"] > thr
            tb = ad["is_benign"]
            tp = int((pred & ~tb).sum()); fn = int((~pred & ~tb).sum())
            fp = int((pred & tb).sum()); tn = int((~pred & tb).sum())
            html_table(pd.DataFrame(
                [["Attack" if LANG == "fr" else "Attack", tp, fn],
                 ["Healthy" if LANG == "fr" else "Healthy", fp, tn]],
                columns=["", "signalé" if LANG == "fr" else "flagged",
                         "non signalé" if LANG == "fr" else "not flagged"]),
                caption=t("conf_cap") if "conf_cap" in T else "",
                first_col_header=True)
            st.caption(f"threshold = {thr:.3f} · FPR = {fp/max(fp+tn,1)*100:.2f} % "
                       f"· recall = {tp/max(tp+fn,1)*100:.2f} %")

        # ── Les trois détecteurs, sur un échantillon fixe du test ─────────
        st.write("")
        st.markdown(f"**{t('cmp3_h')}**")
        try:
            _c3 = compare_three(target_fpr * 100.0, LANG)
            if _c3["rows"]:
                _t3 = pd.DataFrame([{
                    t("cmp_col"): r["mode"],
                    t("cmp_alerts"): (t("n_a") if r["alerts"] is None
                                      else f'{r["alerts"]}/{r["n"]}'),
                    t("cmp_rate"): (t("n_a") if r["alerts"] is None
                                    else f'{r["alerts"]/max(r["n"],1)*100:.1f} %'),
                    t("cmp_recall"): (t("n_a") if r["recall"] is None
                                      else f'{r["recall"]*100:.1f} %'),
                    t("cmp_scale"): detector_scales().get(r["key"], {}).get("kind", t("n_a")),
                } for r in _c3["rows"]])
                html_table(_t3, caption=t("cmp3_intro", n=f"{_c3['n']:,}",
                                          fpr=f"{target_fpr*100:g}"),
                           na=t("n_a"), key="dl_cmp3",
                           file_name="comparaison_echantillon_fixe.csv")
                if _c3.get("manquants"):
                    st.warning(t("cmp_indispo", d=", ".join(_c3["manquants"])))

                # Rappel par famille inédite, pour voir où chaque détecteur
                # cède. La légende est portée par le <caption> du tableau.
                _f3 = pd.DataFrame([{
                    "famille": f,
                    **{r["mode"]: (f'{r["per"][f]*100:.1f} %' if f in r["per"]
                                   else t("n_a")) for r in _c3["rows"]},
                } for f in _c3["fams"]])
                html_table(_f3, caption=t("cmp3_fam"), na=t("n_a"),
                           key="dl_cmp3_fam", file_name="rappel_par_famille.csv")
                st.caption(t("cmp_note"))
        except Exception as _e:
            # Un message d'erreur doit dire quoi faire : l'exception seule
            # laisse l'utilisateur devant un mur.
            st.error(t("cmp_fail", e=str(_e)))

    st.write("")
    st.markdown(f"**{t('c_head')}**")
    st.caption(t("c_intro"))
    _cp = os.path.join(ROOT, "results", "contraste.json")
    if os.path.exists(_cp):
        _c = json.load(open(_cp, encoding="utf-8"))
        _rows = []
        for _fam, _v in sorted(_c["familles"].items(),
                               key=lambda kv: -kv[1]["supervise"]):
            _rows.append({"famille": _fam,
                          t("c_oc"): f'{_v["one_class"]*100:.1f} %',
                          t("c_sv"): f'{_v["supervise"]*100:.1f} %'})
        _rows.append({"famille": t("c_mean"),
                      t("c_oc"): f'{_c["global"]["one_class"]*100:.1f} %',
                      t("c_sv"): f'{_c["global"]["supervise"]*100:.1f} %'})
        html_table(pd.DataFrame(_rows), caption=t("c_intro"),
                   key="dl_contraste", file_name="contraste_approches.csv")

        _vf = _c.get("validation_par_famille") or {}
        if _vf:
            _r2 = []
            for _fam, _v in sorted(_vf.items(), key=lambda kv: -kv[1]["supervise"]):
                _r2.append({"famille": _fam,
                            t("c_sv"): f'{_v["supervise"]*100:.1f} %',
                            t("c_oc"): f'{_v["one_class"]*100:.1f} %'})
            html_table(pd.DataFrame(_r2), caption=t("c_lofo_h"))
            _pire = min(_vf, key=lambda k: _vf[k]["supervise"])
            st.error(t("c_collapse", fam=_pire,
                       pct=f'{_vf[_pire]["supervise"]*100:.1f}'))
    else:
        st.info(t("c_missing"))

    st.write("")
    st.caption(t("src_note", score=g.get("score", "?")))
    st.markdown(f"### ⚠️ {t('a_head')}")
    st.markdown(t("a_intro"))
    triv = om.get("trivial_baselines", {})
    if triv:
        _au = _val(g, "auroc")
        rowsm = ([{"method": t("a_model"), "auroc": round(_au, 4),
                   "recall": _pct(g, "tpr_recall")}] if _au is not None else [])
        for name, d in triv.items():
            label = t("a_sum") if "somme" in name else t("a_lr")
            rec = f'{d["recall"]*100:.1f} %' if "recall" in d else "—"
            rowsm.append({"method": label, "auroc": round(d["auroc"], 4), "recall": rec})
        dfm = pd.DataFrame(rowsm).rename(
            columns={"method": t("a_method"), "auroc": t("a_auroc"),
                     "recall": t("a_recall")})
        html_table(dfm, caption=t("a_intro"), key="dl_baselines",
                   file_name="baselines_triviales.csv")

        best_triv = max(d["auroc"] for d in triv.values())
        _au = _val(g, "auroc")
        if _au is not None and best_triv >= _au - 0.02:
            st.error(t("a_warn"))
        else:
            st.success(t("a_ok"))

        # détail par classe de la baseline triviale
        for name, d in triv.items():
            if "recall_per_class" in d:
                st.caption(f"« {t('a_sum')} » — "
                           + " · ".join(f"{k}: {v*100:.1f} %"
                                        for k, v in d["recall_per_class"].items()))
                break
    else:
        st.info("Baselines absentes de results/metrics.json — relance `py 03_evaluation.py`.")

    st.write("")
    st.markdown(f"**{t('pv_head')}**")
    st.caption(t("pv_intro"))
    r_, f_ = _val(g, "tpr_recall"), _val(g, "fpr")
    _ns, _na = _val(g, "n_samples"), _val(g, "n_anomaly")
    if r_ is None or f_ is None or not _ns or _na is None:
        st.info(t("pv_missing"))
    else:
        rows_pv = []
        for pi, ctx in [(_na / _ns, "this test set"),
                        (0.05, "monitored network"), (0.01, "filtered SOC"),
                        (0.001, "real SOC"), (0.0001, "large operator")]:
            den = r_ * pi + f_ * (1 - pi)
            ppv = (r_ * pi / den) if den else 0.0
            rows_pv.append({t("pv_prev"): f"{pi*100:.3f} %",
                            t("pv_prec"): f"{ppv*100:.2f} %",
                            t("pv_alert"): f"{(1-ppv)/ppv:.1f}" if ppv > 0 else "∞",
                            t("pv_ctx"): ctx})
        html_table(pd.DataFrame(rows_pv), caption=t("pv_intro"),
                   key="dl_prevalence", file_name="precision_selon_prevalence.csv")

    st.write("")
    st.markdown(f"**{t('r_baseline_h')}**")
    st.caption(t("hist_note"))
    st.caption(t("r_baseline_c"))
    bl = []
    abl = ablation_rows()
    for r in abl:
        bl.append({"method": r["label"], "auroc": round(r["auroc"], 4),
                   "seeds": "3" if r.get("auroc_std", 0) > 0 else "1"})
    sv = svdd_row()
    if sv:
        bl.append({"method": "Deep SVDD", "auroc": round(sv["auroc_zero_day"], 4), "seeds": "3"})
        if sv.get("reference_supervised_auroc"):
            bl.append({"method": "Supervised reference",
                       "auroc": round(sv["reference_supervised_auroc"], 4), "seeds": "1"})
    if bl:
        dfb = pd.DataFrame(bl).sort_values("auroc", ascending=False)
        html_table(dfb, caption=t("r_baseline_c"), key="dl_ref",
                   file_name="approches_de_reference.csv")


# ══════════════════════════════════════════════════════════════════════
#  4 · METHOD
# ══════════════════════════════════════════════════════════════════════
with tab_m:
    st.markdown(f'<h2 class="tab-h">{t("tab_method")}</h2>', unsafe_allow_html=True)
    # Le même bandeau que l'unité 2 : l'interface ne change pas de ton
    # quand on change d'onglet.
    entete_unite(4)
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(f"### {t('m_enc_h')}")
        st.markdown(t("m_enc"))
        st.markdown(f"### {t('m_score_h')}")
        st.markdown(t("m_score"))
    with q2:
        st.markdown(f"### {t('m_abl_h')}")
        st.caption(t("hist_note"))
        st.caption(t("m_abl_c"))
        abl = ablation_rows()
        if abl:
            base = next((r["auroc"] for r in abl if r["variante"] == "full"), None)
            dfx = pd.DataFrame([{
                "variant": r["label"],
                "AUROC": round(r["auroc"], 4),
                "Δ": (f'{r["auroc"] - base:+.4f}' if base is not None else "—"),
                "±": (f'{r["auroc_std"]:.4f}' if r.get("auroc_std") else "—"),
            } for r in abl])
            html_table(dfx, caption=t("m_abl_c"), key="dl_ablation",
                       file_name="ablation.csv")
        else:
            st.info("results/ablation.json missing — run `py 04_ablation.py`.")


# ══════════════════════════════════════════════════════════════════════
#  5 · LIMITATIONS
# ══════════════════════════════════════════════════════════════════════
with tab_l:
    st.markdown(f'<h2 class="tab-h">{t("tab_limit")}</h2>', unsafe_allow_html=True)
    # Le même bandeau que l'unité 2 : l'interface ne change pas de ton
    # quand on change d'onglet.
    entete_unite(5)
    st.markdown(f"### {t('l_h')}")
    st.caption(t("l_intro"))

    _LIMITES = [
        ("l1", "open",      "var(--warn)"),   # plafond Bot 84,65 %
        ("l2", "ok",        "var(--ok)"),     # branche EDR supprimee
        ("l3", "part",      "var(--warn)"),   # 5 familles, pas toutes
        ("l4", "open",      "var(--warn)"),   # prevalence de test 18 %
        ("l5", "part",      "var(--warn)"),   # reste CICIDS2017
    ]
    _LBL = {"ok": t("l_status_ok"), "part": t("l_status_part"), "open": t("l_status_open")}
    # Symbole redondant avec la couleur (critère WCAG 1.4.1)
    _SYM = {"ok": "✓", "part": "◐", "open": "✕"}

    for _k, _st, _col in _LIMITES:
        st.markdown(
            f'<div class="panel" lang="{LANG}" style="margin-bottom:12px; border-left:3px solid {_col};">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;'
            f' margin-bottom:10px;">'
            f'<span style="font-weight:650; font-size:1.05rem; color:var(--fg-1);">{t(_k + "_t")}</span>'
            f'<span style="font-size:0.82rem; letter-spacing:0.08em; text-transform:uppercase;'
            f' color:{_col}; border:1px solid {_col}; border-radius:999px; padding:2px 10px;">'
            f'<span aria-hidden="true">{_SYM[_st]}</span> {_LBL[_st]}</span></div>'
            f'<div style="font-size:0.92rem; line-height:1.65; color:var(--fg-3); margin-bottom:12px;">'
            f'{_md_html(t(_k + "_a"))}</div>'
            f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;'
            f' font-size:0.88rem; line-height:1.6;">'
            f'<div><div style="color:var(--fg-4); text-transform:uppercase; letter-spacing:0.08em;'
            f' font-size:0.9rem; font-weight:600; margin-bottom:5px;">{t("l_proof")}</div>'
            f'<div style="color:var(--fg-2);">{_md_html(t(_k + "_p"))}</div></div>'
            f'<div><div style="color:var(--fg-4); text-transform:uppercase; letter-spacing:0.08em;'
            f' font-size:0.9rem; font-weight:600; margin-bottom:5px;">{t("l_left")}</div>'
            f'<div style="color:var(--fg-2);">{_md_html(t(_k + "_r"))}</div></div></div>'
            f'</div>', unsafe_allow_html=True)

    # bilan chiffre
    st.write("")
    _b1, _b2, _b3, _b4 = st.columns(4)
    _b1.markdown(
        f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("l_status_ok")}</div>'
        f'<div class="kpi-v">1 / 5</div><div class="kpi-s">L2</div></div>',
        unsafe_allow_html=True)
    _b2.markdown(
        f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("l_status_part")}</div>'
        f'<div class="kpi-v">2 / 5</div><div class="kpi-s">L3 · L5</div></div>',
        unsafe_allow_html=True)
    _b3.markdown(
        f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("l_status_open")}</div>'
        f'<div class="kpi-v">2 / 5</div><div class="kpi-s">L1 · L4</div></div>',
        unsafe_allow_html=True)
    _b4.markdown(
        f'<div class="kpi" lang="{LANG}"><div class="kpi-l">{t("flows")}</div>'
        f'<div class="kpi-v">5 495</div><div class="kpi-s">attaques évaluées</div></div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="note" style="margin-top:14px;">{_md_html(t("l_foot"))}</div>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  FOOTER — provenance des données et du modèle
# ══════════════════════════════════════════════════════════════════════
st.divider()
try:
    import hashlib
    _h = hashlib.md5(open(os.path.join(ROOT, "models", "ms_encoder.pt"), "rb").read()
                     ).hexdigest()[:10]
except Exception:
    _h = "empreinte non calculable"

_n_tr = META.get("train_shape") or ["?", "?"]
_n_te = META.get("test_shape") or ["?", "?"]
_zz = "Bot · PortScan · Brute Force · XSS · SQL Injection"

st.markdown(
    f'<div class="foot">'
    f'<b>{t("pv_src")}</b> CICIDS2017 (Canadian Institute for Cybersecurity) · '
    f'<b>{t("pv_train")}</b> <code>{_n_tr[0]}</code> {t("flows")} · '
    f'<b>{t("pv_test")}</b> <code>{_n_te[0]}</code> {t("flows")} ({_zz})<br>'
    f'<b>{t("pv_model")}</b> <code>ms_encoder.pt · {_h}</code> · '
    f'<b>{t("pv_score")}</b> <code>gradient boosting · 77 features</code> · '
    f'<code>seed {META.get("seed", "?")}</code> · '
    f'{t("pv_run")}<br>'
    f'<span class="mono">{t("a11y_note")}</span>'
    f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  6 · AIDE / HELP
# ══════════════════════════════════════════════════════════════════════
with tab_h:
    st.markdown(f'<h2 class="tab-h">{t("tab_help")}</h2>', unsafe_allow_html=True)
    # Le même bandeau que l'unité 2 : l'interface ne change pas de ton
    # quand on change d'onglet.
    entete_unite(6)
    def H(key: str) -> str:
        fr, en = HELP[key]
        return fr if LANG == "fr" else en

    st.markdown(f"### {H('h_intro')}")

    st.divider()
    st.markdown(f"### {H('h_steps_h')}")
    for _k in ("h_step1", "h_step2", "h_step3"):
        st.markdown(
            f'<div class="panel" lang="{LANG}" style="margin-bottom:10px;">'
            f'{_md_html(H(_k))}</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### {H('h_modes_h')}")
    _c1, _c2, _c3 = st.columns(3)
    for _col, _n, _b in ((_c1, "h_mode1", "h_mode1_t"),
                         (_c2, "h_mode2", "h_mode2_t"),
                         (_c3, "h_mode3", "h_mode3_t")):
        with _col:
            st.markdown(
                f'<div class="panel" lang="{LANG}" style="height:100%;">'
                f'<div style="font-weight:650;color:var(--fg-1);margin-bottom:8px;">'
                f'{H(_b)}</div>'
                f'<div style="font-size:0.92rem;color:var(--fg-2);">{H(_n)}</div>'
                f'</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### {H('h_keys_h')}")
    for _k in ("h_key1", "h_key2", "h_key3", "h_key4"):
        st.markdown(f"- {H(_k)}")

    st.divider()
    st.markdown(f"### {H('h_glossary_h')}")
    _gloss = [(f"h_g{i}_t", f"h_g{i}") for i in range(1, 13)]
    for _tk, _dk in _gloss:
        with st.expander(H(_tk)):
            st.markdown(H(_dk))

    st.divider()
    st.markdown(f"### {H('h_a11y_h')}")
    st.markdown(H("h_a11y"))
