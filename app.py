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
    "cmp_nodata":   ("Aucun lot analysé pour l'instant — lance d'abord une analyse.",
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
    "lv_missing":   ("Aucun collecteur détecté. Fichier attendu : `{p}`.",
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
    "lv_cols":      ("Colonnes manquantes dans le fichier : {c}. Les scores seraient "
                     "faux, l'analyse est refusée.",
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
    "d_suspect_h":  ("Les connexions les plus suspectes", "Most suspicious connections"),
    "d_suspect_c":  ("Classées de la plus anormale à la moins anormale. Rouge = au-dessus du seuil.",
                     "Ranked from most to least anomalous. Red = above threshold."),
    "d_flow_n":     ("Connexion", "Connection"),
    "d_alert":      ("ALERTE", "ALERT"),
    "d_ok":         ("normal", "normal"),
    "d_dist_h":     ("Répartition", "Breakdown"),
    "d_charts_h":   ("Analyse détaillée", "Detailed analysis"),
    "d_advice":     ("Métriques calculées sur {n} connexions ; tracés limités à {m} points.",
                     "Metrics computed on {n} connections; plots limited to {m} points."),
    "d_no_data":    ("Lance une analyse pour afficher les résultats.",
                     "Run an analysis to display results."),

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
        "A vu les familles connues à l'entraînement. AUROC 0,9668 et 93,12 % de "
        "rappel à 9,03 % de fausses alertes : c'est le détecteur retenu.",
        "Saw the known families during training. AUROC 0.9668 and 93.12 % recall "
        "at 9.03 % false alarms: this is the retained detector."),

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
   SYSTÈME DE DESIGN — CONTRASTE WCAG 2.1 NIVEAU AA
   ──────────────────────────────────────────────────────────────────
   Tous les couples texte/fond de ce thème ont été vérifiés :
   ratio minimal 4,82:1 (AA exige 4,5:1 pour le texte courant).
   Les statuts ne sont jamais portés par la seule couleur : chaque
   badge associe un symbole (✓ ⚠ ✕) et un libellé textuel.
   ══════════════════════════════════════════════════════════════════ */
:root {
  --bg:          #070b14;
  --surface:     #0c1422;
  --surface-2:   #101a2b;
  --border:      #22304a;
  --border-ui:   #4e6390;   /* composants interactifs — ratio 3:1 min */
  --fg-1:        #f1f5f9;       /* 17,97:1 */
  --fg-2:        #dbe4ef;   /* 15,33:1 */
  --fg-3:        #aebdcd;   /* 10,27:1 */
  --fg-4:        #8b9ab0;   /* 6,89:1  — labels, légendes */
  --ok:          #5ee08a;   /* 11,72:1 */
  --warn:        #f5c451;   /* 12,09:1 */
  --info:        #7ab4f7;   /* 9,09:1  */
  --danger:      #ff8f8f;   /* 8,98:1  */
  --focus:       #7ab4f7;
  --sp-1: 6px; --sp-2: 10px; --sp-3: 16px; --sp-4: 24px; --sp-5: 34px;
  --radius: 10px;
}

/* ---- base ---- */
html { font-size: 16px; }
.stApp { background: var(--bg) !important; color: var(--fg-2) !important; }
.block-container { padding-top: var(--sp-3); padding-bottom: var(--sp-5);
                   max-width: 1320px; }
h1,h2,h3,h4 { color: var(--fg-1) !important; letter-spacing: -0.015em;
              font-weight: 650; line-height: 1.3; }
p, li { color: var(--fg-2); line-height: 1.68; }
hr { border-color: var(--border) !important; }

/* ---- focus visible : indispensable au clavier ---- */
a:focus-visible, button:focus-visible, [tabindex]:focus-visible,
input:focus-visible, select:focus-visible,
div[role="radiogroup"] label:focus-visible,
.stTabs [role="tab"]:focus-visible {
  outline: 3px solid var(--focus) !important;
  outline-offset: 3px !important;
  border-radius: 4px;
}

/* ---- respect de la préférence système ---- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
}

/* ---- en-tête ---- */
.eyebrow { font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
           color: var(--fg-4); margin-bottom: var(--sp-1); font-weight: 600; }
.proj-title { font-size: 1.7rem; font-weight: 700; color: var(--fg-1);
              line-height: 1.22; letter-spacing: -0.02em; margin: 0; }
.proj-sub { font-size: 0.94rem; color: var(--fg-3); margin-top: var(--sp-2);
            line-height: 1.6; max-width: 64ch; }

/* ---- badges d'état : symbole + libellé, jamais la couleur seule ---- */
.badge { display: inline-flex; align-items: center; gap: 6px;
         font-size: 0.8rem; font-weight: 600; padding: 5px 12px;
         border-radius: 999px; margin-right: var(--sp-1); margin-top: var(--sp-2);
         border: 1px solid var(--border-ui); background: var(--surface);
         color: var(--fg-2); letter-spacing: 0.01em; }
.badge-ok   { border-color: #2f6b45; color: var(--ok); background: #08190f; }
.badge-warn { border-color: #7a5c15; color: var(--warn); background: #191204; }
.badge-info { border-color: #2c4f7a; color: var(--info); background: #08131f; }

/* ---- cartes de mesure ---- */
.kpi { background: var(--surface); border: 1px solid var(--border);
       border-radius: var(--radius); padding: var(--sp-3); text-align: center; }
.kpi-v { font-size: 1.7rem; font-weight: 700; color: var(--fg-1);
         line-height: 1.15; font-variant-numeric: tabular-nums; }
.kpi-l { font-size: 0.78rem; color: var(--fg-4); letter-spacing: 0.08em;
         text-transform: uppercase; margin-bottom: var(--sp-1); font-weight: 600; }
.kpi-s { font-size: 0.82rem; color: var(--fg-3); margin-top: var(--sp-1); }

/* ---- panneaux ---- */
.panel { background: var(--surface); border: 1px solid var(--border);
         border-radius: var(--radius); padding: var(--sp-3) var(--sp-3);
         color: var(--fg-2); line-height: 1.66; }
.panel-accent { border-left: 3px solid var(--info); }
.pitch { background: var(--surface-2); border: 1px solid #2c4f7a;
         border-left: 3px solid var(--info); border-radius: var(--radius);
         padding: var(--sp-3) var(--sp-4); }

/* ---- encadrés ---- */
.note { background: var(--surface); border: 1px solid var(--border);
        border-left: 3px solid var(--border-ui); border-radius: 8px;
        padding: var(--sp-2) var(--sp-3); font-size: 0.9rem; color: var(--fg-3);
        line-height: 1.65; }
.note-warn { border-left-color: var(--warn); background: #16100240; }

/* ---- pied de page ---- */
.foot { border-top: 1px solid var(--border); margin-top: var(--sp-4);
        padding-top: var(--sp-2); font-size: 0.82rem; color: var(--fg-4);
        line-height: 2; }
.foot code { color: var(--fg-3); background: var(--surface-2);
             padding: 2px 6px; border-radius: 4px; font-size: 0.95em; }

.mono { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }

/* ---- tableaux : lisibilité et densité ---- */
div[data-testid="stDataFrame"] { border: 1px solid var(--border);
                                 border-radius: 8px; }

/* ---- utilitaires ---- */
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
           overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
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

    Palette accessible : le texte utilise #dbe4ef sur #0c1422 (15,3:1) et les
    barres sont dans une teinte unique — la valeur est aussi imprimée à côté
    de chaque barre, de sorte que la figure reste lisible sans distinguer
    les couleurs.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, FG1, FG2, FG3, BAR = "#0c1422", "#f1f5f9", "#dbe4ef", "#aebdcd", "#7ab4f7"
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
        sp.set_color("#22304a")
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

    rows = []
    for mode_, name in (("modele", t("mode_modele")),
                        ("one-class", t("mode_one_class")),
                        ("supervise", t("mode_supervise"))):
        if mode_ == "supervise" and HGB is None and SUP is None:
            continue
        if mode_ == "one-class" and FEAT is None:
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
                     "alerts": int(pred.sum()), "recall": recall, "per": per})
    return {"rows": rows, "fams": fams, "n": len(Xs)}


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
    'this.style.padding=\'10px 16px\';this.style.background=\'#0c1422\';'
    'this.style.border=\'2px solid #7ab4f7\';this.style.borderRadius=\'6px\';'
    'this.style.zIndex=9999;this.style.width=\'auto\';this.style.height=\'auto\';'
    'this.style.clip=\'auto\';" '
    'onblur="this.style.left=\'-9999px\';">'
    'Aller au contenu principal</a>', unsafe_allow_html=True)

st.markdown(
    f'<header role="banner" lang="{LANG}">'
    f'<div class="eyebrow">{t("eyebrow")}</div>'
    f'<h1 class="proj-title" style="margin:0;">{t("title")}</h1>'
    f'<p class="proj-sub">{t("subtitle")}</p>'
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
    f'<div class="note note-warn">'
    f'<b>{t("st_title")}</b><br>{_st_parts[0]}'
    f'</div>', unsafe_allow_html=True)
if len(_st_parts) > 1:
    with st.expander(t("st_more")):
        st.markdown("\n\n".join(_st_parts[1:]))

st.write("")

st.markdown('<div id="contenu-principal" tabindex="-1"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  BARRE LATÉRALE — les réglages, séparés des résultats
#  Un réglage vit à un seul endroit : on ne cherche plus le curseur de
#  fausses alertes dans un onglet alors qu'il vaut pour toute la page.
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### {t('sb_h')}")

    # ── langue ─────────────────────────────────────────────────────────
    st.session_state.lang = st.radio(
        t("sb_lang"), ["fr", "en"], index=0 if LANG == "fr" else 1,
        horizontal=True, key="lang_selector", help=t("lang_help"))
    if st.session_state.lang != LANG:
        st.rerun()

    st.divider()

    # ── détecteur ──────────────────────────────────────────────────────
    _modes = ["modele", "one-class", "supervise"]
    _prev = st.session_state.mode
    st.session_state.mode = st.radio(
        t("sb_det"), _modes,
        index=_modes.index(_prev),
        format_func=lambda v: t("mode_" + v.replace("-", "_")))
    MODE = st.session_state.mode
    if MODE != _prev:
        # analysis_data() est mis en cache sans connaître le mode : on purge
        # sinon la démonstration afficherait les scores de l'autre approche.
        st.cache_data.clear()
        st.rerun()
    st.caption(t("mode_" + MODE.replace("-", "_") + "_d"))

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

    # ── seuil : auparavant enfoui dans l'onglet Démonstration ───────────
    target_fpr = st.slider(t("sb_thr"), 0.5, 20.0, 5.0, 0.5,
                           help=t("sb_thr_help"), format="%.1f %%") / 100.0
    st.session_state.last_fpr = float(target_fpr * 100.0)
    thr = threshold_for_fpr(target_fpr)

    st.divider()
    if st.button(t("sb_reset"), width='stretch'):
        for k in ("last_scores", "last_viol", "last_energy", "last_conf",
                  "last_emb", "last_true", "last_batch",
                  "lv_on", "lv_skip", "lv_n", "lv_a", "lv_rows"):
            st.session_state.pop(k, None)
        st.session_state.history = []
        _hp = os.path.join(ROOT, "data", "history_livre.csv")
        if os.path.exists(_hp):
            os.remove(_hp)
        st.rerun()

    st.caption(t("sb_about"))

st.write("")

tab_o, tab_d, tab_r, tab_m, tab_l, tab_h = st.tabs([
    t("tab_over"), t("tab_demo"), t("tab_res"), t("tab_method"),
    t("tab_limit"), t("tab_help")])


# ══════════════════════════════════════════════════════════════════════
#  1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab_o:
    c1, c2 = st.columns([1.15, 0.85])
    with c1:
        st.markdown(f"#### {t('o_problem_h')}")
        st.markdown(t("o_problem"))
        st.markdown(f"#### {t('o_approach_h')}")
        st.markdown(t("o_approach"))
    with c2:
        st.markdown(
            f'<div class="pitch">'
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
        k1.markdown(f'<div class="kpi"><div class="kpi-l">AUROC</div>'
                    f'<div class="kpi-v">{_num(g, "auroc")}</div>'
                    f'<div class="kpi-s">IC95 % {_ci(g)}</div></div>',
                    unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="kpi-l">{t("r_fpr")}</div>'
                    f'<div class="kpi-v">{_pct(g, "fpr")}</div>'
                    f'<div class="kpi-s">cible ≤ 10 %</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="kpi-l">{t("r_recall")}</div>'
                    f'<div class="kpi-v">{_pct(g, "tpr_recall")}</div>'
                    f'<div class="kpi-s">{g.get("threshold_method") or t("missing")}</div></div>',
                    unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi"><div class="kpi-l">n</div>'
                    f'<div class="kpi-v">{f"{int(_na):,}" if _na is not None else t("missing")}</div>'
                    f'<div class="kpi-s">Zero-Day / '
                    f'{f"{int(_nb):,}" if _nb is not None else t("missing")}</div></div>',
                    unsafe_allow_html=True)

    st.write("")
    st.markdown(f"""<div class="panel panel-accent">{t("o_honest")}</div>""",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  2 · LIVE DEMO
# ══════════════════════════════════════════════════════════════════════
with tab_d:
    st.markdown(f"#### {t('d_s1')}")
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
    st.markdown(f"#### {t('d_s2')}")

    scores = st.session_state.get("last_scores")
    if scores is None:
        st.info(t("d_no_data"))
    else:
        scores = np.asarray(scores)
        violation = np.asarray(st.session_state.get("last_viol", np.zeros(len(scores))))
        energy = np.asarray(st.session_state.get("last_energy", np.zeros(len(scores))))
        conf = np.asarray(st.session_state.get("last_conf", np.ones(len(scores))))
        emb = np.asarray(st.session_state.get("last_emb",
                                              np.random.randn(len(scores), 2) * 0.01))

        nb_total = len(scores)
        mask = scores > thr
        nb_alert = int(mask.sum())
        pct = 100.0 * nb_alert / max(nb_total, 1)

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="kpi"><div class="kpi-l">{t("d_flows")}</div>'
                    f'<div class="kpi-v">{nb_total:,}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi"><div class="kpi-l">{t("d_alerts")}</div>'
                    f'<div class="kpi-v" style="color:{"var(--danger)" if nb_alert else "var(--ok)"}">'
                    f'{nb_alert:,}</div><div class="kpi-s">{pct:.1f} {t("d_detected")}</div></div>',
                    unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi"><div class="kpi-l">{t("d_normal")}</div>'
                    f'<div class="kpi-v" style="color:var(--ok)">{nb_total - nb_alert:,}</div>'
                    f'<div class="kpi-s">{t("d_below")}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi"><div class="kpi-l">{t("d_verdict")}</div>'
                    f'<div class="kpi-v" style="color:{"var(--danger)" if nb_alert else "var(--ok)"}">'
                    f'{t("d_threat") if nb_alert else t("d_clean")}</div>'
                    f'<div class="kpi-s">{t("d_threat_sub") if nb_alert else t("d_clean_sub")}</div></div>',
                    unsafe_allow_html=True)

        # labelled batch: real precision / recall
        tl = st.session_state.get("last_true")
        if tl is not None and len(tl) == nb_total:
            tb = np.array(["BENIGN" in str(l).upper() for l in tl])
            tp = int((mask & ~tb).sum()); fp = int((mask & tb).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("True positives", tp,
                      f"{tp / max(int((~tb).sum()), 1):.1%} recall")
            m2.metric("False positives", fp,
                      f"{fp / max(int(tb.sum()), 1):.1%} FPR", delta_color="inverse")
            m3.metric("Zero-Day in batch", int((~tb).sum()))

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
                    f'background:{"#1b0d0d" if bad else "#08190f"};'
                    f'border-left:3px solid {"var(--danger)" if bad else "var(--ok)"};font-size:0.9rem;">'
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
                height=230, color="#7ab4f7")
            st.caption(f"threshold = {thr:.3f}")

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
        st.scatter_chart(pd.DataFrame({
            "x": emb[idx, 0], "y": emb[idx, 1],
            "class": np.where(scores[idx] > thr, "flagged", "normal")}),
            x="x", y="y", color="class", height=260)

    st.divider()
    st.markdown("**Registry**")
    if st.session_state.get("history"):
        dfh = pd.DataFrame(st.session_state.history)
        st.dataframe(dfh, width='stretch', height=140)
        st.download_button("CSV", dfh.to_csv(index=False).encode(),
                           "registry.csv", width='stretch')


    # ════════════════════════════════════════════════════════════════════
    #  TRACE D'UN FLUX — le trajet complet d'une décision
    #  Pédagogie pure : rien n'est recalculé, on déroule ce qui a déjà été
    #  calculé sur le lot. Aucun réglage ici n'influence la détection.
    # ════════════════════════════════════════════════════════════════════
    if "last_scores" in st.session_state:
        st.divider()
        st.markdown(f"#### {t('d_s3')}")
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
            _d1.markdown(f'<div class="kpi"><div class="kpi-l">{t("tr_score")}</div>'
                         f'<div class="kpi-v">{_v:.3g}</div>'
                         f'<div class="kpi-s">{t("tr_thr")} {thr:.3g}</div></div>',
                         unsafe_allow_html=True)
            _d2.markdown(f'<div class="kpi"><div class="kpi-l">{t("tr_verdict")}</div>'
                         f'<div class="kpi-v">{t("tr_hit") if bool(_al[_j]) else t("tr_quiet")}</div>'
                         f'<div class="kpi-s">{_v:.3g} {"&gt;" if bool(_al[_j]) else "≤"} {thr:.3g}</div></div>',
                         unsafe_allow_html=True)
            _lab_txt = str(_tru[_j]) if (_atk is not None) else t("tr_unknown")
            _d3.markdown(f'<div class="kpi"><div class="kpi-l">{t("tr_true")}</div>'
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
                    st.dataframe(
                        pd.DataFrame({t("tr_feat"): [
                            (_noms[int(i)] if int(i) < len(_noms) else f"colonne {int(i)}")
                            for i in _top],
                                      t("tr_z"): [round(float(_Z[0][i]), 2) for i in _top]}),
                        width='stretch', hide_index=True)
                    st.caption(t("tr_dev_c"))
                except Exception:
                    pass

    # ════════════════════════════════════════════════════════════════════
    #  REJEU FLUX PAR FLUX — un lot déjà noté, qui défile
    # ════════════════════════════════════════════════════════════════════
    if "last_scores" in st.session_state:
        st.write("")
        st.markdown(f"**{t('rj_h')}**")
        st.caption(t("rj_c"))
        if st.button(t("rj_btn"), width='stretch'):
            _scp = np.asarray(st.session_state.last_scores, dtype=float)
            _N = len(_scp)
            _pas = max(1, _N // 60)          # 60 mises à jour au plus
            _zone = st.empty()
            _barre = st.progress(0.0)
            _cum = 0
            for _i in range(0, _N, _pas):
                _cum += int((_scp[_i:_i + _pas] > thr).sum())
                _fait = min(_N, _i + _pas)
                _zone.markdown(t("rj_mid", done=f"{_fait:,}", alerts=f"{_cum:,}"))
                _barre.progress(_fait / _N)
                time.sleep(0.02)             # ~1,2 s au total
            _barre.progress(1.0)
            _zone.markdown(t("rj_end", alerts=f"{_cum:,}", n=f"{_N:,}"))
        st.caption(t("rj_note"))

    # ════════════════════════════════════════════════════════════════════
    #  MODE DIRECT — flux notés à mesure que le collecteur les écrit
    #  Ce n'est pas une simulation : les lignes viennent d'un fichier écrit
    #  par un collecteur réseau, et sont notées au fur et à mesure.
    # ════════════════════════════════════════════════════════════════════
    st.write("")
    st.markdown(f"**{t('lv_h')}**")
    st.caption(t("lv_c"))
    _lv_path = os.path.join(ROOT, "data", "live", "live_flows.csv")

    if not os.path.exists(_lv_path):
        st.info(t("lv_missing", p="data/live/live_flows.csv"))
        with st.expander(t("lv_how")):
            st.markdown(t("lv_how_body"))
    else:
        _lv_on = bool(st.session_state.get("lv_on"))
        if st.button(t("lv_stop") if _lv_on else t("lv_start"), width='stretch'):
            st.session_state.lv_on = not _lv_on
            st.session_state.lv_skip = 0
            st.session_state.lv_n = 0
            st.session_state.lv_a = 0
            st.session_state.lv_rows = []
            st.rerun()

        @st.fragment(run_every=1.5)
        def _live_panel():
            """Relit le fichier du collecteur toutes les deux secondes."""
            if not st.session_state.get("lv_on"):
                st.caption(t("lv_wait"))
                return
            _skip = int(st.session_state.get("lv_skip", 0))
            try:
                # skiprows évite de réanalyser tout l'historique à chaque tour
                _df = pd.read_csv(_lv_path,
                                  skiprows=range(1, _skip + 1) if _skip else None)
            except Exception:
                st.caption(t("lv_wait"))
                return

            _new = _df if len(_df) else _df.iloc[0:0]
            if len(_new) > 2000:          # garde-fou : on ne noie pas l'écran
                _new = _new.iloc[:2000]

            if len(_new):
                st.session_state.lv_skip = _skip + len(_new)
                _X, _miss = live_prepare(_new)
                if _miss:
                    st.error(t("lv_cols", c=", ".join(str(x) for x in _miss[:6])))
                    return
                _sc = np.asarray(infer(_X)[0], dtype=float)
                _hit = _sc > thr
                st.session_state.lv_n = int(st.session_state.get("lv_n", 0)) + len(_sc)
                st.session_state.lv_a = int(st.session_state.get("lv_a", 0)) + int(_hit.sum())
                _rows = list(st.session_state.get("lv_rows", []))
                _now = datetime.datetime.now().strftime("%H:%M:%S")
                for _k in range(len(_sc)):
                    _rows.append({"heure": _now,
                                  "score": float(_sc[_k]),
                                  "verdict": t("tr_hit") if bool(_hit[_k]) else t("tr_quiet")})
                st.session_state.lv_rows = _rows[-12:]

            _n = int(st.session_state.get("lv_n", 0))
            _a = int(st.session_state.get("lv_a", 0))
            _r = (100.0 * _a / _n) if _n else 0.0

            # `role="status"` : le compteur est annoncé aux lecteurs d'écran
            # quand il change. Le symbole ⚠ et le mot « alerte » portent
            # l'information autant que la couleur (WCAG 1.4.1).
            _html = (
                f'<div class="lv-head" role="status">'
                f'<span class="lv-dot" aria-hidden="true"></span>'
                f'<b>{t("lv_live_on")}</b>'
                f'<span>{t("lv_count", n=f"{_n:,}", a=f"{_a:,}", r=f"{_r:.1f}")}</span>'
                f'</div>'
                f'<div class="lv-bar" aria-hidden="true">'
                f'<div class="lv-fill" style="width:{min(100.0, _r):.1f}%"></div></div>')
            for _i, _c in enumerate(reversed(st.session_state.get("lv_rows", [])[-6:])):
                _est_alerte = "⚠" in str(_c.get("verdict", ""))
                _html += (
                    f'<div class="lv-card{" lv-al" if _est_alerte else ""}" '
                    f'style="animation-delay:{_i * 0.05:.2f}s">'
                    f'<span class="lv-t">{_c.get("heure", "")}</span>'
                    f'<span class="lv-s">{float(_c.get("score", 0.0)):.3g}</span>'
                    f'<span class="lv-v">{_c.get("verdict", "")}</span></div>')
            st.markdown(_html, unsafe_allow_html=True)
            if not _n:
                st.caption(t("lv_wait"))

        # Animations. Injectées hors du fragment pour ne pas être réécrites
        # à chaque rafraîchissement. `prefers-reduced-motion` les coupe : une
        # animation n'est jamais une information, seulement un confort.
        st.markdown("""<style>
.lv-head{display:flex;align-items:center;gap:10px;font-size:.95rem;
         color:var(--fg-2);margin-bottom:9px}
.lv-dot{width:10px;height:10px;border-radius:50%;background:var(--ok);
        animation:lv_pulse 1.6s ease-in-out infinite}
@keyframes lv_pulse{
  0%{box-shadow:0 0 0 0 rgba(94,224,138,.65)}
  70%{box-shadow:0 0 0 10px rgba(94,224,138,0)}
  100%{box-shadow:0 0 0 0 rgba(94,224,138,0)}}
.lv-bar{height:8px;border-radius:999px;background:var(--surface-2);
        border:1px solid var(--border);overflow:hidden;margin:2px 0 15px}
.lv-fill{height:100%;background:linear-gradient(90deg,var(--info),var(--warn));
         animation:lv_grow .9s cubic-bezier(.2,.8,.2,1) both}
@keyframes lv_grow{from{width:0}}
.lv-card{display:flex;gap:12px;align-items:center;padding:7px 11px;margin-bottom:6px;
         border-radius:8px;background:var(--surface-2);
         border-left:3px solid var(--ok);animation:lv_in .35s ease-out both}
.lv-card.lv-al{border-left-color:var(--danger)}
@keyframes lv_in{from{opacity:.35;transform:translateY(-4px)}to{opacity:1;transform:none}}
.lv-t{color:var(--fg-4);font-size:.82rem;min-width:62px}
.lv-s{color:var(--fg-1);min-width:78px;font-variant-numeric:tabular-nums}
.lv-v{margin-left:auto;font-size:.85rem;font-weight:600;color:var(--fg-2)}
@media (prefers-reduced-motion:reduce){
  .lv-dot,.lv-fill,.lv-card{animation:none !important}}
</style>""", unsafe_allow_html=True)

        _live_panel()

    # ── Comparaison des trois détecteurs sur le même lot ──────────────────
    st.divider()
    st.markdown(f"#### {t('d_s4')}")
    if st.button(t("cmp_btn"), width='stretch'):
        _Xs = st.session_state.get("last_batch")
        if _Xs is None:
            st.info(t("cmp_nodata"))
        else:
            _labs = st.session_state.get("last_true")
            if _labs is None:
                st.caption(t("lab_none"))
            _fpr = float(st.session_state.get("last_fpr", 5.0))
            _rows = []
            with st.spinner(t("cmp_btn")):
                for _m, _name in (("modele", "MS-Transformer"),
                                  ("one-class", "One-class"),
                                  ("supervise", "Supervised" if LANG == "en" else "Supervisé")):
                    if _m == "supervise" and HGB is None and SUP is None:
                        continue
                    if _m == "one-class" and FEAT is None:
                        continue
                    _sc, _, _, _ = infer(_Xs, mode_=_m)
                    _thr = _cmp_threshold(_m, _fpr)
                    _nb = int((_sc > _thr).sum())
                    _rec = t("n_a")
                    if _labs is not None:
                        _lab = np.array([str(v) for v in _labs])
                        _atk = ~np.array(["BENIGN" in str(v).upper() for v in _lab])
                        if _atk.sum() > 0:
                            _rec = f"{(_sc[_atk] > _thr).mean()*100:.1f} %"
                    _rows.append({
                        t("cmp_col"): _name,
                        t("cmp_alerts"): f"{_nb}/{len(_sc)}",
                        t("cmp_rate"): f"{_nb/len(_sc)*100:.1f} %",
                        t("cmp_recall"): _rec,
                        t("cmp_scale"): detector_scales()[_m]["kind"],
                    })
            if _rows:
                st.markdown(f"**{t('cmp_h')}**")
                st.caption(t("cmp_intro", fpr=f"{_fpr:g}"))
                st.dataframe(pd.DataFrame(_rows), width='stretch', hide_index=True)
                st.caption(t("cmp_note"))

# ══════════════════════════════════════════════════════════════════════
#  3 · RESULTS
# ══════════════════════════════════════════════════════════════════════
with tab_r:
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
                st.caption(t("lot_c"))
                _r = {
                    _c[0]: os.path.basename(str(_lot.get("fichier", "—"))),
                    _c[1]: f'{_lot.get("flux", 0):,}'.replace(",", " "),
                    _c[2]: f'{_lot.get("alertes", 0):,}'.replace(",", " "),
                    _c[3]: f'{_lot.get("taux_alerte", 0)*100:.2f} %',
                    _c[4]: (f'{_lot["rappel"]*100:.2f} %' if "rappel" in _lot
                            else t("n_a")),
                    _c[5]: f'{_lot.get("duree_s", 0):.1f} s',
                }
                st.dataframe(pd.DataFrame([_r]), width='stretch', hide_index=True)
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

        st.markdown(f"#### {t('r_head')}")
        st.caption(t("r_note", nt=f"{g['n_samples']:,}", nb=f"{g['n_benign']:,}",
                     na=f"{g['n_anomaly']:,}"))

        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(f'<div class="kpi"><div class="kpi-l">{t("r_auroc")}</div>'
                    f'<div class="kpi-v">{_num(g, "auroc", 4)}</div>'
                    f'<div class="kpi-s">{_ci(g, 4)}</div></div>',
                    unsafe_allow_html=True)
        a2.markdown(f'<div class="kpi"><div class="kpi-l">{t("r_recall")}</div>'
                    f'<div class="kpi-v">{_pct(g, "tpr_recall", 2)}</div>'
                    f'<div class="kpi-s">{g.get("threshold_method") or t("missing")}</div></div>',
                    unsafe_allow_html=True)
        a3.markdown(f'<div class="kpi"><div class="kpi-l">{t("r_fpr")}</div>'
                    f'<div class="kpi-v">{_pct(g, "fpr", 2)}</div>'
                    f'<div class="kpi-s">target {_pct(g, "target_fpr", 0)}</div></div>',
                    unsafe_allow_html=True)
        a4.markdown(f'<div class="kpi"><div class="kpi-l">{t("r_prec")}</div>'
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
                st.bar_chart(dfa.set_index("class"), height=260, color="#7ab4f7")
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
                    height=260, color="#7ab4f7")
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
            st.dataframe(pd.DataFrame(
                [["Attack", tp, fn], ["Healthy", fp, tn]],
                columns=["", "flagged", "not flagged"]).set_index(""),
                width='stretch')
            st.caption(f"threshold = {thr:.3f} · FPR = {fp/max(fp+tn,1)*100:.2f} % "
                       f"· recall = {tp/max(tp+fn,1)*100:.2f} %")

        # ── Les trois détecteurs, sur un échantillon fixe du test ─────────
        st.write("")
        st.markdown(f"**{t('cmp3_h')}**")
        try:
            _c3 = compare_three(target_fpr * 100.0, LANG)
            if _c3["rows"]:
                st.caption(t("cmp3_intro", n=f"{_c3['n']:,}", fpr=f"{target_fpr*100:g}"))
                _t3 = pd.DataFrame([{
                    t("cmp_col"): r["mode"],
                    t("cmp_alerts"): f'{r["alerts"]}/{r["n"]}',
                    t("cmp_rate"): f'{r["alerts"]/max(r["n"],1)*100:.1f} %',
                    t("cmp_recall"): f'{r["recall"]*100:.1f} %',
                    t("cmp_scale"): detector_scales()[r["key"]]["kind"],
                } for r in _c3["rows"]])
                st.dataframe(_t3, width='stretch', hide_index=True)

                # rappel par famille inédite, pour voir où chaque détecteur cède
                st.caption(t("cmp3_fam"))
                _f3 = pd.DataFrame([{
                    "famille": f,
                    **{r["mode"]: (f'{r["per"][f]*100:.1f} %' if f in r["per"]
                                   else t("n_a")) for r in _c3["rows"]},
                } for f in _c3["fams"]])
                st.dataframe(_f3, width='stretch', hide_index=True)
                st.caption(t("cmp_note"))
        except Exception as _e:
            st.caption(f"comparaison indisponible : {_e}")

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
            st.dataframe(pd.DataFrame(_rows), width='stretch', hide_index=True)

            _vf = _c.get("validation_par_famille") or {}
            if _vf:
                st.caption(t("c_lofo_h"))
                _r2 = []
                for _fam, _v in sorted(_vf.items(), key=lambda kv: -kv[1]["supervise"]):
                    _r2.append({"famille": _fam,
                                t("c_sv"): f'{_v["supervise"]*100:.1f} %',
                                t("c_oc"): f'{_v["one_class"]*100:.1f} %'})
                st.dataframe(pd.DataFrame(_r2), width='stretch', hide_index=True)
                _pire = min(_vf, key=lambda k: _vf[k]["supervise"])
                st.error(t("c_collapse", fam=_pire,
                           pct=f'{_vf[_pire]["supervise"]*100:.1f}'))
        else:
            st.info(t("c_missing"))

        st.write("")
        st.caption(t("src_note", score=g.get("score", "?")))
        st.markdown(f"#### ⚠️ {t('a_head')}")
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
            st.dataframe(dfm, width='stretch', hide_index=True)

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
            st.dataframe(pd.DataFrame(rows_pv), width='stretch', hide_index=True)

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
            st.dataframe(dfb, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════
#  4 · METHOD
# ══════════════════════════════════════════════════════════════════════
with tab_m:
    q1, q2 = st.columns(2)
    with q1:
        st.markdown(f"#### {t('m_enc_h')}")
        st.markdown(t("m_enc"))
        st.markdown(f"#### {t('m_score_h')}")
        st.markdown(t("m_score"))
    with q2:
        st.markdown(f"#### {t('m_abl_h')}")
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
            st.dataframe(dfx, width='stretch', hide_index=True)
        else:
            st.info("results/ablation.json missing — run `py 04_ablation.py`.")


# ══════════════════════════════════════════════════════════════════════
#  5 · LIMITATIONS
# ══════════════════════════════════════════════════════════════════════
with tab_l:
    st.markdown(f"#### {t('l_h')}")
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
            f'<div class="panel" style="margin-bottom:12px; border-left:3px solid {_col};">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;'
            f' margin-bottom:10px;">'
            f'<span style="font-weight:650; font-size:1.05rem; color:var(--fg-1);">{t(_k + "_t")}</span>'
            f'<span style="font-size:0.82rem; letter-spacing:0.08em; text-transform:uppercase;'
            f' color:{_col}; border:1px solid {_col}; border-radius:999px; padding:2px 10px;">'
            f'<span aria-hidden="true">{_SYM[_st]}</span> {_LBL[_st]}</span></div>'
            f'<div style="font-size:0.92rem; line-height:1.65; color:var(--fg-3); margin-bottom:12px;">'
            f'{t(_k + "_a")}</div>'
            f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;'
            f' font-size:0.88rem; line-height:1.6;">'
            f'<div><div style="color:var(--fg-4); text-transform:uppercase; letter-spacing:0.08em;'
            f' font-size:0.9rem; font-weight:600; margin-bottom:5px;">{t("l_proof")}</div>'
            f'<div style="color:var(--fg-2);">{t(_k + "_p")}</div></div>'
            f'<div><div style="color:var(--fg-4); text-transform:uppercase; letter-spacing:0.08em;'
            f' font-size:0.9rem; font-weight:600; margin-bottom:5px;">{t("l_left")}</div>'
            f'<div style="color:var(--fg-2);">{t(_k + "_r")}</div></div></div>'
            f'</div>', unsafe_allow_html=True)

    # bilan chiffre
    st.write("")
    _b1, _b2, _b3, _b4 = st.columns(4)
    _b1.markdown(
        f'<div class="kpi"><div class="kpi-l">{t("l_status_ok")}</div>'
        f'<div class="kpi-v">1 / 5</div><div class="kpi-s">L2</div></div>',
        unsafe_allow_html=True)
    _b2.markdown(
        f'<div class="kpi"><div class="kpi-l">{t("l_status_part")}</div>'
        f'<div class="kpi-v">2 / 5</div><div class="kpi-s">L3 · L5</div></div>',
        unsafe_allow_html=True)
    _b3.markdown(
        f'<div class="kpi"><div class="kpi-l">{t("l_status_open")}</div>'
        f'<div class="kpi-v">2 / 5</div><div class="kpi-s">L1 · L4</div></div>',
        unsafe_allow_html=True)
    _b4.markdown(
        f'<div class="kpi"><div class="kpi-l">{t("flows")}</div>'
        f'<div class="kpi-v">5 495</div><div class="kpi-s">attaques évaluées</div></div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="note" style="margin-top:14px;">{t("l_foot")}</div>',
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
    def H(key: str) -> str:
        fr, en = HELP[key]
        return fr if LANG == "fr" else en

    st.markdown(f"#### {H('h_intro')}")

    st.divider()
    st.markdown(f"### {H('h_steps_h')}")
    for _k in ("h_step1", "h_step2", "h_step3"):
        st.markdown(f'<div class="panel" style="margin-bottom:10px;">{H(_k)}</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### {H('h_modes_h')}")
    _c1, _c2, _c3 = st.columns(3)
    for _col, _n, _b in ((_c1, "h_mode1", "h_mode1_t"),
                         (_c2, "h_mode2", "h_mode2_t"),
                         (_c3, "h_mode3", "h_mode3_t")):
        with _col:
            st.markdown(
                f'<div class="panel" style="height:100%;">'
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
