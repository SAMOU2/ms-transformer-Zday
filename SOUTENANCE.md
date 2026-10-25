# Dossier de soutenance

Support de préparation : plan minuté, chiffres à retenir, questions du jury,
et les formulations à éviter.

---

## 1. Le projet en trente secondes

> Détecter des attaques réseau **jamais vues à l'entraînement** sur CICIDS2017.
> J'ai comparé quatorze méthodes sous un protocole strict — les familles d'attaque
> du test n'apparaissent jamais pendant l'entraînement.
> Le détecteur retenu est un gradient boosting sur les features réseau :
> **AUROC 0,967 et 85 % de rappel pour 10 % de fausses alertes**.
> L'architecture transformer que j'avais conçue plafonne à 0,82 ; je l'ai mesurée,
> et je l'explique.

---

## 2. Plan de présentation (10 minutes)

| Temps | Contenu | Message |
|---|---|---|
| 0:00 – 1:30 | **Problème** : un IDS supervisé ne reconnaît que ce qu'il a vu | pourquoi le Zero-Day |
| 1:30 – 2:30 | **Protocole** : découpage par famille, 5 familles tenues à l'écart | le cœur méthodologique |
| 2:30 – 4:30 | **Architecture** : MS-Transformer, double modalité, 195 260 paramètres | la contribution personnelle |
| 4:30 – 6:30 | **Résultats** : les 14 méthodes, le classement | la comparaison |
| 6:30 – 8:00 | **Ablation** : ce qui est justifié, ce qui ne l'est pas | l'honnêteté |
| 8:00 – 9:00 | **Limites** : Bot, branche EDR synthétique, sélection sur le test | la lucidité |
| 9:00 – 10:00 | **Démo** : l'application, trois modes | la preuve |

---

## 3. Les chiffres à connaître par cœur

| Élément | Valeur |
|---|---|
| Jeu d'entraînement | 73 266 flux (40 000 bénins + 33 266 attaques) |
| Jeu de test | 30 496 flux (25 001 bénins + 5 495 attaques) |
| **AUROC cross-famille (modèle retenu)** | **0,9672** (IC 95 % : 0,9653 – 0,9691) |
| Rappel @ 10 % de FA | 84,8 % |
| FPR réel | 9,4 % |
| Précision / F1 | 66,4 % / 74,5 % |
| TPR @ 1 % FPR | 56,1 % |
| Baseline sans apprentissage (`Σ\|x\|`) | 0,5913 |
| Transformer (meilleure configuration) | 0,8187 |
| Paramètres du transformer | 195 260 |

### Par famille

| Famille | AUROC | Rappel |
|---|---|---|
| SQL Injection | 0,9991 | 100,00 % |
| XSS | 0,9984 | 99,69 % |
| Brute Force | 0,9949 | 97,01 % |
| PortScan | 0,9812 | 96,66 % |
| Bot | 0,9056 | 49,34 % |

---

## 4. Les cinq messages à faire passer

**① Le protocole change le résultat.**
Découpage aléatoire : 0,99. Découpage par famille : 0,90 à 0,97. L'écart ne mesure pas
une faiblesse du modèle, mais la difficulté réelle de la tâche. La plupart des articles
qui annoncent 99 % sur CICIDS utilisent le premier.

**② La famille de modèle compte plus que l'architecture.**
Même espace de features, même protocole : régression logistique 0,9026, gradient
boosting 0,9672. Le rappel passe de 40 % à 85 %.

**③ L'embedding du transformer n'apporte rien.**
Donné en complément à un modèle fort : 0,9463 au lieu de 0,9672. Il dilue les features.

**④ L'ablation est partiellement dégradante, et je le dis.**
Le masque FSM justifie sa présence (−3,1 points sans lui). La perte contrastive et la
branche EDR dégradent. Un composant sur trois est justifié.

**⑤ La limite sur Bot est dans les données, pas dans le modèle.**
AUROC 0,9056 mais 49 % de rappel : les flux Bot produisent peu de régularités
statistiques. Le capter exigerait plus de 10 % de fausses alertes.

---

## 5. Questions probables du jury

**« Pourquoi le transformer fait-il moins bien que le gradient boosting ? »**
Parce que le transformer compresse 77 features en 64 dimensions, et que cette
compression perd de l'information utile. Le gradient boosting travaille sur les
features brutes et construit des conjonctions (paquet court **et** durée brève
**et** drapeau SYN) qu'une représentation continue lisse.

**« 0,9672, n'est-ce pas trop beau ? »**
C'est une AUROC mesurée sur des familles jamais vues, avec un bénin réaliste de
25 001 flux. À 5 % de fausses alertes, le rappel tombe à 80,9 %, ce qui est cohérent
avec cette AUROC. Le chiffre de 0,99 que d'autres publient suppose un découpage
aléatoire, plus facile.

**« Pourquoi le rappel sur Bot est-il si bas ? »**
Sa courbe ROC est correcte mais ses scores se concentrent juste sous le seuil :
le capturer demanderait d'accepter plus de fausses alertes. C'est un arbitrage
opérationnel, pas un défaut du modèle. Bot cherche précisément à ressembler au
trafic normal.

**« Votre branche EDR est-elle réelle ? »**
Non. Les 25 colonnes sont un proxy dérivé du trafic réseau. Je le déclare
explicitement, et l'ablation montre que le retirer améliore le résultat de 3,7 points.

**« Avez-vous testé beaucoup de modèles et gardé le meilleur ? »**
Oui, sept variantes supervisées, et je le signale comme limite : le chiffre est
légèrement optimiste. L'écart avec la deuxième famille de modèles (Random Forest,
0,9452) reste toutefois important.

**« Pourquoi pas un autoencodeur ? »**
Testé : 0,7390, le plus mauvais des six variantes de l'ablation.

**« Qu'est-ce qui est reproductible ? »**
Tout : cinq scripts, des graines fixes, des empreintes MD5 dans le README, et un
guide d'exécution (`GUIDE.md`) qui donne l'ordre et les durées.

**« Quelle serait la suite ? »**
Trois pistes : un découpage temporel plutôt que par famille, une branche EDR
réellement collectée, et un seuil par profil de trafic pour améliorer Bot sans
augmenter globalement les fausses alertes.

---

## 6. Trois pièges à éviter

**Ne dis pas « mon modèle détecte 96 % des Zero-Day ».** Dis : « AUROC de 0,967,
rappel de 85 % à 10 % de fausses alertes ». La nuance est exactement ce que le jury
cherche à évaluer.

**Ne présente pas 0,99.** Ce chiffre n'existe dans ce projet que sur un découpage
aléatoire, et il est documenté comme invalide.

**N'annonce pas la double modalité comme un acquis.** La branche EDR est une
simulation ; l'ablation la déclare nuisible.

---

## 7. Ce que le projet démontre

- Un détecteur qui reconnaît des familles d'attaque **jamais vues**, à 0,967 d'AUROC ;
- **+37,6 points** au-dessus d'une baseline sans apprentissage ;
- Le protocole par famille est plus exigeant que le découpage aléatoire, et l'écart
  explique les scores de 0,99 publiés sur ce jeu ;
- Le masque FSM apporte une contribution mesurable ; le contraste et la branche EDR
  n'en apportent pas.

## Ce qu'il ne démontre pas

- Une détection opérationnelle de Bot au point de fonctionnement retenu ;
- L'intérêt d'une architecture profonde sur ces features ;
- La validité de la branche EDR, qui reste une simulation.

---

## 8. Paragraphe de conclusion, à lire tel quel

> Sur des familles d'attaque jamais vues à l'entraînement, mon détecteur atteint une
> AUROC de 0,967 et un rappel de 85 % pour 10 % de fausses alertes. Ce résultat est
> obtenu avec un gradient boosting sur les features réseau : l'architecture
> transformer que j'ai conçue et entraînée plafonne à 0,82, et son embedding n'améliore
> pas le modèle tabulaire. L'étude d'ablation montre qu'un composant de l'architecture
> est justifié, deux ne le sont pas. La famille Bot reste sous 50 % de rappel, ce qui
> marque la limite du signal disponible dans les statistiques de flux. Ce travail
> mesure donc autant ce qui fonctionne que ce qui ne fonctionne pas — et c'est, je
> crois, ce qu'on attend d'une démarche de recherche.
