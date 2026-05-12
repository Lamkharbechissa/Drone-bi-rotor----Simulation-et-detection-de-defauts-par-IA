# 🛸 Détection et prédiction de défauts par Machine Learning
### Projet PJT — Jumeau Numérique · 2025–2026

> Génération de données de capteurs simulées, exploration et ingénierie de features en vue d'entraîner un modèle de Machine Learning capable de détecter et d'anticiper les défauts d'un système mécanique embarqué.

---

## 📋 Pipeline du projet

```
Génération de données  →  EDA  →  Feature Engineering  →  Modèle ML
```

---

## 1. Génération de la base de données

Les lignes générées reproduisent **physiquement** ce que les capteurs mesureraient dans chaque scénario de défaut, avec l'ajout d'un **bruit réaliste**.

### Structure des données

Chaque ligne contient les variables suivantes :

| Colonne | Description |
|---|---|
| `timestamp` | Horodatage de la mesure |
| `distance` | Distance mesurée (altitude du drone) |
| `courant_droite` | Courant moteur droit |
| `courant_gauche` | Courant moteur gauche |
| `ax_G`, `ay_G`, `az_G` | Accélérations bras gauche (x, y, z) |
| `ax_D`, `ay_D`, `az_D` | Accélérations bras droit (x, y, z) |
| `angle_gauche` | Angle bras gauche |
| `angle_droite` | Angle bras droit |
| `angle_nacelle` | Angle de la nacelle |
| `label_defaut` | Type de défaut détecté |
| `phase` | État du système |

### Variable `phase`

La variable `phase` indique l'état du système parmi trois valeurs possibles :

- **Régime normal** — fonctionnement nominal
- **Pré-défaut** — début de dégradation
- **Défaut franc** — défaut avéré

Cette structuration en phases permet d'entraîner un modèle capable de réaliser une **prédiction anticipée des défauts**, avant que ceux-ci ne deviennent critiques.

---

## 2. Exploration et Visualisation des données (EDA)

L'**Exploratory Data Analysis (EDA)** consiste à observer et comprendre les données à l'aide de graphiques et d'analyses descriptives, avant d'appliquer des algorithmes de Machine Learning.

À partir de la base de données générée, cette étape permet d'analyser les comportements des différentes variables mesurées par les capteurs.

### Questions clés auxquelles répond l'EDA

Avant de lancer un algorithme d'apprentissage automatique, il est nécessaire de répondre aux questions suivantes :

- Les défauts laissent-ils réellement des **traces visibles** dans les signaux des capteurs ?
- Existe-t-il des **différences mesurables** entre le régime normal, le pré-défaut et le défaut ?
- Les données contiennent-elles suffisamment d'**information discriminante** pour permettre à un modèle de Machine Learning d'apprendre et de détecter ces défauts ?

L'objectif de cette étape est de vérifier que les données possèdent des caractéristiques discriminantes permettant d'identifier les différents états du système.

---

## 3. Feature Engineering

Le **Feature Engineering** consiste à transformer les données brutes issues des capteurs en nouveaux indicateurs plus informatifs pour les algorithmes de Machine Learning.

Les capteurs fournissent des mesures instantanées (courants, accélérations, angles, distance), mais ces valeurs seules ne permettent pas toujours de comprendre directement le comportement physique du système.

L'objectif est donc de calculer de nouvelles **features** qui traduisent des phénomènes mécaniques observables, tels que les déséquilibres, les vibrations ou les pertes d'altitude.

### 3.1 Features calculées

| Feature | Formule | Interprétation physique |
|---|---|---|
| `delta_courant` | `\|courant_main − courant_tail\|` | Détecte un déséquilibre entre les deux moteurs |
| `vibration_rms` | `√mean(ax² + ay² + az²)` sur fenêtre de 20 points | Mesure l'intensité globale des vibrations du bras |
| `asymetrie_angle` | `\|angle_gauche − angle_droite\|` | Détecte un déséquilibre mécanique entre les deux bras |
| `tendance_altitude` | Pente de la régression linéaire sur les 20 dernières valeurs de `distance` | Identifie une montée, une stabilité ou une chute du drone |
| `variance_courant` | Variance de `courant_main` sur les 10 derniers points | Indique si le courant est stable ou fluctue fortement |
| `energie_gyro` | `mean(\|gx\| + \|gy\| + \|gz\|)` sur fenêtre de 10 points | Mesure l'agitation globale du drone |

### Fenêtres glissantes

Certaines features (`vibration_rms`, `variance_courant`, `energie_gyro`) sont calculées à partir de plusieurs points dans le temps grâce à une **fenêtre glissante**. Cela permet d'analyser l'évolution récente des mesures plutôt qu'une valeur instantanée, et donc de mieux capturer les tendances ou anomalies.

### 3.2 Résultat

Après le Feature Engineering, le jeu de données contient à la fois :
- Les **variables brutes** des capteurs
- Les **nouvelles variables calculées**, plus représentatives du comportement physique du système

Ces nouvelles features permettent aux modèles de Machine Learning de **détecter et prédire plus efficacement les défauts**.
