# 🤖 Pipeline IA — Détection de défauts drone
### Machine Learning · Simulation capteurs · Maintenance prédictive · 2026

> Pipeline complet de machine learning pour la détection automatique de défauts sur drone à partir de données capteurs. Chaque étape est documentée avec le code Python utilisé, les choix techniques effectués et les résultats obtenus sur le dataset `drone_simulation_Good.csv`.

---

## 🎯 Objectif du projet

Classifier automatiquement l'état d'un drone parmi **21 classes de défauts** (dont l'état normal) à partir de données issues de capteurs embarqués (courants, accéléromètres, gyroscopes, ultrason), en prédisant la variable cible `label_defaut`.

---

## 📋 Pipeline global

```
Collecte des données  →  EDA  →  Prétraitement  →  Feature Engineering  →  Sélection  →  Modèles  →  Évaluation
```

---

## 1. Collecte des données

Le dataset exploité est `drone_simulation_Good.csv`, chargé depuis Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/PJT_2026/drone_simulation_Good.csv')
df.head()
```

### Variables du dataset

| Catégorie | Variable(s) | Description |
|---|---|---|
| Capteurs électriques | `courant_main`, `courant_tail` | Courant rotor principal et queue |
| Mouvement | `distance` | Distance mesurée (altitude) |
| Accéléromètres | `ax_G/D`, `ay_G/D`, `az_G/D` | Accélérations capteurs gauche/droit |
| Angles | `angle_gauche`, `angle_droite`, `angle_nacelle` | Angles des deux bras et de la nacelle |
| Labels cibles | `label_defaut`, `amplitude_defaut`, `localisation_defaut` | Variables à prédire |

### Distribution des variables cibles

```python
df['amplitude_defaut'].value_counts()
df['localisation_defaut'].value_counts()
df['label_defaut'].value_counts()
```

**Distribution de `label_defaut`** — 21 classes, dataset équilibré à 500 obs./classe de défaut :

| Classe | Observations |
|---|---|
| normal | 2000 |
| coupure_moteur, blocage_moteur, usure_moteur, defaut_esc, helice_fendue, helice_desequilibree, helice_inversee, desequilibre_balancier, rupture_bras, fissure_bras, frottement_glissiere, blocage_glissiere, derive_mpu6050, mpu6050_defaillant, hcsr04_defaillant, acs712_derive, decharge_batterie, coupure_alimentation, boulon_desserre, deformation_base | 500 chacune |

> ⚠️ **Déséquilibre des classes** : la classe `normal` contient 2000 observations, soit 4 fois plus que chaque classe de défaut. Ce déséquilibre doit être pris en compte lors de l'entraînement (pondération des classes ou rééchantillonnage).

---

## 2. Analyse Exploratoire des Données (EDA)

### 2.1 Exploration générale

La fonction `explore_data()` réalise un diagnostic complet : dimensions, types de colonnes, valeurs manquantes, doublons et cardinalité.

```python
explore_data(df)
```

> ✅ Le dataset est **propre dès le départ** : aucune valeur manquante et aucun doublon. L'étape de gestion des valeurs manquantes sera sans effet sur ce dataset, mais reste présente dans le pipeline pour sa généricité.

### 2.2 Analyse des distributions

Pour chaque variable numérique (hors colonnes cibles), on trace un histogramme avec courbe KDE et les lignes de moyenne et médiane, puis un boxplot.

```python
target_cols = ['label_defaut', 'defaut_amplitude', 'defaut_localisation']
plot_distributions(df, target_cols=target_cols)
```

**Observations clés :**
- Forte asymétrie (*skewness*) sur `courant_main` (2.24) et `courant_tail` (4.14), indiquant la présence de défauts qui tirent la distribution vers la droite
- Variables avec le plus grand nombre d'outliers (détection par IQR) : `courant_main` (3346), `courant_tail` (2256), `angle_gauche` (2475)

> 💡 Le grand nombre d'outliers **n'est pas du bruit** : il reflète les comportements anormaux simulés (coupure moteur, vibrations, déséquilibre). Ces valeurs sont des **signaux de défaut** et doivent être traitées par remplacement médiane plutôt que suppression.

### 2.3 Étude des corrélations

```python
plot_correlation_matrix(df, target_cols=target_cols, method="pearson")
# Seuil de forte corrélation : |r| >= 0.7
```

**Deux corrélations notables :**
- `az_G` ↔ `az_D` : r = 0.68
- `angle_gauche` ↔ `angle_droite` : r = −0.63

> **Conclusion EDA** : aucune variable ne dépasse le seuil de forte corrélation (|r| ≥ 0.70). Les variables sont suffisamment indépendantes pour être conservées toutes. La PCA permettra néanmoins d'éliminer la redondance résiduelle après le feature engineering.

---

## 3. Prétraitement des Données

### 3.1 Gestion des valeurs manquantes

Stratégie adoptée :
- Suppression des colonnes avec > 50% de NaN
- Imputation par la **médiane** si |skewness| > 1 (distribution asymétrique)
- Imputation par la **moyenne** sinon
- Imputation par le **mode** pour les variables catégorielles

```python
df = handle_missing_values(df, drop_threshold=0.5)
# Sur ce dataset : aucune valeur manquante -> aucune modification
```

### 3.2 Traitement des valeurs aberrantes

Stratégie adoptée : `replace_median` — les outliers détectés par la règle IQR (×1.5) sont remplacés par la médiane de la colonne, sans suppression de lignes.

$$x < Q_1 - 1.5 \times IQR \quad \text{ou} \quad x > Q_3 + 1.5 \times IQR, \quad \text{avec } IQR = Q_3 - Q_1$$

```python
df = handle_outliers(df, strategy='replace_median')
# Les outliers sont remplacés par la médiane de leur colonne
# (pas de suppression de lignes pour préserver le signal de défaut)
```

> La suppression des outliers aurait éliminé des mesures de défaut réelles. Le remplacement par la médiane limite leur impact sur les modèles linéaires tout en conservant la structure temporelle des séries.

### 3.3 Normalisation et standardisation

Stratégie adoptée : mode `compare` — pour chaque colonne, Min-Max et Z-score sont comparés ; la méthode dont l'écart-type résultant est le plus proche de 1 est retenue automatiquement.

$$\text{Min-Max : } x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$

$$\text{Z-score : } x' = \frac{x - \mu}{\sigma}, \quad \mu = 0, \sigma = 1$$

```python
df = scale_features(df, method='compare')
# Pour chaque colonne : compare std_minmax et std_zscore,
# retient la méthode dont std est la plus proche de 1
```

---

## 4. Feature Engineering

Le Feature Engineering transforme les données brutes capteurs en variables pertinentes pour la détection de défauts. **Quatre familles de features** sont construites, exploitant les propriétés physiques, temporelles et dynamiques du système.

### 4.1 Features physiques

```python
df = add_physical_features(df, col_main='courant_main', col_tail='courant_tail')
```

| Feature | Formule | Intérêt |
|---|---|---|
| `delta_I` | `\|courant_main − courant_tail\|` | Déséquilibre moteur, pré-défaut |
| `ratio_I` | `courant_main / courant_tail` | Normalisation du déséquilibre |
| `norm_accel_G` | `√(ax_G² + ay_G² + az_G²)` | Intensité vibratoire gauche |
| `norm_accel_D` | `√(ax_D² + ay_D² + az_D²)` | Intensité vibratoire droite |
| `delta_norm_accel` | `\|‖aG‖ − ‖aD‖\|` | Déséquilibre vibratoire G/D |
| `delta_angle` | `\|angle_gauche − angle_droite\|` | Instabilité angulaire |

### 4.2 Features temporelles (fenêtres glissantes)

Sur une fenêtre glissante de N = 10 mesures, six statistiques sont extraites pour chacune des 12 variables brutes, soit **12 × 6 = 72 nouvelles features**.

```python
BASE_COLS = ['courant_main', 'courant_tail', 'angle_gauche', 'angle_droite',
             'angle_nacelle', 'ax_G', 'ay_G', 'az_G',
             'ax_D', 'ay_D', 'az_D', 'distance']
base_cols = [c for c in BASE_COLS if c in df.columns]
df = add_temporal_features(df, cols=base_cols, window=10)
```

| Feature | Formule | Intérêt |
|---|---|---|
| `rms_col` | `√(1/N · Σx²)` | Énergie du signal |
| `var_col` | `1/N · Σ(xi − x̄)²` | Dispersion, instabilité |
| `mean_col` | `1/N · Σxi` | Tendance globale lissée |
| `std_col` | `√σ²` | Anomalies locales |
| `max_col` | `max(xi)` | Pics anormaux |
| `min_col` | `min(xi)` | Perturbations basses |

### 4.3 Features dynamiques

Ces features capturent l'évolution temporelle du signal et permettent la **détection anticipée** des transitions vers un défaut.

```python
df = add_dynamic_features(df, cols=base_cols, slope_window=5)
```

| Feature | Formule | Intérêt |
|---|---|---|
| `deriv1_col` | `x(t) − x(t−1)` | Changements brusques |
| `slope_col` | `x(t) = at + b` (régression sur 5 pts) | Dérives progressives, pré-défaut |
| `deriv2_col` | `x(t) − 2x(t−1) + x(t−2)` | Dynamique rapide du système |

### 4.4 Features combinées

**Interactions entre variables** — produits entre paires pour capturer les relations non linéaires :

```python
interaction_pairs = [
    ('courant_main', 'courant_tail'),   # déséquilibre moteur
    ('norm_accel_G', 'norm_accel_D'),   # déséquilibre vibratoire
    ('angle_gauche', 'angle_droite'),   # déséquilibre angulaire
    ('courant_main', 'norm_accel_G'),   # courant / vibration gauche
    ('courant_tail', 'norm_accel_D'),   # courant / vibration droite
    ('distance', 'angle_nacelle'),      # effet distance / orientation
]
df = add_combined_features(df, interaction_pairs=interaction_pairs)
```

**Indicateurs de seuil binaires** — le seuil utilisé est la moyenne de la colonne, calculée automatiquement :

$$thresh\_col(t) = \begin{cases} 1 & \text{si } col(t) > \bar{col} \\ 0 & \text{sinon} \end{cases}$$

---

## 5. Sélection de Features

### 5.1 Random Forest — Importance MDI

Le modèle Random Forest évalue la contribution de chaque variable à la réduction d'impureté de Gini. Un seuil de 0.005 est appliqué.

```python
importance_df, selected_cols = compute_feature_importance(
    df,
    target_col="label_defaut",
    importance_threshold=0.005,
    n_estimators=200,
    top_n=30,
    plot=True,
)
```

Paramètres retenus : 200 arbres, seuil d'importance ≥ 0.005. Les features en dessous de ce seuil sont éliminées pour simplifier le modèle et réduire le risque de surapprentissage.

### 5.2 Analyse en Composantes Principales (PCA)

$$Z = X \cdot W$$

où X est la matrice des données standardisées, W la matrice des vecteurs propres et Z la matrice des données projetées.

```python
df_pca, pca, scaler = apply_pca(
    df,
    feature_cols=selected_cols,
    target_col="label_defaut",
    variance_threshold=0.95,
    plot=True,
)
```

| Avantages | Limites |
|---|---|
| Réduction du bruit | Composantes peu interprétables |
| Élimination de la multicolinéarité | Perte d'information possible |
| Accélération de l'entraînement | Nécessite une standardisation préalable |

---

## 6. Division des données

```python
target_cols = ['label_defaut', 'label_id']

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    df,
    target_cols,
    test_size=0.2,    # 20% test
    val_size=0.1,     # 10% validation
    stratify=True,    # conservation des proportions de classes
    time_series=False
)
```

| Ensemble | Proportion | Rôle |
|---|---|---|
| Entraînement | 70% | Apprentissage des paramètres du modèle |
| Validation | 10% | Ajustement des hyperparamètres |
| Test | 20% | Évaluation finale des performances |

La stratification (`stratify=True`) garantit que les proportions de chaque classe de défaut sont respectées dans les trois ensembles.

---

## 7. Entraînement des modèles

Quatre modèles de classification sont entraînés sur `label_defaut` et évalués sur le jeu de validation (600 observations).

### 7.1 Régression Logistique *(baseline)*

Modèle de référence. Un scaling Z-score est appliqué au préalable car la régression logistique est sensible aux échelles.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train["label_defaut"])
```

**Résultats :** Train Acc : **93.57%** · Val Acc : **92.17%** · F1 macro : 0.90 · Overfitting : Faible  
Meilleure classe : `vibration` (F1 = 0.98) · Plus difficile : `déséquilibre` (F1 = 0.77)

### 7.2 Random Forest

Ensemble de 200 arbres de décision avec vote majoritaire. Ne nécessite pas de scaling.

```python
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train["label_defaut"])
```

**Résultats :** Train Acc : **100%** (surapprentissage) · Val Acc : **92.17%** · F1 macro : 0.90 · Overfitting : Fort  
Meilleure classe : `vibration` (F1 = 0.98) · Plus difficile : `déséquilibre` (F1 = 0.75)

### 7.3 Gradient Boosting

Construction séquentielle d'arbres : chaque arbre corrige les erreurs du précédent.

```python
model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train["label_defaut"])
```

**Résultats :** Train Acc : **99%** · Val Acc : **92%** · F1 macro : 0.90 · Overfitting : Modéré  
Meilleure classe : `vibration` (F1 = 0.97) · Plus difficile : `déséquilibre` (F1 = 0.77)

### 7.4 SVM (Support Vector Machine)

Recherche de l'hyperplan optimal maximisant la marge entre les classes. Un scaling Z-score est requis.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

model = SVC(probability=True, random_state=42)
model.fit(X_train_scaled, y_train["label_defaut"])
```

**Résultats :** Train Acc : **95.29%** · Val Acc : **91.5%** · F1 macro : 0.89 · Overfitting : Faible  
Meilleure classe : `vibration` (F1 = 0.97) · Plus difficile : `déséquilibre` (F1 = 0.74)

---

## 8. Évaluation et comparaison des modèles

### Tableau récapitulatif — jeu de validation (600 obs.)

| Modèle | Train Acc. | Val Acc. | F1 macro | Recall macro | Overfitting |
|---|---|---|---|---|---|
| Logistic Regression | 93.57% | **92.17%** | **0.90** | **0.89** | Faible ✅ |
| Random Forest | 100% | **92.17%** | **0.90** | **0.89** | Fort ⚠️ |
| Gradient Boosting | 99% | 92.00% | **0.90** | **0.89** | Modéré ⚠️ |
| SVM | 95.29% | 91.50% | 0.89 | 0.87 | Faible ✅ |

### Métriques d'évaluation

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Critère de sélection — priorité au Recall

> Dans le cadre de la détection de défauts industriels, le **recall est la métrique la plus critique**. Un faux négatif (défaut non détecté) peut entraîner une panne du système, des coûts élevés de réparation ou des risques de sécurité. Un faux positif (fausse alarme) est bien moins critique.

**Modèle recommandé :** La **Régression Logistique** offre le meilleur compromis — performances identiques aux méthodes ensemblistes, sans surapprentissage, avec un temps d'inférence très rapide. Pour des performances maximales, le **Gradient Boosting** avec optimisation des hyperparamètres (GridSearchCV) serait à privilégier.

---

## 9. Amélioration continue

- [ ] **Modèles séquentiels** — LSTM ou CNN-1D pour exploiter la dépendance temporelle des séries de capteurs
- [ ] **Gestion du déséquilibre de classes** — technique SMOTE ou pondération (`class_weight='balanced'`) pour améliorer le recall sur les classes minoritaires
- [ ] **Optimisation des hyperparamètres** — GridSearchCV ou Optuna sur le Gradient Boosting ou XGBoost
- [ ] **Validation croisée temporelle** — `TimeSeriesSplit` pour respecter la dépendance temporelle lors de l'évaluation
- [ ] **Réentraînement continu** — intégration des nouvelles données d'essais physiques pour adapter le modèle aux conditions réelles

---

## 10. Conclusion

Ce pipeline constitue une approche complète pour la détection et la prédiction des défauts sur drone. Les résultats obtenus montrent qu'il est possible d'atteindre **plus de 92% d'accuracy sur 21 classes de défauts** avec des modèles classiques de machine learning, à condition de bien préparer les données.

L'étape de **Feature Engineering** s'est avérée déterminante : les features physiques (déséquilibre de courant, normes d'accélération), temporelles (RMS, variance glissante) et dynamiques (dérivées, pentes) encodent la connaissance métier directement dans les variables d'entrée, rendant les défauts plus détectables pour tous les algorithmes testés.

Le modèle retenu sera déployé comme outil de **maintenance prédictive**, permettant d'anticiper les pannes et de réduire les coûts opérationnels du drone.
