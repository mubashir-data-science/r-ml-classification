# 🤖 Survival Prediction using ML in R

Binary classification comparing **Logistic Regression, Random Forest, and SVM** using the caret framework. Achieves AUC-ROC > 0.87.

![R](https://img.shields.io/badge/R-4.3-blue?style=flat-square&logo=r)
![caret](https://img.shields.io/badge/caret-ML_Framework-green?style=flat-square)
![RandomForest](https://img.shields.io/badge/RandomForest-Best_Model-orange?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-0.87%2B-brightgreen?style=flat-square)

---

## 🎯 Model Results

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| **Random Forest** | **84.2%** | **0.89** |
| Logistic Regression | 81.5% | 0.86 |
| SVM (RBF) | 82.1% | 0.87 |

---

## 🧠 Pipeline

```
Load Data → EDA → Feature Engineering → 
Train/Test Split → 5-Fold CV → Model Training → 
ROC Analysis → Feature Importance
```

## 🔧 Engineered Features
- `family_size` = sibsp + parch + 1
- `is_alone` = alone passenger flag
- `fare_per_person` = fare / family_size
- `age_group` = Child / Teen / YoungAdult / Adult / Senior

---

## ⚙️ Run

```r
# Install R from: https://cran.r-project.org/
Rscript ml_classification.R
```

## 📂 Structure

```
2_r_ml_classification/
├── ml_classification.R    ← Full ML pipeline
├── titanic_data.csv       ← Dataset (800 passengers)
├── 01_survival_by_gender.png
├── 05_roc_curves.png
├── 06_feature_importance.png
└── README.md
```

---

*Built by Muhammad Mubashir*
