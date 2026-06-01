# 🏠 Real Estate Price Prediction
### End-to-End Machine Learning Pipeline on the Ames Housing Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-green?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=white"/>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/ShivamSinghai/Real-Estate-Price-Predictor" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Try%20the%20App-FF4B4B?style=for-the-badge"/>
  </a>
  &nbsp;
  <a href="https://colab.research.google.com/github/ShivamSingh3406/Real-Estate-Price-Prediction-Model/blob/main/Real_Estate_Price_Prediction_final.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height:28px"/>
  </a>
</p>

---

## 📌 Project Overview

This project builds a **complete, production-grade machine learning pipeline** to predict residential property sale prices using the **Ames Housing Dataset** — a rich, real-world dataset containing 2,930 property records and 82 features covering structural, qualitative, and locational attributes.

The notebook demonstrates a rigorous, professional workflow: from raw data ingestion through to model selection, evaluation, and serialization — with every decision thoroughly documented and justified.

---

## 🎯 Objective

> Predict the **Sale Price** of residential properties with high accuracy by applying systematic data preprocessing, feature engineering, and comparative model evaluation.

---

## 📁 Repository Structure

```
📦 Real-Estate-Price-Prediction
├── 📓 Real_Estate_Price_Prediction_Clean.ipynb   ← Main notebook (single-flow)
├── 📊 Original_raw_dataset_AmesHousing.csv       ← Raw input dataset
├── 🤖 real_estate_model.pkl                      ← Saved best model (Random Forest)
└── 📄 README.md                                  ← You are here
```

---

## 🔬 Pipeline Overview

The notebook follows a strict, single-flow architecture — no mid-notebook reloads, no scattered imports, no broken dependencies.

| # | Stage | Key Actions |
|---|-------|-------------|
| 1 | **Data Loading & Inspection** | Shape, dtypes, descriptive statistics |
| 2 | **Missing Value Treatment** | Median/zero imputation for numerical; 'None'/mode for categorical |
| 3 | **Duplicate Check** | Row and column-level validation |
| 4 | **Outlier Treatment** | IQR-based capping for continuous; domain-logic for discrete |
| 5 | **Categorical Cleaning** | Rare grouping (<1%), spelling correction, identifier removal |
| 6 | **Variable Transformation** | Log1p and Yeo-Johnson to normalize skewed distributions |
| 7 | **Encoding** | Binary → Ordinal → One-Hot (drop_first) |
| 8 | **Multicollinearity (VIF)** | 9 high-VIF features pruned iteratively |
| 9 | **Correlation Analysis** | 14 near-zero correlation features removed |
| 10 | **Feature Scaling** | StandardScaler (normal dist.) + RobustScaler (skewed/outlier features) |
| 11 | **EDA** | Heatmaps, scatter plots, residual analysis |
| 12 | **Train-Test Split** | 80/20 split, `random_state=42` |
| 13 | **Model Training & Evaluation** | 5 models compared on test-set metrics |
| 14 | **Final Model Selection** | Random Forest — best R² and lowest RMSE |
| 15 | **Model Serialization** | Saved as `real_estate_model.pkl` |

---

## 🤖 Models Trained & Compared

| Model | R² (Test) | MAE (USD) | RMSE (USD) |
|-------|:---------:|:---------:|:----------:|
| Simple Linear Regression (1 feature) | ~0.487 | ~$42,572 | ~$63,417 |
| Multiple Linear Regression | ~0.871 | ~$19,082 | ~$31,814 |
| Ridge Regression (L2) | ~0.870 | ~$19,165 | ~$31,862 |
| Lasso Regression (L1) | ~0.870 | ~$19,157 | ~$31,835 |
| **Random Forest Regressor** ✅ | **~0.919** | **~$15,292** | **~$25,244** |

### 🏆 Winner: Random Forest Regressor

Random Forest outperforms all linear models because it:
- Captures **non-linear interactions** between features
- Uses **bagging** to reduce variance and generalize better
- Is robust to **feature scale differences**

---

## 📊 Key Findings from EDA

**Top Positive Price Drivers:**
- `Overall Qual` (r ≈ 0.80) — Quality of material and finish is the single strongest predictor
- `Exter Qual` (r ≈ 0.74) — Exterior quality strongly influences buyer perception
- `Gr Liv Area` (r ≈ 0.71) — Above-ground living area is a primary value driver
- `Kitchen Qual`, `Garage Cars`, `Bsmt Qual`, `Year Built` — All exceed r = 0.55

**Top Negative Price Drivers:**
- `Garage Type_Detchd` (r ≈ -0.37) — Detached garages reduce value
- `Foundation_CBlock` (r ≈ -0.35) — Older foundation types lower price
- `Neighborhood_OldTown`, `Neighborhood_Edwards` — Location consistently impacts value

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical operations |
| `matplotlib` + `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, modeling, evaluation |
| `statsmodels` | VIF multicollinearity analysis |
| `pickle` | Model serialization |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/ShivamSingh3406/Real-Estate-Price-Prediction-Model.git
cd Real-Estate-Price-Prediction-Model
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

**3. Launch the notebook**
```bash
jupyter notebook Real_Estate_Price_Prediction_Clean.ipynb
```

**4. Run All Cells**
- `Kernel` → `Restart & Run All`
- The dataset CSV must be in the same directory as the notebook.

---

## 💾 Using the Saved Model

```python
import pickle

# Load the trained Random Forest model
with open('real_estate_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict on new preprocessed data
predictions = model.predict(X_new)
```

> ⚠️ Note: Input data must go through the same preprocessing pipeline before inference.

---

## 📈 Results Summary

The final **Random Forest Regressor** achieved:

```
R²   : 0.919   → Explains 91.9% of price variance on unseen data
MAE  : $15,292  → Average prediction error of ~$15K
RMSE : $25,244  → Typical error magnitude in sale price units
```

---

## 👨‍💻 Author

**Shivam Singh**  
M.Sc. Data Science & AI — BITS Pilani  
Business Analytics with Gen & Agentic AI — BITS School of Management  
10+ years of domain expertise in manufacturing | Transitioning into AI/ML

[![GitHub](https://img.shields.io/badge/GitHub-ShivamSingh3406-black?style=flat-square&logo=github)](https://github.com/ShivamSingh3406)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  <i>Built with precision. Documented with purpose. Designed for impact.</i>
</p>
