# Interpretable Ensemble Learning Framework (IELF) for Heart Disease Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Scikit-Learn | XGBoost | Interpret](https://img.shields.io/badge/Framework-XGBoost%20%7C%20EBM%20%7C%20SHAP-orange)](#tech-stack)

An open-source implementation and extension of the **Interpretable Ensemble Learning Framework (IELF)** for Cardiovascular Disease (CVD) risk prediction. This project pairs high-capacity ensemble gradient boosting (**XGBoost**) with intrinsically transparent generalized additive models (**Explainable Boosting Machines - EBM**), backed by **SHAP** and **LIME** for post-hoc clinical decision explanation.

---

## 📌 Base Paper Reference

* **Title:** Ensemble Learning Approach with Explainable AI for Improved Heart Disease Prediction
* **Authors:** Ayomide Adekoya, Faisal Saeed, Wad Ghaban, Sultan Noman Qasem
* **Journal:** *Frontiers in Pharmacology* (2025)
* **DOI / PMC:** [10.3389/fphar.2025.1654681](https://doi.org/10.3389/fphar.2025.1654681) | [PMC12739384](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12739384/)

---

## 🔬 Motivation & Research Gap

While modern gradient boosting and deep neural networks achieve high benchmark accuracy for cardiovascular triage, they typically operate as **uninterpretable "black boxes"**, preventing clinicians from validating the biological rationale behind individual patient predictions. Conversely, glass-box models (such as logistic regression or single decision trees) frequently underperform when capturing non-linear interactions across high-dimensional biomarker data.

### Proposed Improvement / Extension in this Repository
1. **Hybrid Architecture:** Soft-voting ensemble combining **Explainable Boosting Machines (EBM)** and **XGBoost** to achieve competitive discrimination while preserving interpretability.
2. **Leakage-Safe Validation:** Full nested 5-fold cross-validation with train-only resampling (SMOTE) and preprocessing.
3. **Multi-Scale Interpretability:**
   * **Global explanations:** Feature attribution and stability quantification across cross-validation folds using Kendall's $\tau$ and Overlap@10.
   * **Local explanations:** Case-level breakdown combining additive EBM risk curves, SHAP waterfall plots, and LIME explanations.
4. **Calibration & Subgroup Fairness:** Brier score evaluation and subgroup error profiling across gender and age strata.

---

## 📊 Benchmark Datasets

| Dataset | Cohort Size | Features | Class Ratio | Primary Clinical Endpoint |
|---|:---:|:---:|:---:|---|
| **UCI Cleveland** | 303 | 13 | Balanced (~54% CVD) | Diagnostic presence of angiographic CAD ($\ge 50\%$ stenosis) |
| **Framingham Heart Study** | 4,240 | 15 | Imbalanced (5.58 : 1) | 10-year prospective risk of incident coronary heart disease (CHD) |

---

## 📈 Base Paper Benchmark Performance

| Evaluation Metric | Cleveland Dataset ($n=303$) | Framingham Dataset ($n=4,240$) |
|---|:---:|:---:|
| **ROC-AUC** | **0.899** | **0.696** |
| **Accuracy** | **88.5%** | **82.6%** |
| **Cross-Validation** | Nested 5-Fold CV | Nested 5-Fold CV (Train-only SMOTE) |
| **Explainability** | Stable Global Attributions | EBM Additive + SHAP + LIME |

---

## 🏗 Repository Structure

```text
heart-disease-ielf-xai/
├── data/
│   ├── raw/                  # Original Cleveland & Framingham CSV datasets
│   └── processed/            # Scaled, encoded, and harmonized splits
├── notebooks/
│   ├── 01_eda_and_harmonization.ipynb
│   ├── 02_model_training_nested_cv.ipynb
│   └── 03_xai_shap_lime_stability.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading & schema standardization
│   ├── preprocessing.py      # Missing value imputation, scaling, train-only SMOTE
│   ├── feature_selection.py  # Hybrid feature selection (SHAP + EBM + ANOVA/Chi2)
│   ├── models.py             # EBM, XGBoost, and soft-voting ensemble definitions
│   ├── evaluation.py         # AUC, Sensitivity, Specificity, Brier score calibration
│   └── explainability.py     # SHAP, LIME, and explanation stability (Kendall's tau)
├── configs/
│   └── config.yaml           # Hyperparameters, grid-search boundaries, seeds
├── results/
│   ├── figures/              # Calibration curves, SHAP summary & waterfall plots
│   └── metrics_summary.csv   # Aggregated cross-validation benchmarks
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/heart-disease-ielf-xai.git
cd heart-disease-ielf-xai

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dependencies (`requirements.txt`)
* `numpy>=1.23.0`
* `pandas>=2.0.0`
* `scikit-learn>=1.3.0`
* `xgboost>=2.0.0`
* `interpret>=0.6.0` (for Explainable Boosting Machines)
* `shap>=0.44.0`
* `lime>=0.2.0`
* `scipy>=1.11.0`
* `matplotlib>=3.7.0`
* `seaborn>=0.12.0`

### 3. Run Preprocessing & Training
```bash
# Run complete nested cross-validation pipeline on Cleveland benchmark
python src/train.py --dataset cleveland --folds 5

# Run training on Framingham longitudinal cohort with train-only resampling
python src/train.py --dataset framingham --folds 5 --smote
```

### 4. Generate Explainability Reports
```bash
# Generates global summary plots, local waterfall graphs, and stability metrics
python src/explain.py --model_path results/ielf_model.pkl --sample_idx 42
```

---

## 🧪 Methodological Highlights & Best Practices

1. **Strict Leakage Prevention:** Preprocessing steps (median imputation, standard scaling) and SMOTE resampling are fitted strictly within the inner training folds of the cross-validation loop.
2. **Clinical Calibration:** Beyond raw accuracy, predicted probabilities are calibrated (Platt scaling / Isotonic regression) to ensure clinical risk scores represent genuine likelihood of cardiac events.
3. **Metric Stability:** Feature attributions are evaluated across all outer test folds to verify that feature rankings do not fluctuate erratically under cohort variation.

---

## 📜 Citation

If you use this repository or build upon this implementation, please cite the original base paper:

```bibtex
@article{adekoya2025ensemble,
  title={Ensemble learning approach with explainable AI for improved heart disease prediction},
  author={Adekoya, Ayomide and Saeed, Faisal and Ghaban, Wad and Qasem, Sultan Noman},
  journal={Frontiers in Pharmacology},
  volume={16},
  pages={1654681},
  year={2025},
  publisher={Frontiers Media SA},
  doi={10.3389/fphar.2025.1654681}
}
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
