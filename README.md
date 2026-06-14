# 🌫️ Spatio-Temporal Air Quality Imputation using ML & Deep Learning

> **B.Tech Honors Research Project** — IIIT Sri City | Mentor: Dr. Mainak Thakur  
> Reconstructing missing PM2.5, NO₂, and O₃ readings across 28 CPCB monitoring stations (Delhi, India) using classical ML, LSTM, GNN, Transformer, and KAN architectures across three real-world missingness scenarios.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-blue?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Paper accepted:** *"Improving Air Quality Data by Treating Missing Values with Machine Learning and Spatio-Temporal Variables"* — IGARSS 2025 (IEEE International Geoscience and Remote Sensing Symposium)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Statistics](#-dataset-statistics)
- [System Architecture](#-system-architecture)
- [Missing Data Scenarios](#-missing-data-scenarios)
- [Feature Engineering](#-feature-engineering)
- [Models Implemented](#-models-implemented)
- [Evaluation Protocol](#-evaluation-protocol)
- [Key Results](#-key-results)
- [Engineering Challenges](#-engineering-challenges)
- [Lessons Learned](#-lessons-learned)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Reproducibility](#-reproducibility)
- [Future Work](#-future-work)
- [References](#-references)

---

## 📌 Problem Statement

Air quality monitoring stations across Delhi frequently experience **sensor failures, power outages, and maintenance interruptions**, causing 9–12% of pollutant observations to go missing. These gaps directly degrade downstream forecasting accuracy and environmental policy decisions.

Traditional methods (mean substitution, linear interpolation, Kriging) fail because they ignore:
- **Temporal dependencies** — pollutant levels at t depend on t-1 day and t-1 week
- **Spatial correlations** — neighbouring stations share pollutant patterns
- **Inter-pollutant interactions** — NO₂ drives O₃ formation; PM2.5 correlates with NO₂

This project builds an end-to-end **ML-based imputation framework** that models all three dimensions simultaneously across three progressively difficult missingness scenarios.

---

## 📊 Dataset Statistics

| Property | Value |
|----------|-------|
| **Source** | Central Pollution Control Board (CPCB), India |
| **Study Area** | Delhi NCR, India |
| **Time Period** | January 2023 – December 2025 |
| **Temporal Resolution** | 15-minute intervals |
| **Total Records** | ~4.1 million observations |
| **Monitoring Stations** | 28 CPCB stations |
| **Target Pollutants** | PM2.5, NO₂, O₃ (Ozone) |
| **Missing Rate (PM2.5)** | 9.6% |
| **Missing Rate (NO₂)** | 8.93% |
| **Missing Rate (O₃)** | 11.41% |
| **Total Experiments Run** | 50+ |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CPCB Raw CSVs (~4.1M rows)                       │
│            PM2.5 · NO₂ · O₃ across 28 stations                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Cleaning & Merging                          │
│   Timestamp alignment · Deduplication · Station ID mapping         │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  Missingness Analysis                               │
│   Classify each row → Univariate / Bivariate / Trivariate          │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   Feature Engineering                               │
│         STC Features │ PCC Features │ CC Features (STC+PCC)        │
│    Temporal lags · Spatial KNN avg · Inter-pollutant values        │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     Model Training                                  │
│   RF · XGBoost · LSTM · GNN (StackedGNN) · Transformer · KAN      │
│              5-Fold Cross-Validation (80/20 split)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      Evaluation                                     │
│              MAE · RMSE · R² · MAPE · KS-Test                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  Reconstructed Dataset                              │
│           Complete air quality records for downstream use          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Missing Data Scenarios

Three progressively harder missingness patterns were identified in the real CPCB data and handled separately:

| Scenario | Observed | Missing | Complexity | Frequency |
|----------|----------|---------|------------|-----------|
| **Univariate** | 2 pollutants | 1 pollutant | Low | 12.05% |
| **Bivariate** | 1 pollutant | 2 pollutants | Moderate | 1.16% |
| **Trivariate** | 0 pollutants | All 3 | High | 5.18% |

Each scenario spawns multiple sub-cases (e.g., univariate has 3 sub-cases × 3 feature sets × 3 models = 27 experiments for that scenario alone). Models are **always trained on complete records** with artificial masking, preventing leakage.

---

## ⚙️ Feature Engineering

Three feature strategies were designed and systematically benchmarked:

### STC — Spatio-Temporal Covariates
Uses the **same pollutant's own history** and spatial neighbourhood:

| Feature | Description |
|---------|-------------|
| `X_prev_day` | Pollutant value 24h prior (lag-1) |
| `X_prev_week` | Pollutant value 168h prior (lag-7) |
| `Spatial_Avg_X` | Mean of 5 nearest stations via KNN |
| `Lat, Lon, Time` | Location and timestamp |

### PCC — Pollutant Cross-Covariates
Uses **other pollutants** as predictors (exploiting chemical interactions):
```
X_target(t) = f(Time, Lat, Lon, OtherPollutants, Their_Lags, Their_SpatialAvg)
```
Intuition: NO₂ influences O₃ formation; PM2.5 formation correlates with NO₂.

### CC — Combined Covariates *(generally best)*
Merges STC + PCC into a full feature vector:
```
X_t = f(Time, Lat, Lon, Target_STC, OtherPollutants, Their_STC)
```

---

## 🧠 Models Implemented

### Classical Baselines
| Model | Why Used |
|-------|----------|
| **Random Forest (RF)** | Handles nonlinear relationships, robust to noisy sensor data |
| **XGBoost** | High accuracy on tabular data via gradient boosting |
| **MLR (Linear Regression)** | Establishes a statistical lower bound |
| **Kriging / IDW** | Spatial interpolation baseline |

### Deep Learning Models
| Model | Architecture | Key Design Choice |
|-------|-------------|-------------------|
| **LSTM** | Stacked LSTM + Dense | Temporal sequence modelling with EarlyStopping |
| **Transformer** | 4-layer, 8-head MHA + GlobalAvgPooling | Sinusoidal positional encoding for Lat/Lon |
| **StackedGNN** | GCNConv + 2×GATv2Conv + BatchNorm | KNN graph over 28 stations; attention-based spatial aggregation |
| **KAN** | Kolmogorov-Arnold Network (width=[5,5,1]) | Learnable B-spline activations; novel 2024 architecture |

---

## 📐 Evaluation Protocol

- **Training data:** Complete observations only (no imputed values used in training)
- **Artificial masking:** Values masked on complete records to generate ground truth for evaluation
- **Split:** 80% train / 20% validation
- **Cross-validation:** 5-fold CV for all models
- **Distribution validation:** Kolmogorov-Smirnov (KS) test to verify imputed values match real data distribution
- **Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(|y - ŷ|) | Average error magnitude |
| **RMSE** | √mean((y - ŷ)²) | Penalises large errors |
| **R²** | 1 - SS_res/SS_tot | Goodness of fit (1 = perfect) |
| **MAPE** | mean(|y - ŷ|/y) × 100 | Scale-independent error % |
| **KS Statistic** | max(|F₁(x) - F₂(x)|) | Distribution alignment |

---

## 📈 Key Results

### Univariate Scenario — Predicting PM2.5 (given NO₂ + O₃)

| Feature Set | Model | MAE (µg/m³) | RMSE | R² |
|-------------|-------|-------------|------|----|
| STC | Random Forest | 15.86 ± 0.06 | 26.69 | 0.91 |
| PCC | Random Forest | 17.12 ± 0.04 | 29.22 | 0.89 |
| CC | XGBoost | **6.73 ± 0.04** | 12.75 | **0.87** |
| CC (PCC only) | XGBoost | 6.33 ± 0.03 | 12.11 | 0.88 |

### Bivariate Scenario — Predicting NO₂ + O₃ (given PM2.5 only)

| Model | NO₂ MAE | NO₂ R² | O₃ MAE | O₃ R² |
|-------|---------|--------|--------|--------|
| MLR | 65.01 | 0.04 | 22.44 | 0.09 |
| XGBoost | 31.78 | 0.70 | 11.88 | 0.68 |
| LSTM | **11.17** | **0.73** | — | — |

### Trivariate Scenario — Predicting all 3 (time + location only)

| Target | Model | MAE | RMSE | R² |
|--------|-------|-----|------|----|
| PM2.5 | LSTM | 18.18 | 27.84 | 0.73 |
| NO₂ | LSTM | 13.29 | 18.44 | 0.59 |

### KS-Test — Distribution Alignment of Imputed Values

| Feature Set | KS Statistic (lower = better) |
|-------------|-------------------------------|
| STC | 0.1260 |
| PCC | 0.1217 |
| **CC** | **0.1125** |

CC features produce imputations closest to the real data distribution.

---

## 🔧 Engineering Challenges

- **Data scale:** Merging thousands of individual CPCB CSVs with inconsistent column naming and encoding
- **Irregular timestamps:** 15-minute intervals with gaps, daylight-saving anomalies, and station-specific outage patterns
- **Temporal leakage prevention:** Ensuring lag features at test time are never computed from future observations
- **Graph construction:** Building a meaningful spatial graph over 28 stations where edges reflect feature-space proximity, not just geographic distance
- **Scenario classification:** Efficiently labelling 4.1M rows into the correct missingness category per row per timestamp
- **Memory management:** Processing full dataset in chunks to avoid OOM errors during spatial KNN feature generation
- **Model variance:** Deep learning models (especially GNN) required careful hyperparameter tuning to match XGBoost stability

---

## 💡 Lessons Learned

- **XGBoost consistently outperformed deep learning** on structured tabular air quality data — gradient-boosted trees remain hard to beat when features are well-engineered
- **Spatial features mattered more than long temporal windows** — the 5-station KNN average was the single most predictive feature across almost all scenarios
- **CC features dominated STC and PCC individually** in 9 out of 12 scenario/model combinations, validating the hypothesis that inter-pollutant chemistry adds signal
- **LSTM outperformed RF and XGBoost in the bivariate scenario** — when fewer features are available, sequential memory becomes more valuable
- **KS-test is essential** — visual inspection of imputed values alone is insufficient; distributional alignment must be verified statistically
- **Transformer performance was constrained by the small number of stations (28)** — attention mechanisms benefit from larger graphs or longer sequences
- **Deep learning models required 5–10× more hyperparameter tuning** than tree-based models to achieve comparable performance

---

## 🏗️ Repository Structure

```
air-quality-imputation/
│
├── notebooks/
│   ├── README.md                         # Notebook guide (what each one does)
│   ├── 01_data_processing/
│   │   ├── 01_data_merging_and_eda.ipynb
│   │   └── 02_missing_data_analysis.ipynb
│   ├── 02_baseline_models/
│   │   ├── 01_univariate_random_forest.ipynb
│   │   ├── 02_univariate_xgboost.ipynb
│   │   ├── 03_univariate_kriging_baseline.ipynb
│   │   └── 04_bivariate_rf_xgboost.ipynb
│   ├── 03_deep_learning/
│   │   ├── 01_lstm_univariate.ipynb
│   │   ├── 02_lstm_bivariate.ipynb
│   │   ├── 03_lstm_full_pipeline.ipynb
│   │   └── 04_lstm_trivariate.ipynb
│   ├── 04_graph_neural_networks/
│   │   ├── 01_gnn_bivariate.ipynb
│   │   └── 02_stacked_gnn_pipeline.ipynb
│   ├── 05_advanced_models/
│   │   ├── 01_kan_univariate.ipynb
│   │   ├── 02_kan_custom_implementation.ipynb
│   │   └── 03_kriging_spatial_interpolation.ipynb
│   └── 06_results_analysis/              # Comparison plots (WIP)
│
├── src/
│   └── transformer_model.py              # Standalone Transformer (4-layer, 8-head)
│
├── results/
│   └── metrics_summary.md               # All MAE/RMSE/R² tables
│
├── docs/
│   └── architecture.md                  # Detailed model architecture notes
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/air-quality-imputation.git
cd air-quality-imputation

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook notebooks/
```

> **Data:** Raw CPCB data is not included due to file size (~2.3 GB). Download from [cpcb.nic.in](https://cpcb.nic.in) and place in `data/raw/`. The merging pipeline in `notebooks/01_data_processing/` reconstructs the full dataset.

---

## 🔁 Reproducibility

- Fixed random seeds (`random_state=42`) across all sklearn and XGBoost models
- PyTorch `torch.manual_seed(42)` set before all GNN and KAN experiments
- MinMaxScaler fitted only on training fold data in each CV split (no leakage)
- All experiments logged with consistent metric reporting format
- Model checkpoints saved as `.keras` / `.joblib` (excluded from repo via `.gitignore`; available on request)

---

## 🚀 Future Work

- **Real-time imputation API** using FastAPI — serve predictions via HTTP for live CPCB feed integration
- **Meteorological covariates** — add temperature, humidity, wind speed/direction to improve trivariate scenario accuracy
- **Dockerised deployment** — containerise the full inference pipeline for reproducible production use
- **Automated retraining pipeline** using Apache Airflow — detect data drift and trigger retraining monthly
- **Model monitoring** — track imputation quality metrics in production using evidently.ai
- **Extended architectures** — test TFT (Temporal Fusion Transformer) and ST-GNN for joint spatio-temporal modelling

---

## 📚 References

1. Datla, M. V., Mandal, S., & Thakur, M. — *"Improving Air Quality Data by Treating Missing Values with Machine Learning and Spatio-Temporal Variables"* — **IGARSS 2025**
2. Shoari Nejad et al. — *"A Transformer-based Model for Multivariate Temporal Sensor Data with Missing Values"* — ICARUS, Maynooth University
3. Alsaber et al. — *"Handling Complex Missing Data Using Random Forest for Air Quality Monitoring"* — Int. J. Environmental Research and Public Health
4. Liu et al. — *"KAN: Kolmogorov-Arnold Networks"* — arXiv 2024
5. Brody et al. — *"How Attentive are Graph Attention Networks?"* (GATv2) — ICLR 2022

---

## 👤 Authors

**M. Vikas Reddy** (S20240010141) · **D. Naga Varshith** (S20240010063)  
IIIT Sri City | B.Tech Honors | Mentor: Dr. Mainak Thakur

---

*Work-in-progress: Next semester will extend results with 2023–2025 full evaluation and deploy a real-time imputation API.*
