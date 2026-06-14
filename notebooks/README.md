# Notebooks Guide

Run notebooks in order within each folder. All notebooks originally ran on local machines with Windows paths — update `data_path` variables to point to your local data directory.

---

## 01_data_processing/

| Notebook | What it does |
|----------|-------------|
| `01_data_merging_and_eda.ipynb` | Loads raw CPCB CSVs for NO₂, PM2.5, O₃; reshapes from wide to long format; merges on station ID and timestamp; adds spatial features (K-NN average) |
| `02_missing_data_analysis.ipynb` | Quantifies missing data across the 3 scenarios; checks lag feature availability; plots missingness patterns |

---

## 02_baseline_models/

| Notebook | What it does |
|----------|-------------|
| `01_univariate_random_forest.ipynb` | RF models for all 3 pollutants under STC, PCC, CC feature sets; 5-fold CV; saves `.joblib` models |
| `02_univariate_xgboost.ipynb` | XGBoost equivalents; includes K-NN spatial feature construction |
| `03_univariate_kriging_baseline.ipynb` | Kriging-style spatial interpolation baseline |
| `04_bivariate_rf_xgboost.ipynb` | MLR, XGBoost, and LSTM for bivariate scenarios; GeoDataFrame-based nearest-station lookup |

---

## 03_deep_learning/

| Notebook | What it does |
|----------|-------------|
| `01_lstm_univariate.ipynb` | Stacked LSTM for single-pollutant prediction; MinMaxScaler; EarlyStopping |
| `02_lstm_bivariate.ipynb` | LSTM predicting 2 pollutants simultaneously |
| `03_lstm_full_pipeline.ipynb` | Full training pipeline across all 12 univariate sub-cases; saves best models |
| `04_lstm_trivariate.ipynb` | LSTM predicting all 3 pollutants using time/location features only |

---

## 04_graph_neural_networks/

| Notebook | What it does |
|----------|-------------|
| `01_gnn_bivariate.ipynb` | StackedGNN (GCNConv + GATv2Conv) for bivariate scenario; dynamic graph construction from feature distances |
| `02_stacked_gnn_pipeline.ipynb` | Full GNN pipeline with visualisation of graph structure and architecture diagram |

---

## 05_advanced_models/

| File | What it does |
|------|-------------|
| `01_kan_univariate.ipynb` | KAN (Kolmogorov-Arnold Network) for PM2.5 prediction; uses `pykan` library |
| `02_kan_custom_implementation.ipynb` | Custom PyTorch KAN implementation without external library |
| `03_kriging_spatial_interpolation.ipynb` | Spatial-only interpolation using inverse-distance weighting |
| `04_transformer_model.py` | Standalone Transformer with 4-layer MHA; positional encoding for coordinates |

---

## 06_results_analysis/

> To be populated with comparison plots and final metric tables across all scenarios and models.
