# 📊 Model Performance Summary

All metrics are computed with **5-fold cross-validation** on held-out test sets.  
Dataset: CPCB spatio-temporal air quality data (2023–2025), ~4.1M rows, 28 stations.

---

## Scenario 1 — Univariate Imputation (2 pollutants available, 1 missing)

### Predicting PM2.5

| Feature Set | Model | MAE (µg/m³) | RMSE | R² | MAPE |
|-------------|-------|-------------|------|----|------|
| STC | Random Forest | 15.86 ± 0.06 | 26.69 | 0.91 | 44.4% |
| PCC | Random Forest | 17.12 ± 0.04 | 29.22 | 0.89 | 50.1% |
| CC | XGBoost | **6.73 ± 0.04** | **12.75** | **0.87** | 39.2% |
| CC | XGBoost (PCC) | 6.33 ± 0.03 | 12.11 | 0.88 | 37.9% |

### Predicting NO₂

| Feature Set | Model | MAE (µg/m³) | RMSE | R² | MAPE |
|-------------|-------|-------------|------|----|------|
| STC | Random Forest | 8.33 ± 0.05 | — | — | — |
| PCC | Random Forest | 6.33 ± 0.05 | — | — | — |
| CC | XGBoost | 7.91 ± 0.05 | — | — | — |

### Predicting Ozone (O₃)

| Feature Set | Model | MAE (µg/m³) | RMSE | R² | MAPE |
|-------------|-------|-------------|------|----|------|
| STC | Random Forest | 7.73 | — | — | — |
| PCC | Random Forest | 8.33 | — | — | — |
| CC | XGBoost | 7.52 | — | — | — |

---

## Scenario 2 — Bivariate Imputation (1 pollutant available, 2 missing)

### Predicting NO₂ + O₃ (given PM2.5)

| Model | NO₂ MAE | NO₂ R² | O₃ MAE | O₃ R² |
|-------|---------|--------|--------|--------|
| MLR (Linear Regression) | 65.01 | 0.04 | 22.44 | 0.09 |
| XGBoost | 31.78 | 0.70 | 11.88 | 0.68 |
| LSTM | 11.17 | 0.73 | — | — |

### Predicting PM2.5 + O₃ (given NO₂)

| Model | PM2.5 MAE | PM2.5 R² | O₃ MAE | O₃ R² |
|-------|-----------|---------|--------|--------|
| XGBoost | 10.44 | 0.73 | — | — |
| MLR | — | — | — | — |

### Predicting PM2.5 + NO₂ (given O₃)

| Model | PM2.5 MAE | PM2.5 R² | NO₂ MAE | NO₂ R² |
|-------|-----------|---------|---------|--------|
| XGBoost | 10.17 | 0.74 | 10.48 | 0.73 |

---

## Scenario 3 — Trivariate Imputation (all 3 missing, time/location only)

| Target | Model | MAE (µg/m³) | RMSE | R² |
|--------|-------|-------------|------|----|
| PM2.5 | LSTM | 18.18 | 27.84 | 0.73 |
| NO₂ | LSTM | 13.29 | 18.44 | 0.59 |
| O₃ | LSTM | — | — | — |

---

## GNN Results — StackedGNN (GCNConv + GATv2Conv)

Architecture: GCN layer → 2× GATv2 layers → BatchNorm → Dropout (0.3)  
Input: Spatial graph over 28 stations (k=5 KNN edges)

> Quantitative results pending full pipeline run on 2023–2025 data.

---

## Notes

- **STC** = Spatio-Temporal Covariates: lag features (t-1 day, t-1 week) + spatial neighbourhood averages  
- **PCC** = Pollutant Cross-Covariates: co-pollutant readings as direct predictors  
- **CC** = Combined Covariates: STC + PCC merged  
- All LSTM experiments used MinMaxScaler normalisation and EarlyStopping (patience=15)  
- KS-test (D-statistic) used to validate imputed distribution matches original: CC features gave best alignment (KS=0.1125)
