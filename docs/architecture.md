# 🏛️ Model Architecture Details

## 1. StackedGNN (Graph Neural Network)

Designed to exploit **spatial relationships** between the 28 CPCB monitoring stations.

```
Input node features (per station):
  - NO2_prev_day, PM2.5_prev_day, Ozone_prev_day
  - NO2_prev_week, PM2.5_prev_week, Ozone_prev_week
  - Lat, Lon

Graph construction: K-Nearest Neighbours (k=5) on feature distance

Architecture:
  GCNConv(input_dim → hidden_dim)
      ↓
  GATv2Conv(hidden_dim → hidden_dim, heads=2)
      ↓
  GATv2Conv(hidden_dim → hidden_dim, heads=2)
      ↓
  BatchNorm1d
      ↓
  Dropout(0.3)
      ↓
  Linear(hidden_dim → output_dim)

Training: Adam, MSE loss, 5-fold CV
```

---

## 2. LSTM (Long Short-Term Memory)

Used for all three missing data scenarios.

```
Input shape: (batch, timesteps, features)

Univariate scenario features:
  [Time, Lat, Lon, pollutant_prev_day, pollutant_prev_week,
   Spatial_Avg_pollutant, co-pollutant values]

Architecture:
  LSTM(units=128, return_sequences=True)
      ↓
  Dropout(0.3)
      ↓
  LSTM(units=64)
      ↓
  Dense(32, relu)
      ↓
  Dense(output_dim)   ← 1 for univariate, 2 for bivariate, 3 for trivariate

Training:
  - MinMaxScaler normalisation
  - EarlyStopping (patience=15, restore_best_weights=True)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - Batch size: 32, Max epochs: 50–150
```

---

## 3. Transformer

Multi-head self-attention for spatio-temporal sequence modelling.

```
Input: sliding windows of n_steps=4 timesteps

Positional encoding:
  Lat_sin = sin(radians(Lat))
  Lat_cos = cos(radians(Lat))
  Lon_sin / Lon_cos similarly

Architecture (4 stacked blocks):
  MultiHeadAttention(num_heads=8, key_dim=input_dim)
      ↓ residual + LayerNorm
  FeedForward(Dense(128, relu) → Dense(input_dim))
      ↓ residual + LayerNorm
      ↓ [×4]
  GlobalAveragePooling1D
      ↓
  Dense(128, relu) + Dropout(0.4)
      ↓
  Dense(3)  ← predicts [NO2, PM2.5, Ozone] simultaneously

Training: Adam(lr=1e-3), MSE loss, EarlyStopping(patience=15), max 150 epochs
```

---

## 4. KAN (Kolmogorov-Arnold Network)

Novel architecture (Liu et al., 2024) replacing fixed activation functions with learned B-spline functions on edges.

```
Architecture: KAN(width=[5, 5, 1], grid=3, k=3)

Input features (5): [Ozone, NO2, Lat, Lon, Time]
Output: PM2.5 prediction

Advantages over MLP:
  - Interpretable activation functions
  - Potentially better extrapolation
  - Fewer parameters for equivalent expressivity

Note: Experimental implementation; results pending full hyperparameter tuning.
```

---

## 5. Feature Engineering Strategies

### STC — Spatio-Temporal Covariates
Uses only the **same pollutant's own history** + spatial context:
```
Features for predicting PM2.5:
  [Time, Lat, Lon, PM2.5_prev_day, PM2.5_prev_week, Spatial_Avg_PM2.5]
```

### PCC — Pollutant Cross-Covariates
Uses **other pollutants** as predictors (no temporal lag):
```
Features for predicting PM2.5:
  [Time, Lat, Lon, NO2, Ozone]
```

### CC — Combined Covariates
Merges STC + PCC:
```
Features for predicting PM2.5:
  [Time, Lat, Lon, PM2.5_prev_day, PM2.5_prev_week,
   Spatial_Avg_PM2.5, NO2, Ozone]
```
CC consistently performs best across scenarios and models.
