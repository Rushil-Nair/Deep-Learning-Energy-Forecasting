import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import holidays
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import sys
import logging
import warnings
import statsmodels.api as sm
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")
# Set plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)

# --- 2. CONFIGURATION (PC OPTIMIZED) ---
print("Starting Simple GRU Optimization (PC Mode)...")
BASE_PATH = 'data'
FILE_PATH = os.path.join(BASE_PATH, 'final_cleaned_data_energy_multicity.csv')
OUTPUT_DIR = BASE_PATH
TARGET = 'elec_mw'
CHAMPION_PATH = os.path.join(OUTPUT_DIR, "best_champion_gru_simple.pth")
RETRAINED_PATH = os.path.join(OUTPUT_DIR, "best_retrained_gru_simple.pth")
PARAM_PATH = os.path.join(OUTPUT_DIR, "best_params_gru_simple.json")

# Hardware Settings
NUM_WORKERS = 32
PERSISTENT_WORKERS = True
PIN_MEMORY = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tuning Constraints
N_TRIALS = 50
TIMEOUT_SECONDS = 7 * 60 * 60
MAX_EPOCHS_PER_TRIAL = 30
TRIAL_PATIENCE = 10

# --- 3. DATA LOADING ---
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found: {FILE_PATH}")

df = pd.read_csv(FILE_PATH, index_col=0, parse_dates=True)
df.index.name = 'datetime'

# --- 4. FEATURE ENGINEERING ---
def create_features(df):
    df_out = df.copy()
    df_out['year'] = df_out.index.year
    df_out['quarter'] = df_out.index.quarter
    df_out['month'] = df_out.index.month
    df_out['hour'] = df_out.index.hour
    df_out['dayofweek'] = df_out.index.dayofweek

    idx = np.arange(len(df_out))
    df_out['sin_8760h'] = np.sin(2 * np.pi * idx / 8760)
    df_out['cos_8760h'] = np.cos(2 * np.pi * idx / 8760)

    us_hols = holidays.US()
    df_out['is_holiday'] = df_out.index.map(lambda x: 1 if x in us_hols else 0)

    for lag in [1, 2, 3, 24, 48, 168]:
        df_out[f'lag_{lag}'] = df_out[TARGET].shift(lag)

    for w in [24, 168]:
        shifted = df_out[TARGET].shift(1)
        df_out[f'rm_{w}'] = shifted.rolling(w).mean()
        df_out[f'rs_{w}'] = shifted.rolling(w).std()
        df_out[f'rmed_{w}'] = shifted.rolling(w).median()

    temp_col = 'temp'
    if temp_col in df_out.columns:
        df_out['txh'] = df_out[temp_col] * df_out['hour']
        df_out['t2'] = df_out[temp_col] ** 2

    return df_out.dropna()

df = create_features(df)

# --- 5. SPLIT & SCALE ---
train_df = df.loc[:'2015-12-31']
val_df = df.loc['2016-01-01':'2016-12-31']
test_df = df.loc['2017-01-01':]

all_cols = [c for c in df.columns if c != TARGET]
scaler_x = StandardScaler().fit(train_df[all_cols])
scaler_y = StandardScaler().fit(train_df[[TARGET]])

LOOKBACK = 168
HORIZON = 24

def make_seq(df_in):
    X = scaler_x.transform(df_in[all_cols])
    y = scaler_y.transform(df_in[[TARGET]])
    Xs, ys = [], []
    for i in range(len(X) - LOOKBACK - HORIZON + 1):
        Xs.append(X[i : i+LOOKBACK])
        ys.append(y[i+LOOKBACK : i+LOOKBACK+HORIZON].flatten())
    return np.array(Xs), np.array(ys)

X_train, y_train = make_seq(train_df)
X_val, y_val = make_seq(val_df)
X_test, y_test_scaled = make_seq(test_df)

class TSData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.tensor(X).float(), torch.tensor(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# --- 6. MODEL DEFINITION (SIMPLE GRU) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(SimpleGRU, self).__init__()

        # Standard GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # LayerNorm
        self.ln = nn.LayerNorm(hidden_size)

        # Final Projection
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GRU Pass
        gru_out, _ = self.gru(x)

        # Last time step only
        last_step = gru_out[:, -1, :]

        # Norm -> Dropout -> Output
        out = self.ln(last_step)
        return self.fc(self.dropout(out))

# --- TRACKER CLASS ---
class BestModelTracker:
    def __init__(self):
        self.best_val_loss = float('inf')

    def check_and_save(self, current_val_loss, model):
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            if hasattr(model, '_orig_mod'):
                torch.save(model._orig_mod.state_dict(), CHAMPION_PATH)
            else:
                torch.save(model.state_dict(), CHAMPION_PATH)

tracker = BestModelTracker()

# --- 7. OBJECTIVE FUNCTION ---
def objective(trial):
    # Hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    train_loader = DataLoader(TSData(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(TSData(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)

    # Simple GRU
    model = SimpleGRU(len(all_cols), hidden_size, num_layers, HORIZON, dropout).to(device)
    try: model = torch.compile(model)
    except: pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_trial_loss = float('inf')
    patience_counter = 0

    for epoch in range(MAX_EPOCHS_PER_TRIAL):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                val_loss += criterion(model(X), y).item()

        avg_val_loss = val_loss / len(val_loader)

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_trial_loss:
            best_trial_loss = avg_val_loss
            patience_counter = 0
            tracker.check_and_save(avg_val_loss, model)
        else:
            patience_counter += 1
            if patience_counter >= TRIAL_PATIENCE:
                break

    return best_trial_loss

# --- 8. EXECUTE TUNING ---
sampler = TPESampler(seed=42, multivariate=True)
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

print(f"Running Simple GRU optimization on {torch.cuda.get_device_name(0)}...")
study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=True)

print("\n" + "="*40)
print(f"Best Val MSE: {study.best_value:.6f}")
print("Best Params:", study.best_params)
print("="*40)

with open(PARAM_PATH, 'w') as f:
    json.dump(study.best_params, f)

# --- 9. RETRAIN ---
print("\nRetraining Simple GRU with Best Parameters...")
best_params = study.best_params
model = SimpleGRU(
    len(all_cols),
    best_params['hidden_size'],
    best_params['num_layers'],
    HORIZON,
    best_params['dropout']
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
criterion = nn.MSELoss()
scheduler = OneCycleLR(optimizer, max_lr=best_params['lr'], epochs=50, steps_per_epoch=len(DataLoader(TSData(X_train, y_train), batch_size=best_params['batch_size'])))

full_train_loader = DataLoader(TSData(X_train, y_train), batch_size=best_params['batch_size'], shuffle=True, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)
full_val_loader = DataLoader(TSData(X_val, y_val), batch_size=best_params['batch_size'], shuffle=False, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)

best_retrain_loss = float('inf')
patience = 0

for epoch in range(50):
    model.train()
    train_l = 0
    for X, y in full_train_loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_l += loss.item()

    model.eval()
    val_l = 0
    preds, trues = [], []
    with torch.no_grad():
        for X, y in full_val_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = criterion(model(X), y)
            val_l += loss.item()
            preds.append(model(X).cpu().numpy())
            trues.append(y.cpu().numpy())

    avg_val = val_l/len(full_val_loader)
    val_r2 = r2_score(np.concatenate(trues), np.concatenate(preds))
    print(f"Retrain Ep {epoch+1}: Val {avg_val:.5f} R2 {val_r2:.4f}")

    if avg_val < best_retrain_loss:
        best_retrain_loss = avg_val
        torch.save(model.state_dict(), RETRAINED_PATH)
        patience = 0
    else:
        patience += 1
        if patience >= 10: break

# --- 10. EVALUATION ---
print("\nEvaluating Simple GRU...")
model.load_state_dict(torch.load(RETRAINED_PATH))
model.eval()

test_loader = DataLoader(TSData(X_test, y_test_scaled), batch_size=best_params['batch_size'], shuffle=False, num_workers=NUM_WORKERS)

preds = []
with torch.no_grad():
    for X, _ in test_loader:
        preds.append(model(X.to(device)).cpu().numpy())

yp = scaler_y.inverse_transform(np.concatenate(preds))
yt = scaler_y.inverse_transform(y_test_scaled)

# Metrics
mse = mean_squared_error(yt, yp)
rmse = np.sqrt(mse)
mae = mean_absolute_error(yt, yp)
r2 = r2_score(yt, yp)
mape = np.mean(np.abs((yt - yp) / (yt + 1e-6))) * 100
smape = 100/len(yt) * np.sum(2 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-6))

print("\n" + "="*40)
print("** FINAL TUNED SIMPLE GRU RESULTS **")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE:     {rmse:.2f} MW")
print(f"MAE:      {mae:.2f} MW")
print(f"MAPE:     {mape:.2f}%")
print(f"sMAPE:    {smape:.2f}%")
print("="*40)