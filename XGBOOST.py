import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import sys
import xgboost as xgb
import optuna
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import warnings

# --- 1. SETUP: Define Paths & Config ---
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

print("Starting XGBoost (Multi-City Data) on M4 CPU (Highly Optimized)...")

# Detect current working directory
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data') 
output_dir = os.path.join(current_dir, 'output') 

os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

clean_file_path = os.path.join(data_dir, 'final_cleaned_data_energy_multicity.csv')
MODEL_SAVE_PATH = os.path.join(output_dir, 'XGBoost_MultiCity_Best.pkl')
TARGET = 'elec_mw'

# Config
N_FORECAST = 24
VAL_SPLIT_DATE = '2016-01-01'
TEST_SPLIT_DATE = '2017-01-01'
OPTUNA_TRIALS = 20 
TIMEOUT_SECONDS = 3600 

# --- 2. LOAD DATA ---
print(f"\nLoading dataset from {clean_file_path}...")
if not os.path.exists(clean_file_path):
    print(f"ERROR: Could not find file at: {clean_file_path}")
    sys.exit(1)

df = pd.read_csv(clean_file_path, index_col=0, parse_dates=True)
df.index.name = 'datetime' 
print(f"Successfully loaded data. Shape: {df.shape}")

# --- 3. FEATURE ENGINEERING ---
print("Applying Feature Engineering...")

def create_full_features(df):
    df_out = df.copy()
    
    # Time Features
    df_out['year'] = df_out.index.year
    df_out['quarter'] = df_out.index.quarter
    df_out['month'] = df_out.index.month
    df_out['hour'] = df_out.index.hour
    df_out['dayofweek'] = df_out.index.dayofweek
    
    # Cyclical Encoding
    time_index = np.arange(len(df_out))
    df_out['sin_8760h'] = np.sin(2 * np.pi * (1 / 8760) * time_index)
    df_out['cos_8760h'] = np.cos(2 * np.pi * (1 / 8760) * time_index)
    
    # Holidays
    us_holidays = holidays.US()
    df_out['is_holiday'] = df_out.index.map(lambda x: 1 if x in us_holidays else 0)
    
    # Lag Features
    lags = [1, 2, 3, 24, 48, 168]
    for lag in lags:
        df_out[f'lag_{lag}h'] = df_out[TARGET].shift(lag)
        
    # Rolling Statistics
    windows = [24, 168]
    for window in windows:
        shifted = df_out[TARGET].shift(1) 
        df_out[f'rolling_mean_{window}h'] = shifted.rolling(window=window).mean()
        df_out[f'rolling_std_{window}h'] = shifted.rolling(window=window).std()
        df_out[f'rolling_median_{window}h'] = shifted.rolling(window=window).median()
        df_out[f'rolling_q25_{window}h'] = shifted.rolling(window=window).quantile(0.25)
        df_out[f'rolling_q75_{window}h'] = shifted.rolling(window=window).quantile(0.75)
        
    # Weather Interactions
    if 'temp' in df_out.columns and 'hour' in df_out.columns:
        df_out['temp_hour_interact'] = df_out['temp'] * df_out['hour']
        df_out['temp_sq'] = df_out['temp'] ** 2
        
    return df_out

df = create_full_features(df)
df = df.dropna()

# --- 4. PREPARE TARGETS ---
print("Creating Multi-Output Targets...")
feature_cols = [c for c in df.columns if c != TARGET]

# Categoricals (XGBoost handles these well with enable_categorical=True)
cat_cols = ['hour', 'dayofweek', 'month', 'year', 'quarter', 'is_holiday']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

def create_tabular_io(df_subset, feature_cols, target_col, horizon=24):
    X_list, y_list = [], []
    data_feat = df_subset[feature_cols].values
    data_target = df_subset[target_col].values
    
    for i in range(len(df_subset) - horizon):
        X_list.append(data_feat[i])
        y_list.append(data_target[i+1 : i+1+horizon])
        
    return pd.DataFrame(X_list, columns=feature_cols), np.array(y_list)

# --- 5. SPLIT DATA ---
train_mask = df.index < VAL_SPLIT_DATE
val_mask = (df.index >= VAL_SPLIT_DATE) & (df.index < TEST_SPLIT_DATE)
test_mask = df.index >= TEST_SPLIT_DATE

X_train, y_train = create_tabular_io(df[train_mask], feature_cols, TARGET, N_FORECAST)
X_val, y_val = create_tabular_io(df[val_mask], feature_cols, TARGET, N_FORECAST)
X_test, y_test = create_tabular_io(df[test_mask], feature_cols, TARGET, N_FORECAST)

test_dates = df[test_mask].index[:-N_FORECAST] 

# --- 6. OPTUNA (CPU Optimized) ---
print("\nDefining Optuna objective (M4 CPU)...")
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    params = {
        # --- CPU SETTINGS FOR MAC M4 ---
        'device': 'cpu',         # Fallback to standard CPU
        'n_jobs': -1,            # Use ALL M4 Cores (Very Fast)
        'tree_method': 'hist',   # Fastest histogram-based method
        # -------------------------------
        
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'random_state': 42,
        'enable_categorical': True,
    }
    
    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, preds))
    return val_rmse

if __name__ == "__main__":
    print(f"Starting Optuna (Max Trials: {OPTUNA_TRIALS})...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=TIMEOUT_SECONDS)

    print("\nOptuna Study Complete.")
    print(f"Best Val RMSE: {study.best_value:.4f}")

    # --- 7. TRAIN FINAL MODEL ---
    print("\nTraining Final Model...")
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    final_params = study.best_params
    final_params['device'] = 'cpu'
    final_params['n_jobs'] = -1
    final_params['tree_method'] = 'hist'
    final_params['enable_categorical'] = True

    final_model = MultiOutputRegressor(xgb.XGBRegressor(**final_params))
    final_model.fit(X_train_val, y_train_val)
    print("Final model trained on M4 CPU.")

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- 8. EVALUATION ---
    print("\nEvaluating on Test Set...")
    y_pred = final_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    print("\n" + "="*40)
    print(f"** FINAL RESULTS (XGBoost CPU) **")
    print(f"MAE:  {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"R2:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("="*40 + "\n")

    # --- 9. PLOTTING ---
    print("Generating Plots...")
    
    # Feature Importance
    importances = np.zeros(len(feature_cols))
    for estimator in final_model.estimators_:
        importances += estimator.feature_importances_
    importances /= 24 

    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=fi_df, palette='magma')
    plt.title('XGBoost Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'XGB_Feature_Importance.png'))

    # Forecast Plot
    y_test_t1 = y_test[:, 0]
    y_pred_t1 = y_pred[:, 0]

    plt.figure(figsize=(18, 6))
    plt.plot(test_dates, y_test_t1, label='Actual', alpha=0.6, color='navy')
    plt.plot(test_dates, y_pred_t1, label='Predicted', alpha=0.7, color='orange', linestyle='--')
    plt.title(f'Test Set Forecast (t+1) - R2: {r2:.4f}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'XGB_Forecast.png'))
