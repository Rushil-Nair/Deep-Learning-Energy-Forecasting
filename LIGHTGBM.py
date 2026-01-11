import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import sys
import logging
import lightgbm as lgb
import optuna
import holidays
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import warnings

# --- 1. SETUP: Define Paths & Config ---
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

print("Starting LightGBM (Multi-City Data + Comprehensive Analysis)...")

# Detect current working directory to make paths relative
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data') # Assumes you have a 'data' folder
output_dir = os.path.join(current_dir, 'output') # Results will be saved here

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Path to your file
clean_file_path = os.path.join(data_dir, 'final_cleaned_data_energy_multicity.csv')
MODEL_SAVE_PATH = os.path.join(output_dir, 'LightGBM_MultiCity_Best.pkl')
TARGET = 'elec_mw'

# Config
N_FORECAST = 24
VAL_SPLIT_DATE = '2016-01-01'
TEST_SPLIT_DATE = '2017-01-01'
OPTUNA_TRIALS = 20
TIMEOUT_SECONDS = 3600 

# --- 2. LOAD AND PREPARE DATA ---
print(f"\nLoading dataset from {clean_file_path}...")
if not os.path.exists(clean_file_path):
    print(f"ERROR: Could not find file at: {clean_file_path}")
    print("Please ensure you have created a 'data' folder and placed the csv file inside it.")
    sys.exit(1)

# Robust loading
df = pd.read_csv(clean_file_path, index_col=0, parse_dates=True)
df.index.name = 'datetime' 

print(f"Successfully loaded data. Shape: {df.shape}")

# --- 3. FEATURE ENGINEERING ---
print("Applying Kaggle-style feature engineering...")

def create_full_features(df):
    df_out = df.copy()
    
    # 1. Time Features
    df_out['year'] = df_out.index.year
    df_out['quarter'] = df_out.index.quarter
    df_out['month'] = df_out.index.month
    df_out['hour'] = df_out.index.hour
    df_out['dayofweek'] = df_out.index.dayofweek
    
    # 2. Cyclical Encoding
    time_index = np.arange(len(df_out))
    df_out['sin_8760h'] = np.sin(2 * np.pi * (1 / 8760) * time_index)
    df_out['cos_8760h'] = np.cos(2 * np.pi * (1 / 8760) * time_index)
    
    # 3. Holidays
    us_holidays = holidays.US()
    df_out['is_holiday'] = df_out.index.map(lambda x: 1 if x in us_holidays else 0)
    
    # 4. Lag Features
    lags = [1, 2, 3, 24, 48, 168]
    for lag in lags:
        df_out[f'lag_{lag}h'] = df_out[TARGET].shift(lag)
        
    # 5. Rolling Statistics
    windows = [24, 168]
    for window in windows:
        shifted = df_out[TARGET].shift(1) 
        df_out[f'rolling_mean_{window}h'] = shifted.rolling(window=window).mean()
        df_out[f'rolling_std_{window}h'] = shifted.rolling(window=window).std()
        df_out[f'rolling_median_{window}h'] = shifted.rolling(window=window).median()
        df_out[f'rolling_q25_{window}h'] = shifted.rolling(window=window).quantile(0.25)
        df_out[f'rolling_q75_{window}h'] = shifted.rolling(window=window).quantile(0.75)
        
    # 6. Weather Interactions
    temp_col = 'temp'
    if temp_col in df_out.columns and 'hour' in df_out.columns:
        df_out['temp_hour_interact'] = df_out[temp_col] * df_out['hour']
        df_out['temp_sq'] = df_out[temp_col] ** 2
        
    return df_out

df = create_full_features(df)
df = df.dropna()

# --- 4. PREPARE TABULAR ARRAYS (MIMO Strategy) ---
print("\nCreating Multi-Output Targets...")
feature_cols = [c for c in df.columns if c != TARGET]

# Convert categoricals for LightGBM
cat_cols = ['hour', 'dayofweek', 'month', 'year', 'quarter', 'is_holiday']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

def create_tabular_io(df_subset, feature_cols, target_col, horizon=24):
    X_list, y_list = [], []
    
    data_feat = df_subset[feature_cols].values
    data_target = df_subset[target_col].values
    
    # MIMO: Input at t predicts t+1...t+24
    for i in range(len(df_subset) - horizon):
        X_list.append(data_feat[i])
        y_list.append(data_target[i+1 : i+1+horizon])
        
    return pd.DataFrame(X_list, columns=feature_cols), np.array(y_list)

# --- 5. SPLIT DATA ---
train_mask = df.index < VAL_SPLIT_DATE
val_mask = (df.index >= VAL_SPLIT_DATE) & (df.index < TEST_SPLIT_DATE)
test_mask = df.index >= TEST_SPLIT_DATE

print("Generating Train/Val/Test sets...")
X_train, y_train = create_tabular_io(df[train_mask], feature_cols, TARGET, N_FORECAST)
X_val, y_val = create_tabular_io(df[val_mask], feature_cols, TARGET, N_FORECAST)
X_test, y_test = create_tabular_io(df[test_mask], feature_cols, TARGET, N_FORECAST)

test_dates = df[test_mask].index[:-N_FORECAST] 

print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Val shape:   {X_val.shape}, {y_val.shape}")
print(f"Test shape:  {X_test.shape}, {y_test.shape}")

# --- 6. OPTUNA OPTIMIZATION ---
print("\nDefining Optuna objective...")
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'random_state': 42,
        # 'device': 'gpu', # Uncomment only if you have compiled LGBM with OpenCL
        'n_jobs': -1 # Use all M4 cores (Often faster than GPU for Tabular)
    }
    
    # Use MultiOutputRegressor with LGBM
    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return val_rmse

# Main execution block required for multiprocessing on Mac
if __name__ == "__main__":
    print(f"Starting Optuna (Max Trials: {OPTUNA_TRIALS})...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=TIMEOUT_SECONDS)

    print("\nOptuna Study Complete.")
    print(f"Best Val RMSE: {study.best_value:.4f}")
    print("Best Params:", study.best_params)

    # --- 7. TRAIN FINAL MODEL ---
    print("\nTraining Final Model on Train + Val...")

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    final_params = study.best_params
    final_params['n_jobs'] = -1 
    # final_params['device'] = 'gpu' # Uncomment only if compiled with OpenCL

    final_model = MultiOutputRegressor(lgb.LGBMRegressor(**final_params))

    final_model.fit(X_train_val, y_train_val)

    # Save Model
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- 8. FINAL EVALUATION ---
    print("\nEvaluating on Test Set...")
    y_pred = final_model.predict(X_test)

    # Overall Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    print("\n" + "="*40)
    print(f"** FINAL RESULTS: LightGBM + Multi-City Data **")
    print(f"MAE:  {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"R2:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("="*40 + "\n")

    # --- 9. ADVANCED PLOTTING & ANALYSIS ---
    print("Generating Analysis Plots...")

    # A. Feature Importance
    importances = np.zeros(len(feature_cols))
    for estimator in final_model.estimators_:
        importances += estimator.feature_importances_
    importances /= 24 # Average

    # Create DataFrame for plotting
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(20) # Top 20

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
    plt.title('Top 20 Feature Importances (Averaged over 24h Horizon)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LGBM_Feature_Importance.png'))
    # plt.show() # Commented out to prevent blocking execution if running in background

    # B. Error Distribution (Histogram)
    residuals = (y_test - y_pred).flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=100, kde=True, color='purple')
    plt.title('Error Distribution (Residuals)')
    plt.xlabel('Error (MW)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'LGBM_Error_Distribution.png'))

    # C. Horizon Accuracy
    horizon_rmse = [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(24)]
    horizon_r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(24)]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Forecast Horizon (Hours)')
    ax1.set_ylabel('RMSE (MW)', color=color)
    ax1.plot(range(1, 25), horizon_rmse, marker='o', color=color, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('R2 Score', color=color)
    ax2.plot(range(1, 25), horizon_r2, marker='s', color=color, label='R2')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Forecast Accuracy by Horizon')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'LGBM_Horizon_Accuracy.png'))

    # D. Full Forecast Plot (t+1)
    y_test_t1 = y_test[:, 0]
    y_pred_t1 = y_pred[:, 0]

    plt.figure(figsize=(18, 6))
    plt.plot(test_dates, y_test_t1, label='Actual (t+1)', alpha=0.6, color='navy')
    plt.plot(test_dates, y_pred_t1, label='LGBM Predicted (t+1)', alpha=0.7, color='orange', linestyle='--')
    plt.title(f'Full Test Set Forecast (t+1 Step) - R2: {r2:.4f}', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'LGBM_Full_Forecast.png'))