import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# --- Fixed static path config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_FOLDER = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_FOLDER, exist_ok=True)

DRUG_COLUMNS = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

# --- Helper Function (from your script) ---
def smape(y_true, y_pred):
    """ Symmetric MAPE (handles zero values safely) """
    epsilon = np.finfo(np.float64).eps
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

def get_forecast_results(file_path, freq='D', future_periods=365):
    """
    Main function to run the forecast for all 8 drugs.
    This contains the exact logic from your run_forecast function.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
        
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    
    all_results = []

    for col in DRUG_COLUMNS:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in the uploaded file. Skipping.")
            continue

        print(f"\nTraining model for {col} ...")

        # Prepare dataframe for Prophet
        dfg = df[['datum', col]].rename(columns={'datum': 'ds', col: 'y'}).dropna()
        
        if len(dfg) < 2:
            print(f"Skipping {col} due to insufficient data.")
            continue

        # Train-test split
        train_size = int(len(dfg) * 0.8)
        train, test = dfg.iloc[:train_size], dfg.iloc[train_size:]

        # Train Prophet model
        model = Prophet()
        model.fit(train)

        # Predict test period + future
        total_future_periods = len(test) + future_periods
        future = model.make_future_dataframe(periods=total_future_periods, freq=freq)
        forecast = model.predict(future)

        # Extract forecast for test set using your script's exact logic
        forecast_test = forecast.iloc[-(len(test) + future_periods):-future_periods][['ds', 'yhat']]
        test_results = test.merge(forecast_test, on="ds")

        if test_results.empty:
            print(f"Warning: Could not align test data for {col}. Check data frequency. Skipping metrics and plot.")
            continue
            
        # Metrics
        mae = mean_absolute_error(test_results['y'], test_results['yhat'])
        rmse = np.sqrt(mean_squared_error(test_results['y'], test_results['yhat']))
        smape_val = smape(test_results['y'], test_results['yhat'])
        accuracy = max(0, 100 - min(smape_val, 100))

        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | SMAPE: {smape_val:.2f}% | Accuracy: {accuracy:.2f}%")

        # --- Plotting (saving instead of showing) ---
        plt.figure(figsize=(12, 6))
        plt.plot(train['ds'], train['y'], label="Train", color="blue")
        plt.plot(test['ds'], test['y'], label="Test (Actual)", color="green")
        plt.plot(test_results['ds'], test_results['yhat'], label="Test (Predicted)", color="red")
        
        # Future predictions
        forecast_future = forecast.tail(future_periods)
        plt.plot(forecast_future['ds'], forecast_future['yhat'], label="Future Forecast", color="orange", linestyle="dashed")

        plt.title(f"Sales Forecast for {col}")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_filename = f"{col}_forecast_{uuid.uuid4().hex[:8]}.png"
        plot_path = os.path.join(PLOTS_FOLDER, plot_filename)  # Save to absolute static path
        plt.savefig(plot_path)
        plt.close()

        all_results.append({
            'drug': col,
            'accuracy': accuracy,
            'plot': plot_filename
        })

    return all_results
