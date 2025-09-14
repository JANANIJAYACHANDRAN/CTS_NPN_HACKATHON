import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.linear_model import LinearRegression

forecast_steps = 30
top_n = 3
PLOTS_FOLDER = os.path.join("static", "plots")
os.makedirs(PLOTS_FOLDER, exist_ok=True)

def analyze_drug(series, freq="D"):
    temp = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(temp)
    future = model.make_future_dataframe(periods=forecast_steps, freq=freq)
    forecast = model.predict(future)
    x = np.arange(len(forecast))
    y = forecast['yhat'].values
    reg = LinearRegression().fit(x.reshape(-1,1), y)
    slope = reg.coef_[0]
    return slope, forecast

def get_trend_strength(file_path, freq="D"):
    df = pd.read_csv(file_path)
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum']).sort_values('datum')
    df = df.set_index('datum')
    drug_columns = [c for c in df.columns if c.lower() != "datum"]
    
    results = []
    for col in drug_columns:
        try:
            slope, forecast = analyze_drug(df[col].dropna(), freq=freq)
            
            # Save plot
            plt.figure(figsize=(10,5))
            plt.plot(forecast['ds'], forecast['yhat'], label="Forecast", color="blue")
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
            plt.scatter(df.index, df[col], color="black", s=10, label="Historical")
            plt.title(f"Trend for {col} (Slope={slope:.4f})")
            plt.legend()
            
            plot_filename = f"{col}_forecast.png"
            plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            
            results.append({
                "drug": col,
                "slope": slope,
                "plot": plot_filename
            })
        except Exception as e:
            print(f"Skipping {col}: {e}")
    
    # Sort by slope and take top_n
    results_sorted = sorted(results, key=lambda x: x['slope'], reverse=True)[:top_n]
    return results_sorted
