import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# --- Fixed static path configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_FOLDER = os.path.join(BASE_DIR, "static", "plots")  # Always correct, regardless of CWD
os.makedirs(PLOTS_FOLDER, exist_ok=True)
top_n = 3
forecast_steps = 90

def analyze_seasonality_drug(series):
    if len(series) < 24:
        return 0, {}, None
    temp_df = pd.DataFrame({'ds': series.index, 'y': series.values})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        seasonality_prior_scale=50,
        changepoint_prior_scale=0.01
    )
    if len(series) >= 12:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=8)
    if len(series) >= 4:
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=4)
    try:
        model.fit(temp_df)
        future = model.make_future_dataframe(periods=forecast_steps, freq='M')
        forecast = model.predict(future)
        metrics = {}
        seasonal_component = forecast['yearly'][:len(series)]
        trend_component = forecast['trend'][:len(series)]
        seasonal_var = np.var(seasonal_component)
        total_var = np.var(trend_component + seasonal_component)
        seasonal_strength = seasonal_var / total_var if total_var > 0 else 0
        metrics['prophet_seasonal_strength'] = seasonal_strength
        fft_values = np.abs(fft(series.values - np.mean(series.values)))
        freqs = fftfreq(len(series.values))
        dominant_freq_idx = np.argsort(fft_values[1:len(fft_values)//2])[-3:]
        dominant_freqs = freqs[dominant_freq_idx + 1]
        metrics['dominant_frequencies'] = dominant_freqs.tolist()
        max_lag = min(len(series) // 2, 12)
        autocorr_values = [abs(series.autocorr(lag=lag)) if not np.isnan(series.autocorr(lag=lag)) else 0 for lag in range(1, max_lag + 1)]
        metrics['max_autocorr'] = max(autocorr_values) if autocorr_values else 0
        metrics['seasonal_lags'] = autocorr_values
        fitted_values = forecast['yhat'][:len(series)]
        residuals = series.values - fitted_values
        cv_residuals = np.std(residuals) / np.mean(series.values) if np.mean(series.values) != 0 else 0
        metrics['cv_residuals'] = cv_residuals
        combined_score = (
            seasonal_strength * 0.4 +
            metrics['max_autocorr'] * 0.3 +
            (1 - cv_residuals) * 0.3
        )
        return combined_score, metrics, forecast
    except Exception as e:
        print(f"Error in seasonality calculation: {e}")
        return 0, {}, None

def create_seasonality_plots(drug_name, series, forecast, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Seasonality Analysis: {drug_name}', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.plot(series.index, series.values, 'o-', color='black', label='Historical Data', markersize=4)
    ax1.plot(forecast['ds'][:len(series)], forecast['yhat'][:len(series)], color='blue', label='Prophet Fit', linewidth=2)
    ax1.fill_between(forecast['ds'][:len(series)], forecast['yhat_lower'][:len(series)], forecast['yhat_upper'][:len(series)], alpha=0.3, color='blue')
    ax1.set_title('Time Series with Seasonal Fit')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if 'yearly' in forecast.columns:
        ax2.plot(forecast['ds'][:len(series)], forecast['yearly'][:len(series)], color='red', linewidth=2, label='Yearly Seasonality')
    if 'weekly' in forecast.columns:
        ax2.plot(forecast['ds'][:len(series)], forecast['weekly'][:len(series)], color='green', linewidth=2, label='Weekly Seasonality', alpha=0.7)
    ax2.set_title('Seasonal Components')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Seasonal Effect')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if 'seasonal_lags' in metrics and metrics['seasonal_lags']:
        lags = range(1, len(metrics['seasonal_lags']) + 1)
        ax3.bar(lags, metrics['seasonal_lags'], color='purple', alpha=0.7)
        ax3.axhline(y=0.2, color='red', linestyle='--', label='Significance Threshold')
        ax3.set_title('Autocorrelation by Lag (Months)')
        ax3.set_xlabel('Lag (months)')
        ax3.set_ylabel('Autocorrelation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(forecast['ds'][:len(series)], forecast['trend'][:len(series)], color='orange', linewidth=2, label='Trend')
    ax4.plot(forecast['ds'][:len(series)], forecast['yearly'][:len(series)], color='red', linewidth=2, label='Seasonal')
    ax4.set_title('Trend vs Seasonal Components')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Component Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f"{drug_name}_seasonality_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_filename


def get_seasonality_analysis(file_path):
    df = pd.read_csv(file_path)
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum']).sort_values('datum').set_index('datum')
    drug_columns = [c for c in df.columns if c.lower() != 'datum']

    results = []
    for col in drug_columns:
        try:
            series = df[col].dropna()
            if len(series) < 12:
                continue
            seasonal_score, metrics, forecast = analyze_seasonality_drug(series)
            if forecast is not None:
                plot_filename = create_seasonality_plots(col, series, forecast, metrics)
                results.append({
                    'drug': col,
                    'score': seasonal_score,
                    'plot': plot_filename
                })
        except Exception as e:
            print(f"Skipping {col} due to error: {e}")
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    return results_sorted
