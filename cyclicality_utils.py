import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# --- Fixed static path configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory where this script is located
PLOTS_FOLDER = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_FOLDER, exist_ok=True)

top_n = 3
forecast_steps = 120

def analyze_cyclicality_drug(series, min_cycle_length=6, max_cycle_length=60):
    """
    This is the helper function with your original, unchanged model logic.
    It was named detect_cycles in your script.
    """
    if len(series) < min_cycle_length * 2:
        return 0, {}, None, None

    temp_df = pd.DataFrame({'ds': series.index, 'y': series.values})
    model = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        seasonality_mode='additive', changepoint_prior_scale=0.1,
        n_changepoints=min(25, len(series)//4)
    )

    try:
        model.fit(temp_df)
        future = model.make_future_dataframe(periods=forecast_steps, freq='M')
        forecast = model.predict(future)
        trend = forecast['trend'][:len(series)]
        detrended = series.values - trend.values
        metrics = {}

        # FFT-based cycle detection
        fft_values = np.abs(fft(detrended - np.mean(detrended)))
        freqs = fftfreq(len(detrended))
        positive_freqs = freqs[:len(freqs)//2][1:]
        positive_fft = fft_values[:len(fft_values)//2][1:]
        dominant_indices = np.argsort(positive_fft)[-5:]
        dominant_freqs = positive_freqs[dominant_indices]
        dominant_periods = 1 / dominant_freqs
        valid_periods = dominant_periods[(dominant_periods >= min_cycle_length) & (dominant_periods <= max_cycle_length)]
        metrics['dominant_cycles'] = valid_periods.tolist()
        metrics['cycle_strength'] = np.max(positive_fft) / np.mean(positive_fft) if len(positive_fft) > 0 else 0

        # Autocorrelation
        max_lag = min(len(detrended) // 2, max_cycle_length)
        autocorr_values = [pd.Series(detrended).autocorr(lag=lag) for lag in range(min_cycle_length, max_lag + 1)]
        autocorr_values = [x if not np.isnan(x) else 0 for x in autocorr_values]
        metrics['max_autocorr_cycle'] = max([abs(x) for x in autocorr_values]) if autocorr_values else 0
        metrics['autocorr_values'] = autocorr_values

        # Peak detection
        peaks, _ = signal.find_peaks(detrended, height=np.std(detrended)*0.5, distance=min_cycle_length//2)
        troughs, _ = signal.find_peaks(-detrended, height=np.std(detrended)*0.5, distance=min_cycle_length//2)
        metrics['num_peaks'] = len(peaks)
        metrics['num_troughs'] = len(troughs)
        
        # Cycle regularity
        all_extrema = sorted(np.concatenate([peaks, troughs]))
        if len(all_extrema) > 2:
            extrema_distances = np.diff(all_extrema)
            cycle_regularity = np.std(extrema_distances) / np.mean(extrema_distances) if np.mean(extrema_distances) > 0 else float('inf')
            metrics['cycle_regularity'] = 1 / (1 + cycle_regularity)
        else:
            metrics['cycle_regularity'] = 0

        # Amplitude consistency
        if len(peaks) > 0 and len(troughs) > 0:
            peak_values = detrended[peaks]
            trough_values = detrended[troughs]
            amplitude_consistency = 1 - (np.std(peak_values - np.mean(trough_values)) / (np.mean(peak_values) - np.mean(trough_values))) if (np.mean(peak_values) - np.mean(trough_values)) != 0 else 0
            metrics['amplitude_consistency'] = max(0, amplitude_consistency)
        else:
            metrics['amplitude_consistency'] = 0

        # Combined cyclicality score
        cycle_score = (
            min(metrics['cycle_strength'] / 10, 1) * 0.3 +
            metrics['max_autocorr_cycle'] * 0.3 +
            min(metrics['num_peaks'] / 5, 1) * 0.2 +
            metrics['cycle_regularity'] * 0.1 +
            metrics['amplitude_consistency'] * 0.1
        )
        return cycle_score, metrics, forecast, detrended

    except Exception as e:
        print(f"Error in cyclicality calculation: {e}")
        return 0, {}, None, None

def create_cyclicality_plot(drug_name, series, forecast, detrended, metrics):
    """ Creates and saves the cyclicality plot. """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Cyclicality Analysis: {drug_name}', fontsize=16, fontweight='bold')

    # Plot 1
    ax1 = axes[0, 0]
    ax1.plot(series.index, series.values, 'o-', color='blue', label='Original Data', markersize=4)
    ax1.plot(series.index, forecast['trend'][:len(series)], color='red', linewidth=2, label='Trend')
    ax1.set_title('Original Data with Trend')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2
    ax2 = axes[0, 1]
    ax2.plot(series.index, detrended, color='green', linewidth=2, label='Detrended (Cycles)')
    peaks, _ = signal.find_peaks(detrended, height=np.std(detrended)*0.5, distance=3)
    troughs, _ = signal.find_peaks(-detrended, height=np.std(detrended)*0.5, distance=3)
    if len(peaks) > 0:
        ax2.scatter(series.index[peaks], detrended[peaks], color='red', s=50, label='Peaks', zorder=5)
    if len(troughs) > 0:
        ax2.scatter(series.index[troughs], detrended[troughs], color='blue', s=50, label='Troughs', zorder=5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Detrended Data (Cyclical Component)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3
    ax3 = axes[1, 0]
    fft_values = np.abs(fft(detrended - np.mean(detrended)))
    freqs = fftfreq(len(detrended))
    positive_freqs = freqs[:len(freqs)//2][1:]
    positive_fft = fft_values[:len(fft_values)//2][1:]
    periods = 1 / positive_freqs
    mask = (periods >= 6) & (periods <= 60)
    ax3.plot(periods[mask], positive_fft[mask], color='purple', linewidth=2)
    ax3.set_title('Frequency Spectrum (Cycle Detection)')
    ax3.set_xlabel('Cycle Length (periods)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4
    ax4 = axes[1, 1]
    lags = range(6, min(len(detrended) // 2, 60) + 1)
    autocorr_subset = metrics['autocorr_values'][:len(lags)]
    ax4.plot(lags[:len(autocorr_subset)], autocorr_subset, color='orange', linewidth=2, marker='o')
    ax4.axhline(y=0.3, color='red', linestyle='--', label='Significance Threshold')
    ax4.axhline(y=-0.3, color='red', linestyle='--')
    ax4.set_title('Autocorrelation (Cycle Lags)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"{drug_name}_cyclicality_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_filename

def get_cyclicality_analysis(file_path):
    """ Main function called by app.py, structured like get_trend_strength. """
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
            
            cycle_score, metrics, forecast, detrended = analyze_cyclicality_drug(series)
            
            if forecast is not None and detrended is not None:
                plot_filename = create_cyclicality_plot(col, series, forecast, detrended, metrics)
                results.append({
                    'drug': col,
                    'score': cycle_score,
                    'plot': plot_filename
                })
        except Exception as e:
            print(f"Skipping {col} due to error: {e}")
            
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    return results_sorted
