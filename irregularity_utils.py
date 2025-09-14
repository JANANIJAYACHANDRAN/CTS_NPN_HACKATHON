import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prophet import Prophet
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import jarque_bera
import warnings
warnings.filterwarnings('ignore')

# --- Fixed static path configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of this script directory
PLOTS_FOLDER = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_FOLDER, exist_ok=True)

top_n = 3
forecast_steps = 60

def analyze_irregularity_drug(series):
    """
    This is the helper function with your original, unchanged model logic.
    It was named detect_irregularity in your script.
    """
    if len(series) < 12:
        return 0, {}, None, None

    temp_df = pd.DataFrame({'ds': series.index, 'y': series.values})
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode='additive', changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0, interval_width=0.8
    )
    if len(series) >= 12:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=6)

    try:
        model.fit(temp_df)
        future = model.make_future_dataframe(periods=forecast_steps, freq='M')
        forecast = model.predict(future)
        fitted_values = forecast['yhat'][:len(series)]
        residuals = series.values - fitted_values.values
        metrics = {}

        # 1. Basic volatility measures
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_cv'] = metrics['residual_std'] / np.mean(series.values) if np.mean(series.values) != 0 else 0
        
        # 2. Outlier detection
        lower_bound = forecast['yhat_lower'][:len(series)]
        upper_bound = forecast['yhat_upper'][:len(series)]
        total_outliers = np.sum(series.values < lower_bound.values) + np.sum(series.values > upper_bound.values)
        metrics['outlier_percentage'] = total_outliers / len(series) * 100
        metrics['num_outliers'] = total_outliers
        
        # 3. Statistical tests for normality of residuals
        if len(residuals) >= 8:
            try:
                jb_stat, jb_pvalue = jarque_bera(residuals)
                metrics['jarque_bera_pvalue'] = jb_pvalue
            except:
                metrics['jarque_bera_pvalue'] = 1.0
        else:
            metrics['jarque_bera_pvalue'] = 1.0

        # 4. Changepoint irregularity
        changepoints = model.changepoints
        metrics['num_changepoints'] = len(changepoints)
        
        # 7. Autocorrelation breakdown
        max_lag = min(len(series) // 3, 12)
        autocorr_values = [abs(series.autocorr(lag=lag)) if not np.isnan(series.autocorr(lag=lag)) else 0 for lag in range(1, max_lag + 1)]
        metrics['mean_autocorr'] = np.mean(autocorr_values) if autocorr_values else 0
        metrics['autocorr_breakdown'] = 1 - metrics['mean_autocorr']

        # 8. Model prediction error
        mape = np.mean(np.abs((series.values - fitted_values) / series.values)) * 100 if np.all(series.values != 0) else 0
        metrics['mape'] = mape

        # 9. Combined irregularity score (your original logic)
        outlier_score = min(metrics['outlier_percentage'] / 20, 1)
        volatility_score = min(metrics['residual_cv'], 1)
        changepoint_score = min(metrics['num_changepoints'] / 10, 1)
        normality_score = 1 - metrics['jarque_bera_pvalue']
        autocorr_score = metrics['autocorr_breakdown']
        prediction_error_score = min(metrics['mape'] / 50, 1)
        
        irregularity_score = (
            outlier_score * 0.25 + volatility_score * 0.20 + changepoint_score * 0.15 +
            normality_score * 0.15 + autocorr_score * 0.15 + prediction_error_score * 0.10
        )
        return irregularity_score, metrics, forecast, residuals

    except Exception as e:
        print(f"Error in irregularity calculation: {e}")
        return 0, {}, None, None

def create_irregularity_plot(drug_name, series, forecast, residuals, metrics):
    """ Creates and saves the irregularity plot, matching your original script. """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Irregularity Analysis: {drug_name}', fontsize=16, fontweight='bold')

    # Plot 1: Original data with model fit and outliers
    ax1 = axes[0, 0]
    ax1.plot(series.index, series.values, 'o-', color='blue', label='Original Data', markersize=4)
    ax1.plot(series.index, forecast['yhat'][:len(series)], color='red', linewidth=2, label='Prophet Fit')
    ax1.fill_between(series.index, forecast['yhat_lower'][:len(series)], forecast['yhat_upper'][:len(series)], alpha=0.3, color='red', label='Prediction Interval')
    lower_bound = forecast['yhat_lower'][:len(series)]
    upper_bound = forecast['yhat_upper'][:len(series)]
    outliers = (series.values < lower_bound.values) | (series.values > upper_bound.values)
    if np.any(outliers):
        ax1.scatter(series.index[outliers], series.values[outliers], color='orange', s=50, label='Outliers', zorder=5)
    ax1.set_title('Data with Model Fit and Outliers')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals (irregularity component)
    ax2 = axes[0, 1]
    ax2.plot(series.index, residuals, color='green', linewidth=2, label='Residuals')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(y=np.std(residuals), color='red', linestyle='--', alpha=0.7, label='+1 Std')
    ax2.axhline(y=-np.std(residuals), color='red', linestyle='--', alpha=0.7, label='-1 Std')
    ax2.set_title('Residuals (Irregular Component)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Residual Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residual distribution
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=min(15, len(residuals)//3), density=True, alpha=0.7, color='purple', label='Residuals')
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(np.min(residuals), np.max(residuals), 100)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    ax3.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Distribution')
    ax3.set_title(f'Residual Distribution (JB p-value: {metrics["jarque_bera_pvalue"]:.4f})')
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rolling volatility
    ax4 = axes[1, 1]
    window_size = min(6, len(series) // 4)
    if window_size >= 3:
        rolling_std = series.rolling(window=window_size).std()
        ax4.plot(series.index, rolling_std, color='orange', linewidth=2, label=f'Rolling Std ({window_size} periods)')
        ax4.set_title('Rolling Volatility')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Rolling Standard Deviation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor rolling volatility', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Rolling Volatility (Insufficient Data)')
    
    plt.tight_layout()
    plot_filename = f"{drug_name}_irregularity_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(PLOTS_FOLDER, plot_filename)  # Use corrected absolute path
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_filename

def get_irregularity_analysis(file_path):
    """ Main function called by app.py, structured like get_trend_strength. """
    df = pd.read_csv(file_path)
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum']).sort_values('datum').set_index('datum')
    drug_columns = [c for c in df.columns if c.lower() != 'datum']
    
    results = []
    for col in drug_columns:
        try:
            series = df[col].dropna()
            if len(series) < 8:
                continue
            
            irregularity_score, metrics, forecast, residuals = analyze_irregularity_drug(series)
            
            if forecast is not None and residuals is not None:
                plot_filename = create_irregularity_plot(col, series, forecast, residuals, metrics)
                results.append({
                    'drug': col,
                    'score': irregularity_score,
                    'plot': plot_filename
                })
        except Exception as e:
            print(f"Skipping {col} due to error: {e}")
            
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    return results_sorted
