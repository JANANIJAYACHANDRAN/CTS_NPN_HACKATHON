# app.py

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os

from trend_utils import get_trend_strength
from seasonality_utils import get_seasonality_analysis
from cyclicality_utils import get_cyclicality_analysis
from irregularity_utils import get_irregularity_analysis # 1. ADDED IMPORT
from forecast_utils import get_forecast_results
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Main and Component Routes ---
@app.route('/')
def index():
    """Renders the main login/home page."""
    return render_template('index.html')

@app.route('/trend')
def trend_page():
    """Renders the Trend analysis upload page."""
    return render_template('trend.html')

@app.route('/seasonality')
def seasonality_page():
    """Renders the Seasonality analysis upload page."""
    return render_template('seasonality.html')

@app.route('/cyclicality')
def cyclicality_page():
    return render_template('cyclicality.html')

@app.route('/irregularity')
def irregularity_page():
    return render_template('irregularity.html')

@app.route('/forecast')
def forecast_page():
    return render_template('forecast.html')

# --- Analysis Endpoint for Trend ---
@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles file upload and analysis for Trend."""
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if not file or file.filename == '':
        return "No file selected", 400
    
    freq = request.form.get('trend_type', 'W') 
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    top_drugs = get_trend_strength(filepath, freq=freq)
    return render_template('result.html', top_drugs=top_drugs)

# --- Analysis Endpoint for Seasonality ---
@app.route('/analyze_seasonality', methods=['POST'])
def analyze_seasonality():
    """Handles file upload and analysis for Seasonality."""
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if not file or file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    top_seasonal_drugs = get_seasonality_analysis(filepath)
    return render_template('seasonality_result.html', top_seasonal_drugs=top_seasonal_drugs)

# --- Analysis Endpoint for Cyclicality ---
@app.route('/analyze_cyclicality', methods=['POST'])
def analyze_cyclicality():
    """Handles file upload and analysis for Cyclicality."""
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if not file or file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    top_cyclical_drugs = get_cyclicality_analysis(filepath)
    return render_template('cyclicality_result.html', top_cyclical_drugs=top_cyclical_drugs)

# 3. ADDED NEW ENDPOINT FOR IRREGULARITY
# --- Analysis Endpoint for Irregularity ---
@app.route('/analyze_irregularity', methods=['POST'])
def analyze_irregularity():
    """Handles file upload and analysis for Irregularity."""
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if not file or file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    top_irregular_drugs = get_irregularity_analysis(filepath)
    return render_template('irregularity_result.html', top_irregular_drugs=top_irregular_drugs)

@app.route('/forecast_sales', methods=['POST'])
def forecast():
    """Handles file upload and analysis for Forecasting."""
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    if not file or file.filename == '': return "No file selected", 400
    
    # Get frequency from the dropdown on the forecast.html page
    freq = request.form.get('forecast_freq', 'D')
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Note: The future_periods is hardcoded to 365 in your script's example
    forecast_results = get_forecast_results(filepath, freq=freq, future_periods=365)
    return render_template('forecast_result.html', forecast_results=forecast_results)



if __name__ == '__main__':
    app.run(debug=True)