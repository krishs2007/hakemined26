#!/usr/bin/env python3
"""
SmartContainer Risk Engine — Flask Web Application
Serves the main website and connects to all ML pipeline modules.

Usage:
    python webapp.py
    Open http://localhost:5000 in your browser.
"""
import sys
import os
import json
import pickle
import threading
import time
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.preprocessor import load_and_prepare
from src.feature_engineer import build_direct_features, build_historical_features, FEATURE_COLS
from src.anomaly_detector import train_anomaly_detector, get_anomaly_scores
from src.model_trainer import train_model, load_model, get_probabilities
from src.risk_scorer import compute_risk_score, classify_risk_level
from src.explainer import generate_explanations
from src.dashboard import generate_dashboard

app = Flask(__name__)
CORS(app)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join('models', 'rf_model.pkl')
ANOMALY_DETECTOR_PATH = os.path.join('models', 'anomaly_detector.pkl')
PREDICTIONS_PATH = os.path.join('outputs', 'predictions.csv')
DASHBOARD_PATH = os.path.join('outputs', 'summary_report.png')
REPORT_PATH = os.path.join('outputs', 'report.html')
UPLOAD_DIR = 'data'

# ── Global pipeline state ─────────────────────────────────────────────────────
pipeline_state = {
    'status': 'idle',       # idle | running | completed | error
    'message': '',
    'progress': 0,
    'started_at': None,
    'completed_at': None,
    'error': None,
}
pipeline_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _set_state(status, message='', progress=0, error=None):
    with pipeline_lock:
        pipeline_state['status'] = status
        pipeline_state['message'] = message
        pipeline_state['progress'] = progress
        pipeline_state['error'] = error
        if status == 'running' and pipeline_state['started_at'] is None:
            pipeline_state['started_at'] = time.time()
        if status in ('completed', 'error'):
            pipeline_state['completed_at'] = time.time()


def _load_predictions():
    """Load predictions CSV and return DataFrame, or None if not found."""
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    return pd.read_csv(PREDICTIONS_PATH)


def _generate_html_report(results_df):
    """Generate a simple HTML summary report."""
    os.makedirs('outputs', exist_ok=True)
    critical = int((results_df['Risk_Level'] == 'Critical').sum())
    medium = int((results_df['Risk_Level'] == 'Medium Risk').sum())
    low = int((results_df['Risk_Level'] == 'Low Risk').sum())
    total = len(results_df)

    rows = []
    for _, row in results_df.iterrows():
        level = row['Risk_Level']
        color = '#e74c3c' if level == 'Critical' else ('#f39c12' if level == 'Medium Risk' else '#2ecc71')
        explanation = row.get('Explanation_Summary', '')
        rows.append(
            f"<tr>"
            f"<td>{row['Container_ID']}</td>"
            f"<td>{row['Risk_Score']:.1f}</td>"
            f"<td style='color:{color};font-weight:bold'>{level}</td>"
            f"<td>{explanation}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SmartContainer Risk Engine — Report</title>
  <style>
    body {{font-family: Arial, sans-serif; margin: 20px; background: #f5f6fa;}}
    h1 {{color: #2c3e50;}}
    .summary {{display:flex; gap:20px; margin-bottom:20px;}}
    .card {{background:#fff; border-radius:8px; padding:16px 24px; box-shadow:0 2px 8px rgba(0,0,0,.1);}}
    table {{border-collapse:collapse; width:100%; background:#fff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.1);}}
    th {{background:#2c3e50; color:#fff; padding:10px 14px; text-align:left;}}
    td {{padding:8px 14px; border-bottom:1px solid #eee;}}
    tr:hover td {{background:#f0f4f8;}}
  </style>
</head>
<body>
  <h1>🚢 SmartContainer Risk Engine — Summary Report</h1>
  <div class="summary">
    <div class="card"><h3>Total</h3><p style="font-size:2em;margin:0">{total:,}</p></div>
    <div class="card" style="color:#e74c3c"><h3>Critical</h3><p style="font-size:2em;margin:0">{critical:,}</p></div>
    <div class="card" style="color:#f39c12"><h3>Medium Risk</h3><p style="font-size:2em;margin:0">{medium:,}</p></div>
    <div class="card" style="color:#2ecc71"><h3>Low Risk</h3><p style="font-size:2em;margin:0">{low:,}</p></div>
  </div>
  <table>
    <thead><tr><th>Container ID</th><th>Risk Score</th><th>Risk Level</th><th>Explanation</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>"""
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    return REPORT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (executes in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_thread():
    try:
        _set_state('running', 'Loading data…', progress=5)

        # Change working directory so relative paths resolve correctly
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Step 1: load data
        train_df, test_df = load_and_prepare()
        _set_state('running', 'Engineering features…', progress=20)

        # Step 2: feature engineering
        train_feat = build_direct_features(train_df)
        train_feat = build_historical_features(train_feat, train_df)
        test_feat = build_direct_features(test_df)
        test_feat = build_historical_features(test_feat, train_df)
        _set_state('running', 'Training anomaly detector…', progress=35)

        # Step 3: anomaly detection
        detector = train_anomaly_detector(train_feat)
        anomaly_scores = get_anomaly_scores(detector, test_feat)
        # Save anomaly detector
        os.makedirs('models', exist_ok=True)
        with open(ANOMALY_DETECTOR_PATH, 'wb') as f:
            pickle.dump(detector, f)
        _set_state('running', 'Training Random Forest classifier…', progress=50)

        # Step 4: train model
        rf_model = train_model(train_feat, FEATURE_COLS)
        probs = get_probabilities(rf_model, test_feat[FEATURE_COLS])
        _set_state('running', 'Computing risk scores…', progress=65)

        # Step 5: risk scoring
        test_feat['Risk_Score'] = compute_risk_score(probs, anomaly_scores, test_feat)
        test_feat['Risk_Level'] = classify_risk_level(test_feat['Risk_Score'].values)
        _set_state('running', 'Generating explanations…', progress=75)

        # Step 6: explanations
        test_feat['Explanation_Summary'] = generate_explanations(test_feat)
        _set_state('running', 'Saving predictions…', progress=85)

        # Step 7: save predictions with enriched columns
        os.makedirs('outputs', exist_ok=True)
        output = test_df[['Container_ID', 'Origin_Country', 'Destination_Port']].copy()
        output['Risk_Score'] = test_feat['Risk_Score'].values
        output['Risk_Level'] = test_feat['Risk_Level'].values
        output['Explanation_Summary'] = test_feat['Explanation_Summary'].values
        output.to_csv(PREDICTIONS_PATH, index=False)
        _set_state('running', 'Generating dashboard…', progress=92)

        # Step 8: dashboard
        dashboard_df = test_feat.copy()
        dashboard_df['Container_ID'] = test_df['Container_ID'].values
        dashboard_df['Origin_Country'] = test_df['Origin_Country'].values
        dashboard_df['Destination_Port'] = test_df['Destination_Port'].values
        generate_dashboard(dashboard_df)

        # Step 9: HTML report
        _generate_html_report(output)

        _set_state('completed', 'Pipeline finished successfully.', progress=100)

    except Exception as exc:
        _set_state('error', str(exc), error=traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    with pipeline_lock:
        state = dict(pipeline_state)
    state['model_trained'] = os.path.exists(MODEL_PATH)
    state['predictions_exist'] = os.path.exists(PREDICTIONS_PATH)
    state['dashboard_exists'] = os.path.exists(DASHBOARD_PATH)
    return jsonify(state)


@app.route('/api/run-pipeline', methods=['POST'])
def api_run_pipeline():
    with pipeline_lock:
        if pipeline_state['status'] == 'running':
            return jsonify({'error': 'Pipeline is already running.'}), 409

    # Reset state and launch background thread
    with pipeline_lock:
        pipeline_state['status'] = 'running'
        pipeline_state['message'] = 'Initializing…'
        pipeline_state['progress'] = 0
        pipeline_state['started_at'] = time.time()
        pipeline_state['completed_at'] = None
        pipeline_state['error'] = None

    thread = threading.Thread(target=_run_pipeline_thread, daemon=True)
    thread.start()

    return jsonify({'status': 'started', 'message': 'Pipeline started in background.'})


@app.route('/api/predictions')
def api_predictions():
    df = _load_predictions()
    if df is None:
        return jsonify([])

    # Optional filters
    risk_level = request.args.get('risk_level')
    search = request.args.get('search', '').strip()
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    if risk_level:
        df = df[df['Risk_Level'] == risk_level]
    if search:
        mask = df.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        df = df[mask]

    total = len(df)
    page_df = df.iloc[offset: offset + limit]

    records = []
    for _, row in page_df.iterrows():
        records.append({
            'Container_ID': str(row['Container_ID']),
            'Risk_Score': float(row['Risk_Score']),
            'Risk_Level': str(row['Risk_Level']),
            'Origin_Country': str(row.get('Origin_Country', '')),
            'Destination_Port': str(row.get('Destination_Port', '')),
            'Explanation_Summary': str(row.get('Explanation_Summary', '')),
        })

    return jsonify({'total': total, 'offset': offset, 'limit': limit, 'data': records})


@app.route('/api/predictions/<container_id>')
def api_prediction_detail(container_id):
    df = _load_predictions()
    if df is None:
        abort(404)
    row = df[df['Container_ID'].astype(str) == str(container_id)]
    if row.empty:
        abort(404)
    return jsonify(row.iloc[0].to_dict())


@app.route('/api/stats')
def api_stats():
    df = _load_predictions()
    if df is None:
        return jsonify({})

    total = len(df)
    level_counts = df['Risk_Level'].value_counts().to_dict()

    top_countries = (
        df.groupby('Origin_Country')['Risk_Score'].mean()
        .nlargest(10)
        .round(1)
        .to_dict()
    ) if 'Origin_Country' in df.columns else {}

    top_ports = (
        df.groupby('Destination_Port')['Risk_Score'].mean()
        .nlargest(10)
        .round(1)
        .to_dict()
    ) if 'Destination_Port' in df.columns else {}

    return jsonify({
        'total': total,
        'critical': int(level_counts.get('Critical', 0)),
        'medium_risk': int(level_counts.get('Medium Risk', 0)),
        'low_risk': int(level_counts.get('Low Risk', 0)),
        'avg_score': round(float(df['Risk_Score'].mean()), 1),
        'median_score': round(float(df['Risk_Score'].median()), 1),
        'max_score': round(float(df['Risk_Score'].max()), 1),
        'min_score': round(float(df['Risk_Score'].min()), 1),
        'top_countries': top_countries,
        'top_ports': top_ports,
    })


@app.route('/api/dashboard-image')
def api_dashboard_image():
    if not os.path.exists(DASHBOARD_PATH):
        abort(404)
    return send_file(DASHBOARD_PATH, mimetype='image/png')


@app.route('/api/report')
def api_report():
    if not os.path.exists(REPORT_PATH):
        abort(404)
    return send_file(REPORT_PATH, mimetype='text/html')


@app.route('/api/download/predictions')
def api_download_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        abort(404)
    return send_file(PREDICTIONS_PATH, as_attachment=True, download_name='predictions.csv')


@app.route('/api/download/dashboard')
def api_download_dashboard():
    if not os.path.exists(DASHBOARD_PATH):
        abort(404)
    return send_file(DASHBOARD_PATH, as_attachment=True, download_name='summary_report.png')


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are accepted.'}), 400
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(save_path)
    return jsonify({'message': f'File saved to {save_path}.'})


@app.route('/api/feature-importance')
def api_feature_importance():
    if not os.path.exists(MODEL_PATH):
        return jsonify([])
    with open(MODEL_PATH, 'rb') as fh:
        model = pickle.load(fh)
    importances = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLS, importances.tolist()),
                   key=lambda x: x[1], reverse=True)
    return jsonify([{'feature': f, 'importance': round(i, 4)} for f, i in pairs])


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000, use_reloader=False)
