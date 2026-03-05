#!/usr/bin/env python3
"""
SmartContainer Risk Engine - Pipeline Orchestrator
Single entry point that runs everything end-to-end.

Usage:
    python pipeline.py
"""
import sys
import os
import time
import numpy as np
import pandas as pd

# Global random seed for full reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.preprocessor import load_and_prepare
from src.feature_engineer import build_direct_features, build_historical_features, FEATURE_COLS
from src.anomaly_detector import train_anomaly_detector, get_anomaly_scores
from src.model_trainer import train_model, get_probabilities
from src.risk_scorer import compute_risk_score, classify_risk_level
from src.risk_scorer import THRESHOLD_LOW_MEDIUM, THRESHOLD_MEDIUM_CRITICAL
from src.explainer import generate_explanations
from src.dashboard import generate_dashboard

# Version tracking
PIPELINE_VERSION = "1.0.0"
FEATURE_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"

def run_pipeline(hist_path=None, rt_path=None):
    start_time = time.time()
    num_features = len(FEATURE_COLS)

    print("=" * 60)
    print("  SmartContainer Risk Engine")
    print("  Pipeline v" + PIPELINE_VERSION + " | Features v" + FEATURE_VERSION + " | Model v" + MODEL_VERSION)
    print("  Random Seed: " + str(RANDOM_SEED))
    print("  Risk Thresholds: Low<" + str(THRESHOLD_LOW_MEDIUM) + ", Medium<" + str(THRESHOLD_MEDIUM_CRITICAL) + ", Critical>=" + str(THRESHOLD_MEDIUM_CRITICAL))
    print("=" * 60)

    print("")
    print("[Step 1/8] Loading data...")
    train_df, test_df = load_and_prepare(hist_path, rt_path)

    print("")
    print("[Step 2/8] Engineering features...")
    print("  Building " + str(num_features) + " features...")
    train_feat = build_direct_features(train_df)
    train_feat = build_historical_features(train_feat, train_df)
    test_feat = build_direct_features(test_df)
    test_feat = build_historical_features(test_feat, train_df)

    print("")
    print("[Step 3/8] Training anomaly detector...")
    detector = train_anomaly_detector(train_feat)
    anomaly_scores = get_anomaly_scores(detector, test_feat)

    print("")
    print("[Step 4/8] Training classifier...")
    rf_model = train_model(train_feat, FEATURE_COLS)
    probs = get_probabilities(rf_model, test_feat[FEATURE_COLS])

    print("")
    print("[Step 5/8] Computing risk scores...")
    test_feat['Risk_Score'] = compute_risk_score(probs, anomaly_scores, test_feat)
    test_feat['Risk_Level'] = classify_risk_level(test_feat['Risk_Score'].values)

    print("")
    print("[Step 6/8] Generating explanations...")
    test_feat['Explanation_Summary'] = generate_explanations(test_feat)

    print("")
    print("[Step 7/8] Saving predictions...")
    os.makedirs('outputs', exist_ok=True)
    output = test_df[['Container_ID']].copy()
    output['Risk_Score'] = test_feat['Risk_Score'].values
    output['Risk_Level'] = test_feat['Risk_Level'].values
    output['Explanation_Summary'] = test_feat['Explanation_Summary'].values
    output_path = 'outputs/predictions.csv'
    output.to_csv(output_path, index=False)
    print("  Saved: " + output_path + " (" + str(len(output)) + " rows)")

    print("")
    print("[Step 8/8] Generating dashboard...")
    dashboard_df = test_feat.copy()
    dashboard_df['Container_ID'] = test_df['Container_ID'].values
    png_path, html_path = generate_dashboard(dashboard_df)
    print(f"  PNG:  {png_path}")
    print(f"  HTML: {html_path}")

    print("")
    print("=" * 60)
    print("  VALIDATION vs Ground Truth")
    print("=" * 60)
    risk_to_original = {
        'Critical': 'Critical',
        'Medium Risk': 'Low Risk',
        'Low Risk': 'Clear',
    }
    mapped_predictions = test_feat['Risk_Level'].map(risk_to_original)

    from sklearn.metrics import classification_report, accuracy_score
    ground_truth = test_df['Clearance_Status'].values
    print("")
    print("  NOTE: Risk levels mapped to original labels for comparison:")
    print("    Critical    -> Critical")
    print("    Medium Risk -> Low Risk")
    print("    Low Risk    -> Clear")
    print("")
    print(classification_report(ground_truth, mapped_predictions))
    acc = accuracy_score(ground_truth, mapped_predictions)
    print("  Overall accuracy: " + str(round(acc, 4)))

    print("")
    print("  Risk Score by Ground Truth Label:")
    for label in ['Clear', 'Low Risk', 'Critical']:
        mask = test_df['Clearance_Status'] == label
        scores = test_feat.loc[mask, 'Risk_Score']
        if len(scores) > 0:
            print("    " + label.ljust(12) + ": mean=" + str(round(scores.mean(), 1)) + ", median=" + str(round(scores.median(), 1)) + ", min=" + str(round(scores.min(), 1)) + ", max=" + str(round(scores.max(), 1)))

    elapsed = time.time() - start_time
    print("")
    print("  Pipeline completed in " + str(round(elapsed, 1)) + " seconds.")
    print("=" * 60)

    return output


if __name__ == '__main__':
    run_pipeline()