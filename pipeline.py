"""
SmartContainer Risk Engine — Pipeline Orchestrator
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessor import load_and_prepare
from src.feature_engineer import build_direct_features, build_historical_features, FEATURE_COLS
from src.anomaly_detector import train_anomaly_detector, get_anomaly_scores
from src.model_trainer import train_model, get_probabilities, evaluate_model
from src.risk_scorer import compute_risk_score, classify_risk_level
from src.risk_scorer import THRESHOLD_LOW_MEDIUM, THRESHOLD_MEDIUM_CRITICAL
from src.explainer import generate_explanations
from src.dashboard import generate_dashboard

# Version tracking
PIPELINE_VERSION = "1.0.0"
FEATURE_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"

def run_pipeline(hist_path=None, rt_path=None):
    """
    Execute the full SmartContainer Risk Engine pipeline.

    Steps:
        1. Load & prepare data (leakage-safe split)
        2. Engineer features (direct + historical)
        3. Train anomaly detector (Isolation Forest)
        4. Train classifier (Random Forest)
        5. Compute risk scores (model + anomaly + rules)
        6. Generate explanations
        7. Save predictions CSV
        8. Generate dashboard
        9. Validate against ground truth

    Args:
        hist_path: path to Historical Data CSV (auto-detected if None)
        rt_path: path to Real-Time Data CSV (auto-detected if None)

    Returns:
        DataFrame with predictions
    """
    start_time = time.time()

    print("=" * 60)
    print("  SmartContainer Risk Engine")
    print(f"  Pipeline v{PIPELINE_VERSION} | Features v{FEATURE_VERSION} | Model v{MODEL_VERSION}")
    print(f"  Random Seed: {RANDOM_SEED}")
    print(f"  Risk Thresholds: Low<{{THRESHOLD_LOW_MEDIUM}}, "
          f"Medium<{{THRESHOLD_MEDIUM_CRITICAL}}, Critical>={{THRESHOLD_MEDIUM_CRITICAL}}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[Step 1/8] Loading data...")
    train_df, test_df = load_and_prepare(hist_path, rt_path)

    # ------------------------------------------------------------------
    # Step 2: Engineer features
    # ------------------------------------------------------------------
    print("\n[Step 2/8] Engineering features...")
    print(f"  Building {{len(FEATURE_COLS)}} features...")

    train_feat = build_direct_features(train_df)
    train_feat = build_historical_features(train_feat, train_df)

    # IMPORTANT: test features use train_df as reference (no leakage)
    test_feat = build_direct_features(test_df)
    test_feat = build_historical_features(test_feat, train_df)

    # ------------------------------------------------------------------
    # Step 3: Train anomaly detector
    # ------------------------------------------------------------------
    print("\n[Step 3/8] Training anomaly detector...")
    detector = train_anomaly_detector(train_feat)
    anomaly_scores = get_anomaly_scores(detector, test_feat)

    # ------------------------------------------------------------------
    # Step 4: Train classifier
    # ------------------------------------------------------------------
    print("\n[Step 4/8] Training classifier...")
    rf_model = train_model(train_feat, FEATURE_COLS)
    probs = get_probabilities(rf_model, test_feat[FEATURE_COLS])

    # ------------------------------------------------------------------
    # Step 5: Compute risk scores
    # ------------------------------------------------------------------
    print("\n[Step 5/8] Computing risk scores...")
    test_feat['Risk_Score'] = compute_risk_score(probs, anomaly_scores, test_feat)
    test_feat['Risk_Level'] = classify_risk_level(test_feat['Risk_Score'].values)

    # ------------------------------------------------------------------
    # Step 6: Generate explanations
    # ------------------------------------------------------------------
    print("\n[Step 6/8] Generating explanations...")
    test_feat['Explanation_Summary'] = generate_explanations(test_feat)

    # ------------------------------------------------------------------
    # Step 7: Save predictions
    # ------------------------------------------------------------------
    print("\n[Step 7/8] Saving predictions...")
    os.makedirs('outputs', exist_ok=True)

    output = test_df[['Container_ID']].copy()
    output['Risk_Score'] = test_feat['Risk_Score'].values
    output['Risk_Level'] = test_feat['Risk_Level'].values
    output['Explanation_Summary'] = test_feat['Explanation_Summary'].values

    output_path = 'outputs/predictions.csv'
    output.to_csv(output_path, index=False)
    print(f"  Saved: {{output_path}} ({{len(output):,}} rows)")

    # ------------------------------------------------------------------
    # Step 8: Generate dashboard
    # ------------------------------------------------------------------
    print("\n[Step 8/8] Generating dashboard...")
    # Merge original columns needed for dashboard
    dashboard_df = test_feat.copy()
    dashboard_df['Container_ID'] = test_df['Container_ID'].values
    generate_dashboard(dashboard_df)

    # ------------------------------------------------------------------
    # Validation (since we have ground truth labels)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VALIDATION vs Ground Truth")
    print("=" * 60)

    # Map our 3-level risk to the original 3-level labels for comparison
    # Note: This is a derived mapping for evaluation purposes only
    risk_to_original = {
        'Critical': 'Critical',
        'Medium Risk': 'Low Risk',
        'Low Risk': 'Clear',
    }
    mapped_predictions = test_feat['Risk_Level'].map(risk_to_original)

    from sklearn.metrics import classification_report, accuracy_score
    ground_truth = test_df['Clearance_Status'].values
    print("\n  NOTE: Risk levels mapped to original labels for comparison:")
    print("    Critical    -> Critical")
    print("    Medium Risk -> Low Risk")
    print("    Low Risk    -> Clear")
    print()
    print(classification_report(ground_truth, mapped_predictions))
    print(f"  Overall accuracy: {{accuracy_score(ground_truth, mapped_predictions):.4f}}")

    # Risk score stats by ground truth label
    print("\n  Risk Score by Ground Truth Label:")
    for label in ['Clear', 'Low Risk', 'Critical']:
        mask = test_df['Clearance_Status'] == label
        scores = test_feat.loc[mask, 'Risk_Score']
        if len(scores) > 0:
            print(f"    {{label:12s}}: mean={{scores.mean():.1f}}, "
                  f"median={{scores.median():.1f}}, "
                  f"min={{scores.min():.1f}}, max={{scores.max():.1f}}")

    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {{elapsed:.1f}} seconds.")
    print("=" * 60)

    return output


if __name__ == '__main__':
    run_pipeline()