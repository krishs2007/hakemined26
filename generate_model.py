#!/usr/bin/env python3
"""
SmartContainer Risk Engine - Model Pre-trainer
Convenience script to train and save the Random Forest model and Anomaly Detector
without running the full pipeline (no predictions, no dashboard).

Usage:
    python generate_model.py

Outputs:
    models/rf_model.pkl         - Trained Random Forest classifier
    models/anomaly_detector.pkl - Trained Isolation Forest anomaly detector
"""
import sys
import os
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.preprocessor import load_and_prepare
from src.feature_engineer import build_direct_features, build_historical_features, FEATURE_COLS
from src.anomaly_detector import train_anomaly_detector
from src.model_trainer import train_model

MODEL_DIR = 'models'
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_detector.pkl')


def generate_models(hist_path=None, rt_path=None):
    """
    Load data, engineer features, train and save both models.

    Args:
        hist_path: Optional path to Historical Data CSV
        rt_path:   Optional path to Real-Time Data CSV
    """
    print("=" * 60)
    print("  SmartContainer Risk Engine - Model Generator")
    print("=" * 60)

    print("\n[Step 1/3] Loading and preparing data...")
    train_df, test_df = load_and_prepare(hist_path, rt_path)

    print("\n[Step 2/3] Engineering features...")
    train_feat = build_direct_features(train_df)
    train_feat = build_historical_features(train_feat, train_df)

    print("\n[Step 3/3] Training and saving models...")

    # Train Random Forest (model_trainer already saves it to models/rf_model.pkl)
    rf_model = train_model(train_feat, FEATURE_COLS)

    # Train and save Anomaly Detector
    detector = train_anomaly_detector(train_feat)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ANOMALY_MODEL_PATH, 'wb') as f:
        pickle.dump(detector, f)
    print(f"  Anomaly detector saved to: {ANOMALY_MODEL_PATH}")

    print("\n" + "=" * 60)
    print("  Models saved successfully:")
    print(f"    {RF_MODEL_PATH}")
    print(f"    {ANOMALY_MODEL_PATH}")
    print("=" * 60)

    return rf_model, detector


if __name__ == '__main__':
    generate_models()
