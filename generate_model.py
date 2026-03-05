#!/usr/bin/env python3
"""
SmartContainer Risk Engine — Standalone Model Training Script
Trains and saves the Random Forest classifier and Anomaly Detector.

Usage:
    python generate_model.py
"""
import sys
import os
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.preprocessor import load_and_prepare
from src.feature_engineer import build_direct_features, build_historical_features, FEATURE_COLS
from src.model_trainer import train_model
from src.anomaly_detector import train_anomaly_detector

ANOMALY_DETECTOR_PATH = os.path.join('models', 'anomaly_detector.pkl')


def main():
    print("=" * 60)
    print("  SmartContainer Risk Engine — Model Generation")
    print("=" * 60)

    print("\n[Step 1/4] Loading data...")
    train_df, test_df = load_and_prepare()

    print("\n[Step 2/4] Engineering features...")
    train_feat = build_direct_features(train_df)
    train_feat = build_historical_features(train_feat, train_df)

    print("\n[Step 3/4] Training Random Forest classifier...")
    rf_model = train_model(train_feat, FEATURE_COLS)
    print("  ✓ RF model saved to models/rf_model.pkl")

    print("\n[Step 4/4] Training Anomaly Detector (Isolation Forest)...")
    detector = train_anomaly_detector(train_feat)
    os.makedirs('models', exist_ok=True)
    with open(ANOMALY_DETECTOR_PATH, 'wb') as f:
        pickle.dump(detector, f)
    print(f"  ✓ Anomaly detector saved to {ANOMALY_DETECTOR_PATH}")

    print("\n" + "=" * 60)
    print("  Model generation complete!")
    print("  models/rf_model.pkl        — Random Forest classifier")
    print("  models/anomaly_detector.pkl — Isolation Forest anomaly detector")
    print("=" * 60)


if __name__ == '__main__':
    main()
