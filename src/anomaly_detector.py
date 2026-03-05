"""
SmartContainer Risk Engine — Anomaly Detector Module
Runs Isolation Forest as a secondary unsupervised layer.
Catches multivariate outliers the supervised model might miss.

Normalization parameters are fitted on TRAINING data and reused
for test/inference to ensure stable scoring across runs.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

MODULE_VERSION = "1.0.0"
RANDOM_SEED = 42

# Features used for anomaly detection
ANOMALY_FEATURES = [
    'weight_diff_abs_pct', 'value_per_kg',
    'log_dwell', 'log_value', 'log_weight',
]

# Anomaly score contributes up to this many points to final risk score
ANOMALY_MAX_SCORE = 30


class AnomalyDetector:
    """
    Wraps Isolation Forest with stable normalization.
    Normalization min/max are fitted on training scores and reused on test.
    """

    def __init__(self, contamination=0.05):
        self.iso = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.train_score_min = None
        self.train_score_max = None

    def fit(self, train_feat):
        """Fit Isolation Forest on training data and record normalization params."""
        print(f"[anomaly_detector v{MODULE_VERSION}] Fitting Isolation Forest...")
        self.iso.fit(train_feat[ANOMALY_FEATURES])

        # Compute raw scores on training data to establish normalization range
        raw_scores = -self.iso.score_samples(train_feat[ANOMALY_FEATURES])
        self.train_score_min = raw_scores.min()
        self.train_score_max = raw_scores.max()

        print(f"  Training anomaly score range: [{self.train_score_min:.4f}, {self.train_score_max:.4f}]")
        return self

    def score(self, df_feat):
        """
        Score new data using fitted model + training normalization params.
        Returns normalized scores in [0, ANOMALY_MAX_SCORE] range.
        """
        raw_scores = -self.iso.score_samples(df_feat[ANOMALY_FEATURES])

        # Normalize using TRAINING min/max (stable across runs)
        score_range = self.train_score_max - self.train_score_min
        if score_range == 0:
            score_range = 1.0  # prevent division by zero

        normalized = (raw_scores - self.train_score_min) / score_range * ANOMALY_MAX_SCORE
        normalized = np.clip(normalized, 0, ANOMALY_MAX_SCORE)

        return normalized


def train_anomaly_detector(train_feat):
    """Convenience function: create + fit anomaly detector."""
    detector = AnomalyDetector()
    detector.fit(train_feat)
    return detector


def get_anomaly_scores(detector, df_feat):
    """Convenience function: score data with fitted detector."""
    scores = detector.score(df_feat)
    print(f"  Anomaly scores — mean: {scores.mean():.2f}, "
          f"median: {np.median(scores):.2f}, max: {scores.max():.2f}")
    return scores


if __name__ == '__main__':
    from preprocessor import load_and_prepare
    from feature_engineer import build_direct_features, build_historical_features

    train_df, test_df = load_and_prepare()
    train_feat = build_direct_features(train_df)
    train_feat = build_historical_features(train_feat, train_df)
    test_feat = build_direct_features(test_df)
    test_feat = build_historical_features(test_feat, train_df)

    detector = train_anomaly_detector(train_feat)
    scores = get_anomaly_scores(detector, test_feat)
    print(f"\nAnomaly scores shape: {scores.shape}")