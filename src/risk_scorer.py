"""
SmartContainer Risk Engine — Risk Scorer Module
Combines model probabilities + anomaly scores + rule flags
into a final 0–100 risk score per container.
"""
import numpy as np

MODULE_VERSION = "1.0.0"

# === Configurable Threshold Constants ===
# Adjust these to calibrate risk level cutoffs
THRESHOLD_LOW_MEDIUM = 31       # scores below this → Low Risk
THRESHOLD_MEDIUM_CRITICAL = 61  # scores at/above this → Critical

# Component weights (must sum to 1.0)
WEIGHT_MODEL = 0.70      # supervised model component
WEIGHT_ANOMALY = 0.20    # unsupervised anomaly component
WEIGHT_RULES = 0.10      # rule-based hard flags component

# Max rule-based score contribution
RULE_MAX_SCORE = 10


def compute_risk_score(probs, anomaly_scores, df_feat):
    """
    Combine model probabilities + anomaly scores + rule flags
    into final 0–100 risk score.

    Args:
        probs: array shape (n, 3) → [P(Clear), P(LowRisk), P(Critical)]
        anomaly_scores: array shape (n,) → 0–30 range
        df_feat: DataFrame with feature columns for rule evaluation

    Returns:
        Array of final risk scores, clipped to [0, 100], rounded to 1 decimal
    """
    # Model component (70% of score)
    # P(Low Risk) contributes up to 30 points, P(Critical) up to 100
    model_score = (probs[:, 1] * 30 + probs[:, 2] * 100) * WEIGHT_MODEL

    # Anomaly component (20% of score)
    anomaly_component = anomaly_scores * WEIGHT_ANOMALY

    # Rule-based hard flags (10% of score — catches obvious cases)
    rule_score = np.zeros(len(df_feat))
    rule_score += (df_feat['weight_flag_severe'].values * 8)    # >30% weight diff
    rule_score += (df_feat['is_zero_value'].values * 5)         # declared value = 0
    rule_score += (df_feat['is_zero_weight'].values * 5)        # declared weight = 0
    rule_score += (df_feat['is_very_long'].values * 4)          # dwell > 118hrs
    rule_score = np.clip(rule_score, 0, RULE_MAX_SCORE)

    # Combine
    final_score = model_score + anomaly_component + rule_score
    final_score = np.clip(final_score, 0, 100)

    print(f"[risk_scorer v{MODULE_VERSION}] Risk scores computed")
    print(f"  Model component     — mean: {model_score.mean():.2f}")
    print(f"  Anomaly component   — mean: {anomaly_component.mean():.2f}")
    print(f"  Rule component      — mean: {rule_score.mean():.2f}")
    print(f"  Final score         — mean: {final_score.mean():.2f}, "
          f"median: {np.median(final_score):.2f}, max: {final_score.max():.2f}")

    return final_score.round(1)


def classify_risk_level(risk_scores):
    """
    3-level classification based on risk scores:
        0–30   → Low Risk
        31–60  → Medium Risk
        61–100 → Critical

    Thresholds are configurable via module constants.
    """
    levels = np.where(
        risk_scores >= THRESHOLD_MEDIUM_CRITICAL, 'Critical',
        np.where(risk_scores >= THRESHOLD_LOW_MEDIUM, 'Medium Risk', 'Low Risk')
    )

    unique, counts = np.unique(levels, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"  Risk level distribution: {dist}")

    return levels