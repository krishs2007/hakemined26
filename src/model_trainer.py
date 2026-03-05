"""
SmartContainer Risk Engine — Model Trainer Module
Trains a Random Forest classifier with class-weight balancing
to handle extreme class imbalance (Critical=1%, Clear=78%).
"""
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

MODULE_VERSION = "1.0.0"
RANDOM_SEED = 42

# Label encoding
LABEL_MAP = {'Clear': 0, 'Low Risk': 1, 'Critical': 2}
LABEL_REVERSE = {0: 'Clear', 1: 'Low Risk', 2: 'Critical'}

# Model save path
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')

def train_model(train_feat, feature_cols):
    """
    Train Random Forest with class weights to handle imbalance.
    Critical is weighted 50x more than Clear.

    Args:
        train_feat: DataFrame with features + Clearance_Status column
        feature_cols: list of feature column names

    Returns:
        Trained RandomForestClassifier model
    """
    X = train_feat[feature_cols]
    y = train_feat['Clearance_Status'].map(LABEL_MAP)

    print(f"[model_trainer v{MODULE_VERSION}] Training Random Forest...")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(X):,}")
    print(f"  Class distribution: {y.value_counts().to_dict()}")

    # Class weights handle imbalance: Critical=1%, Clear=78%
    # Weight Critical 50x more than Clear
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight={0: 1, 1: 2, 2: 50},
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Print feature importances (top 10)
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances),
                      key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 Feature Importances:")
    for name, imp in feat_imp[:10]:
        print(f"    {name:30s} {imp:.4f} ({imp*100:.1f}%)")

    # Training accuracy (sanity check)
    train_pred = model.predict(X)
    train_acc = (train_pred == y).mean()
    print(f"\n  Training accuracy: {train_acc:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {MODEL_PATH}")

    return model

def load_model():
    """Load a previously trained model from disk."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"[model_trainer] Loaded model from {MODEL_PATH}")
    return model

def get_probabilities(model, X):
    """
    Get prediction probabilities.

    Returns:
        Array of shape (n, 3) -> [P(Clear), P(Low Risk), P(Critical)] per row
    """
    return model.predict_proba(X)

def evaluate_model(model, test_feat, feature_cols):
    """
    Evaluate model on test data with ground truth labels.
    Prints classification report and confusion matrix.
    """
    X_test = test_feat[feature_cols]
    y_true = test_feat['Clearance_Status'].map(LABEL_MAP)
    y_pred = model.predict(X_test)

    print(f"\n  === Model Evaluation (Test Set) ===")
    print(classification_report(
        y_true, y_pred,
        target_names=['Clear', 'Low Risk', 'Critical']
    ))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {cm}")

    return y_pred

if __name__ == '__main__':
    from preprocessor import load_and_prepare
    from feature_engineer import build_direct_features, build_historical_features, FEATURE_COLS

    train_df, test_df = load_and_prepare()
    train_feat = build_direct_features(train_df)
    train_feat = build_historical_features(train_feat, train_df)
    test_feat = build_direct_features(test_df)
    test_feat = build_historical_features(test_feat, train_df)

    model = train_model(train_feat, FEATURE_COLS)
    evaluate_model(model, test_feat, FEATURE_COLS)