"""
SmartContainer Risk Engine — Feature Engineering Module
Builds all 26 features from raw fields.
Features are built on both train and test, but all statistics
come from training data only to avoid leakage.
"""
import pandas as pd
import numpy as np

MODULE_VERSION = "1.0.0"

# === 26 features total ===
FEATURE_COLS = [
    # Weight features (top signal — ~37.6% importance)
    'weight_diff_abs_pct', 'weight_over_declared', 'is_zero_weight',
    'weight_flag_mild', 'weight_flag_moderate', 'weight_flag_severe',

    # Value features
    'value_per_kg', 'log_value', 'log_weight', 'is_zero_value',

    # Dwell time features (2nd signal — ~10.9% importance)
    'log_dwell', 'is_fast_clear', 'is_long_dwell', 'is_very_long',

    # Temporal features
    'declaration_hour', 'is_off_hours', 'is_weekend',

    # Regime + importer flags
    'is_transit', 'is_new_importer',

    # Historical behavioral stats (train-only aggregates)
    'importer_critical_rate',      # ~22.5% importance
    'importer_shipment_count',
    'importer_avg_weight_disc',
    'country_critical_rate',
    'port_critical_rate',
    'hs_critical_rate',            # ~11.7% importance
    'hs_shipment_count',
]

def build_direct_features(df):
    """
    Part A: Direct computed features — no leakage risk.
    These are derived purely from each row's own fields.
    """
    df = df.copy()

    # --- WEIGHT FEATURES (strongest signal: 0.38 correlation with Critical) ---
    df['weight_diff_pct'] = (
        (df['Measured_Weight'] - df['Declared_Weight'])
        / df['Declared_Weight'].replace(0, np.nan) * 100
    )
    df['weight_diff_abs_pct'] = df['weight_diff_pct'].abs().fillna(0)
    df['weight_over_declared'] = (df['weight_diff_pct'] > 0).astype(int)
    df['is_zero_weight'] = (df['Declared_Weight'] == 0).astype(int)

    # Thresholds from data analysis:
    # Clear avg: 0.02% | Low Risk: 4.5% | Critical: 12.8%
    df['weight_flag_mild'] = (df['weight_diff_abs_pct'] > 5).astype(int)
    df['weight_flag_moderate'] = (df['weight_diff_abs_pct'] > 15).astype(int)
    df['weight_flag_severe'] = (df['weight_diff_abs_pct'] > 30).astype(int)

    # --- VALUE FEATURES ---
    df['value_per_kg'] = (
        df['Declared_Value'] / df['Declared_Weight'].replace(0, np.nan)
    ).fillna(0)
    df['log_value'] = np.log1p(df['Declared_Value'])
    df['log_weight'] = np.log1p(df['Declared_Weight'])
    df['is_zero_value'] = (df['Declared_Value'] == 0).astype(int)

    # --- DWELL TIME FEATURES (2nd strongest: 0.18 correlation) ---
    # Thresholds from data: 5th pct=10.2hrs, 95th=98.0hrs, 99th=118.3hrs
    # Critical avg: 86.9hrs vs Clear avg: 40.5hrs
    df['log_dwell'] = np.log1p(df['Dwell_Time_Hours'])
    df['is_fast_clear'] = (df['Dwell_Time_Hours'] < 10).astype(int)
    df['is_long_dwell'] = (df['Dwell_Time_Hours'] > 98).astype(int)
    df['is_very_long'] = (df['Dwell_Time_Hours'] > 118).astype(int)

    # --- TEMPORAL FEATURES ---
    df['declaration_hour'] = df['Declaration_Time'].str.split(':').str[0].astype(int)
    df['is_off_hours'] = (
        (df['declaration_hour'] < 6) | (df['declaration_hour'] > 22)
    ).astype(int)
    df['declaration_dow'] = df['Declaration_Date'].dt.dayofweek
    df['is_weekend'] = (df['declaration_dow'] >= 5).astype(int)

    # --- REGIME FLAG ---
    df['is_transit'] = (df['Trade_Regime'] == 'Transit').astype(int)

    return df

def build_historical_features(df, train_ref):
    """
    Part B: Historical statistics features.
    Compute risk statistics from training data and map onto df.
    train_ref must ALWAYS be the TRAINING set, never the test set.
    """
    tr = train_ref.copy()
    tr['is_critical'] = (tr['Clearance_Status'] == 'Critical').astype(int)
    tr['wda'] = (
        (tr['Measured_Weight'] - tr['Declared_Weight'])
        / tr['Declared_Weight'].replace(0, np.nan) * 100
    ).abs().fillna(0)

    pop_rate = tr['is_critical'].mean()  # ~0.0104 = 1.04% baseline

    # Importer risk profile (strongest behavioral signal)
    # ~360 importers have critical_rate > 10%
    imp_stats = tr.groupby('Importer_ID').agg(
        importer_critical_rate=('is_critical', 'mean'),
        importer_shipment_count=('Container_ID', 'count'),
        importer_avg_weight_disc=('wda', 'mean'),
    ).reset_index()

    # Country risk (NO = 12.5%, UA = 7.5%, DK = 6.5% critical rate)
    country_stats = tr.groupby('Origin_Country').agg(
        country_critical_rate=('is_critical', 'mean'),
    ).reset_index()

    # Port risk (PORT_60 = 12.5%, PORT_14 = 6.7%)
    port_stats = tr.groupby('Destination_Port').agg(
        port_critical_rate=('is_critical', 'mean'),
    ).reset_index()

    # HS Code risk (108 HS codes have any Critical history)
    hs_stats = tr.groupby('HS_Code').agg(
        hs_critical_rate=('is_critical', 'mean'),
        hs_shipment_count=('Container_ID', 'count'),
    ).reset_index()

    # Merge all onto df
    df = df.copy()
    df = df.merge(imp_stats, on='Importer_ID', how='left')
    df = df.merge(country_stats, on='Origin_Country', how='left')
    df = df.merge(port_stats, on='Destination_Port', how='left')
    df = df.merge(hs_stats, on='HS_Code', how='left')

    # Fill unknowns with population baseline
    for col in ['importer_critical_rate', 'country_critical_rate',
                'port_critical_rate', 'hs_critical_rate']:
        df[col] = df[col].fillna(pop_rate)

    df['importer_shipment_count'] = df['importer_shipment_count'].fillna(1)
    df['importer_avg_weight_disc'] = df['importer_avg_weight_disc'].fillna(0)
    df['hs_shipment_count'] = df['hs_shipment_count'].fillna(1)
    df['is_new_importer'] = (df['importer_shipment_count'] == 1).astype(int)

    print(f"[feature_engineer v{MODULE_VERSION}] Built {len(FEATURE_COLS)} features")
    print(f"  Population critical rate (baseline): {pop_rate:.4f}")

    return df

if __name__ == '__main__':
    from preprocessor import load_and_prepare
    train_df, test_df = load_and_prepare()
    train_feat = build_direct_features(train_df)
    train_feat = build_historical_features(train_feat, train_df)
    test_feat = build_direct_features(test_df)
    test_feat = build_historical_features(test_feat, train_df)
    print(f"\nTrain features shape: {train_feat.shape}")
    print(f"Test features shape:  {test_feat.shape}")
    print(f"Feature columns:      {len(FEATURE_COLS)}")
