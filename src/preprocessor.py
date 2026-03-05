"""
SmartContainer Risk Engine — Preprocessor Module
Loads CSVs, renames columns, fixes types, creates leakage-safe train/test split.
"""
import pandas as pd
import numpy as np
import os

# Global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Version tracking
MODULE_VERSION = "1.0.0"

def load_and_prepare(hist_path=None, rt_path=None):
    """
    Load historical and real-time data, clean columns, and create
    a leakage-safe train/test split.

    The real-time dataset is a subset of historical — all 8,481 real-time
    Container IDs exist in historical data with identical labels.
    We MUST exclude real-time Container IDs from training to prevent data leakage.

    Returns:
        train_df: Historical data MINUS real-time IDs (for training)
        test_df:  Real-time data (for inference/evaluation)
    """
    # Auto-detect data file locations
    if hist_path is None:
        if os.path.exists('Historical Data.csv'):
            hist_path = 'Historical Data.csv'
        elif os.path.exists('data/Historical_Data.csv'):
            hist_path = 'data/Historical_Data.csv'
        else:
            raise FileNotFoundError(
                "Cannot find Historical Data CSV. Place it at repo root as "
                "'Historical Data.csv' or in data/Historical_Data.csv"
            )

    if rt_path is None:
        if os.path.exists('Real-Time Data.csv'):
            rt_path = 'Real-Time Data.csv'
        elif os.path.exists('data/Real-Time_Data.csv'):
            rt_path = 'data/Real-Time_Data.csv'
        else:
            raise FileNotFoundError(
                "Cannot find Real-Time Data CSV. Place it at repo root as "
                "'Real-Time Data.csv' or in data/Real-Time_Data.csv"
            )

    print(f"[preprocessor v{MODULE_VERSION}] Loading data...")
    print(f"  Historical: {hist_path}")
    print(f"  Real-Time:  {rt_path}")

    hist = pd.read_csv(hist_path)
    rt = pd.read_csv(rt_path)

    # Rename long/messy column names to clean versions
    rename_map = {
        'Declaration_Date (YYYY-MM-DD)': 'Declaration_Date',
        'Trade_Regime (Import / Export / Transit)': 'Trade_Regime',
    }
    hist.rename(columns=rename_map, inplace=True)
    rt.rename(columns=rename_map, inplace=True)

    # Fix data types
    hist['Declaration_Date'] = pd.to_datetime(hist['Declaration_Date'])
    rt['Declaration_Date'] = pd.to_datetime(rt['Declaration_Date'])
    hist['HS_Code'] = hist['HS_Code'].astype(str)
    rt['HS_Code'] = rt['HS_Code'].astype(str)

    # === LEAKAGE-SAFE SPLIT ===
    # Real-time IDs are a subset of historical IDs.
    # Exclude them from training to prevent data leakage.
    rt_ids = set(rt['Container_ID'])
    train_df = hist[~hist['Container_ID'].isin(rt_ids)].copy()
    test_df = rt.copy()

    print(f"\n  === Leakage-Safe Split ===")
    print(f"  Historical total:  {len(hist):,} rows")
    print(f"  Real-time IDs:     {len(rt_ids):,}")
    print(f"  Training set:      {len(train_df):,} rows (historical MINUS real-time IDs)")
    print(f"  Test set:          {len(test_df):,} rows (= real-time data)")
    print(f"  Overlap check:     {len(set(train_df['Container_ID']) & rt_ids)} shared IDs (must be 0)")
    print(f"\n  Train labels: {train_df['Clearance_Status'].value_counts().to_dict()}")
    print(f"  Test labels:  {test_df['Clearance_Status'].value_counts().to_dict()}")

    return train_df, test_df


if __name__ == '__main__':
    train, test = load_and_prepare()
    print(f"\nDone. Train shape: {train.shape}, Test shape: {test.shape}")
