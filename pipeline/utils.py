"""Shared utilities for all pipeline modules"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler

# Feature columns (standard across all datasets)
FEATURE_COLS = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 
    'NumAromaticRings', 'NumHeteroatoms', 'TPSA', 'NumRings', 'NumAliphaticRings', 
    'NumSaturatedRings', 'FractionCsp3', 'NumValenceElectrons', 'MaxPartialCharge', 
    'MinPartialCharge', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'QED', 'BertzCT', 
    'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'MolMR', 'BalabanJ', 'HallKierAlpha', 
    'NumSaturatedCarbocycles', 'NumAromaticCarbocycles', 'NumSaturatedHeterocycles', 
    'NumAromaticHeterocycles', 'fr_NH2', 'fr_COO', 'fr_benzene', 'fr_furan', 'fr_halogen'
]

# View definitions for Co-Training
VIEW1_CANDIDATES = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumValenceElectrons', 
    'TPSA', 'MaxPartialCharge', 'MinPartialCharge', 'LabuteASA', 'MolMR', 
    'QED', 'NumHeteroatoms'
]

VIEW2_CANDIDATES = [
    'NumRotatableBonds', 'NumAromaticRings', 'NumRings', 'NumAliphaticRings', 
    'NumSaturatedRings', 'FractionCsp3', 'PEOE_VSA1', 'PEOE_VSA2', 'BertzCT', 
    'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'BalabanJ', 'HallKierAlpha', 
    'NumSaturatedCarbocycles', 'NumAromaticCarbocycles', 
    'NumSaturatedHeterocycles', 'NumAromaticHeterocycles', 
    'fr_NH2', 'fr_COO', 'fr_benzene', 'fr_furan', 'fr_halogen'
]

V1_IDX = [FEATURE_COLS.index(f) for f in VIEW1_CANDIDATES if f in FEATURE_COLS]
V2_IDX = [FEATURE_COLS.index(f) for f in VIEW2_CANDIDATES if f in FEATURE_COLS]

# Hyperparameters
MAX_ITERATIONS = 20
SAMPLES_PER_ITER = 50
CONFIDENCE_THRESHOLD = 0.80
OUTLIER_STD_THRESHOLD = 3


def clip_outliers(df, std_threshold=3):
    """
    Clip outliers using standard deviation method (matching notebook).
    Returns clipped dataframe and outlier count.
    """
    df_clipped = df.copy()
    outlier_count = 0
    
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        # Count and clip outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            outlier_count += outliers
            df_clipped[col] = df[col].clip(lower_bound, upper_bound)
    
    return df_clipped, outlier_count


def process_enhanced_data(enhanced_dir, target_col, dataset_name, processed_dir):
    """
    Process enhanced data following notebook steps:
    1. Load enhanced data
    2. Extract features
    3. Clip outliers (3 std)
    4. Remove non-finite values
    5. Scale with StandardScaler
    6. Save processed data
    
    Returns: (X_labeled, y_labeled, X_unlabeled, scaler)
    """
    print(f"\n{'='*60}")
    print(f"📊 Processing Enhanced Data: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Create processed directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load enhanced data
    labeled_path = os.path.join(enhanced_dir, "labeled_features.csv")
    unlabeled_path = os.path.join(enhanced_dir, "unlabeled_features.csv")
    
    if not os.path.exists(labeled_path):
        print(f"   ❌ No labeled CSV found at {labeled_path}")
        return None, None, None, None
    
    print(f"   🔹 Loading enhanced data from: {enhanced_dir}")
    df_labeled = pd.read_csv(labeled_path)
    df_unlabeled = pd.read_csv(unlabeled_path) if os.path.exists(unlabeled_path) else None
    
    # Identify feature columns (exclude identifiers and targets)
    if dataset_name == 'tox21':
        exclude_cols = ['mol_id', 'smiles', 'canonical_smiles', 'toxic']
    else:
        exclude_cols = ['smiles', 'canonical_smiles', 'FDA_APPROVED', 'CT_TOX']
    
    all_features = [col for col in df_labeled.columns if col not in exclude_cols]
    
    # Ensure all FEATURE_COLS are present
    missing_features = set(FEATURE_COLS) - set(all_features)
    if missing_features:
        print(f"   ⚠️ Warning: Missing features: {missing_features}")
        # Use only available features
        all_features = [f for f in FEATURE_COLS if f in all_features]
    
    print(f"   ✓ Using {len(all_features)} features")
    
    # Extract features and target
    X_labeled = df_labeled[all_features].copy()
    y_labeled = df_labeled[target_col].copy()
    
    if df_unlabeled is not None:
        X_unlabeled = df_unlabeled[all_features].copy()
    else:
        X_unlabeled = None
    
    print(f"   ✓ Loaded {len(X_labeled)} labeled samples")
    if X_unlabeled is not None:
        print(f"   ✓ Loaded {len(X_unlabeled)} unlabeled samples")
    
    # Step 1: Clip outliers (3 std)
    print(f"\n   🔹 Step 1: Clipping outliers (±{OUTLIER_STD_THRESHOLD} std)...")
    X_labeled_clipped, labeled_outliers = clip_outliers(X_labeled, OUTLIER_STD_THRESHOLD)
    if X_unlabeled is not None:
        X_unlabeled_clipped, unlabeled_outliers = clip_outliers(X_unlabeled, OUTLIER_STD_THRESHOLD)
        print(f"      ✓ Clipped {labeled_outliers} labeled outliers, {unlabeled_outliers} unlabeled outliers")
    else:
        print(f"      ✓ Clipped {labeled_outliers} labeled outliers")
    
    # Step 2: Remove non-finite values
    print(f"\n   🔹 Step 2: Removing non-finite values...")
    labeled_non_finite_mask = ~np.isfinite(X_labeled_clipped).all(axis=1)
    X_labeled_clean = X_labeled_clipped[~labeled_non_finite_mask].copy()
    y_labeled_clean = y_labeled[~labeled_non_finite_mask].copy()
    
    if X_unlabeled is not None:
        unlabeled_non_finite_mask = ~np.isfinite(X_unlabeled_clipped).all(axis=1)
        X_unlabeled_clean = X_unlabeled_clipped[~unlabeled_non_finite_mask].copy()
        removed_unlabeled = unlabeled_non_finite_mask.sum()
    else:
        X_unlabeled_clean = None
        removed_unlabeled = 0
    
    removed_labeled = labeled_non_finite_mask.sum()
    print(f"      ✓ Removed {removed_labeled} labeled rows, {removed_unlabeled} unlabeled rows")
    print(f"      ✓ Clean data: {len(X_labeled_clean)} labeled, {len(X_unlabeled_clean) if X_unlabeled_clean is not None else 0} unlabeled")
    
    # Step 3: Scale with StandardScaler (fit on labeled only)
    print(f"\n   🔹 Step 3: Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    scaler.fit(X_labeled_clean)
    
    X_labeled_scaled = scaler.transform(X_labeled_clean)
    if X_unlabeled_clean is not None:
        X_unlabeled_scaled = scaler.transform(X_unlabeled_clean)
    else:
        X_unlabeled_scaled = None
    
    # Convert back to DataFrame
    X_labeled_scaled_df = pd.DataFrame(
        X_labeled_scaled,
        columns=all_features,
        index=X_labeled_clean.index
    )
    
    if X_unlabeled_scaled is not None:
        X_unlabeled_scaled_df = pd.DataFrame(
            X_unlabeled_scaled,
            columns=all_features,
            index=X_unlabeled_clean.index
        )
    else:
        X_unlabeled_scaled_df = None
    
    print(f"      ✓ Scaled data shapes: Labeled {X_labeled_scaled_df.shape}, Unlabeled {X_unlabeled_scaled_df.shape if X_unlabeled_scaled_df is not None else (0, 0)}")
    
    # Step 4: Save processed data
    print(f"\n   🔹 Step 4: Saving processed data...")
    
    # Combine features with target for labeled data
    df_labeled_processed = X_labeled_scaled_df.copy()
    df_labeled_processed[target_col] = y_labeled_clean.values
    
    # Save processed data
    labeled_processed_path = os.path.join(processed_dir, "labeled_processed.csv")
    df_labeled_processed.to_csv(labeled_processed_path, index=False)
    print(f"      ✓ Saved: {labeled_processed_path}")
    
    if X_unlabeled_scaled_df is not None:
        unlabeled_processed_path = os.path.join(processed_dir, "unlabeled_processed.csv")
        X_unlabeled_scaled_df.to_csv(unlabeled_processed_path, index=False)
        print(f"      ✓ Saved: {unlabeled_processed_path}")
    
    # Save scaler
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"      ✓ Saved: {scaler_path}")
    
    # Save feature info
    feature_info = {
        'feature_names': all_features,
        'n_features': len(all_features),
        'target_variable': target_col,
        'dataset': dataset_name,
        'scaling_method': 'StandardScaler',
        'outlier_handling': f'clipping_{OUTLIER_STD_THRESHOLD}std'
    }
    feature_info_path = os.path.join(processed_dir, "feature_info.json")
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"      ✓ Saved: {feature_info_path}")
    
    # Return numpy arrays for training
    return X_labeled_scaled, y_labeled_clean.values, X_unlabeled_scaled, scaler
