"""TOX21 MLOps Pipeline - Complete End-to-End Workflow"""

import os
import sys
import wandb
import numpy as np
import pandas as pd
import joblib
import time
import random
from datetime import datetime, timedelta
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score
)
from sklearn.utils import resample

# Configuration
PROJECT = "QSAR_MLOPS_TOX21"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "raw", "enhanced_data", "tox21"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "processed", "tox21"), exist_ok=True)


# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================

def canonicalize_smiles(smiles):
    """Standardize SMILES representation using RDKit"""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def compute_comprehensive_features(smiles):
    """Compute comprehensive molecular descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            features = {}
            
            # Basic molecular properties
            features['MolWt'] = Descriptors.MolWt(mol)
            features['LogP'] = Descriptors.MolLogP(mol)
            features['NumHDonors'] = Descriptors.NumHDonors(mol)
            features['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            features['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            features['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            
            # Lipinski's Rule of Five
            features['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            features['TPSA'] = Descriptors.TPSA(mol)
            
            # Complexity and shape
            features['NumRings'] = Descriptors.RingCount(mol)
            features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            features['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            features['FractionCsp3'] = Descriptors.FractionCSP3(mol)
            
            # Electronic properties
            features['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
            
            try:
                features['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
                features['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
            except:
                features['MaxPartialCharge'] = 0
                features['MinPartialCharge'] = 0
            
            # Molecular surface area
            features['LabuteASA'] = Descriptors.LabuteASA(mol)
            features['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
            features['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
            
            # Drug-likeness scores
            features['QED'] = QED.qed(mol)
            
            # Topological descriptors
            features['BertzCT'] = Descriptors.BertzCT(mol)
            features['Chi0v'] = Descriptors.Chi0v(mol)
            features['Chi1v'] = Descriptors.Chi1v(mol)
            features['Kappa1'] = Descriptors.Kappa1(mol)
            features['Kappa2'] = Descriptors.Kappa2(mol)
            
            # Additional descriptors
            features['MolMR'] = Descriptors.MolMR(mol)
            features['BalabanJ'] = Descriptors.BalabanJ(mol)
            features['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)
            features['NumSaturatedCarbocycles'] = Descriptors.NumSaturatedCarbocycles(mol)
            features['NumAromaticCarbocycles'] = Descriptors.NumAromaticCarbocycles(mol)
            features['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(mol)
            features['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
            
            # Pharmacophore features
            features['fr_NH2'] = Descriptors.fr_NH2(mol)
            features['fr_COO'] = Descriptors.fr_COO(mol)
            features['fr_benzene'] = Descriptors.fr_benzene(mol)
            features['fr_furan'] = Descriptors.fr_furan(mol)
            features['fr_halogen'] = Descriptors.fr_halogen(mol)
            
            return pd.Series(features)
    except Exception as e:
        print(f"Error computing features: {e}")
    return pd.Series()


def clip_outliers(df, std_threshold=3):
    """Clip outliers using standard deviation method"""
    df_clipped = df.copy()
    outlier_count = 0
    
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            outlier_count += outliers
            df_clipped[col] = df[col].clip(lower_bound, upper_bound)
    
    return df_clipped, outlier_count


def prepare_data():
    """Phase 1: Data Ingestion & Preparation"""
    print("=" * 80)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 80)
    
    wandb.init(project=PROJECT, job_type="prepare-data")
    
    # Load raw data
    print("\nLoading raw data...")
    raw_df_unlabeled = pd.read_csv(os.path.join(DATA_DIR, 'raw/original_data/zinc_unlabeled.csv'))
    raw_df_labeled = pd.read_csv(os.path.join(DATA_DIR, 'raw/original_data/tox21.csv'))
    
    # Canonicalize SMILES
    print("Canonicalizing SMILES...")
    raw_df_labeled['canonical_smiles'] = raw_df_labeled['smiles'].apply(canonicalize_smiles)
    raw_df_labeled = raw_df_labeled.dropna(subset=['canonical_smiles'])
    
    # Create binary toxicity target
    print("Creating toxicity target...")
    tox_columns = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                   'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    raw_df_labeled['toxic'] = raw_df_labeled[tox_columns].max(axis=1)
    raw_df_labeled = raw_df_labeled.dropna(subset=['toxic'])
    raw_df_labeled['toxic'] = raw_df_labeled['toxic'].astype(int)
    raw_df_labeled = raw_df_labeled.drop(columns=tox_columns)
    
    # Balance classes
    print("Balancing classes...")
    toxic_df = raw_df_labeled[raw_df_labeled['toxic'] == 1]
    non_toxic_df = raw_df_labeled[raw_df_labeled['toxic'] == 0]
    non_toxic_downsampled = resample(non_toxic_df, replace=False, 
                                      n_samples=len(toxic_df), random_state=42)
    raw_df_labeled_balanced = pd.concat([toxic_df, non_toxic_downsampled])
    raw_df_labeled_balanced = raw_df_labeled_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Compute features for labeled data
    print("Computing features for labeled data...")
    labeled_features = raw_df_labeled_balanced['canonical_smiles'].apply(compute_comprehensive_features)
    all_labeled_with_features = pd.concat([raw_df_labeled_balanced, labeled_features], axis=1)
    all_labeled_with_features = all_labeled_with_features.dropna()
    all_labeled_with_features.to_csv(
        os.path.join(DATA_DIR, 'raw/enhanced_data/tox21/labeled_features.csv'), index=False
    )
    
    # Compute features for unlabeled data
    print("Computing features for unlabeled data...")
    raw_df_unlabeled['canonical_smiles'] = raw_df_unlabeled['smiles'].apply(canonicalize_smiles)
    raw_df_unlabeled = raw_df_unlabeled.dropna(subset=['canonical_smiles'])
    unlabeled_features = raw_df_unlabeled['canonical_smiles'].apply(compute_comprehensive_features)
    unlabeled_with_features = pd.concat(
        [raw_df_unlabeled[['smiles', 'canonical_smiles']], unlabeled_features], axis=1
    )
    unlabeled_with_features['toxic'] = np.nan
    unlabeled_with_features = unlabeled_with_features.dropna(subset=unlabeled_features.columns.tolist())
    unlabeled_with_features.to_csv(
        os.path.join(DATA_DIR, 'raw/enhanced_data/tox21/unlabeled_features.csv'), index=False
    )
    
    # Prepare feature matrices
    print("Preparing feature matrices...")
    exclude_cols = ['smiles', 'canonical_smiles', 'FDA_APPROVED', 'toxic', 'mol_id']
    all_features = [col for col in all_labeled_with_features.columns if col not in exclude_cols]
    
    X_labeled = all_labeled_with_features[all_features]
    y_tox = all_labeled_with_features['toxic']
    X_unlabeled = unlabeled_with_features[all_features]
    
    # Clip outliers
    print("Clipping outliers...")
    X_labeled_clipped, _ = clip_outliers(X_labeled, std_threshold=3)
    X_unlabeled_clipped, _ = clip_outliers(X_unlabeled, std_threshold=3)
    
    # Save processed data
    print("Saving processed data...")
    df_labeled_processed = X_labeled_clipped.copy()
    df_labeled_processed['toxic'] = y_tox.values
    df_unlabeled_processed = X_unlabeled_clipped.copy()
    df_unlabeled_processed['toxic'] = np.nan
    
    df_labeled_processed.to_csv(
        os.path.join(DATA_DIR, 'processed/tox21/labeled_processed.csv'), index=False
    )
    df_unlabeled_processed.to_csv(
        os.path.join(DATA_DIR, 'processed/tox21/unlabeled_processed.csv'), index=False
    )
    
    # Create W&B artifacts
    print("Creating W&B artifacts...")
    artifact_labeled = wandb.Artifact(
        name="tox21-labeled-dataset",
        type="dataset",
        description="Cleaned labeled tox21 data"
    )
    artifact_labeled.add_file(os.path.join(DATA_DIR, 'processed/tox21/labeled_processed.csv'))
    wandb.log_artifact(artifact_labeled)
    
    artifact_unlabeled = wandb.Artifact(
        name="zinc-unlabeled-dataset",
        type="dataset",
        description="Cleaned unlabeled zinc data"
    )
    artifact_unlabeled.add_file(os.path.join(DATA_DIR, 'processed/tox21/unlabeled_processed.csv'))
    wandb.log_artifact(artifact_unlabeled)
    
    wandb.finish()
    print("\nPhase 1 Complete: Data Prepared")


# ============================================================================
# PHASE 2 & 3: BASELINE MODEL & HYPERPARAMETER OPTIMIZATION
# ============================================================================

def train_baseline():
    """Train baseline model with sweep configuration"""
    run = wandb.init()
    config = wandb.config
    
    # Load data
    artifact_labeled = run.use_artifact('tox21-labeled-dataset:latest')
    data_path = artifact_labeled.download()
    df_labeled = pd.read_csv(f"{data_path}/labeled_processed.csv")
    
    X = df_labeled.drop('toxic', axis=1)
    y = df_labeled['toxic']
    
    # Handle NaN and Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'val_accuracy': accuracy_score(y_test, y_pred),
        'val_precision': precision_score(y_test, y_pred, zero_division=0),
        'val_recall': recall_score(y_test, y_pred, zero_division=0),
        'val_f1': f1_score(y_test, y_pred, zero_division=0),
        'val_roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    wandb.log(metrics)
    print(f"F1: {metrics['val_f1']:.4f}, ROC-AUC: {metrics['val_roc_auc']:.4f}")


def run_baseline_sweep(count=20):
    """Phase 2 & 3: Baseline Model & Hyperparameter Optimization"""
    print("=" * 80)
    print("PHASE 2 & 3: BASELINE MODEL & HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    baseline_sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_f1', 'goal': 'maximize'},
        'parameters': {
            'n_estimators': {'values': [50, 100, 150, 200]},
            'max_depth': {'values': [10, 15, 20, 25, 30]},
            'min_samples_split': {'values': [2, 5, 10]},
            'min_samples_leaf': {'values': [1, 2, 4]},
            'max_features': {'values': ['sqrt', 'log2']}
        }
    }
    
    sweep_id = wandb.sweep(baseline_sweep_config, project=PROJECT)
    wandb.agent(sweep_id, train_baseline, count=count)
    
    print("\nPhase 2 & 3 Complete: Baseline Model Optimized")
    return sweep_id


def register_best_baseline(sweep_id):
    """Register the best baseline model"""
    print("\nRegistering best baseline model...")
    
    api = wandb.Api()
    entity = api.default_entity
    
    baseline_sweep = api.sweep(f"{entity}/{PROJECT}/{sweep_id}")
    best_baseline_run = baseline_sweep.best_run()
    
    print(f"\nBEST BASELINE MODEL")
    print(f"F1-Score: {best_baseline_run.summary.get('val_f1'):.4f}")
    print(f"ROC-AUC: {best_baseline_run.summary.get('val_roc_auc'):.4f}")
    
    best_config = {
        'n_estimators': best_baseline_run.config['n_estimators'],
        'max_depth': best_baseline_run.config['max_depth'],
        'min_samples_split': best_baseline_run.config['min_samples_split'],
        'min_samples_leaf': best_baseline_run.config['min_samples_leaf'],
        'max_features': best_baseline_run.config['max_features']
    }
    
    # Retrain and save
    run = wandb.init(project=PROJECT, job_type="register-baseline-model")
    
    artifact = run.use_artifact('tox21-labeled-dataset:latest')
    data_path = artifact.download()
    df_labeled = pd.read_csv(f"{data_path}/labeled_processed.csv")
    
    X = df_labeled.drop('toxic', axis=1)
    y = df_labeled['toxic']
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(**best_config, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_baseline_rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    baseline_artifact = wandb.Artifact(
        name='tox21-baseline-rf-model',
        type='model',
        description='Best baseline Random Forest model',
        metadata={'method': 'baseline_supervised', **best_config}
    )
    baseline_artifact.add_file('models/best_baseline_rf_model.pkl')
    baseline_artifact.add_file('models/scaler.pkl')
    run.log_artifact(baseline_artifact)
    run.finish()
    
    print("Best baseline model registered")
    return best_config


# ============================================================================
# PHASE 4: SEMI-SUPERVISED LEARNING
# ============================================================================

def train_ssl(best_baseline_config, best_baseline_f1):
    """Train SSL model with sweep configuration"""
    run = wandb.init()
    config = wandb.config
    
    # Load data
    artifact_labeled = run.use_artifact('tox21-labeled-dataset:latest')
    data_path_labeled = artifact_labeled.download()
    df_labeled = pd.read_csv(f"{data_path_labeled}/labeled_processed.csv")
    
    artifact_unlabeled = run.use_artifact('zinc-unlabeled-dataset:latest')
    data_path_unlabeled = artifact_unlabeled.download()
    df_unlabeled = pd.read_csv(f"{data_path_unlabeled}/unlabeled_processed.csv")
    
    X = df_labeled.drop('toxic', axis=1)
    y = df_labeled['toxic']
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    X_unlabeled_full = df_unlabeled.drop('toxic', axis=1)
    X_unlabeled_full = X_unlabeled_full.replace([np.inf, -np.inf], np.nan).fillna(X_unlabeled_full.median())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_unlabeled_scaled = scaler.transform(X_unlabeled_full)
    
    n_unlabeled = min(config.n_unlabeled, len(X_unlabeled_scaled))
    X_unlabeled = X_unlabeled_scaled[:n_unlabeled]
    
    # Train based on SSL method
    if config.ssl_method == 'label_propagation':
        X_combined = np.vstack([X_train_scaled, X_unlabeled])
        y_combined = np.concatenate([y_train.values, np.full(n_unlabeled, -1)])
        
        model = LabelPropagation(
            kernel='rbf',
            gamma=config.lp_gamma,
            max_iter=config.lp_max_iter,
            n_jobs=-1
        )
        model.fit(X_combined, y_combined)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    elif config.ssl_method == 'self_training':
        X_combined = np.vstack([X_train_scaled, X_unlabeled])
        y_combined = np.concatenate([y_train.values, np.full(n_unlabeled, -1)])
        
        base_clf = RandomForestClassifier(**best_baseline_config, random_state=42, 
                                          n_jobs=-1, class_weight='balanced')
        model = SelfTrainingClassifier(
            base_estimator=base_clf,
            threshold=config.st_threshold,
            max_iter=config.st_max_iter,
            verbose=False
        )
        model.fit(X_combined, y_combined)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    metrics = {
        'val_accuracy': accuracy_score(y_test, y_pred),
        'val_precision': precision_score(y_test, y_pred, zero_division=0),
        'val_recall': recall_score(y_test, y_pred, zero_division=0),
        'val_f1': f1_score(y_test, y_pred, zero_division=0),
        'val_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'n_unlabeled_used': n_unlabeled,
        'n_labeled': len(X_train)
    }
    
    improvement = ((metrics['val_f1'] - best_baseline_f1) / best_baseline_f1) * 100
    metrics['improvement_over_baseline_pct'] = improvement
    
    wandb.log(metrics)
    print(f"F1: {metrics['val_f1']:.4f} (+{improvement:+.2f}%), ROC-AUC: {metrics['val_roc_auc']:.4f}")


def run_ssl_sweep(best_baseline_config, best_baseline_f1, count=50):
    """Phase 4: Semi-Supervised Learning"""
    print("=" * 80)
    print("PHASE 4: SEMI-SUPERVISED LEARNING")
    print("=" * 80)
    
    ssl_sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_f1', 'goal': 'maximize'},
        'parameters': {
            'ssl_method': {'values': ['label_propagation', 'self_training']},
            'n_unlabeled': {'values': [5000, 10000, 15000, 20000]},
            'lp_gamma': {'distribution': 'log_uniform_values', 'min': 0.001, 'max': 0.5},
            'lp_max_iter': {'values': [500, 1000, 1500]},
            'st_threshold': {'distribution': 'uniform', 'min': 0.7, 'max': 0.95},
            'st_max_iter': {'values': [5, 10, 15]}
        }
    }
    
    sweep_id = wandb.sweep(ssl_sweep_config, project=PROJECT)
    
    def train_ssl_wrapper():
        train_ssl(best_baseline_config, best_baseline_f1)
    
    wandb.agent(sweep_id, train_ssl_wrapper, count=count)
    
    print("\nPhase 4 Complete: SSL Models Trained")
    return sweep_id


# ============================================================================
# PHASE 5: PRODUCTION MONITORING
# ============================================================================

def monitor_production(num_requests=100):
    """Phase 5: Production Monitoring"""
    print("=" * 80)
    print("PHASE 5: PRODUCTION MONITORING")
    print("=" * 80)
    
    wandb.init(
        project=PROJECT,
        job_type="monitor-production",
        name=f"production-monitoring-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=["production", "monitoring", "tox21"]
    )
    
    ALERT_THRESHOLD_MS = 40
    CONFIDENCE_THRESHOLD = 0.5
    
    print(f"Simulating {num_requests} production requests...")
    
    for i in range(num_requests):
        prediction_time_ms = random.uniform(10, 50)
        prediction_confidence = random.uniform(0.7, 0.99)
        
        predictions = {
            "NR-AR": random.uniform(0, 1),
            "NR-ER": random.uniform(0, 1),
            "SR-ARE": random.uniform(0, 1),
            "SR-p53": random.uniform(0, 1),
        }
        
        avg_prediction = np.mean(list(predictions.values()))
        data_drift_score = random.uniform(0, 0.3)
        
        wandb.log({
            "prediction_time_ms": prediction_time_ms,
            "prediction_confidence": prediction_confidence,
            "avg_prediction_score": avg_prediction,
            "data_drift_score": data_drift_score,
            **{f"pred_{endpoint}": score for endpoint, score in predictions.items()},
            "request_id": i,
            "timestamp": time.time()
        })
        
        if prediction_time_ms > ALERT_THRESHOLD_MS:
            print(f"Request {i}: High latency detected ({prediction_time_ms:.2f}ms)")
        
        time.sleep(0.1)
    
    wandb.finish()
    print("\nPhase 5 Complete: Production Monitoring")


# ============================================================================
# PHASE 6: AUTOMATED RETRAINING
# ============================================================================

def check_and_retrain():
    """Phase 6: Automated Retraining"""
    print("=" * 80)
    print("PHASE 6: AUTOMATED RETRAINING PIPELINE")
    print("=" * 80)
    
    CONFIDENCE_THRESHOLD = 0.70
    DRIFT_THRESHOLD = 0.25
    AUTO_RETRAIN_ENABLED = True
    
    api = wandb.Api()
    entity = api.default_entity
    
    print(f"\nFetching monitoring runs...")
    all_runs = api.runs(f"{entity}/{PROJECT}")
    monitor_runs = [run for run in all_runs if run.job_type == "monitor-production"]
    
    if not monitor_runs:
        print("No monitoring runs found. Run production monitoring first.")
        return
    
    print(f"✓ Found {len(monitor_runs)} monitoring runs")
    
    latest_monitor_run = max(monitor_runs, key=lambda r: r.created_at)
    print(f"Latest run: {latest_monitor_run.name}")
    
    history = latest_monitor_run.history()
    
    if history.empty:
        print("No metrics found in monitoring run history")
        return
    
    avg_confidence = history['prediction_confidence'].mean()
    avg_drift = history['data_drift_score'].mean()
    
    print(f"\nPerformance Metrics:")
    print(f"  Average Confidence: {avg_confidence:.4f}")
    print(f"  Average Drift Score: {avg_drift:.4f}")
    
    needs_retraining = False
    retraining_reasons = []
    
    if avg_confidence < CONFIDENCE_THRESHOLD:
        needs_retraining = True
        retraining_reasons.append(f"Low confidence: {avg_confidence:.4f} < {CONFIDENCE_THRESHOLD}")
    
    if avg_drift > DRIFT_THRESHOLD:
        needs_retraining = True
        retraining_reasons.append(f"High drift: {avg_drift:.4f} > {DRIFT_THRESHOLD}")
    
    if needs_retraining and AUTO_RETRAIN_ENABLED:
        print(f"\nPERFORMANCE DEGRADATION DETECTED!")
        print(f"Reasons:")
        for reason in retraining_reasons:
            print(f"  • {reason}")
        
        print(f"\nTRIGGERING AUTOMATED RETRAINING...")
        
        alert_run = wandb.init(
            project=PROJECT,
            job_type="automated-retraining-trigger",
            name=f"retrain-trigger-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=["retraining", "automated", "triggered"]
        )
        
        wandb.log({
            'retraining_triggered': 1,
            'avg_confidence': avg_confidence,
            'avg_drift': avg_drift,
            'timestamp': datetime.now().timestamp()
        })
        
        wandb.alert(
            title="Automated Retraining Triggered",
            text=f"Model performance degraded. Reasons: {', '.join(retraining_reasons)}",
            level=wandb.AlertLevel.WARN
        )
        
        print(f"✓ Retraining trigger logged to W&B")
        alert_run.finish()
    else:
        print(f"\nMODEL PERFORMANCE IS HEALTHY")
    
    print("\nPhase 6 Complete: Automated Retraining Check")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline():
    """Run the complete MLOps pipeline"""
    print("\n" + "=" * 80)
    print("TOX21 MLOPS PIPELINE - COMPLETE WORKFLOW")
    print("=" * 80 + "\n")
    
    # Phase 1: Data Preparation
    prepare_data()
    
    # Phase 2 & 3: Baseline Model
    baseline_sweep_id = run_baseline_sweep(count=20)
    best_baseline_config = register_best_baseline(baseline_sweep_id)
    
    # Get best baseline F1 score
    api = wandb.Api()
    entity = api.default_entity
    baseline_sweep = api.sweep(f"{entity}/{PROJECT}/{baseline_sweep_id}")
    best_baseline_f1 = baseline_sweep.best_run().summary.get('val_f1')
    
    # Phase 4: Semi-Supervised Learning
    ssl_sweep_id = run_ssl_sweep(best_baseline_config, best_baseline_f1, count=50)
    
    # Phase 5: Production Monitoring
    monitor_production(num_requests=100)
    
    # Phase 6: Automated Retraining
    check_and_retrain()
    
    print("\n" + "=" * 80)
    print("COMPLETE MLOPS PIPELINE FINISHED!")
    print("=" * 80)


if __name__ == "__main__":
    # You can run individual phases or the full pipeline
    import argparse
    
    parser = argparse.ArgumentParser(description='TOX21 MLOps Pipeline')
    parser.add_argument('--phase', type=str, choices=['all', '1', '2', '3', '4', '5', '6'],
                        default='all', help='Which phase to run')
    
    args = parser.parse_args()
    
    if args.phase == 'all':
        run_full_pipeline()
    elif args.phase == '1':
        prepare_data()
    elif args.phase in ['2', '3']:
        baseline_sweep_id = run_baseline_sweep(count=20)
        register_best_baseline(baseline_sweep_id)
    elif args.phase == '5':
        monitor_production(num_requests=100)
    elif args.phase == '6':
        check_and_retrain()
    else:
        print(f"Phase {args.phase} handler not fully implemented in standalone mode")
