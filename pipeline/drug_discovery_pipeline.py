"""Drug Discovery Pipeline - FDA Approval Prediction using Self-Training"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Fix imports - add project root to path if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline.utils import (
    FEATURE_COLS, process_enhanced_data, MAX_ITERATIONS, SAMPLES_PER_ITER, CONFIDENCE_THRESHOLD
)

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "app", "models")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
ENHANCED_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "enhanced_data")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Configuration
DATASET_NAME = 'drug_discovery'
TARGET = 'FDA_APPROVED'
METHOD = 'self_training'
THRESHOLD = 0.8
UNLABELED_SIZE = 5000

ENHANCED_DIR = os.path.join(ENHANCED_DATA_DIR, DATASET_NAME)
PROCESSED_DIR = os.path.join(PROCESSED_DATA_DIR, DATASET_NAME)


def train_self_training(X_train, y_train, X_unlabeled, X_val, y_val, threshold=0.75, target_size=5000):
    """
    Self-Training using sklearn's SelfTrainingClassifier with intelligent sample selection.
    Always reaches target_size by selecting best quality samples first, then filling with remaining.
    """
    print(f"   🔄 Starting Self-Training (threshold={threshold}, target={target_size})")
    
    # Initial baseline model
    base_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    base_model.fit(X_train, y_train)
    
    # Evaluate baseline
    y_pred_base = base_model.predict(X_val)
    y_proba_base = base_model.predict_proba(X_val)[:, 1]
    baseline_f1 = f1_score(y_val, y_pred_base)
    baseline_auc = roc_auc_score(y_val, y_proba_base)
    
    print(f"      Baseline: F1={baseline_f1:.4f}, AUC={baseline_auc:.4f}")
    
    # Intelligent selection: Filter unlabeled samples
    print(f"      Selecting {target_size} unlabeled samples from {len(X_unlabeled)} candidates...")
    
    unlabeled_proba = base_model.predict_proba(X_unlabeled)
    unlabeled_conf = np.max(unlabeled_proba, axis=1)
    unlabeled_pred = np.argmax(unlabeled_proba, axis=1)
    
    # Filter by confidence threshold
    high_conf_mask = unlabeled_conf > threshold
    high_conf_indices = np.where(high_conf_mask)[0]
    
    # Sort all by confidence (descending)
    all_sorted_indices = np.argsort(unlabeled_conf)[::-1]
    
    # Phase 1: Select beneficial samples (high quality)
    beneficial_indices = []
    if len(high_conf_indices) > 0:
        sorted_high_conf = high_conf_indices[np.argsort(unlabeled_conf[high_conf_indices])[::-1]]
        
        print(f"      Phase 1: Testing {min(len(sorted_high_conf), target_size * 2)} high-confidence samples...")
        
        for idx in sorted_high_conf[:min(len(sorted_high_conf), target_size * 2)]:
            if len(beneficial_indices) >= target_size:
                break
                
            X_temp = np.vstack([X_train, X_unlabeled[idx:idx+1]])
            y_temp = np.concatenate([y_train, [unlabeled_pred[idx]]])
            
            temp_model = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            temp_model.fit(X_temp, y_temp)
            
            y_pred_temp = temp_model.predict(X_val)
            temp_f1 = f1_score(y_val, y_pred_temp)
            
            if temp_f1 >= baseline_f1 * 0.95:
                beneficial_indices.append(idx)
        
        print(f"      ✓ Selected {len(beneficial_indices)} beneficial samples")
    
    # Phase 2: Fill to target_size with remaining high-confidence samples
    selected_indices = set(beneficial_indices)
    
    if len(selected_indices) < target_size:
        remaining_needed = target_size - len(selected_indices)
        print(f"      Phase 2: Filling {remaining_needed} more samples from high-confidence pool...")
        
        if len(high_conf_indices) > 0:
            sorted_high_conf = high_conf_indices[np.argsort(unlabeled_conf[high_conf_indices])[::-1]]
            for idx in sorted_high_conf:
                if len(selected_indices) >= target_size:
                    break
                if idx not in selected_indices:
                    selected_indices.add(idx)
        
        print(f"      ✓ Added {len(selected_indices) - len(beneficial_indices)} high-confidence samples")
    
    # Phase 3: If still not enough, use all remaining samples by confidence
    if len(selected_indices) < target_size:
        remaining_needed = target_size - len(selected_indices)
        print(f"      Phase 3: Filling {remaining_needed} more samples from all remaining pool...")
        
        for idx in all_sorted_indices:
            if len(selected_indices) >= target_size:
                break
            if idx not in selected_indices:
                selected_indices.add(idx)
        
        print(f"      ✓ Added {len(selected_indices) - (target_size - remaining_needed)} additional samples")
    
    # Convert to array and ensure we have exactly target_size (or all available if less)
    selected_indices = np.array(list(selected_indices))[:min(target_size, len(X_unlabeled))]
    
    if len(selected_indices) < target_size:
        print(f"      ⚠️ Only {len(selected_indices)} samples available (target was {target_size})")
    else:
        print(f"      ✓ Selected exactly {len(selected_indices)} samples")
    
    # Prepare data for SelfTrainingClassifier
    X_selected = X_unlabeled[selected_indices]
    X_combined = np.vstack([X_train, X_selected])
    y_combined = np.concatenate([y_train, np.full(len(X_selected), -1)])
    
    print(f"      Combined: {len(X_train)} labeled + {len(X_selected)} unlabeled = {len(X_combined)} total")
    
    # Use sklearn's SelfTrainingClassifier
    st_classifier = SelfTrainingClassifier(
        base_estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        threshold=threshold,
        max_iter=10,
        verbose=False
    )
    
    print(f"      Training SelfTrainingClassifier...")
    st_classifier.fit(X_combined, y_combined)
    
    # Check how many were pseudo-labeled
    n_pseudo_labeled = (st_classifier.transduction_ != -1).sum() - len(y_train)
    print(f"      📝 Pseudo-labeled {n_pseudo_labeled}/{len(X_selected)} samples ({n_pseudo_labeled/len(X_selected)*100:.1f}%)")
    
    # Evaluate on validation set
    y_pred_val = st_classifier.predict(X_val)
    y_proba_val = st_classifier.predict_proba(X_val)[:, 1]
    val_f1 = f1_score(y_val, y_pred_val)
    val_auc = roc_auc_score(y_val, y_proba_val)
    
    print(f"      Validation: F1={val_f1:.4f}, AUC={val_auc:.4f}")
    
    return st_classifier


def train_and_evaluate(X_labeled, y_labeled, X_unlabeled, scaler):
    """Train model and evaluate on test set."""
    print(f"\n🚀 Training Drug Discovery Model...")
    
    # Split for validation (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )
    
    # Further split train for validation during training
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model
    if X_unlabeled is None or len(X_unlabeled) == 0:
        print("   ℹ️ No unlabeled data. Running Supervised Learning.")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_fit, y_train_fit)
    else:
        # Use all available unlabeled data up to target size
        model = train_self_training(
            X_train_fit, y_train_fit, X_unlabeled, X_val, y_val, 
            THRESHOLD, UNLABELED_SIZE
        )
    
    # Evaluate on test set
    print("   📈 Evaluating on Test Set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    
    print(f"      F1: {metrics['f1_score']:.4f} | AUC: {metrics['roc_auc']:.4f}")
    return model, metrics


def run_pipeline():
    """Main pipeline execution for drug discovery."""
    print("="*60)
    print("🔬 DRUG DISCOVERY PIPELINE (FDA Approval)")
    print("="*60)
    
    # Step 1: Process enhanced data
    X_lab, y_lab, X_unlab, scaler = process_enhanced_data(
        ENHANCED_DIR, TARGET, DATASET_NAME, PROCESSED_DIR
    )
    
    if X_lab is None:
        print(f"   ❌ Failed to process data. Exiting.")
        return None
    
    # Step 2: Train and evaluate
    model, metrics = train_and_evaluate(X_lab, y_lab, X_unlab, scaler)
    
    # Step 3: Save model
    model_name = f"{DATASET_NAME}_model.pkl"
    artifact = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "type": METHOD.replace('_', '-').title(),
        "dataset": DATASET_NAME,
        "target": TARGET
    }
    path = os.path.join(MODEL_OUTPUT_DIR, model_name)
    joblib.dump(artifact, path)
    print(f"   💾 Saved model: {path}")
    
    # Also save scaler separately
    scaler_model_path = os.path.join(MODEL_OUTPUT_DIR, f"{DATASET_NAME}_scaler.pkl")
    joblib.dump(scaler, scaler_model_path)
    print(f"   💾 Saved scaler: {scaler_model_path}")
    
    print(f"\n✅ Drug Discovery Pipeline Complete")
    return model, metrics, scaler


if __name__ == "__main__":
    run_pipeline()
