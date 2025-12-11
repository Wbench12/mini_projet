"""QSAR Semi-Supervised Production Pipeline

This script runs the full Co-Training (Multi-View) algorithm matching the 
notebook's logic exactly.

Steps:
1. Load Labeled and Unlabeled Data.
2. Split Labeled Data (80% Train / 20% Test) for validation.
3. Split Features into View 1 (Physics) and View 2 (Structure).
4. Run Co-Training Loop (Pseudo-labeling) on the Train split.
5. Validate on the Test split.
6. Save the final Model + Metrics to app/models/.
"""
import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "app", "models")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

FDA_DIR = os.path.join(BASE_DIR, "data", "processed", "drug_discovery")
TOX_DIR = os.path.join(BASE_DIR, "data", "processed", "ct_tox")

# Hyperparameters (Matched to Notebook)
MAX_ITERATIONS = 20
SAMPLES_PER_ITER = 50       
CONFIDENCE_THRESHOLD = 0.80 

# Best Unlabeled Sizes (From your results)
FDA_UNLABELED_SIZE = 500    
TOX_UNLABELED_SIZE = 5000   

# ------------------------------------------------------------------------------
# Feature Definitions (Matched exactly)
# ------------------------------------------------------------------------------
FEATURE_COLS = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 
    'NumAromaticRings', 'NumHeteroatoms', 'TPSA', 'NumRings', 'NumAliphaticRings', 
    'NumSaturatedRings', 'FractionCsp3', 'NumValenceElectrons', 'MaxPartialCharge', 
    'MinPartialCharge', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'QED', 'BertzCT', 
    'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'MolMR', 'BalabanJ', 'HallKierAlpha', 
    'NumSaturatedCarbocycles', 'NumAromaticCarbocycles', 'NumSaturatedHeterocycles', 
    'NumAromaticHeterocycles', 'fr_NH2', 'fr_COO', 'fr_benzene', 'fr_furan', 'fr_halogen'
]

# View 1: Physicochemical
VIEW1_CANDIDATES = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumValenceElectrons', 
    'TPSA', 'MaxPartialCharge', 'MinPartialCharge', 'LabuteASA', 'MolMR', 
    'QED', 'NumHeteroatoms'
]

# View 2: Structural/Graph
VIEW2_CANDIDATES = [
    'NumRotatableBonds', 'NumAromaticRings', 'NumRings', 'NumAliphaticRings', 
    'NumSaturatedRings', 'FractionCsp3', 'PEOE_VSA1', 'PEOE_VSA2', 'BertzCT', 
    'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'BalabanJ', 'HallKierAlpha', 
    'NumSaturatedCarbocycles', 'NumAromaticCarbocycles', 
    'NumSaturatedHeterocycles', 'NumAromaticHeterocycles', 
    'fr_NH2', 'fr_COO', 'fr_benzene', 'fr_furan', 'fr_halogen'
]

# Map to indices
V1_IDX = [FEATURE_COLS.index(f) for f in VIEW1_CANDIDATES if f in FEATURE_COLS]
V2_IDX = [FEATURE_COLS.index(f) for f in VIEW2_CANDIDATES if f in FEATURE_COLS]

# ==============================================================================
# 2. MODEL CLASS
# ==============================================================================

class CoTrainingEnsemble:
    """Ensemble model used for inference."""
    def __init__(self, model1, model2, v1_idx, v2_idx):
        self.model1 = model1
        self.model2 = model2
        self.v1_idx = v1_idx
        self.v2_idx = v2_idx
        self.classes_ = model1.classes_
        self.n_features_in_ = len(FEATURE_COLS)

    def predict(self, X):
        avg_prob = self.predict_proba(X)[:, 1]
        return (avg_prob >= 0.5).astype(int)
        
    def predict_proba(self, X):
        proba1 = self.model1.predict_proba(X[:, self.v1_idx])
        proba2 = self.model2.predict_proba(X[:, self.v2_idx])
        return (proba1 + proba2) / 2

# ==============================================================================
# 3. DATA LOADING
# ==============================================================================

def load_datasets(directory, target_col, n_unlabeled):
    """Loads labeled and unlabeled data."""
    print(f"🔹 Loading data from: {directory}")
    
    # 1. Load Labeled Data
    labeled_path = os.path.join(directory, "labeled_processed.csv")
    if not os.path.exists(labeled_path):
        print(f"   ❌ No labeled CSV found at {labeled_path}")
        return None, None, None
    
    df_lab = pd.read_csv(labeled_path)
    
    if target_col not in df_lab.columns:
        print(f"   ❌ Target column '{target_col}' not found")
        return None, None, None
    
    df_lab = df_lab.dropna(subset=[target_col])
    X_labeled = df_lab[FEATURE_COLS].values
    y_labeled = df_lab[target_col].values
    print(f"   ✓ Loaded {len(X_labeled)} labeled samples")
    
    # 2. Load Unlabeled Data
    unlabeled_path = os.path.join(directory, "unlabeled_processed.csv")
    X_unlabeled = None
    if os.path.exists(unlabeled_path):
        print(f"   ✓ Found unlabeled file")
        df_unlab = pd.read_csv(unlabeled_path)
        if len(df_unlab) > n_unlabeled:
            df_unlab = df_unlab.sample(n_unlabeled, random_state=42)
        X_unlabeled = df_unlab[FEATURE_COLS].values
        print(f"   ✓ Using {len(X_unlabeled)} unlabeled samples")
    else:
        print("   ⚠️ No unlabeled data found.")

    return X_labeled, y_labeled, X_unlabeled

# ==============================================================================
# 4. TRAINING & EVALUATION LOGIC
# ==============================================================================

def train_and_evaluate(X_labeled, y_labeled, X_unlabeled, task_name):
    """
    Splits labeled data, runs Co-Training, evaluates on hold-out set.
    """
    print(f"\n🚀 Processing {task_name}...")

    # 1. Split for Validation (Critical for honest metrics)
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
    )
    
    # 2. Initialize Views & Models
    X_train_v1 = X_train[:, V1_IDX]
    X_train_v2 = X_train[:, V2_IDX]
    y_train_curr = y_train.copy()
    
    clf1 = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=42)
    
    # 3. Co-Training Loop
    if X_unlabeled is not None and len(X_unlabeled) > 0:
        X_pool_v1 = X_unlabeled[:, V1_IDX]
        X_pool_v2 = X_unlabeled[:, V2_IDX]
        mask_avail = np.ones(len(X_unlabeled), dtype=bool)
        
        print(f"   🔄 Starting Co-Training Loop ({MAX_ITERATIONS} iters)")
        
        for i in range(MAX_ITERATIONS):
            if not np.any(mask_avail): break
            
            # Train on current set
            clf1.fit(X_train_v1, y_train_curr)
            clf2.fit(X_train_v2, y_train_curr)
            
            pool_idx = np.where(mask_avail)[0]
            if len(pool_idx) == 0: break
            
            prob1 = clf1.predict_proba(X_pool_v1[pool_idx])
            prob2 = clf2.predict_proba(X_pool_v2[pool_idx])
            
            new_indices = []
            new_labels = []
            
            for probs in [prob1, prob2]:
                conf = np.max(probs, axis=1)
                pred = np.argmax(probs, axis=1)
                
                # Thresholding
                high_conf_mask = conf > CONFIDENCE_THRESHOLD
                candidates = np.where(high_conf_mask)[0]
                
                if len(candidates) > 0:
                    sorted_locs = candidates[np.argsort(conf[candidates])[::-1]]
                    selected = sorted_locs[:SAMPLES_PER_ITER]
                    original_indices = pool_idx[selected]
                    new_indices.extend(original_indices)
                    new_labels.extend(pred[selected])
            
            if not new_indices: break
                
            unique_indices, unique_pos = np.unique(new_indices, return_index=True)
            final_indices = unique_indices
            final_labels = np.array(new_labels)[unique_pos]
            
            # Add to training set
            X_train_v1 = np.vstack((X_train_v1, X_unlabeled[final_indices][:, V1_IDX]))
            X_train_v2 = np.vstack((X_train_v2, X_unlabeled[final_indices][:, V2_IDX]))
            y_train_curr = np.concatenate((y_train_curr, final_labels))
            mask_avail[final_indices] = False
            
            if (i+1) % 5 == 0 or i==0: 
                print(f"      Iter {i+1}: Train size {len(y_train_curr)}")
    else:
        print("   ℹ️ No unlabeled data. Running Supervised Learning.")

    # 4. Final Retrain on Augmented Data
    clf1.fit(X_train_v1, y_train_curr)
    clf2.fit(X_train_v2, y_train_curr)
    
    # 5. Create Ensemble
    model = CoTrainingEnsemble(clf1, clf2, V1_IDX, V2_IDX)
    
    # 6. Evaluation (On Hold-out Test Set)
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

# ==============================================================================
# 5. MAIN PIPELINE
# ==============================================================================

def main():
    print("="*60)
    print("🔬 QSAR SEMI-SUPERVISED PIPELINE (Co-Training + Metrics)")
    print("="*60)
    
    # --- TASK 1: FDA APPROVAL ---
    X_lab, y_lab, X_unlab = load_datasets(FDA_DIR, "FDA_APPROVED", FDA_UNLABELED_SIZE)
    if X_lab is not None:
        model, metrics = train_and_evaluate(X_lab, y_lab, X_unlab, "FDA Approval")
        
        # Save Artifact
        artifact = {"model": model, "metrics": metrics, "type": "Co-Training"}
        path = os.path.join(MODEL_OUTPUT_DIR, "fda_approval_model.pkl")
        joblib.dump(artifact, path)
        print(f"   💾 Saved FDA Artifact: {path}")

    # --- TASK 2: CLINICAL TOXICITY ---
    X_lab, y_lab, X_unlab = load_datasets(TOX_DIR, "CT_TOX", TOX_UNLABELED_SIZE)
    if X_lab is not None:
        model, metrics = train_and_evaluate(X_lab, y_lab, X_unlab, "Clinical Toxicity")
        
        # Save Artifact
        artifact = {"model": model, "metrics": metrics, "type": "Co-Training"}
        path = os.path.join(MODEL_OUTPUT_DIR, "ct_tox_model.pkl")
        joblib.dump(artifact, path)
        print(f"   💾 Saved Tox Artifact: {path}")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("   Models updated with notebook logic and metrics.")
    print("   Restart Uvicorn to apply changes.")
    print("="*60)

if __name__ == "__main__":
    main()
