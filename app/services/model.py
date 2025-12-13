import os
import logging
import joblib
import numpy as np
import sys
import __main__ 
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, MolSurf, QED, Fragments, Crippen
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CUSTOM MODEL CLASS (MUST MATCH PIPELINE)
# ==============================================================================

class CoTrainingEnsemble:
    def __init__(self, model1, model2, v1_idx, v2_idx):
        self.model1 = model1
        self.model2 = model2
        self.v1_idx = v1_idx
        self.v2_idx = v2_idx
        self.classes_ = model1.classes_
        self.n_features_in_ = len(v1_idx) + len(v2_idx)

    def predict(self, X):
        avg_prob = self.predict_proba(X)[:, 1]
        return (avg_prob >= 0.5).astype(int)
        
    def predict_proba(self, X):
        proba1 = self.model1.predict_proba(X[:, self.v1_idx])
        proba2 = self.model2.predict_proba(X[:, self.v2_idx])
        return (proba1 + proba2) / 2

# Patch namespace for unpickling
setattr(__main__, "CoTrainingEnsemble", CoTrainingEnsemble)

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

# Model paths (matching pipeline naming)
DRUG_DISCOVERY_PATH = os.path.join(MODEL_DIR, "drug_discovery_model.pkl")
CT_TOX_PATH = os.path.join(MODEL_DIR, "ct_tox_model.pkl")
TOX21_PATH = os.path.join(MODEL_DIR, "tox21_model.pkl")

# Feature columns (must match pipeline)
FEATURE_COLS = [
    'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 
    'NumAromaticRings', 'NumHeteroatoms', 'TPSA', 'NumRings', 'NumAliphaticRings', 
    'NumSaturatedRings', 'FractionCsp3', 'NumValenceElectrons', 'MaxPartialCharge', 
    'MinPartialCharge', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 'QED', 'BertzCT', 
    'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'MolMR', 'BalabanJ', 'HallKierAlpha', 
    'NumSaturatedCarbocycles', 'NumAromaticCarbocycles', 'NumSaturatedHeterocycles', 
    'NumAromaticHeterocycles', 'fr_NH2', 'fr_COO', 'fr_benzene', 'fr_furan', 'fr_halogen'
]

@dataclass
class ModelInfo:
    name: str
    type: str
    path: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    dataset: Optional[str] = None
    target: Optional[str] = None

@dataclass
class MolecularProperties:
    canonical_smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    h_donors: int
    h_acceptors: int
    rotatable_bonds: int
    qed: float

# ==============================================================================
# 3. FEATURE EXTRACTION
# ==============================================================================

def calculate_descriptors(mol):
    """Calculate molecular descriptors matching pipeline feature order"""
    return [
        Descriptors.MolWt(mol), 
        Crippen.MolLogP(mol), 
        Descriptors.NumHDonors(mol), 
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), 
        Descriptors.NumAromaticRings(mol), 
        Descriptors.NumHeteroatoms(mol),
        Descriptors.TPSA(mol), 
        Descriptors.RingCount(mol), 
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumSaturatedRings(mol), 
        Descriptors.FractionCSP3(mol), 
        Descriptors.NumValenceElectrons(mol),
        Descriptors.MaxPartialCharge(mol), 
        Descriptors.MinPartialCharge(mol), 
        MolSurf.LabuteASA(mol),
        MolSurf.PEOE_VSA1(mol), 
        MolSurf.PEOE_VSA2(mol), 
        QED.qed(mol), 
        GraphDescriptors.BertzCT(mol),
        GraphDescriptors.Chi0v(mol), 
        GraphDescriptors.Chi1v(mol), 
        GraphDescriptors.Kappa1(mol),
        GraphDescriptors.Kappa2(mol), 
        Crippen.MolMR(mol), 
        GraphDescriptors.BalabanJ(mol),
        GraphDescriptors.HallKierAlpha(mol), 
        Descriptors.NumSaturatedCarbocycles(mol),
        Descriptors.NumAromaticCarbocycles(mol), 
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.NumAromaticHeterocycles(mol), 
        Fragments.fr_NH2(mol), 
        Fragments.fr_COO(mol),
        Fragments.fr_benzene(mol), 
        Fragments.fr_furan(mol), 
        Fragments.fr_halogen(mol)
    ]

# ==============================================================================
# 4. MODEL SERVICE
# ==============================================================================

class QSARModelService:
    def __init__(self):
        # Drug prediction models (both used together)
        self.drug_discovery_model = None  # FDA approval model
        self.ct_tox_model = None           # Clinical toxicity model (used for drug prediction)
        self.drug_discovery_scaler = None
        self.ct_tox_scaler = None
        self.drug_discovery_metrics = {}
        self.ct_tox_metrics = {}
        
        # Toxicity prediction model
        self.tox21_model = None             # TOX21 toxicity model
        self.tox21_scaler = None
        self.tox21_metrics = {}
        
        self.load_models()

    def _load_artifact(self, path, name):
        """Load model artifact with scaler and metrics"""
        if not os.path.exists(path):
            logger.error(f"❌ {name} file NOT FOUND at {path}")
            return None, None, {}
            
        try:
            artifact = joblib.load(path)
            
            # New format: Dictionary with model, scaler, metrics
            if isinstance(artifact, dict):
                model = artifact.get("model")
                scaler = artifact.get("scaler")
                metrics = artifact.get("metrics", {})
                dataset = artifact.get("dataset", "")
                target = artifact.get("target", "")
                
                if model is None:
                    logger.error(f"❌ {name} artifact missing 'model' key")
                    return None, None, {}
                
                logger.info(f"✅ {name} loaded successfully (dataset: {dataset}, target: {target})")
                if scaler is not None:
                    logger.info(f"   ✓ Scaler loaded")
                if metrics:
                    logger.info(f"   ✓ Metrics available: {list(metrics.keys())}")
                
                return model, scaler, metrics
                
            # Legacy format: Raw model (no scaler/metrics)
            else:
                logger.warning(f"⚠️ {name} loaded (Legacy format, no scaler/metrics)")
                return artifact, None, {}
                
        except Exception as e:
            logger.error(f"❌ Critical error loading {name}: {e}")
            if "CoTrainingEnsemble" in str(e):
                logger.error("HINT: Namespace mismatch. Ensure CoTrainingEnsemble is defined.")
            return None, None, {}

    def load_models(self):
        """Load all three models"""
        # Load drug discovery model (FDA approval)
        self.drug_discovery_model, self.drug_discovery_scaler, self.drug_discovery_metrics = self._load_artifact(
            DRUG_DISCOVERY_PATH, "Drug Discovery Model"
        )
        
        # Load ct_tox model (used for drug prediction)
        self.ct_tox_model, self.ct_tox_scaler, self.ct_tox_metrics = self._load_artifact(
            CT_TOX_PATH, "CT_TOX Model"
        )
        
        # Load tox21 model (used for toxicity prediction)
        self.tox21_model, self.tox21_scaler, self.tox21_metrics = self._load_artifact(
            TOX21_PATH, "TOX21 Model"
        )

    def _prepare_features(self, smiles: str, scaler=None):
        """
        Prepare features from SMILES string.
        Returns: (mol, features_array)
        Features are scaled if scaler is provided.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None, None
            
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(canonical_smiles)
            
            # Calculate descriptors
            features = np.array(calculate_descriptors(mol)).reshape(1, -1)
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)
            
            # Scale if scaler provided
            if scaler is not None:
                features = scaler.transform(features)
            
            return mol, features
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None

    def get_molecular_properties(self, smiles: str) -> Optional[MolecularProperties]:
        """Get basic molecular properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            return MolecularProperties(
                canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
                molecular_weight=float(Descriptors.MolWt(mol)),
                logp=float(Crippen.MolLogP(mol)),
                tpsa=float(Descriptors.TPSA(mol)),
                h_donors=int(Descriptors.NumHDonors(mol)),
                h_acceptors=int(Descriptors.NumHAcceptors(mol)),
                rotatable_bonds=int(Descriptors.NumRotatableBonds(mol)),
                qed=float(QED.qed(mol))
            )
        except Exception as e:
            logger.error(f"Error getting molecular properties: {e}")
            return None

    def predict_drug_approval(self, smiles: str):
        """
        Predict FDA approval using BOTH drug_discovery and ct_tox models.
        Combines predictions from both models.
        
        Returns: (prediction, confidence, probabilities_dict) or None
        """
        # Check if both models are available
        if not self.drug_discovery_model or not self.ct_tox_model:
            logger.warning("Drug prediction requires both drug_discovery and ct_tox models")
            return None
        
        # Get predictions from both models
        drug_result = self._predict_generic(
            smiles,
            self.drug_discovery_model,
            self.drug_discovery_scaler,
            "Drug Discovery"
        )
        
        ct_tox_result = self._predict_generic(
            smiles,
            self.ct_tox_model,
            self.ct_tox_scaler,
            "CT_TOX"
        )
        
        if not drug_result or not ct_tox_result:
            return None
        
        drug_pred, drug_conf, drug_probs = drug_result
        ct_tox_pred, ct_tox_conf, ct_tox_probs = ct_tox_result
        
        # Combine predictions: weighted average
        # drug_discovery gets 60% weight, ct_tox gets 40% weight
        drug_prob_1 = drug_probs.get("1", 0.0)
        ct_tox_prob_1 = ct_tox_probs.get("1", 0.0)
        
        # Invert ct_tox (non-toxic = good for approval)
        ct_tox_approval_prob = 1.0 - ct_tox_prob_1
        
        # Weighted combination
        combined_prob_1 = 0.6 * drug_prob_1 + 0.4 * ct_tox_approval_prob
        combined_prob_0 = 1.0 - combined_prob_1
        
        # Final prediction
        prediction = 1 if combined_prob_1 >= 0.5 else 0
        confidence = combined_prob_1 if prediction == 1 else combined_prob_0
        
        probabilities = {
            "0": float(combined_prob_0),
            "1": float(combined_prob_1),
            "drug_discovery": {
                "prediction": int(drug_pred),
                "confidence": float(drug_conf),
                "probabilities": drug_probs
            },
            "ct_tox": {
                "prediction": int(ct_tox_pred),
                "confidence": float(ct_tox_conf),
                "probabilities": ct_tox_probs
            }
        }
        
        return int(prediction), confidence, probabilities

    def predict_toxicity(self, smiles: str):
        """
        Predict toxicity using TOX21 model.
        
        Returns: (prediction, confidence, probabilities_dict) or None
        """
        return self._predict_generic(
            smiles, 
            self.tox21_model, 
            self.tox21_scaler,
            "TOX21 Toxicity"
        )

    def _predict_generic(self, smiles, model, scaler, name):
        """
        Generic prediction function that handles scaling
        
        Returns: (prediction, confidence, probabilities_dict) or None
        """
        if not model:
            logger.warning(f"{name} Model not loaded")
            return None
        
        _, features = self._prepare_features(smiles, scaler)
        if features is None:
            return None

        try:
            probs = model.predict_proba(features)[0]
            prob_0, prob_1 = float(probs[0]), float(probs[1])
            prediction = 1 if prob_1 >= 0.5 else 0
            confidence = prob_1 if prediction == 1 else prob_0
            
            return int(prediction), confidence, {"0": prob_0, "1": prob_1}
        except Exception as e:
            logger.error(f"{name} prediction error: {e}")
            return None

# Singleton Instance
_model_service = QSARModelService()

def get_models():
    return _model_service

def get_drug_model_info():
    """Get drug discovery model info"""
    status = "Loaded" if _model_service.drug_discovery_model else "Not Loaded"
    return ModelInfo(
        name="Drug Discovery (FDA Approval) Predictor",
        type=_model_service.drug_discovery_metrics.get("type", "Unknown"),
        path=DRUG_DISCOVERY_PATH,
        status=status,
        metrics=_model_service.drug_discovery_metrics,
        dataset="drug_discovery",
        target="FDA_APPROVED"
    )

def get_ct_tox_model_info():
    """Get CT_TOX model info"""
    status = "Loaded" if _model_service.ct_tox_model else "Not Loaded"
    return ModelInfo(
        name="Clinical Toxicity (CT_TOX) Predictor",
        type=_model_service.ct_tox_metrics.get("type", "Unknown"),
        path=CT_TOX_PATH,
        status=status,
        metrics=_model_service.ct_tox_metrics,
        dataset="ct_tox",
        target="CT_TOX"
    )

def get_toxicity_model_info():
    """Get TOX21 toxicity model info"""
    status = "Loaded" if _model_service.tox21_model else "Not Loaded"
    return ModelInfo(
        name="TOX21 Toxicity Predictor",
        type=_model_service.tox21_metrics.get("type", "Unknown"),
        path=TOX21_PATH,
        status=status,
        metrics=_model_service.tox21_metrics,
        dataset="tox21",
        target="toxic"
    )