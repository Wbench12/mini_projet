import os
import logging
import joblib
import numpy as np
import sys
import __main__ 
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, MolSurf, QED, Fragments, Crippen
from dataclasses import dataclass, field
from typing import Dict

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

CT_TOX_PATH = os.path.join(MODEL_DIR, "ct_tox_model.pkl")
FDA_PATH = os.path.join(MODEL_DIR, "fda_approval_model.pkl")

@dataclass
class ModelInfo:
    name: str
    type: str
    path: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict) # Added metrics support

@dataclass
class MolecularProperties:
    canonical_smiles: str; molecular_weight: float; logp: float; tpsa: float
    h_donors: int; h_acceptors: int; rotatable_bonds: int; qed: float

# ==============================================================================
# 3. FEATURE EXTRACTION
# ==============================================================================

def calculate_descriptors(mol):
    return [
        Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol), Descriptors.NumHeteroatoms(mol),
        Descriptors.TPSA(mol), Descriptors.RingCount(mol), Descriptors.NumAliphaticRings(mol),
        Descriptors.NumSaturatedRings(mol), Descriptors.FractionCSP3(mol), Descriptors.NumValenceElectrons(mol),
        Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol), MolSurf.LabuteASA(mol),
        MolSurf.PEOE_VSA1(mol), MolSurf.PEOE_VSA2(mol), QED.qed(mol), GraphDescriptors.BertzCT(mol),
        GraphDescriptors.Chi0v(mol), GraphDescriptors.Chi1v(mol), GraphDescriptors.Kappa1(mol),
        GraphDescriptors.Kappa2(mol), Crippen.MolMR(mol), GraphDescriptors.BalabanJ(mol),
        GraphDescriptors.HallKierAlpha(mol), Descriptors.NumSaturatedCarbocycles(mol),
        Descriptors.NumAromaticCarbocycles(mol), Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.NumAromaticHeterocycles(mol), Fragments.fr_NH2(mol), Fragments.fr_COO(mol),
        Fragments.fr_benzene(mol), Fragments.fr_furan(mol), Fragments.fr_halogen(mol)
    ]

# ==============================================================================
# 4. MODEL SERVICE
# ==============================================================================

class QSARModelService:
    def __init__(self):
        self.fda_model = None
        self.tox_model = None
        self.fda_metrics = {}
        self.tox_metrics = {}
        self.load_models()

    def _load_artifact(self, path, name):
        """Helper to load dictionary artifact or raw model."""
        if not os.path.exists(path):
            logger.error(f"❌ {name} file NOT FOUND at {path}")
            return None, {}
            
        try:
            artifact = joblib.load(path)
            
            # Scenario A: New Format (Dictionary)
            if isinstance(artifact, dict) and "model" in artifact:
                logger.info(f"✅ {name} loaded successfully (with metrics).")
                return artifact["model"], artifact.get("metrics", {})
                
            # Scenario B: Old Format (Raw Model)
            else:
                logger.info(f"⚠️ {name} loaded (Legacy format, no metrics).")
                return artifact, {}
                
        except Exception as e:
            logger.error(f"❌ Critical error loading {name}: {e}")
            if "CoTrainingEnsemble" in str(e):
                logger.error("HINT: Namespace mismatch. Ensure CoTrainingEnsemble is defined.")
            return None, {}

    def load_models(self):
        """Loads both models using the helper."""
        self.fda_model, self.fda_metrics = self._load_artifact(FDA_PATH, "FDA Model")
        self.tox_model, self.tox_metrics = self._load_artifact(CT_TOX_PATH, "Toxicity Model")

    def _prepare_features(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return None, None
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(canonical_smiles)
            features = np.nan_to_num(np.array(calculate_descriptors(mol)).reshape(1, -1))
            return mol, features
        except: return None, None

    def get_molecular_properties(self, smiles: str) -> MolecularProperties:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return None
            return MolecularProperties(
                Chem.MolToSmiles(mol, canonical=True), float(Descriptors.MolWt(mol)), float(Crippen.MolLogP(mol)),
                float(Descriptors.TPSA(mol)), int(Descriptors.NumHDonors(mol)), int(Descriptors.NumHAcceptors(mol)),
                int(Descriptors.NumRotatableBonds(mol)), float(QED.qed(mol))
            )
        except: return None

    def predict_fda_approval(self, smiles: str):
        return self._predict_generic(smiles, self.fda_model, "FDA")

    def predict_toxicity(self, smiles: str):
        return self._predict_generic(smiles, self.tox_model, "Toxicity")

    def _predict_generic(self, smiles, model, name):
        if not model:
            logger.warning(f"{name} Model not loaded")
            return None
        _, feats = self._prepare_features(smiles)
        if feats is None: return None

        try:
            probs = model.predict_proba(feats)[0]
            prob_0, prob_1 = float(probs[0]), float(probs[1])
            prediction = 1 if prob_1 >= 0.5 else 0
            confidence = prob_1 if prediction == 1 else prob_0
            return int(prediction), confidence, {"0": prob_0, "1": prob_1}
        except Exception as e:
            logger.error(f"{name} prediction error: {e}")
            return None

# Singleton Instance
_model_service = QSARModelService()

def get_models(): return _model_service

def get_fda_model_info():
    status = "Loaded" if _model_service.fda_model else "Not Loaded"
    return ModelInfo("FDA Approval Predictor", "CoTraining Ensemble", FDA_PATH, status, _model_service.fda_metrics)

def get_toxicity_model_info():
    status = "Loaded" if _model_service.tox_model else "Not Loaded"
    return ModelInfo("Clinical Toxicity Predictor", "CoTraining Ensemble", CT_TOX_PATH, status, _model_service.tox_metrics)
