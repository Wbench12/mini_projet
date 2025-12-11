# app/api/routes/qsar.py
"""
QSAR Prediction Endpoints
Provides FDA approval and toxicity predictions from SMILES strings
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

from app.services.model import (
    get_models,
    get_fda_model_info,
    get_toxicity_model_info,
    ModelInfo,
    MolecularProperties
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class SMILESInput(BaseModel):
    """Single SMILES string input"""
    smiles: str = Field(..., description="SMILES representation of the molecule", min_length=1)
    
    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        return v.strip()


class BatchSMILESInput(BaseModel):
    """Batch SMILES input"""
    molecules: List[SMILESInput] = Field(..., description="List of molecules to predict", min_items=1, max_items=100)


class FDAPrediction(BaseModel):
    """FDA approval prediction result"""
    prediction: int = Field(..., description="0 = Not Approved, 1 = Approved")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Probability distribution")


class ToxicityPrediction(BaseModel):
    """Clinical toxicity prediction result"""
    prediction: int = Field(..., description="0 = Non-Toxic, 1 = Toxic")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Probability distribution")


class CompletePrediction(BaseModel):
    """Complete prediction with both FDA and toxicity"""
    smiles: str
    canonical_smiles: str
    molecular_properties: Optional[MolecularProperties]
    fda_approval: FDAPrediction
    toxicity: ToxicityPrediction
    recommendation: str = Field(..., description="Overall assessment")
    risk_level: str = Field(..., description="Risk classification")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total: int
    successful: int
    failed: int
    predictions: List[CompletePrediction]
    errors: List[dict]


class ModelsInfo(BaseModel):
    """Information about both models"""
    fda_model: ModelInfo
    toxicity_model: ModelInfo
    rdkit_available: bool
    features_count: int


# ============================================================================
# Helper Functions
# ============================================================================

def determine_recommendation(fda_pred: int, tox_pred: int, fda_conf: float, tox_conf: float) -> tuple[str, str]:
    """
    Determine overall recommendation and risk level
    
    Returns:
        (recommendation, risk_level)
    """
    if fda_pred == 1 and tox_pred == 0:
        # Best case: Approved and non-toxic
        if fda_conf > 0.8 and tox_conf > 0.8:
            return ("✅ HIGHLY RECOMMENDED - High approval probability, low toxicity risk", "LOW")
        else:
            return ("✅ RECOMMENDED - Likely approved with acceptable safety profile", "LOW-MODERATE")
            
    elif fda_pred == 1 and tox_pred == 1:
        # Approved but toxic
        if tox_conf > 0.7:
            return ("⚠️ CAUTION - May be approved but has toxicity concerns", "MODERATE-HIGH")
        else:
            return ("⚠️ PROCEED WITH CAUTION - Approval likely but safety unclear", "MODERATE")
            
    elif fda_pred == 0 and tox_pred == 0:
        # Not approved but non-toxic
        return ("❌ NOT RECOMMENDED - Low approval probability despite low toxicity", "MODERATE")
        
    else:
        # Not approved and toxic
        if fda_conf > 0.7 and tox_conf > 0.7:
            return ("🛑 STRONGLY NOT RECOMMENDED - Low approval probability with toxicity risk", "HIGH")
        else:
            return ("❌ NOT RECOMMENDED - Unlikely to be approved", "MODERATE-HIGH")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/models-info", response_model=ModelsInfo)
async def models_info():
    """Get information about both FDA approval and toxicity models"""
    try:
        models = get_models()
        return ModelsInfo(
            fda_model=get_fda_model_info(),
            toxicity_model=get_toxicity_model_info(),
            rdkit_available=True,  # Will fail at import if not available
            features_count=37
        )
    except Exception as e:
        logger.exception("Failed to get models info")
        raise HTTPException(status_code=503, detail=f"Models not available: {str(e)}")


@router.post("/predict", response_model=CompletePrediction)
async def predict_molecule(input_data: SMILESInput):
    """
    Predict FDA approval and toxicity for a single molecule
    
    - **smiles**: SMILES string representation of the molecule
    
    Returns predictions for:
    - FDA approval probability
    - Clinical toxicity risk
    - Overall recommendation
    """
    try:
        models = get_models()
        smiles = input_data.smiles
        
        # Get molecular properties
        mol_props = models.get_molecular_properties(smiles)
        if not mol_props:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # Predict FDA approval
        fda_result = models.predict_fda_approval(smiles)
        if not fda_result:
            raise HTTPException(status_code=400, detail="FDA prediction failed")
        fda_pred, fda_conf, fda_probs = fda_result
        
        # Predict toxicity
        tox_result = models.predict_toxicity(smiles)
        if not tox_result:
            raise HTTPException(status_code=400, detail="Toxicity prediction failed")
        tox_pred, tox_conf, tox_probs = tox_result
        
        # Create predictions
        fda_prediction = FDAPrediction(
            prediction=fda_pred,
            prediction_label="✅ FDA APPROVED" if fda_pred == 1 else "❌ NOT APPROVED",
            confidence=fda_conf,
            probabilities=fda_probs
        )
        
        toxicity_prediction = ToxicityPrediction(
            prediction=tox_pred,
            prediction_label="🔴 TOXIC" if tox_pred == 1 else "🟢 NON-TOXIC",
            confidence=tox_conf,
            probabilities=tox_probs
        )
        
        # Determine recommendation
        recommendation, risk_level = determine_recommendation(fda_pred, tox_pred, fda_conf, tox_conf)
        
        return CompletePrediction(
            smiles=smiles,
            canonical_smiles=mol_props.canonical_smiles,
            molecular_properties=mol_props,
            fda_approval=fda_prediction,
            toxicity=toxicity_prediction,
            recommendation=recommendation,
            risk_level=risk_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchSMILESInput):
    """
    Predict FDA approval and toxicity for multiple molecules
    
    Maximum 100 molecules per batch
    """
    predictions = []
    errors = []
    successful = 0
    
    try:
        models = get_models()
        
        for idx, mol_input in enumerate(batch.molecules):
            try:
                smiles = mol_input.smiles
                
                # Get molecular properties
                mol_props = models.get_molecular_properties(smiles)
                if not mol_props:
                    errors.append({"index": idx, "smiles": smiles, "error": "Invalid SMILES"})
                    continue
                
                # Predict FDA approval
                fda_result = models.predict_fda_approval(smiles)
                if not fda_result:
                    errors.append({"index": idx, "smiles": smiles, "error": "FDA prediction failed"})
                    continue
                fda_pred, fda_conf, fda_probs = fda_result
                
                # Predict toxicity
                tox_result = models.predict_toxicity(smiles)
                if not tox_result:
                    errors.append({"index": idx, "smiles": smiles, "error": "Toxicity prediction failed"})
                    continue
                tox_pred, tox_conf, tox_probs = tox_result
                
                # Create predictions
                fda_prediction = FDAPrediction(
                    prediction=fda_pred,
                    prediction_label="✅ FDA APPROVED" if fda_pred == 1 else "❌ NOT APPROVED",
                    confidence=fda_conf,
                    probabilities=fda_probs
                )
                
                toxicity_prediction = ToxicityPrediction(
                    prediction=tox_pred,
                    prediction_label="🔴 TOXIC" if tox_pred == 1 else "🟢 NON-TOXIC",
                    confidence=tox_conf,
                    probabilities=tox_probs
                )
                
                recommendation, risk_level = determine_recommendation(fda_pred, tox_pred, fda_conf, tox_conf)
                
                predictions.append(CompletePrediction(
                    smiles=smiles,
                    canonical_smiles=mol_props.canonical_smiles,
                    molecular_properties=mol_props,
                    fda_approval=fda_prediction,
                    toxicity=toxicity_prediction,
                    recommendation=recommendation,
                    risk_level=risk_level
                ))
                successful += 1
                
            except Exception as e:
                errors.append({"index": idx, "smiles": mol_input.smiles, "error": str(e)})
                
        return BatchPredictionResponse(
            total=len(batch.molecules),
            successful=successful,
            failed=len(errors),
            predictions=predictions,
            errors=errors
        )
        
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@router.get("/predict-smiles")
async def predict_from_query(
    smiles: str = Query(..., description="SMILES string of the molecule")
) -> CompletePrediction:
    """
    Predict from query parameter (convenient for GET requests)
    
    Example: /qsar/predict-smiles?smiles=CC(=O)Oc1ccccc1C(=O)O
    """
    return await predict_molecule(SMILESInput(smiles=smiles))
