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
    get_drug_model_info,
    get_ct_tox_model_info,
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


class DrugPrediction(BaseModel):
    """Drug discovery (FDA approval) prediction result"""
    prediction: int = Field(..., description="0 = Not Approved, 1 = Approved")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Probability distribution (includes individual model predictions)")


class ToxicityPrediction(BaseModel):
    """Clinical toxicity prediction result"""
    prediction: int = Field(..., description="0 = Non-Toxic, 1 = Toxic")
    prediction_label: str = Field(..., description="Human-readable prediction")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Probability distribution")


class DrugPredictionResponse(BaseModel):
    """Complete drug prediction response"""
    smiles: str
    canonical_smiles: str
    molecular_properties: Optional[MolecularProperties]
    prediction: DrugPrediction


class ToxicityPredictionResponse(BaseModel):
    """Complete toxicity prediction response"""
    smiles: str
    canonical_smiles: str
    molecular_properties: Optional[MolecularProperties]
    prediction: ToxicityPrediction


class CompletePrediction(BaseModel):
    """Complete prediction with both FDA and toxicity"""
    smiles: str
    canonical_smiles: str
    molecular_properties: Optional[MolecularProperties]
    drug_approval: DrugPrediction
    toxicity: ToxicityPrediction
    recommendation: str = Field(..., description="Overall assessment")
    risk_level: str = Field(..., description="Risk classification")


class BatchDrugResponse(BaseModel):
    """Batch drug prediction response"""
    total: int
    successful: int
    failed: int
    predictions: List[DrugPredictionResponse]
    errors: List[dict]


class BatchToxicityResponse(BaseModel):
    """Batch toxicity prediction response"""
    total: int
    successful: int
    failed: int
    predictions: List[ToxicityPredictionResponse]
    errors: List[dict]


class ModelsInfo(BaseModel):
    """Information about all three models"""
    drug_discovery_model: ModelInfo
    ct_tox_model: ModelInfo
    toxicity_model: ModelInfo
    rdkit_available: bool
    features_count: int


# ============================================================================
# Helper Functions
# ============================================================================

def determine_recommendation(drug_pred: int, tox_pred: int, drug_conf: float, tox_conf: float) -> tuple[str, str]:
    """
    Determine overall recommendation and risk level
    
    Returns:
        (recommendation, risk_level)
    """
    if drug_pred == 1 and tox_pred == 0:
        # Best case: Approved and non-toxic
        if drug_conf > 0.8 and tox_conf > 0.8:
            return ("✅ HIGHLY RECOMMENDED - High approval probability, low toxicity risk", "LOW")
        else:
            return ("✅ RECOMMENDED - Likely approved with acceptable safety profile", "LOW-MODERATE")
            
    elif drug_pred == 1 and tox_pred == 1:
        # Approved but toxic
        if tox_conf > 0.7:
            return ("⚠️ CAUTION - May be approved but has toxicity concerns", "MODERATE-HIGH")
        else:
            return ("⚠️ PROCEED WITH CAUTION - Approval likely but safety unclear", "MODERATE")
            
    elif drug_pred == 0 and tox_pred == 0:
        # Not approved but non-toxic
        return ("❌ NOT RECOMMENDED - Low approval probability despite low toxicity", "MODERATE")
        
    else:
        # Not approved and toxic
        if drug_conf > 0.7 and tox_conf > 0.7:
            return ("🛑 STRONGLY NOT RECOMMENDED - Low approval probability with toxicity risk", "HIGH")
        else:
            return ("❌ NOT RECOMMENDED - Unlikely to be approved", "MODERATE-HIGH")


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/models-info", response_model=ModelsInfo)
async def models_info():
    """Get information about all three models (drug_discovery, ct_tox, tox21)"""
    try:
        models = get_models()
        return ModelsInfo(
            drug_discovery_model=get_drug_model_info(),
            ct_tox_model=get_ct_tox_model_info(),
            toxicity_model=get_toxicity_model_info(),
            rdkit_available=True,
            features_count=35
        )
    except Exception as e:
        logger.exception("Failed to get models info")
        raise HTTPException(status_code=503, detail=f"Models not available: {str(e)}")


@router.post("/drug-predict", response_model=DrugPredictionResponse)
async def predict_drug(input_data: SMILESInput):
    """
    Predict FDA approval for a single molecule using BOTH drug_discovery and ct_tox models
    
    - **smiles**: SMILES string representation of the molecule
    
    Returns combined prediction from:
    - Drug Discovery model (FDA approval)
    - CT_TOX model (clinical toxicity, inverted for approval signal)
    """
    try:
        models = get_models()
        smiles = input_data.smiles
        
        # Get molecular properties
        mol_props = models.get_molecular_properties(smiles)
        if not mol_props:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # Predict FDA approval (uses both models)
        drug_result = models.predict_drug_approval(smiles)
        if not drug_result:
            raise HTTPException(status_code=400, detail="Drug approval prediction failed (requires both drug_discovery and ct_tox models)")
        drug_pred, drug_conf, drug_probs = drug_result
        
        # Create prediction
        drug_prediction = DrugPrediction(
            prediction=drug_pred,
            prediction_label="✅ FDA APPROVED" if drug_pred == 1 else "❌ NOT APPROVED",
            confidence=drug_conf,
            probabilities=drug_probs  # Includes individual model predictions
        )
        
        return DrugPredictionResponse(
            smiles=smiles,
            canonical_smiles=mol_props.canonical_smiles,
            molecular_properties=mol_props,
            prediction=drug_prediction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Drug prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/toxicity-predict", response_model=ToxicityPredictionResponse)
async def predict_toxicity(input_data: SMILESInput):
    """
    Predict clinical toxicity for a single molecule using TOX21 model
    
    - **smiles**: SMILES string representation of the molecule
    
    Returns prediction for:
    - TOX21 toxicity risk
    """
    try:
        models = get_models()
        smiles = input_data.smiles
        
        # Get molecular properties
        mol_props = models.get_molecular_properties(smiles)
        if not mol_props:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # Predict toxicity using TOX21 model
        tox_result = models.predict_toxicity(smiles)
        if not tox_result:
            raise HTTPException(status_code=400, detail="Toxicity prediction failed (TOX21 model not available)")
        tox_pred, tox_conf, tox_probs = tox_result
        
        # Create prediction
        toxicity_prediction = ToxicityPrediction(
            prediction=tox_pred,
            prediction_label="🔴 TOXIC" if tox_pred == 1 else "🟢 NON-TOXIC",
            confidence=tox_conf,
            probabilities=tox_probs
        )
        
        return ToxicityPredictionResponse(
            smiles=smiles,
            canonical_smiles=mol_props.canonical_smiles,
            molecular_properties=mol_props,
            prediction=toxicity_prediction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Toxicity prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict", response_model=CompletePrediction)
async def predict_complete(input_data: SMILESInput):
    """
    Predict both FDA approval and toxicity for a single molecule
    
    - **smiles**: SMILES string representation of the molecule
    
    Returns predictions for:
    - FDA approval probability (combined from drug_discovery + ct_tox)
    - TOX21 toxicity risk
    - Overall recommendation
    """
    try:
        models = get_models()
        smiles = input_data.smiles
        
        # Get molecular properties
        mol_props = models.get_molecular_properties(smiles)
        if not mol_props:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # Predict FDA approval (uses both drug_discovery and ct_tox)
        drug_result = models.predict_drug_approval(smiles)
        if not drug_result:
            raise HTTPException(status_code=400, detail="Drug approval prediction failed")
        drug_pred, drug_conf, drug_probs = drug_result
        
        # Predict toxicity using TOX21
        tox_result = models.predict_toxicity(smiles)
        if not tox_result:
            raise HTTPException(status_code=400, detail="Toxicity prediction failed")
        tox_pred, tox_conf, tox_probs = tox_result
        
        # Create predictions
        drug_prediction = DrugPrediction(
            prediction=drug_pred,
            prediction_label="✅ FDA APPROVED" if drug_pred == 1 else "❌ NOT APPROVED",
            confidence=drug_conf,
            probabilities=drug_probs
        )
        
        toxicity_prediction = ToxicityPrediction(
            prediction=tox_pred,
            prediction_label="🔴 TOXIC" if tox_pred == 1 else "🟢 NON-TOXIC",
            confidence=tox_conf,
            probabilities=tox_probs
        )
        
        # Determine recommendation
        recommendation, risk_level = determine_recommendation(drug_pred, tox_pred, drug_conf, tox_conf)
        
        return CompletePrediction(
            smiles=smiles,
            canonical_smiles=mol_props.canonical_smiles,
            molecular_properties=mol_props,
            drug_approval=drug_prediction,
            toxicity=toxicity_prediction,
            recommendation=recommendation,
            risk_level=risk_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Complete prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/drug-predict-batch", response_model=BatchDrugResponse)
async def predict_drug_batch(batch: BatchSMILESInput):
    """
    Predict FDA approval for multiple molecules (uses both drug_discovery and ct_tox)
    
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
                
                # Predict FDA approval (uses both models)
                drug_result = models.predict_drug_approval(smiles)
                if not drug_result:
                    errors.append({"index": idx, "smiles": smiles, "error": "Drug prediction failed"})
                    continue
                drug_pred, drug_conf, drug_probs = drug_result
                
                # Create prediction
                drug_prediction = DrugPrediction(
                    prediction=drug_pred,
                    prediction_label="✅ FDA APPROVED" if drug_pred == 1 else "❌ NOT APPROVED",
                    confidence=drug_conf,
                    probabilities=drug_probs
                )
                
                predictions.append(DrugPredictionResponse(
                    smiles=smiles,
                    canonical_smiles=mol_props.canonical_smiles,
                    molecular_properties=mol_props,
                    prediction=drug_prediction
                ))
                successful += 1
                
            except Exception as e:
                errors.append({"index": idx, "smiles": mol_input.smiles, "error": str(e)})
                
        return BatchDrugResponse(
            total=len(batch.molecules),
            successful=successful,
            failed=len(errors),
            predictions=predictions,
            errors=errors
        )
        
    except Exception as e:
        logger.exception("Batch drug prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@router.post("/toxicity-predict-batch", response_model=BatchToxicityResponse)
async def predict_toxicity_batch(batch: BatchSMILESInput):
    """
    Predict toxicity for multiple molecules (uses TOX21 model)
    
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
                
                # Predict toxicity using TOX21
                tox_result = models.predict_toxicity(smiles)
                if not tox_result:
                    errors.append({"index": idx, "smiles": smiles, "error": "Toxicity prediction failed"})
                    continue
                tox_pred, tox_conf, tox_probs = tox_result
                
                # Create prediction
                toxicity_prediction = ToxicityPrediction(
                    prediction=tox_pred,
                    prediction_label="🔴 TOXIC" if tox_pred == 1 else "🟢 NON-TOXIC",
                    confidence=tox_conf,
                    probabilities=tox_probs
                )
                
                predictions.append(ToxicityPredictionResponse(
                    smiles=smiles,
                    canonical_smiles=mol_props.canonical_smiles,
                    molecular_properties=mol_props,
                    prediction=toxicity_prediction
                ))
                successful += 1
                
            except Exception as e:
                errors.append({"index": idx, "smiles": mol_input.smiles, "error": str(e)})
                
        return BatchToxicityResponse(
            total=len(batch.molecules),
            successful=successful,
            failed=len(errors),
            predictions=predictions,
            errors=errors
        )
        
    except Exception as e:
        logger.exception("Batch toxicity prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@router.get("/drug-predict-smiles")
async def predict_drug_from_query(
    smiles: str = Query(..., description="SMILES string of the molecule")
) -> DrugPredictionResponse:
    """
    Predict FDA approval from query parameter (convenient for GET requests)
    
    Example: /qsar/drug-predict-smiles?smiles=CC(=O)Oc1ccccc1C(=O)O
    """
    return await predict_drug(SMILESInput(smiles=smiles))


@router.get("/toxicity-predict-smiles")
async def predict_toxicity_from_query(
    smiles: str = Query(..., description="SMILES string of the molecule")
) -> ToxicityPredictionResponse:
    """
    Predict toxicity from query parameter (convenient for GET requests)
    
    Example: /qsar/toxicity-predict-smiles?smiles=CC(=O)Oc1ccccc1C(=O)O
    """
    return await predict_toxicity(SMILESInput(smiles=smiles))