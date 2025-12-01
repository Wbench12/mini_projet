import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.model import get_model, predict_batch, ModelInfo

logger = logging.getLogger(__name__)
router = APIRouter()


class QSARFeatures(BaseModel):
    """Input feature vector for a single molecule.

    Replace this schema with the actual descriptor set used by your QSAR pipeline.
    For now we keep a generic numeric vector.
    """

    features: List[float] = Field(..., description="List of numeric descriptors")


class QSARBatch(BaseModel):
    """Batch prediction request containing multiple molecules."""

    items: List[QSARFeatures]


class QSARPrediction(BaseModel):
    index: int
    predicted: float


class QSARBatchResponse(BaseModel):
    model: ModelInfo
    predictions: List[QSARPrediction]


@router.get("/model-info", response_model=ModelInfo)
async def model_info():
    """Return metadata about the loaded semi-supervised QSAR model."""
    model = get_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@router.post("/predict", response_model=QSARBatchResponse)
async def predict(batch: QSARBatch):
    """Run batch predictions.

    Currently uses a placeholder model. Integrate the real pipeline by
    updating `services/model.py` to load your trained semi-supervised QSAR model.
    """
    model = get_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_matrix = [item.features for item in batch.items]
    try:
        preds = predict_batch(feature_matrix)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))

    response = QSARBatchResponse(
        model=model,
        predictions=[QSARPrediction(index=i, predicted=p) for i, p in enumerate(preds)],
    )
    return response