from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..database import get_db
from .. import models, schemas

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"]
)

@router.get("/", response_model=List[schemas.PredictionResponse])
def list_predictions(
    subject_id: int = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List prediction history. Option to filter by subject_id."""
    query = db.query(models.Prediction)
    if subject_id:
        query = query.filter(models.Prediction.subject_id == subject_id)
        
    return query.order_by(models.Prediction.created_at.desc()).limit(limit).all()

@router.get("/{prediction_id}", response_model=schemas.PredictionResponse)
def get_prediction_detail(prediction_id: int, db: Session = Depends(get_db)):
    """Get single prediction details."""
    pred = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction record not found")
    return pred
