from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any
from ..database import get_db
from .. import models, schemas
from ..services.signal_processing import slice_raw_data_into_windows, process_single_window
from ..services.feature_extraction import extract_features_from_window, average_features_across_windows
from ..services.prediction import predict_glucose

router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"]
)

def run_pipeline_sync(subject_id: int, db: Session) -> Dict[str, Any]:
    """Execute the full processing and prediction pipeline synchronously."""
    # 1. Fetch raw readings
    readings = db.query(models.RawReading).filter(
        models.RawReading.subject_id == subject_id
    ).order_by(models.RawReading.timestamp.asc()).all()
    
    if not readings:
        raise HTTPException(
            status_code=400,
            detail="No raw PPG data found for this subject. Please upload a CSV or stream data first."
        )
        
    subject = db.query(models.Subject).filter(models.Subject.id == subject_id).first()
    
    # Create PipelineRun record in database
    run_record = models.PipelineRun(
        subject_id=subject_id,
        status="running",
        started_at=datetime.utcnow()
    )
    db.add(run_record)
    db.commit()
    db.refresh(run_record)
    
    pipeline_log = []
    pipeline_log.append(f"Started pipeline run {run_record.id} for subject {subject.name} (ID: {subject_id}) at {run_record.started_at}")
    pipeline_log.append(f"Retrieved {len(readings)} raw readings from database.")
    
    try:
        # Convert readings to DataFrame
        data_dict = {
            "IR": [r.ir_value for r in readings],
            "RED": [r.red_value for r in readings],
            "Timestamp": [r.timestamp for r in readings]
        }
        df = pd.DataFrame(data_dict)
        
        # 2. Slice into 15-second windows (6000 samples @ 400Hz)
        window_size = 6000
        windows = slice_raw_data_into_windows(df, window_size_samples=window_size)
        pipeline_log.append(f"Sliced data into {len(windows)} window(s) of 15 seconds.")
        
        if not windows:
            raise ValueError(f"Insufficient data to slice. Got {len(readings)} samples, need at least 3000 samples (7.5s) for a padded window.")
            
        window_features = []
        
        # 3. Process each window
        for idx, (ir_raw, red_raw) in enumerate(windows):
            pipeline_log.append(f"\n--- Processing Window {idx+1}/{len(windows)} ---")
            
            # Run signal processing
            proc_out = process_single_window(ir_raw, red_raw, fs=400.0)
            pipeline_log.append(proc_out["log"])
            
            # Check if beats were detected
            if len(proc_out["ir_beats"]) < 3 or len(proc_out["red_beats"]) < 3:
                pipeline_log.append(f"Warning: Low beat count in window {idx+1}. Skipping feature extraction for this window.")
                continue
                
            # Extract features (24 features)
            win_feats = extract_features_from_window(ir_raw, red_raw, proc_out, fs=400.0)
            window_features.append(win_feats)
            pipeline_log.append(f"Extracted 24 features from window {idx+1}.")
            
        if not window_features:
            raise ValueError("All windows failed quality check or had insufficient beat counts for feature extraction.")
            
        # 4. Average features across windows
        avg_features = average_features_across_windows(window_features)
        pipeline_log.append(f"\n--- Feature Averaging ---")
        pipeline_log.append(f"Averaged features across {len(window_features)} successful windows.")
        
        # 5. Predict blood glucose level
        pipeline_log.append(f"\n--- Blood Glucose Prediction ---")
        predicted_glucose, classification = predict_glucose(avg_features)
        pipeline_log.append(f"Predicted Blood Glucose: {predicted_glucose:.2f} mg/dL (Classification: {classification})")
        
        # 6. Save prediction in database
        prediction_record = models.Prediction(
            subject_id=subject_id,
            pipeline_run_id=run_record.id,
            predicted_glucose=predicted_glucose,
            actual_glucose=subject.glucose_level,
            features_json=avg_features,
            created_at=datetime.utcnow()
        )
        db.add(prediction_record)
        
        # Update run status
        run_record.status = "completed"
        run_record.completed_at = datetime.utcnow()
        run_record.log = "\n".join(pipeline_log)
        db.commit()
        db.refresh(prediction_record)
        
        return {
            "success": True,
            "prediction_id": prediction_record.id,
            "predicted_glucose": predicted_glucose,
            "classification": classification,
            "actual_glucose": subject.glucose_level,
            "run_id": run_record.id,
            "features": avg_features,
            "log": run_record.log
        }
        
    except Exception as e:
        pipeline_log.append(f"\nERROR: Pipeline failed: {str(e)}")
        # Update run status
        run_record.status = "failed"
        run_record.completed_at = datetime.utcnow()
        run_record.log = "\n".join(pipeline_log)
        db.commit()
        
        return {
            "success": False,
            "error": str(e),
            "run_id": run_record.id,
            "log": run_record.log
        }

@router.post("/run/{subject_id}")
def run_pipeline(subject_id: int, db: Session = Depends(get_db)):
    """Run the entire processing and prediction pipeline on the subject's raw data."""
    result = run_pipeline_sync(subject_id, db)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Pipeline execution failed."))
    return result

@router.get("/status/{run_id}", response_model=schemas.PipelineRunResponse)
def get_pipeline_run_status(run_id: int, db: Session = Depends(get_db)):
    """Get status and logs of a pipeline run."""
    run = db.query(models.PipelineRun).filter(models.PipelineRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return run
