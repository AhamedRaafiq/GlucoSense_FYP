from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Subject Schemas
class SubjectBase(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    glucose_level: Optional[float] = None

class SubjectCreate(SubjectBase):
    pass

class SubjectResponse(SubjectBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Raw Reading Schemas
class RawReadingBase(BaseModel):
    timestamp: float
    ir_value: float
    red_value: float

class RawReadingCreate(RawReadingBase):
    pass

class RawReadingResponse(RawReadingBase):
    id: int
    subject_id: int

    class Config:
        from_attributes = True

class RawReadingBulkCreate(BaseModel):
    subject_id: int
    readings: List[RawReadingCreate]

# Pipeline Run Schemas
class PipelineRunResponse(BaseModel):
    id: int
    subject_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    log: Optional[str] = None

    class Config:
        from_attributes = True

# Prediction Schemas
class PredictionResponse(BaseModel):
    id: int
    subject_id: int
    pipeline_run_id: int
    predicted_glucose: float
    actual_glucose: Optional[float] = None
    features_json: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True

# Dashboard Stats Schemas
class DashboardStats(BaseModel):
    total_subjects: int
    total_predictions: int
    average_predicted_glucose: float
    average_actual_glucose: Optional[float] = None
    model_mae: float
    model_r2: float
    model_rmse: float

class GlucoseDistributionItem(BaseModel):
    range_label: str  # e.g., "Normal (70-100)"
    count: int
    color: str

class PredictedVsActualItem(BaseModel):
    id: int
    subject_name: str
    actual: Optional[float]
    predicted: float
    created_at: datetime

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
