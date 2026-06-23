from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class Subject(Base):
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    glucose_level = Column(Float, nullable=True)  # Reference actual glucose level (mg/dL)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    raw_readings = relationship("RawReading", back_populates="subject", cascade="all, delete-orphan")
    pipeline_runs = relationship("PipelineRun", back_populates="subject", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="subject", cascade="all, delete-orphan")

class RawReading(Base):
    __tablename__ = "raw_readings"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)  # Relative timestamp in seconds or unix timestamp
    ir_value = Column(Float, nullable=False)
    red_value = Column(Float, nullable=False)

    subject = relationship("Subject", back_populates="raw_readings")

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    log = Column(Text, nullable=True)

    subject = relationship("Subject", back_populates="pipeline_runs")
    predictions = relationship("Prediction", back_populates="pipeline_run", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False, index=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    predicted_glucose = Column(Float, nullable=False)
    actual_glucose = Column(Float, nullable=True)
    features_json = Column(JSON, nullable=False)  # Extracted features dictionary
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    subject = relationship("Subject", back_populates="predictions")
    pipeline_run = relationship("PipelineRun", back_populates="predictions")
