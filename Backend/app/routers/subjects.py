from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ..database import get_db
from .. import models, schemas

router = APIRouter(
    prefix="/subjects",
    tags=["subjects"]
)

@router.get("/", response_model=List[schemas.SubjectResponse])
def get_subjects(db: Session = Depends(get_db)):
    return db.query(models.Subject).order_by(models.Subject.created_at.desc()).all()

@router.post("/", response_model=schemas.SubjectResponse, status_code=status.HTTP_201_CREATED)
def create_subject(subject: schemas.SubjectCreate, db: Session = Depends(get_db)):
    db_subject = models.Subject(
        name=subject.name,
        age=subject.age,
        gender=subject.gender,
        glucose_level=subject.glucose_level
    )
    db.add(db_subject)
    db.commit()
    db.refresh(db_subject)
    return db_subject

@router.get("/{subject_id}", response_model=schemas.SubjectResponse)
def get_subject(subject_id: int, db: Session = Depends(get_db)):
    subject = db.query(models.Subject).filter(models.Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    return subject

@router.delete("/{subject_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subject(subject_id: int, db: Session = Depends(get_db)):
    subject = db.query(models.Subject).filter(models.Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    db.delete(subject)
    db.commit()
    return None
