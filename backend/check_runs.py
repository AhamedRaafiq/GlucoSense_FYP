import sys
import os

# Set correct python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models import PipelineRun, Subject, RawReading

db = SessionLocal()
try:
    print("--- SUBJECTS ---")
    subjects = db.query(Subject).all()
    for s in subjects:
        readings_count = db.query(RawReading).filter(RawReading.subject_id == s.id).count()
        print(f"Subject: {s.name} (ID: {s.id}), Glucose Ref: {s.glucose_level}, Raw Readings Count: {readings_count}")
        
    print("\n--- PIPELINE RUNS ---")
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(10).all()
    for r in runs:
        print(f"Run ID: {r.id}, Subject ID: {r.subject_id}, Status: {r.status}, Started: {r.started_at}")
        if r.log:
            print("--- LOG ---")
            print(r.log)
            print("------------")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    db.close()
