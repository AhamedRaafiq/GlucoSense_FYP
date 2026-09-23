import os
import io
import time
import math
import asyncio
import threading
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Any
import serial
from ..database import get_db, SessionLocal
from .. import models, schemas

router = APIRouter(
    prefix="/data",
    tags=["data_collection"]
)

# Thread-safe global state for active collections and streams
active_sessions = {}  # subject_id -> {"thread": Thread, "stop_event": Event, "mode": str}
stream_queues = {}   # subject_id -> List[asyncio.Queue]
lock = threading.Lock()

class SerialReaderThread(threading.Thread):
    def __init__(self, subject_id: int, port: str, baud: int, loop: asyncio.AbstractEventLoop, simulate: bool = False):
        super().__init__()
        self.subject_id = subject_id
        self.port = port
        self.baud = baud
        self.loop = loop
        self.simulate = simulate
        self.stop_event = threading.Event()
        
    def run(self):
        db = SessionLocal()
        ser = None
        
        # Buffer to insert readings in bulk
        reading_buffer = []
        last_insert_time = time.time()
        
        try:
            if not self.simulate:
                ser = serial.Serial(self.port, self.baud, timeout=0.1)
                
            start_time = time.time()
            sim_t = 0.0
            
            while not self.stop_event.is_set():
                if self.simulate:
                    # Simulating PPG wave at ~100Hz (sleep 10ms)
                    time.sleep(0.01)
                    sim_t += 0.01
                    # Base signal with pulse components
                    hr_freq = 1.2  # 72 BPM
                    ir_val = 80000 + 8000 * math.sin(2 * math.pi * hr_freq * sim_t) + 2000 * math.sin(2 * math.pi * 2 * hr_freq * sim_t) + 300 * math.sin(sim_t * 0.1)
                    red_val = 75000 + 7200 * math.sin(2 * math.pi * hr_freq * sim_t + 0.1) + 1800 * math.sin(2 * math.pi * 2 * hr_freq * sim_t + 0.1) + 250 * math.sin(sim_t * 0.1)
                    # Add noise
                    ir_val += np.random.normal(0, 50) if 'np' in globals() else 0.0
                    red_val += np.random.normal(0, 50) if 'np' in globals() else 0.0
                else:
                    if ser and ser.in_waiting > 0:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if not line:
                            continue
                        try:
                            # Expecting comma separated IR,RED
                            parts = line.split(',')
                            if len(parts) >= 2:
                                ir_val = float(parts[0])
                                red_val = float(parts[1])
                            else:
                                continue
                        except ValueError:
                            continue
                    else:
                        time.sleep(0.005)
                        continue
                
                timestamp = time.time() - start_time
                
                # Create reading
                reading_data = {
                    "timestamp": timestamp,
                    "ir_value": float(ir_val),
                    "red_value": float(red_val)
                }
                
                # 1. Broadcaster (send to active async queues)
                with lock:
                    if self.subject_id in stream_queues:
                        for q in stream_queues[self.subject_id]:
                            # Run thread-safe call to push to asyncio queue using saved event loop
                            if self.loop.is_running():
                                self.loop.call_soon_threadsafe(q.put_nowait, reading_data)
                
                # 2. Database Buffer
                reading_buffer.append(
                    models.RawReading(
                        subject_id=self.subject_id,
                        timestamp=timestamp,
                        ir_value=ir_val,
                        red_value=red_val
                    )
                )
                
                # Save to database in bulk every 1.0 second or every 400 readings
                if len(reading_buffer) >= 400 or (time.time() - last_insert_time >= 1.0 and reading_buffer):
                    db.add_all(reading_buffer)
                    db.commit()
                    reading_buffer = []
                    last_insert_time = time.time()
                    
            # Insert any leftover readings
            if reading_buffer:
                db.add_all(reading_buffer)
                db.commit()
                
        except Exception as e:
            print(f"Error in SerialReaderThread: {e}")
        finally:
            if ser:
                ser.close()
            db.close()

@router.post("/upload")
async def upload_csv_data(
    subject_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload raw PPG data in CSV format.
    Expects headers: 'IR', 'RED' (or 'ir_value', 'red_value').
    Optional 'Timestamp' column.
    """
    subject = db.query(models.Subject).filter(models.Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Check column names
        cols = {c.upper(): c for c in df.columns}
        ir_col = cols.get('IR') or cols.get('IR_VALUE')
        red_col = cols.get('RED') or cols.get('RED_VALUE')
        
        if not ir_col or not red_col:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain IR and RED columns (case-insensitive)."
            )
            
        time_col = cols.get('TIMESTAMP') or cols.get('TIME')
        
        # Determine timestamps
        timestamps = []
        if time_col:
            try:
                # Try to convert to float directly first
                first_val = df.iloc[0][time_col]
                float(first_val)
                raw_times = df[time_col].astype(float).values
                timestamps = raw_times - raw_times[0]
            except (ValueError, TypeError):
                # Try parsing as datetime/time strings
                try:
                    parsed_times = pd.to_datetime(df[time_col], errors='coerce')
                    if parsed_times.isna().sum() > len(parsed_times) * 0.5:
                        raise ValueError("Too many unparseable date/time strings")
                    parsed_times = parsed_times.ffill().bfill()
                    deltas = parsed_times - parsed_times.iloc[0]
                    timestamps = deltas.dt.total_seconds().values
                except Exception:
                    # Fallback to index-based if parsing fails
                    timestamps = [float(idx / 400.0) for idx in range(len(df))]
        else:
            timestamps = [float(idx / 400.0) for idx in range(len(df))]
            
        # Clean existing raw readings
        db.query(models.RawReading).filter(models.RawReading.subject_id == subject_id).delete()
        db.commit()
        
        # Parse and insert readings
        readings = []
        for idx, row in df.iterrows():
            t_val = float(timestamps[idx])
            readings.append(
                models.RawReading(
                    subject_id=subject_id,
                    timestamp=t_val,
                    ir_value=float(row[ir_col]),
                    red_value=float(row[red_col])
                )
            )
            
        # Bulk save (chunked to prevent DB overloading)
        chunk_size = 5000
        for i in range(0, len(readings), chunk_size):
            db.add_all(readings[i:i+chunk_size])
            db.commit()
            
        return {
            "success": True,
            "message": f"Successfully uploaded and saved {len(readings)} readings."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

import serial.tools.list_ports

@router.get("/serial/ports")
def list_serial_ports():
    """List available COM ports."""
    try:
        ports = [port.device for port in serial.tools.list_ports.comports()]
        return {"ports": ports}
    except Exception as e:
        return {"ports": [], "error": str(e)}

@router.post("/serial/start")
async def start_serial_collection(
    subject_id: int,
    port: str = "COM7",
    baud: int = 115200,
    simulate: bool = Query(False, description="Simulate sensor inputs if hardware is unavailable"),
    db: Session = Depends(get_db)
):
    """Start reading PPG sensor data via Serial or Simulation."""
    subject = db.query(models.Subject).filter(models.Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
        
    with lock:
        if subject_id in active_sessions:
            return {"success": True, "message": "Collection already running for this subject."}
            
        # Clear old readings
        db.query(models.RawReading).filter(models.RawReading.subject_id == subject_id).delete()
        db.commit()
        
        loop = asyncio.get_running_loop()
        thread = SerialReaderThread(subject_id, port, baud, loop, simulate=simulate)
        thread.start()
        
        active_sessions[subject_id] = {
            "thread": thread,
            "stop_event": thread.stop_event,
            "mode": "simulation" if simulate else "serial"
        }
        
    return {
        "success": True,
        "message": f"Data collection started in {'simulation' if simulate else 'serial'} mode."
    }

@router.post("/serial/stop")
def stop_serial_collection(subject_id: int):
    """Stop reading from Serial/Simulation."""
    with lock:
        if subject_id not in active_sessions:
            return {"success": True, "message": "No active collection found for this subject."}
            
        session = active_sessions[subject_id]
        session["stop_event"].set()
        session["thread"].join()
        
        del active_sessions[subject_id]
        
    return {
        "success": True,
        "message": "Data collection stopped."
    }

@router.get("/serial/status/{subject_id}")
def get_serial_status(subject_id: int):
    """Get active collection status for a subject."""
    with lock:
        is_active = subject_id in active_sessions
        mode = active_sessions[subject_id]["mode"] if is_active else None
        
    return {
        "is_active": is_active,
        "mode": mode
    }

@router.get("/serial/stream/{subject_id}")
async def stream_live_data(subject_id: int):
    """SSE Stream endpoint for live charting on the frontend."""
    # Create asyncio queue for this stream client
    q = asyncio.Queue()
    
    with lock:
        if subject_id not in stream_queues:
            stream_queues[subject_id] = []
        stream_queues[subject_id].append(q)
        
    async def event_generator():
        try:
            while True:
                # Get reading from serial thread
                reading = await q.get()
                yield f"data: {json.dumps(reading)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # Clean up queue when client disconnects
            with lock:
                if subject_id in stream_queues and q in stream_queues[subject_id]:
                    stream_queues[subject_id].remove(q)
                    if not stream_queues[subject_id]:
                        del stream_queues[subject_id]

    import json
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.websocket("/serial/ws/{subject_id}")
async def websocket_endpoint(websocket: WebSocket, subject_id: int):
    """WebSocket endpoint for raw low-latency real-time PPG data streaming."""
    await websocket.accept()
    q = asyncio.Queue()
    
    with lock:
        if subject_id not in stream_queues:
            stream_queues[subject_id] = []
        stream_queues[subject_id].append(q)
        
    try:
        import json
        while True:
            # Retrieve computed reading from SerialReaderThread
            reading = await q.get()
            # Send immediately to connected websocket client
            await websocket.send_json(reading)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket session error: {e}")
    finally:
        with lock:
            if subject_id in stream_queues and q in stream_queues[subject_id]:
                stream_queues[subject_id].remove(q)
                if not stream_queues[subject_id]:
                    del stream_queues[subject_id]
