import React, { useState, useEffect, useRef } from 'react';
import { ChevronRight, ChevronLeft, Upload, Play, Square, Settings, CheckCircle2, AlertTriangle, FileSpreadsheet, Cpu } from 'lucide-react';
import RealTimeChart from '../components/RealTimeChart';
import GlucoseGauge from '../components/GlucoseGauge';
import { subjectService, dataService, pipelineService } from '../services/api';

const DataCollectionPage = () => {
  const [step, setStep] = useState(1);
  const [subject, setSubject] = useState({ name: '', age: '', gender: 'Male', glucose_level: '' });
  const [createdSubject, setCreatedSubject] = useState(null);
  
  // Data Collection state
  const [collectMode, setCollectMode] = useState('upload'); // 'upload' or 'stream'
  const [serialPort, setSerialPort] = useState('COM7');
  const [simulateSensor, setSimulateSensor] = useState(false);
  const [availablePorts, setAvailablePorts] = useState([]);
  const [streamData, setStreamData] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [pointsCount, setPointsCount] = useState(0);
  const [file, setFile] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  
  // Pipeline state
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState(0); // 0 to 6
  const [pipelineResult, setPipelineResult] = useState(null);
  const [pipelineError, setPipelineError] = useState(null);

  const sseSourceRef = useRef(null);
  const allDataRef = useRef([]);
  const updateIntervalRef = useRef(null);

  // Clean up SSE stream and intervals on unmount
  useEffect(() => {
    return () => {
      if (sseSourceRef.current) {
        sseSourceRef.current.close();
      }
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
    };
  }, []);

  // Fetch available COM ports when switching to stream mode
  useEffect(() => {
    if (collectMode === 'stream') {
      const fetchPorts = async () => {
        try {
          const res = await dataService.listSerialPorts();
          if (res && res.ports && res.ports.length > 0) {
            setAvailablePorts(res.ports);
            setSerialPort(res.ports[0]);
            setSimulateSensor(false);
          } else {
            setAvailablePorts([]);
            setSimulateSensor(true); // default to simulation if no hardware is found
          }
        } catch (err) {
          console.error("Failed to fetch COM ports:", err);
          setAvailablePorts([]);
          setSimulateSensor(true);
        }
      };
      fetchPorts();
    }
  }, [collectMode]);

  const handleCreateSubject = async (e) => {
    e.preventDefault();
    if (!subject.name) return;
    try {
      const data = await subjectService.createSubject({
        name: subject.name,
        age: subject.age ? parseInt(subject.age) : null,
        gender: subject.gender,
        glucose_level: subject.glucose_level ? parseFloat(subject.glucose_level) : null
      });
      setCreatedSubject(data);
      setStep(2);
    } catch (err) {
      alert("Failed to register subject.");
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadSuccess(false);
    }
  };

  const handleFileUpload = async () => {
    if (!file || !createdSubject) return;
    try {
      setUploadSuccess(false);
      await dataService.uploadCSV(createdSubject.id, file);
      setUploadSuccess(true);
      // Auto move to step 3 after slight delay
      setTimeout(() => {
        setStep(3);
        runPredictionPipeline();
      }, 1000);
    } catch (err) {
      alert(err.response?.data?.detail || "CSV upload failed. Verify file columns.");
    }
  };

  const startLiveStreaming = async () => {
    if (!createdSubject) return;
    try {
      allDataRef.current = [];
      setStreamData([]);
      setPointsCount(0);
      
      // Start serial reader in backend
      await dataService.startSerial(createdSubject.id, serialPort, simulateSensor);
      setStreaming(true);
      
      // Establish WebSocket Client Connection
      const socket = new WebSocket(dataService.getWebSocketUrl(createdSubject.id));
      sseSourceRef.current = socket;
      
      socket.onmessage = (event) => {
        const reading = JSON.parse(event.data);
        allDataRef.current.push(reading);
      };
      
      socket.onerror = (err) => {
        console.error("WebSocket connection error:", err);
        stopLiveStreaming(false);
        alert("Live streaming disconnected or failed to connect.");
      };

      // Throttle UI rendering to 10 updates per second (100ms interval)
      // This prevents the browser UI thread from freezing at 400Hz high-frequency updates
      updateIntervalRef.current = setInterval(() => {
        const currentData = allDataRef.current;
        const currentLen = currentData.length;
        
        if (currentLen > 0) {
          setPointsCount(currentLen);
          // Pass only the last 150 points for visualization to keep DOM nodes small
          setStreamData([...currentData.slice(-150)]);
        }
        
        const limit = simulateSensor ? 1200 : 6000;
        if (currentLen >= limit) {
          clearInterval(updateIntervalRef.current);
          stopLiveStreaming(true);
        }
      }, 100);
      
    } catch (err) {
      alert("Failed to initialize serial data stream.");
    }
  };

  const stopLiveStreaming = async (autoMove = false) => {
    if (!createdSubject) return;
    try {
      if (sseSourceRef.current) {
        sseSourceRef.current.close();
      }
      if (updateIntervalRef.current) {
        clearInterval(updateIntervalRef.current);
      }
      setStreaming(false);
      
      // Call stop serial in the backend asynchronously so it doesn't block the UI transition
      dataService.stopSerial(createdSubject.id).catch(err => {
        console.error("Error stopping serial in backend:", err);
      });
      
      if (autoMove) {
        setStep(3);
        runPredictionPipeline();
      }
    } catch (err) {
      console.error("Error stopping serial read:", err);
    }
  };

  const runPredictionPipeline = async () => {
    if (!createdSubject) return;
    setPipelineLoading(true);
    setPipelineError(null);
    setPipelineResult(null);
    setPipelineProgress(0);
    
    // Simulate frontend visual progress of stages
    const interval = setInterval(() => {
      setPipelineProgress(prev => {
        if (prev < 5) return prev + 1;
        clearInterval(interval);
        return prev;
      });
    }, 900);
    
    try {
      const res = await pipelineService.runPipeline(createdSubject.id);
      clearInterval(interval);
      setPipelineProgress(6);
      setPipelineResult(res);
      // Auto move to step 4 when complete
      setTimeout(() => setStep(4), 1000);
    } catch (err) {
      clearInterval(interval);
      setPipelineError(err.response?.data?.detail || "Processing pipeline failed. Make sure you collected sufficient data.");
      setPipelineLoading(false);
    }
  };

  const resetWizard = () => {
    setSubject({ name: '', age: '', gender: 'Male', glucose_level: '' });
    setCreatedSubject(null);
    setStreamData([]);
    setPointsCount(0);
    setFile(null);
    setUploadSuccess(false);
    setPipelineResult(null);
    setPipelineError(null);
    setStep(1);
  };

  const progressSteps = [
    "Slicing 15s Windows",
    "Filtering Raw Signals",
    "Beat Detection & Averaging",
    "Extracting 19 channel features",
    "Engineering 24 Features",
    "Robust Scaling",
    "XGBoost estimation complete!"
  ];

  return (
    <div style={styles.container}>
      <div style={styles.wizardHeader}>
        <h1 style={{ margin: 0, fontSize: '1.875rem' }}>Non-Invasive Estimator</h1>
        <p style={{ margin: 0, fontSize: '0.875rem', color: '#64748b' }}>Predict blood glucose level step-by-step.</p>
      </div>

      {/* Wizard Steps indicator bar */}
      <div className="wizard-steps">
        <div className={`wizard-step ${step === 1 ? 'active' : step > 1 ? 'completed' : ''}`}>1</div>
        <div className={`wizard-step ${step === 2 ? 'active' : step > 2 ? 'completed' : ''}`}>2</div>
        <div className={`wizard-step ${step === 3 ? 'active' : step > 3 ? 'completed' : ''}`}>3</div>
        <div className={`wizard-step ${step === 4 ? 'active' : ''}`}>4</div>
      </div>

      {/* STEP 1: SUBJECT INFO */}
      {step === 1 && (
        <div className="card" style={styles.wizardCard}>
          <h3 style={styles.cardTitle}>Step 1: Patient Registration</h3>
          <form onSubmit={handleCreateSubject} style={styles.form}>
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <input 
                type="text" 
                className="form-input" 
                value={subject.name} 
                onChange={(e) => setSubject({ ...subject, name: e.target.value })}
                placeholder="e.g. John Doe"
                required
              />
            </div>
            
            <div style={styles.row}>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="form-label">Age</label>
                <input 
                  type="number" 
                  className="form-input" 
                  value={subject.age} 
                  onChange={(e) => setSubject({ ...subject, age: e.target.value })}
                  placeholder="e.g. 45"
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label className="form-label">Gender</label>
                <select 
                  className="form-select" 
                  value={subject.gender}
                  onChange={(e) => setSubject({ ...subject, gender: e.target.value })}
                >
                  <option>Male</option>
                  <option>Female</option>
                  <option>Other</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Clinical Reference Glucose (Optional - mg/dL)</label>
              <input 
                type="number" 
                className="form-input" 
                value={subject.glucose_level} 
                onChange={(e) => setSubject({ ...subject, glucose_level: e.target.value })}
                placeholder="Reference lab value for training calibration"
              />
            </div>

            <button type="submit" className="btn btn-primary" style={styles.nextBtn}>
              Register Patient
              <ChevronRight size={16} />
            </button>
          </form>
        </div>
      )}

      {/* STEP 2: DATA COLLECTION */}
      {step === 2 && (
        <div className="card" style={styles.wizardCard}>
          <div style={styles.cardHeader}>
            <h3 style={styles.cardTitle}>Step 2: Raw Signal Acquisition</h3>
            <span style={styles.patientBadge}>Subject: {createdSubject?.name}</span>
          </div>

          <div style={styles.tabContainer}>
            <button 
              onClick={() => setCollectMode('upload')}
              style={{ ...styles.tab, borderBottomColor: collectMode === 'upload' ? '#6366f1' : 'transparent', color: collectMode === 'upload' ? '#6366f1' : '#94a3b8' }}
            >
              <FileSpreadsheet size={16} />
              Upload Raw CSV
            </button>
            <button 
              onClick={() => setCollectMode('stream')}
              style={{ ...styles.tab, borderBottomColor: collectMode === 'stream' ? '#6366f1' : 'transparent', color: collectMode === 'stream' ? '#6366f1' : '#94a3b8' }}
            >
              <Cpu size={16} />
              Live Sensor Stream
            </button>
          </div>

          {collectMode === 'upload' ? (
            <div style={styles.tabContent}>
              <p style={{ fontSize: '0.9rem' }}>
                Upload a raw diagnostic recording CSV file. The file should contain column headers `IR` and `RED` (with values output from sensor photodiode).
              </p>
              
              <div style={styles.fileDropZone}>
                <Upload size={32} style={{ color: '#6366f1', marginBottom: '0.75rem' }} />
                <input 
                  type="file" 
                  accept=".csv" 
                  onChange={handleFileChange}
                  style={styles.fileInput}
                />
                <span style={{ fontSize: '0.9rem', fontWeight: '500' }}>
                  {file ? file.name : "Click to select CSV file"}
                </span>
                <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                  Accepts standard data logger CSV format.
                </span>
              </div>

              {uploadSuccess && (
                <div style={styles.successAlert}>
                  <CheckCircle2 size={16} style={{ color: '#10b981' }} />
                  <span>CSV data uploaded successfully. Starting pipeline...</span>
                </div>
              )}

              <div style={styles.btnRow}>
                <button onClick={() => setStep(1)} className="btn btn-secondary">
                  <ChevronLeft size={16} />
                  Back
                </button>
                <button 
                  onClick={handleFileUpload} 
                  disabled={!file} 
                  className={`btn btn-primary ${!file ? 'btn-disabled' : ''}`}
                >
                  Upload & Estimate
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          ) : (
            <div style={styles.tabContent}>
              <div style={styles.streamConfig}>
                <div className="form-group" style={{ flex: 1, margin: 0 }}>
                  <label className="form-label">Serial COM Port</label>
                  {availablePorts.length > 0 ? (
                    <select 
                      className="form-select" 
                      value={serialPort} 
                      onChange={(e) => setSerialPort(e.target.value)}
                    >
                      {availablePorts.map((p) => (
                        <option key={p} value={p}>{p}</option>
                      ))}
                    </select>
                  ) : (
                    <input 
                      type="text" 
                      className="form-input" 
                      value={serialPort} 
                      onChange={(e) => setSerialPort(e.target.value)}
                      placeholder="e.g. COM3"
                    />
                  )}
                </div>
                <div className="form-group" style={{ flex: 1, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1.5rem' }}>
                  <input 
                    type="checkbox" 
                    checked={simulateSensor} 
                    onChange={(e) => setSimulateSensor(e.target.checked)}
                    id="simulate"
                  />
                  <label htmlFor="simulate" className="form-label" style={{ margin: 0, cursor: 'pointer' }}>
                    Simulate MAX30102 Signal
                  </label>
                </div>
              </div>
              
              {availablePorts.length === 0 && (
                <div style={{ color: '#f59e0b', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.25rem', marginTop: '-0.5rem' }}>
                  <AlertTriangle size={14} />
                  <span>No hardware COM ports detected. Simulation mode enabled. Connect your ESP32 and refresh stream.</span>
                </div>
              )}

              {/* Streaming dashboard indicator */}
              <div style={styles.streamDashboard}>
                <div style={styles.streamInfo}>
                  <span style={styles.infoLabel}>Readings captured</span>
                  <span style={styles.infoValue}>{pointsCount} / {simulateSensor ? 1200 : 6000}</span>
                </div>
                <div style={styles.streamButtons}>
                  {!streaming ? (
                    <button onClick={startLiveStreaming} className="btn btn-primary" style={styles.streamBtn}>
                      <Play size={16} />
                      Start Stream
                    </button>
                  ) : (
                    <button onClick={() => stopLiveStreaming(false)} className="btn btn-danger" style={styles.streamBtn}>
                      <Square size={16} />
                      Stop Stream
                    </button>
                  )}
                </div>
              </div>

              {/* Real-time Numeric Values */}
              {streaming && streamData.length > 0 && (
                <div style={styles.liveDataBox}>
                  <div style={styles.liveValueItem}>
                    <span style={styles.liveValueLabel}>Latest IR Value</span>
                    <span style={styles.liveValueVal}>{streamData[streamData.length - 1].ir_value.toLocaleString()}</span>
                  </div>
                  <div style={styles.liveValueItem}>
                    <span style={styles.liveValueLabel}>Latest RED Value</span>
                    <span style={styles.liveValueValRed}>{streamData[streamData.length - 1].red_value.toLocaleString()}</span>
                  </div>
                </div>
              )}
 
              {/* Live Signal Chart */}
              {streamData.length > 0 && (
                <div style={{ marginTop: '1.5rem' }}>
                  <h4 style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Real-time Waveform Monitor
                  </h4>
                  <RealTimeChart data={streamData} />
                </div>
              )}

              <div style={styles.btnRow}>
                <button onClick={() => setStep(1)} className="btn btn-secondary">
                  <ChevronLeft size={16} />
                  Back
                </button>
                <button 
                  onClick={() => stopLiveStreaming(true)} 
                  disabled={pointsCount < (simulateSensor ? 300 : 1500)} 
                  className={`btn btn-primary ${pointsCount < (simulateSensor ? 300 : 1500) ? 'btn-disabled' : ''}`}
                >
                  Process & Predict
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* STEP 3: PIPELINE PROCESSING */}
      {step === 3 && (
        <div className="card" style={styles.wizardCard}>
          <h3 style={styles.cardTitle}>Step 3: Signal Processing Automation Pipeline</h3>
          
          {pipelineError ? (
            <div style={styles.errorContainer}>
              <AlertTriangle size={32} style={{ color: '#ef4444', marginBottom: '0.75rem' }} />
              <h4 style={{ margin: 0, fontWeight: 'bold' }}>Pipeline Execution Failed</h4>
              <p style={{ margin: '0.5rem 0', fontSize: '0.9rem', color: '#cbd5e1', textAlign: 'center' }}>
                {pipelineError}
              </p>
              <button onClick={() => setStep(2)} className="btn btn-secondary" style={{ marginTop: '1rem' }}>
                Reacquire Data
              </button>
            </div>
          ) : (
            <div style={styles.pipelineProgressList}>
              <div style={styles.progressAnimation}>
                <div className="spinner" style={{ width: '48px', height: '48px' }}></div>
                <p style={{ marginTop: '1rem', fontWeight: 'bold' }}>Computing Diagnostic Pipeline...</p>
              </div>
              
              <div style={styles.stagesBox}>
                {progressSteps.map((stepText, index) => (
                  <div key={index} style={{
                    ...styles.progressStepItem,
                    opacity: pipelineProgress >= index ? 1 : 0.25,
                    color: pipelineProgress > index ? '#10b981' : pipelineProgress === index ? '#6366f1' : '#94a3b8'
                  }}>
                    {pipelineProgress > index ? (
                      <CheckCircle2 size={16} style={{ color: '#10b981', flexShrink: 0 }} />
                    ) : (
                      <div style={{
                        ...styles.bullet,
                        backgroundColor: pipelineProgress === index ? '#6366f1' : '#334155',
                        boxShadow: pipelineProgress === index ? '0 0 8px #6366f1' : 'none'
                      }}></div>
                    )}
                    <span>{stepText}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* STEP 4: PREDICTION RESULT */}
      {step === 4 && (
        <div style={styles.resultGrid}>
          {/* Gauge card */}
          <div className="card" style={styles.gaugeCard}>
            <h3 style={{ ...styles.cardTitle, textAlign: 'center', marginBottom: '1.5rem' }}>Glucose Prediction Result</h3>
            {pipelineResult && <GlucoseGauge value={pipelineResult.predicted_glucose} />}
            <button onClick={resetWizard} className="btn btn-secondary" style={{ marginTop: '2rem', width: '100%' }}>
              Test Another Patient
            </button>
          </div>

          {/* Features card */}
          <div className="card" style={styles.featuresCard}>
            <h3 style={styles.cardTitle}>Extracted Diagnostic Features</h3>
            <p style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '1rem' }}>
              Average computed features representing patient physiology. Only top features are listed.
            </p>
            <div style={styles.featuresTableWrapper}>
              <table>
                <thead>
                  <tr>
                    <th>Feature Label</th>
                    <th>Computed Value</th>
                  </tr>
                </thead>
                <tbody>
                  {pipelineResult && Object.entries(pipelineResult.features).slice(0, 14).map(([key, val]) => (
                    <tr key={key}>
                      <td style={{ fontSize: '0.85rem', color: '#94a3b8' }}>{key}</td>
                      <td style={{ fontSize: '0.85rem', fontWeight: 'bold', fontFamily: 'monospace' }}>
                        {val.toFixed(5)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2rem',
    maxWidth: '800px',
    margin: '0 auto',
  },
  wizardHeader: {
    textAlign: 'center',
    marginBottom: '1rem',
  },
  liveDataBox: {
    display: 'flex',
    gap: '2rem',
    justifyContent: 'center',
    background: 'rgba(15, 23, 42, 0.4)',
    padding: '1rem',
    borderRadius: '0.75rem',
    border: '1px solid rgba(255, 255, 255, 0.05)',
    marginTop: '1rem',
  },
  liveValueItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '0.25rem',
    flex: 1,
  },
  liveValueLabel: {
    fontSize: '0.75rem',
    color: '#94a3b8',
    textTransform: 'uppercase',
    fontWeight: 'bold',
  },
  liveValueVal: {
    fontSize: '1.75rem',
    fontWeight: '700',
    color: '#06b6d4',
    fontFamily: 'monospace',
  },
  liveValueValRed: {
    fontSize: '1.75rem',
    fontWeight: '700',
    color: '#ef4444',
    fontFamily: 'monospace',
  },
  wizardCard: {
    padding: '2rem',
    background: 'rgba(30, 41, 59, 0.4)',
  },
  cardTitle: {
    fontSize: '1.25rem',
    fontWeight: '700',
    marginBottom: '1.5rem',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
  },
  row: {
    display: 'flex',
    gap: '1rem',
  },
  nextBtn: {
    marginTop: '1.5rem',
    alignSelf: 'flex-end',
    gap: '0.5rem',
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.5rem',
    borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    paddingBottom: '0.75rem',
  },
  patientBadge: {
    background: 'rgba(99, 102, 241, 0.1)',
    color: '#a5b4fc',
    fontSize: '0.8rem',
    fontWeight: '600',
    padding: '0.35rem 0.75rem',
    borderRadius: '0.5rem',
  },
  tabContainer: {
    display: 'flex',
    borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    marginBottom: '1.5rem',
  },
  tab: {
    background: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    padding: '0.75rem 1.5rem',
    fontSize: '0.9rem',
    fontWeight: '600',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    transition: 'all 0.2s',
  },
  tabContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  fileDropZone: {
    border: '2px dashed rgba(99, 102, 241, 0.3)',
    borderRadius: '0.75rem',
    padding: '2.5rem',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    position: 'relative',
    background: 'rgba(99, 102, 241, 0.02)',
    transition: 'all 0.3s',
  },
  fileInput: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    opacity: 0,
    cursor: 'pointer',
  },
  btnRow: {
    display: 'flex',
    justifyContent: 'space-between',
    marginTop: '1.5rem',
  },
  successAlert: {
    background: 'rgba(16, 185, 129, 0.1)',
    border: '1px solid rgba(16, 185, 129, 0.2)',
    color: '#10b981',
    borderRadius: '0.5rem',
    padding: '0.75rem 1rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.85rem',
  },
  streamConfig: {
    display: 'flex',
    gap: '1.5rem',
    alignItems: 'center',
    background: 'rgba(15, 23, 42, 0.3)',
    padding: '1.25rem',
    borderRadius: '0.75rem',
    border: '1px solid rgba(255, 255, 255, 0.04)',
  },
  streamDashboard: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    background: 'rgba(30, 41, 59, 0.5)',
    padding: '1rem 1.5rem',
    borderRadius: '0.75rem',
    border: '1px solid rgba(255, 255, 255, 0.08)',
  },
  streamInfo: {
    display: 'flex',
    flexDirection: 'column',
  },
  infoLabel: {
    fontSize: '0.75rem',
    color: '#64748b',
    textTransform: 'uppercase',
    fontWeight: 'bold',
  },
  infoValue: {
    fontSize: '1.25rem',
    fontWeight: '800',
    color: 'white',
  },
  streamButtons: {
    display: 'flex',
  },
  streamBtn: {
    gap: '0.5rem',
    padding: '0.6rem 1.25rem',
  },
  pipelineProgressList: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '2rem',
    padding: '1rem 0',
  },
  progressAnimation: {
    textAlign: 'center',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  stagesBox: {
    width: '100%',
    maxWidth: '420px',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    background: 'rgba(15, 23, 42, 0.4)',
    padding: '1.5rem',
    borderRadius: '0.75rem',
    border: '1px solid rgba(255, 255, 255, 0.05)',
  },
  progressStepItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    fontSize: '0.9rem',
    fontWeight: '500',
    transition: 'all 0.3s ease',
  },
  bullet: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  errorContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '2rem 0',
  },
  resultGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
    gap: '2rem',
  },
  gaugeCard: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem',
  },
  featuresCard: {
    padding: '2rem',
  },
  featuresTableWrapper: {
    maxHeight: '340px',
    overflowY: 'auto',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '0.5rem',
  }
};

export default DataCollectionPage;
