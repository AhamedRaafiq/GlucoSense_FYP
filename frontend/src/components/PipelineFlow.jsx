import React, { useState } from 'react';
import { ArrowRight, Cpu, FileText, CheckCircle2, ChevronRight } from 'lucide-react';

const PipelineFlow = () => {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      id: 1,
      title: "1. ESP32 Raw Data",
      description: "ESP32-S3 collects PPG sensor data using MAX30102.",
      details: "Samples dual-wavelength light absorption (IR and RED) at 400Hz with 18-bit ADC. Threshold checks for finger contact.",
      file: "main_code01.c",
      input: "Finger Photodiode reflection",
      output: "Serial print: 'ir_value,red_value'"
    },
    {
      id: 2,
      title: "2. Data Logger",
      description: "Saves serial raw readings to local storage.",
      details: "Reads COM port at 115200 baud. Real-time plot visualization of signals and logs CSV format (Timestamp, IR, RED).",
      file: "data_logger_code02.py",
      input: "Serial values",
      output: "05_Data_Storage/01_Raw/*.csv"
    },
    {
      id: 3,
      title: "3. Verified Raw",
      description: "Maintains high quality good recordings only.",
      details: "Saves only verified PPG signals without movement artifacts or bad contact readings.",
      file: "05_Data_Storage/02_Verified_Raw",
      input: "Raw CSVs",
      output: "Verified raw files"
    },
    {
      id: 4,
      title: "4. Window Slicer",
      description: "Slices verified PPG signals into 15-second windows.",
      details: "Creates non-overlapping segments of 6000 samples (@ 400Hz) for feature extraction and analysis consistency.",
      file: "step3_window_slicer_code03.ipynb",
      input: "Verified CSVs",
      output: "05_Data_Storage/03_Windowed/{SubjectName}_Win{N}.csv"
    },
    {
      id: 5,
      title: "5. Signal Processing",
      description: "Removes noise and baseline wanders.",
      details: "Spike removal (median filter, k=3) → Signal inversion (-1x) → Lowpass filter (Butterworth 16Hz) → Highpass filter (Butterworth 0.5Hz) → MinMax normalization.",
      file: "Automated_Signal_Processing_Code04.py",
      input: "Windowed CSV",
      output: "04_Filtered/ (Filtered Full & Ensemble averages)"
    },
    {
      id: 6,
      title: "6. Feature Extraction",
      description: "Extracts clinical and statistical features.",
      details: "Extracts 19 time-domain, frequency-domain, derivative, and morphological features per channel (RED & IR).",
      file: "Feature_Extraction_Code05.py",
      input: "Filtered signals & beat segments",
      output: "05_Features_/ (19 features × 2 channels = 38 flat)"
    },
    {
      id: 7,
      title: "7. Average Features",
      description: "Averages features across windows per subject.",
      details: "Combines window features of a subject into a single representation to increase statistics reliability.",
      file: "Average_Feature_Extraction_Code06.py",
      input: "Window features",
      output: "06_Averaged_Features/{Subject}_AveFeature.csv"
    },
    {
      id: 8,
      title: "8. Dataset Creation",
      description: "Combines averaged features with reference glucose.",
      details: "Extracts subject ID, matches it with reference glucose level from excel metadata collection sheet, and constructs master table.",
      file: "Data_Set_Creation_Code07.py",
      input: "Averaged features + Metadata collection Excel",
      output: "07_Final_Data_Set/MASTER_Dataset.csv"
    },
    {
      id: 9,
      title: "9. Feature Engineering",
      description: "Creates engineered ratio/difference features.",
      details: "Selects 18 IR base features + creates 5 ratio/difference combinations (e.g. Ratio_TEO_Mean, Diff_Spectral_Entropy) + 1 Ensemble ratio = 24 features.",
      file: "Data_set_with_24_Features_creation_08.py",
      input: "38 features MASTER dataset",
      output: "08_Data_set_with_24_features/ (24 features master)"
    },
    {
      id: 10,
      title: "10. Data Cleaning",
      description: "Outlier handling and NaN imputation.",
      details: "Handles missing values by median imputation and clips outliers using Interquartile Range method (IQR x 1.5).",
      file: "Cleaned_dataset_without_NaN_and_outliers_Creation_code09.py",
      input: "24 features master dataset",
      output: "09_Cleaned_dataset/ (Cleaned master dataset)"
    },
    {
      id: 11,
      title: "11. Scaling & Split",
      description: "Robust scaling and stratified train-test split.",
      details: "Stratifies split based on glucose classes (Hypo, Normal, Prediabetic, Diabetic) to prevent bias. Fits RobustScaler (median & IQR) on X_train only.",
      file: "Train_Test_Split_and_Robust_Scaling_Code10.py",
      input: "Cleaned dataset",
      output: "10_Robust_Scaled_and_Train_Test_Splitted_Data_Set/ (train/test CSVs & JSON)"
    },
    {
      id: 12,
      title: "12. XGBoost Predictor",
      description: "Predicts blood glucose level using boosting.",
      details: "Trains XGBRegressor on 15 selected features. Predicts blood glucose in mg/dL. Retrains automatically with new data.",
      file: "XGBoost_ML_Code11.py",
      input: "15 scaled features",
      output: "Blood Glucose Level (mg/dL)"
    }
  ];

  return (
    <div style={styles.container}>
      <div style={styles.stepsGrid}>
        {steps.map((step, idx) => (
          <div
            key={step.id}
            onClick={() => setActiveStep(idx)}
            style={{
              ...styles.stepCard,
              borderColor: activeStep === idx ? '#6366f1' : 'rgba(255, 255, 255, 0.08)',
              background: activeStep === idx ? 'rgba(99, 102, 241, 0.1)' : 'rgba(30, 41, 59, 0.35)'
            }}
          >
            <div style={styles.stepHeader}>
              <span style={{
                ...styles.stepNumber,
                backgroundColor: activeStep === idx ? '#6366f1' : '#334155',
                color: 'white'
              }}>{step.id}</span>
              <h4 style={styles.stepTitle}>{step.title.split('. ')[1]}</h4>
            </div>
            <p style={styles.stepDesc}>{step.description}</p>
          </div>
        ))}
      </div>

      <div style={styles.detailCard}>
        <div style={styles.detailHeader}>
          <Cpu size={24} style={styles.detailIcon} />
          <h3 style={styles.detailTitle}>{steps[activeStep].title}</h3>
        </div>
        
        <p style={styles.detailText}>{steps[activeStep].details}</p>
        
        <div style={styles.metaGrid}>
          <div style={styles.metaItem}>
            <span style={styles.metaLabel}>Code/Folder File</span>
            <span style={styles.metaValue}>{steps[activeStep].file}</span>
          </div>
          <div style={styles.metaItem}>
            <span style={styles.metaLabel}>Input data</span>
            <span style={styles.metaValue}>{steps[activeStep].input}</span>
          </div>
          <div style={styles.metaItem}>
            <span style={styles.metaLabel}>Output output</span>
            <span style={styles.metaValue}>{steps[activeStep].output}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2.5rem',
  },
  stepsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
    gap: '1rem',
  },
  stepCard: {
    padding: '1.25rem',
    borderRadius: '0.75rem',
    border: '1px solid',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  stepHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    marginBottom: '0.5rem',
  },
  stepNumber: {
    width: '1.75rem',
    height: '1.75rem',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '0.8rem',
    fontWeight: 'bold',
  },
  stepTitle: {
    fontSize: '0.95rem',
    fontWeight: '600',
    margin: 0,
  },
  stepDesc: {
    fontSize: '0.8rem',
    color: '#94a3b8',
    margin: 0,
    lineHeight: '1.4',
  },
  detailCard: {
    background: 'rgba(30, 41, 59, 0.6)',
    backdropFilter: 'blur(12px)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    borderRadius: '1rem',
    padding: '2rem',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.4)',
  },
  detailHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    marginBottom: '1rem',
    borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    paddingBottom: '1rem',
  },
  detailIcon: {
    color: '#06b6d4',
  },
  detailTitle: {
    fontSize: '1.5rem',
    fontWeight: '700',
    margin: 0,
  },
  detailText: {
    fontSize: '1.05rem',
    color: '#cbd5e1',
    marginBottom: '1.5rem',
    lineHeight: '1.6',
  },
  metaGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: '1.5rem',
  },
  metaItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
    background: 'rgba(15, 23, 42, 0.4)',
    padding: '1rem',
    borderRadius: '0.5rem',
    border: '1px solid rgba(255, 255, 255, 0.04)',
  },
  metaLabel: {
    fontSize: '0.75rem',
    fontWeight: '600',
    textTransform: 'uppercase',
    color: '#64748b',
    letterSpacing: '0.05em',
  },
  metaValue: {
    fontSize: '0.9rem',
    color: '#38bdf8',
    wordBreak: 'break-all',
  }
};

export default PipelineFlow;
