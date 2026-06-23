import React from 'react';
import { Cpu, HelpCircle, HardDrive, Info } from 'lucide-react';

const AboutPage = () => {
  return (
    <div style={styles.container}>
      <h1 style={styles.pageTitle}>About the Project</h1>
      <p style={styles.introText}>
        This system is a non-invasive diagnostics device developed for blood glucose estimation using Photoplethysmography (PPG) waveform modeling and Gradient Boosting machine learning.
      </p>

      {/* Hardware Setup Card */}
      <div className="card" style={styles.card}>
        <div style={styles.cardHeader}>
          <Cpu size={24} style={{ color: '#06b6d4' }} />
          <h3 style={styles.cardTitle}>Hardware Pin Connections</h3>
        </div>
        <p style={{ fontSize: '0.9rem', marginBottom: '1.25rem' }}>
          Pin layout connecting the MAX30102 PPG sensor to the ESP32-S3 microcontroller board.
        </p>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>ESP32-S3 Pin</th>
                <th>MAX30102 Pin</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ fontWeight: 'bold', color: '#6366f1' }}>GPIO 1</td>
                <td style={{ fontWeight: 'bold' }}>SDA</td>
                <td>I2C Data line with 400kHz speed.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 'bold', color: '#6366f1' }}>GPIO 2</td>
                <td style={{ fontWeight: 'bold' }}>SCL</td>
                <td>I2C Clock line.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 'bold', color: '#10b981' }}>3.3V</td>
                <td style={{ fontWeight: 'bold' }}>VIN</td>
                <td>Power supply input.</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 'bold', color: '#64748b' }}>GND</td>
                <td style={{ fontWeight: 'bold' }}>GND</td>
                <td>Common ground reference.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Method details card */}
      <div className="card" style={styles.card}>
        <div style={styles.cardHeader}>
          <Info size={24} style={{ color: '#6366f1' }} />
          <h3 style={styles.cardTitle}>Methodology Breakdown</h3>
        </div>
        <div style={styles.sections}>
          <div style={styles.sectionItem}>
            <h4 style={styles.subTitle}>1. PPG Signal Acquisition</h4>
            <p style={styles.sectionDesc}>
              The MAX30102 sensor shines red light (660nm) and infrared light (880nm) through the tissue. It measures the reflected light intensity, which corresponds to changes in blood volume. The signals are digitized at 400 Hz by the internal 18-bit ADC.
            </p>
          </div>
          
          <div style={styles.sectionItem}>
            <h4 style={styles.subTitle}>2. Waveform Conditioning</h4>
            <p style={styles.sectionDesc}>
              Spike removal filters high-frequency spikes. Inversion aligns peaks correctly. A lowpass Butterworth filter (cutoff 16Hz) removes power line noise. A highpass Butterworth filter (cutoff 0.5Hz) removes low-frequency breathing artifacts and baseline wanders.
            </p>
          </div>

          <div style={styles.sectionItem}>
            <h4 style={styles.subTitle}>3. Feature Extraction & Engineering</h4>
            <p style={styles.sectionDesc}>
              Features such as Shannon Entropy, Spectral Entropy, Pulse Width, BPM, HRV, Teager Energy Operator (TEO), derivative averages (VPG, SDPPG), and the Ensemble Ratio are calculated. Standardized ratios are computed between RED and IR wavelengths.
            </p>
          </div>

          <div style={styles.sectionItem}>
            <h4 style={styles.subTitle}>4. Estimation Model</h4>
            <p style={styles.sectionDesc}>
              The features are scaled using a RobustScaler (to mitigate noise deviations) and fed to an XGBoost regressor tree. The trained estimator maps these complex physiological metrics to a predicted blood glucose level (mg/dL).
            </p>
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
    gap: '2rem',
  },
  pageTitle: {
    fontSize: '2rem',
    fontWeight: '800',
    marginBottom: '0.25rem',
  },
  introText: {
    fontSize: '1.05rem',
    color: '#cbd5e1',
    maxWidth: '700px',
    marginBottom: '0.5rem',
  },
  card: {
    padding: '2rem',
    background: 'rgba(30, 41, 59, 0.35)',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    marginBottom: '1rem',
  },
  cardTitle: {
    fontSize: '1.25rem',
    fontWeight: '700',
    margin: 0,
  },
  sections: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  sectionItem: {
    background: 'rgba(15, 23, 42, 0.2)',
    padding: '1.25rem',
    borderRadius: '0.75rem',
    border: '1px solid rgba(255, 255, 255, 0.03)',
  },
  subTitle: {
    fontSize: '1.05rem',
    fontWeight: '600',
    color: '#38bdf8',
    marginBottom: '0.5rem',
  },
  sectionDesc: {
    fontSize: '0.9rem',
    color: '#cbd5e1',
    margin: 0,
    lineHeight: '1.6',
  }
};

export default AboutPage;
