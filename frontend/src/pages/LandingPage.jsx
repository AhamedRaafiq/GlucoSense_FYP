import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Cpu, Activity, Award, ShieldAlert, BarChart3 } from 'lucide-react';
import PipelineFlow from '../components/PipelineFlow';

const LandingPage = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationId;
    let width = canvas.width = canvas.offsetWidth;
    let height = canvas.height = canvas.offsetHeight;
    
    let t = 0;
    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      
      // Simulating a double peak PPG wave
      for (let x = 0; x < width; x++) {
        const xOffset = x * 0.02 + t;
        const hr = math_ppg_wave(xOffset);
        const y = height / 2 + hr * (height * 0.3);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      t += 0.05;
      animationId = requestAnimationFrame(draw);
    };

    const math_ppg_wave = (x) => {
      // Combination of sines to simulate a pulse
      return Math.sin(x) * 0.6 + Math.sin(x * 2) * 0.25 + Math.sin(x * 4) * 0.08;
    };

    draw();

    const handleResize = () => {
      if (canvas) {
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div style={styles.container}>
      <div className="hero-glow"></div>
      
      {/* Hero Section */}
      <section style={styles.hero}>
        <div style={styles.heroContent}>
          <div style={styles.badgeContainer}>
            <span style={styles.heroBadge}>
              <Activity size={12} style={{ color: '#06b6d4' }} />
              State of the Art Non-Invasive Diagnostics
            </span>
          </div>
          <h1 style={styles.heroTitle}>
            Predict Diabetes using <span className="gradient-text">PPG Signals</span>
          </h1>
          <p style={styles.heroText}>
            A research-backed pipeline that processes dual-wavelength light absorption from a MAX30102 sensor, extracts physiological features, and estimates blood glucose levels non-invasively using XGBoost machine learning.
          </p>
          <div style={styles.ctaGroup}>
            <Link to="/collect" className="btn btn-primary" style={styles.ctaBtn}>
              Start Collection
              <ArrowRight size={16} />
            </Link>
            <Link to="/dashboard" className="btn btn-secondary" style={styles.ctaBtn}>
              View Dashboard
            </Link>
          </div>
        </div>
        <div style={styles.heroVisual}>
          <div style={styles.waveLabel}>Live Signal Simulator</div>
          <canvas ref={canvasRef} style={styles.canvas}></canvas>
        </div>
      </section>

      {/* Features Overview */}
      <section style={styles.features}>
        <h2 style={styles.sectionTitle}>Key Advantages</h2>
        <div className="grid grid-3">
          <div className="card" style={styles.featCard}>
            <Award size={32} style={{ color: '#10b981', marginBottom: '1rem' }} />
            <h3 style={styles.featTitle}>100% Pain-Free</h3>
            <p style={{ margin: 0, fontSize: '0.9rem' }}>
              Eliminates the discomfort of daily finger-pricking. Uses optical sensors to measure blood volume variations.
            </p>
          </div>
          <div className="card" style={styles.featCard}>
            <Cpu size={32} style={{ color: '#06b6d4', marginBottom: '1rem' }} />
            <h3 style={styles.featTitle}>Edge Processing</h3>
            <p style={{ margin: 0, fontSize: '0.9rem' }}>
              Interfaced with an ESP32-S3 microcontroller running real-time signal logging for robust diagnostic pipelines.
            </p>
          </div>
          <div className="card" style={styles.featCard}>
            <BarChart3 size={32} style={{ color: '#6366f1', marginBottom: '1rem' }} />
            <h3 style={styles.featTitle}>XGBoost Estimator</h3>
            <p style={{ margin: 0, fontSize: '0.9rem' }}>
              Utilizes a gradient boosting regressor trained on 15 custom features to guarantee optimal blood glucose mapping.
            </p>
          </div>
        </div>
      </section>

      {/* Interactive System Pipeline Architecture */}
      <section style={styles.architecture}>
        <div style={styles.sectionHeader}>
          <h2 style={styles.sectionTitle}>System Pipeline Architecture</h2>
          <p style={styles.sectionSubtitle}>
            Interactive layout of the 12 pipeline stages, from raw MAX30102 photodiode reads to the XGBoost final glucose estimation.
          </p>
        </div>
        <PipelineFlow />
      </section>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5rem',
    position: 'relative',
  },
  hero: {
    display: 'flex',
    alignItems: 'center',
    gap: '3rem',
    padding: '3rem 0',
    flexWrap: 'wrap',
  },
  heroContent: {
    flex: '1.2',
    minWidth: '320px',
  },
  badgeContainer: {
    marginBottom: '1rem',
  },
  heroBadge: {
    background: 'rgba(6, 182, 212, 0.08)',
    border: '1px solid rgba(6, 182, 212, 0.2)',
    color: '#06b6d4',
    padding: '0.4rem 1rem',
    borderRadius: '9999px',
    fontSize: '0.75rem',
    fontWeight: '600',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.5rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  heroTitle: {
    fontSize: '3.2rem',
    lineHeight: '1.15',
    fontWeight: '800',
    marginBottom: '1.5rem',
  },
  heroText: {
    fontSize: '1.1rem',
    lineHeight: '1.6',
    marginBottom: '2.5rem',
  },
  ctaGroup: {
    display: 'flex',
    gap: '1rem',
    flexWrap: 'wrap',
  },
  ctaBtn: {
    padding: '0.8rem 1.75rem',
    fontSize: '0.95rem',
  },
  heroVisual: {
    flex: '0.8',
    minWidth: '320px',
    background: 'rgba(30, 41, 59, 0.4)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '1.25rem',
    padding: '1.5rem',
    position: 'relative',
    height: '240px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
  },
  waveLabel: {
    position: 'absolute',
    top: '1rem',
    left: '1.5rem',
    fontSize: '0.75rem',
    fontWeight: 'bold',
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  canvas: {
    width: '100%',
    height: '140px',
  },
  features: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2rem',
  },
  sectionTitle: {
    fontSize: '2rem',
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: '0.5rem',
  },
  featCard: {
    textAlign: 'center',
    padding: '2.5rem 1.5rem',
  },
  featTitle: {
    fontSize: '1.2rem',
    fontWeight: '700',
    marginBottom: '0.75rem',
  },
  architecture: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2.5rem',
  },
  sectionHeader: {
    textAlign: 'center',
  },
  sectionSubtitle: {
    maxWidth: '600px',
    margin: '0 auto',
    fontSize: '0.95rem',
  }
};

export default LandingPage;
