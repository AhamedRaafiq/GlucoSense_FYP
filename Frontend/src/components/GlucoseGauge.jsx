import React from 'react';

const GlucoseGauge = ({ value }) => {
  // Glucose range config
  const getRangeDetails = (val) => {
    if (val < 70) return { label: 'Hypoglycemic', color: '#ef4444', text: 'Low Blood Glucose' };
    if (val <= 100) return { label: 'Normal', color: '#10b981', text: 'Healthy Blood Glucose' };
    if (val <= 125) return { label: 'Pre-diabetic', color: '#f59e0b', text: 'Elevated Blood Glucose' };
    if (val <= 180) return { label: 'Diabetic', color: '#ef4444', text: 'High Blood Glucose' };
    return { label: 'Hyperglycemic', color: '#b91c1c', text: 'Very High Blood Glucose' };
  };

  const details = getRangeDetails(value);
  
  // SVG gauge constants
  const size = 200;
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  
  // Map value (40 to 240) to percentage of gauge (0 to 1)
  const minVal = 40;
  const maxVal = 240;
  const valNormalized = Math.min(Math.max(value, minVal), maxVal);
  const percentage = (valNormalized - minVal) / (maxVal - minVal);
  
  const strokeDashoffset = circumference - (percentage * circumference);

  return (
    <div style={styles.container}>
      <div style={styles.gaugeWrapper}>
        <svg width={size} height={size} style={styles.svg}>
          {/* Background circle */}
          <circle
            className="gauge-bg"
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth={strokeWidth}
          />
          {/* Progress circle */}
          <circle
            className="gauge-progress"
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke={details.color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{
              transform: 'rotate(-90deg)',
              transformOrigin: '50% 50%',
              transition: 'stroke-dashoffset 1s ease-out, stroke 1s ease',
              filter: `drop-shadow(0 0 6px ${details.color})`
            }}
          />
        </svg>
        <div style={styles.content}>
          <span style={styles.value}>{value.toFixed(1)}</span>
          <span style={styles.unit}>mg/dL</span>
        </div>
      </div>
      <div style={styles.info}>
        <span style={{ ...styles.badge, backgroundColor: `${details.color}18`, color: details.color }}>
          {details.label}
        </span>
        <p style={styles.desc}>{details.text}</p>
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '1.5rem',
  },
  gaugeWrapper: {
    position: 'relative',
    width: '200px',
    height: '200px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  svg: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    zIndex: 2,
  },
  value: {
    fontSize: '2.5rem',
    fontWeight: '800',
    color: 'white',
    lineHeight: '1',
  },
  unit: {
    fontSize: '0.8rem',
    color: '#64748b',
    fontWeight: '600',
    marginTop: '0.25rem',
  },
  info: {
    textAlign: 'center',
  },
  badge: {
    display: 'inline-block',
    padding: '0.35rem 1rem',
    fontSize: '0.85rem',
    fontWeight: '700',
    borderRadius: '9999px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: '0.5rem',
  },
  desc: {
    fontSize: '0.9rem',
    color: '#94a3b8',
    margin: 0,
  }
};

export default GlucoseGauge;
