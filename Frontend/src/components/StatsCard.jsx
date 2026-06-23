import React from 'react';

const StatsCard = ({ title, value, icon, change, changeType }) => {
  return (
    <div className="card" style={styles.card}>
      <div style={styles.header}>
        <span style={styles.title}>{title}</span>
        <div style={styles.iconContainer}>{icon}</div>
      </div>
      <div style={styles.body}>
        <span style={styles.value}>{value}</span>
        {change && (
          <span style={{ 
            ...styles.change, 
            color: changeType === 'positive' ? '#10b981' : changeType === 'negative' ? '#ef4444' : '#94a3b8' 
          }}>
            {change}
          </span>
        )}
      </div>
    </div>
  );
};

const styles = {
  card: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    minHeight: '120px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  title: {
    fontSize: '0.875rem',
    fontWeight: '600',
    color: '#94a3b8',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  iconContainer: {
    color: '#6366f1',
    background: 'rgba(99, 102, 241, 0.1)',
    padding: '0.5rem',
    borderRadius: '0.5rem',
  },
  body: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '0.5rem',
    marginTop: '1rem',
  },
  value: {
    fontSize: '2rem',
    fontWeight: '800',
    color: 'white',
    lineHeight: '1',
  },
  change: {
    fontSize: '0.75rem',
    fontWeight: '600',
  }
};

export default StatsCard;
