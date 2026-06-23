import React from 'react';
import { Heart } from 'lucide-react';

const Footer = () => {
  return (
    <footer style={styles.footer}>
      <div style={styles.container}>
        <p style={styles.text}>
          © 2026 Non-Invasive Glucose Prediction System. Built for FYP.
        </p>
        <p style={styles.credit}>
          Made with <Heart size={14} style={styles.heart} /> and Advanced AI.
        </p>
      </div>
    </footer>
  );
};

const styles = {
  footer: {
    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
    padding: '1.5rem 2rem',
    background: '#0f172a',
    marginTop: 'auto',
  },
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '1rem',
  },
  text: {
    color: '#64748b',
    fontSize: '0.85rem',
    margin: 0,
  },
  credit: {
    color: '#64748b',
    fontSize: '0.85rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem',
    margin: 0,
  },
  heart: {
    color: '#ef4444',
  },
};

export default Footer;
