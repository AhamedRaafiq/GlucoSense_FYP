import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, LayoutDashboard, Database, Info } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path ? 'active-nav-link' : '';
  };

  return (
    <nav style={styles.nav}>
      <div style={styles.navContainer}>
        <Link to="/" style={styles.logo}>
          <Activity size={24} style={styles.logoIcon} />
          <span style={styles.logoText}>Antigravity PPG</span>
        </Link>
        
        <div style={styles.navLinks}>
          <Link to="/dashboard" style={{ ...styles.link, ...styles[isActive('/dashboard')] }}>
            <LayoutDashboard size={18} />
            <span>Dashboard</span>
          </Link>
          <Link to="/collect" style={{ ...styles.link, ...styles[isActive('/collect')] }}>
            <Database size={18} />
            <span>Data Collection</span>
          </Link>
          <Link to="/about" style={{ ...styles.link, ...styles[isActive('/about')] }}>
            <Info size={18} />
            <span>About</span>
          </Link>
        </div>
      </div>
    </nav>
  );
};

const styles = {
  nav: {
    background: 'rgba(15, 23, 42, 0.8)',
    backdropFilter: 'blur(12px)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
    position: 'sticky',
    top: 0,
    zIndex: 100,
    padding: '1rem 2rem',
  },
  navContainer: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    textDecoration: 'none',
  },
  logoIcon: {
    color: '#06b6d4',
  },
  logoText: {
    fontWeight: 800,
    fontSize: '1.25rem',
    background: 'linear-gradient(135deg, #f8fafc 0%, #cbd5e1 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    letterSpacing: '-0.03em',
  },
  navLinks: {
    display: 'flex',
    gap: '2rem',
  },
  link: {
    color: '#94a3b8',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.9rem',
    fontWeight: 500,
    transition: 'color 0.2s',
  },
  'active-nav-link': {
    color: '#6366f1',
    fontWeight: 600,
  },
};

export default Navbar;
