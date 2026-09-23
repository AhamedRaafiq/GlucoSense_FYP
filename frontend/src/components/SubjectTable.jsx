import React from 'react';
import { Trash2, Calendar, User } from 'lucide-react';

const SubjectTable = ({ subjects, onDelete }) => {
  const formatDate = (dateStr) => {
    const d = new Date(dateStr);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  };

  return (
    <div className="table-container">
      {subjects.length === 0 ? (
        <div style={styles.empty}>
          <User size={32} style={{ color: '#64748b', marginBottom: '0.5rem' }} />
          <p style={{ margin: 0, color: '#64748b' }}>No subjects added yet.</p>
        </div>
      ) : (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Age</th>
              <th>Gender</th>
              <th>Actual Glucose</th>
              <th>Created Date</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {subjects.map((subj) => (
              <tr key={subj.id}>
                <td style={{ fontWeight: 'bold', color: '#6366f1' }}>#{subj.id}</td>
                <td style={{ fontWeight: '500' }}>{subj.name}</td>
                <td>{subj.age || 'N/A'}</td>
                <td>{subj.gender || 'N/A'}</td>
                <td style={{ fontWeight: '600' }}>
                  {subj.glucose_level ? `${subj.glucose_level} mg/dL` : (
                    <span style={{ color: '#64748b', fontWeight: 'normal', fontSize: '0.8rem' }}>Unspecified</span>
                  )}
                </td>
                <td style={{ color: '#94a3b8' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                    <Calendar size={14} />
                    {formatDate(subj.created_at)}
                  </div>
                </td>
                <td>
                  <button 
                    onClick={() => onDelete(subj.id)}
                    style={styles.deleteBtn}
                    title="Delete Subject"
                  >
                    <Trash2 size={16} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

const styles = {
  empty: {
    padding: '3rem',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
  },
  deleteBtn: {
    background: 'transparent',
    border: 'none',
    color: '#64748b',
    cursor: 'pointer',
    padding: '0.35rem',
    borderRadius: '0.375rem',
    transition: 'all 0.2s',
    display: 'flex',
    alignItems: 'center',
  },
  // We can add simple hover styles inside components if we wish
};

// Add raw CSS styling specifically for this component to change trash button color on hover
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.innerHTML = `
    button[title="Delete Subject"]:hover {
      background-color: rgba(239, 68, 68, 0.15) !important;
      color: #ef4444 !important;
    }
  `;
  document.head.appendChild(style);
}

export default SubjectTable;
