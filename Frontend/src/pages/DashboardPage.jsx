import React, { useState, useEffect } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ScatterChart, Scatter, Label, Legend } from 'recharts';
import { Users, Activity, TrendingUp, RefreshCw, BarChart2, CheckCircle2, AlertTriangle } from 'lucide-react';
import StatsCard from '../components/StatsCard';
import SubjectTable from '../components/SubjectTable';
import { dashboardService, subjectService, predictionService } from '../services/api';

const DashboardPage = () => {
  const [stats, setStats] = useState(null);
  const [distribution, setDistribution] = useState([]);
  const [predVsActual, setPredVsActual] = useState([]);
  const [featImportance, setFeatImportance] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [sData, distData, pvaData, featData, subjList, predList] = await Promise.all([
        dashboardService.getStats(),
        dashboardService.getDistribution(),
        dashboardService.getPredictedVsActual(),
        dashboardService.getFeatureImportance(),
        subjectService.getSubjects(),
        predictionService.getPredictions()
      ]);
      
      setStats(sData);
      setDistribution(distData);
      setPredVsActual(pvaData);
      setFeatImportance(featData.slice(0, 10)); // Top 10 features
      setSubjects(subjList);
      setPredictions(predList);
    } catch (e) {
      console.error("Error loading dashboard details:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleRetrain = async () => {
    try {
      setRetraining(true);
      setRetrainResult(null);
      const res = await dashboardService.retrainModel();
      setRetrainResult({
        success: res.success,
        message: res.message,
        metrics: res.metrics
      });
      // Refresh dashboard stats after retraining
      fetchDashboardData();
    } catch (e) {
      setRetrainResult({
        success: false,
        message: e.response?.data?.detail || "Retraining failed due to insufficient training records."
      });
    } finally {
      setRetraining(false);
    }
  };

  const handleDeleteSubject = async (id) => {
    if (window.confirm("Are you sure you want to delete this subject? All their readings and predictions will be deleted.")) {
      try {
        await subjectService.deleteSubject(id);
        fetchDashboardData(); // Reload
      } catch (e) {
        alert("Failed to delete subject.");
      }
    }
  };

  if (loading) {
    return (
      <div style={styles.loadingContainer}>
        <div className="spinner"></div>
        <p style={{ marginTop: '1rem', color: '#64748b' }}>Syncing dashboard data...</p>
      </div>
    );
  }

  // Pre-calculated diagonal line (identity line y=x) for predicted vs actual chart
  const identityLine = [
    { x: 60, y: 60 },
    { x: 180, y: 180 }
  ];

  return (
    <div style={styles.container}>
      <h1 style={styles.pageTitle}>System Dashboard</h1>
      
      {/* Stats row */}
      <div className="grid grid-4" style={styles.statsRow}>
        <StatsCard 
          title="Total Subjects" 
          value={stats?.total_subjects || 0} 
          icon={<Users size={20} />} 
        />
        <StatsCard 
          title="Total Predictions" 
          value={stats?.total_predictions || 0} 
          icon={<Activity size={20} />} 
        />
        <StatsCard 
          title="Avg Predicted Glucose" 
          value={stats?.average_predicted_glucose ? `${stats.average_predicted_glucose.toFixed(1)}` : 'N/A'} 
          icon={<TrendingUp size={20} />} 
          change="mg/dL"
          changeType="neutral"
        />
        <StatsCard 
          title="Model R² Score" 
          value={stats?.model_r2 ? `${stats.model_r2.toFixed(3)}` : '0.175'} 
          icon={<BarChart2 size={20} />} 
          change="XGBoost"
          changeType="positive"
        />
      </div>

      {/* Retrain model card */}
      <div className="card" style={styles.retrainCard}>
        <div style={styles.retrainInfo}>
          <h3 style={styles.retrainTitle}>Model Pipeline Calibration</h3>
          <p style={styles.retrainDesc}>
            Retrain the XGBoost regressor model with all subject records currently saved in the database. A minimum of 10 subjects with reference values is required.
          </p>
        </div>
        <button 
          onClick={handleRetrain} 
          disabled={retraining} 
          className="btn btn-primary"
          style={styles.retrainBtn}
        >
          {retraining ? (
            <>
              <div className="spinner" style={{ width: '16px', height: '16px' }}></div>
              Retraining...
            </>
          ) : (
            <>
              <RefreshCw size={16} />
              Retrain Model
            </>
          )}
        </button>
      </div>

      {/* Retraining Feedback alerts */}
      {retrainResult && (
        <div style={{
          ...styles.alert,
          borderColor: retrainResult.success ? '#10b981' : '#ef4444',
          background: retrainResult.success ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)'
        }}>
          {retrainResult.success ? (
            <CheckCircle2 size={20} style={{ color: '#10b981', flexShrink: 0 }} />
          ) : (
            <AlertTriangle size={20} style={{ color: '#ef4444', flexShrink: 0 }} />
          )}
          <div>
            <h4 style={{ margin: 0, fontWeight: 'bold' }}>
              {retrainResult.success ? 'Model Retraining Succeeded' : 'Model Retraining Failed'}
            </h4>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#cbd5e1' }}>
              {retrainResult.message}
            </p>
            {retrainResult.metrics && (
              <div style={styles.metricsGrid}>
                <span><strong>MAE:</strong> {retrainResult.metrics.mae.toFixed(3)} mg/dL</span>
                <span><strong>RMSE:</strong> {retrainResult.metrics.rmse.toFixed(3)} mg/dL</span>
                <span><strong>R²:</strong> {retrainResult.metrics.r2.toFixed(3)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-2" style={styles.chartsGrid}>
        {/* Glucose distribution histogram */}
        <div className="card" style={styles.chartCard}>
          <h3 style={styles.chartTitle}>Glucose Classification Distribution</h3>
          <p style={styles.chartSubtitle}>Frequency counts across clinical diagnostics bounds.</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={distribution} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="range_label" stroke="#64748b" fontSize={10} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
              <Tooltip 
                contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ color: 'white' }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importance horizontal chart */}
        <div className="card" style={styles.chartCard}>
          <h3 style={styles.chartTitle}>XGBoost Feature Importance (Top 10)</h3>
          <p style={styles.chartSubtitle}>Relative feature contributions to glucose estimation.</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart 
              data={featImportance} 
              layout="vertical"
              margin={{ top: 10, right: 10, left: 10, bottom: 0 }}
            >
              <XAxis type="number" stroke="#64748b" fontSize={11} />
              <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={10} width={130} tickLine={false} />
              <Tooltip 
                contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ color: 'white' }}
              />
              <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Predicted vs Actual scatter plot */}
        <div className="card" style={{ ...styles.chartCard, gridColumn: 'span 2' }}>
          <h3 style={styles.chartTitle}>Predicted vs Actual Glucose Levels</h3>
          <p style={styles.chartSubtitle}>Plot of estimations vs clinical measurements. Diagonal line represents perfect alignment (Y=X).</p>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <XAxis type="number" dataKey="actual" name="Actual Glucose" unit=" mg/dL" domain={[60, 185]} stroke="#64748b" fontSize={11}>
                <Label value="Actual Glucose (mg/dL)" offset={-5} position="insideBottom" fill="#94a3b8" fontSize={11} />
              </XAxis>
              <YAxis type="number" dataKey="predicted" name="Predicted Glucose" unit=" mg/dL" domain={[60, 185]} stroke="#64748b" fontSize={11}>
                <Label value="Predicted Glucose (mg/dL)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} fill="#94a3b8" fontSize={11} />
              </YAxis>
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }} 
                contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ color: 'white' }}
              />
              <Scatter name="Subjects" data={predVsActual} fill="#06b6d4" />
              {/* Identity line Y=X */}
              <Scatter name="Ideal Line" data={identityLine} line={{ stroke: '#64748b', strokeWidth: 1, strokeDasharray: '4 4' }} shape={() => null} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Subjects management table */}
      <div style={styles.tableSection}>
        <h3 style={styles.chartTitle}>Subject Records</h3>
        <p style={styles.chartSubtitle}>Manage all subjects and clinical actual readings currently stored in the system.</p>
        <SubjectTable subjects={subjects} onDelete={handleDeleteSubject} />
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
    marginBottom: '0.5rem',
  },
  statsRow: {
    marginTop: '0.5rem',
  },
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '400px',
  },
  retrainCard: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '1.5rem',
    background: 'rgba(99, 102, 241, 0.05)',
    borderColor: 'rgba(99, 102, 241, 0.2)',
  },
  retrainInfo: {
    flex: '1',
    minWidth: '280px',
  },
  retrainTitle: {
    fontSize: '1.1rem',
    fontWeight: '700',
    marginBottom: '0.25rem',
    color: '#a5b4fc',
  },
  retrainDesc: {
    fontSize: '0.85rem',
    margin: 0,
    color: '#cbd5e1',
  },
  retrainBtn: {
    padding: '0.75rem 1.5rem',
    gap: '0.5rem',
    flexShrink: 0,
  },
  alert: {
    border: '1px solid',
    borderRadius: '0.75rem',
    padding: '1rem 1.25rem',
    display: 'flex',
    gap: '0.75rem',
    alignItems: 'flex-start',
  },
  metricsGrid: {
    display: 'flex',
    gap: '1.5rem',
    fontSize: '0.8rem',
    marginTop: '0.5rem',
    color: '#a5b4fc',
  },
  chartsGrid: {
    marginTop: '0.5rem',
  },
  chartCard: {
    padding: '1.5rem',
    background: 'rgba(30, 41, 59, 0.35)',
  },
  chartTitle: {
    fontSize: '1.1rem',
    fontWeight: '700',
    margin: 0,
  },
  chartSubtitle: {
    fontSize: '0.8rem',
    color: '#64748b',
    marginBottom: '1.5rem',
  },
  tableSection: {
    marginTop: '1rem',
  }
};

export default DashboardPage;
