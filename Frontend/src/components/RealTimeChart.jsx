import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const RealTimeChart = ({ data }) => {
  // Keep only last 150 points for performance
  const chartData = data.slice(-150);

  // Custom tool tip component
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div style={styles.tooltip}>
          <p style={styles.tooltipTime}>{`Time: ${payload[0].payload.timestamp.toFixed(2)}s`}</p>
          <p style={{ ...styles.tooltipValue, color: '#38bdf8' }}>
            {`IR: ${payload[0].value.toFixed(0)}`}
          </p>
          {payload[1] && (
            <p style={{ ...styles.tooltipValue, color: '#ef4444' }}>
              {`RED: ${payload[1].value.toFixed(0)}`}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div style={styles.container}>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={(tick) => `${tick.toFixed(1)}s`}
            stroke="#64748b" 
            fontSize={11}
          />
          <YAxis 
            yAxisId="ir" 
            domain={['auto', 'auto']} 
            stroke="#38bdf8" 
            fontSize={11}
          />
          <YAxis 
            yAxisId="red" 
            orientation="right" 
            domain={['auto', 'auto']} 
            stroke="#ef4444" 
            fontSize={11}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line 
            yAxisId="ir"
            type="monotone" 
            dataKey="ir_value" 
            stroke="#06b6d4" 
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line 
            yAxisId="red"
            type="monotone" 
            dataKey="red_value" 
            stroke="#ef4444" 
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const styles = {
  container: {
    width: '100%',
    background: 'rgba(15, 23, 42, 0.4)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '0.75rem',
    padding: '1.25rem 0.5rem 0.5rem 0.5rem',
  },
  tooltip: {
    background: 'rgba(15, 23, 42, 0.95)',
    border: '1px solid rgba(255, 255, 255, 0.15)',
    borderRadius: '0.5rem',
    padding: '0.75rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
  },
  tooltipTime: {
    color: '#94a3b8',
    fontSize: '0.75rem',
    fontWeight: 'bold',
    marginBottom: '0.25rem',
  },
  tooltipValue: {
    fontSize: '0.85rem',
    fontWeight: '600',
    margin: 0,
  }
};

export default RealTimeChart;
