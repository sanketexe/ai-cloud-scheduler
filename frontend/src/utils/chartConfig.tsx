/**
 * Chart Configuration Utility
 * 
 * Provides consistent styling, labels, legends, and tooltips for all charts
 * across the dashboard to meet requirement 6.5
 */

import numeral from 'numeral';

// Chart color palette
export const CHART_COLORS = {
  primary: '#2196f3',
  secondary: '#f50057',
  success: '#4caf50',
  warning: '#ff9800',
  error: '#f44336',
  info: '#00bcd4',
  purple: '#9c27b0',
  teal: '#009688',
  indigo: '#3f51b5',
  pink: '#e91e63',
};

export const CHART_COLOR_ARRAY = [
  CHART_COLORS.primary,
  CHART_COLORS.success,
  CHART_COLORS.warning,
  CHART_COLORS.error,
  CHART_COLORS.info,
  CHART_COLORS.purple,
  CHART_COLORS.teal,
  CHART_COLORS.indigo,
  CHART_COLORS.pink,
  CHART_COLORS.secondary,
];

// Common tooltip styling
export const TOOLTIP_STYLE = {
  backgroundColor: '#1a1d3a',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '8px',
  padding: '12px',
};

// Common axis styling
export const AXIS_STYLE = {
  stroke: '#b0bec5',
  fontSize: 12,
};

// Common grid styling
export const GRID_STYLE = {
  strokeDasharray: '3 3',
  stroke: 'rgba(255,255,255,0.1)',
};

// Currency formatter
export const formatCurrency = (value: number): string => {
  return numeral(value).format('$0,0');
};

// Compact currency formatter (for axis labels)
export const formatCurrencyCompact = (value: number): string => {
  return numeral(value).format('$0a');
};

// Percentage formatter
export const formatPercentage = (value: number): string => {
  return `${value.toFixed(1)}%`;
};

// Number formatter
export const formatNumber = (value: number): string => {
  return numeral(value).format('0,0');
};

/**
 * Custom tooltip component for currency values
 */
export const CurrencyTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div style={TOOLTIP_STYLE}>
        <p style={{ margin: 0, marginBottom: 8, fontWeight: 600, color: '#fff' }}>
          {label}
        </p>
        {payload.map((entry: any, index: number) => (
          <p
            key={index}
            style={{
              margin: 0,
              marginBottom: 4,
              color: entry.color,
              fontSize: 14,
            }}
          >
            <span style={{ fontWeight: 600 }}>{entry.name}:</span>{' '}
            {formatCurrency(entry.value)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

/**
 * Custom tooltip component for percentage values
 */
export const PercentageTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div style={TOOLTIP_STYLE}>
        <p style={{ margin: 0, marginBottom: 8, fontWeight: 600, color: '#fff' }}>
          {label}
        </p>
        {payload.map((entry: any, index: number) => (
          <p
            key={index}
            style={{
              margin: 0,
              marginBottom: 4,
              color: entry.color,
              fontSize: 14,
            }}
          >
            <span style={{ fontWeight: 600 }}>{entry.name}:</span>{' '}
            {formatPercentage(entry.value)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

/**
 * Custom tooltip component for pie charts with currency
 */
export const PieChartCurrencyTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0];
    return (
      <div style={TOOLTIP_STYLE}>
        <p style={{ margin: 0, marginBottom: 4, fontWeight: 600, color: '#fff' }}>
          {data.name}
        </p>
        <p style={{ margin: 0, color: data.payload.fill, fontSize: 14 }}>
          <span style={{ fontWeight: 600 }}>Cost:</span> {formatCurrency(data.payload.cost || data.value)}
        </p>
        <p style={{ margin: 0, color: '#b0bec5', fontSize: 12 }}>
          {formatPercentage(data.payload.value || data.percent * 100)}
        </p>
      </div>
    );
  }
  return null;
};

/**
 * Legend configuration for multi-series charts
 */
export const LEGEND_CONFIG = {
  wrapperStyle: {
    paddingTop: '20px',
  },
  iconType: 'circle' as const,
  iconSize: 10,
};

/**
 * Responsive container default props
 */
export const RESPONSIVE_CONTAINER_PROPS = {
  width: '100%' as const,
  height: 350,
};

/**
 * Get color by index from the color array
 */
export const getChartColor = (index: number): string => {
  return CHART_COLOR_ARRAY[index % CHART_COLOR_ARRAY.length];
};

/**
 * Generate label for pie chart slices
 */
export const renderPieLabel = (entry: any): string => {
  const percentage = entry.percent ? (entry.percent * 100).toFixed(0) : entry.value;
  return `${entry.name}: ${percentage}%`;
};

/**
 * Custom label component for pie charts with better visibility
 */
export const CustomPieLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }: {
  cx: number;
  cy: number;
  midAngle: number;
  innerRadius: number;
  outerRadius: number;
  percent: number;
  name: string;
}) => {
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  if (percent < 0.05) return null; // Don't show label for slices < 5%

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      style={{ fontSize: 12, fontWeight: 600 }}
    >
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

/**
 * Hover effect configuration for interactive charts
 */
export const HOVER_CONFIG = {
  cursor: 'pointer',
  activeDot: { r: 8, strokeWidth: 2, stroke: '#fff' },
};
