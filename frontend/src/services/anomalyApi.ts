// API service for AI-Powered Cost Anomaly Detection
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Add request interceptor for authentication
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Anomaly Detection Types
export interface AnomalyConfiguration {
  sensitivity_level: 'conservative' | 'balanced' | 'aggressive';
  threshold_percentage: number;
  baseline_period_days: number;
  min_cost_threshold: number;
  excluded_services: string[];
  maintenance_windows: Array<{
    start: string;
    end: string;
  }>;
  notification_channels: string[];
  escalation_rules: {
    high_severity_threshold?: number;
    escalation_delay_minutes?: number;
    escalation_channels?: string[];
  };
}

export interface AnomalyDetectionRequest {
  account_id: string;
  time_range: {
    start_date: string;
    end_date: string;
  };
  services?: string[];
  regions?: string[];
  cost_threshold?: number;
  include_forecasts?: boolean;
}

export interface Anomaly {
  anomaly_id: string;
  account_id: string;
  detection_timestamp: string;
  severity: 'low' | 'medium' | 'high';
  confidence_score: number;
  anomaly_score: number;
  estimated_impact_usd: number;
  affected_services: string[];
  affected_regions: string[];
  description: string;
  root_cause: string;
}

export interface AnomalyDetails extends Anomaly {
  root_cause_analysis: {
    primary_cause: string;
    contributing_factors: string[];
    affected_resources: string[];
  };
  time_series_data: Array<{
    timestamp: string;
    cost: number;
    baseline: number;
  }>;
  recommendations: string[];
  similar_anomalies: Array<{
    anomaly_id: string;
    similarity_score: number;
    date: string;
  }>;
}

export interface ForecastRequest {
  account_id: string;
  forecast_horizon_days: number;
  confidence_level: number;
  include_seasonality: boolean;
  services?: string[];
  granularity: 'daily' | 'weekly' | 'monthly';
}

export interface Forecast {
  date: string;
  predicted_cost: number;
  confidence_level: number;
  services: string[];
  granularity: string;
  factors: {
    trend: number;
    seasonality: number;
    baseline: number;
  };
}

export interface ForecastResponse {
  forecasts: Forecast[];
  confidence_intervals: {
    upper_bound: number[];
    lower_bound: number[];
    confidence_level: number;
  };
  metadata: {
    account_id: string;
    forecast_generated_at: string;
    horizon_days: number;
    granularity: string;
    services_included: string[];
    seasonality_included: boolean;
    model_version: string;
    forecast_accuracy_estimate: number;
  };
}

export interface ModelPerformance {
  model_id: string;
  time_range: {
    start_date: string;
    end_date: string;
  };
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    false_positive_rate: number;
    false_negative_rate: number;
    detection_latency_ms: number;
  };
  resource_usage: {
    average_memory_mb: number;
    average_cpu_percent: number;
    average_prediction_time_ms: number;
  };
  business_impact: {
    total_predictions: number;
    anomalies_detected: number;
    alerts_generated: number;
    estimated_cost_savings_usd: number;
  };
  trend_analysis: {
    accuracy_trend: string;
    performance_degradation: boolean;
    recommendation: string;
  };
}

export interface ModelInfo {
  model_id: string;
  model_name: string;
  model_type: string;
  status: string;
  environment: string;
  accuracy: number;
  last_updated: string;
}

export interface DriftAnalysis {
  model_id: string;
  drift_analysis: {
    overall_drift_score: number;
    drift_threshold: number;
    drift_detected: boolean;
    drift_type: string;
    confidence_level: number;
    analysis_timestamp: string;
    feature_drift: {
      [feature: string]: {
        drift_score: number;
        significant: boolean;
      };
    };
    recommendations: string[];
    historical_drift: Array<{
      date: string;
      drift_score: number;
    }>;
  };
}

export interface Alert {
  alert_id: string;
  account_id: string;
  anomaly_id: string;
  severity: 'low' | 'medium' | 'high';
  status: 'active' | 'acknowledged' | 'snoozed' | 'resolved';
  title: string;
  description: string;
  created_at: string;
  estimated_impact_usd: number;
  affected_services: string[];
  notification_channels: string[];
  acknowledged_by?: string;
  acknowledged_at?: string;
}

export interface AuditEvent {
  event_id: string;
  event_type: string;
  timestamp: string;
  description: string;
  resource_id?: string;
  user_id: string;
  metadata: any;
  source: string;
}

export interface SystemStatus {
  system_status: {
    overall_status: string;
    last_updated: string;
    uptime_percentage: number;
  };
  statistics: {
    total_accounts_monitored: number;
    anomalies_detected_24h: number;
    alerts_generated_24h: number;
    forecasts_generated_24h: number;
    api_requests_24h: number;
    average_response_time_ms: number;
  };
  model_status: {
    total_models: number;
    deployed_models: number;
    models_with_drift: number;
    average_accuracy: number;
  };
  performance_metrics: {
    cpu_usage_percent: number;
    memory_usage_percent: number;
    disk_usage_percent: number;
    network_throughput_mbps: number;
  };
}

// Anomaly Detection API Service
export const anomalyApiService = {
  // Configuration Management
  async getConfiguration(accountId: string): Promise<{ account_id: string; configuration: AnomalyConfiguration; status: string }> {
    const response = await api.get('/api/v1/config', {
      params: { account_id: accountId }
    });
    return response.data;
  },

  async updateConfiguration(accountId: string, config: AnomalyConfiguration): Promise<{ account_id: string; configuration: AnomalyConfiguration; status: string; message: string }> {
    const response = await api.post('/api/v1/config', config, {
      params: { account_id: accountId }
    });
    return response.data;
  },

  // Anomaly Detection
  async detectAnomalies(request: AnomalyDetectionRequest): Promise<{
    anomalies: Anomaly[];
    summary: {
      total_anomalies: number;
      high_severity: number;
      medium_severity: number;
      low_severity: number;
      total_estimated_impact: number;
      detection_time_ms: number;
      model_confidence: number;
    };
    metadata: any;
  }> {
    const response = await api.post('/api/v1/anomalies/detect', request);
    return response.data;
  },

  async getAnomalies(params: {
    account_id: string;
    start_date: string;
    end_date: string;
    severity?: string;
    limit?: number;
  }): Promise<{
    anomalies: Anomaly[];
    total_count: number;
    filters: any;
    metadata: any;
  }> {
    const response = await api.get('/api/v1/anomalies', { params });
    return response.data;
  },

  async getAnomalyDetails(anomalyId: string): Promise<AnomalyDetails> {
    const response = await api.get(`/api/v1/anomalies/${anomalyId}`);
    return response.data;
  },

  // Forecasting
  async generateForecast(request: ForecastRequest): Promise<ForecastResponse> {
    const response = await api.post('/api/v1/forecasts/generate', request);
    return response.data;
  },

  async getForecasts(params: {
    account_id: string;
    horizon_days: number;
    services?: string;
  }): Promise<{
    forecasts: Forecast[];
    account_id: string;
    horizon_days: number;
    services: string[];
    generated_at: string;
    cache_hit: boolean;
  }> {
    const response = await api.get('/api/v1/forecasts', { params });
    return response.data;
  },

  // Model Performance
  async getModelPerformance(params: {
    model_id?: string;
    start_date: string;
    end_date: string;
  }): Promise<{
    performance_metrics: ModelPerformance;
    drift_analysis?: DriftAnalysis['drift_analysis'];
    recommendations: string[];
  }> {
    const response = await api.get('/api/v1/models/performance', { params });
    return response.data;
  },

  async listModels(): Promise<{
    models: ModelInfo[];
    total_count: number;
    retrieved_at: string;
  }> {
    const response = await api.get('/api/v1/models');
    return response.data;
  },

  async getModelDrift(modelId: string): Promise<DriftAnalysis> {
    const response = await api.get(`/api/v1/models/${modelId}/drift`);
    return response.data;
  },

  // Alert Management
  async getAlerts(params: {
    account_id: string;
    status?: string;
    severity?: string;
    limit?: number;
  }): Promise<{
    alerts: Alert[];
    total_count: number;
    account_id: string;
    filters: any;
    retrieved_at: string;
  }> {
    const response = await api.get('/api/v1/alerts', { params });
    return response.data;
  },

  async acknowledgeAlert(alertId: string, userId: string, notes?: string): Promise<{
    alert_id: string;
    status: string;
    acknowledged_by: string;
    acknowledged_at: string;
    notes?: string;
    message: string;
  }> {
    const response = await api.post(`/api/v1/alerts/${alertId}/acknowledge`, null, {
      params: { user_id: userId, notes }
    });
    return response.data;
  },

  async snoozeAlert(alertId: string, durationMinutes: number, userId: string): Promise<{
    alert_id: string;
    status: string;
    snoozed_by: string;
    snoozed_at: string;
    snooze_until: string;
    duration_minutes: number;
    message: string;
  }> {
    const response = await api.post(`/api/v1/alerts/${alertId}/snooze`, null, {
      params: { duration_minutes: durationMinutes, user_id: userId }
    });
    return response.data;
  },

  // Audit Trail
  async getAuditTrail(params: {
    event_types?: string;
    user_id?: string;
    start_date: string;
    end_date: string;
    limit?: number;
  }): Promise<{
    events: AuditEvent[];
    total_count: number;
    metadata: any;
  }> {
    const response = await api.get('/api/v1/audit', { params });
    return response.data;
  },

  // Health & Status
  async healthCheck(): Promise<{
    status: string;
    timestamp: string;
    version: string;
    uptime_seconds: number;
    dependencies: {
      [service: string]: string;
    };
  }> {
    const response = await api.get('/api/v1/health');
    return response.data;
  },

  async getSystemStatus(): Promise<SystemStatus> {
    const response = await api.get('/api/v1/status');
    return response.data;
  },
};

export default anomalyApiService;