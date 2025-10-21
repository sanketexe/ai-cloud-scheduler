// API service for FinOps Platform
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// FinOps Types
export interface CostData {
  date: string;
  service: string;
  team: string;
  cost: number;
  usage_quantity: number;
  usage_unit: string;
}

export interface Budget {
  id: string;
  name: string;
  amount: number;
  spent: number;
  utilization: number;
  period: string;
  status: string;
  team: string;
  alerts: Array<{
    threshold: number;
    triggered: boolean;
  }>;
}

export interface OptimizationRecommendation {
  id: string;
  type: string;
  resource: string;
  resourceType: string;
  currentConfig: string;
  recommendedConfig: string;
  monthlySavings: number;
  annualSavings: number;
  confidence: number;
  riskLevel: string;
  status: string;
  team: string;
}

export interface Alert {
  id: string;
  name: string;
  type: string;
  condition: string;
  threshold: number;
  currentValue: number;
  status: string;
  severity: string;
  enabled: boolean;
  channels: string[];
  team: string;
}

export interface ComplianceData {
  overallScore: number;
  taggingCompliance: number;
  policyCompliance: number;
  securityCompliance: number;
  violations: Array<{
    id: string;
    resource: string;
    policy: string;
    violation: string;
    severity: string;
    status: string;
  }>;
}

// FinOps API Functions
export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Cost Management
  async getCosts(params: {
    startDate: string;
    endDate: string;
    granularity?: string;
    groupBy?: string[];
    filters?: any;
  }): Promise<{ data: CostData[]; total: number }> {
    const response = await api.get('/api/v1/costs', { params });
    return response.data;
  },

  async getCostAttribution(params: {
    startDate: string;
    endDate: string;
    dimension: string;
  }) {
    const response = await api.get('/api/v1/costs/attribution', { params });
    return response.data;
  },

  async getCostAnomalies(params?: {
    startDate?: string;
    endDate?: string;
    confidenceThreshold?: number;
  }) {
    const response = await api.get('/api/v1/costs/anomalies', { params });
    return response.data;
  },

  // Budget Management
  async getBudgets(params?: { status?: string; team?: string }): Promise<Budget[]> {
    const response = await api.get('/api/v1/budgets', { params });
    return response.data.budgets;
  },

  async createBudget(budget: Omit<Budget, 'id'>): Promise<Budget> {
    const response = await api.post('/api/v1/budgets', budget);
    return response.data;
  },

  async updateBudget(id: string, budget: Partial<Budget>): Promise<Budget> {
    const response = await api.patch(`/api/v1/budgets/${id}`, budget);
    return response.data;
  },

  async deleteBudget(id: string): Promise<void> {
    await api.delete(`/api/v1/budgets/${id}`);
  },

  // Optimization
  async getOptimizationRecommendations(params?: {
    type?: string;
    minSavings?: number;
    confidence?: string;
  }): Promise<OptimizationRecommendation[]> {
    const response = await api.get('/api/v1/optimization/recommendations', { params });
    return response.data.recommendations;
  },

  async implementRecommendation(id: string, params: {
    implementationDate?: string;
    notes?: string;
  }): Promise<void> {
    await api.post(`/api/v1/optimization/recommendations/${id}/implement`, params);
  },

  async getRIRecommendations() {
    const response = await api.get('/api/v1/optimization/reserved-instances');
    return response.data;
  },

  // Reports
  async generateReport(params: {
    type: string;
    name: string;
    parameters: any;
    format: string;
    delivery?: any;
  }) {
    const response = await api.post('/api/v1/reports/generate', params);
    return response.data;
  },

  async getReports() {
    const response = await api.get('/api/v1/reports');
    return response.data;
  },

  async getReportStatus(id: string) {
    const response = await api.get(`/api/v1/reports/${id}`);
    return response.data;
  },

  async downloadReport(id: string) {
    const response = await api.get(`/api/v1/reports/${id}/download`, {
      responseType: 'blob',
    });
    return response.data;
  },

  // Alerts
  async getAlerts(): Promise<Alert[]> {
    const response = await api.get('/api/v1/alerts');
    return response.data.alerts;
  },

  async createAlert(alert: Omit<Alert, 'id'>): Promise<Alert> {
    const response = await api.post('/api/v1/alerts', alert);
    return response.data;
  },

  async updateAlert(id: string, alert: Partial<Alert>): Promise<Alert> {
    const response = await api.patch(`/api/v1/alerts/${id}`, alert);
    return response.data;
  },

  async deleteAlert(id: string): Promise<void> {
    await api.delete(`/api/v1/alerts/${id}`);
  },

  async testAlert(id: string): Promise<void> {
    await api.post(`/api/v1/alerts/${id}/test`);
  },

  // Compliance
  async getComplianceOverview(): Promise<ComplianceData> {
    const response = await api.get('/api/v1/compliance/overview');
    return response.data;
  },

  async getTaggingCompliance() {
    const response = await api.get('/api/v1/compliance/tagging');
    return response.data;
  },

  async getPolicyViolations() {
    const response = await api.get('/api/v1/compliance/violations');
    return response.data;
  },

  // Configuration
  async getCloudProviders() {
    const response = await api.get('/api/v1/config/cloud-providers');
    return response.data;
  },

  async updateNotificationSettings(settings: any) {
    const response = await api.put('/api/v1/config/notifications', settings);
    return response.data;
  },

  // Webhooks
  async createWebhook(webhook: any) {
    const response = await api.post('/api/v1/webhooks', webhook);
    return response.data;
  },

  async getWebhooks() {
    const response = await api.get('/api/v1/webhooks');
    return response.data;
  },
};

export default apiService;