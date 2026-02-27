/**
 * Multi-Cloud Cost Comparison API Service
 * 
 * Provides API functions for multi-cloud cost comparison, TCO analysis,
 * and migration planning functionality.
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Multi-Cloud Types
export interface ComputeSpec {
  cpu_cores: number;
  memory_gb: number;
  instance_type_preference?: string;
  operating_system: string;
  architecture: string;
  gpu_required: boolean;
  gpu_type?: string;
}

export interface StorageSpec {
  primary_storage_gb: number;
  storage_type: string;
  backup_storage_gb?: number;
  iops_requirement?: number;
  throughput_mbps?: number;
}

export interface NetworkSpec {
  bandwidth_mbps: number;
  data_transfer_gb_monthly: number;
  load_balancer_required: boolean;
  cdn_required: boolean;
  vpn_required: boolean;
}

export interface DatabaseSpec {
  database_type: string;
  storage_gb: number;
  connections: number;
  backup_retention_days: number;
  high_availability: boolean;
}

export interface UsagePatterns {
  hours_per_day: number;
  days_per_week: number;
  peak_usage_multiplier: number;
  seasonal_variation: boolean;
  auto_scaling: boolean;
}

export interface WorkloadSpec {
  name: string;
  description?: string;
  compute_spec: ComputeSpec;
  storage_spec: StorageSpec;
  network_spec: NetworkSpec;
  database_spec?: DatabaseSpec;
  additional_services: string[];
  usage_patterns: UsagePatterns;
  compliance_requirements: string[];
  regions: string[];
}

export interface CostBreakdown {
  compute: number;
  storage: number;
  network: number;
  database?: number;
  additional_services: number;
  support: number;
  total: number;
}

export interface CostComparison {
  id: string;
  workload_id: string;
  comparison_date: string;
  aws_monthly_cost?: number;
  gcp_monthly_cost?: number;
  azure_monthly_cost?: number;
  aws_annual_cost?: number;
  gcp_annual_cost?: number;
  azure_annual_cost?: number;
  cost_breakdown: Record<string, CostBreakdown>;
  recommendations: string[];
  pricing_data_version?: string;
  lowest_cost_provider?: string;
  cost_difference_percentage?: Record<string, number>;
}

export interface TCOAnalysis {
  id: string;
  workload_id: string;
  analysis_date: string;
  time_horizon_years: number;
  aws_tco: Record<string, any>;
  gcp_tco: Record<string, any>;
  azure_tco: Record<string, any>;
  hidden_costs: Record<string, any>;
  operational_costs: Record<string, any>;
  cost_projections: Record<string, any>;
  total_tco_comparison: Record<string, number>;
  recommended_provider?: string;
}

export interface RiskAssessment {
  overall_risk_level: string;
  technical_risks: string[];
  business_risks: string[];
  mitigation_strategies: string[];
  success_probability: number;
}

export interface MigrationAnalysis {
  id: string;
  workload_id: string;
  source_provider: string;
  target_provider: string;
  analysis_date: string;
  migration_cost: number;
  migration_timeline_days: number;
  break_even_months?: number;
  cost_breakdown: Record<string, number>;
  risk_assessment: RiskAssessment;
  recommendations: string[];
  monthly_savings?: number;
  annual_savings?: number;
  roi_percentage?: number;
}

export interface ServicePricing {
  provider: string;
  service_name: string;
  service_category: string;
  region: string;
  pricing_unit: string;
  price_per_unit: number;
  currency: string;
  effective_date: string;
  pricing_details?: Record<string, any>;
}

export interface CloudProvider {
  name: string;
  provider_type: string;
  supported_regions: string[];
  supported_services: string[];
  pricing_model: string;
}

export interface CloudService {
  name: string;
  category: string;
  description: string;
  pricing_units: string[];
  regions: string[];
}

export interface MigrationRequest {
  workload_id: string;
  source_provider: 'aws' | 'gcp' | 'azure';
  target_provider: 'aws' | 'gcp' | 'azure';
  migration_timeline_preference?: number;
  downtime_tolerance_hours?: number;
  team_size: number;
  include_training_costs: boolean;
}

export interface TCORequest {
  workload_id: string;
  time_horizon_years: number;
  include_hidden_costs: boolean;
  discount_rate: number;
}

export interface WorkloadValidation {
  is_valid: boolean;
  errors: Array<{
    message: string;
    field?: string;
    code?: string;
  }>;
  warnings: string[];
  estimated_monthly_cost_range?: {
    min_cost: number;
    max_cost: number;
    currency: string;
  };
}

export interface WorkloadListResponse {
  workloads: Array<{
    id: string;
    name: string;
    description?: string;
    created_at: string;
    updated_at: string;
    regions: string[];
    compliance_requirements: string[];
  }>;
  total_count: number;
  page: number;
  page_size: number;
}

export interface ComparisonListResponse {
  comparisons: CostComparison[];
  total_count: number;
  page: number;
  page_size: number;
}

// Multi-Cloud API Service
export const multiCloudApi = {
  // Workload Cost Comparison
  async compareWorkloadCosts(workload: WorkloadSpec): Promise<CostComparison> {
    const response = await api.post('/api/v1/multi-cloud/compare', workload);
    return response.data;
  },

  // TCO Analysis
  async calculateTCO(request: TCORequest): Promise<TCOAnalysis> {
    const response = await api.post('/api/v1/multi-cloud/tco', request);
    return response.data;
  },

  // Migration Analysis
  async analyzeMigration(request: MigrationRequest): Promise<MigrationAnalysis> {
    const response = await api.post('/api/v1/multi-cloud/migration', request);
    return response.data;
  },

  // Service Pricing
  async getServicePricing(
    provider: string,
    service: string,
    region?: string
  ): Promise<ServicePricing> {
    const params = region ? { region } : {};
    const response = await api.get(`/api/v1/multi-cloud/pricing/${provider}/${service}`, { params });
    return response.data;
  },

  // Supported Providers
  async getSupportedProviders(): Promise<CloudProvider[]> {
    const response = await api.get('/api/v1/multi-cloud/providers');
    return response.data;
  },

  // Provider Services
  async getProviderServices(provider: string, category?: string): Promise<CloudService[]> {
    const params = category ? { category } : {};
    const response = await api.get(`/api/v1/multi-cloud/services/${provider}`, { params });
    return response.data;
  },

  // Workload Management
  async getWorkloadSpecifications(page = 1, pageSize = 20): Promise<WorkloadListResponse> {
    const response = await api.get('/api/v1/multi-cloud/workloads', {
      params: { page, page_size: pageSize }
    });
    return response.data;
  },

  async getWorkloadComparisons(
    workloadId: string,
    page = 1,
    pageSize = 10
  ): Promise<ComparisonListResponse> {
    const response = await api.get(`/api/v1/multi-cloud/comparisons/${workloadId}`, {
      params: { page, page_size: pageSize }
    });
    return response.data;
  },

  // Workload Validation
  async validateWorkloadSpecification(workload: WorkloadSpec): Promise<WorkloadValidation> {
    const response = await api.post('/api/v1/multi-cloud/validate', workload);
    return response.data;
  },

  // Utility Functions
  getProviderDisplayName(provider: string): string {
    const names: Record<string, string> = {
      aws: 'Amazon Web Services',
      gcp: 'Google Cloud Platform',
      azure: 'Microsoft Azure'
    };
    return names[provider] || provider;
  },

  formatCurrency(amount: number, currency = 'USD'): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  },

  calculateSavingsPercentage(originalCost: number, newCost: number): number {
    if (originalCost === 0) return 0;
    return ((originalCost - newCost) / originalCost) * 100;
  }
};

export default multiCloudApi;