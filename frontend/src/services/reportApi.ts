/**
 * API service for migration report generation and export
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

// Types
export interface ReportSummary {
  report_id: string;
  project_id: string;
  generated_at: string;
  primary_recommendation: string;
  confidence_score: number;
  estimated_monthly_cost?: number;
  estimated_savings?: number;
}

export interface ExecutiveSummary {
  organization_name: string;
  assessment_date: string;
  primary_recommendation: string;
  estimated_monthly_cost?: number;
  estimated_savings?: number;
  migration_duration_weeks: number;
  confidence_score: number;
  key_benefits: string[];
  critical_considerations: string[];
}

export interface TechnicalAnalysis {
  workload_summary: {
    total_compute_cores: number;
    total_memory_gb: number;
    total_storage_tb: number;
    database_types: string[];
    peak_transaction_rate: number;
    application_count: number;
  };
  performance_requirements: {
    availability_target: number;
    max_latency_ms: number;
    throughput_requirements: any;
    scalability_requirements: any;
  };
  compliance_requirements: {
    regulatory_frameworks: string[];
    data_residency_requirements: string[];
    industry_certifications: string[];
    security_standards: string[];
  };
  technical_constraints: {
    preferred_technologies: string[];
    integration_requirements: any;
    security_requirements: any;
    backup_requirements: any;
  };
  provider_evaluations: Array<{
    provider: string;
    overall_score: number;
    cost_score: number;
    performance_score: number;
    compliance_score: number;
    migration_complexity_score: number;
    estimated_monthly_cost?: number;
    migration_duration_weeks: number;
    strengths: string[];
    weaknesses: string[];
  }>;
  comparison_matrix: {
    key_differences: string[];
    cost_comparison: any;
    scoring_weights: any;
  };
  risk_assessment: any;
}

export interface ImplementationRoadmap {
  migration_phases: Array<{
    phase: string;
    duration_weeks: number;
    description: string;
    key_activities: string[];
  }>;
  timeline_overview: {
    total_duration_weeks: number;
    estimated_start_date: string;
    estimated_completion_date: string;
    critical_path: string[];
  };
  resource_requirements: {
    project_manager: string;
    cloud_architect: string;
    migration_engineers: string;
    application_teams: string;
    estimated_budget: string;
  };
  success_criteria: string[];
  potential_challenges: string[];
  mitigation_strategies: string[];
}

export interface AssessmentInputs {
  organization_profile: {
    company_name: string;
    company_size: string;
    industry: string;
    current_infrastructure: string;
    it_team_size: number;
    cloud_experience_level: string;
    geographic_presence: string[];
  };
  workload_profile: {
    total_compute_cores: number;
    total_memory_gb: number;
    total_storage_tb: number;
    database_types: string[];
    peak_transaction_rate: number;
    applications: any[];
  };
  requirements_summary: {
    performance: {
      availability_target: number;
      max_latency_ms: number;
    };
    compliance: {
      regulatory_frameworks: string[];
      data_residency: string[];
    };
    budget: {
      migration_budget: number;
      target_monthly_cost?: number;
    };
    technical: {
      preferred_technologies: string[];
      security_requirements: any;
    };
  };
  scoring_methodology: {
    weights_used: any;
    evaluation_criteria: string[];
    scoring_scale: string;
  };
  assumptions: string[];
}

export interface ComprehensiveReport {
  report_id: string;
  project_id: string;
  generated_at: string;
  executive_summary: ExecutiveSummary;
  technical_analysis: TechnicalAnalysis;
  implementation_roadmap: ImplementationRoadmap;
  assessment_inputs: AssessmentInputs;
  appendices: any;
}

export interface ShareableLink {
  link_token: string;
  expires_at: string;
  share_url: string;
}

export interface ReportStatus {
  status: 'generating' | 'completed' | 'failed' | 'not_ready' | 'ready';
  message: string;
  report_id?: string;
}

// API Functions
export const reportApi = {
  /**
   * Generate a comprehensive migration report
   */
  async generateReport(projectId: string): Promise<ReportSummary> {
    const response = await axios.post(
      `${API_BASE_URL}/migration-advisor/${projectId}/reports/generate`
    );
    return response.data;
  },

  /**
   * Get the latest comprehensive report for a project
   */
  async getLatestReport(projectId: string): Promise<ComprehensiveReport> {
    const response = await axios.get(
      `${API_BASE_URL}/migration-advisor/${projectId}/reports/latest`
    );
    return response.data;
  },

  /**
   * Export report as PDF
   */
  async exportPDF(projectId: string): Promise<Blob> {
    const response = await axios.get(
      `${API_BASE_URL}/migration-advisor/${projectId}/reports/export/pdf`,
      {
        responseType: 'blob',
      }
    );
    return response.data;
  },

  /**
   * Create a shareable link for the report
   */
  async createShareableLink(projectId: string, expiresInDays: number = 30): Promise<ShareableLink> {
    const response = await axios.post(
      `${API_BASE_URL}/migration-advisor/${projectId}/reports/share`,
      null,
      {
        params: { expires_in_days: expiresInDays }
      }
    );
    return response.data;
  },

  /**
   * Get report generation status
   */
  async getReportStatus(projectId: string): Promise<ReportStatus> {
    const response = await axios.get(
      `${API_BASE_URL}/migration-advisor/${projectId}/reports/status`
    );
    return response.data;
  },

  /**
   * View a shared report using a token
   */
  async viewSharedReport(linkToken: string): Promise<Partial<ComprehensiveReport>> {
    const response = await axios.get(
      `${API_BASE_URL}/migration-advisor/shared/${linkToken}`
    );
    return response.data;
  },

  /**
   * Download PDF and trigger browser download
   */
  async downloadPDF(projectId: string, organizationName: string = 'Report'): Promise<void> {
    try {
      const blob = await this.exportPDF(projectId);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Generate filename
      const date = new Date().toISOString().split('T')[0];
      const filename = `Migration_Report_${organizationName.replace(/\s+/g, '_')}_${date}.pdf`;
      link.download = filename;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download PDF:', error);
      throw error;
    }
  }
};

export default reportApi;