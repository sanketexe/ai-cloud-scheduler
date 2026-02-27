import React, { useState, useEffect } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  Container,
  LinearProgress,
  Tabs,
  Tab,
  Card,
  CardContent,
  Divider,
} from '@mui/material';
import { SmartToy, Assignment, Chat } from '@mui/icons-material';
import { useNavigate, useParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { migrationApi, MigrationProject } from '../services/migrationApi';
import OrganizationProfileForm from '../components/MigrationWizard/OrganizationProfileForm';
import WorkloadProfileForm from '../components/MigrationWizard/WorkloadProfileForm';
import RequirementsForm from '../components/MigrationWizard/RequirementsForm';
import AIAssistant from '../components/MigrationWizard/AIAssistant';
import MigrationChatInterface from '../components/MigrationChatbot/MigrationChatInterface';

const steps = [
  'Organization Profile',
  'Workload Analysis',
  'Requirements',
  'Review & Submit',
];

const MigrationWizard: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [project, setProject] = useState<MigrationProject | null>(null);
  const [activeTab, setActiveTab] = useState(0); // 0 = Wizard, 1 = AI Assistant
  const [assessmentStatus, setAssessmentStatus] = useState({
    organization_complete: false,
    workload_complete: false,
    requirements_complete: false,
    overall_progress: 0,
  });

  // Form data state with proper initialization
  const [organizationData, setOrganizationData] = useState<any>({
    company_size: 'MEDIUM',
    industry: 'Technology',
    current_infrastructure: 'ON_PREMISES',
    geographic_presence: ['North America'],
    it_team_size: 10,
    cloud_experience_level: 'BEGINNER',
  });
  const [workloadData, setWorkloadData] = useState<any>({
    total_compute_cores: 4,
    total_memory_gb: 16,
    total_storage_tb: 1,
    database_types: [],
    data_volume_tb: 0.5,
    peak_transaction_rate: 100,
  });
  const [requirementsData, setRequirementsData] = useState<any>({
    performance: {
      latency_target_ms: 100,
      availability_target: 99.9,
      disaster_recovery_rto_minutes: 60,
      disaster_recovery_rpo_minutes: 15,
      geographic_distribution: [],
    },
    compliance: {
      regulatory_frameworks: [],
      data_residency_requirements: [],
      industry_certifications: [],
      security_standards: [],
    },
    budget: {
      current_monthly_cost: 5000,
      migration_budget: 50000,
      target_monthly_cost: 4000,
      cost_optimization_priority: 'MEDIUM',
    },
    technical: {
      required_services: ['Compute', 'Storage', 'Database'],
      ml_ai_required: false,
      analytics_required: false,
      container_orchestration: false,
      serverless_required: false,
    },
  });

  // Validation state
  const [validationErrors, setValidationErrors] = useState<Record<string, string[]>>({});

  // Form data persistence and validation
  useEffect(() => {
    console.log('Organization data changed:', organizationData);
    // Persist to localStorage for recovery
    if (organizationData && projectId) {
      localStorage.setItem(`migration-org-${projectId}`, JSON.stringify(organizationData));
    }
  }, [organizationData, projectId]);

  useEffect(() => {
    console.log('Workload data changed:', workloadData);
    // Persist to localStorage for recovery
    if (workloadData && projectId) {
      localStorage.setItem(`migration-workload-${projectId}`, JSON.stringify(workloadData));
    }
  }, [workloadData, projectId]);

  useEffect(() => {
    console.log('Requirements data changed:', requirementsData);
    // Persist to localStorage for recovery
    if (requirementsData && projectId) {
      localStorage.setItem(`migration-requirements-${projectId}`, JSON.stringify(requirementsData));
    }
  }, [requirementsData, projectId]);

  // Load persisted data on initialization
  useEffect(() => {
    if (projectId) {
      const savedOrgData = localStorage.getItem(`migration-org-${projectId}`);
      if (savedOrgData) {
        try {
          const parsedData = JSON.parse(savedOrgData);
          setOrganizationData(parsedData);
        } catch (error) {
          console.error('Failed to parse saved organization data:', error);
        }
      }

      const savedWorkloadData = localStorage.getItem(`migration-workload-${projectId}`);
      if (savedWorkloadData) {
        try {
          const parsedData = JSON.parse(savedWorkloadData);
          setWorkloadData(parsedData);
        } catch (error) {
          console.error('Failed to parse saved workload data:', error);
        }
      }

      const savedRequirementsData = localStorage.getItem(`migration-requirements-${projectId}`);
      if (savedRequirementsData) {
        try {
          const parsedData = JSON.parse(savedRequirementsData);
          setRequirementsData(parsedData);
        } catch (error) {
          console.error('Failed to parse saved requirements data:', error);
        }
      }
    }
  }, [projectId]);

  useEffect(() => {
    const initProject = async () => {
      try {
        setInitializing(true);

        if (projectId) {
          // Load existing project
          await loadProject();
          await loadAssessmentStatus();
        } else {
          // Create new project
          const newProject = await migrationApi.createProject({
            organization_name: 'My Organization',
          });
          setProject(newProject);
          // Update URL with new project ID
          navigate(`/migration-wizard/${newProject.project_id}`, { replace: true });
        }
      } catch (error) {
        console.error('Failed to initialize project:', error);
        // Set a mock project to allow user to continue
        setProject({
          project_id: 'demo-project',
          organization_name: 'Demo Organization',
          created_date: new Date().toISOString(),
          status: 'ASSESSMENT',
          current_phase: 'Organization Profile',
          estimated_completion: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000).toISOString(),
        });
      } finally {
        setInitializing(false);
      }
    };

    initProject();
  }, [projectId, navigate]);

  const loadProject = async () => {
    try {
      const data = await migrationApi.getProject(projectId!);
      setProject(data);
    } catch (error) {
      toast.error('Failed to load migration project');
      console.error(error);
    }
  };

  const loadAssessmentStatus = async () => {
    try {
      const status = await migrationApi.getAssessmentStatus(projectId!);
      setAssessmentStatus(status);

      // Set active step based on completion
      if (!status.organization_complete) {
        setActiveStep(0);
      } else if (!status.workload_complete) {
        setActiveStep(1);
      } else if (!status.requirements_complete) {
        setActiveStep(2);
      } else {
        setActiveStep(3);
      }
    } catch (error) {
      console.error('Failed to load assessment status', error);
    }
  };

  const handleNext = async () => {
    // Validate current step before proceeding
    if (!canProceed()) {
      toast.error('Please complete all required fields before continuing');
      return;
    }

    setLoading(true);
    try {
      // Save current step data
      if (activeStep === 0 && organizationData) {
        console.log('Submitting organization data:', organizationData);
        await migrationApi.submitOrganizationProfile(projectId!, organizationData);
        toast.success('Organization profile saved');
      } else if (activeStep === 1 && workloadData) {
        console.log('Submitting workload data:', workloadData);
        await migrationApi.submitWorkloadProfile(projectId!, workloadData);
        toast.success('Workload profile saved');
      } else if (activeStep === 2 && requirementsData) {
        console.log('Submitting requirements data:', requirementsData);
        await migrationApi.submitRequirements(projectId!, requirementsData);
        toast.success('Requirements saved');
      }

      if (activeStep === steps.length - 1) {
        // Complete assessment and generate recommendations automatically
        toast.success('Assessment complete! Generating recommendations...');

        try {
          // Check if recommendations already exist
          const recommendationsExist = await migrationApi.checkRecommendationsExist(projectId!);

          if (!recommendationsExist) {
            // Generate new recommendations
            await migrationApi.generateRecommendations(projectId!);
            toast.success('Recommendations generated successfully!');
          } else {
            toast.success('Using existing recommendations');
          }

          navigate(`/migration/${projectId}/recommendations`);
        } catch (error: any) {
          console.error('Failed to generate recommendations:', error);
          const errorMessage = error.response?.data?.detail ||
            error.response?.data?.error?.message ||
            error.message ||
            'Failed to generate recommendations';
          toast.error(errorMessage);

          // Still navigate to recommendations page to show error state
          navigate(`/migration/${projectId}/recommendations`);
        }
      } else {
        setActiveStep((prevStep) => prevStep + 1);
        await loadAssessmentStatus();
      }
    } catch (error: any) {
      console.error('Error in handleNext:', error);
      const errorMessage = error.response?.data?.detail ||
        error.response?.data?.error?.message ||
        error.message ||
        'Failed to save data';
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const getStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <OrganizationProfileForm
            data={organizationData}
            onChange={setOrganizationData}
          />
        );
      case 1:
        return (
          <WorkloadProfileForm
            data={workloadData}
            onChange={setWorkloadData}
          />
        );
      case 2:
        return (
          <RequirementsForm
            data={requirementsData}
            onChange={setRequirementsData}
          />
        );
      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review Your Assessment
            </Typography>
            <Paper sx={{ p: 3, mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Organization Profile
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Company Size: {organizationData?.company_size}
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Industry: {organizationData?.industry}
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Current Infrastructure: {organizationData?.current_infrastructure}
              </Typography>
            </Paper>
            <Paper sx={{ p: 3, mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Workload Profile
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Compute Cores: {workloadData?.total_compute_cores}
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Memory: {workloadData?.total_memory_gb} GB
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Storage: {workloadData?.total_storage_tb} TB
              </Typography>
            </Paper>
            <Paper sx={{ p: 3, mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Requirements Summary
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Availability Target: {requirementsData?.performance?.availability_target}%
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Budget: ${requirementsData?.budget?.migration_budget?.toLocaleString()}
              </Typography>
            </Paper>
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };

  const isStepComplete = (step: number) => {
    switch (step) {
      case 0:
        return assessmentStatus.organization_complete;
      case 1:
        return assessmentStatus.workload_complete;
      case 2:
        return assessmentStatus.requirements_complete;
      default:
        return false;
    }
  };

  // Validation functions for each step
  const validateOrganizationData = (data: any): string[] => {
    const errors: string[] = [];

    if (!data) {
      errors.push('Organization data is required');
      return errors;
    }

    if (!data.company_size) {
      errors.push('Company size is required');
    }

    if (!data.industry) {
      errors.push('Industry is required');
    }

    if (!data.current_infrastructure) {
      errors.push('Current infrastructure type is required');
    }

    if (!data.geographic_presence || data.geographic_presence.length === 0) {
      errors.push('At least one geographic region is required');
    }

    if (!data.it_team_size || data.it_team_size < 1) {
      errors.push('IT team size must be at least 1');
    }

    if (!data.cloud_experience_level) {
      errors.push('Cloud experience level is required');
    }

    return errors;
  };

  const validateWorkloadData = (data: any): string[] => {
    const errors: string[] = [];

    if (!data) {
      errors.push('Workload data is required');
      return errors;
    }

    if (typeof data.total_compute_cores !== 'number' || data.total_compute_cores < 0) {
      errors.push('Total compute cores must be a valid number (0 or greater)');
    }

    if (typeof data.total_memory_gb !== 'number' || data.total_memory_gb < 0) {
      errors.push('Total memory must be a valid number (0 or greater)');
    }

    if (typeof data.total_storage_tb !== 'number' || data.total_storage_tb < 0) {
      errors.push('Total storage must be a valid number (0 or greater)');
    }

    if (typeof data.data_volume_tb !== 'number' || data.data_volume_tb < 0) {
      errors.push('Data volume must be a valid number (0 or greater)');
    }

    if (typeof data.peak_transaction_rate !== 'number' || data.peak_transaction_rate < 0) {
      errors.push('Peak transaction rate must be a valid number (0 or greater)');
    }

    return errors;
  };

  const validateRequirementsData = (data: any): string[] => {
    const errors: string[] = [];

    if (!data) {
      errors.push('Requirements data is required');
      return errors;
    }

    // Performance requirements validation
    if (!data.performance) {
      errors.push('Performance requirements are required');
    } else {
      if (typeof data.performance.latency_target_ms !== 'number' || data.performance.latency_target_ms < 0) {
        errors.push('Latency target must be a valid number (0 or greater)');
      }

      if (typeof data.performance.availability_target !== 'number' ||
        data.performance.availability_target < 90 ||
        data.performance.availability_target > 100) {
        errors.push('Availability target must be between 90 and 100 percent');
      }

      if (typeof data.performance.disaster_recovery_rto_minutes !== 'number' ||
        data.performance.disaster_recovery_rto_minutes < 0) {
        errors.push('RTO must be a valid number (0 or greater)');
      }

      if (typeof data.performance.disaster_recovery_rpo_minutes !== 'number' ||
        data.performance.disaster_recovery_rpo_minutes < 0) {
        errors.push('RPO must be a valid number (0 or greater)');
      }
    }

    // Budget requirements validation
    if (!data.budget) {
      errors.push('Budget requirements are required');
    } else {
      if (typeof data.budget.current_monthly_cost !== 'number' || data.budget.current_monthly_cost < 0) {
        errors.push('Current monthly cost must be a valid number (0 or greater)');
      }

      if (typeof data.budget.migration_budget !== 'number' || data.budget.migration_budget < 0) {
        errors.push('Migration budget must be a valid number (0 or greater)');
      }

      if (typeof data.budget.target_monthly_cost !== 'number' || data.budget.target_monthly_cost < 0) {
        errors.push('Target monthly cost must be a valid number (0 or greater)');
      }

      if (!data.budget.cost_optimization_priority) {
        errors.push('Cost optimization priority is required');
      }
    }

    // Compliance requirements validation
    if (!data.compliance) {
      errors.push('Compliance requirements are required');
    }

    // Technical requirements validation
    if (!data.technical) {
      errors.push('Technical requirements are required');
    } else {
      if (!data.technical.required_services || data.technical.required_services.length === 0) {
        errors.push('At least one required service must be selected');
      }
    }

    return errors;
  };

  // Real-time validation effect
  useEffect(() => {
    const newErrors: Record<string, string[]> = {};

    // Validate current step
    switch (activeStep) {
      case 0:
        newErrors.organization = validateOrganizationData(organizationData);
        break;
      case 1:
        newErrors.workload = validateWorkloadData(workloadData);
        break;
      case 2:
        newErrors.requirements = validateRequirementsData(requirementsData);
        break;
    }

    setValidationErrors(newErrors);
  }, [activeStep, organizationData, workloadData, requirementsData]);

  const canProceed = () => {
    console.log('Checking canProceed for step:', activeStep);
    console.log('Current data:', { organizationData, workloadData, requirementsData });

    switch (activeStep) {
      case 0:
        const orgErrors = validateOrganizationData(organizationData);
        console.log('Organization validation errors:', orgErrors);
        return orgErrors.length === 0;
      case 1:
        const workloadErrors = validateWorkloadData(workloadData);
        console.log('Workload validation errors:', workloadErrors);
        return workloadErrors.length === 0;
      case 2:
        const reqErrors = validateRequirementsData(requirementsData);
        console.log('Requirements validation errors:', reqErrors);
        return reqErrors.length === 0;
      case 3:
        return true;
      default:
        return false;
    }
  };

  const getCurrentStepErrors = (): string[] => {
    switch (activeStep) {
      case 0:
        return validationErrors.organization || [];
      case 1:
        return validationErrors.workload || [];
      case 2:
        return validationErrors.requirements || [];
      default:
        return [];
    }
  };

  if (initializing || !project) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <LinearProgress />
          <Typography sx={{ mt: 2 }}>
            {initializing ? 'Initializing migration wizard...' : 'Loading migration project...'}
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <>
      {/* AI Assistant - Floating Chat Widget */}
      <AIAssistant
        context={{
          organization: organizationData,
          workload: workloadData,
          requirements: requirementsData,
        }}
        currentStep={steps[activeStep]}
      />

      <Container maxWidth="lg">
        <Box sx={{ mt: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Cloud Migration Assessment
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            {project.organization_name} - Project ID: {project.project_id}
          </Typography>

          <Paper sx={{ p: 3, mt: 3 }}>
            <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
              {steps.map((label, index) => (
                <Step key={label} completed={isStepComplete(index)}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>

            {loading && <LinearProgress sx={{ mb: 2 }} />}

            <Box sx={{ minHeight: 400 }}>
              {getStepContent(activeStep)}
            </Box>

            {/* Validation Feedback */}
            {getCurrentStepErrors().length > 0 && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'error.light', borderRadius: 1, border: '1px solid', borderColor: 'error.main' }}>
                <Typography variant="subtitle2" sx={{ color: 'error.dark', fontWeight: 'bold', mb: 1 }}>
                  Please fix the following issues to continue:
                </Typography>
                {getCurrentStepErrors().map((error, index) => (
                  <Typography key={index} variant="body2" sx={{ color: 'error.dark', ml: 1 }}>
                    • {error}
                  </Typography>
                ))}
              </Box>
            )}


            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
              <Button
                disabled={activeStep === 0 || loading}
                onClick={handleBack}
              >
                Back
              </Button>
              <Box sx={{ flex: '1 1 auto' }} />
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="contained"
                  onClick={handleNext}
                  disabled={!canProceed() || loading}
                  sx={{
                    minWidth: 120,
                    backgroundColor: canProceed() ? 'primary.main' : 'grey.400'
                  }}
                >
                  {activeStep === steps.length - 1 ? 'Complete Assessment' : 'Next'}
                </Button>
              </Box>
            </Box>
          </Paper>

          {/* Progress indicator */}
          <Paper sx={{ p: 2, mt: 3 }}>
            <Typography variant="body2" gutterBottom>
              Overall Progress: {assessmentStatus.overall_progress}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={assessmentStatus.overall_progress}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Paper>
        </Box>
      </Container>
    </>
  );
};

export default MigrationWizard;
