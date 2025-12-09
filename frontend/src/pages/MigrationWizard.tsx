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
} from '@mui/material';
import { useNavigate, useParams } from 'react-router-dom';
import toast from 'react-hot-toast';
import { migrationApi, MigrationProject } from '../services/migrationApi';
import OrganizationProfileForm from '../components/MigrationWizard/OrganizationProfileForm';
import WorkloadProfileForm from '../components/MigrationWizard/WorkloadProfileForm';
import RequirementsForm from '../components/MigrationWizard/RequirementsForm';
import AIAssistant from '../components/MigrationWizard/AIAssistant';

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
  const [assessmentStatus, setAssessmentStatus] = useState({
    organization_complete: false,
    workload_complete: false,
    requirements_complete: false,
    overall_progress: 0,
  });

  // Form data state
  const [organizationData, setOrganizationData] = useState<any>(null);
  const [workloadData, setWorkloadData] = useState<any>(null);
  const [requirementsData, setRequirementsData] = useState<any>(null);

  useEffect(() => {
    initializeProject();
  }, [projectId]);

  const initializeProject = async () => {
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
    setLoading(true);
    try {
      // Save current step data
      if (activeStep === 0 && organizationData) {
        await migrationApi.submitOrganizationProfile(projectId!, organizationData);
        toast.success('Organization profile saved');
      } else if (activeStep === 1 && workloadData) {
        await migrationApi.submitWorkloadProfile(projectId!, workloadData);
        toast.success('Workload profile saved');
      } else if (activeStep === 2 && requirementsData) {
        await migrationApi.submitRequirements(projectId!, requirementsData);
        toast.success('Requirements saved');
      }

      if (activeStep === steps.length - 1) {
        // Complete assessment and navigate to recommendations
        toast.success('Assessment complete! Generating recommendations...');
        navigate(`/migration/${projectId}/recommendations`);
      } else {
        setActiveStep((prevStep) => prevStep + 1);
        await loadAssessmentStatus();
      }
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save data');
      console.error(error);
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

  const canProceed = () => {
    switch (activeStep) {
      case 0:
        return organizationData !== null;
      case 1:
        return workloadData !== null;
      case 2:
        return requirementsData !== null;
      case 3:
        return true;
      default:
        return false;
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

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
            <Button
              disabled={activeStep === 0 || loading}
              onClick={handleBack}
            >
              Back
            </Button>
            <Box sx={{ flex: '1 1 auto' }} />
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={!canProceed() || loading}
            >
              {activeStep === steps.length - 1 ? 'Complete Assessment' : 'Next'}
            </Button>
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
