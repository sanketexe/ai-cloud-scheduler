/**
 * Migration Timeline Component
 * 
 * Visual timeline showing migration phases, dependencies,
 * and key milestones with interactive elements.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Tooltip,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent,
} from '@mui/lab';
import {
  Assessment as AssessmentIcon,
  Build as BuildIcon,
  CloudUpload as CloudUploadIcon,
  Security as SecurityIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  Warning as WarningIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Task as TaskIcon
} from '@mui/icons-material';

import { MigrationAnalysis } from '../../services/multiCloudApi';

interface MigrationPhase {
  id: string;
  name: string;
  description: string;
  duration: number; // in days
  startDay: number;
  endDay: number;
  status: 'pending' | 'in-progress' | 'completed' | 'blocked';
  dependencies: string[];
  tasks: MigrationTask[];
  riskLevel: 'low' | 'medium' | 'high';
  icon: React.ReactNode;
}

interface MigrationTask {
  id: string;
  name: string;
  description: string;
  duration: number; // in hours
  assignee?: string;
  status: 'pending' | 'in-progress' | 'completed';
  dependencies: string[];
}

interface MigrationTimelineProps {
  analysis: MigrationAnalysis;
}

const MigrationTimeline: React.FC<MigrationTimelineProps> = ({ analysis }) => {
  const [expandedPhase, setExpandedPhase] = useState<string | null>(null);

  // Generate migration phases based on analysis
  const generateMigrationPhases = (): MigrationPhase[] => {
    const totalDays = analysis.migration_timeline_days;
    
    return [
      {
        id: 'assessment',
        name: 'Assessment & Planning',
        description: 'Analyze current infrastructure and plan migration strategy',
        duration: Math.ceil(totalDays * 0.15),
        startDay: 1,
        endDay: Math.ceil(totalDays * 0.15),
        status: 'completed',
        dependencies: [],
        riskLevel: 'low',
        icon: <AssessmentIcon />,
        tasks: [
          {
            id: 'inventory',
            name: 'Infrastructure Inventory',
            description: 'Document current infrastructure and dependencies',
            duration: 16,
            status: 'completed',
            dependencies: []
          },
          {
            id: 'requirements',
            name: 'Requirements Analysis',
            description: 'Define migration requirements and success criteria',
            duration: 12,
            status: 'completed',
            dependencies: ['inventory']
          },
          {
            id: 'strategy',
            name: 'Migration Strategy',
            description: 'Develop detailed migration approach and timeline',
            duration: 20,
            status: 'completed',
            dependencies: ['requirements']
          }
        ]
      },
      {
        id: 'preparation',
        name: 'Environment Preparation',
        description: 'Set up target environment and prepare migration tools',
        duration: Math.ceil(totalDays * 0.25),
        startDay: Math.ceil(totalDays * 0.15) + 1,
        endDay: Math.ceil(totalDays * 0.4),
        status: 'in-progress',
        dependencies: ['assessment'],
        riskLevel: 'medium',
        icon: <BuildIcon />,
        tasks: [
          {
            id: 'target-setup',
            name: 'Target Environment Setup',
            description: 'Configure target cloud environment and networking',
            duration: 32,
            status: 'in-progress',
            dependencies: []
          },
          {
            id: 'tools-setup',
            name: 'Migration Tools Setup',
            description: 'Install and configure migration tools and utilities',
            duration: 16,
            status: 'pending',
            dependencies: ['target-setup']
          },
          {
            id: 'security-config',
            name: 'Security Configuration',
            description: 'Configure security policies and access controls',
            duration: 24,
            status: 'pending',
            dependencies: ['target-setup']
          }
        ]
      },
      {
        id: 'migration',
        name: 'Data & Application Migration',
        description: 'Migrate applications, data, and configurations',
        duration: Math.ceil(totalDays * 0.4),
        startDay: Math.ceil(totalDays * 0.4) + 1,
        endDay: Math.ceil(totalDays * 0.8),
        status: 'pending',
        dependencies: ['preparation'],
        riskLevel: 'high',
        icon: <CloudUploadIcon />,
        tasks: [
          {
            id: 'data-migration',
            name: 'Data Migration',
            description: 'Migrate databases and file systems',
            duration: 48,
            status: 'pending',
            dependencies: []
          },
          {
            id: 'app-migration',
            name: 'Application Migration',
            description: 'Migrate applications and services',
            duration: 40,
            status: 'pending',
            dependencies: ['data-migration']
          },
          {
            id: 'config-migration',
            name: 'Configuration Migration',
            description: 'Migrate configurations and settings',
            duration: 16,
            status: 'pending',
            dependencies: ['app-migration']
          }
        ]
      },
      {
        id: 'testing',
        name: 'Testing & Validation',
        description: 'Test migrated systems and validate functionality',
        duration: Math.ceil(totalDays * 0.15),
        startDay: Math.ceil(totalDays * 0.8) + 1,
        endDay: Math.ceil(totalDays * 0.95),
        status: 'pending',
        dependencies: ['migration'],
        riskLevel: 'medium',
        icon: <SecurityIcon />,
        tasks: [
          {
            id: 'functional-testing',
            name: 'Functional Testing',
            description: 'Test application functionality and features',
            duration: 24,
            status: 'pending',
            dependencies: []
          },
          {
            id: 'performance-testing',
            name: 'Performance Testing',
            description: 'Validate performance and scalability',
            duration: 16,
            status: 'pending',
            dependencies: ['functional-testing']
          },
          {
            id: 'security-testing',
            name: 'Security Testing',
            description: 'Verify security controls and compliance',
            duration: 12,
            status: 'pending',
            dependencies: ['functional-testing']
          }
        ]
      },
      {
        id: 'cutover',
        name: 'Go-Live & Cutover',
        description: 'Switch to new environment and decommission old systems',
        duration: Math.ceil(totalDays * 0.05),
        startDay: Math.ceil(totalDays * 0.95) + 1,
        endDay: totalDays,
        status: 'pending',
        dependencies: ['testing'],
        riskLevel: 'high',
        icon: <CheckCircleIcon />,
        tasks: [
          {
            id: 'cutover-prep',
            name: 'Cutover Preparation',
            description: 'Prepare for production cutover',
            duration: 8,
            status: 'pending',
            dependencies: []
          },
          {
            id: 'production-cutover',
            name: 'Production Cutover',
            description: 'Switch to new production environment',
            duration: 4,
            status: 'pending',
            dependencies: ['cutover-prep']
          },
          {
            id: 'decommission',
            name: 'Old System Decommission',
            description: 'Safely decommission old infrastructure',
            duration: 8,
            status: 'pending',
            dependencies: ['production-cutover']
          }
        ]
      }
    ];
  };

  const phases = generateMigrationPhases();

  const getStatusColor = (status: string): 'success' | 'primary' | 'error' | 'secondary' => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'in-progress':
        return 'primary';
      case 'blocked':
        return 'error';
      default:
        return 'secondary';
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  const getPhaseProgress = (phase: MigrationPhase) => {
    const completedTasks = phase.tasks.filter(task => task.status === 'completed').length;
    return (completedTasks / phase.tasks.length) * 100;
  };

  const handlePhaseToggle = (phaseId: string) => {
    setExpandedPhase(expandedPhase === phaseId ? null : phaseId);
  };

  return (
    <Box>
      {/* Timeline Overview */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Migration Timeline Overview
        </Typography>
        <Alert severity="info" sx={{ mb: 2 }}>
          Total Duration: {analysis.migration_timeline_days} days | 
          Estimated Completion: {new Date(Date.now() + analysis.migration_timeline_days * 24 * 60 * 60 * 1000).toLocaleDateString()}
        </Alert>
      </Box>

      {/* Timeline Visualization */}
      <Timeline position="alternate">
        {phases.map((phase, index) => (
          <TimelineItem key={phase.id}>
            <TimelineOppositeContent
              sx={{ m: 'auto 0' }}
              align={index % 2 === 0 ? 'right' : 'left'}
              variant="body2"
              color="text.secondary"
            >
              Day {phase.startDay} - {phase.endDay}
              <br />
              ({phase.duration} days)
            </TimelineOppositeContent>
            
            <TimelineSeparator>
              <TimelineDot color={getStatusColor(phase.status)} variant="outlined">
                {phase.icon}
              </TimelineDot>
              {index < phases.length - 1 && <TimelineConnector />}
            </TimelineSeparator>
            
            <TimelineContent sx={{ py: '12px', px: 2 }}>
              <Card 
                variant="outlined" 
                sx={{ 
                  cursor: 'pointer',
                  '&:hover': { boxShadow: 2 }
                }}
                onClick={() => handlePhaseToggle(phase.id)}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="h6" component="span">
                      {phase.name}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                      <Chip
                        label={phase.status.replace('-', ' ')}
                        color={getStatusColor(phase.status) as any}
                        size="small"
                      />
                      <Chip
                        label={`${phase.riskLevel} risk`}
                        color={getRiskColor(phase.riskLevel) as any}
                        size="small"
                        variant="outlined"
                      />
                      <IconButton size="small">
                        {expandedPhase === phase.id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                      </IconButton>
                    </Box>
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {phase.description}
                  </Typography>
                  
                  <Box sx={{ mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="caption">Progress</Typography>
                      <Typography variant="caption">
                        {Math.round(getPhaseProgress(phase))}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={getPhaseProgress(phase)}
                      color={getStatusColor(phase.status) as any}
                    />
                  </Box>

                  <Collapse in={expandedPhase === phase.id}>
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Tasks ({phase.tasks.length})
                      </Typography>
                      <List dense>
                        {phase.tasks.map((task) => (
                          <ListItem key={task.id} sx={{ pl: 0 }}>
                            <ListItemIcon sx={{ minWidth: 32 }}>
                              {task.status === 'completed' ? (
                                <CheckCircleIcon color="success" fontSize="small" />
                              ) : task.status === 'in-progress' ? (
                                <PlayArrowIcon color="primary" fontSize="small" />
                              ) : (
                                <ScheduleIcon color="disabled" fontSize="small" />
                              )}
                            </ListItemIcon>
                            <ListItemText
                              primary={
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                  <Typography variant="body2">
                                    {task.name}
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {task.duration}h
                                  </Typography>
                                </Box>
                              }
                              secondary={task.description}
                            />
                          </ListItem>
                        ))}
                      </List>
                      
                      {phase.dependencies.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            Dependencies: {phase.dependencies.join(', ')}
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Collapse>
                </CardContent>
              </Card>
            </TimelineContent>
          </TimelineItem>
        ))}
      </Timeline>

      {/* Critical Path Analysis */}
      <Box sx={{ mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Critical Path & Dependencies
        </Typography>
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Critical Dependencies Identified:
          </Typography>
          <Typography variant="body2">
            • Data migration must complete before application migration can begin<br/>
            • Security configuration is required before production cutover<br/>
            • All testing phases must pass before go-live approval
          </Typography>
        </Alert>
      </Box>
    </Box>
  );
};

export default MigrationTimeline;