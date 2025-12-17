import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  Badge,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  CheckCircle as ApproveIcon,
  Cancel as RejectIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  AttachMoney as MoneyIcon,
  Security as SecurityIcon,
  Visibility as ViewIcon,
  Comment as CommentIcon,
  History as HistoryIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';

// Types
interface PendingAction {
  action_id: string;
  action_type: string;
  resource_id: string;
  resource_type: string;
  estimated_monthly_savings: number;
  risk_level: 'low' | 'medium' | 'high';
  scheduled_execution_time: string;
  safety_checks_passed: boolean;
  rollback_plan: any;
  created_at: string;
  requester: string;
  justification: string;
  impact_assessment: {
    affected_resources: string[];
    potential_risks: string[];
    mitigation_steps: string[];
  };
}

interface ApprovalDecision {
  action_id: string;
  decision: 'approved' | 'rejected';
  comments: string;
  conditions?: string[];
}

interface ApprovalHistory {
  approval_id: string;
  action_id: string;
  decision: 'approved' | 'rejected';
  approver: string;
  comments: string;
  timestamp: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`approval-tabpanel-${index}`}
      aria-labelledby={`approval-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const ActionApproval: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedAction, setSelectedAction] = useState<PendingAction | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [approvalDialog, setApprovalDialog] = useState(false);
  const [approvalComments, setApprovalComments] = useState('');
  const [bulkSelection, setBulkSelection] = useState<string[]>([]);
  const [filterRisk, setFilterRisk] = useState<string>('all');
  const queryClient = useQueryClient();

  // Fetch pending actions
  const { data: pendingActions, isLoading: pendingLoading } = useQuery<PendingAction[]>(
    'pending-approvals',
    async () => {
      const response = await fetch('/api/automation/approvals/pending');
      if (!response.ok) throw new Error('Failed to fetch pending approvals');
      return response.json();
    },
    { refetchInterval: 30000 }
  );

  // Fetch approval history
  const { data: approvalHistory, isLoading: historyLoading } = useQuery<ApprovalHistory[]>(
    'approval-history',
    async () => {
      const response = await fetch('/api/automation/approvals/history');
      if (!response.ok) throw new Error('Failed to fetch approval history');
      return response.json();
    }
  );

  // Approve/Reject action mutation
  const approvalMutation = useMutation(
    async (decision: ApprovalDecision) => {
      const response = await fetch(`/api/automation/approvals/${decision.action_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(decision),
      });
      if (!response.ok) throw new Error('Failed to process approval');
      return response.json();
    },
    {
      onSuccess: (_, variables) => {
        queryClient.invalidateQueries('pending-approvals');
        queryClient.invalidateQueries('approval-history');
        toast.success(`Action ${variables.decision} successfully`);
        setApprovalDialog(false);
        setDetailsOpen(false);
        setApprovalComments('');
      },
      onError: () => {
        toast.error('Failed to process approval');
      },
    }
  );

  // Bulk approval mutation
  const bulkApprovalMutation = useMutation(
    async ({ actionIds, decision, comments }: { 
      actionIds: string[], 
      decision: 'approved' | 'rejected', 
      comments: string 
    }) => {
      const response = await fetch('/api/automation/approvals/bulk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action_ids: actionIds, decision, comments }),
      });
      if (!response.ok) throw new Error('Failed to process bulk approval');
      return response.json();
    },
    {
      onSuccess: (_, variables) => {
        queryClient.invalidateQueries('pending-approvals');
        queryClient.invalidateQueries('approval-history');
        toast.success(`${variables.actionIds.length} actions ${variables.decision} successfully`);
        setBulkSelection([]);
      },
      onError: () => {
        toast.error('Failed to process bulk approval');
      },
    }
  );

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleActionDetails = (action: PendingAction) => {
    setSelectedAction(action);
    setDetailsOpen(true);
  };

  const handleApprovalAction = (action: PendingAction, decision: 'approved' | 'rejected') => {
    setSelectedAction(action);
    setApprovalDialog(true);
  };

  const handleSubmitApproval = (decision: 'approved' | 'rejected') => {
    if (selectedAction) {
      approvalMutation.mutate({
        action_id: selectedAction.action_id,
        decision,
        comments: approvalComments,
      });
    }
  };

  const handleBulkSelection = (actionId: string) => {
    setBulkSelection(prev => 
      prev.includes(actionId) 
        ? prev.filter(id => id !== actionId)
        : [...prev, actionId]
    );
  };

  const handleBulkApproval = (decision: 'approved' | 'rejected') => {
    if (bulkSelection.length > 0) {
      bulkApprovalMutation.mutate({
        actionIds: bulkSelection,
        decision,
        comments: `Bulk ${decision} - ${bulkSelection.length} actions`,
      });
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'error';
      default: return 'default';
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const filteredPendingActions = pendingActions?.filter(action => 
    filterRisk === 'all' || action.risk_level === filterRisk
  ) || [];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
            Action Approval Workflow
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Review and approve automated cost optimization actions
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          {bulkSelection.length > 0 && (
            <>
              <Button
                variant="contained"
                color="success"
                onClick={() => handleBulkApproval('approved')}
                disabled={bulkApprovalMutation.isLoading}
              >
                Approve Selected ({bulkSelection.length})
              </Button>
              <Button
                variant="outlined"
                color="error"
                onClick={() => handleBulkApproval('rejected')}
                disabled={bulkApprovalMutation.isLoading}
              >
                Reject Selected ({bulkSelection.length})
              </Button>
            </>
          )}
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Risk Filter</InputLabel>
            <Select
              value={filterRisk}
              onChange={(e) => setFilterRisk(e.target.value)}
              label="Risk Filter"
            >
              <MenuItem value="all">All Risks</MenuItem>
              <MenuItem value="low">Low Risk</MenuItem>
              <MenuItem value="medium">Medium Risk</MenuItem>
              <MenuItem value="high">High Risk</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Badge badgeContent={filteredPendingActions.length} color="warning">
                  <ScheduleIcon color="warning" />
                </Badge>
                <Typography variant="h6" sx={{ ml: 2 }}>
                  Pending Approvals
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 600 }}>
                {filteredPendingActions.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <MoneyIcon color="success" />
                <Typography variant="h6" sx={{ ml: 2 }}>
                  Potential Savings
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                {formatCurrency(
                  filteredPendingActions.reduce((sum, action) => sum + action.estimated_monthly_savings, 0)
                )}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <ErrorIcon color="error" />
                <Typography variant="h6" sx={{ ml: 2 }}>
                  High Risk Actions
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ fontWeight: 600 }}>
                {filteredPendingActions.filter(action => action.risk_level === 'high').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Pending Approvals" />
            <Tab label="Approval History" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          {filteredPendingActions.length === 0 ? (
            <Alert severity="info">
              No pending approvals at this time.
            </Alert>
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell padding="checkbox">
                      <input
                        type="checkbox"
                        checked={bulkSelection.length === filteredPendingActions.length}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setBulkSelection(filteredPendingActions.map(a => a.action_id));
                          } else {
                            setBulkSelection([]);
                          }
                        }}
                      />
                    </TableCell>
                    <TableCell>Action Type</TableCell>
                    <TableCell>Resource</TableCell>
                    <TableCell>Risk Level</TableCell>
                    <TableCell>Estimated Savings</TableCell>
                    <TableCell>Scheduled Time</TableCell>
                    <TableCell>Requester</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredPendingActions.map((action) => (
                    <TableRow key={action.action_id} hover>
                      <TableCell padding="checkbox">
                        <input
                          type="checkbox"
                          checked={bulkSelection.includes(action.action_id)}
                          onChange={() => handleBulkSelection(action.action_id)}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {action.action_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box>
                          <Typography variant="body2">{action.resource_id}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {action.resource_type}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={action.risk_level}
                          color={getRiskColor(action.risk_level) as any}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="success.main">
                          {formatCurrency(action.estimated_monthly_savings)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(action.scheduled_execution_time).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">{action.requester}</Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Tooltip title="View Details">
                            <IconButton
                              size="small"
                              onClick={() => handleActionDetails(action)}
                            >
                              <ViewIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Approve">
                            <IconButton
                              size="small"
                              color="success"
                              onClick={() => handleApprovalAction(action, 'approved')}
                            >
                              <ApproveIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Reject">
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => handleApprovalAction(action, 'rejected')}
                            >
                              <RejectIcon />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Action ID</TableCell>
                  <TableCell>Decision</TableCell>
                  <TableCell>Approver</TableCell>
                  <TableCell>Comments</TableCell>
                  <TableCell>Timestamp</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {approvalHistory?.map((approval) => (
                  <TableRow key={approval.approval_id} hover>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {approval.action_id.substring(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={approval.decision}
                        color={approval.decision === 'approved' ? 'success' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{approval.approver}</Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{approval.comments}</Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(approval.timestamp).toLocaleString()}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
      </Card>

      {/* Action Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Action Details: {selectedAction?.action_type.replace(/_/g, ' ')}
        </DialogTitle>
        <DialogContent>
          {selectedAction && (
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Alert 
                    severity={selectedAction.safety_checks_passed ? 'success' : 'warning'}
                    sx={{ mb: 2 }}
                  >
                    Safety checks {selectedAction.safety_checks_passed ? 'passed' : 'failed'}
                  </Alert>
                </Grid>
                
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Resource ID
                  </Typography>
                  <Typography variant="body1">{selectedAction.resource_id}</Typography>
                </Grid>
                
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Resource Type
                  </Typography>
                  <Typography variant="body1">{selectedAction.resource_type}</Typography>
                </Grid>
                
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Risk Level
                  </Typography>
                  <Chip
                    label={selectedAction.risk_level}
                    color={getRiskColor(selectedAction.risk_level) as any}
                    size="small"
                  />
                </Grid>
                
                <Grid item xs={6}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Estimated Monthly Savings
                  </Typography>
                  <Typography variant="body1" color="success.main">
                    {formatCurrency(selectedAction.estimated_monthly_savings)}
                  </Typography>
                </Grid>
                
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="text.secondary">
                    Justification
                  </Typography>
                  <Typography variant="body1">{selectedAction.justification}</Typography>
                </Grid>
                
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="h6" sx={{ mb: 2 }}>Impact Assessment</Typography>
                  
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                    Affected Resources
                  </Typography>
                  <List dense>
                    {selectedAction.impact_assessment.affected_resources.map((resource, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <InfoIcon color="info" />
                        </ListItemIcon>
                        <ListItemText primary={resource} />
                      </ListItem>
                    ))}
                  </List>
                  
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1, mt: 2 }}>
                    Potential Risks
                  </Typography>
                  <List dense>
                    {selectedAction.impact_assessment.potential_risks.map((risk, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <WarningIcon color="warning" />
                        </ListItemIcon>
                        <ListItemText primary={risk} />
                      </ListItem>
                    ))}
                  </List>
                  
                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1, mt: 2 }}>
                    Mitigation Steps
                  </Typography>
                  <List dense>
                    {selectedAction.impact_assessment.mitigation_steps.map((step, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <SecurityIcon color="success" />
                        </ListItemIcon>
                        <ListItemText primary={step} />
                      </ListItem>
                    ))}
                  </List>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
          <Button
            variant="outlined"
            color="error"
            onClick={() => selectedAction && handleApprovalAction(selectedAction, 'rejected')}
          >
            Reject
          </Button>
          <Button
            variant="contained"
            color="success"
            onClick={() => selectedAction && handleApprovalAction(selectedAction, 'approved')}
          >
            Approve
          </Button>
        </DialogActions>
      </Dialog>

      {/* Approval Decision Dialog */}
      <Dialog
        open={approvalDialog}
        onClose={() => setApprovalDialog(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Approval Decision
        </DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Comments (optional)"
            value={approvalComments}
            onChange={(e) => setApprovalComments(e.target.value)}
            sx={{ mt: 2 }}
            placeholder="Add any comments or conditions for this approval decision..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApprovalDialog(false)}>Cancel</Button>
          <Button
            variant="outlined"
            color="error"
            onClick={() => handleSubmitApproval('rejected')}
            disabled={approvalMutation.isLoading}
          >
            Reject
          </Button>
          <Button
            variant="contained"
            color="success"
            onClick={() => handleSubmitApproval('approved')}
            disabled={approvalMutation.isLoading}
          >
            Approve
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ActionApproval;