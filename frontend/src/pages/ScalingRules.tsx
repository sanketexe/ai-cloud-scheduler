import React, { useState } from 'react';
import {
    Box,
    Typography,
    Grid,
    Card,
    CardContent,
    Chip,
    Button,
    IconButton,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Tabs,
    Tab,
    Switch,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    CircularProgress,
    Tooltip,
    Alert,
    TextField,
    MenuItem,
    Select,
    FormControl,
    InputLabel,
    Divider,
    InputAdornment,
} from '@mui/material';
import {
    Add as AddIcon,
    Refresh as RefreshIcon,
    Delete as DeleteIcon,
    PlayArrow as RunIcon,
    Science as TestIcon,
    Storage as StorageIcon,
    Computer as Ec2Icon,
    DataUsage as RdsIcon,
    AutoMode as AutoIcon,
    TrendingUp as TrendingUpIcon,
    CheckCircle as SuccessIcon,
    Error as ErrorIcon,
    Warning as WarningIcon,
    History as HistoryIcon,
    Speed as SpeedIcon,
    Rule as RuleIcon,
    ExpandMore as ExpandIcon,
    Edit as EditIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import toast from 'react-hot-toast';
import {
    scalingRulesApi,
    ScalingRule,
    ScalingRuleCreate,
    RuleExecution,
    ScalingStats,
} from '../services/scalingRulesApi';

// ── Tab Panel ────────────────────────────────────────
interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
    return (
        <div role="tabpanel" hidden={value !== index}>
            {value === index && <Box sx={{ pt: 3 }}>{children}</Box>}
        </div>
    );
}

// ── Template presets ──────────────────────────────────
const RULE_TEMPLATES = [
    {
        label: 'EBS — Increase Storage',
        value: {
            service_type: 'ebs',
            metric_namespace: 'AWS/EBS',
            metric_name: 'VolumeQueueLength',
            metric_dimension_name: 'VolumeId',
            metric_statistic: 'Average',
            threshold_operator: 'gt',
            threshold_value: 5,
            scaling_direction: 'scale_up',
            scaling_action: { action: 'increase_storage', amount_gb: 4 },
            max_scaling_limit: { max_size_gb: 100 },
        },
    },
    {
        label: 'EC2 — Resize Instance',
        value: {
            service_type: 'ec2',
            metric_namespace: 'AWS/EC2',
            metric_name: 'CPUUtilization',
            metric_dimension_name: 'InstanceId',
            metric_statistic: 'Average',
            threshold_operator: 'gt',
            threshold_value: 80,
            scaling_direction: 'scale_up',
            scaling_action: { action: 'resize_instance', target_instance_type: 'm5.xlarge' },
            max_scaling_limit: { max_instance_type: 'm5.4xlarge' },
        },
    },
    {
        label: 'RDS — Scale DB Instance',
        value: {
            service_type: 'rds',
            metric_namespace: 'AWS/RDS',
            metric_name: 'CPUUtilization',
            metric_dimension_name: 'DBInstanceIdentifier',
            metric_statistic: 'Average',
            threshold_operator: 'gt',
            threshold_value: 85,
            scaling_direction: 'scale_up',
            scaling_action: { action: 'resize_db_instance', target_db_instance_class: 'db.m5.large' },
            max_scaling_limit: { max_instance_class: 'db.m5.4xlarge' },
        },
    },
];

const OPERATOR_LABELS: Record<string, string> = {
    gt: '> Greater than',
    lt: '< Less than',
    gte: '≥ Greater or equal',
    lte: '≤ Less or equal',
};

const SERVICE_ICONS: Record<string, React.ReactElement> = {
    ebs: <StorageIcon sx={{ fontSize: 18 }} />,
    ec2: <Ec2Icon sx={{ fontSize: 18 }} />,
    rds: <RdsIcon sx={{ fontSize: 18 }} />,
    asg: <AutoIcon sx={{ fontSize: 18 }} />,
};

const SERVICE_COLORS: Record<string, string> = {
    ebs: '#ff9800',
    ec2: '#2196f3',
    rds: '#9c27b0',
    asg: '#4caf50',
};

// ═══════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════
const ScalingRules: React.FC = () => {
    const [tabValue, setTabValue] = useState(0);
    const [createOpen, setCreateOpen] = useState(false);
    const [selectedTemplate, setSelectedTemplate] = useState(0);
    const queryClient = useQueryClient();

    // Form state
    const [form, setForm] = useState<ScalingRuleCreate>({
        name: '',
        description: '',
        service_type: 'ebs',
        resource_filter: {},
        metric_namespace: 'AWS/EBS',
        metric_name: 'VolumeQueueLength',
        metric_dimension_name: 'VolumeId',
        metric_statistic: 'Average',
        threshold_operator: 'gt',
        threshold_value: 5,
        evaluation_periods: 3,
        evaluation_interval_seconds: 300,
        scaling_direction: 'scale_up',
        scaling_action: { action: 'increase_storage', amount_gb: 4 },
        max_scaling_limit: { max_size_gb: 100 },
        cooldown_seconds: 1800,
        is_enabled: true,
    });

    const [resourceIdsInput, setResourceIdsInput] = useState('');

    // ── Queries ──
    const { data: rules = [], isLoading: rulesLoading } = useQuery<ScalingRule[]>(
        'scaling-rules',
        scalingRulesApi.getRules,
        { refetchInterval: 15000 }
    );

    const { data: stats } = useQuery<ScalingStats>(
        'scaling-rules-stats',
        scalingRulesApi.getStats,
        { refetchInterval: 15000 }
    );

    const { data: allExecutions = [] } = useQuery<RuleExecution[]>(
        'scaling-rules-executions',
        scalingRulesApi.getAllExecutions,
        { refetchInterval: 10000 }
    );

    // ── Mutations ──
    const createMutation = useMutation(
        (data: ScalingRuleCreate) => scalingRulesApi.createRule(data),
        {
            onSuccess: () => {
                toast.success('Scaling rule created!');
                queryClient.invalidateQueries('scaling-rules');
                queryClient.invalidateQueries('scaling-rules-stats');
                setCreateOpen(false);
            },
            onError: () => { toast.error('Failed to create rule'); },
        }
    );

    const deleteMutation = useMutation(
        (id: string) => scalingRulesApi.deleteRule(id),
        {
            onSuccess: () => {
                toast.success('Rule deleted');
                queryClient.invalidateQueries('scaling-rules');
                queryClient.invalidateQueries('scaling-rules-stats');
            },
            onError: () => { toast.error('Failed to delete rule'); },
        }
    );

    const toggleMutation = useMutation(
        (id: string) => scalingRulesApi.toggleRule(id),
        {
            onSuccess: () => {
                queryClient.invalidateQueries('scaling-rules');
                queryClient.invalidateQueries('scaling-rules-stats');
            },
        }
    );

    const evaluateMutation = useMutation(
        () => scalingRulesApi.evaluateAll(),
        {
            onSuccess: (data) => {
                const triggered = data.rules_triggered || 0;
                toast.success(`Evaluation complete: ${triggered} rule(s) triggered`);
                queryClient.invalidateQueries('scaling-rules');
                queryClient.invalidateQueries('scaling-rules-stats');
                queryClient.invalidateQueries('scaling-rules-executions');
            },
            onError: () => { toast.error('Evaluation failed'); },
        }
    );

    const testMutation = useMutation(
        (id: string) => scalingRulesApi.testRule(id),
        {
            onSuccess: (data) => {
                const breached = data.resources_breached || 0;
                toast.success(`Dry run: ${breached} of ${data.resources_checked} resource(s) would trigger`);
            },
            onError: () => { toast.error('Dry run failed'); },
        }
    );

    // ── Handlers ──
    const applyTemplate = (idx: number) => {
        setSelectedTemplate(idx);
        const tmpl = RULE_TEMPLATES[idx].value;
        setForm(prev => ({
            ...prev,
            ...tmpl,
        }));
    };

    const handleCreate = () => {
        const data = { ...form };
        // Parse resource IDs
        if (resourceIdsInput.trim()) {
            data.resource_filter = {
                ...data.resource_filter,
                resource_ids: resourceIdsInput.split(',').map(s => s.trim()).filter(Boolean),
            };
        }
        createMutation.mutate(data);
    };

    const openCreateDialog = () => {
        applyTemplate(0);
        setForm(prev => ({ ...prev, name: '', description: '' }));
        setResourceIdsInput('');
        setCreateOpen(true);
    };

    const formatCurrency = (n: number) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n);

    const formatAction = (rule: ScalingRule) => {
        const action = rule.scaling_action;
        if (action.action === 'increase_storage') return `+${action.amount_gb || 0} GB storage`;
        if (action.action === 'resize_instance') return `→ ${action.target_instance_type}`;
        if (action.action === 'resize_db_instance') return `→ ${action.target_db_instance_class}`;
        return JSON.stringify(action);
    };

    const formatOperator = (op: string) => {
        const symbols: Record<string, string> = { gt: '>', lt: '<', gte: '≥', lte: '≤' };
        return symbols[op] || op;
    };

    // ═══════════════════════════════════════════════════
    // Render
    // ═══════════════════════════════════════════════════
    return (
        <Box>
            {/* Page Header */}
            <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        <AutoIcon sx={{ mr: 1, verticalAlign: 'bottom', color: '#ff9800' }} />
                        Auto-Scaling Rules
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mt: 0.5 }}>
                        Pre-authorize automatic resource scaling — set rules, define limits, let the system handle the rest
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                        variant="outlined"
                        startIcon={<RefreshIcon />}
                        onClick={() => {
                            queryClient.invalidateQueries('scaling-rules');
                            queryClient.invalidateQueries('scaling-rules-stats');
                            queryClient.invalidateQueries('scaling-rules-executions');
                        }}
                    >
                        Refresh
                    </Button>
                    <Tooltip title="Evaluate all enabled rules now against current CloudWatch metrics">
                        <Button
                            variant="outlined"
                            color="warning"
                            startIcon={<RunIcon />}
                            onClick={() => evaluateMutation.mutate()}
                            disabled={evaluateMutation.isLoading}
                        >
                            {evaluateMutation.isLoading ? 'Evaluating...' : 'Run All Rules'}
                        </Button>
                    </Tooltip>
                    <Button
                        variant="contained"
                        startIcon={<AddIcon />}
                        onClick={openCreateDialog}
                        sx={{
                            background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
                            '&:hover': { background: 'linear-gradient(135deg, #f57c00 0%, #e65100 100%)' },
                        }}
                    >
                        New Rule
                    </Button>
                </Box>
            </Box>

            {/* Stats Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #1a237e 0%, #283593 100%)' }}>
                        <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="caption" color="text.secondary">Active Rules</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                                        {stats?.active_rules ?? 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        of {stats?.total_rules ?? 0} total
                                    </Typography>
                                </Box>
                                <RuleIcon sx={{ fontSize: 48, color: 'rgba(255,152,0,0.4)' }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #004d40 0%, #00695c 100%)' }}>
                        <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="caption" color="text.secondary">Total Executions</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                                        {stats?.total_executions ?? 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        {stats?.success_rate ?? 0}% success rate
                                    </Typography>
                                </Box>
                                <SpeedIcon sx={{ fontSize: 48, color: 'rgba(105,240,174,0.3)' }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%)' }}>
                        <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="caption" color="text.secondary">Successful</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700 }}>
                                        {stats?.successful_executions ?? 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        scaling actions
                                    </Typography>
                                </Box>
                                <SuccessIcon sx={{ fontSize: 48, color: 'rgba(76,175,80,0.3)' }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                    <Card sx={{ background: 'linear-gradient(135deg, #3e2723 0%, #4e342e 100%)' }}>
                        <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="caption" color="text.secondary">Cost Impact</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#ff9800' }}>
                                        {formatCurrency(stats?.total_cost_impact ?? 0)}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        monthly estimate
                                    </Typography>
                                </Box>
                                <TrendingUpIcon sx={{ fontSize: 48, color: 'rgba(255,152,0,0.3)' }} />
                            </Box>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Tabs */}
            <Tabs
                value={tabValue}
                onChange={(_e, v) => setTabValue(v)}
                sx={{
                    '& .MuiTab-root': { textTransform: 'none', fontWeight: 600 },
                    borderBottom: '1px solid rgba(255,255,255,0.1)',
                }}
            >
                <Tab icon={<RuleIcon />} iconPosition="start" label="Rules" />
                <Tab icon={<HistoryIcon />} iconPosition="start" label="Execution History" />
            </Tabs>

            {/* ── Tab 0: Rules ─────────────────────────────────── */}
            <TabPanel value={tabValue} index={0}>
                {rulesLoading ? (
                    <Box sx={{ textAlign: 'center', py: 6 }}><CircularProgress /></Box>
                ) : rules.length === 0 ? (
                    <Alert severity="info" sx={{ mt: 2 }}>
                        No scaling rules configured yet. Click <strong>New Rule</strong> to create your first auto-scaling rule.
                    </Alert>
                ) : (
                    <TableContainer component={Paper} sx={{ background: 'transparent' }}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>Rule Name</TableCell>
                                    <TableCell>Service</TableCell>
                                    <TableCell>Metric</TableCell>
                                    <TableCell>Threshold</TableCell>
                                    <TableCell>Action</TableCell>
                                    <TableCell align="center">Triggered</TableCell>
                                    <TableCell align="center">Last Run</TableCell>
                                    <TableCell align="center">Enabled</TableCell>
                                    <TableCell align="center">Actions</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {rules.map((rule) => (
                                    <TableRow key={rule.id} hover sx={{ opacity: rule.is_enabled ? 1 : 0.5 }}>
                                        <TableCell>
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>{rule.name}</Typography>
                                            {rule.description && (
                                                <Typography variant="caption" color="text.secondary">
                                                    {rule.description}
                                                </Typography>
                                            )}
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                icon={SERVICE_ICONS[rule.service_type] || <AutoIcon sx={{ fontSize: 18 }} />}
                                                label={rule.service_type.toUpperCase()}
                                                size="small"
                                                sx={{
                                                    bgcolor: `${SERVICE_COLORS[rule.service_type] || '#757575'}22`,
                                                    color: SERVICE_COLORS[rule.service_type] || '#757575',
                                                    fontWeight: 600,
                                                }}
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <Typography variant="body2">{rule.metric_name}</Typography>
                                            <Typography variant="caption" color="text.secondary">{rule.metric_namespace}</Typography>
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                label={`${formatOperator(rule.threshold_operator)} ${rule.threshold_value}`}
                                                size="small"
                                                variant="outlined"
                                                sx={{ fontFamily: 'monospace', fontWeight: 600 }}
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                label={formatAction(rule)}
                                                size="small"
                                                sx={{
                                                    bgcolor: rule.scaling_direction === 'scale_up' ? 'rgba(76,175,80,0.15)' : 'rgba(244,67,54,0.15)',
                                                    color: rule.scaling_direction === 'scale_up' ? '#66bb6a' : '#ef5350',
                                                    fontWeight: 600,
                                                }}
                                            />
                                        </TableCell>
                                        <TableCell align="center">
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                {rule.trigger_count}×
                                            </Typography>
                                        </TableCell>
                                        <TableCell align="center">
                                            {rule.last_triggered_at ? (
                                                <Tooltip title={new Date(rule.last_triggered_at).toLocaleString()}>
                                                    <Typography variant="body2">
                                                        {new Date(rule.last_triggered_at).toLocaleDateString()}
                                                    </Typography>
                                                </Tooltip>
                                            ) : (
                                                <Typography variant="caption" color="text.secondary">Never</Typography>
                                            )}
                                        </TableCell>
                                        <TableCell align="center">
                                            <Switch
                                                checked={rule.is_enabled}
                                                size="small"
                                                onChange={() => toggleMutation.mutate(rule.id)}
                                            />
                                        </TableCell>
                                        <TableCell align="center">
                                            <Box sx={{ display: 'flex', gap: 0.5, justifyContent: 'center' }}>
                                                <Tooltip title="Dry-Run Test">
                                                    <IconButton
                                                        size="small"
                                                        sx={{ color: '#7c4dff' }}
                                                        onClick={() => testMutation.mutate(rule.id)}
                                                    >
                                                        <TestIcon />
                                                    </IconButton>
                                                </Tooltip>
                                                <Tooltip title="Delete Rule">
                                                    <IconButton
                                                        size="small"
                                                        color="error"
                                                        onClick={() => deleteMutation.mutate(rule.id)}
                                                    >
                                                        <DeleteIcon />
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

            {/* ── Tab 1: Execution History ──────────────────────── */}
            <TabPanel value={tabValue} index={1}>
                {allExecutions.length === 0 ? (
                    <Alert severity="info">
                        No scaling actions executed yet. Create rules and run them to see execution history here.
                    </Alert>
                ) : (
                    <TableContainer component={Paper} sx={{ background: 'transparent' }}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>Time</TableCell>
                                    <TableCell>Rule</TableCell>
                                    <TableCell>Resource</TableCell>
                                    <TableCell>Metric Value</TableCell>
                                    <TableCell>Action</TableCell>
                                    <TableCell>Status</TableCell>
                                    <TableCell>Cost Impact</TableCell>
                                    <TableCell>Duration</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {allExecutions.map((exec) => (
                                    <TableRow key={exec.id} hover>
                                        <TableCell>
                                            <Typography variant="body2">
                                                {new Date(exec.triggered_at).toLocaleString()}
                                            </Typography>
                                        </TableCell>
                                        <TableCell>
                                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                                {exec.rule_name}
                                            </Typography>
                                        </TableCell>
                                        <TableCell>
                                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                                                {exec.resource_id}
                                            </Typography>
                                        </TableCell>
                                        <TableCell>
                                            <Typography variant="body2">
                                                {exec.metric_value_at_trigger.toFixed(1)} (threshold: {exec.threshold_value})
                                            </Typography>
                                        </TableCell>
                                        <TableCell>
                                            <Typography variant="body2">
                                                {exec.action_taken?.action || 'N/A'}
                                            </Typography>
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                icon={exec.status === 'success' ? <SuccessIcon /> : exec.status === 'failed' ? <ErrorIcon /> : <WarningIcon />}
                                                label={exec.status}
                                                size="small"
                                                color={exec.status === 'success' ? 'success' : exec.status === 'failed' ? 'error' : 'warning'}
                                            />
                                        </TableCell>
                                        <TableCell>
                                            {exec.cost_impact !== null
                                                ? formatCurrency(exec.cost_impact)
                                                : '—'
                                            }
                                        </TableCell>
                                        <TableCell>
                                            {exec.execution_duration_ms !== null
                                                ? `${exec.execution_duration_ms}ms`
                                                : '—'
                                            }
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )}
            </TabPanel>

            {/* ═══════════════════════════════════════════════════ */}
            {/* Create Rule Dialog                                 */}
            {/* ═══════════════════════════════════════════════════ */}
            <Dialog
                open={createOpen}
                onClose={() => setCreateOpen(false)}
                maxWidth="md"
                fullWidth
                PaperProps={{
                    sx: {
                        background: 'linear-gradient(135deg, #1a1d3a 0%, #2a2d5a 100%)',
                        border: '1px solid rgba(255,152,0,0.3)',
                    },
                }}
            >
                <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <AutoIcon sx={{ color: '#ff9800' }} />
                    Create Auto-Scaling Rule
                </DialogTitle>
                <DialogContent>
                    {/* Template selector */}
                    <Alert severity="info" sx={{ mb: 3, mt: 1 }}>
                        Choose a template to get started, then customize the values.
                    </Alert>
                    <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
                        {RULE_TEMPLATES.map((tmpl, idx) => (
                            <Chip
                                key={idx}
                                label={tmpl.label}
                                onClick={() => applyTemplate(idx)}
                                color={selectedTemplate === idx ? 'primary' : 'default'}
                                variant={selectedTemplate === idx ? 'filled' : 'outlined'}
                                sx={{ cursor: 'pointer' }}
                            />
                        ))}
                    </Box>

                    <Grid container spacing={2}>
                        {/* Name & Description */}
                        <Grid item xs={8}>
                            <TextField
                                fullWidth label="Rule Name" size="small" required
                                value={form.name}
                                onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                                placeholder="e.g. Increase production EBS storage"
                            />
                        </Grid>
                        <Grid item xs={4}>
                            <FormControl fullWidth size="small">
                                <InputLabel>Service</InputLabel>
                                <Select
                                    value={form.service_type}
                                    label="Service"
                                    onChange={e => setForm(p => ({ ...p, service_type: e.target.value }))}
                                >
                                    <MenuItem value="ebs">EBS (Storage)</MenuItem>
                                    <MenuItem value="ec2">EC2 (Compute)</MenuItem>
                                    <MenuItem value="rds">RDS (Database)</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth label="Description" size="small" multiline rows={2}
                                value={form.description}
                                onChange={e => setForm(p => ({ ...p, description: e.target.value }))}
                                placeholder="What this rule does..."
                            />
                        </Grid>

                        <Grid item xs={12}>
                            <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                                <Typography variant="caption" color="text.secondary">METRIC & THRESHOLD</Typography>
                            </Divider>
                        </Grid>

                        {/* Metric */}
                        <Grid item xs={4}>
                            <TextField
                                fullWidth label="Metric Namespace" size="small"
                                value={form.metric_namespace}
                                onChange={e => setForm(p => ({ ...p, metric_namespace: e.target.value }))}
                            />
                        </Grid>
                        <Grid item xs={4}>
                            <TextField
                                fullWidth label="Metric Name" size="small"
                                value={form.metric_name}
                                onChange={e => setForm(p => ({ ...p, metric_name: e.target.value }))}
                            />
                        </Grid>
                        <Grid item xs={4}>
                            <TextField
                                fullWidth label="Dimension Name" size="small"
                                value={form.metric_dimension_name}
                                onChange={e => setForm(p => ({ ...p, metric_dimension_name: e.target.value }))}
                            />
                        </Grid>

                        {/* Threshold */}
                        <Grid item xs={3}>
                            <FormControl fullWidth size="small">
                                <InputLabel>Operator</InputLabel>
                                <Select
                                    value={form.threshold_operator}
                                    label="Operator"
                                    onChange={e => setForm(p => ({ ...p, threshold_operator: e.target.value }))}
                                >
                                    {Object.entries(OPERATOR_LABELS).map(([k, v]) => (
                                        <MenuItem key={k} value={k}>{v}</MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={3}>
                            <TextField
                                fullWidth label="Threshold Value" size="small" type="number"
                                value={form.threshold_value}
                                onChange={e => setForm(p => ({ ...p, threshold_value: Number(e.target.value) }))}
                            />
                        </Grid>
                        <Grid item xs={3}>
                            <TextField
                                fullWidth label="Eval Periods" size="small" type="number"
                                value={form.evaluation_periods}
                                onChange={e => setForm(p => ({ ...p, evaluation_periods: Number(e.target.value) }))}
                                helperText="Consecutive breaches"
                            />
                        </Grid>
                        <Grid item xs={3}>
                            <TextField
                                fullWidth label="Interval (sec)" size="small" type="number"
                                value={form.evaluation_interval_seconds}
                                onChange={e => setForm(p => ({ ...p, evaluation_interval_seconds: Number(e.target.value) }))}
                                helperText="Between checks"
                            />
                        </Grid>

                        <Grid item xs={12}>
                            <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                                <Typography variant="caption" color="text.secondary">SCALING ACTION</Typography>
                            </Divider>
                        </Grid>

                        {/* Action config — dynamic based on service type */}
                        {form.service_type === 'ebs' && (
                            <>
                                <Grid item xs={6}>
                                    <TextField
                                        fullWidth label="Amount to Add (GB)" size="small" type="number"
                                        value={form.scaling_action.amount_gb || 4}
                                        onChange={e => setForm(p => ({
                                            ...p,
                                            scaling_action: { ...p.scaling_action, action: 'increase_storage', amount_gb: Number(e.target.value) }
                                        }))}
                                        InputProps={{ endAdornment: <InputAdornment position="end">GB</InputAdornment> }}
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        fullWidth label="Max Size Limit (GB)" size="small" type="number"
                                        value={(form.max_scaling_limit || {}).max_size_gb || 100}
                                        onChange={e => setForm(p => ({
                                            ...p,
                                            max_scaling_limit: { max_size_gb: Number(e.target.value) }
                                        }))}
                                        InputProps={{ endAdornment: <InputAdornment position="end">GB</InputAdornment> }}
                                    />
                                </Grid>
                            </>
                        )}

                        {form.service_type === 'ec2' && (
                            <>
                                <Grid item xs={6}>
                                    <TextField
                                        fullWidth label="Target Instance Type" size="small"
                                        value={form.scaling_action.target_instance_type || 'm5.xlarge'}
                                        onChange={e => setForm(p => ({
                                            ...p,
                                            scaling_action: { action: 'resize_instance', target_instance_type: e.target.value }
                                        }))}
                                        placeholder="e.g. m5.xlarge"
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        fullWidth label="Max Instance Type Limit" size="small"
                                        value={(form.max_scaling_limit || {}).max_instance_type || 'm5.4xlarge'}
                                        onChange={e => setForm(p => ({
                                            ...p,
                                            max_scaling_limit: { max_instance_type: e.target.value }
                                        }))}
                                        placeholder="e.g. m5.4xlarge"
                                    />
                                </Grid>
                            </>
                        )}

                        {form.service_type === 'rds' && (
                            <>
                                <Grid item xs={6}>
                                    <TextField
                                        fullWidth label="Target DB Instance Class" size="small"
                                        value={form.scaling_action.target_db_instance_class || 'db.m5.large'}
                                        onChange={e => setForm(p => ({
                                            ...p,
                                            scaling_action: { action: 'resize_db_instance', target_db_instance_class: e.target.value }
                                        }))}
                                        placeholder="e.g. db.m5.large"
                                    />
                                </Grid>
                                <Grid item xs={6}>
                                    <TextField
                                        fullWidth label="Max DB Instance Class Limit" size="small"
                                        value={(form.max_scaling_limit || {}).max_instance_class || 'db.m5.4xlarge'}
                                        onChange={e => setForm(p => ({
                                            ...p,
                                            max_scaling_limit: { max_instance_class: e.target.value }
                                        }))}
                                        placeholder="e.g. db.m5.4xlarge"
                                    />
                                </Grid>
                            </>
                        )}

                        <Grid item xs={12}>
                            <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                                <Typography variant="caption" color="text.secondary">TARGET RESOURCES</Typography>
                            </Divider>
                        </Grid>

                        <Grid item xs={8}>
                            <TextField
                                fullWidth label="Resource IDs (comma-separated)" size="small"
                                value={resourceIdsInput}
                                onChange={e => setResourceIdsInput(e.target.value)}
                                placeholder="e.g. vol-0abc123, vol-0def456"
                                helperText="Leave empty to target all resources matching filters"
                            />
                        </Grid>
                        <Grid item xs={4}>
                            <TextField
                                fullWidth label="Cooldown (seconds)" size="small" type="number"
                                value={form.cooldown_seconds}
                                onChange={e => setForm(p => ({ ...p, cooldown_seconds: Number(e.target.value) }))}
                                helperText="Min wait between triggers"
                            />
                        </Grid>
                    </Grid>
                </DialogContent>
                <DialogActions sx={{ px: 3, pb: 2 }}>
                    <Button onClick={() => setCreateOpen(false)}>Cancel</Button>
                    <Button
                        variant="contained"
                        onClick={handleCreate}
                        disabled={!form.name || createMutation.isLoading}
                        sx={{
                            background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
                            '&:hover': { background: 'linear-gradient(135deg, #f57c00 0%, #e65100 100%)' },
                        }}
                    >
                        {createMutation.isLoading ? 'Creating...' : 'Create Rule'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default ScalingRules;
