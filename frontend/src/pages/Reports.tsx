import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { motion } from 'framer-motion';
import {
  Add,
  Download,
  Schedule,
  Visibility,
  Delete,
  Assessment,
  PictureAsPdf,
  TableChart,
} from '@mui/icons-material';
import numeral from 'numeral';

// Mock data
const reports = [
  {
    id: 1,
    name: 'Monthly Cost Summary',
    type: 'cost_summary',
    schedule: 'Monthly',
    lastRun: '2024-01-15T09:00:00Z',
    nextRun: '2024-02-15T09:00:00Z',
    status: 'active',
    recipients: ['finance@company.com', 'cto@company.com'],
    format: 'PDF',
  },
  {
    id: 2,
    name: 'Budget Performance Report',
    type: 'budget_analysis',
    schedule: 'Weekly',
    lastRun: '2024-01-14T10:00:00Z',
    nextRun: '2024-01-21T10:00:00Z',
    status: 'active',
    recipients: ['budget-managers@company.com'],
    format: 'Excel',
  },
  {
    id: 3,
    name: 'Optimization Opportunities',
    type: 'optimization',
    schedule: 'Daily',
    lastRun: '2024-01-15T08:00:00Z',
    nextRun: '2024-01-16T08:00:00Z',
    status: 'active',
    recipients: ['devops@company.com'],
    format: 'PDF',
  },
  {
    id: 4,
    name: 'Compliance Report',
    type: 'compliance',
    schedule: 'Quarterly',
    lastRun: '2024-01-01T12:00:00Z',
    nextRun: '2024-04-01T12:00:00Z',
    status: 'paused',
    recipients: ['compliance@company.com'],
    format: 'PDF',
  },
];

const reportTemplates = [
  { id: 'cost_summary', name: 'Cost Summary Report', description: 'Comprehensive cost breakdown and trends' },
  { id: 'budget_analysis', name: 'Budget Analysis Report', description: 'Budget performance and variance analysis' },
  { id: 'optimization', name: 'Optimization Report', description: 'Cost optimization opportunities and recommendations' },
  { id: 'compliance', name: 'Compliance Report', description: 'Tagging compliance and governance status' },
  { id: 'chargeback', name: 'Chargeback Report', description: 'Cost allocation by teams and projects' },
];

const Reports: React.FC = () => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newReport, setNewReport] = useState({
    name: '',
    type: '',
    schedule: 'monthly',
    recipients: '',
    format: 'PDF',
  });

  const handleCreateReport = () => {
    console.log('Creating report:', newReport);
    setCreateDialogOpen(false);
    setNewReport({
      name: '',
      type: '',
      schedule: 'monthly',
      recipients: '',
      format: 'PDF',
    });
  };

  const handleDownloadReport = (reportId: number) => {
    console.log('Downloading report:', reportId);
  };

  const handleDeleteReport = (reportId: number) => {
    console.log('Deleting report:', reportId);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return '#4caf50';
      case 'paused':
        return '#ff9800';
      case 'error':
        return '#f44336';
      default:
        return '#9e9e9e';
    }
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'PDF':
        return <PictureAsPdf sx={{ fontSize: 16 }} />;
      case 'Excel':
        return <TableChart sx={{ fontSize: 16 }} />;
      default:
        return <Assessment sx={{ fontSize: 16 }} />;
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Reports & Analytics
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Report
        </Button>
      </Box>

      {/* Report Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Active Reports
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  {reports.filter(r => r.status === 'active').length}
                </Typography>
                <Chip
                  label="Scheduled"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    color: '#4caf50',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Reports Generated
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  127
                </Typography>
                <Chip
                  label="This month"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(33, 150, 243, 0.2)',
                    color: '#2196f3',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Recipients
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  24
                </Typography>
                <Chip
                  label="Unique users"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(156, 39, 176, 0.2)',
                    color: '#9c27b0',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        <Grid item xs={12} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ color: 'text.secondary', mb: 1 }}>
                  Avg Generation Time
                </Typography>
                <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                  2.3s
                </Typography>
                <Chip
                  label="Performance"
                  size="small"
                  sx={{
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    color: '#4caf50',
                  }}
                />
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Reports Table */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 3 }}>
                  Scheduled Reports
                </Typography>
                <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Report Name</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Schedule</TableCell>
                        <TableCell>Last Run</TableCell>
                        <TableCell>Next Run</TableCell>
                        <TableCell align="center">Status</TableCell>
                        <TableCell align="center">Format</TableCell>
                        <TableCell align="center">Recipients</TableCell>
                        <TableCell align="center">Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {reports.map((report) => (
                        <TableRow key={report.id}>
                          <TableCell component="th" scope="row">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {report.name}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={report.type.replace('_', ' ')}
                              size="small"
                              sx={{
                                backgroundColor: 'rgba(33, 150, 243, 0.2)',
                                color: '#2196f3',
                                textTransform: 'capitalize',
                              }}
                            />
                          </TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Schedule sx={{ fontSize: 16, mr: 1, color: 'text.secondary' }} />
                              <Typography variant="body2">
                                {report.schedule}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {new Date(report.lastRun).toLocaleDateString()}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {new Date(report.lastRun).toLocaleTimeString()}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {new Date(report.nextRun).toLocaleDateString()}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              {new Date(report.nextRun).toLocaleTimeString()}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={report.status}
                              size="small"
                              sx={{
                                backgroundColor: `${getStatusColor(report.status)}20`,
                                color: getStatusColor(report.status),
                                textTransform: 'capitalize',
                              }}
                            />
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                              {getFormatIcon(report.format)}
                              <Typography variant="body2" sx={{ ml: 1 }}>
                                {report.format}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            <Typography variant="body2">
                              {report.recipients.length} recipients
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                              <Tooltip title="Download Latest">
                                <IconButton 
                                  size="small" 
                                  onClick={() => handleDownloadReport(report.id)}
                                >
                                  <Download sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="View Report">
                                <IconButton size="small">
                                  <Visibility sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Delete Report">
                                <IconButton 
                                  size="small" 
                                  onClick={() => handleDeleteReport(report.id)}
                                >
                                  <Delete sx={{ fontSize: 16, color: '#f44336' }} />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Create Report Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Report</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Report Name"
                value={newReport.name}
                onChange={(e) => setNewReport({ ...newReport, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Report Type</InputLabel>
                <Select
                  value={newReport.type}
                  label="Report Type"
                  onChange={(e) => setNewReport({ ...newReport, type: e.target.value })}
                >
                  {reportTemplates.map((template) => (
                    <MenuItem key={template.id} value={template.id}>
                      <Box>
                        <Typography variant="body2">{template.name}</Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          {template.description}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Schedule</InputLabel>
                <Select
                  value={newReport.schedule}
                  label="Schedule"
                  onChange={(e) => setNewReport({ ...newReport, schedule: e.target.value })}
                >
                  <MenuItem value="daily">Daily</MenuItem>
                  <MenuItem value="weekly">Weekly</MenuItem>
                  <MenuItem value="monthly">Monthly</MenuItem>
                  <MenuItem value="quarterly">Quarterly</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Format</InputLabel>
                <Select
                  value={newReport.format}
                  label="Format"
                  onChange={(e) => setNewReport({ ...newReport, format: e.target.value })}
                >
                  <MenuItem value="PDF">PDF</MenuItem>
                  <MenuItem value="Excel">Excel</MenuItem>
                  <MenuItem value="CSV">CSV</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Recipients (comma-separated emails)"
                value={newReport.recipients}
                onChange={(e) => setNewReport({ ...newReport, recipients: e.target.value })}
                placeholder="user1@company.com, user2@company.com"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleCreateReport} variant="contained">
            Create Report
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Reports;