import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  FilterList,
  Download,
  Refresh,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import numeral from 'numeral';
import dayjs from 'dayjs';

// Mock data
const costAnalysisData = [
  { date: '2024-01-01', compute: 1200, storage: 800, network: 300, database: 500, total: 2800 },
  { date: '2024-01-02', compute: 1350, storage: 820, network: 280, database: 520, total: 2970 },
  { date: '2024-01-03', compute: 1180, storage: 790, network: 320, database: 480, total: 2770 },
  { date: '2024-01-04', compute: 1420, storage: 850, network: 310, database: 540, total: 3120 },
  { date: '2024-01-05', compute: 1290, storage: 810, network: 290, database: 510, total: 2900 },
  { date: '2024-01-06', compute: 1380, storage: 830, network: 300, database: 530, total: 3040 },
  { date: '2024-01-07', compute: 1250, storage: 800, network: 285, database: 495, total: 2830 },
];

const teamCostData = [
  { team: 'Engineering', cost: 18500, percentage: 42, trend: 'up', change: 8.5 },
  { team: 'Data Science', cost: 12300, percentage: 28, trend: 'down', change: -3.2 },
  { team: 'DevOps', cost: 8900, percentage: 20, trend: 'up', change: 12.1 },
  { team: 'QA', cost: 4400, percentage: 10, trend: 'stable', change: 1.2 },
];

const serviceBreakdown = [
  { name: 'EC2 Instances', cost: 15830, percentage: 35, color: '#ff9800' },
  { name: 'S3 Storage', cost: 11308, percentage: 25, color: '#4caf50' },
  { name: 'RDS Database', cost: 9046, percentage: 20, color: '#2196f3' },
  { name: 'Lambda Functions', cost: 4523, percentage: 10, color: '#9c27b0' },
  { name: 'CloudFront CDN', cost: 2261, percentage: 5, color: '#607d8b' },
  { name: 'Other Services', cost: 2262, percentage: 5, color: '#795548' },
];

const CostAnalysis: React.FC = () => {
  const [dateRange, setDateRange] = useState({
    start: dayjs().subtract(7, 'day'),
    end: dayjs(),
  });
  const [groupBy, setGroupBy] = useState('service');
  const [filterBy, setFilterBy] = useState('all');
  const [loading, setLoading] = useState(false);

  const handleRefresh = () => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  };

  const handleExport = () => {
    // Implement export functionality
    console.log('Exporting cost analysis data...');
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <TrendingUp sx={{ color: '#f44336', fontSize: 16 }} />;
      case 'down':
        return <TrendingDown sx={{ color: '#4caf50', fontSize: 16 }} />;
      default:
        return <TrendingUp sx={{ color: '#9e9e9e', fontSize: 16 }} />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up':
        return '#f44336';
      case 'down':
        return '#4caf50';
      default:
        return '#9e9e9e';
    }
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDayjs}>
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            Cost Analysis
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleRefresh}
              disabled={loading}
            >
              Refresh
            </Button>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleExport}
            >
              Export
            </Button>
          </Box>
        </Box>

        {/* Filters */}
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3 }}>
              <FilterList sx={{ mr: 1, verticalAlign: 'middle' }} />
              Filters & Options
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={3}>
                <DatePicker
                  label="Start Date"
                  value={dateRange.start}
                  onChange={(newValue) => setDateRange({ ...dateRange, start: newValue! })}
                  renderInput={(params) => <TextField {...params} fullWidth />}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <DatePicker
                  label="End Date"
                  value={dateRange.end}
                  onChange={(newValue) => setDateRange({ ...dateRange, end: newValue! })}
                  renderInput={(params) => <TextField {...params} fullWidth />}
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Group By</InputLabel>
                  <Select
                    value={groupBy}
                    label="Group By"
                    onChange={(e) => setGroupBy(e.target.value)}
                  >
                    <MenuItem value="service">Service</MenuItem>
                    <MenuItem value="team">Team</MenuItem>
                    <MenuItem value="project">Project</MenuItem>
                    <MenuItem value="region">Region</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={3}>
                <FormControl fullWidth>
                  <InputLabel>Filter By</InputLabel>
                  <Select
                    value={filterBy}
                    label="Filter By"
                    onChange={(e) => setFilterBy(e.target.value)}
                  >
                    <MenuItem value="all">All Resources</MenuItem>
                    <MenuItem value="production">Production Only</MenuItem>
                    <MenuItem value="development">Development Only</MenuItem>
                    <MenuItem value="tagged">Tagged Resources</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Cost Trend Chart */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} lg={8}>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 3 }}>
                    Daily Cost Breakdown
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={costAnalysisData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="date" stroke="#b0bec5" />
                      <YAxis stroke="#b0bec5" tickFormatter={(value) => numeral(value).format('$0a')} />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: '#1a1d3a',
                          border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: '8px',
                        }}
                        formatter={(value: any) => [numeral(value).format('$0,0'), '']}
                      />
                      <Bar dataKey="compute" stackId="a" fill="#ff9800" name="Compute" />
                      <Bar dataKey="storage" stackId="a" fill="#4caf50" name="Storage" />
                      <Bar dataKey="network" stackId="a" fill="#2196f3" name="Network" />
                      <Bar dataKey="database" stackId="a" fill="#9c27b0" name="Database" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          <Grid item xs={12} lg={4}>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 3 }}>
                    Service Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={serviceBreakdown}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="percentage"
                      >
                        {serviceBreakdown.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: '#1a1d3a',
                          border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: '8px',
                        }}
                        formatter={(value: any, name: any, props: any) => [
                          `${value}% (${numeral(props.payload.cost).format('$0,0')})`,
                          props.payload.name
                        ]}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                  <Box sx={{ mt: 2 }}>
                    {serviceBreakdown.slice(0, 4).map((service) => (
                      <Box key={service.name} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Box
                          sx={{
                            width: 12,
                            height: 12,
                            borderRadius: '50%',
                            backgroundColor: service.color,
                            mr: 1,
                          }}
                        />
                        <Typography variant="body2" sx={{ flexGrow: 1, fontSize: '0.8rem' }}>
                          {service.name}
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.8rem' }}>
                          {service.percentage}%
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>

        {/* Team Cost Analysis */}
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
                    Cost by Team
                  </Typography>
                  <TableContainer component={Paper} sx={{ backgroundColor: 'transparent' }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Team</TableCell>
                          <TableCell align="right">Cost</TableCell>
                          <TableCell align="right">Percentage</TableCell>
                          <TableCell align="right">Trend</TableCell>
                          <TableCell align="right">Change</TableCell>
                          <TableCell align="right">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {teamCostData.map((team) => (
                          <TableRow key={team.team}>
                            <TableCell component="th" scope="row">
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {team.team}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {numeral(team.cost).format('$0,0')}
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Chip
                                label={`${team.percentage}%`}
                                size="small"
                                sx={{
                                  backgroundColor: 'rgba(33, 150, 243, 0.2)',
                                  color: '#2196f3',
                                }}
                              />
                            </TableCell>
                            <TableCell align="right">
                              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                {getTrendIcon(team.trend)}
                              </Box>
                            </TableCell>
                            <TableCell align="right">
                              <Typography
                                variant="body2"
                                sx={{
                                  color: getTrendColor(team.trend),
                                  fontWeight: 600,
                                }}
                              >
                                {team.change > 0 ? '+' : ''}{team.change}%
                              </Typography>
                            </TableCell>
                            <TableCell align="right">
                              <Tooltip title="View Details">
                                <IconButton size="small">
                                  <TrendingUp sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
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
      </Box>
    </LocalizationProvider>
  );
};

export default CostAnalysis;