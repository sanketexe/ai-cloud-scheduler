import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Box,
  Divider,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  TrendingUp as TrendingUpIcon,
  AccountBalance as BudgetIcon,
  Lightbulb as OptimizationIcon,
  Assessment as ReportsIcon,
  Notifications as AlertsIcon,
  Security as ComplianceIcon,
  Settings as SettingsIcon,
  AttachMoney as MoneyIcon,
  Cloud as CloudIcon,
  NotificationsActive as NotificationsActiveIcon,
  AutoMode as AutomationIcon,
  Psychology as AnomalyIcon,
  CompareArrows as MultiCloudIcon,
  FlightTakeoff as MigrationIcon,
  SmartToy as AIIcon,
} from '@mui/icons-material';

const drawerWidth = 280;

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
  { text: 'Cost Analysis', icon: <TrendingUpIcon />, path: '/cost-analysis' },
  { text: 'Multi-Cloud Comparison', icon: <MultiCloudIcon />, path: '/multi-cloud' },
  { text: 'Migration Planner', icon: <MigrationIcon />, path: '/migration-planner' },
  { text: 'AI Dashboard', icon: <AIIcon />, path: '/ai-dashboard' },
  { text: 'AWS Cost Analysis', icon: <CloudIcon />, path: '/aws-cost-analysis' },
  { text: 'AWS Cost Alerts', icon: <NotificationsActiveIcon />, path: '/aws-cost-alerts' },
  { text: 'Budget Management', icon: <BudgetIcon />, path: '/budgets' },
  { text: 'Optimization', icon: <OptimizationIcon />, path: '/optimization' },
  { text: 'Automation', icon: <AutomationIcon />, path: '/automation' },
  { text: 'AI Anomaly Detection', icon: <AnomalyIcon />, path: '/anomaly-detection' },
  { text: 'Reports', icon: <ReportsIcon />, path: '/reports' },
  { text: 'Alerts', icon: <AlertsIcon />, path: '/alerts' },
  { text: 'Compliance', icon: <ComplianceIcon />, path: '/compliance' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];

const Sidebar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          background: 'linear-gradient(180deg, #0a0e27 0%, #1a1d3a 100%)',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)',
        },
      }}
    >
      {/* Logo Section */}
      <Box
        sx={{
          p: 3,
          display: 'flex',
          alignItems: 'center',
          gap: 2,
        }}
      >
        <MoneyIcon sx={{ fontSize: 40, color: 'primary.main' }} />
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 700, color: 'white' }}>
            FinOps Platform
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Financial Operations
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

      {/* Navigation Menu */}
      <List sx={{ px: 2, py: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
            <ListItemButton
              onClick={() => navigate(item.path)}
              selected={location.pathname === item.path}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  background: 'linear-gradient(135deg, #2196f3 0%, #1976d2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
                  },
                },
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.05)',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  color: location.pathname === item.path ? 'white' : 'text.secondary',
                  minWidth: 40,
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                sx={{
                  '& .MuiListItemText-primary': {
                    fontWeight: location.pathname === item.path ? 600 : 400,
                    color: location.pathname === item.path ? 'white' : 'text.primary',
                  },
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      {/* Status Section */}
      <Box sx={{ mt: 'auto', p: 2 }}>
        <Box
          sx={{
            p: 2,
            borderRadius: 2,
            background: 'rgba(76, 175, 80, 0.1)',
            border: '1px solid rgba(76, 175, 80, 0.3)',
          }}
        >
          <Typography variant="caption" sx={{ color: '#4caf50', fontWeight: 600 }}>
            System Status
          </Typography>
          <Typography variant="body2" sx={{ color: 'white', mt: 0.5 }}>
            All systems operational
          </Typography>
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;