import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  IconButton,
  Badge,
  Avatar,
  Chip,
} from '@mui/material';
import {
  Notifications as NotificationsIcon,
  AccountCircle as AccountIcon,
} from '@mui/icons-material';

const Header: React.FC = () => {
  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        background: 'rgba(26, 29, 58, 0.95)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {/* Left side - empty for now since we have sidebar */}
        <Box />

        {/* Center - Status indicators */}
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Chip
            label="AWS Connected"
            size="small"
            sx={{
              background: 'rgba(255, 152, 0, 0.2)',
              color: '#ff9800',
              border: '1px solid rgba(255, 152, 0, 0.3)',
            }}
          />
          <Chip
            label="GCP Connected"
            size="small"
            sx={{
              background: 'rgba(76, 175, 80, 0.2)',
              color: '#4caf50',
              border: '1px solid rgba(76, 175, 80, 0.3)',
            }}
          />
          <Chip
            label="Azure Connected"
            size="small"
            sx={{
              background: 'rgba(33, 150, 243, 0.2)',
              color: '#2196f3',
              border: '1px solid rgba(33, 150, 243, 0.3)',
            }}
          />
        </Box>

        {/* Right side - User actions */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton color="inherit">
            <Badge badgeContent={3} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>
          
          <IconButton color="inherit">
            <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
              <AccountIcon />
            </Avatar>
          </IconButton>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;