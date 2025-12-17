import React, { useState } from 'react';
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Card,
} from '@mui/material';
import PolicyConfiguration from '../components/PolicyConfiguration';
import ActionApproval from '../components/ActionApproval';
import SavingsReports from '../components/SavingsReports';

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
      id={`automation-settings-tabpanel-${index}`}
      aria-labelledby={`automation-settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

const AutomationSettings: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
          Automation Management
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure policies, manage approvals, and analyze savings from automated cost optimization
        </Typography>
      </Box>

      {/* Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Policy Configuration" />
            <Tab label="Action Approvals" />
            <Tab label="Savings Reports" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <PolicyConfiguration />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <ActionApproval />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <SavingsReports />
        </TabPanel>
      </Card>
    </Box>
  );
};

export default AutomationSettings;