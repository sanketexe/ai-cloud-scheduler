import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { HelmetProvider } from 'react-helmet-async';

// Components
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import OnboardingQuickStart from './pages/OnboardingQuickStart';
import Dashboard from './pages/Dashboard';
import CostAnalysis from './pages/CostAnalysis';
import BudgetManagement from './pages/BudgetManagement';
import Optimization from './pages/Optimization';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import Alerts from './pages/Alerts';
import Compliance from './pages/Compliance';
import SchedulerDashboard from './pages/SchedulerDashboard';
import MigrationPlanner from './pages/MigrationPlanner';
import Home from './pages/Home';
import MigrationWizard from './pages/MigrationWizard';
import MigrationResults from './pages/MigrationResults';
import MigrationDashboard from './pages/MigrationDashboard';
import ProviderRecommendations from './pages/ProviderRecommendations';
import ResourceOrganization from './pages/ResourceOrganization';
import DimensionalFiltering from './pages/DimensionalFiltering';
import MigrationReport from './pages/MigrationReport';
import PlatformFloatingChat from './components/AI/PlatformFloatingChat';

// Theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#f50057',
      light: '#ff5983',
      dark: '#c51162',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1d3a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0bec5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, #1a1d3a 0%, #2a2d5a 100%)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
  },
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

// Reusable layout wrapper
const PageLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Box sx={{ display: 'flex', minHeight: '100vh' }}>
    <Sidebar />
    <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
      <Header />
      <Box component="main" sx={{ flexGrow: 1, p: 3, mt: 8 }}>
        {children}
      </Box>
    </Box>
  </Box>
);

function App() {
  return (
    <HelmetProvider>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Router>
            <Routes>
              {/* Main Entry Points */}
              <Route path="/" element={<Home />} />
              <Route path="/onboarding" element={<OnboardingQuickStart />} />

              {/* Migration Wizard - No Sidebar/Header */}
              <Route path="/migration-wizard" element={<MigrationWizard />} />
              <Route path="/migration-wizard/:projectId" element={<MigrationWizard />} />
              <Route path="/migration/:projectId/recommendations" element={<ProviderRecommendations />} />
              <Route path="/migration/:projectId/results" element={<MigrationResults />} />
              <Route path="/migration/:projectId/dashboard" element={<MigrationDashboard />} />
              <Route path="/migration/:projectId/resources" element={<ResourceOrganization />} />
              <Route path="/migration/:projectId/filtering" element={<DimensionalFiltering />} />
              <Route path="/migration/:projectId/report" element={<MigrationReport />} />

              {/* Dashboard Routes */}
              <Route path="/dashboard" element={<PageLayout><Dashboard /></PageLayout>} />
              <Route path="/scheduler" element={<PageLayout><SchedulerDashboard /></PageLayout>} />
              <Route path="/cost-analysis" element={<PageLayout><CostAnalysis /></PageLayout>} />
              <Route path="/budgets" element={<PageLayout><BudgetManagement /></PageLayout>} />
              <Route path="/optimization" element={<PageLayout><Optimization /></PageLayout>} />
              <Route path="/reports" element={<PageLayout><Reports /></PageLayout>} />
              <Route path="/alerts" element={<PageLayout><Alerts /></PageLayout>} />
              <Route path="/compliance" element={<PageLayout><Compliance /></PageLayout>} />
              <Route path="/settings" element={<PageLayout><Settings /></PageLayout>} />
              <Route path="/migration-planner" element={<PageLayout><MigrationPlanner /></PageLayout>} />
            </Routes>
            <PlatformFloatingChat />
          </Router>
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#1a1d3a',
                color: '#fff',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              },
            }}
          />
        </ThemeProvider>
      </QueryClientProvider>
    </HelmetProvider>
  );
}

export default App;