import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Grid,
  Paper,
  useTheme,
  alpha,
} from '@mui/material';
import {
  CloudUpload as MigrationIcon,
  TrendingUp,
  CompareArrows,
  Assessment,
  Security,
  Speed,
} from '@mui/icons-material';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const theme = useTheme();

  const handleMigrationClick = () => {
    navigate('/migration-wizard');
  };

  const handleFinOpsClick = () => {
    navigate('/dashboard');
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, #1a2332 100%)`,
        display: 'flex',
        alignItems: 'center',
        py: 4,
      }}
    >
      <Container maxWidth="lg">
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Typography
            variant="h2"
            sx={{
              fontWeight: 700,
              mb: 2,
              background: 'linear-gradient(45deg, #2196f3 30%, #21cbf3 90%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Cloud Intelligence Platform
          </Typography>
          <Typography variant="h5" color="text.secondary" sx={{ mb: 1 }}>
            Intelligent Multi-Cloud Management & Cost Optimization
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Choose your journey to cloud excellence
          </Typography>
        </Box>

        {/* Main Options */}
        <Grid container spacing={4} sx={{ mb: 6 }}>
          {/* Migration Cost Comparison Card */}
          <Grid item xs={12} md={6}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: `0 12px 40px ${alpha(theme.palette.primary.main, 0.3)}`,
                  borderColor: theme.palette.primary.main,
                },
              }}
              onClick={handleMigrationClick}
            >
              <CardContent sx={{ flexGrow: 1, p: 4 }}>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mb: 3,
                  }}
                >
                  <Box
                    sx={{
                      p: 2,
                      borderRadius: '50%',
                      background: `linear-gradient(135deg, ${alpha(
                        theme.palette.primary.main,
                        0.2
                      )} 0%, ${alpha(theme.palette.primary.dark, 0.3)} 100%)`,
                    }}
                  >
                    <MigrationIcon sx={{ fontSize: 60, color: 'primary.main' }} />
                  </Box>
                </Box>

                <Typography
                  variant="h4"
                  align="center"
                  gutterBottom
                  sx={{ fontWeight: 600, mb: 2 }}
                >
                  Migration Cost Comparison
                </Typography>

                <Typography
                  variant="body1"
                  color="text.secondary"
                  align="center"
                  sx={{ mb: 3 }}
                >
                  Planning to migrate to the cloud? Compare costs across AWS, GCP, and
                  Azure to find the best fit for your startup.
                </Typography>

                {/* Features */}
                <Box sx={{ mt: 3 }}>
                  {[
                    { icon: <CompareArrows />, text: 'Multi-cloud cost comparison' },
                    { icon: <Assessment />, text: 'Database assessment & sizing' },
                    { icon: <Speed />, text: 'Quick migration planning' },
                  ].map((feature, index) => (
                    <Box
                      key={index}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        mb: 2,
                        p: 1.5,
                        borderRadius: 1,
                        background: alpha(theme.palette.primary.main, 0.05),
                      }}
                    >
                      <Box sx={{ color: 'primary.main', mr: 2 }}>{feature.icon}</Box>
                      <Typography variant="body2">{feature.text}</Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>

              <CardActions sx={{ p: 3, pt: 0 }}>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  onClick={handleMigrationClick}
                  sx={{
                    py: 1.5,
                    fontSize: '1.1rem',
                    fontWeight: 600,
                    background: 'linear-gradient(45deg, #2196f3 30%, #21cbf3 90%)',
                    '&:hover': {
                      background: 'linear-gradient(45deg, #1976d2 30%, #00bcd4 90%)',
                    },
                  }}
                >
                  Start Migration Analysis
                </Button>
              </CardActions>
            </Card>
          </Grid>

          {/* FinOps Management Card */}
          <Grid item xs={12} md={6}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: `0 12px 40px ${alpha(theme.palette.secondary.main, 0.3)}`,
                  borderColor: theme.palette.secondary.main,
                },
              }}
              onClick={handleFinOpsClick}
            >
              <CardContent sx={{ flexGrow: 1, p: 4 }}>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mb: 3,
                  }}
                >
                  <Box
                    sx={{
                      p: 2,
                      borderRadius: '50%',
                      background: `linear-gradient(135deg, ${alpha(
                        theme.palette.secondary.main,
                        0.2
                      )} 0%, ${alpha(theme.palette.secondary.dark, 0.3)} 100%)`,
                    }}
                  >
                    <TrendingUp sx={{ fontSize: 60, color: 'secondary.main' }} />
                  </Box>
                </Box>

                <Typography
                  variant="h4"
                  align="center"
                  gutterBottom
                  sx={{ fontWeight: 600, mb: 2 }}
                >
                  Cost Optimization & Management
                </Typography>

                <Typography
                  variant="body1"
                  color="text.secondary"
                  align="center"
                  sx={{ mb: 3 }}
                >
                  Already in the cloud? Optimize your spending with real-time cost
                  tracking, budgets, and intelligent recommendations.
                </Typography>

                {/* Features */}
                <Box sx={{ mt: 3 }}>
                  {[
                    { icon: <Assessment />, text: 'Real-time cost analytics' },
                    { icon: <Security />, text: 'Budget alerts & governance' },
                    { icon: <TrendingUp />, text: 'AI-powered optimization' },
                  ].map((feature, index) => (
                    <Box
                      key={index}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        mb: 2,
                        p: 1.5,
                        borderRadius: 1,
                        background: alpha(theme.palette.secondary.main, 0.05),
                      }}
                    >
                      <Box sx={{ color: 'secondary.main', mr: 2 }}>{feature.icon}</Box>
                      <Typography variant="body2">{feature.text}</Typography>
                    </Box>
                  ))}
                </Box>
              </CardContent>

              <CardActions sx={{ p: 3, pt: 0 }}>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  onClick={handleFinOpsClick}
                  sx={{
                    py: 1.5,
                    fontSize: '1.1rem',
                    fontWeight: 600,
                    background: 'linear-gradient(45deg, #f50057 30%, #ff5983 90%)',
                    '&:hover': {
                      background: 'linear-gradient(45deg, #c51162 30%, #f50057 90%)',
                    },
                  }}
                >
                  Go to FinOps Dashboard
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>

        {/* Footer Info */}
        <Paper
          sx={{
            p: 3,
            textAlign: 'center',
            background: alpha(theme.palette.background.paper, 0.5),
            backdropFilter: 'blur(10px)',
          }}
        >
          <Typography variant="body2" color="text.secondary">
            Not sure which option to choose?{' '}
            <Typography
              component="span"
              variant="body2"
              sx={{ color: 'primary.main', fontWeight: 600 }}
            >
              Migration Cost Comparison
            </Typography>{' '}
            is perfect for startups planning their cloud journey, while{' '}
            <Typography
              component="span"
              variant="body2"
              sx={{ color: 'secondary.main', fontWeight: 600 }}
            >
              Cost Optimization
            </Typography>{' '}
            is ideal for businesses already running in the cloud.
          </Typography>
        </Paper>
      </Container>
    </Box>
  );
};

export default LandingPage;
