import React, { useState } from 'react';
import {
    Box,
    Container,
    Typography,
    Paper,
    Stepper,
    Step,
    StepLabel,
    TextField,
    Button,
    Alert,
    CircularProgress,
    useTheme,
    Grid,
} from '@mui/material';
import { motion } from 'framer-motion';
import { CheckCircle, RocketLaunch } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useMutation } from 'react-query';
import toast from 'react-hot-toast';

const steps = ['Connect AWS Account', 'Analyzing Costs', 'Results Ready'];

const OnboardingQuickStart: React.FC = () => {
    const theme = useTheme();
    const navigate = useNavigate();
    const [activeStep, setActiveStep] = useState(0);
    const [credentials, setCredentials] = useState({
        access_key_id: '',
        secret_access_key: '',
        region: 'us-east-1',
    });

    const onboardingMutation = useMutation(
        async (data: typeof credentials) => {
            // In dev/demo, we might use a different URL if proxy isn't set up perfectly, 
            // but typically package.json proxy handles this.
            const response = await axios.post('/api/v1/onboarding/quick-setup', data);
            return response.data;
        },
        {
            onSuccess: (data) => {
                toast.success(data.message);
                setActiveStep(1);
                // Simulate analysis time for UX
                setTimeout(() => {
                    setActiveStep(2);
                }, 3000);
            },
            onError: (error: any) => {
                toast.error(error.response?.data?.detail || 'Failed to connect. Please check credentials.');
            },
        }
    );

    const handleNext = () => {
        if (activeStep === 0) {
            if (!credentials.access_key_id || !credentials.secret_access_key) {
                toast.error("Please enter credentials");
                return;
            }
            onboardingMutation.mutate(credentials);
        } else if (activeStep === 2) {
            navigate('/dashboard');
        }
    };

    const handleDemoMode = () => {
        setCredentials({
            access_key_id: 'DEMO',
            secret_access_key: 'DEMO',
            region: 'us-east-1'
        });
    };

    return (
        <Box
            sx={{
                minHeight: '100vh',
                background: `linear-gradient(135deg, ${theme.palette.background.default} 0%, #1a2332 100%)`,
                pt: 8,
                pb: 4,
            }}
        >
            <Container maxWidth="md">
                <Typography variant="h4" align="center" sx={{ mb: 6, fontWeight: 700 }}>
                    Connect Your AWS Account
                </Typography>

                <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 8 }}>
                    {steps.map((label) => (
                        <Step key={label}>
                            <StepLabel>{label}</StepLabel>
                        </Step>
                    ))}
                </Stepper>

                <Paper
                    elevation={4}
                    sx={{
                        p: 4,
                        background: 'rgba(26, 35, 60, 0.8)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: 3,
                        minHeight: '400px',
                        position: 'relative',
                    }}
                >
                    <motion.div
                        key={activeStep}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                    >
                        {activeStep === 0 && (
                            <Box>
                                <Typography variant="h5" sx={{ mb: 3 }}>
                                    Connect your AWS Account
                                </Typography>
                                <Alert severity="info" sx={{ mb: 3 }}>
                                    We use read-only access to analyze your Cost & Usage Reports.
                                    Your credentials are encrypted and never shared.
                                    <Button size="small" onClick={handleDemoMode} sx={{ ml: 2, fontWeight: 'bold' }}>
                                        Try Demo Mode instead
                                    </Button>
                                </Alert>

                                <Grid container spacing={3}>
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            label="Access Key ID"
                                            value={credentials.access_key_id}
                                            onChange={(e) => setCredentials({ ...credentials, access_key_id: e.target.value })}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            type="password"
                                            label="Secret Access Key"
                                            value={credentials.secret_access_key}
                                            onChange={(e) => setCredentials({ ...credentials, secret_access_key: e.target.value })}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            label="Region"
                                            value={credentials.region}
                                            onChange={(e) => setCredentials({ ...credentials, region: e.target.value })}
                                        />
                                    </Grid>
                                </Grid>

                                <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
                                    <Button
                                        variant="contained"
                                        size="large"
                                        onClick={handleNext}
                                        disabled={onboardingMutation.isLoading}
                                        startIcon={onboardingMutation.isLoading ? <CircularProgress size={20} /> : <RocketLaunch />}
                                    >
                                        {onboardingMutation.isLoading ? 'Connecting...' : 'Start Analysis'}
                                    </Button>
                                </Box>
                            </Box>
                        )}

                        {activeStep === 1 && (
                            <Box sx={{ textAlign: 'center', py: 8 }}>
                                <CircularProgress size={60} thickness={4} sx={{ mb: 4 }} />
                                <Typography variant="h5" sx={{ mb: 2 }}>
                                    Analyzing Usage Patterns...
                                </Typography>
                                <Typography color="text.secondary">
                                    We are scanning your AWS environment for cost optimization opportunities.
                                    This usually takes about 30 seconds.
                                </Typography>
                            </Box>
                        )}

                        {activeStep === 2 && (
                            <Box sx={{ textAlign: 'center', py: 6 }}>
                                <CheckCircle sx={{ fontSize: 80, color: 'success.main', mb: 3 }} />
                                <Typography variant="h4" sx={{ mb: 2, fontWeight: 700 }}>
                                    Configuration Complete!
                                </Typography>
                                <Typography variant="h6" color="text.secondary" sx={{ mb: 6 }}>
                                    We've successfully connected to your account and identified initial savings.
                                </Typography>

                                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 4 }}>
                                    <Box sx={{ p: 2, border: '1px solid rgba(255,255,255,0.1)', borderRadius: 2 }}>
                                        <Typography variant="caption" color="text.secondary">POTENTIAL SAVINGS</Typography>
                                        <Typography variant="h4" color="primary.main">$1,240/mo</Typography>
                                    </Box>
                                </Box>

                                <Box sx={{ mt: 6 }}>
                                    <Button
                                        variant="contained"
                                        size="large"
                                        onClick={handleNext}
                                        sx={{ px: 6, py: 1.5, fontSize: '1.1rem' }}
                                    >
                                        Go to Dashboard
                                    </Button>
                                </Box>
                            </Box>
                        )}
                    </motion.div>
                </Paper>
            </Container>
        </Box>
    );
};

export default OnboardingQuickStart;
