import React from 'react';
import { Box, LinearProgress, Typography } from '@mui/material';

interface ProgressIndicatorProps {
  message?: string;
  progress?: number;
  showPercentage?: boolean;
}

/**
 * Progress Indicator Component
 * 
 * Displays a progress bar for slow operations
 * Validates: Requirements 5.3, 6.2
 */
const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ 
  message = 'Processing...', 
  progress,
  showPercentage = false
}) => {
  return (
    <Box sx={{ width: '100%', py: 2 }}>
      {message && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            {message}
          </Typography>
          {showPercentage && progress !== undefined && (
            <Typography variant="body2" color="text.secondary">
              {Math.round(progress)}%
            </Typography>
          )}
        </Box>
      )}
      <LinearProgress 
        variant={progress !== undefined ? 'determinate' : 'indeterminate'}
        value={progress}
        sx={{
          height: 6,
          borderRadius: 3,
          backgroundColor: 'rgba(255,255,255,0.1)',
          '& .MuiLinearProgress-bar': {
            borderRadius: 3,
          },
        }}
      />
    </Box>
  );
};

export default ProgressIndicator;
