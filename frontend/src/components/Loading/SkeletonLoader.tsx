import React from 'react';
import { Box, Skeleton, Card, CardContent, Grid } from '@mui/material';

interface SkeletonLoaderProps {
  variant?: 'dashboard' | 'table' | 'chart' | 'card' | 'text';
  count?: number;
  height?: number | string;
}

/**
 * Skeleton Loader Component
 * 
 * Provides skeleton loading states for different UI patterns
 * Validates: Requirements 5.3, 6.2
 */
const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({ 
  variant = 'card', 
  count = 1,
  height = 200 
}) => {
  const renderDashboardSkeleton = () => (
    <Box>
      {/* Header Skeleton */}
      <Skeleton variant="text" width="40%" height={50} sx={{ mb: 4 }} />
      
      {/* Stats Cards Skeleton */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {[1, 2, 3, 4].map((i) => (
          <Grid item xs={12} sm={6} md={3} key={i}>
            <Card>
              <CardContent>
                <Skeleton variant="text" width="60%" height={24} />
                <Skeleton variant="text" width="80%" height={40} sx={{ my: 1 }} />
                <Skeleton variant="text" width="40%" height={20} />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Charts Skeleton */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
              <Skeleton variant="rectangular" width="100%" height={350} sx={{ borderRadius: 2 }} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Skeleton variant="text" width="40%" height={32} sx={{ mb: 2 }} />
              <Skeleton variant="circular" width={200} height={200} sx={{ mx: 'auto', my: 2 }} />
              {[1, 2, 3].map((i) => (
                <Box key={i} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Skeleton variant="circular" width={12} height={12} sx={{ mr: 1 }} />
                  <Skeleton variant="text" width="60%" />
                  <Skeleton variant="text" width="20%" sx={{ ml: 'auto' }} />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderTableSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
        {Array.from({ length: count }).map((_, i) => (
          <Box key={i} sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <Skeleton variant="rectangular" width="20%" height={40} />
            <Skeleton variant="rectangular" width="30%" height={40} />
            <Skeleton variant="rectangular" width="25%" height={40} />
            <Skeleton variant="rectangular" width="25%" height={40} />
          </Box>
        ))}
      </CardContent>
    </Card>
  );

  const renderChartSkeleton = () => (
    <Card>
      <CardContent>
        <Skeleton variant="text" width="30%" height={32} sx={{ mb: 2 }} />
        <Skeleton 
          variant="rectangular" 
          width="100%" 
          height={typeof height === 'number' ? height : 300} 
          sx={{ borderRadius: 2 }} 
        />
      </CardContent>
    </Card>
  );

  const renderCardSkeleton = () => (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i} sx={{ mb: 2 }}>
          <CardContent>
            <Skeleton variant="text" width="60%" height={24} />
            <Skeleton variant="text" width="80%" height={40} sx={{ my: 1 }} />
            <Skeleton variant="text" width="40%" height={20} />
          </CardContent>
        </Card>
      ))}
    </>
  );

  const renderTextSkeleton = () => (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton 
          key={i} 
          variant="text" 
          width={`${Math.random() * 30 + 70}%`} 
          height={24} 
          sx={{ mb: 1 }} 
        />
      ))}
    </>
  );

  switch (variant) {
    case 'dashboard':
      return renderDashboardSkeleton();
    case 'table':
      return renderTableSkeleton();
    case 'chart':
      return renderChartSkeleton();
    case 'card':
      return renderCardSkeleton();
    case 'text':
      return renderTextSkeleton();
    default:
      return renderCardSkeleton();
  }
};

export default SkeletonLoader;
