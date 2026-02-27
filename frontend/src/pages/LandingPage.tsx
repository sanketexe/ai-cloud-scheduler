import React from 'react';
import { Navigate } from 'react-router-dom';

const LandingPage: React.FC = () => {
  return <Navigate to="/onboarding" replace />;
};

export default LandingPage;
