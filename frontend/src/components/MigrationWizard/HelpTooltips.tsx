import React from 'react';
import { Tooltip, IconButton, Box, Typography } from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

interface HelpTooltipProps {
  title: string;
  content: string;
  placement?: 'top' | 'bottom' | 'left' | 'right';
}

export const HelpTooltip: React.FC<HelpTooltipProps> = ({ 
  title, 
  content, 
  placement = 'right' 
}) => {
  return (
    <Tooltip
      title={
        <Box sx={{ p: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 0.5 }}>
            {title}
          </Typography>
          <Typography variant="body2">{content}</Typography>
        </Box>
      }
      placement={placement}
      arrow
    >
      <IconButton size="small" sx={{ ml: 0.5 }}>
        <HelpOutlineIcon fontSize="small" />
      </IconButton>
    </Tooltip>
  );
};

// Predefined help content for common terms
export const HELP_CONTENT = {
  // Provider-related
  WEIGHTED_SCORING: {
    title: 'Weighted Scoring',
    content: 'Our algorithm evaluates 12 dimensions with different importance levels. Compliance (×3.0) and Workload Fit (×2.5) have the highest impact on your final score.'
  },
  
  HARD_ELIMINATORS: {
    title: 'Hard Eliminators',
    content: 'Certain requirements automatically eliminate providers. For example, FedRAMP certification is only available on AWS and Azure.'
  },
  
  SCORE_PREVIEW: {
    title: 'Real-Time Scoring',
    content: 'Scores update as you answer questions. The more information you provide, the more accurate your recommendations become.'
  },
  
  // Complexity-related
  MIGRATION_COMPLEXITY: {
    title: 'Migration Complexity',
    content: 'Calculated based on data volume, compliance requirements, team experience, and hybrid needs. Ranges from LOW (1-2 weeks) to HIGH (2-6 months).'
  },
  
  DATA_VOLUME: {
    title: 'Data Volume',
    content: 'Total amount of data to migrate. < 1 TB is LOW complexity, 1-100 TB is MEDIUM, > 100 TB is HIGH complexity.'
  },
  
  // Workload-related
  WORKLOAD_TYPES: {
    title: 'Workload Types',
    content: 'Select all that apply. Different providers excel at different workloads (e.g., GCP for AI/ML, Oracle for databases).'
  },
  
  WEB_APPLICATIONS: {
    title: 'Web Applications',
    content: 'Customer-facing websites, web portals, APIs, and web services. All major providers support this well.'
  },
  
  DATABASES: {
    title: 'Databases',
    content: 'Relational (PostgreSQL, MySQL) or NoSQL (MongoDB, Cassandra) databases. Oracle Cloud excels if you use Oracle Database.'
  },
  
  ANALYTICS: {
    title: 'Analytics & Big Data',
    content: 'Data warehousing, ETL pipelines, business intelligence. GCP\'s BigQuery is industry-leading for this workload.'
  },
  
  AI_ML: {
    title: 'AI/ML Workloads',
    content: 'Machine learning model training and inference. GCP (TensorFlow, Vertex AI) and Azure (Azure ML) are strong choices.'
  },
  
  ENTERPRISE_APPS: {
    title: 'Enterprise Applications',
    content: 'ERP, CRM, HR systems. Azure excels with Microsoft Dynamics, Oracle with Oracle ERP, IBM with WebSphere.'
  },
  
  // Compliance-related
  COMPLIANCE_FRAMEWORKS: {
    title: 'Compliance Requirements',
    content: 'Regulatory frameworks your organization must comply with. This heavily influences provider selection.'
  },
  
  FEDRAMP: {
    title: 'FedRAMP',
    content: 'Federal Risk and Authorization Management Program. Required for US government work. Only AWS and Azure are FedRAMP authorized.'
  },
  
  HIPAA: {
    title: 'HIPAA',
    content: 'Health Insurance Portability and Accountability Act. Required for healthcare data. All major providers offer HIPAA compliance with BAA.'
  },
  
  SOC2: {
    title: 'SOC 2',
    content: 'Service Organization Control 2. Audits security, availability, processing integrity, confidentiality, and privacy controls.'
  },
  
  PCI_DSS: {
    title: 'PCI DSS',
    content: 'Payment Card Industry Data Security Standard. Required if you process, store, or transmit credit card data.'
  },
  
  GDPR: {
    title: 'GDPR',
    content: 'General Data Protection Regulation. EU regulation for data protection and privacy. Requires data residency in EU.'
  },
  
  // Tech stack-related
  TECH_STACK: {
    title: 'Technology Stack',
    content: 'Your current programming languages, frameworks, and tools. Some providers integrate better with specific stacks.'
  },
  
  MICROSOFT_STACK: {
    title: 'Microsoft Stack',
    content: '.NET, C#, SQL Server, Windows Server, Active Directory. Azure provides the best integration for Microsoft technologies.'
  },
  
  ORACLE_STACK: {
    title: 'Oracle Stack',
    content: 'Oracle Database, Oracle ERP, PeopleSoft, JD Edwards. Oracle Cloud offers best performance and BYOL savings.'
  },
  
  IBM_STACK: {
    title: 'IBM Stack',
    content: 'WebSphere, Db2, IBM MQ, CICS. IBM Cloud provides deep integration with IBM software and middleware.'
  },
  
  // Budget-related
  BUDGET_PRIORITY: {
    title: 'Cost Optimization Priority',
    content: 'How important is cost savings? HIGH priority favors GCP and Oracle (competitive pricing), LOW priority focuses on features.'
  },
  
  MONTHLY_BUDGET: {
    title: 'Monthly Cloud Budget',
    content: 'Expected monthly spending on cloud services. Used to eliminate providers that don\'t fit your budget constraints.'
  },
  
  // Provider-specific
  AWS_STRENGTHS: {
    title: 'AWS Strengths',
    content: 'Largest service catalog (200+ services), global reach (30+ regions), mature migration tools, strong enterprise support.'
  },
  
  AZURE_STRENGTHS: {
    title: 'Azure Strengths',
    content: 'Microsoft ecosystem integration, hybrid cloud (Azure Arc), 90+ compliance certifications, enterprise agreements.'
  },
  
  GCP_STRENGTHS: {
    title: 'GCP Strengths',
    content: 'AI/ML leadership (TensorFlow, Vertex AI), BigQuery analytics, competitive pricing, Kubernetes expertise.'
  },
  
  IBM_STRENGTHS: {
    title: 'IBM Cloud Strengths',
    content: 'IBM software integration, Red Hat OpenShift, Watson AI, financial services focus, bare metal servers.'
  },
  
  ORACLE_STRENGTHS: {
    title: 'Oracle Cloud Strengths',
    content: 'Oracle Database performance, Autonomous Database, BYOL program, Oracle ERP integration, database-heavy workloads.'
  },
  
  // Capability scores
  CAPABILITY_COMPUTE: {
    title: 'Compute Capability',
    content: 'Virtual machines, containers, serverless functions. Rated on variety, performance, and pricing.'
  },
  
  CAPABILITY_STORAGE: {
    title: 'Storage Capability',
    content: 'Object storage, block storage, file storage. Rated on durability, performance, and cost.'
  },
  
  CAPABILITY_NETWORKING: {
    title: 'Networking Capability',
    content: 'VPCs, load balancers, CDN, DNS. Rated on global reach, performance, and features.'
  },
  
  CAPABILITY_DATABASES: {
    title: 'Database Capability',
    content: 'Managed relational and NoSQL databases. Rated on variety, performance, and management features.'
  },
  
  CAPABILITY_AI_ML: {
    title: 'AI/ML Capability',
    content: 'Machine learning platforms, pre-trained models, AI services. Rated on ease of use and advanced features.'
  },
  
  CAPABILITY_SECURITY: {
    title: 'Security Capability',
    content: 'Identity management, encryption, compliance tools. Rated on comprehensiveness and ease of use.'
  }
};

// Helper function to get help content
export const getHelpContent = (key: keyof typeof HELP_CONTENT) => {
  return HELP_CONTENT[key];
};

export default HelpTooltip;
