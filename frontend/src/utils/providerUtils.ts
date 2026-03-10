/**
 * Provider Utilities
 * Centralized utilities for cloud provider display and information
 */

export interface ProviderDetails {
  name: string;
  icon: string;
  color: string;
  keyStrength: string;
  strengths: string[];
  bestFor: string[];
  watchFor: string[];
}

export const PROVIDER_ICONS: Record<string, string> = {
  'AWS': '☁️',
  'Azure': '🔷',
  'GCP': '🌐',
  'IBM': '🔷',
  'Oracle': '🔴'
};

export const PROVIDER_COLORS: Record<string, string> = {
  'AWS': '#FF9900',
  'Azure': '#0078D4',
  'GCP': '#4285F4',
  'IBM': '#0F62FE',
  'Oracle': '#F80000'
};

export const PROVIDER_DETAILS: Record<string, ProviderDetails> = {
  'AWS': {
    name: 'Amazon Web Services',
    icon: '☁️',
    color: '#FF9900',
    keyStrength: 'Largest service catalog and global reach',
    strengths: [
      'Most mature cloud platform with 200+ services',
      'Extensive global infrastructure (30+ regions)',
      'Strong serverless and container support',
      'Best migration tooling (AWS Migration Hub, DMS)',
      'Largest partner ecosystem'
    ],
    bestFor: [
      'Startups needing rapid scaling',
      'E-commerce and retail applications',
      'Microservices architectures',
      'Organizations requiring service variety'
    ],
    watchFor: [
      'Complex pricing structure',
      'Egress fees can be significant',
      'Steeper learning curve'
    ]
  },
  'Azure': {
    name: 'Microsoft Azure',
    icon: '🔷',
    color: '#0078D4',
    keyStrength: 'Best for Microsoft ecosystem and hybrid cloud',
    strengths: [
      'Seamless Microsoft product integration',
      'Leading hybrid cloud capabilities (Azure Arc)',
      'Strong enterprise and government focus',
      'Excellent compliance certifications',
      'Best for .NET and Windows workloads'
    ],
    bestFor: [
      'Organizations using Microsoft stack',
      'Enterprise and government sectors',
      'Hybrid cloud deployments',
      'Finance and healthcare industries'
    ],
    watchFor: [
      'Portal complexity',
      'Premium support can be expensive',
      'Some services lag behind AWS'
    ]
  },
  'GCP': {
    name: 'Google Cloud Platform',
    icon: '🌐',
    color: '#4285F4',
    keyStrength: 'Best for AI/ML and cost optimization',
    strengths: [
      'Industry-leading AI/ML services (Vertex AI)',
      'Superior data analytics (BigQuery)',
      'Competitive pricing with sustained use discounts',
      'Excellent Kubernetes support (GKE)',
      'Strong open source focus'
    ],
    bestFor: [
      'Data analytics and ML projects',
      'Cost-conscious organizations',
      'Container-based applications',
      'Small to medium businesses'
    ],
    watchFor: [
      'Smaller partner network',
      'Fewer regions than AWS/Azure',
      'Less enterprise tooling'
    ]
  },
  'IBM': {
    name: 'IBM Cloud',
    icon: '🔷',
    color: '#0F62FE',
    keyStrength: 'Best for IBM software and hybrid enterprise',
    strengths: [
      'Excellent IBM software integration (Db2, WebSphere, MQ)',
      'Strong regulated-industry compliance',
      'Advanced hybrid cloud with LinuxONE',
      'Watson AI services',
      'Bare metal server options'
    ],
    bestFor: [
      'Organizations with IBM investments',
      'Financial services and banking',
      'Hybrid cloud deployments',
      'AIX and IBM i workloads'
    ],
    watchFor: [
      'Narrower service catalog',
      'Smaller developer community',
      'Limited third-party integrations'
    ]
  },
  'Oracle': {
    name: 'Oracle Cloud Infrastructure',
    icon: '🔴',
    color: '#F80000',
    keyStrength: 'Best for Oracle Database and ERP',
    strengths: [
      'Optimized for Oracle Database workloads',
      'BYOL program saves significant costs',
      'Autonomous Database capabilities',
      'Strong security with always-on encryption',
      'Excellent for Oracle ERP and Fusion apps'
    ],
    bestFor: [
      'Organizations running Oracle Database',
      'Oracle ERP, PeopleSoft, JD Edwards users',
      'Workloads with Oracle licenses',
      'Cost-conscious Oracle customers'
    ],
    watchFor: [
      'Limited appeal outside Oracle ecosystem',
      'Smaller ISV partner network',
      'Fewer managed services vs competitors'
    ]
  }
};

export const getProviderIcon = (provider: string): string => {
  return PROVIDER_ICONS[provider] || '☁️';
};

export const getProviderColor = (provider: string): string => {
  return PROVIDER_COLORS[provider] || '#666666';
};

export const getProviderDetails = (provider: string): ProviderDetails | null => {
  return PROVIDER_DETAILS[provider] || null;
};

export const getAllProviders = (): string[] => {
  return Object.keys(PROVIDER_DETAILS);
};

export const getProviderName = (provider: string): string => {
  return PROVIDER_DETAILS[provider]?.name || provider;
};
