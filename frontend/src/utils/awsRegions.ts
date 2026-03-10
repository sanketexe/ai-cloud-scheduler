/**
 * AWS Regions Configuration
 * Complete list of AWS regions with display names
 */

export interface AWSRegion {
  code: string;
  name: string;
  location: string;
}

export const AWS_REGIONS: AWSRegion[] = [
  // US Regions
  { code: 'us-east-1', name: 'US East (N. Virginia)', location: 'United States' },
  { code: 'us-east-2', name: 'US East (Ohio)', location: 'United States' },
  { code: 'us-west-1', name: 'US West (N. California)', location: 'United States' },
  { code: 'us-west-2', name: 'US West (Oregon)', location: 'United States' },

  // Canada
  { code: 'ca-central-1', name: 'Canada (Central)', location: 'Canada' },

  // Europe
  { code: 'eu-north-1', name: 'Europe (Stockholm)', location: 'Europe' },
  { code: 'eu-west-1', name: 'Europe (Ireland)', location: 'Europe' },
  { code: 'eu-west-2', name: 'Europe (London)', location: 'Europe' },
  { code: 'eu-west-3', name: 'Europe (Paris)', location: 'Europe' },
  { code: 'eu-central-1', name: 'Europe (Frankfurt)', location: 'Europe' },
  { code: 'eu-central-2', name: 'Europe (Zurich)', location: 'Europe' },
  { code: 'eu-south-1', name: 'Europe (Milan)', location: 'Europe' },
  { code: 'eu-south-2', name: 'Europe (Spain)', location: 'Europe' },

  // Asia Pacific
  { code: 'ap-northeast-1', name: 'Asia Pacific (Tokyo)', location: 'Asia Pacific' },
  { code: 'ap-northeast-2', name: 'Asia Pacific (Seoul)', location: 'Asia Pacific' },
  { code: 'ap-northeast-3', name: 'Asia Pacific (Osaka)', location: 'Asia Pacific' },
  { code: 'ap-southeast-1', name: 'Asia Pacific (Singapore)', location: 'Asia Pacific' },
  { code: 'ap-southeast-2', name: 'Asia Pacific (Sydney)', location: 'Asia Pacific' },
  { code: 'ap-southeast-3', name: 'Asia Pacific (Jakarta)', location: 'Asia Pacific' },
  { code: 'ap-southeast-4', name: 'Asia Pacific (Melbourne)', location: 'Asia Pacific' },
  { code: 'ap-south-1', name: 'Asia Pacific (Mumbai)', location: 'Asia Pacific' },
  { code: 'ap-south-2', name: 'Asia Pacific (Hyderabad)', location: 'Asia Pacific' },
  { code: 'ap-east-1', name: 'Asia Pacific (Hong Kong)', location: 'Asia Pacific' },

  // China
  { code: 'cn-north-1', name: 'China (Beijing)', location: 'China' },
  { code: 'cn-northwest-1', name: 'China (Ningxia)', location: 'China' },

  // Middle East
  { code: 'me-south-1', name: 'Middle East (Bahrain)', location: 'Middle East' },
  { code: 'me-central-1', name: 'Middle East (UAE)', location: 'Middle East' },

  // Africa
  { code: 'af-south-1', name: 'Africa (Cape Town)', location: 'Africa' },

  // South America
  { code: 'sa-east-1', name: 'South America (São Paulo)', location: 'South America' },
];

export const getRegionByCode = (code: string): AWSRegion | undefined => {
  return AWS_REGIONS.find(region => region.code === code);
};

export const getRegionsByLocation = (location: string): AWSRegion[] => {
  return AWS_REGIONS.filter(region => region.location === location);
};

export const getPopularRegions = (): AWSRegion[] => {
  return AWS_REGIONS.filter(region =>
    ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1'].includes(region.code)
  );
};