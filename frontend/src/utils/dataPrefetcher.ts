/**
 * Data Prefetcher
 * 
 * Pre-loads data for upcoming pages to improve navigation performance
 * Validates: Requirements 5.2
 */

import apiService from '../services/api';

interface PrefetchConfig {
  page: string;
  dataLoaders: Array<{
    key: string;
    loader: () => Promise<any>;
    priority?: number;
  }>;
}

// Cache for prefetched data
const prefetchCache = new Map<string, {
  data: any;
  timestamp: number;
  ttl: number;
}>();

// Default TTL: 2 minutes (data stays fresh for quick navigation)
const DEFAULT_PREFETCH_TTL = 2 * 60 * 1000;

/**
 * Prefetch configurations for demo pages
 */
const PREFETCH_CONFIGS: PrefetchConfig[] = [
  {
    page: 'dashboard',
    dataLoaders: [
      {
        key: 'dashboard-data',
        loader: async () => {
          const response = await fetch('http://localhost:8000/api/dashboard');
          return response.json();
        },
        priority: 10,
      },
      {
        key: 'budgets',
        loader: async () => {
          const response = await fetch('http://localhost:8000/api/budgets');
          return response.json();
        },
        priority: 8,
      },
    ],
  },
  {
    page: 'cost-analysis',
    dataLoaders: [
      {
        key: 'costs',
        loader: () => apiService.getCosts({
          startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          endDate: new Date().toISOString().split('T')[0],
          granularity: 'DAILY',
        }),
        priority: 10,
      },
      {
        key: 'cost-attribution',
        loader: () => apiService.getCostAttribution({
          startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          endDate: new Date().toISOString().split('T')[0],
          dimension: 'SERVICE',
        }),
        priority: 8,
      },
    ],
  },
  {
    page: 'budgets',
    dataLoaders: [
      {
        key: 'budgets',
        loader: () => apiService.getBudgets(),
        priority: 10,
      },
    ],
  },
  {
    page: 'optimization',
    dataLoaders: [
      {
        key: 'recommendations',
        loader: () => apiService.getOptimizationRecommendations(),
        priority: 10,
      },
      {
        key: 'ri-recommendations',
        loader: () => apiService.getRIRecommendations(),
        priority: 7,
      },
    ],
  },
  {
    page: 'compliance',
    dataLoaders: [
      {
        key: 'compliance-overview',
        loader: () => apiService.getComplianceOverview(),
        priority: 10,
      },
      {
        key: 'tagging-compliance',
        loader: () => apiService.getTaggingCompliance(),
        priority: 8,
      },
    ],
  },
];

/**
 * Get prefetched data from cache
 */
export function getPrefetchedData(key: string): any | null {
  const cached = prefetchCache.get(key);
  
  if (!cached) return null;
  
  const now = Date.now();
  if (now - cached.timestamp > cached.ttl) {
    prefetchCache.delete(key);
    return null;
  }
  
  console.log(`[Prefetch] Cache hit for ${key}`);
  return cached.data;
}

/**
 * Store prefetched data in cache
 */
function setPrefetchedData(key: string, data: any, ttl: number = DEFAULT_PREFETCH_TTL) {
  prefetchCache.set(key, {
    data,
    timestamp: Date.now(),
    ttl,
  });
  console.log(`[Prefetch] Cached ${key} with TTL ${ttl}ms`);
}

/**
 * Prefetch data for a specific page
 */
export async function prefetchPage(pageName: string): Promise<void> {
  const config = PREFETCH_CONFIGS.find(c => c.page === pageName);
  
  if (!config) {
    console.warn(`[Prefetch] No configuration found for page: ${pageName}`);
    return;
  }
  
  console.log(`[Prefetch] Starting prefetch for ${pageName}`);
  const startTime = Date.now();
  
  // Sort loaders by priority
  const sortedLoaders = [...config.dataLoaders].sort(
    (a, b) => (b.priority || 0) - (a.priority || 0)
  );
  
  // Execute all loaders in parallel
  const promises = sortedLoaders.map(async (loader) => {
    try {
      const data = await loader.loader();
      setPrefetchedData(loader.key, data);
      return { key: loader.key, success: true };
    } catch (error) {
      console.error(`[Prefetch] Failed to load ${loader.key}:`, error);
      return { key: loader.key, success: false };
    }
  });
  
  const results = await Promise.all(promises);
  const duration = Date.now() - startTime;
  const successCount = results.filter(r => r.success).length;
  
  console.log(
    `[Prefetch] Completed ${successCount}/${results.length} loaders for ${pageName} in ${duration}ms`
  );
}

/**
 * Prefetch data for multiple pages
 */
export async function prefetchPages(pageNames: string[]): Promise<void> {
  console.log(`[Prefetch] Prefetching ${pageNames.length} pages`);
  
  const promises = pageNames.map(page => prefetchPage(page));
  await Promise.all(promises);
}

/**
 * Prefetch demo flow pages in sequence
 * This is optimized for the 5-minute demo flow
 */
export async function prefetchDemoFlow(): Promise<void> {
  const demoPages = [
    'dashboard',
    'cost-analysis',
    'budgets',
    'optimization',
    'compliance',
  ];
  
  console.log('[Prefetch] Starting demo flow prefetch');
  
  // Prefetch all demo pages
  await prefetchPages(demoPages);
  
  console.log('[Prefetch] Demo flow prefetch complete');
}

/**
 * Clear prefetch cache
 */
export function clearPrefetchCache(): void {
  prefetchCache.clear();
  console.log('[Prefetch] Cache cleared');
}

/**
 * Clear specific cache entry
 */
export function clearPrefetchEntry(key: string): void {
  prefetchCache.delete(key);
  console.log(`[Prefetch] Cleared cache entry: ${key}`);
}

/**
 * Get cache statistics
 */
export function getPrefetchStats() {
  const entries = Array.from(prefetchCache.entries());
  const now = Date.now();
  
  return {
    totalEntries: entries.length,
    validEntries: entries.filter(([_, v]) => now - v.timestamp <= v.ttl).length,
    expiredEntries: entries.filter(([_, v]) => now - v.timestamp > v.ttl).length,
    cacheSize: entries.reduce((sum, [_, v]) => sum + JSON.stringify(v.data).length, 0),
  };
}
