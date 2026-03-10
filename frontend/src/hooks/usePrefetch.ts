/**
 * usePrefetch Hook
 * 
 * React hook for managing data prefetching
 * Validates: Requirements 5.2
 */

import { useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { prefetchPage, prefetchDemoFlow } from '../utils/dataPrefetcher';

// Map routes to page names for prefetching
const ROUTE_TO_PAGE_MAP: Record<string, string> = {
  '/dashboard': 'dashboard',
  '/cost-analysis': 'cost-analysis',
  '/budgets': 'budgets',
  '/optimization': 'optimization',
  '/compliance': 'compliance',
};

// Define which pages to prefetch based on current page
const PREFETCH_STRATEGY: Record<string, string[]> = {
  '/dashboard': ['cost-analysis', 'budgets'],
  '/cost-analysis': ['budgets', 'optimization'],
  '/budgets': ['optimization', 'compliance'],
  '/optimization': ['compliance', 'dashboard'],
  '/compliance': ['dashboard', 'cost-analysis'],
};

/**
 * Hook to automatically prefetch data for likely next pages
 */
export function usePrefetch() {
  const location = useLocation();
  
  useEffect(() => {
    const currentPath = location.pathname;
    const pagesToPrefetch = PREFETCH_STRATEGY[currentPath];
    
    if (pagesToPrefetch && pagesToPrefetch.length > 0) {
      // Prefetch after a short delay to avoid interfering with current page load
      const timeoutId = setTimeout(() => {
        console.log(`[usePrefetch] Prefetching for ${currentPath}:`, pagesToPrefetch);
        pagesToPrefetch.forEach(page => {
          prefetchPage(page).catch(err => {
            console.warn(`[usePrefetch] Failed to prefetch ${page}:`, err);
          });
        });
      }, 1000);
      
      return () => clearTimeout(timeoutId);
    }
  }, [location.pathname]);
}

/**
 * Hook to prefetch demo flow on mount
 */
export function useDemoFlowPrefetch() {
  useEffect(() => {
    // Prefetch demo flow after initial render
    const timeoutId = setTimeout(() => {
      console.log('[useDemoFlowPrefetch] Starting demo flow prefetch');
      prefetchDemoFlow().catch(err => {
        console.warn('[useDemoFlowPrefetch] Failed to prefetch demo flow:', err);
      });
    }, 2000);
    
    return () => clearTimeout(timeoutId);
  }, []);
}

/**
 * Hook to manually trigger prefetch for a specific page
 */
export function useManualPrefetch() {
  const prefetch = useCallback((pageName: string) => {
    console.log(`[useManualPrefetch] Manually prefetching ${pageName}`);
    return prefetchPage(pageName);
  }, []);
  
  return prefetch;
}

/**
 * Hook to prefetch on link hover (for navigation links)
 */
export function usePrefetchOnHover() {
  const prefetchOnHover = useCallback((route: string) => {
    const pageName = ROUTE_TO_PAGE_MAP[route];
    if (pageName) {
      console.log(`[usePrefetchOnHover] Prefetching ${pageName} on hover`);
      prefetchPage(pageName).catch(err => {
        console.warn(`[usePrefetchOnHover] Failed to prefetch ${pageName}:`, err);
      });
    }
  }, []);
  
  return prefetchOnHover;
}
