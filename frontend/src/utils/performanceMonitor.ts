/**
 * Performance Monitor
 * 
 * Tracks and logs performance metrics for dashboard and page loads
 * Validates: Requirements 5.1, 5.2
 */

import React from 'react';

interface PerformanceMetric {
  name: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  metadata?: Record<string, any>;
}

class PerformanceMonitor {
  private metrics: Map<string, PerformanceMetric> = new Map();
  private thresholds = {
    dashboardLoad: 3000, // 3 seconds
    pageNavigation: 1000, // 1 second
    apiCall: 2000, // 2 seconds
  };

  /**
   * Start tracking a performance metric
   */
  start(name: string, metadata?: Record<string, any>): void {
    this.metrics.set(name, {
      name,
      startTime: performance.now(),
      metadata,
    });
    console.log(`[Performance] Started tracking: ${name}`);
  }

  /**
   * End tracking and calculate duration
   */
  end(name: string): number | null {
    const metric = this.metrics.get(name);
    if (!metric) {
      console.warn(`[Performance] No metric found for: ${name}`);
      return null;
    }

    const endTime = performance.now();
    const duration = endTime - metric.startTime;

    metric.endTime = endTime;
    metric.duration = duration;

    // Check against thresholds
    this.checkThreshold(name, duration);

    console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`);
    return duration;
  }

  /**
   * Check if duration exceeds threshold
   */
  private checkThreshold(name: string, duration: number): void {
    let threshold: number | undefined;
    let metricType: string = 'custom';

    if (name.includes('dashboard') || name.includes('Dashboard')) {
      threshold = this.thresholds.dashboardLoad;
      metricType = 'Dashboard Load';
    } else if (name.includes('navigation') || name.includes('Navigation')) {
      threshold = this.thresholds.pageNavigation;
      metricType = 'Page Navigation';
    } else if (name.includes('api') || name.includes('API')) {
      threshold = this.thresholds.apiCall;
      metricType = 'API Call';
    }

    if (threshold && duration > threshold) {
      console.warn(
        `[Performance] ⚠️ ${metricType} exceeded threshold: ${duration.toFixed(2)}ms > ${threshold}ms`
      );
    } else if (threshold) {
      console.log(
        `[Performance] ✓ ${metricType} within threshold: ${duration.toFixed(2)}ms < ${threshold}ms`
      );
    }
  }

  /**
   * Get metric by name
   */
  getMetric(name: string): PerformanceMetric | undefined {
    return this.metrics.get(name);
  }

  /**
   * Get all metrics
   */
  getAllMetrics(): PerformanceMetric[] {
    return Array.from(this.metrics.values());
  }

  /**
   * Get metrics summary
   */
  getSummary(): {
    totalMetrics: number;
    averageDuration: number;
    slowestMetric: PerformanceMetric | null;
    fastestMetric: PerformanceMetric | null;
  } {
    const metrics = this.getAllMetrics().filter(m => m.duration !== undefined);
    
    if (metrics.length === 0) {
      return {
        totalMetrics: 0,
        averageDuration: 0,
        slowestMetric: null,
        fastestMetric: null,
      };
    }

    const durations = metrics.map(m => m.duration!);
    const averageDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
    
    const slowestMetric = metrics.reduce((prev, current) => 
      (current.duration! > prev.duration!) ? current : prev
    );
    
    const fastestMetric = metrics.reduce((prev, current) => 
      (current.duration! < prev.duration!) ? current : prev
    );

    return {
      totalMetrics: metrics.length,
      averageDuration,
      slowestMetric,
      fastestMetric,
    };
  }

  /**
   * Clear all metrics
   */
  clear(): void {
    this.metrics.clear();
    console.log('[Performance] Cleared all metrics');
  }

  /**
   * Log summary to console
   */
  logSummary(): void {
    const summary = this.getSummary();
    
    console.group('[Performance] Summary');
    console.log(`Total Metrics: ${summary.totalMetrics}`);
    console.log(`Average Duration: ${summary.averageDuration.toFixed(2)}ms`);
    
    if (summary.slowestMetric) {
      console.log(
        `Slowest: ${summary.slowestMetric.name} (${summary.slowestMetric.duration!.toFixed(2)}ms)`
      );
    }
    
    if (summary.fastestMetric) {
      console.log(
        `Fastest: ${summary.fastestMetric.name} (${summary.fastestMetric.duration!.toFixed(2)}ms)`
      );
    }
    
    console.groupEnd();
  }

  /**
   * Track Web Vitals (Core Web Vitals)
   */
  trackWebVitals(): void {
    if ('PerformanceObserver' in window) {
      // Largest Contentful Paint (LCP)
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        console.log(`[Performance] LCP: ${lastEntry.startTime.toFixed(2)}ms`);
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

      // First Input Delay (FID)
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          const fid = entry.processingStart - entry.startTime;
          console.log(`[Performance] FID: ${fid.toFixed(2)}ms`);
        });
      });
      fidObserver.observe({ entryTypes: ['first-input'] });

      // Cumulative Layout Shift (CLS)
      let clsScore = 0;
      const clsObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          if (!entry.hadRecentInput) {
            clsScore += entry.value;
          }
        });
        console.log(`[Performance] CLS: ${clsScore.toFixed(4)}`);
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });
    }
  }
}

// Export singleton instance
export const performanceMonitor = new PerformanceMonitor();

// Export hook for React components
export function usePerformanceTracking(metricName: string, metadata?: Record<string, any>) {
  React.useEffect(() => {
    performanceMonitor.start(metricName, metadata);
    
    return () => {
      performanceMonitor.end(metricName);
    };
  }, [metricName, metadata]);
}
