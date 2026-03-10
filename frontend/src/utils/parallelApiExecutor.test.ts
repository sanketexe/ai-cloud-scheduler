/**
 * Tests for Parallel API Executor
 * 
 * Validates: Requirements 5.5
 */

import { executeParallel, executeBatched, executeWithRetry, APICall } from './parallelApiExecutor';

describe('Parallel API Executor', () => {
  describe('executeParallel', () => {
    it('should execute multiple API calls concurrently', async () => {
      const calls: APICall[] = [
        {
          name: 'call1',
          fn: async () => {
            await new Promise(resolve => setTimeout(resolve, 100));
            return 'result1';
          },
        },
        {
          name: 'call2',
          fn: async () => {
            await new Promise(resolve => setTimeout(resolve, 100));
            return 'result2';
          },
        },
        {
          name: 'call3',
          fn: async () => {
            await new Promise(resolve => setTimeout(resolve, 100));
            return 'result3';
          },
        },
      ];

      const startTime = Date.now();
      const results = await executeParallel(calls);
      const duration = Date.now() - startTime;

      // All calls should complete
      expect(results).toHaveLength(3);
      expect(results.every(r => r.success)).toBe(true);
      
      // Parallel execution should be faster than sequential (< 200ms vs 300ms)
      expect(duration).toBeLessThan(200);
      
      // Results should match
      expect(results[0].data).toBe('result1');
      expect(results[1].data).toBe('result2');
      expect(results[2].data).toBe('result3');
    });

    it('should handle failed API calls gracefully', async () => {
      const calls: APICall[] = [
        {
          name: 'success',
          fn: async () => 'success',
        },
        {
          name: 'failure',
          fn: async () => {
            throw new Error('API Error');
          },
        },
      ];

      const results = await executeParallel(calls);

      expect(results).toHaveLength(2);
      expect(results[0].success).toBe(true);
      expect(results[0].data).toBe('success');
      expect(results[1].success).toBe(false);
      expect(results[1].error?.message).toBe('API Error');
    });

    it('should respect priority ordering', async () => {
      const executionOrder: string[] = [];
      
      const calls: APICall[] = [
        {
          name: 'low-priority',
          fn: async () => {
            executionOrder.push('low');
            return 'low';
          },
          priority: 1,
        },
        {
          name: 'high-priority',
          fn: async () => {
            executionOrder.push('high');
            return 'high';
          },
          priority: 10,
        },
        {
          name: 'medium-priority',
          fn: async () => {
            executionOrder.push('medium');
            return 'medium';
          },
          priority: 5,
        },
      ];

      await executeParallel(calls);

      // Higher priority should be processed first
      expect(executionOrder[0]).toBe('high');
    });

    it('should handle timeout for slow API calls', async () => {
      const calls: APICall[] = [
        {
          name: 'slow-call',
          fn: async () => {
            await new Promise(resolve => setTimeout(resolve, 2000));
            return 'result';
          },
          timeout: 100,
        },
      ];

      const results = await executeParallel(calls);

      expect(results[0].success).toBe(false);
      expect(results[0].error?.message).toBe('Timeout');
    });

    it('should track duration for each call', async () => {
      const calls: APICall[] = [
        {
          name: 'call1',
          fn: async () => {
            await new Promise(resolve => setTimeout(resolve, 50));
            return 'result';
          },
        },
      ];

      const results = await executeParallel(calls);

      expect(results[0].duration).toBeGreaterThanOrEqual(50);
      expect(results[0].duration).toBeLessThan(100);
    });
  });

  describe('executeBatched', () => {
    it('should execute calls in batches', async () => {
      const calls: APICall[] = Array.from({ length: 10 }, (_, i) => ({
        name: `call${i}`,
        fn: async () => `result${i}`,
      }));

      const results = await executeBatched(calls, 3);

      expect(results).toHaveLength(10);
      expect(results.every(r => r.success)).toBe(true);
    });
  });

  describe('executeWithRetry', () => {
    it('should retry failed calls', async () => {
      let attemptCount = 0;
      
      const calls: APICall[] = [
        {
          name: 'flaky-call',
          fn: async () => {
            attemptCount++;
            if (attemptCount < 2) {
              throw new Error('Temporary failure');
            }
            return 'success';
          },
        },
      ];

      const results = await executeWithRetry(calls, 2);

      expect(results[0].success).toBe(true);
      expect(results[0].data).toBe('success');
      expect(attemptCount).toBe(2);
    });

    it('should not retry successful calls', async () => {
      let callCount = 0;
      
      const calls: APICall[] = [
        {
          name: 'success-call',
          fn: async () => {
            callCount++;
            return 'success';
          },
        },
      ];

      await executeWithRetry(calls, 2);

      expect(callCount).toBe(1);
    });
  });
});
