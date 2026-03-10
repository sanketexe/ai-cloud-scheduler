/**
 * Parallel API Executor
 * 
 * Executes multiple API calls concurrently to minimize total load time
 * Validates: Requirements 5.5
 */

export interface APICall<T = any> {
  name: string;
  fn: () => Promise<T>;
  priority?: number; // Higher priority executes first
  timeout?: number; // Optional timeout in ms
}

export interface APIResult<T = any> {
  name: string;
  success: boolean;
  data?: T;
  error?: Error;
  duration: number;
}

/**
 * Execute multiple API calls in parallel
 * Returns results for all calls, even if some fail
 */
export async function executeParallel<T = any>(
  calls: APICall<T>[]
): Promise<APIResult<T>[]> {
  const startTime = Date.now();
  
  console.log(`[ParallelAPI] Executing ${calls.length} calls in parallel`);

  // Sort by priority (higher first)
  const sortedCalls = [...calls].sort((a, b) => 
    (b.priority || 0) - (a.priority || 0)
  );

  // Execute all calls concurrently
  const promises = sortedCalls.map(async (call) => {
    const callStartTime = Date.now();
    
    try {
      // Add timeout if specified
      const promise = call.timeout 
        ? Promise.race([
            call.fn(),
            new Promise<never>((_, reject) => 
              setTimeout(() => reject(new Error('Timeout')), call.timeout)
            )
          ])
        : call.fn();

      const data = await promise;
      const duration = Date.now() - callStartTime;
      
      console.log(`[ParallelAPI] ${call.name} completed in ${duration}ms`);
      
      return {
        name: call.name,
        success: true,
        data,
        duration,
      };
    } catch (error) {
      const duration = Date.now() - callStartTime;
      
      console.error(`[ParallelAPI] ${call.name} failed after ${duration}ms:`, error);
      
      return {
        name: call.name,
        success: false,
        error: error as Error,
        duration,
      };
    }
  });

  const results = await Promise.all(promises);
  const totalDuration = Date.now() - startTime;
  
  const successCount = results.filter(r => r.success).length;
  console.log(
    `[ParallelAPI] Completed ${successCount}/${calls.length} calls in ${totalDuration}ms`
  );

  return results;
}

/**
 * Execute API calls in batches to avoid overwhelming the server
 */
export async function executeBatched<T = any>(
  calls: APICall<T>[],
  batchSize: number = 5
): Promise<APIResult<T>[]> {
  const results: APIResult<T>[] = [];
  
  console.log(`[ParallelAPI] Executing ${calls.length} calls in batches of ${batchSize}`);

  for (let i = 0; i < calls.length; i += batchSize) {
    const batch = calls.slice(i, i + batchSize);
    const batchResults = await executeParallel(batch);
    results.push(...batchResults);
  }

  return results;
}

/**
 * Execute API calls with retry logic for failed calls
 */
export async function executeWithRetry<T = any>(
  calls: APICall<T>[],
  maxRetries: number = 2
): Promise<APIResult<T>[]> {
  let results = await executeParallel(calls);
  
  // Retry failed calls
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const failedCalls = calls.filter((call, index) => !results[index].success);
    
    if (failedCalls.length === 0) break;
    
    console.log(`[ParallelAPI] Retrying ${failedCalls.length} failed calls (attempt ${attempt})`);
    
    const retryResults = await executeParallel(failedCalls);
    
    // Update results with retry outcomes
    let retryIndex = 0;
    results = results.map((result) => {
      if (!result.success && retryIndex < retryResults.length) {
        return retryResults[retryIndex++];
      }
      return result;
    });
  }

  return results;
}
