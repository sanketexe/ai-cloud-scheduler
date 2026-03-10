/**
 * Retry Logic with Exponential Backoff
 * Validates: Requirements 4.4
 */

export interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  retryableStatuses: number[];
}

export const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  initialDelay: 1000, // 1 second
  maxDelay: 10000, // 10 seconds
  backoffMultiplier: 2,
  retryableStatuses: [408, 429, 500, 502, 503, 504],
};

/**
 * Check if error is retryable
 */
export function isRetryableError(error: any, config: RetryConfig = DEFAULT_RETRY_CONFIG): boolean {
  // Network errors are retryable
  if (!error.response) {
    return true;
  }

  // Check if status code is retryable
  const status = error.response?.status;
  return config.retryableStatuses.includes(status);
}

/**
 * Calculate delay for retry attempt
 */
export function calculateRetryDelay(
  attemptNumber: number,
  config: RetryConfig = DEFAULT_RETRY_CONFIG
): number {
  const delay = config.initialDelay * Math.pow(config.backoffMultiplier, attemptNumber - 1);
  return Math.min(delay, config.maxDelay);
}

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Retry a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig = DEFAULT_RETRY_CONFIG,
  onRetry?: (attempt: number, error: any) => void
): Promise<T> {
  let lastError: any;

  for (let attempt = 1; attempt <= config.maxRetries + 1; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry if this is the last attempt
      if (attempt > config.maxRetries) {
        break;
      }

      // Don't retry if error is not retryable
      if (!isRetryableError(error, config)) {
        break;
      }

      // Calculate delay and wait
      const delay = calculateRetryDelay(attempt, config);
      
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log(
        `Retry attempt ${attempt}/${config.maxRetries} after ${delay}ms`,
        errorMessage
      );

      // Call retry callback if provided
      if (onRetry) {
        onRetry(attempt, error);
      }

      await sleep(delay);
    }
  }

  // All retries failed, throw the last error
  throw lastError;
}

/**
 * Retry with custom condition
 */
export async function retryWithCondition<T>(
  fn: () => Promise<T>,
  shouldRetry: (error: any, attempt: number) => boolean,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const fullConfig = { ...DEFAULT_RETRY_CONFIG, ...config };
  let lastError: any;

  for (let attempt = 1; attempt <= fullConfig.maxRetries + 1; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry if this is the last attempt
      if (attempt > fullConfig.maxRetries) {
        break;
      }

      // Check custom retry condition
      if (!shouldRetry(error, attempt)) {
        break;
      }

      // Calculate delay and wait
      const delay = calculateRetryDelay(attempt, fullConfig);
      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Retry with rate limit handling
 */
export async function retryWithRateLimit<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const fullConfig = { ...DEFAULT_RETRY_CONFIG, ...config };
  let lastError: any;

  for (let attempt = 1; attempt <= fullConfig.maxRetries + 1; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;

      // Don't retry if this is the last attempt
      if (attempt > fullConfig.maxRetries) {
        break;
      }

      // Check if it's a rate limit error
      const isRateLimit = error.response?.status === 429;
      if (!isRateLimit && !isRetryableError(error, fullConfig)) {
        break;
      }

      // Use Retry-After header if available
      let delay: number;
      if (isRateLimit) {
        const retryAfter = error.response?.headers['retry-after'];
        if (retryAfter) {
          delay = parseInt(retryAfter, 10) * 1000; // Convert to milliseconds
        } else {
          delay = calculateRetryDelay(attempt, fullConfig);
        }
      } else {
        delay = calculateRetryDelay(attempt, fullConfig);
      }

      console.log(
        `Rate limit retry attempt ${attempt}/${fullConfig.maxRetries} after ${delay}ms`
      );

      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Queue for managing retry requests
 */
export class RetryQueue {
  private queue: Array<() => Promise<any>> = [];
  private processing = false;
  private concurrency = 1;

  constructor(concurrency: number = 1) {
    this.concurrency = concurrency;
  }

  /**
   * Add request to queue
   */
  async enqueue<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try {
          const result = await fn();
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });

      this.processQueue();
    });
  }

  /**
   * Process queued requests
   */
  private async processQueue() {
    if (this.processing || this.queue.length === 0) {
      return;
    }

    this.processing = true;

    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.concurrency);
      await Promise.all(batch.map(fn => fn()));
    }

    this.processing = false;
  }

  /**
   * Get queue size
   */
  getQueueSize(): number {
    return this.queue.length;
  }

  /**
   * Clear queue
   */
  clear() {
    this.queue = [];
  }
}

/**
 * Global retry queue for rate-limited requests
 */
export const globalRetryQueue = new RetryQueue(3);
