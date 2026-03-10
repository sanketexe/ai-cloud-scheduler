import axios, { AxiosError } from 'axios';

/**
 * Error types for categorization
 */
export enum ErrorType {
  NETWORK = 'NETWORK',
  AUTHENTICATION = 'AUTHENTICATION',
  AUTHORIZATION = 'AUTHORIZATION',
  VALIDATION = 'VALIDATION',
  NOT_FOUND = 'NOT_FOUND',
  RATE_LIMIT = 'RATE_LIMIT',
  SERVER = 'SERVER',
  UNKNOWN = 'UNKNOWN',
}

/**
 * Structured error response
 */
export interface ErrorResponse {
  type: ErrorType;
  message: string;
  userMessage: string;
  technicalDetails?: string;
  statusCode?: number;
  retryable: boolean;
  retryAfter?: number;
  fallbackAvailable: boolean;
}

/**
 * Format user-friendly error messages
 * Validates: Requirements 4.3
 */
export class ErrorMessageFormatter {
  /**
   * Format error into user-friendly message
   */
  static formatError(error: any): ErrorResponse {
    // Handle Axios errors
    if (axios.isAxiosError(error)) {
      return this.formatAxiosError(error);
    }

    // Handle generic errors
    return {
      type: ErrorType.UNKNOWN,
      message: error.message || 'An unexpected error occurred',
      userMessage: 'Something went wrong. Please try again.',
      technicalDetails: error.stack,
      retryable: true,
      fallbackAvailable: false,
    };
  }

  /**
   * Format Axios-specific errors
   */
  private static formatAxiosError(error: AxiosError): ErrorResponse {
    const response = error.response;
    const statusCode = response?.status;

    // Network errors (no response)
    if (!response) {
      return {
        type: ErrorType.NETWORK,
        message: 'Network connection failed',
        userMessage: 'Unable to connect to the server. Please check your internet connection and try again.',
        technicalDetails: error.message,
        retryable: true,
        fallbackAvailable: true,
      };
    }

    // Handle specific status codes
    switch (statusCode) {
      case 400:
        return {
          type: ErrorType.VALIDATION,
          message: 'Invalid request',
          userMessage: this.extractUserMessage(response.data) || 'The request contains invalid data. Please check your input.',
          technicalDetails: JSON.stringify(response.data),
          statusCode,
          retryable: false,
          fallbackAvailable: false,
        };

      case 401:
        return {
          type: ErrorType.AUTHENTICATION,
          message: 'Authentication failed',
          userMessage: 'Your session has expired. Please log in again.',
          technicalDetails: JSON.stringify(response.data),
          statusCode,
          retryable: false,
          fallbackAvailable: false,
        };

      case 403:
        return {
          type: ErrorType.AUTHORIZATION,
          message: 'Access denied',
          userMessage: 'You don\'t have permission to access this resource. Please check your AWS IAM permissions.',
          technicalDetails: JSON.stringify(response.data),
          statusCode,
          retryable: false,
          fallbackAvailable: false,
        };

      case 404:
        return {
          type: ErrorType.NOT_FOUND,
          message: 'Resource not found',
          userMessage: 'The requested resource was not found.',
          technicalDetails: JSON.stringify(response.data),
          statusCode,
          retryable: false,
          fallbackAvailable: false,
        };

      case 429:
        const retryAfter = this.extractRetryAfter(response.headers);
        return {
          type: ErrorType.RATE_LIMIT,
          message: 'Rate limit exceeded',
          userMessage: `Too many requests. Please wait ${retryAfter ? `${retryAfter} seconds` : 'a moment'} before trying again.`,
          technicalDetails: JSON.stringify(response.data),
          statusCode,
          retryable: true,
          retryAfter,
          fallbackAvailable: true,
        };

      case 500:
      case 502:
      case 503:
      case 504:
        return {
          type: ErrorType.SERVER,
          message: 'Server error',
          userMessage: 'The server encountered an error. We\'ll use cached data if available.',
          technicalDetails: JSON.stringify(response.data),
          statusCode,
          retryable: true,
          fallbackAvailable: true,
        };

      default:
        return {
          type: ErrorType.UNKNOWN,
          message: `HTTP ${statusCode} error`,
          userMessage: 'An unexpected error occurred. Please try again.',
          technicalDetails: JSON.stringify(response.data),
          statusCode,
          retryable: true,
          fallbackAvailable: false,
        };
    }
  }

  /**
   * Extract user-friendly message from API response
   */
  private static extractUserMessage(data: any): string | undefined {
    if (!data) return undefined;

    // Try different common error message fields
    if (typeof data === 'string') return data;
    if (data.message) return data.message;
    if (data.error?.message) return data.error.message;
    if (data.detail) return data.detail;
    
    return undefined;
  }

  /**
   * Extract retry-after value from headers
   */
  private static extractRetryAfter(headers: any): number | undefined {
    if (!headers) return undefined;
    
    const retryAfter = headers['retry-after'] || headers['Retry-After'];
    if (retryAfter) {
      const seconds = parseInt(retryAfter, 10);
      return isNaN(seconds) ? undefined : seconds;
    }
    
    return undefined;
  }

  /**
   * Get remediation steps for common errors
   */
  static getRemediationSteps(errorType: ErrorType): string[] {
    switch (errorType) {
      case ErrorType.NETWORK:
        return [
          'Check your internet connection',
          'Verify the backend server is running',
          'Try refreshing the page',
        ];

      case ErrorType.AUTHENTICATION:
        return [
          'Log out and log back in',
          'Clear your browser cache',
          'Check if your session has expired',
        ];

      case ErrorType.AUTHORIZATION:
        return [
          'Verify your AWS credentials are configured',
          'Check your IAM permissions include Cost Explorer access',
          'Contact your AWS administrator for access',
        ];

      case ErrorType.RATE_LIMIT:
        return [
          'Wait a few moments before trying again',
          'Reduce the frequency of requests',
          'Consider implementing request batching',
        ];

      case ErrorType.SERVER:
        return [
          'Wait a moment and try again',
          'Check the system status page',
          'Contact support if the issue persists',
        ];

      default:
        return [
          'Try refreshing the page',
          'Clear your browser cache',
          'Contact support if the issue persists',
        ];
    }
  }
}

/**
 * Log API calls for troubleshooting
 * Validates: Requirements 4.5
 */
export class APICallLogger {
  private static logs: Array<{
    timestamp: string;
    method: string;
    url: string;
    status?: number;
    duration?: number;
    error?: string;
  }> = [];

  private static maxLogs = 100;

  /**
   * Log API request
   */
  static logRequest(method: string, url: string): number {
    const startTime = Date.now();
    
    this.addLog({
      timestamp: new Date().toISOString(),
      method,
      url,
    });

    return startTime;
  }

  /**
   * Log API response
   */
  static logResponse(method: string, url: string, status: number, startTime: number) {
    const duration = Date.now() - startTime;
    
    this.addLog({
      timestamp: new Date().toISOString(),
      method,
      url,
      status,
      duration,
    });

    console.log(`[API] ${method} ${url} - ${status} (${duration}ms)`);
  }

  /**
   * Log API error
   */
  static logError(method: string, url: string, error: string, startTime: number) {
    const duration = Date.now() - startTime;
    
    this.addLog({
      timestamp: new Date().toISOString(),
      method,
      url,
      duration,
      error,
    });

    console.error(`[API] ${method} ${url} - ERROR: ${error} (${duration}ms)`);
  }

  /**
   * Add log entry
   */
  private static addLog(log: any) {
    this.logs.push(log);
    
    // Keep only the most recent logs
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }
  }

  /**
   * Get all logs
   */
  static getLogs() {
    return [...this.logs];
  }

  /**
   * Clear all logs
   */
  static clearLogs() {
    this.logs = [];
  }

  /**
   * Export logs as JSON
   */
  static exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }

  /**
   * Get logs for a specific URL pattern
   */
  static getLogsByUrl(urlPattern: string) {
    return this.logs.filter(log => log.url.includes(urlPattern));
  }

  /**
   * Get error logs only
   */
  static getErrorLogs() {
    return this.logs.filter(log => log.error);
  }

  /**
   * Get statistics
   */
  static getStats() {
    const totalRequests = this.logs.length;
    const errorCount = this.logs.filter(log => log.error).length;
    const avgDuration = this.logs
      .filter(log => log.duration)
      .reduce((sum, log) => sum + (log.duration || 0), 0) / totalRequests;

    return {
      totalRequests,
      errorCount,
      successCount: totalRequests - errorCount,
      errorRate: totalRequests > 0 ? (errorCount / totalRequests) * 100 : 0,
      avgDuration: Math.round(avgDuration),
    };
  }
}
