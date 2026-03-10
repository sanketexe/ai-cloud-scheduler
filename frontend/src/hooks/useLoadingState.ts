import { useState, useCallback } from 'react';

interface LoadingState {
  isLoading: boolean;
  error: string | null;
  progress?: number;
}

interface UseLoadingStateReturn {
  isLoading: boolean;
  error: string | null;
  progress?: number;
  startLoading: (message?: string) => void;
  stopLoading: () => void;
  setError: (error: string) => void;
  clearError: () => void;
  setProgress: (progress: number) => void;
  withLoading: <T>(fn: () => Promise<T>) => Promise<T>;
}

/**
 * Custom hook for managing loading states
 * 
 * Provides a consistent way to handle loading, error, and progress states
 * Validates: Requirements 5.3, 6.2
 */
export const useLoadingState = (initialLoading = false): UseLoadingStateReturn => {
  const [state, setState] = useState<LoadingState>({
    isLoading: initialLoading,
    error: null,
    progress: undefined,
  });

  const startLoading = useCallback((message?: string) => {
    setState({
      isLoading: true,
      error: null,
      progress: undefined,
    });
  }, []);

  const stopLoading = useCallback(() => {
    setState(prev => ({
      ...prev,
      isLoading: false,
    }));
  }, []);

  const setError = useCallback((error: string) => {
    setState(prev => ({
      ...prev,
      isLoading: false,
      error,
    }));
  }, []);

  const clearError = useCallback(() => {
    setState(prev => ({
      ...prev,
      error: null,
    }));
  }, []);

  const setProgress = useCallback((progress: number) => {
    setState(prev => ({
      ...prev,
      progress: Math.min(100, Math.max(0, progress)),
    }));
  }, []);

  const withLoading = useCallback(async <T,>(fn: () => Promise<T>): Promise<T> => {
    startLoading();
    try {
      const result = await fn();
      stopLoading();
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An error occurred';
      setError(errorMessage);
      throw error;
    }
  }, [startLoading, stopLoading, setError]);

  return {
    isLoading: state.isLoading,
    error: state.error,
    progress: state.progress,
    startLoading,
    stopLoading,
    setError,
    clearError,
    setProgress,
    withLoading,
  };
};

export default useLoadingState;
