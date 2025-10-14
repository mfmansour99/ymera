// ============================================================================
// ERROR BOUNDARY - Production Grade with Recovery
// ============================================================================

import React, { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: (error: Error, reset: () => void) => ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorCount: number;
}

export class ErrorBoundary extends React.Component<Props, State> {
  private resetTimeout: NodeJS.Timeout | null = null;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState(
      (prev) => ({
        errorInfo,
        errorCount: prev.errorCount + 1,
      }),
      () => {
        this.props.onError?.(error, errorInfo);

        // Log to error tracking service in production
        if (process.env.NODE_ENV === 'production') {
          console.error('ErrorBoundary caught:', error, errorInfo);
          // Send to Sentry, LogRocket, etc.
        }

        // Auto-recover after 5 seconds if error count < 3
        if (this.state.errorCount < 3) {
          this.resetTimeout = setTimeout(() => this.resetError(), 5000);
        }
      }
    );
  }

  componentWillUnmount(): void {
    if (this.resetTimeout) {
      clearTimeout(this.resetTimeout);
    }
  }

  resetError = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error!, this.resetError);
      }

      return (
        <div
          role="alert"
          style={{
            padding: '2rem',
            background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: '12px',
            color: '#ef4444',
            maxWidth: '600px',
            margin: '2rem auto',
          }}
        >
          <h2 style={{ margin: '0 0 1rem 0', fontSize: '20px', fontWeight: 'bold' }}>
            Something went wrong
          </h2>
          <p style={{ margin: '0 0 1rem 0', fontSize: '14px' }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>

          {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
            <details
              style={{
                marginTop: '1rem',
                padding: '1rem',
                background: 'rgba(0,0,0,0.2)',
                borderRadius: '8px',
                fontSize: '12px',
                fontFamily: 'monospace',
              }}
            >
              <summary style={{ cursor: 'pointer', marginBottom: '0.5rem' }}>
                Error Details
              </summary>
              <pre
                style={{
                  margin: '0.5rem 0 0 0',
                  overflow: 'auto',
                  maxHeight: '200px',
                }}
              >
                {this.state.error?.stack}
                {'\n\nComponent Stack:\n'}
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}

          <button
            onClick={this.resetError}
            style={{
              marginTop: '1rem',
              padding: '10px 20px',
              background: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: '600',
            }}
          >
            Try Again
          </button>

          {this.state.errorCount >= 3 && (
            <p style={{ marginTop: '1rem', fontSize: '12px', color: '#fca5a5' }}>
              This error persists. Please refresh the page or contact support.
            </p>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

// ============================================================================
// MEMOIZATION UTILITIES & HOOKS
// ============================================================================

import { useMemo, useCallback, useRef, useEffect } from 'react';

/**
 * Hook for stable object references to prevent unnecessary re-renders
 */
export const useStableObject = <T extends Record<string, any>>(obj: T): T => {
  const ref = useRef<T>(obj);

  useEffect(() => {
    ref.current = obj;
  }, [obj]);

  return useMemo(() => ref.current, []);
};

/**
 * Hook for memoizing expensive computations with dependency tracking
 */
export const useMemoWithPrevious = <T,>(
  factory: () => T,
  deps: React.DependencyList,
  isEqual?: (prev: T | undefined, next: T) => boolean
): T => {
  const prevValueRef = useRef<T>();
  const prevDepsRef = useRef<React.DependencyList>();

  const hasDepChanged = () => {
    if (!prevDepsRef.current) return true;
    if (prevDepsRef.current.length !== deps.length) return true;

    for (let i = 0; i < deps.length; i++) {
      if (!Object.is(prevDepsRef.current[i], deps[i])) {
        return true;
      }
    }
    return false;
  };

  if (hasDepChanged()) {
    const newValue = factory();
    prevValueRef.current = newValue;
    prevDepsRef.current = deps;
    return newValue;
  }

  return prevValueRef.current as T;
};

/**
 * Hook for stable callback references
 */
export const useStableCallback = <T extends (...args: any[]) => any>(
  callback: T
): T => {
  const ref = useRef(callback);

  useEffect(() => {
    ref.current = callback;
  }, [callback]);

  return useCallback(
    ((...args: any[]) => ref.current(...args)) as T,
    []
  );
};

/**
 * Hook for debounced values (useful for search inputs)
 */
export const useDebouncedValue = <T,>(value: T, delayMs: number = 300): T => {
  const [debouncedValue, setDebouncedValue] = React.useState(value);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setDebouncedValue(value);
    }, delayMs);

    return () => clearTimeout(timeout);
  }, [value, delayMs]);

  return debouncedValue;
};

/**
 * Hook for throttled callbacks
 */
export const useThrottledCallback = <T extends (...args: any[]) => any>(
  callback: T,
  delayMs: number = 300
): T => {
  const lastCallRef = useRef<number>(0);
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  return useCallback(
    ((...args: any[]) => {
      const now = Date.now();
      if (now - lastCallRef.current >= delayMs) {
        lastCallRef.current = now;
        callbackRef.current(...args);
      }
    }) as T,
    [delayMs]
  );
};

// ============================================================================
// PERFORMANCE MONITORING HOOK
// ============================================================================

interface PerformanceMetrics {
  renderTime: number;
  componentName: string;
  timestamp: number;
}

export const usePerformanceMonitoring = (componentName: string): void => {
  useEffect(() => {
    const startTime = performance.now();

    return () => {
      const endTime = performance.now();
      const renderTime = endTime - startTime;

      // Only log if render took > 16ms (one frame at 60fps)
      if (renderTime > 16) {
        const metrics: PerformanceMetrics = {
          renderTime,
          componentName,
          timestamp: Date.now(),
        };

        if (process.env.NODE_ENV === 'development') {
          console.warn(
            `⚠️ ${componentName} rendered in ${renderTime.toFixed(2)}ms`,
            metrics
          );
        }

        // Send to analytics in production
        if (process.env.NODE_ENV === 'production') {
          // Send to performance monitoring service
        }
      }
    };
  }, [componentName]);
};

// ============================================================================
// LAZY LOADING COMPONENT
// ============================================================================

interface LazyComponentProps {
  fallback?: ReactNode;
  errorFallback?: (error: Error) => ReactNode;
  children: ReactNode;
}

export const LazyComponent: React.FC<LazyComponentProps> = ({
  fallback = <div style={{ padding: '2rem', textAlign: 'center', color: '#9ca3af' }}>Loading...</div>,
  errorFallback,
  children,
}) => {
  return (
    <React.Suspense fallback={fallback}>
      <ErrorBoundary fallback={errorFallback}>
        {children}
      </ErrorBoundary>
    </React.Suspense>
  );
};

// ============================================================================
// VIRTUALIZED LIST COMPONENT (for large datasets)
// ============================================================================

import { FixedSizeList as List } from 'react-window';

interface VirtualizedListProps<T> {
  items: T[];
  itemSize: number;
  height: number;
  itemKey: (index: number, item: T) => string | number;
  renderItem: (index: number, item: T) => ReactNode;
  overscanCount?: number;
}

export const VirtualizedList = React.memo(
  function VirtualizedListComponent<T>({
    items,
    itemSize,
    height,
    itemKey,
    renderItem,
    overscanCount = 5,
  }: VirtualizedListProps<T>) {
    const Row = React.useMemo(
      () =>
        ({ index, style }: { index: number; style: React.CSSProperties }) => (
          <div key={itemKey(index, items[index])} style={style}>
            {renderItem(index, items[index])}
          </div>
        ),
      [items, itemKey, renderItem]
    );

    return (
      <List
        height={height}
        itemCount={items.length}
        itemSize={itemSize}
        width="100%"
        overscanCount={overscanCount}
      >
        {Row}
      </List>
    );
  }
) as <T,>(props: VirtualizedListProps<T>) => JSX.Element;

// ============================================================================
// DEBOUNCED API CALL HOOK
// ============================================================================

interface UseDebouncedApiProps<T> {
  apiCall: (value: string) => Promise<T>;
  delayMs?: number;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
}

export const useDebouncedApi = <T,>({
  apiCall,
  delayMs = 500,
  onSuccess,
  onError,
}: UseDebouncedApiProps<T>) => {
  const [value, setValue] = React.useState('');
  const [results, setResults] = React.useState<T | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  const debouncedSearch = useThrottledCallback(async (searchValue: string) => {
    if (!searchValue) {
      setResults(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiCall(searchValue);
      setResults(data);
      onSuccess?.(data);
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Unknown error');
      setError(error);
      onError?.(error);
    } finally {
      setLoading(false);
    }
  }, delayMs);

  const handleChange = useCallback((newValue: string) => {
    setValue(newValue);
    debouncedSearch(newValue);
  }, [debouncedSearch]);

  return { value, results, loading, error, onChange: handleChange };
};

// ============================================================================
// INFINITE SCROLL HOOK
// ============================================================================

interface UseInfiniteScrollProps {
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  threshold?: number;
}

export const useInfiniteScroll = ({
  onLoadMore,
  hasMore,
  threshold = 0.8,
}: UseInfiniteScrollProps) => {
  const observerTarget = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!observerTarget.current || !hasMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.is