import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, X } from 'lucide-react';

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

export class ErrorBoundary extends Component<Props, State> {
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
        // Call custom error handler if provided
        this.props.onError?.(error, errorInfo);

        // Log to console in development
        if (process.env.NODE_ENV === 'development') {
          console.group('🚨 ErrorBoundary Caught Error');
          console.error('Error:', error);
          console.error('Error Info:', errorInfo);
          console.groupEnd();
        }

        // Auto-recover after 5 seconds if error count < 3
        if (this.state.errorCount < 3) {
          this.resetTimeout = setTimeout(() => {
            this.resetError();
          }, 5000);
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
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback(this.state.error!, this.resetError);
      }

      // Default fallback UI
      return (
        <div
          role="alert"
          className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4"
        >
          <div className="max-w-2xl w-full">
            <div
              style={{
                backdropFilter: 'blur(20px)',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '16px',
                padding: '2rem',
              }}
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <div
                    style={{
                      width: '64px',
                      height: '64px',
                      borderRadius: '12px',
                      background: 'rgba(239, 68, 68, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <AlertTriangle size={32} style={{ color: '#ef4444' }} />
                  </div>
                  <div>
                    <h2
                      style={{
                        fontSize: '24px',
                        fontWeight: 'bold',
                        color: 'white',
                        marginBottom: '8px',
                      }}
                    >
                      Something went wrong
                    </h2>
                    <p style={{ color: '#fca5a5', fontSize: '14px' }}>
                      {this.state.error?.message || 'An unexpected error occurred'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Error Details (Development Only) */}
              {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                <details
                  style={{
                    marginBottom: '1.5rem',
                    padding: '1rem',
                    background: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '8px',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                  }}
                >
                  <summary
                    style={{
                      cursor: 'pointer',
                      marginBottom: '0.5rem',
                      color: '#fca5a5',
                      fontWeight: '600',
                      fontSize: '14px',
                    }}
                  >
                    🔍 Error Details (Development Only)
                  </summary>
                  <pre
                    style={{
                      margin: '0.5rem 0 0 0',
                      overflow: 'auto',
                      maxHeight: '300px',
                      fontSize: '12px',
                      fontFamily: 'monospace',
                      color: '#fca5a5',
                      background: 'rgba(0, 0, 0, 0.2)',
                      padding: '1rem',
                      borderRadius: '4px',
                    }}
                  >
                    {this.state.error?.stack}
                    {'\n\n=== Component Stack ===\n'}
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}

              {/* Action Buttons */}
              <div className="flex items-center space-x-3">
                <button
                  onClick={this.resetError}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '12px 24px',
                    background: 'linear-gradient(to right, #ef4444, #dc2626)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontWeight: '600',
                    fontSize: '14px',
                  }}
                >
                  <RefreshCw size={16} />
                  Try Again
                </button>

                <button
                  onClick={() => (window.location.href = '/')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '12px 24px',
                    background: 'rgba(255, 255, 255, 0.1)',
                    color: 'white',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontWeight: '600',
                    fontSize: '14px',
                  }}
                >
                  <Home size={16} />
                  Go Home
                </button>
              </div>

              {/* Persistent Error Warning */}
              {this.state.errorCount >= 3 && (
                <div
                  style={{
                    marginTop: '1.5rem',
                    padding: '12px',
                    background: 'rgba(251, 191, 36, 0.1)',
                    border: '1px solid rgba(251, 191, 36, 0.3)',
                    borderRadius: '8px',
                  }}
                >
                  <p
                    style={{
                      margin: 0,
                      fontSize: '13px',
                      color: '#fbbf24',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                  >
                    <AlertTriangle size={14} />
                    This error persists. Please refresh the page or contact support.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// ============================================================================
// EXAMPLE USAGE
// ============================================================================

// Wrap your entire app:
export default function AppWithErrorBoundary() {
  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        // Send to error tracking service (Sentry, LogRocket, etc.)
        console.error('Error caught by boundary:', error, errorInfo);
      }}
    >
      <YourMainApp />
    </ErrorBoundary>
  );
}

// Or wrap specific sections:
function Dashboard() {
  return (
    <ErrorBoundary>
      <DashboardContent />
    </ErrorBoundary>
  );
}

// Custom fallback example:
function PageWithCustomFallback() {
  return (
    <ErrorBoundary
      fallback={(error, reset) => (
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <h2>Oops! Something went wrong in this section.</h2>
          <p>{error.message}</p>
          <button onClick={reset}>Try Again</button>
        </div>
      )}
    >
      <YourComponent />
    </ErrorBoundary>
  );
}

// Demo component to show error
function YourMainApp() {
  const [shouldError, setShouldError] = React.useState(false);

  if (shouldError) {
    throw new Error('This is a test error!');
  }

  return (
    <div style={{ padding: '2rem', textAlign: 'center', color: 'white' }}>
      <h1>App is running normally</h1>
      <button
        onClick={() => setShouldError(true)}
        style={{
          padding: '12px 24px',
          background: '#ef4444',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
          marginTop: '1rem',
        }}
      >
        Trigger Error (Test)
      </button>
    </div>
  );
}