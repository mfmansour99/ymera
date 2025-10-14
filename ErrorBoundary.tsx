import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-8 text-center">
            <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <i className="fas fa-exclamation-triangle text-red-400 text-3xl" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Something went wrong</h2>
            <p className="text-slate-400 mb-6">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="glass-button bg-accent-primary bg-opacity-20 border-accent-primary"
            >
              <i className="fas fa-redo mr-2" />
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}