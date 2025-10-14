import React, { createContext, useContext, useCallback, useMemo, useState, useEffect } from 'react';

// Enhanced Error Taxonomy
export const ErrorSeverity = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical'
};

export const ErrorCategory = {
  NETWORK: 'network',
  AUTHENTICATION: 'authentication',
  AUTHORIZATION: 'authorization',
  VALIDATION: 'validation',
  CLIENT: 'client',
  SERVER: 'server',
  SECURITY: 'security',
  PERFORMANCE: 'performance',
  UI: 'ui',
  AI_SERVICE: 'ai_service'
};

export const RecoveryStrategy = {
  RETRY: 'retry',
  REFRESH: 'refresh',
  REDIRECT: 'redirect',
  MANUAL: 'manual',
  IGNORE: 'ignore',
  ESCALATE: 'escalate'
};

// Error Classifier
class ErrorClassifier {
  static patterns = new Map([
    [/unauthorized|401/i, {
      category: ErrorCategory.AUTHENTICATION,
      severity: ErrorSeverity.HIGH,
      strategy: RecoveryStrategy.REDIRECT,
      userMessage: 'Your session has expired. Please log in again.',
    }],
    
    [/forbidden|403/i, {
      category: ErrorCategory.AUTHORIZATION,
      severity: ErrorSeverity.MEDIUM,
      strategy: RecoveryStrategy.MANUAL,
      userMessage: 'You don\'t have permission to perform this action.',
    }],
    
    [/network|timeout|connection/i, {
      category: ErrorCategory.NETWORK,
      severity: ErrorSeverity.MEDIUM,
      strategy: RecoveryStrategy.RETRY,
      userMessage: 'Connection issue detected. Retrying automatically...',
    }],
    
    [/validation|invalid|required/i, {
      category: ErrorCategory.VALIDATION,
      severity: ErrorSeverity.LOW,
      strategy: RecoveryStrategy.MANUAL,
      userMessage: 'Please check your input and try again.',
    }],
    
    [/ai_service|model_error|generation_failed/i, {
      category: ErrorCategory.AI_SERVICE,
      severity: ErrorSeverity.HIGH,
      strategy: RecoveryStrategy.RETRY,
      userMessage: 'AI service is temporarily unavailable. Retrying...',
    }],
    
    [/rate.?limit|429/i, {
      category: ErrorCategory.PERFORMANCE,
      severity: ErrorSeverity.MEDIUM,
      strategy: RecoveryStrategy.RETRY,
      userMessage: 'Too many requests. Please wait a moment...',
    }],
    
    [/security|csrf|xss|injection/i, {
      category: ErrorCategory.SECURITY,
      severity: ErrorSeverity.CRITICAL,
      strategy: RecoveryStrategy.ESCALATE,
      userMessage: 'Security violation detected. This incident has been logged.',
    }],
  ]);

  static classify(error, context = {}) {
    const errorMessage = this.extractErrorMessage(error).toLowerCase();
    const errorCode = this.extractErrorCode(error);
    
    for (const [pattern, classification] of this.patterns) {
      if (pattern.test(errorMessage) || pattern.test(String(errorCode))) {
        return {
          ...classification,
          technicalMessage: this.extractTechnicalMessage(error),
          stackTrace: error?.stack,
        };
      }
    }
    
    // Default classification based on error type
    if (error?.response?.status) {
      const status = error.response.status;
      if (status >= 500) {
        return {
          category: ErrorCategory.SERVER,
          severity: ErrorSeverity.HIGH,
          strategy: RecoveryStrategy.RETRY,
          userMessage: 'Server error occurred. Retrying automatically...',
          technicalMessage: `HTTP ${status}: ${error.response.statusText}`,
        };
      } else if (status >= 400) {
        return {
          category: ErrorCategory.CLIENT,
          severity: ErrorSeverity.MEDIUM,
          strategy: RecoveryStrategy.MANUAL,
          userMessage: 'Request error. Please check your input.',
          technicalMessage: `HTTP ${status}: ${error.response.statusText}`,
        };
      }
    }
    
    // Fallback
    return {
      category: ErrorCategory.CLIENT,
      severity: ErrorSeverity.LOW,
      strategy: RecoveryStrategy.MANUAL,
      userMessage: 'An unexpected error occurred.',
      technicalMessage: String(error),
    };
  }

  static extractErrorMessage(error) {
    if (typeof error === 'string') return error;
    if (error?.message) return error.message;
    if (error?.response?.data?.message) return error.response.data.message;
    if (error?.response?.data?.error) return error.response.data.error;
    return 'Unknown error';
  }

  static extractErrorCode(error) {
    if (error?.code) return error.code;
    if (error?.response?.status) return error.response.status;
    if (error?.status) return error.status;
    return 'UNKNOWN';
  }

  static extractTechnicalMessage(error) {
    const message = this.extractErrorMessage(error);
    const code = this.extractErrorCode(error);
    return `${code}: ${message}`;
  }
}

// Error Recovery Engine
class ErrorRecovery {
  static retryDelays = [1000, 2000, 4000, 8000, 16000];

  static async executeStrategy(error, originalAction) {
    try {
      switch (error.strategy) {
        case RecoveryStrategy.RETRY:
          return await this.handleRetry(error, originalAction);
          
        case RecoveryStrategy.REFRESH:
          return await this.handleRefresh();
          
        case RecoveryStrategy.REDIRECT:
          return await this.handleRedirect(error);
          
        case RecoveryStrategy.ESCALATE:
          return await this.handleEscalation(error);
          
        default:
          return false;
      }
    } catch (recoveryError) {
      console.error('Error recovery failed:', recoveryError);
      return false;
    }
  }

  static async handleRetry(error, originalAction) {
    if (!originalAction || error.retryCount >= error.maxRetries) {
      return false;
    }

    const delay = this.retryDelays[Math.min(error.retryCount, this.retryDelays.length - 1)];
    
    await new Promise(resolve => setTimeout(resolve, delay));
    
    try {
      await originalAction();
      return true;
    } catch (retryError) {
      console.warn(`Retry ${error.retryCount + 1} failed:`, retryError);
      return false;
    }
  }

  static async handleRefresh() {
    try {
      window.location.reload();
      return true;
    } catch {
      return false;
    }
  }

  static async handleRedirect(error) {
    try {
      if (error.category === ErrorCategory.AUTHENTICATION) {
        window.location.href = '/login';
      } else {
        window.location.href = '/';
      }
      return true;
    } catch {
      return false;
    }
  }

  static async handleEscalation(error) {
    try {
      console.error('SECURITY INCIDENT:', error);
      return true;
    } catch {
      return false;
    }
  }
}

// Context
const ErrorContext = createContext(null);

// Error Provider Component
export const ErrorProvider = ({ children }) => {
  const [errors, setErrors] = useState([]);
  const [retryActions] = useState(new Map());

  const addError = useCallback((error, context = {}, options = {}) => {
    const classification = ErrorClassifier.classify(error, context);
    const errorId = crypto.randomUUID();
    
    const appError = {
      id: errorId,
      error,
      context: {
        ...context,
        url: window.location.href,
        userAgent: navigator.userAgent,
        timestamp: Date.now(),
      },
      timestamp: new Date().toISOString(),
      category: classification.category || ErrorCategory.CLIENT,
      severity: classification.severity || ErrorSeverity.MEDIUM,
      strategy: classification.strategy || RecoveryStrategy.MANUAL,
      userMessage: classification.userMessage || 'An error occurred',
      technicalMessage: classification.technicalMessage || String(error),
      stackTrace: classification.stackTrace,
      userId: context.userId,
      sessionId: context.sessionId,
      retryCount: 0,
      maxRetries: options.maxRetries || 3,
      recovered: false,
      reportedToBackend: false,
    };

    setErrors(prev => [appError, ...prev.slice(0, 99)]);

    // Log to console in development
    if (import.meta.env.DEV) {
      console.group(`🛑 ${appError.category.toUpperCase()} ERROR [${appError.severity}]`);
      console.error('User Message:', appError.userMessage);
      console.error('Technical:', appError.technicalMessage);
      console.error('Context:', appError.context);
      console.error('Original Error:', error);
      if (appError.stackTrace) {
        console.error('Stack:', appError.stackTrace);
      }
      console.groupEnd();
    }

    return errorId;
  }, []);

  const clearErrors = useCallback(() => {
    setErrors([]);
  }, []);

  const clearError = useCallback((errorId) => {
    setErrors(prev => prev.filter(e => e.id !== errorId));
  }, []);

  const retryError = useCallback(async (errorId) => {
    const error = errors.find(e => e.id === errorId);
    const retryAction = retryActions.get(errorId);
    
    if (!error) return false;

    if (error.retryCount >= error.maxRetries) {
      return false;
    }

    setErrors(prev => prev.map(e => 
      e.id === errorId 
        ? { ...e, retryCount: e.retryCount + 1 }
        : e
    ));

    try {
      const recovered = await ErrorRecovery.executeStrategy(error, retryAction);
      
      if (recovered) {
        setErrors(prev => prev.map(e => 
          e.id === errorId 
            ? { ...e, recovered: true }
            : e
        ));
        return true;
      } else {
        return false;
      }
    } catch (retryError) {
      console.error('Retry execution failed:', retryError);
      return false;
    }
  }, [errors, retryActions]);

  const errorStats = useMemo(() => {
    const total = errors.length;
    const byCategory = errors.reduce((acc, error) => {
      acc[error.category] = (acc[error.category] || 0) + 1;
      return acc;
    }, {});
    
    const bySeverity = errors.reduce((acc, error) => {
      acc[error.severity] = (acc[error.severity] || 0) + 1;
      return acc;
    }, {});
    
    const recovered = errors.filter(e => e.recovered).length;
    const recoveryRate = total > 0 ? (recovered / total) * 100 : 0;
    
    const totalRetries = errors.reduce((sum, e) => sum + e.retryCount, 0);
    const averageRetryCount = total > 0 ? totalRetries / total : 0;

    return {
      total,
      byCategory,
      bySeverity,
      recoveryRate,
      averageRetryCount,
    };
  }, [errors]);

  const criticalErrors = useMemo(() => 
    errors.filter(e => e.severity === ErrorSeverity.CRITICAL && !e.recovered),
    [errors]
  );

  const contextValue = useMemo(() => ({
    errors,
    addError,
    clearErrors,
    clearError,
    retryError,
    hasErrors: errors.length > 0,
    criticalErrors,
    errorStats,
  }), [errors, addError, clearErrors, clearError, retryError, criticalErrors, errorStats]);

  return (
    <ErrorContext.Provider value={contextValue}>
      {children}
      <ErrorOverlay />
      <CriticalErrorModal />
    </ErrorContext.Provider>
  );
};

// Error Overlay Component
const ErrorOverlay = () => {
  const { errors, clearError, retryError } = useErrorHandler();
  const [isExpanded, setIsExpanded] = useState(false);
  
  const visibleErrors = errors.filter(e => 
    e.severity >= ErrorSeverity.MEDIUM && !e.recovered
  ).slice(0, 5);

  if (visibleErrors.length === 0) return null;

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '1rem',
        right: '1rem',
        zIndex: 50,
        maxWidth: '28rem',
        background: 'rgba(15, 23, 42, 0.95)',
        backdropFilter: 'blur(16px)',
        borderRadius: '1rem',
        border: '1px solid rgba(239, 68, 68, 0.2)',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
        overflow: 'hidden'
      }}
    >
      <div 
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          padding: '1rem',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          transition: 'background-color 0.2s'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <span style={{ 
            width: '1.25rem', 
            height: '1.25rem', 
            marginRight: '0.5rem',
            color: '#ef4444'
          }}>⚠️</span>
          <span style={{ 
            fontWeight: '500', 
            color: '#f1f5f9',
            fontSize: '0.875rem'
          }}>
            {visibleErrors.length} Error{visibleErrors.length !== 1 ? 's' : ''}
          </span>
        </div>
        <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
          {isExpanded ? '▲' : '▼'}
        </span>
      </div>
      
      {isExpanded && (
        <div style={{ maxHeight: '16rem', overflowY: 'auto' }}>
          {visibleErrors.map((error, index) => (
            <div
              key={error.id}
              style={{
                padding: '0.75rem',
                borderTop: '1px solid rgba(71, 85, 105, 0.5)'
              }}
            >
              <div style={{ 
                display: 'flex', 
                alignItems: 'flex-start', 
                justifyContent: 'space-between',
                marginBottom: '0.5rem'
              }}>
                <div style={{ flex: 1 }}>
                  <p style={{ 
                    fontSize: '0.875rem', 
                    color: '#e2e8f0',
                    marginBottom: '0.25rem'
                  }}>
                    {error.userMessage}
                  </p>
                  <p style={{ 
                    fontSize: '0.75rem', 
                    color: '#94a3b8'
                  }}>
                    {error.category} • {error.timestamp}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    clearError(error.id);
                  }}
                  style={{
                    marginLeft: '0.5rem',
                    color: '#94a3b8',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: '1rem'
                  }}
                >
                  ✕
                </button>
              </div>
              
              {error.strategy === RecoveryStrategy.RETRY && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    retryError(error.id);
                  }}
                  disabled={error.retryCount >= error.maxRetries}
                  style={{
                    fontSize: '0.75rem',
                    background: error.retryCount >= error.maxRetries 
                      ? 'rgba(100, 116, 139, 0.2)' 
                      : 'rgba(100, 244, 172, 0.2)',
                    color: error.retryCount >= error.maxRetries 
                      ? '#64748b' 
                      : '#64f4ac',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '0.5rem',
                    border: 'none',
                    cursor: error.retryCount >= error.maxRetries ? 'not-allowed' : 'pointer',
                    transition: 'background-color 0.2s'
                  }}
                >
                  {error.retryCount >= error.maxRetries 
                    ? 'Max Retries' 
                    : `Retry (${error.retryCount}/${error.maxRetries})`}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Critical Error Modal Component
const CriticalErrorModal = () => {
  const { criticalErrors, clearError } = useErrorHandler();
  const [currentError, setCurrentError] = useState(null);

  useEffect(() => {
    if (criticalErrors.length > 0 && !currentError) {
      setCurrentError(criticalErrors[0]);
    }
  }, [criticalErrors, currentError]);

  if (!currentError) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
        background: 'rgba(0, 0, 0, 0.8)',
        backdropFilter: 'blur(4px)'
      }}
    >
      <div
        style={{
          background: 'linear-gradient(to bottom right, rgb(30, 41, 59), rgb(15, 23, 42))',
          borderRadius: '1.5rem',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          maxWidth: '32rem',
          width: '100%',
          overflow: 'hidden'
        }}
      >
        <div style={{ 
          padding: '1.5rem',
          borderBottom: '1px solid rgb(51, 65, 85)'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center',
            marginBottom: '1rem'
          }}>
            <div style={{
              padding: '0.75rem',
              background: 'rgba(239, 68, 68, 0.2)',
              borderRadius: '9999px',
              marginRight: '1rem'
            }}>
              <span style={{ fontSize: '2rem' }}>🛡️</span>
            </div>
            <div>
              <h2 style={{ 
                fontSize: '1.25rem', 
                fontWeight: 'bold', 
                color: '#f1f5f9',
                margin: 0
              }}>Critical Error</h2>
              <p style={{ 
                fontSize: '0.875rem', 
                color: '#94a3b8',
                margin: 0
              }}>{currentError.category}</p>
            </div>
          </div>
          
          <p style={{ 
            color: '#e2e8f0',
            marginBottom: '1rem'
          }}>
            {currentError.userMessage}
          </p>
          
          <div style={{
            background: 'rgba(15, 23, 42, 0.5)',
            borderRadius: '0.5rem',
            padding: '0.75rem'
          }}>
            <p style={{ 
              fontSize: '0.75rem', 
              color: '#94a3b8',
              marginBottom: '0.25rem'
            }}>Technical Details:</p>
            <p style={{ 
              fontSize: '0.75rem', 
              color: '#cbd5e1',
              fontFamily: 'monospace',
              wordBreak: 'break-all',
              margin: 0
            }}>
              {currentError.technicalMessage}
            </p>
          </div>
        </div>
        
        <div style={{ 
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem'
        }}>
          <button
            onClick={() => {
              clearError(currentError.id);
              setCurrentError(null);
            }}
            style={{
              flex: 1,
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(to right, #64f4ac, #4fd1c5)',
              color: '#000',
              fontWeight: '600',
              borderRadius: '0.75rem',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontSize: '0.875rem'
            }}
          >
            Acknowledge
          </button>
          
          <button
            onClick={() => window.location.reload()}
            style={{
              flex: 1,
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(to right, rgb(71, 85, 105), rgb(51, 65, 85))',
              color: '#f1f5f9',
              fontWeight: '600',
              borderRadius: '0.75rem',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontSize: '0.875rem'
            }}
          >
            Refresh Page
          </button>
        </div>
      </div>
    </div>
  );
};

// Hook for using error handler
export const useErrorHandler = () => {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error('useErrorHandler must be used within ErrorProvider');
  }
  return context;
};

// Hook for component-level error reporting
export const useErrorReporting = () => {
  const { addError } = useErrorHandler();
  
  return {
    reportError: useCallback((error, context) => {
      return addError(error, context, { 
        showToast: true, 
        reportToBackend: true 
      });
    }, [addError]),
    
    reportWarning: useCallback((message, context) => {
      return addError(new Error(message), context, { 
        showToast: false, 
        reportToBackend: false 
      });
    }, [addError]),
    
    reportCritical: useCallback((error, context) => {
      return addError(error, { ...context, severity: ErrorSeverity.CRITICAL }, { 
        showToast: true, 
        reportToBackend: true,
        autoRetry: false 
      });
    }, [addError]),
  };
};

// Error Fallback Component
export function AppErrorFallback({ error, resetErrorBoundary }) {
  const classification = ErrorClassifier.classify(error);
  
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(to bottom right, rgb(15, 23, 42), rgb(0, 0, 0), rgb(15, 23, 42))',
      color: '#f1f5f9',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '1rem'
    }}>
      <div style={{
        textAlign: 'center',
        maxWidth: '42rem',
        padding: '2rem',
        background: 'linear-gradient(to bottom right, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.5))',
        backdropFilter: 'blur(16px)',
        borderRadius: '1.5rem',
        border: '1px solid rgba(239, 68, 68, 0.2)',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          marginBottom: '1.5rem'
        }}>
          <span style={{ fontSize: '4rem', marginRight: '1rem' }}>⚠️</span>
          <span style={{ fontSize: '3rem' }}>🛡️</span>
        </div>
        
        <h1 style={{
          fontSize: '1.875rem',
          fontWeight: 'bold',
          marginBottom: '1rem',
          background: 'linear-gradient(to right, #ef4444, #dc2626)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          System Error Detected
        </h1>
        
        <p style={{
          fontSize: '1.25rem',
          color: '#cbd5e1',
          marginBottom: '1.5rem'
        }}>
          {classification.userMessage || 'An unexpected error occurred'}
        </p>
        
        <div style={{
          background: 'rgba(15, 23, 42, 0.5)',
          borderRadius: '0.5rem',
          padding: '1rem',
          marginBottom: '2rem',
          textAlign: 'left'
        }}>
          <h3 style={{
            fontSize: '0.875rem',
            fontWeight: '600',
            color: '#94a3b8',
            marginBottom: '0.5rem'
          }}>Technical Details:</h3>
          <p style={{
            fontSize: '0.75rem',
            color: '#cbd5e1',
            fontFamily: 'monospace',
            wordBreak: 'break-all',
            margin: 0
          }}>
            {error.message}
          </p>
        </div>
        
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
          justifyContent: 'center'
        }}>
          <button
            onClick={resetErrorBoundary}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(to right, #64f4ac, #4fd1c5)',
              color: '#000',
              fontWeight: '600',
              borderRadius: '0.75rem',
              border: 'none',
              cursor: 'pointer',
              fontSize: '0.875rem',
              transition: 'all 0.2s'
            }}
          >
            <span style={{ marginRight: '0.5rem' }}>🔄</span>
            Try Again
          </button>
          
          <button
            onClick={() => window.location.href = '/'}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(to right, rgb(71, 85, 105), rgb(51, 65, 85))',
              color: '#f1f5f9',
              fontWeight: '600',
              borderRadius: '0.75rem',
              border: 'none',
              cursor: 'pointer',
              fontSize: '0.875rem',
              transition: 'all 0.2s'
            }}
          >
            <span style={{ marginRight: '0.5rem' }}>🏠</span>
            Go Home
          </button>
        </div>
      </div>
    </div>
  );
}