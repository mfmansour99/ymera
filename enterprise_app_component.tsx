import React, { Suspense, lazy, useEffect, useState } from 'react'
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { ErrorBoundary } from 'react-error-boundary'
import { Toaster } from 'react-hot-toast'
import { HelmetProvider, Helmet } from 'react-helmet-async'
import { AnimatePresence, motion } from 'framer-motion'
import { useAuth, Permission } from './store/auth'
import { Loader2, AlertTriangle, Wifi, WifiOff, ShieldAlert } from 'lucide-react'

// Lazy load components for better performance
const Home = lazy(() => import('./pages/Home'))
const Login = lazy(() => import('./pages/Login'))
const Agents = lazy(() => import('./pages/Agents'))
const Chat = lazy(() => import('./pages/Chat'))
const Settings = lazy(() => import('./pages/Settings'))
const Admin = lazy(() => import('./pages/Admin'))
const NotFound = lazy(() => import('./pages/NotFound'))

// Layout components
const Layout = lazy(() => import('./components/Layout'))
const NavBar = lazy(() => import('./components/NavBar'))

// Enhanced Loading Spinner Component
const FullPageSpinner = ({ message = 'Loading...' }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm">
    <div className="flex flex-col items-center space-y-4">
      <div className="relative">
        <div className="w-12 h-12 border-3 border-ymera-glow/30 border-t-ymera-glow rounded-full animate-spin"></div>
        <div className="absolute inset-0 w-12 h-12 border-3 border-transparent border-t-ymera-accent/50 rounded-full animate-spin animate-reverse"></div>
      </div>
      <p className="text-slate-300 text-sm font-medium animate-pulse">{message}</p>
    </div>
  </div>
)

// Enhanced Error Fallback Component
const ErrorFallback = ({ error, resetErrorBoundary }) => (
  <div className="min-h-screen bg-black flex items-center justify-center p-4">
    <div className="max-w-md w-full bg-slate-900 border border-red-500/30 rounded-lg p-6 text-center">
      <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
      <h2 className="text-xl font-bold text-white mb-2">Something went wrong</h2>
      <p className="text-slate-400 mb-4 text-sm">
        {error?.message || 'An unexpected error occurred'}
      </p>
      <div className="flex flex-col sm:flex-row gap-2">
        <button
          onClick={resetErrorBoundary}
          className="flex-1 px-4 py-2 bg-ymera-glow text-black font-medium rounded-md hover:bg-ymera-glow/90 transition-colors"
        >
          Try Again
        </button>
        <button
          onClick={() => window.location.reload()}
          className="flex-1 px-4 py-2 bg-slate-700 text-white font-medium rounded-md hover:bg-slate-600 transition-colors"
        >
          Reload Page
        </button>
      </div>
    </div>
  </div>
)

// Network Status Monitor
const NetworkStatusMonitor = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [showStatus, setShowStatus] = useState(false)

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true)
      setShowStatus(true)
      setTimeout(() => setShowStatus(false), 3000)
    }

    const handleOffline = () => {
      setIsOnline(false)
      setShowStatus(true)
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  if (!showStatus && isOnline) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -50 }}
        className={`fixed top-16 left-1/2 transform -translate-x-1/2 z-40 px-4 py-2 rounded-lg shadow-lg ${
          isOnline 
            ? 'bg-green-600 text-white' 
            : 'bg-red-600 text-white'
        }`}
      >
        <div className="flex items-center space-x-2">
          {isOnline ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          <span className="text-sm font-medium">
            {isOnline ? 'Back online' : 'No internet connection'}
          </span>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

// Protected Route Component with Enhanced Permission Checking
const ProtectedRoute = ({ 
  children, 
  requirePermission, 
  requireRole, 
  fallback = <Navigate to="/login" replace /> 
}) => {
  const { isAuthenticated, hasPermission, hasRole, user } = useAuth()
  
  if (!isAuthenticated()) {
    return fallback
  }

  if (requirePermission && !hasPermission(requirePermission)) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-slate-900 border border-red-500/30 rounded-lg p-6 text-center">
          <ShieldAlert className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Access Denied</h2>
          <p className="text-slate-400 mb-4">
            You don't have permission to access this resource.
          </p>
          <p className="text-xs text-slate-500">
            Required permission: <code>{requirePermission}</code>
          </p>
          <div className="mt-4">
            <button
              onClick={() => window.history.back()}
              className="px-4 py-2 bg-slate-700 text-white font-medium rounded-md hover:bg-slate-600 transition-colors"
            >
              Go Back
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (requireRole && !hasRole(requireRole)) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-slate-900 border border-red-500/30 rounded-lg p-6 text-center">
          <ShieldAlert className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Insufficient Role</h2>
          <p className="text-slate-400 mb-4">
            Your role doesn't allow access to this resource.
          </p>
          <p className="text-xs text-slate-500">
            Current role: <code>{user?.role}</code><br />
            Required role: <code>{requireRole}</code>
          </p>
        </div>
      </div>
    )
  }

  return children
}

// Session Expiry Warning Component
const SessionExpiryWarning = () => {
  const { isSessionExpiring, getSessionTimeRemaining, refreshSession, logout } = useAuth()
  const [showWarning, setShowWarning] = useState(false)
  const [timeLeft, setTimeLeft] = useState(0)

  useEffect(() => {
    const checkSession = () => {
      if (isSessionExpiring()) {
        setShowWarning(true)
        setTimeLeft(Math.ceil(getSessionTimeRemaining() / 1000))
      } else {
        setShowWarning(false)
      }
    }

    // Check every 30 seconds
    const interval = setInterval(checkSession, 30000)
    checkSession() // Initial check

    return () => clearInterval(interval)
  }, [isSessionExpiring, getSessionTimeRemaining])

  useEffect(() => {
    if (showWarning && timeLeft > 0) {
      const timer = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) {
            setShowWarning(false)
            logout('session_expired')
            return 0
          }
          return prev - 1
        })
      }, 1000)

      return () => clearInterval(timer)
    }
  }, [showWarning, timeLeft, logout])

  const handleExtendSession = async () => {
    try {
      await refreshSession()
      setShowWarning(false)
    } catch (error) {
      console.error('Failed to refresh session:', error)
    }
  }

  if (!showWarning) return null

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      >
        <div className="bg-slate-900 border border-yellow-500/50 rounded-lg p-6 max-w-md w-full mx-4">
          <div className="flex items-center mb-4">
            <AlertTriangle className="w-6 h-6 text-yellow-500 mr-3" />
            <h3 className="text-lg font-bold text-white">Session Expiring</h3>
          </div>
          <p className="text-slate-300 mb-4">
            Your session will expire in <strong>{Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, '0')}</strong>
          </p>
          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleExtendSession}
              className="flex-1 px-4 py-2 bg-ymera-glow text-black font-medium rounded-md hover:bg-ymera-glow/90 transition-colors"
            >
              Extend Session
            </button>
            <button
              onClick={() => logout('user_initiated')}
              className="flex-1 px-4 py-2 bg-slate-700 text-white font-medium rounded-md hover:bg-slate-600 transition-colors"
            >
              Logout Now
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}

// Auth Initialization Guard
const AuthInitializationGuard = ({ children }) => {
  const { isInitialized, initialize, isLoading } = useAuth()
  const [initError, setInitError] = useState(null)

  useEffect(() => {
    if (!isInitialized) {
      initialize().catch(error => {
        console.error('Auth initialization failed:', error)
        setInitError(error)
      })
    }
  }, [isInitialized, initialize])

  if (initError) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-slate-900 border border-red-500/30 rounded-lg p-6 text-center">
          <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Initialization Failed</h2>
          <p className="text-slate-400 mb-4">
            Failed to initialize the application. Please refresh the page.
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-ymera-glow text-black font-medium rounded-md hover:bg-ymera-glow/90 transition-colors"
          >
            Reload
          </button>
        </div>
      </div>
    )
  }

  if (!isInitialized || isLoading) {
    return <FullPageSpinner message="Initializing secure session..." />
  }

  return children
}

// Main App Component
const App = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const { isAuthenticated, user } = useAuth()

  // Page transitions
  const pageVariants = {
    initial: { opacity: 0, y: 20 },
    in: { opacity: 1, y: 0 },
    out: { opacity: 0, y: -20 }
  }

  const pageTransition = {
    type: 'tween',
    ease: 'anticipate',
    duration: 0.5
  }

  return (
    <HelmetProvider>
      <Helmet>
        <title>Ymera - AI Agent Platform</title>
        <meta name="description" content="Advanced AI Agent Platform for Enterprise Solutions" />
        <meta name="theme-color" content="#64f4ac" />
      </Helmet>

      <ErrorBoundary
        FallbackComponent={ErrorFallback}
        onError={(error, errorInfo) => {
          console.error('App-level error:', error, errorInfo)
          // In production, send to error monitoring service
        }}
        onReset={() => {
          // Clear any error state and potentially redirect
          navigate('/', { replace: true })
        }}
      >
        <AuthInitializationGuard>
          <div className="min-h-screen bg-black text-white">
            <NavBar />
            <NetworkStatusMonitor />
            <SessionExpiryWarning />

            <main className="pt-16">
              <Suspense fallback={<FullPageSpinner message="Loading page..." />}>
                <AnimatePresence mode="wait" initial={false}>
                  <motion.div
                    key={location.pathname}
                    initial="initial"
                    animate="in"
                    exit="out"
                    variants={pageVariants}
                    transition={pageTransition}
                  >
                    <Routes>
                      {/* Public Routes */}
                      <Route 
                        path="/login" 
                        element={
                          isAuthenticated() ? 
                            <Navigate to="/" replace /> : 
                            <Login />
                        } 
                      />
                      
                      {/* Protected Routes */}
                      <Route 
                        path="/" 
                        element={
                          <ProtectedRoute>
                            <Home />
                          </ProtectedRoute>
                        } 
                      />

                      <Route 
                        path="/agents" 
                        element={
                          <ProtectedRoute requirePermission={Permission.AGENT_READ}>
                            <Agents />
                          </ProtectedRoute>
                        } 
                      />

                      <Route 
                        path="/chat" 
                        element={
                          <ProtectedRoute requirePermission={Permission.AGENT_EXECUTE}>
                            <Chat />
                          </ProtectedRoute>
                        } 
                      />

                      <Route 
                        path="/settings" 
                        element={
                          <ProtectedRoute>
                            <Settings />
                          </ProtectedRoute>
                        } 
                      />

                      {/* Admin Routes */}
                      <Route 
                        path="/admin/*" 
                        element={
                          <ProtectedRoute requirePermission={Permission.SYSTEM_ADMIN}>
                            <Admin />
                          </ProtectedRoute>
                        } 
                      />

                      {/* 404 Route */}
                      <Route path="*" element={<NotFound />} />
                    </Routes>
                  </motion.div>
                </AnimatePresence>
              </Suspense>
            </main>
          </div>

          {/* Global Toast Notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              className: 'bg-slate-900 border border-slate-700 text-slate-100',
              style: {
                background: 'rgb(15 23 42)',
                borderColor: 'rgb(51 65 85)',
                color: 'rgb(241 245 249)'
              },
              success: {
                iconTheme: {
                  primary: '#64f4ac',
                  secondary: '#000000'
                }
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#ffffff'
                }
              }
            }}
          />
        </AuthInitializationGuard>
      </ErrorBoundary>
    </HelmetProvider>
  )
}

export default App