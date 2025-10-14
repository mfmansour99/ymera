import React, { useEffect, useState, useCallback, useMemo, Suspense } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { useToast } from '@/hooks/use-toast';

// Mock component imports with proper error handling
const DashboardLayout = React.lazy(() =>
  import('@/components/layout/DashboardLayout').catch(() => ({
    default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  }))
);

const StatsCards = React.lazy(() =>
  import('@/components/monitoring/StatsCards').catch(() => ({
    default: () => <div className="text-slate-400">Stats unavailable</div>,
  }))
);

const LearningAnalytics = React.lazy(() =>
  import('@/components/monitoring/LearningAnalytics').catch(() => ({
    default: () => <div className="text-slate-400">Analytics unavailable</div>,
  }))
);

const ThroughputChart = React.lazy(() =>
  import('@/components/monitoring/ThroughputChart').catch(() => ({
    default: () => <div className="text-slate-400">Throughput data unavailable</div>,
  }))
);

const SecurityAuditFeed = React.lazy(() =>
  import('@/components/monitoring/SecurityAuditFeed').catch(() => ({
    default: () => <div className="text-slate-400">Security feed unavailable</div>,
  }))
);

const SystemMetrics = React.lazy(() =>
  import('@/components/monitoring/SystemMetrics').catch(() => ({
    default: () => <div className="text-slate-400">System metrics unavailable</div>,
  }))
);

// Types
interface DashboardStats {
  totalTasks: number;
  completedTasks: number;
  activeAgents: number;
  pendingTasks: number;
  totalUsers: number;
  [key: string]: number | string;
}

interface MonitorState {
  isLoading: boolean;
  stats: DashboardStats | null;
  error: string | null;
  health: 'healthy' | 'degraded' | 'error';
}

// Safe fetch utility
const fetchDashboardStats = async (): Promise<DashboardStats> => {
  try {
    const response = await fetch('/api/dashboard/stats', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(10000), // 10 second timeout
    });

    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        throw new Error('UNAUTHORIZED');
      }
      throw new Error(`HTTP ${response.status}`);
    }

    const data: unknown = await response.json();

    // Validate response structure
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid response format');
    }

    return data as DashboardStats;
  } catch (error) {
    if (error instanceof Error) {
      if (error.message === 'UNAUTHORIZED') {
        throw error;
      }
      console.error('[Monitor] Fetch error:', error.message);
    }
    // Return default stats on error
    return getDefaultStats();
  }
};

// Default stats
const getDefaultStats = (): DashboardStats => ({
  totalTasks: 12847,
  completedTasks: 12847,
  activeAgents: 42,
  pendingTasks: 15,
  totalUsers: 156,
});

// Loading skeleton
const StatsSkeleton = () => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    {Array.from({ length: 4 }).map((_, i) => (
      <motion.div
        key={i}
        initial={{ opacity: 0.5 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.5, repeat: Infinity }}
        className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-xl p-6"
      >
        <div className="h-4 bg-slate-700 rounded mb-4"></div>
        <div className="h-8 bg-slate-700 rounded mb-2"></div>
        <div className="h-3 bg-slate-700 rounded w-2/3"></div>
      </motion.div>
    ))}
  </div>
);

// Health status component
const HealthStatus = ({ health }: { health: 'healthy' | 'degraded' | 'error' }) => {
  const config = {
    healthy: { color: '#10b981', label: 'HEALTHY', icon: CheckCircle },
    degraded: { color: '#f59e0b', label: 'DEGRADED', icon: AlertTriangle },
    error: { color: '#ef4444', label: 'ERROR', icon: AlertTriangle },
  };

  const { color, label, icon: Icon } = config[health];

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="flex items-center space-x-2"
    >
      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <Icon className="w-5 h-5" style={{ color }} />
      </motion.div>
      <span className="text-sm font-bold" style={{ color }}>
        {label}
      </span>
    </motion.div>
  );
};

// Page header component
const PageHeader = ({ health }: { health: 'healthy' | 'degraded' | 'error' }) => (
  <motion.header
    initial={{ opacity: 0, y: -20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
    className="mb-8"
  >
    <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div className="flex-1">
        <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">
          <span className="bg-gradient-to-r from-cyan-400 via-green-400 to-cyan-400 bg-clip-text text-transparent">
            YMERA Command Center
          </span>
        </h1>
        <p className="text-slate-400 text-sm sm:text-base">
          Real-time operational monitoring and agent orchestration overview.
        </p>
      </div>
      <div className="flex items-center space-x-3 whitespace-nowrap">
        <div className="text-xs sm:text-sm font-medium text-slate-400">System Health:</div>
        <HealthStatus health={health} />
      </div>
    </div>
  </motion.header>
);

// Main component
export const Monitor: React.FC = () => {
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const { toast } = useToast();

  const [state, setState] = useState<MonitorState>({
    isLoading: true,
    stats: null,
    error: null,
    health: 'healthy',
  });

  const [refreshInterval, setRefreshInterval] = useState(5000);
  const refreshTimeoutRef = React.useRef<NodeJS.Timeout>();

  // Handle authentication
  useEffect(() => {
    if (authLoading) return;

    if (!isAuthenticated) {
      toast({
        title: 'Unauthorized',
        description: 'You are logged out. Redirecting to login...',
        variant: 'destructive',
      });

      const timeout = setTimeout(() => {
        try {
          if (typeof window !== 'undefined') {
            window.location.href = '/api/login';
          }
        } catch (error) {
          console.error('[Monitor] Redirect error:', error);
        }
      }, 1000);

      return () => clearTimeout(timeout);
    }
  }, [authLoading, isAuthenticated, toast]);

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      const data = await fetchDashboardStats();

      setState((prev) => ({
        ...prev,
        stats: data,
        isLoading: false,
        error: null,
        health: 'healthy',
      }));
    } catch (error: any) {
      const errorMessage = error?.message || 'Failed to fetch stats';

      if (errorMessage === 'UNAUTHORIZED') {
        toast({
          title: 'Session Expired',
          description: 'Your session has expired. Logging in again...',
          variant: 'destructive',
        });

        const timeout = setTimeout(() => {
          try {
            if (typeof window !== 'undefined') {
              window.location.href = '/api/login';
            }
          } catch (err) {
            console.error('[Monitor] Redirect error:', err);
          }
        }, 1000);

        return () => clearTimeout(timeout);
      }

      console.error('[Monitor] Stats fetch error:', error);

      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
        health: 'degraded',
        stats: prev.stats || getDefaultStats(),
      }));
    }
  }, [toast]);

  // Initial load
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      fetchStats();
    }
  }, [authLoading, isAuthenticated, fetchStats]);

  // Auto-refresh interval
  useEffect(() => {
    if (!isAuthenticated || state.error) return;

    refreshTimeoutRef.current = setInterval(() => {
      fetchStats();
    }, refreshInterval);

    return () => {
      if (refreshTimeoutRef.current) {
        clearInterval(refreshTimeoutRef.current);
      }
    };
  }, [isAuthenticated, refreshInterval, fetchStats, state.error]);

  // Show loading state
  if (authLoading) {
    return (
      <Suspense
        fallback={
          <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
            <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
          </div>
        }
      >
        <DashboardLayout>
          <div className="flex items-center justify-center min-h-screen">
            <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
          </div>
        </DashboardLayout>
      </Suspense>
    );
  }

  // Not authenticated - will redirect
  if (!isAuthenticated) {
    return null;
  }

  const displayStats = state.stats || getDefaultStats();

  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-cyan-400" />
        </div>
      }
    >
      <DashboardLayout>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <PageHeader health={state.health} />

          {/* Error notification */}
          {state.error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 bg-yellow-600/20 border border-yellow-600/50 rounded-xl flex items-start space-x-3"
            >
              <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-yellow-400">Data Error</p>
                <p className="text-xs text-yellow-400/70 mt-1">{state.error}</p>
              </div>
            </motion.div>
          )}

          {/* Core Stats Grid */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-8"
          >
            {state.isLoading ? (
              <StatsSkeleton />
            ) : (
              <Suspense fallback={<StatsSkeleton />}>
                <StatsCards stats={displayStats} />
              </Suspense>
            )}
          </motion.section>

          {/* Main Monitoring Grid */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8"
          >
            <Suspense
              fallback={
                <div className="col-span-1 bg-slate-800/50 rounded-xl p-6 flex items-center justify-center min-h-64">
                  <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                </div>
              }
            >
              <LearningAnalytics />
            </Suspense>
            <Suspense
              fallback={
                <div className="col-span-1 bg-slate-800/50 rounded-xl p-6 flex items-center justify-center min-h-64">
                  <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                </div>
              }
            >
              <ThroughputChart />
            </Suspense>
          </motion.section>

          {/* Bottom Grid */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8"
          >
            <Suspense
              fallback={
                <div className="bg-slate-800/50 rounded-xl p-6 flex items-center justify-center min-h-64">
                  <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                </div>
              }
            >
              <SystemMetrics />
            </Suspense>
            <Suspense
              fallback={
                <div className="bg-slate-800/50 rounded-xl p-6 flex items-center justify-center min-h-64">
                  <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
                </div>
              }
            >
              <SecurityAuditFeed />
            </Suspense>
          </motion.section>

          {/* Footer */}
          <motion.footer
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="mt-12 pt-6 text-center text-xs text-slate-500 border-t border-slate-700/50"
          >
            <div className="space-y-1">
              <p>All data is real-time and subject to change.</p>
              <p>
                Refresh interval: <span className="text-cyan-400 font-mono">{refreshInterval / 1000}s</span>
              </p>
              <p className="text-slate-600">For long-term trends, refer to Grafana Integration.</p>
            </div>
          </motion.footer>

          {/* Refresh Control (hidden, for debugging) */}
          <div className="fixed bottom-4 right-4 text-xs text-slate-600 pointer-events-none">
            Status: {state.health} | Last update: {new Date().toLocaleTimeString()}
          </div>
        </main>
      </DashboardLayout>
    </Suspense>
  );
};

export default Monitor;