import { useEffect, useState } from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { StatsCards } from '@/components/monitoring/StatsCards';
import { LearningAnalytics } from '@/components/monitoring/LearningAnalytics';
import { ThroughputChart } from '@/components/monitoring/ThroughputChart';
import { SecurityAuditFeed } from '@/components/monitoring/SecurityAuditFeed';
import { SystemMetrics } from '@/components/monitoring/SystemMetrics';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '@/hooks/useAuth';
import { useToast } from '@/hooks/use-toast';
import { isUnauthorizedError } from '@/lib/authUtils';

interface DashboardStats {
  totalTasks: number;
  completedTasks: number;
  activeAgents: number;
  pendingTasks: number;
  totalUsers: number;
}

export function Dashboard() {
  const { isAuthenticated, isLoading } = useAuth();
  const { toast } = useToast();
  const [systemHealth, setSystemHealth] = useState<'HEALTHY' | 'WARNING' | 'CRITICAL'>('HEALTHY');

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      toast({
        title: "Unauthorized",
        description: "You are logged out. Logging in again...",
        variant: "destructive",
      });
      setTimeout(() => {
        window.location.href = "/api/login";
      }, 500);
      return;
    }
  }, [isAuthenticated, isLoading, toast]);

  const { data: stats, isLoading: statsLoading, error } = useQuery<DashboardStats>({
    queryKey: ['/api/dashboard/stats'],
    enabled: isAuthenticated,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Handle unauthorized errors
  useEffect(() => {
    if (error && isUnauthorizedError(error as Error)) {
      toast({
        title: "Unauthorized",
        description: "Session expired. Logging in again...",
        variant: "destructive",
      });
      setTimeout(() => {
        window.location.href = "/api/login";
      }, 500);
    }
  }, [error, toast]);

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="text-lg text-muted-foreground">Loading dashboard...</div>
        </div>
      </DashboardLayout>
    );
  }

  if (!isAuthenticated) {
    return null; // Will redirect to login
  }

  const defaultStats: DashboardStats = {
    totalTasks: 0,
    completedTasks: 0,
    activeAgents: 0,
    pendingTasks: 0,
    totalUsers: 0,
  };

  const dashboardStats = stats || defaultStats;

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <header className="mb-8">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center">
            <div>
              <h1 className="text-4xl font-bold text-foreground mb-2">
                <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Ymera Dashboard
                </span>
              </h1>
              <p className="text-muted-foreground text-lg">
                AI agent orchestration and workflow management.
              </p>
            </div>
            <div className="mt-4 sm:mt-0 flex items-center space-x-4">
              <div className="text-sm font-medium text-muted-foreground">
                System Health:
              </div>
              <div className="flex items-center space-x-2">
                <div 
                  className={`w-3 h-3 rounded-full ${
                    systemHealth === 'HEALTHY' ? 'bg-green-400 animate-pulse' : 
                    systemHealth === 'WARNING' ? 'bg-yellow-400' : 'bg-red-400'
                  }`}
                />
                <span className={`text-sm font-bold ${
                  systemHealth === 'HEALTHY' ? 'text-green-400' : 
                  systemHealth === 'WARNING' ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {systemHealth}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Core Stats Grid */}
        <section className="mb-8">
          {statsLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="bg-card/50 rounded-lg p-6 animate-pulse">
                  <div className="h-4 bg-muted rounded mb-4"></div>
                  <div className="h-8 bg-muted rounded mb-2"></div>
                  <div className="h-3 bg-muted rounded"></div>
                </div>
              ))}
            </div>
          ) : (
            <StatsCards stats={dashboardStats} />
          )}
        </section>

        {/* Main Content Grid */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <LearningAnalytics />
          <ThroughputChart />
        </section>

        {/* Bottom Grid */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <SystemMetrics />
          <SecurityAuditFeed />
        </section>
      </div>
    </DashboardLayout>
  );
}
