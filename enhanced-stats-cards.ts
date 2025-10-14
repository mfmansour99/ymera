import { Activity, Users, Server, AlertTriangle, TrendingUp, TrendingDown } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useMemo } from 'react';

interface StatsCardsProps {
  stats: {
    totalTasks: number;
    completedTasks: number;
    activeAgents: number;
    pendingTasks: number;
    totalUsers: number;
  };
  previousStats?: {
    totalTasks: number;
    completedTasks: number;
    activeAgents: number;
    pendingTasks: number;
    totalUsers: number;
  };
  alerts?: Array<{
    id: string;
    severity: 'critical' | 'warning' | 'info';
    message: string;
  }>;
  wsConnected?: boolean;
}

export function StatsCards({ stats, previousStats, alerts = [], wsConnected = false }: StatsCardsProps) {
  const calculateChange = (current: number, previous?: number): { value: number; isPositive: boolean } => {
    if (!previous || previous === 0) return { value: 0, isPositive: true };
    const change = ((current - previous) / previous) * 100;
    return { value: Math.abs(change), isPositive: change >= 0 };
  };

  const criticalAlerts = alerts.filter(a => a.severity === 'critical').length;
  const totalAlerts = alerts.length;
  const agentLoad = stats.activeAgents > 0 ? Math.min(100, (stats.totalTasks / (stats.activeAgents * 10)) * 100) : 0;

  const statItems = useMemo(() => {
    const tasksChange = calculateChange(stats.completedTasks, previousStats?.completedTasks);
    const agentsChange = calculateChange(stats.activeAgents, previousStats?.activeAgents);
    
    return [
      {
        title: 'Tasks Completed (Total)',
        value: stats.completedTasks.toLocaleString(),
        icon: Activity,
        color: 'text-primary',
        bgColor: 'bg-primary/10',
        borderColor: 'border-primary/20',
        change: tasksChange.value > 0 ? `${tasksChange.isPositive ? '+' : '-'}${tasksChange.value.toFixed(1)}% from last week` : 'No change',
        changeIcon: tasksChange.isPositive ? TrendingUp : TrendingDown,
        changeColor: tasksChange.isPositive ? 'text-green-400' : 'text-red-400',
        testId: 'stat-tasks-completed'
      },
      {
        title: 'Agents Registered',
        value: stats.activeAgents.toString(),
        icon: Users,
        color: 'text-accent',
        bgColor: 'bg-accent/10',
        borderColor: 'border-accent/20',
        change: `Load: ${agentLoad.toFixed(0)}%`,
        changeColor: agentLoad > 80 ? 'text-yellow-400' : 'text-muted-foreground',
        testId: 'stat-agents-registered'
      },
      {
        title: 'Tasks Queued (Current)',
        value: stats.pendingTasks.toString(),
        icon: Server,
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-400/10',
        borderColor: 'border-yellow-400/20',
        change: `WS: ${wsConnected ? 'Connected' : 'Disconnected'}`,
        changeColor: wsConnected ? 'text-green-400' : 'text-red-400',
        testId: 'stat-tasks-queued'
      },
      {
        title: 'Infrastructure Alerts',
        value: totalAlerts.toString(),
        icon: AlertTriangle,
        color: criticalAlerts > 0 ? 'text-red-400' : totalAlerts > 0 ? 'text-yellow-400' : 'text-green-400',
        bgColor: criticalAlerts > 0 ? 'bg-red-400/10' : totalAlerts > 0 ? 'bg-yellow-400/10' : 'bg-green-400/10',
        borderColor: criticalAlerts > 0 ? 'border-red-400/20' : totalAlerts > 0 ? 'border-yellow-400/20' : 'border-green-400/20',
        change: criticalAlerts > 0 ? `${criticalAlerts} Critical Alert${criticalAlerts > 1 ? 's' : ''}` : totalAlerts > 0 ? 'No critical alerts' : 'All systems operational',
        changeColor: criticalAlerts > 0 ? 'text-red-400' : totalAlerts > 0 ? 'text-yellow-400' : 'text-green-400',
        testId: 'stat-infrastructure-alerts'
      },
    ];
  }, [stats, previousStats, agentLoad, criticalAlerts, totalAlerts, wsConnected]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {statItems.map((item) => (
        <Card 
          key={item.title} 
          className={`
            bg-card/50 backdrop-blur-sm border-border/50 
            hover:bg-card/70 transition-all duration-300 
            hover:scale-105 hover:shadow-lg hover:shadow-primary/20
            ${item.borderColor}
          `}
          data-testid={item.testId}
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {item.title}
            </CardTitle>
            <div className={`p-2 rounded-lg ${item.bgColor}`}>
              <item.icon className={`w-5 h-5 ${item.color}`} />
            </div>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${item.color} glow-text mb-2`}>
              {item.value}
            </div>
            <div className={`text-xs flex items-center gap-1 ${item.changeColor || 'text-muted-foreground'}`}>
              {item.changeIcon && <item.changeIcon className="w-3 h-3" />}
              <span>{item.change}</span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}