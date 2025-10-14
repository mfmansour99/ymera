import { Activity, Users, Server, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface StatsCardsProps {
  stats: {
    totalTasks: number;
    completedTasks: number;
    activeAgents: number;
    pendingTasks: number;
    totalUsers: number;
  };
}

export function StatsCards({ stats }: StatsCardsProps) {
  const statItems = [
    {
      title: 'Tasks Completed (Total)',
      value: stats.completedTasks.toLocaleString(),
      icon: Activity,
      color: 'text-primary',
      change: '+23% from last week',
      testId: 'stat-tasks-completed'
    },
    {
      title: 'Agents Registered',
      value: stats.activeAgents.toString(),
      icon: Users,
      color: 'text-accent',
      change: 'Load: 78%',
      testId: 'stat-agents-registered'
    },
    {
      title: 'Tasks Queued (Current)',
      value: stats.pendingTasks.toString(),
      icon: Server,
      color: 'text-yellow-400',
      change: 'WS: Connected',
      testId: 'stat-tasks-queued'
    },
    {
      title: 'Infrastructure Alerts',
      value: '3', // This would come from a separate alerts system
      icon: AlertTriangle,
      color: 'text-red-400',
      change: '2 Critical Alerts',
      changeColor: 'text-red-400',
      testId: 'stat-infrastructure-alerts'
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {statItems.map((item) => (
        <Card 
          key={item.title} 
          className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/70 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-primary/20"
          data-testid={item.testId}
        >
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {item.title}
            </CardTitle>
            <item.icon className={`w-5 h-5 ${item.color}`} />
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${item.color} glow-text`}>
              {item.value}
            </div>
            <div className={`text-xs mt-2 ${item.changeColor || 'text-muted-foreground'}`}>
              {item.change}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
