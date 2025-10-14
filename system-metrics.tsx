import { useEffect, useState } from 'react';
import { Cpu, Globe } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useWebSocket } from '@/hooks/useWebSocket';

interface SystemMetricsProps {
  className?: string;
}

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  color: string;
}

export function SystemMetrics({ className }: SystemMetricsProps) {
  const [cpuUsage, setCpuUsage] = useState(72);
  const [memoryUsage, setMemoryUsage] = useState(45);
  const { lastMessage } = useWebSocket();

  const performanceMetrics: PerformanceMetric[] = [
    { name: 'Disk I/O', value: 1.2, unit: 'GB/s', color: 'text-green-400' },
    { name: 'Network Throughput', value: 847, unit: 'Mbps', color: 'text-primary' },
    { name: 'Active Connections', value: 1247, unit: '', color: 'text-accent' },
    { name: 'Queue Depth', value: 12, unit: '', color: 'text-yellow-400' },
  ];

  const edgeStats = [
    { name: 'Active Edge Nodes', value: '3', color: 'text-green-400' },
    { name: 'Optimal User Node', value: 'US-East-1', color: 'text-accent' },
    { name: 'Core Web Vitals (LCP)', value: '1.2s', color: 'text-green-400' },
    { name: 'Average Response Time', value: '89ms', color: 'text-primary' },
  ];

  // Handle WebSocket messages for real-time updates
  useEffect(() => {
    if (lastMessage?.type === 'METRICS_UPDATE') {
      const metrics = lastMessage.data;
      if (metrics?.cpu) setCpuUsage(metrics.cpu);
      if (metrics?.memory) setMemoryUsage(metrics.memory);
    }
  }, [lastMessage]);

  // Simulate real-time updates with more realistic patterns
  useEffect(() => {
    const interval = setInterval(() => {
      setCpuUsage(prev => {
        const change = (Math.random() - 0.5) * 10;
        const newValue = prev + change;
        return Math.max(10, Math.min(95, newValue));
      });
      
      setMemoryUsage(prev => {
        const change = (Math.random() - 0.5) * 8;
        const newValue = prev + change;
        return Math.max(10, Math.min(90, newValue));
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const CircularProgress = ({ value, label, color }: { value: number; label: string; color: string }) => {
    const circumference = 2 * Math.PI * 45;
    const strokeDashoffset = circumference - (value / 100) * circumference;

    return (
      <div className="text-center">
        <div className="relative w-24 h-24 mx-auto mb-2">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              className="text-muted opacity-20"
            />
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className={color}
              style={{ transition: 'stroke-dashoffset 0.5s ease-in-out' }}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`text-lg font-bold ${color}`}>{Math.round(value)}%</span>
          </div>
        </div>
        <div className="text-sm font-medium text-muted-foreground">{label}</div>
      </div>
    );
  };

  return (
    <div className={`space-y-6 ${className || ''}`}>
      {/* Edge Computing Status */}
      <Card className="bg-card/50 backdrop-blur-sm border-border/50" data-testid="edge-computing-status">
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-foreground flex items-center">
            <Globe className="w-5 h-5 mr-2" />
            Edge Computing Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {edgeStats.map((stat) => (
              <div key={stat.name} className="flex justify-between items-center">
                <span className="text-muted-foreground text-sm">{stat.name}:</span>
                <span className={`font-bold ${stat.color}`}>{stat.value}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* System Performance Metrics */}
      <Card className="bg-card/50 backdrop-blur-sm border-border/50" data-testid="system-performance">
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-foreground flex items-center">
            <Cpu className="w-5 h-5 mr-2" />
            System Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Performance Gauges */}
          <div className="grid grid-cols-2 gap-6 mb-6">
            <CircularProgress value={cpuUsage} label="CPU Usage" color="text-primary" />
            <CircularProgress value={memoryUsage} label="Memory Usage" color="text-accent" />
          </div>

          {/* Additional Metrics */}
          <div className="space-y-3">
            {performanceMetrics.map((metric) => (
              <div key={metric.name} className="flex justify-between items-center">
                <span className="text-muted-foreground text-sm">{metric.name}:</span>
                <span className={`font-bold ${metric.color}`}>
                  {metric.value.toLocaleString()}{metric.unit && ` ${metric.unit}`}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}