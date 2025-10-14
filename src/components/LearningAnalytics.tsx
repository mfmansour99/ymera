import { useEffect, useRef } from 'react';
import { Book, Atom, TrendingUp, AlertTriangle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function LearningAnalytics() {
  const chartRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    // Create a simple line chart using Canvas API
    const drawChart = () => {
      const canvas = chartRef.current!;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const width = rect.width;
      const height = rect.height;
      const padding = 40;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Sample data for knowledge growth
      const data = [50, 70, 85, 100, 90, 120, 140];
      const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

      // Calculate points
      const stepX = (width - padding * 2) / (data.length - 1);
      const maxValue = Math.max(...data);
      const points = data.map((value, index) => ({
        x: padding + index * stepX,
        y: height - padding - (value / maxValue) * (height - padding * 2),
      }));

      // Draw grid lines
      ctx.strokeStyle = 'rgba(100, 244, 172, 0.1)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {
        const y = padding + i * ((height - padding * 2) / 4);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
      }

      // Draw area under curve
      ctx.fillStyle = 'rgba(79, 209, 197, 0.1)';
      ctx.beginPath();
      ctx.moveTo(points[0].x, height - padding);
      points.forEach(point => ctx.lineTo(point.x, point.y));
      ctx.lineTo(points[points.length - 1].x, height - padding);
      ctx.closePath();
      ctx.fill();

      // Draw line
      ctx.strokeStyle = '#4fd1c5';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.forEach(point => ctx.lineTo(point.x, point.y));
      ctx.stroke();

      // Draw points
      ctx.fillStyle = '#4fd1c5';
      points.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw labels
      ctx.fillStyle = '#64748b';
      ctx.font = '12px Inter';
      ctx.textAlign = 'center';
      labels.forEach((label, index) => {
        const x = padding + index * stepX;
        ctx.fillText(label, x, height - 10);
      });
    };

    drawChart();

    const handleResize = () => drawChart();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const learningStats = [
    {
      title: 'Active Training Runs',
      value: '5',
      subtitle: 'Agents: 12',
      icon: Atom,
      color: 'text-indigo-400',
    },
    {
      title: 'Knowledge Base Size',
      value: '1.4 GB',
      subtitle: 'Docs: 8.5K',
      icon: TrendingUp,
      color: 'text-teal-400',
    },
    {
      title: 'Training Error Rate (24h)',
      value: '2.1%',
      subtitle: 'Fails: 12',
      icon: AlertTriangle,
      color: 'text-green-400',
    },
  ];

  return (
    <Card className="lg:col-span-2 bg-card/50 backdrop-blur-sm border-border/50" data-testid="learning-analytics">
      <CardHeader>
        <CardTitle className="text-xl font-bold text-primary flex items-center">
          <Book className="w-6 h-6 mr-2" />
          Learning System Analysis
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Learning Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          {learningStats.map((stat) => (
            <div key={stat.title} className="bg-secondary/50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-muted-foreground">{stat.title}</div>
                <stat.icon className={`w-4 h-4 ${stat.color}`} />
              </div>
              <div className={`text-2xl font-bold ${stat.color}`}>
                {stat.value}
              </div>
              <div className="text-xs text-muted-foreground">{stat.subtitle}</div>
            </div>
          ))}
        </div>

        {/* Knowledge Growth Chart */}
        <div className="bg-secondary/30 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-muted-foreground mb-4">
            Knowledge Growth Over Time (MB)
          </h4>
          <div className="h-48 relative">
            <canvas
              ref={chartRef}
              className="w-full h-full"
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
