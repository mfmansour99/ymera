import { useEffect, useRef, useState, useCallback } from 'react';
import { Activity } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useWebSocket } from '@/hooks/useWebSocket';

interface DataPoint {
  value: number;
  label: string;
  timestamp: number;
}

export function ThroughputChart() {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const { lastMessage, isConnected } = useWebSocket();
  const [data, setData] = useState<DataPoint[]>([]);
  const [isAnimating, setIsAnimating] = useState(false);

  // Initialize with sample data
  useEffect(() => {
    const now = Date.now();
    const initialData: DataPoint[] = Array.from({ length: 6 }, (_, i) => ({
      value: 75 + Math.random() * 20,
      label: new Date(now - (5 - i) * 60000).toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      }),
      timestamp: now - (5 - i) * 60000
    }));
    setData(initialData);
  }, []);

  // Update data from WebSocket messages
  useEffect(() => {
    if (lastMessage?.type === 'throughput_update' && lastMessage.data) {
      setData(prev => {
        const updated = [...prev.slice(1), {
          value: lastMessage.data.value,
          label: new Date().toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
          }),
          timestamp: Date.now()
        }];
        return updated;
      });
    }
  }, [lastMessage]);

  const drawChart = useCallback(() => {
    const canvas = chartRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // High DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Clear canvas with fade effect
    ctx.fillStyle = 'rgba(10, 10, 10, 0.02)';
    ctx.fillRect(0, 0, width, height);

    // Ensure we have data
    if (data.length === 0) return;

    const maxValue = 100;
    const barWidth = chartWidth / data.length - 10;
    const barSpacing = 10;

    // Animated grid lines
    ctx.strokeStyle = 'rgba(100, 244, 172, 0.1)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    
    for (let i = 0; i <= 4; i++) {
      const y = padding + i * (chartHeight / 4);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = 'rgba(100, 116, 139, 0.8)';
      ctx.font = '11px Inter, system-ui';
      ctx.textAlign = 'right';
      const value = maxValue - (i * 25);
      ctx.fillText(`${value}%`, padding - 10, y + 4);
    }
    ctx.setLineDash([]);

    // Draw bars with animation
    data.forEach((point, index) => {
      const x = padding + index * (barWidth + barSpacing);
      const barHeight = (point.value / maxValue) * chartHeight;
      const y = height - padding - barHeight;

      // Create gradient for each bar
      const gradient = ctx.createLinearGradient(x, y, x, height - padding);
      gradient.addColorStop(0, 'rgba(100, 244, 172, 0.9)');
      gradient.addColorStop(0.5, 'rgba(100, 244, 172, 0.7)');
      gradient.addColorStop(1, 'rgba(100, 244, 172, 0.3)');

      // Draw bar with rounded top
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, [4, 4, 0, 0]);
      ctx.fill();

      // Bar glow effect
      ctx.shadowColor = '#64f4ac';
      ctx.shadowBlur = 10;
      ctx.strokeStyle = '#64f4ac';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Value label on top of bar with animation
      if (isAnimating) {
        ctx.fillStyle = '#64f4ac';
        ctx.font = 'bold 12px Inter, system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(point.value).toString(), x + barWidth / 2, y - 8);
      }

      // Time label at bottom
      ctx.fillStyle = '#64748b';
      ctx.font = '11px Inter, system-ui';
      ctx.textAlign = 'center';
      ctx.fillText(point.label, x + barWidth / 2, height - 10);
    });

    // Connection status indicator
    const statusColor = isConnected ? '#10b981' : '#ef4444';
    ctx.fillStyle = statusColor;
    ctx.beginPath();
    ctx.arc(width - 20, 20, 5, 0, Math.PI * 2);
    ctx.fill();

    // Animate
    if (isAnimating) {
      animationRef.current = requestAnimationFrame(drawChart);
    }
  }, [data, isConnected, isAnimating]);

  useEffect(() => {
    drawChart();
    setIsAnimating(true);

    const handleResize = () => {
      drawChart();
    };

    window.addEventListener('resize', handleResize);
    
    // Set up periodic updates if WebSocket is not connected
    const interval = !isConnected ? setInterval(() => {
      setData(prev => {
        const newData = [...prev.slice(1)];
        newData.push({
          value: 75 + Math.random() * 20,
          label: new Date().toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
          }),
          timestamp: Date.now()
        });
        return newData;
      });
    }, 5000) : null;

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (interval) {
        clearInterval(interval);
      }
      setIsAnimating(false);
    };
  }, [drawChart, isConnected]);

  const currentTime = new Date().toLocaleTimeString();
  const averageThroughput = data.reduce((sum, d) => sum + d.value, 0) / data.length;

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50 overflow-hidden" data-testid="throughput-chart">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-foreground flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Live Agent Throughput (5m)
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">
              Avg: {averageThroughput.toFixed(1)}%
            </span>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 relative mb-4">
          <canvas
            ref={chartRef}
            className="w-full h-full"
            style={{ width: '100%', height: '100%' }}
            aria-label="Throughput chart visualization"
          />
        </div>
        <div className="text-xs text-muted-foreground text-center flex items-center justify-center gap-2">
          <span>
            {isConnected ? 'Live data via WebSocket' : 'Simulated data (WebSocket disconnected)'}
          </span>
          <span className="text-muted-foreground/50">•</span>
          <span className="font-mono" data-testid="last-update-time">
            {currentTime}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}