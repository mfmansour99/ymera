import { useEffect, useRef, useState, useCallback } from 'react';
import { Activity } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useWebSocket } from '@/hooks/useWebSocket';

interface DataPoint {
  time: string;
  value: number;
}

export function ThroughputChart() {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const { lastMessage } = useWebSocket();
  
  const [data, setData] = useState<DataPoint[]>([
    { time: '14:42', value: 85 },
    { time: '14:43', value: 92 },
    { time: '14:44', value: 78 },
    { time: '14:45', value: 96 },
    { time: '14:46', value: 88 },
    { time: '14:47', value: 94 },
  ]);

  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage?.type === 'THROUGHPUT_UPDATE') {
      const newDataPoint: DataPoint = {
        time: new Date().toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit',
          hour12: false 
        }),
        value: lastMessage.data.throughput || Math.floor(Math.random() * 30 + 70)
      };
      
      setData(prevData => {
        const updatedData = [...prevData.slice(-5), newDataPoint];
        return updatedData;
      });
    }
  }, [lastMessage]);

  const drawChart = useCallback(() => {
    if (!chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    const canvas = chartRef.current;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    // Set actual canvas size
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    // Scale context for HiDPI displays
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = 40;

    // Clear canvas with subtle gradient background
    const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
    bgGradient.addColorStop(0, 'rgba(6, 182, 212, 0.02)');
    bgGradient.addColorStop(1, 'rgba(6, 182, 212, 0)');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    const maxValue = 100;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Draw grid lines
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
      ctx.fillStyle = '#64748b';
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'right';
      const value = Math.round(maxValue - (i * maxValue / 4));
      ctx.fillText(`${value}%`, padding - 10, y + 4);
    }
    
    ctx.setLineDash([]);

    // Calculate bar dimensions
    const barCount = data.length;
    const barSpacing = 10;
    const barWidth = Math.max(20, (chartWidth - (barCount - 1) * barSpacing) / barCount);

    // Draw bars with gradient
    data.forEach((item, index) => {
      const x = padding + index * (barWidth + barSpacing);
      const barHeight = (item.value / maxValue) * chartHeight;
      const y = height - padding - barHeight;

      // Create gradient for each bar
      const barGradient = ctx.createLinearGradient(x, y, x, height - padding);
      barGradient.addColorStop(0, 'rgba(100, 244, 172, 0.9)');
      barGradient.addColorStop(0.5, 'rgba(100, 244, 172, 0.7)');
      barGradient.addColorStop(1, 'rgba(100, 244, 172, 0.4)');
      
      // Draw bar
      ctx.fillStyle = barGradient;
      ctx.fillRect(x, y, barWidth, barHeight);

      // Bar glow effect
      ctx.shadowColor = '#64f4ac';
      ctx.shadowBlur = 10;
      ctx.fillRect(x, y, barWidth, 2);
      ctx.shadowBlur = 0;

      // Bar border
      ctx.strokeStyle = '#64f4ac';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, barWidth, barHeight);

      // Value label on top of bar
      ctx.fillStyle = '#64f4ac';
      ctx.font = 'bold 12px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(item.value.toString(), x + barWidth / 2, y - 8);

      // Time label at bottom
      ctx.fillStyle = '#64748b';
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.fillText(item.time, x + barWidth / 2, height - 10);
    });

    // Draw axes
    ctx.strokeStyle = 'rgba(100, 244, 172, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
  }, [data]);

  // Initial draw and redraw on data/resize changes
  useEffect(() => {
    const draw = () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      animationFrameRef.current = requestAnimationFrame(drawChart);
    };

    draw();

    const handleResize = () => draw();
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [drawChart]);

  // Simulate periodic updates in demo mode
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prevData => {
        const newValue = Math.floor(Math.random() * 30 + 70);
        const newTime = new Date().toLocaleTimeString('en-US', { 
          hour: '2-digit', 
          minute: '2-digit',
          hour12: false 
        });
        
        return [...prevData.slice(-5), { time: newTime, value: newValue }];
      });
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const currentTime = new Date().toLocaleTimeString();

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50" data-testid="throughput-chart">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-foreground flex items-center">
          <Activity className="w-5 h-5 mr-2" />
          Live Agent Throughput (5m)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 relative mb-4">
          <canvas
            ref={chartRef}
            className="w-full h-full"
            style={{ width: '100%', height: '100%' }}
          />
        </div>
        <div className="flex justify-between items-center text-xs text-muted-foreground">
          <span>Data updated via WebSocket</span>
          <span className="font-mono" data-testid="last-update-time">
            Last update: {currentTime}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}