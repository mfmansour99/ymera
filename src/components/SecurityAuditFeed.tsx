import { useEffect, useState } from 'react';
import { ShieldAlert } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useWebSocket } from '@/hooks/useWebSocket';

interface AuditEntry {
  timestamp: string;
  level: 'CRITICAL' | 'HIGH' | 'WARN' | 'INFO';
  message: string;
  details?: string;
}

export function SecurityAuditFeed() {
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([
    {
      timestamp: '15:45',
      level: 'CRITICAL',
      message: 'DB_CONNECT_FAIL - POSTGRES',
    },
    {
      timestamp: '15:44',
      level: 'HIGH',
      message: 'AGENT_AUTHZ_FAIL - User: 1002',
    },
    {
      timestamp: '15:43',
      level: 'INFO',
      message: 'SW_CACHE_UPDATE - Service Worker',
    },
    {
      timestamp: '15:40',
      level: 'CRITICAL',
      message: 'XSS_ATTEMPT - Origin: 203.0.113.4',
    },
    {
      timestamp: '15:39',
      level: 'INFO',
      message: 'SESSION_REFRESH - User: admin@ymera.ai',
    },
    {
      timestamp: '15:38',
      level: 'WARN',
      message: 'RATE_LIMIT_EXCEEDED - API: /agents',
    },
    {
      timestamp: '15:37',
      level: 'INFO',
      message: 'BACKUP_COMPLETED - Size: 2.3GB',
    },
  ]);

  const { lastMessage } = useWebSocket();

  useEffect(() => {
    // Listen for security alerts from WebSocket
    if (lastMessage?.type === 'SECURITY_ALERT') {
      const newEntry: AuditEntry = {
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour12: false, 
          hour: '2-digit', 
          minute: '2-digit' 
        }),
        level: lastMessage.alert.level || 'INFO',
        message: lastMessage.alert.message,
        details: lastMessage.alert.details,
      };

      setAuditEntries(prev => [newEntry, ...prev.slice(0, 9)]); // Keep last 10 entries
    }
  }, [lastMessage]);

  const getLevelColor = (level: AuditEntry['level']) => {
    switch (level) {
      case 'CRITICAL':
        return 'text-red-400';
      case 'HIGH':
        return 'text-yellow-400';
      case 'WARN':
        return 'text-yellow-400';
      case 'INFO':
        return 'text-green-400';
      default:
        return 'text-blue-400';
    }
  };

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50" data-testid="security-audit-feed">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-red-400 flex items-center">
          <ShieldAlert className="w-5 h-5 mr-2" />
          Security Audit Feed
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-48">
          <div className="font-mono text-xs space-y-1">
            {auditEntries.map((entry, index) => (
              <div
                key={`${entry.timestamp}-${index}`}
                className={`${getLevelColor(entry.level)} opacity-80 hover:opacity-100 transition-opacity`}
                data-testid={`audit-entry-${entry.level.toLowerCase()}`}
              >
                [{entry.timestamp}] {entry.level}: {entry.message}
                {entry.details && (
                  <div className="text-muted-foreground pl-4 text-xs">
                    {entry.details}
                  </div>
                )}
              </div>
            ))}
          </div>
        </ScrollArea>
        <div className="mt-2 text-xs text-muted-foreground">
          Real-time security events • Auto-refresh enabled
        </div>
      </CardContent>
    </Card>
  );
}
