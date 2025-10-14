import { WebSocketServer, WebSocket } from 'ws';
import { storage } from '../storage';

interface WebSocketClient {
  ws: WebSocket;
  userId?: string;
  isAlive: boolean;
}

export class WebSocketManager {
  private clients: Set<WebSocketClient> = new Set();
  private wss: WebSocketServer;

  constructor(wss: WebSocketServer) {
    this.wss = wss;
    this.setupWebSocketServer();
    this.setupHeartbeat();
  }

  private setupWebSocketServer(): void {
    this.wss.on('connection', (ws: WebSocket, request) => {
      console.log('WebSocket client connected');
      
      const client: WebSocketClient = {
        ws,
        isAlive: true,
      };
      
      this.clients.add(client);

      // Handle authentication via query parameters
      const url = new URL(request.url!, `http://${request.headers.host}`);
      const userId = url.searchParams.get('userId');
      if (userId) {
        client.userId = userId;
      }

      // Handle incoming messages
      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());
          await this.handleMessage(client, message);
        } catch (error) {
          console.error('Error handling WebSocket message:', error);
          this.sendToClient(client, {
            type: 'error',
            message: 'Invalid message format',
          });
        }
      });

      // Handle client disconnect
      ws.on('close', () => {
        console.log('WebSocket client disconnected');
        this.clients.delete(client);
      });

      // Handle errors
      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.clients.delete(client);
      });

      // Respond to ping messages
      ws.on('pong', () => {
        client.isAlive = true;
      });

      // Send initial connection confirmation
      this.sendToClient(client, {
        type: 'connected',
        message: 'WebSocket connection established',
        timestamp: new Date().toISOString(),
      });
    });
  }

  private setupHeartbeat(): void {
    // Ping clients every 30 seconds to check if they're alive
    setInterval(() => {
      this.clients.forEach((client) => {
        if (!client.isAlive) {
          console.log('Terminating dead WebSocket connection');
          client.ws.terminate();
          this.clients.delete(client);
          return;
        }

        client.isAlive = false;
        if (client.ws.readyState === WebSocket.OPEN) {
          client.ws.ping();
        }
      });
    }, 30000);
  }

  private async handleMessage(client: WebSocketClient, message: any): Promise<void> {
    switch (message.type) {
      case 'PING':
        this.sendToClient(client, {
          type: 'PONG',
          timestamp: new Date().toISOString(),
        });
        break;

      case 'AUTHENTICATE':
        client.userId = message.userId;
        this.sendToClient(client, {
          type: 'AUTHENTICATED',
          userId: client.userId,
        });
        break;

      case 'SUBSCRIBE_METRICS':
        // Client wants to receive real-time metrics
        this.sendToClient(client, {
          type: 'SUBSCRIBED',
          subscription: 'metrics',
        });
        break;

      case 'GET_DASHBOARD_STATS':
        try {
          const stats = await storage.getDashboardStats();
          this.sendToClient(client, {
            type: 'DASHBOARD_STATS',
            data: stats,
            timestamp: new Date().toISOString(),
          });
        } catch (error) {
          this.sendToClient(client, {
            type: 'ERROR',
            message: 'Failed to fetch dashboard stats',
          });
        }
        break;

      default:
        this.sendToClient(client, {
          type: 'ERROR',
          message: `Unknown message type: ${message.type}`,
        });
    }
  }

  private sendToClient(client: WebSocketClient, data: any): void {
    if (client.ws.readyState === WebSocket.OPEN) {
      client.ws.send(JSON.stringify(data));
    }
  }

  public broadcast(data: any): void {
    const message = JSON.stringify(data);
    this.clients.forEach((client) => {
      if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(message);
      }
    });
  }

  public sendToUser(userId: string, data: any): void {
    const message = JSON.stringify(data);
    this.clients.forEach((client) => {
      if (client.userId === userId && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(message);
      }
    });
  }

  public async broadcastMetrics(): Promise<void> {
    try {
      const stats = await storage.getDashboardStats();
      const recentMetrics = await storage.getSystemMetrics(undefined, 10);
      
      this.broadcast({
        type: 'METRICS_UPDATE',
        data: {
          stats,
          recentMetrics,
          timestamp: new Date().toISOString(),
        },
      });
    } catch (error) {
      console.error('Error broadcasting metrics:', error);
    }
  }

  public broadcastTaskUpdate(taskId: string, status: string, data?: any): void {
    this.broadcast({
      type: 'TASK_UPDATE',
      taskId,
      status,
      data,
      timestamp: new Date().toISOString(),
    });
  }

  public broadcastSecurityAlert(alert: any): void {
    this.broadcast({
      type: 'SECURITY_ALERT',
      alert,
      timestamp: new Date().toISOString(),
    });
  }

  public getConnectedClients(): number {
    return this.clients.size;
  }

  public getActiveConnections(): { total: number; authenticated: number } {
    const total = this.clients.size;
    const authenticated = Array.from(this.clients).filter(client => client.userId).length;
    
    return { total, authenticated };
  }
}
