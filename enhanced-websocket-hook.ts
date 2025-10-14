import { useEffect, useRef, useState, useCallback } from 'react';
import { useAuth } from './useAuth';

interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
  [key: string]: any;
}

interface WebSocketHookReturn {
  isConnected: boolean;
  isConnecting: boolean;
  lastMessage: WebSocketMessage | null;
  sendMessage: (message: any) => boolean;
  disconnect: () => void;
  connectionState: number;
  reconnectAttempts: number;
}

const WS_STATES = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
} as const;

const DEFAULT_OPTIONS = {
  reconnectAttempts: 10,
  reconnectDelay: 1000,
  maxReconnectDelay: 16000,
  reconnectDecay: 2,
  heartbeatInterval: 25000,
  messageQueueSize: 100,
};

export function useWebSocket(
  path: string = '/ws',
  options: Partial<typeof DEFAULT_OPTIONS> = {}
): WebSocketHookReturn {
  const config = { ...DEFAULT_OPTIONS, ...options };
  const { user, isAuthenticated } = useAuth();
  
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [connectionState, setConnectionState] = useState(WS_STATES.CLOSED);
  const [reconnectCount, setReconnectCount] = useState(0);
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const messageQueueRef = useRef<any[]>([]);
  const isUnmountingRef = useRef(false);

  // Get WebSocket URL with proper protocol detection
  const getWebSocketUrl = useCallback(() => {
    const isProduction = import.meta.env.PROD;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = isProduction 
      ? window.location.host 
      : import.meta.env.VITE_WS_HOST || window.location.host;
    
    const params = new URLSearchParams();
    if (user?.id) {
      params.append('userId', user.id);
    }
    
    return `${protocol}//${host}${path}?${params.toString()}`;
  }, [path, user?.id]);

  // Heartbeat logic with error handling
  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WS_STATES.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ 
            type: 'PING', 
            timestamp: Date.now() 
          }));
        } catch (error) {
          console.error('Heartbeat send failed:', error);
        }
      }
    }, config.heartbeatInterval);
  }, [config.heartbeatInterval]);

  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  // Process queued messages
  const processMessageQueue = useCallback(() => {
    while (messageQueueRef.current.length > 0 && wsRef.current?.readyState === WS_STATES.OPEN) {
      const message = messageQueueRef.current.shift();
      try {
        wsRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send queued message:', error);
        messageQueueRef.current.unshift(message); // Re-queue on failure
        break;
      }
    }
  }, []);

  // Connection logic with enhanced error handling
  const connect = useCallback(() => {
    if (isUnmountingRef.current || wsRef.current || connectionState === WS_STATES.CONNECTING || !isAuthenticated) {
      return;
    }

    setIsConnecting(true);
    setConnectionState(WS_STATES.CONNECTING);

    try {
      const ws = new WebSocket(getWebSocketUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        if (isUnmountingRef.current) return;
        
        console.log('WebSocket connected successfully');
        setIsConnected(true);
        setIsConnecting(false);
        setConnectionState(WS_STATES.OPEN);
        setReconnectCount(0);
        
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
        
        startHeartbeat();
        processMessageQueue();
      };

      ws.onmessage = (event) => {
        if (isUnmountingRef.current) return;
        
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);

          // Handle special message types
          if (data.type === 'AUTH_EXPIRED') {
            console.warn('WebSocket: Authentication expired');
            disconnect();
          } else if (data.type === 'PONG') {
            // Heartbeat response received
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        if (isUnmountingRef.current) return;
        
        console.log('WebSocket disconnected', event.code, event.reason);
        setIsConnected(false);
        setIsConnecting(false);
        setConnectionState(WS_STATES.CLOSED);
        stopHeartbeat();
        
        // Only attempt reconnect for abnormal closures
        if (event.code !== 1000 && event.code !== 1001) {
          reconnect();
        }
      };

      ws.onerror = (error) => {
        if (isUnmountingRef.current) return;
        
        console.error('WebSocket error:', error);
        setIsConnecting(false);
        setConnectionState(WS_STATES.CLOSED);
        ws.close();
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
      setIsConnecting(false);
      setConnectionState(WS_STATES.CLOSED);
      reconnect();
    }
  }, [getWebSocketUrl, isAuthenticated, connectionState, startHeartbeat, stopHeartbeat, processMessageQueue]);

  const reconnect = useCallback(() => {
    if (isUnmountingRef.current || reconnectCount >= config.reconnectAttempts) {
      console.error('Max reconnection attempts reached or component unmounting');
      return;
    }

    const newReconnectCount = reconnectCount + 1;
    setReconnectCount(newReconnectCount);
    
    const currentDelay = Math.min(
      config.reconnectDelay * Math.pow(config.reconnectDecay, newReconnectCount - 1),
      config.maxReconnectDelay
    );

    console.log(`Reconnecting in ${currentDelay}ms (attempt ${newReconnectCount}/${config.reconnectAttempts})`);

    reconnectTimeoutRef.current = setTimeout(() => {
      if (!isUnmountingRef.current) {
        connect();
      }
    }, currentDelay);
  }, [config, connect, reconnectCount]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    stopHeartbeat();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setIsConnecting(false);
    setConnectionState(WS_STATES.CLOSED);
    setReconnectCount(0);
    messageQueueRef.current = [];
  }, [stopHeartbeat]);

  const sendMessage = useCallback((message: any): boolean => {
    if (wsRef.current?.readyState === WS_STATES.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        
        // Queue message for retry if it's important
        if (messageQueueRef.current.length < config.messageQueueSize) {
          messageQueueRef.current.push(message);
        }
        return false;
      }
    } else {
      // Queue message if not connected
      if (messageQueueRef.current.length < config.messageQueueSize) {
        messageQueueRef.current.push(message);
        console.log('Message queued for sending when connection is established');
      }
      return false;
    }
  }, [config.messageQueueSize]);

  // Auto-connect effect
  useEffect(() => {
    isUnmountingRef.current = false;
    
    if (isAuthenticated) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      isUnmountingRef.current = true;
      disconnect();
    };
  }, [isAuthenticated]); // Removed connect and disconnect from deps to avoid loops

  // Handle page visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopHeartbeat();
      } else if (isConnected) {
        startHeartbeat();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isConnected, startHeartbeat, stopHeartbeat]);

  return {
    isConnected,
    isConnecting,
    lastMessage,
    sendMessage,
    disconnect,
    connectionState,
    reconnectAttempts: reconnectCount,
  };
}