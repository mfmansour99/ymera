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
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  // Get WebSocket URL
  const getWebSocketUrl = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const params = new URLSearchParams();
    
    if (user?.id) {
      params.append('userId', user.id);
    }
    
    return `${protocol}//${host}${path}?${params.toString()}`;
  }, [path, user?.id]);

  // Heartbeat logic
  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) return;
    
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WS_STATES.OPEN) {
        wsRef.current.send(JSON.stringify({ 
          type: 'PING', 
          timestamp: Date.now() 
        }));
      }
    }, config.heartbeatInterval);
  }, [config.heartbeatInterval]);

  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  // Connection logic
  const connect = useCallback(() => {
    if (wsRef.current || connectionState === WS_STATES.CONNECTING || !isAuthenticated) {
      return;
    }

    setIsConnecting(true);
    setConnectionState(WS_STATES.CONNECTING);

    try {
      const ws = new WebSocket(getWebSocketUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setIsConnecting(false);
        setConnectionState(WS_STATES.OPEN);
        reconnectAttemptsRef.current = 0;
        
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
        
        startHeartbeat();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);

          // Handle special message types
          if (data.type === 'AUTH_EXPIRED') {
            console.warn('WebSocket: Authentication expired');
            disconnect();
            // The auth hook should handle re-authentication
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected', event.code, event.reason);
        setIsConnected(false);
        setIsConnecting(false);
        setConnectionState(WS_STATES.CLOSED);
        stopHeartbeat();
        
        if (event.code !== 1000) { // Not a normal closure
          reconnect();
        }
      };

      ws.onerror = (error) => {
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
  }, [getWebSocketUrl, isAuthenticated, startHeartbeat, stopHeartbeat]);

  const reconnect = useCallback(() => {
    if (reconnectAttemptsRef.current >= config.reconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    reconnectAttemptsRef.current += 1;
    const currentDelay = Math.min(
      config.reconnectDelay * Math.pow(config.reconnectDecay, reconnectAttemptsRef.current - 1),
      config.maxReconnectDelay
    );

    console.log(`Reconnecting in ${currentDelay}ms (attempt ${reconnectAttemptsRef.current})`);

    reconnectTimeoutRef.current = setTimeout(() => {
      connect();
    }, currentDelay);
  }, [config, connect]);

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
  }, [stopHeartbeat]);

  const sendMessage = useCallback((message: any): boolean => {
    if (wsRef.current?.readyState === WS_STATES.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        return false;
      }
    }
    console.warn('WebSocket is not open. Message not sent.');
    return false;
  }, []);

  // Auto-connect effect
  useEffect(() => {
    if (isAuthenticated) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [isAuthenticated, connect, disconnect]);

  return {
    isConnected,
    isConnecting,
    lastMessage,
    sendMessage,
    disconnect,
    connectionState,
  };
}
