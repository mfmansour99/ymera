// hooks/useWebSocket.ts - Enterprise WebSocket management
import { useEffect, useRef, useState, useCallback } from 'react'
import { useAuth } from '../store/auth'

// Message types for type safety
interface WebSocketMessage {
  id: string
  type: string
  data: any
  timestamp: number
  ackId?: string
}

interface WebSocketConfig {
  url: string
  protocols?: string[]
  reconnectAttempts?: number
  reconnectInterval?: number
  heartbeatInterval?: number
  timeout?: number
  debug?: boolean
}

interface WebSocketState {
  isConnected: boolean
  isConnecting: boolean
  isReconnecting: boolean
  error: string | null
  lastMessage: WebSocketMessage | null
  connectionAttempts: number
  latency: number
}

interface UseWebSocketReturn {
  // State
  state: WebSocketState
  
  // Actions
  send: (type: string, data: any) => void
  sendWithAck: (type: string, data: any, timeout?: number) => Promise<WebSocketMessage>
  disconnect: () => void
  reconnect: () => void
  
  // Event handlers
  onMessage: (handler: (message: WebSocketMessage) => void) => () => void
  onConnect: (handler: () => void) => () => void
  onDisconnect: (handler: (event: CloseEvent) => void) => () => void
  onError: (handler: (error: Event) => void) => () => void
}

// Default configuration
const DEFAULT_CONFIG: Required<Omit<WebSocketConfig, 'url' | 'protocols'>> & { protocols?: string[] } = {
  protocols: undefined,
  reconnectAttempts: 5,
  reconnectInterval: 1000,
  heartbeatInterval: 30000,
  timeout: 10000,
  debug: false
}

// Message acknowledgment manager
class AckManager {
  private pendingAcks = new Map<string, {
    resolve: (message: WebSocketMessage) => void
    reject: (error: Error) => void
    timeout: NodeJS.Timeout
  }>()

  addPendingAck(
    id: string, 
    resolve: (message: WebSocketMessage) => void, 
    reject: (error: Error) => void,
    timeoutMs: number
  ): void {
    const timeout = setTimeout(() => {
      const pending = this.pendingAcks.get(id)
      if (pending) {
        this.pendingAcks.delete(id)
        reject(new Error(`Message acknowledgment timeout for ${id}`))
      }
    }, timeoutMs)

    this.pendingAcks.set(id, { resolve, reject, timeout })
  }

  resolveAck(ackId: string, message: WebSocketMessage): boolean {
    const pending = this.pendingAcks.get(ackId)
    if (pending) {
      clearTimeout(pending.timeout)
      this.pendingAcks.delete(ackId)
      pending.resolve(message)
      return true
    }
    return false
  }

  rejectAll(error: Error): void {
    for (const [id, pending] of this.pendingAcks) {
      clearTimeout(pending.timeout)
      pending.reject(error)
    }
    this.pendingAcks.clear()
  }

  clear(): void {
    for (const [, pending] of this.pendingAcks) {
      clearTimeout(pending.timeout)
    }
    this.pendingAcks.clear()
  }
}

// Connection quality monitor
class ConnectionQualityMonitor {
  private latencyHistory: number[] = []
  private lastPingTime = 0
  
  startPing(): number {
    this.lastPingTime = Date.now()
    return this.lastPingTime
  }
  
  recordPong(): number {
    const latency = Date.now() - this.lastPingTime
    this.latencyHistory.unshift(latency)
    
    // Keep only last 10 measurements
    if (this.latencyHistory.length > 10) {
      this.latencyHistory = this.latencyHistory.slice(0, 10)
    }
    
    return latency
  }
  
  getAverageLatency(): number {
    if (this.latencyHistory.length === 0) return 0
    return this.latencyHistory.reduce((sum, l) => sum + l, 0) / this.latencyHistory.length
  }
  
  getConnectionQuality(): 'excellent' | 'good' | 'fair' | 'poor' {
    const avgLatency = this.getAverageLatency()
    if (avgLatency < 100) return 'excellent'
    if (avgLatency < 300) return 'good'
    if (avgLatency < 1000) return 'fair'
    return 'poor'
  }
  
  reset(): void {
    this.latencyHistory = []
    this.lastPingTime = 0
  }
}

// Main WebSocket hook
export const useWebSocket = (config: WebSocketConfig): UseWebSocketReturn => {
  const { token, isAuthenticated } = useAuth()
  const wsRef = useRef<WebSocket | null>(null)
  const ackManagerRef = useRef(new AckManager())
  const qualityMonitorRef = useRef(new ConnectionQualityMonitor())
  const configRef = useRef({ ...DEFAULT_CONFIG, ...config })
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const heartbeatIntervalRef = useRef<NodeJS.Timeout>()
  
  // Event handler refs
  const messageHandlersRef = useRef<Set<(message: WebSocketMessage) => void>>(new Set())
  const connectHandlersRef = useRef<Set<() => void>>(new Set())
  const disconnectHandlersRef = useRef<Set<(event: CloseEvent) => void>>(new Set())
  const errorHandlersRef = useRef<Set<(error: Event) => void>>(new Set())

  // State
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    isReconnecting: false,
    error: null,
    lastMessage: null,
    connectionAttempts: 0,
    latency: 0
  })

  // Update config when props change
  useEffect(() => {
    configRef.current = { ...DEFAULT_CONFIG, ...config }
  }, [config])

  // Log function
  const log = useCallback((message: string, data?: any) => {
    if (configRef.current.debug) {
      console.log(`[WebSocket] ${message}`, data)
    }
  }, [])

  // Clear all timeouts
  const clearTimeouts = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
      heartbeatIntervalRef.current = undefined
    }
  }, [])

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    clearTimeouts()
    
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const pingTime = qualityMonitorRef.current.startPing()
        const pingMessage: WebSocketMessage = {
          id: crypto.randomUUID(),
          type: 'ping',
          data: { timestamp: pingTime },
          timestamp: Date.now()
        }
        
        wsRef.current.send(JSON.stringify(pingMessage))
        log('Heartbeat ping sent')
      }
    }, configRef.current.heartbeatInterval)
  }, [clearTimeouts, log])

  // Handle incoming messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      
      log('Message received', message)

      // Handle pong responses for latency calculation
      if (message.type === 'pong') {
        const latency = qualityMonitorRef.current.recordPong()
        setState(prev => ({ ...prev, latency }))
        return
      }

      // Handle acknowledgments
      if (message.ackId && ackManagerRef.current.resolveAck(message.ackId, message)) {
        return
      }

      // Update state with latest message
      setState(prev => ({ ...prev, lastMessage: message, error: null }))

      // Notify all message handlers
      messageHandlersRef.current.forEach(handler => {
        try {
          handler(message)
        } catch (error) {
          console.error('Message handler error:', error)
        }
      })
    } catch (error) {
      log('Failed to parse message', error)
      setState(prev => ({ 
        ...prev, 
        error: 'Invalid message format received' 
      }))
    }
  }, [log])

  // Handle connection open
  const handleOpen = useCallback(() => {
    log('Connection established')
    
    setState(prev => ({
      ...prev,
      isConnected: true,
      isConnecting: false,
      isReconnecting: false,
      error: null,
      connectionAttempts: 0
    }))

    qualityMonitorRef.current.reset()
    startHeartbeat()

    // Notify connect handlers
    connectHandlersRef.current.forEach(handler => {
      try {
        handler()
      } catch (error) {
        console.error('Connect handler error:', error)
      }
    })
  }, [log, startHeartbeat])

  // Handle connection close
  const handleClose = useCallback((event: CloseEvent) => {
    log('Connection closed', { code: event.code, reason: event.reason })
    
    clearTimeouts()
    ackManagerRef.current.rejectAll(new Error('Connection closed'))

    setState(prev => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
      error: event.reason || `Connection closed (${event.code})`
    }))

    // Notify disconnect handlers
    disconnectHandlersRef.current.forEach(handler => {
      try {
        handler(event)
      } catch (error) {
        console.error('Disconnect handler error:', error)
      }
    })

    // Auto-reconnect if not a normal closure
    if (event.code !== 1000 && event.code !== 1001) {
      scheduleReconnect()
    }
  }, [log, clearTimeouts])

  // Handle connection error
  const handleError = useCallback((event: Event) => {
    log('Connection error', event)
    
    setState(prev => ({
      ...prev,
      error: 'Connection error occurred'
    }))

    // Notify error handlers
    errorHandlersRef.current.forEach(handler => {
      try {
        handler(event)
      } catch (error) {
        console.error('Error handler error:', error)
      }
    })
  }, [log])

  // Schedule reconnection attempt
  const scheduleReconnect = useCallback(() => {
    const config = configRef.current
    
    setState(prev => {
      if (prev.connectionAttempts >= config.reconnectAttempts) {
        return {
          ...prev,
          isReconnecting: false,
          error: `Failed to reconnect after ${config.reconnectAttempts} attempts`
        }
      }

      const nextAttempt = prev.connectionAttempts + 1
      const delay = Math.min(
        config.reconnectInterval * Math.pow(2, nextAttempt - 1),
        30000 // Max 30 seconds
      )

      log(`Scheduling reconnect attempt ${nextAttempt} in ${delay}ms`)

      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, delay)

      return {
        ...prev,
        isReconnecting: true,
        connectionAttempts: nextAttempt
      }
    })
  }, [log])

  // Connect to WebSocket
  const connect = useCallback(() => {
    // Don't connect if already connected or connecting
    if (wsRef.current?.readyState === WebSocket.OPEN || 
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return
    }

    // Don't connect if not authenticated
    if (!isAuthenticated() || !token) {
      log('Cannot connect: Not authenticated')
      setState(prev => ({
        ...prev,
        error: 'Authentication required for WebSocket connection'
      }))
      return
    }

    const config = configRef.current
    setState(prev => ({ ...prev, isConnecting: true, error: null }))

    try {
      // Add authentication token to URL
      const url = new URL(config.url)
      url.searchParams.set('token', token)

      log('Connecting to WebSocket', url.toString())

      wsRef.current = new WebSocket(url.toString(), config.protocols)
      
      wsRef.current.onopen = handleOpen
      wsRef.current.onmessage = handleMessage
      wsRef.current.onclose = handleClose
      wsRef.current.onerror = handleError

      // Connection timeout
      setTimeout(() => {
        if (wsRef.current?.readyState === WebSocket.CONNECTING) {
          log('Connection timeout')
          wsRef.current.close()
        }
      }, config.timeout)

    } catch (error) {
      log('Failed to create WebSocket', error)
      setState(prev => ({
        ...prev,
        isConnecting: false,
        error: `Failed to connect: ${error instanceof Error ? error.message : 'Unknown error'}`
      }))
    }
  }, [isAuthenticated, token, handleOpen, handleMessage, handleClose, handleError, log])

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    log('Disconnecting WebSocket')
    
    clearTimeouts()
    ackManagerRef.current.clear()
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }

    setState(prev => ({
      ...prev,
      isConnected: false,
      isConnecting: false,
      isReconnecting: false,
      connectionAttempts: 0
    }))
  }, [log, clearTimeouts])

  // Send message
  const send = useCallback((type: string, data: any) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      log('Cannot send: WebSocket not connected')
      return
    }

    const message: WebSocketMessage = {
      id: crypto.randomUUID(),
      type,
      data,
      timestamp: Date.now()
    }

    try {
      wsRef.current.send(JSON.stringify(message))
      log('Message sent', message)
    } catch (error) {
      log('Failed to send message', error)
    }
  }, [log])

  // Send message with acknowledgment
  const sendWithAck = useCallback((
    type: string, 
    data: any, 
    timeout = 10000
  ): Promise<WebSocketMessage> => {
    return new Promise((resolve, reject) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'))
        return
      }

      const messageId = crypto.randomUUID()
      const message: WebSocketMessage = {
        id: messageId,
        type,
        data,
        timestamp: Date.now(),
        ackId: messageId
      }

      try {
        // Set up acknowledgment handler
        ackManagerRef.current.addPendingAck(messageId, resolve, reject, timeout)
        
        wsRef.current.send(JSON.stringify(message))
        log('Message sent with ack request', message)
      } catch (error) {
        reject(error)
      }
    })
  }, [log])

  // Reconnect manually
  const reconnect = useCallback(() => {
    log('Manual reconnect requested')
    disconnect()
    setTimeout(connect, 100)
  }, [log, disconnect, connect])

  // Event handler registration functions
  const onMessage = useCallback((handler: (message: WebSocketMessage) => void) => {
    messageHandlersRef.current.add(handler)
    return () => messageHandlersRef.current.delete(handler)
  }, [])

  const onConnect = useCallback((handler: () => void) => {
    connectHandlersRef.current.add(handler)
    return () => connectHandlersRef.current.delete(handler)
  }, [])

  const onDisconnect = useCallback((handler: (event: CloseEvent) => void) => {
    disconnectHandlersRef.current.add(handler)
    return () => disconnectHandlersRef.current.delete(handler)
  }, [])

  const onError = useCallback((handler: (error: Event) => void) => {
    errorHandlersRef.current.add(handler)
    return () => errorHandlersRef.current.delete(handler)
  }, [])

  // Auto-connect effect
  useEffect(() => {
    if (isAuthenticated()) {
      connect()
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [isAuthenticated, connect, disconnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearTimeouts()
      ackManagerRef.current.clear()
      disconnect()
    }
  }, [clearTimeouts, disconnect])

  return {
    state,
    send,
    sendWithAck,
    disconnect,
    reconnect,
    onMessage,
    onConnect,
    onDisconnect,
    onError
  }
}

// Higher-order hook for specific message types
export const useWebSocketSubscription = <T = any>(
  config: WebSocketConfig,
  messageType: string,
  handler: (data: T) => void
) => {
  const websocket = useWebSocket(config)

  useEffect(() => {
    return websocket.onMessage((message) => {
      if (message.type === messageType) {
        handler(message.data as T)
      }
    })
  }, [websocket, messageType, handler])

  return websocket
}

// Hook for real-time agent status updates
export const useAgentUpdates = () => {
  const wsConfig = {
    url: `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/ws/agents`,
    debug: import.meta.env.DEV
  }

  return useWebSocket(wsConfig)
}

// Hook for chat/messaging functionality  
export const useChatWebSocket = () => {
  const wsConfig = {
    url: `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/ws/chat`,
    debug: import.meta.env.DEV
  }

  return useWebSocket(wsConfig)
}

export default useWebSocket