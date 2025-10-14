// hooks/useWebSocket.ts - Fixed Enterprise WebSocket management
import { useEffect, useRef, useState, useCallback } from 'react'

// Simple auth interface for WebSocket
interface AuthContext {
  token?: string
  isAuthenticated: () => boolean
}

// Mock useAuth hook - replace with your actual implementation
const useAuth = (): AuthContext => {
  const token = localStorage.getItem('auth_token')
  return {
    token: token || undefined,
    isAuthenticated: () => !!token
  }
}

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
  state: WebSocketState
  send: (type: string, data: any) => void
  sendWithAck: (type: string, data: any, timeout?: number) => Promise<WebSocketMessage>
  disconnect: () => void
  reconnect: () => void
  onMessage: (handler: (message: WebSocketMessage) => void) => () => void
  onConnect: (handler: () => void) => () => void
  onDisconnect: (handler: (event: CloseEvent) => void) => () => void
  onError: (handler: (error: Event) => void) => () => void
}

const DEFAULT_CONFIG: Required<Omit<WebSocketConfig, 'url' | 'protocols'>> & { protocols?: string[] } = {
  protocols: undefined,
  reconnectAttempts: 5,
  reconnectInterval: 1000,
  heartbeatInterval: 30000,
  timeout: 10000,
  debug: false
}

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
    
    if (this.latencyHistory.length > 10) {
      this.latencyHistory = this.latencyHistory.slice(0, 10)
    }
    
    return latency
  }
  
  getAverageLatency(): number {
    if (this.latencyHistory.length === 0) return 0
    return this.latencyHistory.reduce((sum, l) => sum + l, 0) / this.latencyHistory.length
  }
  
  reset(): void {
    this.latencyHistory = []
    this.lastPingTime = 0
  }
}

// UUID generator fallback
const generateUUID = (): string => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0
    const v = c === 'x' ? r : (r & 0x3 | 0x8)
    return v.toString(16)
  })
}

export const useWebSocket = (config: WebSocketConfig): UseWebSocketReturn => {
  const { token, isAuthenticated } = useAuth()
  const wsRef = useRef<WebSocket | null>(null)
  const ackManagerRef = useRef(new AckManager())
  const qualityMonitorRef = useRef(new ConnectionQualityMonitor())
  const configRef = useRef({ ...DEFAULT_CONFIG, ...config })
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const heartbeatIntervalRef = useRef<NodeJS.Timeout>()
  
  const messageHandlersRef = useRef<Set<(message: WebSocketMessage) => void>>(new Set())
  const connectHandlersRef = useRef<Set<() => void>>(new Set())
  const disconnectHandlersRef = useRef<Set<(event: CloseEvent) => void>>(new Set())
  const errorHandlersRef = useRef<Set<(error: Event) => void>>(new Set())

  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    isReconnecting: false,
    error: null,
    lastMessage: null,
    connectionAttempts: 0,
    latency: 0
  })

  useEffect(() => {
    configRef.current = { ...DEFAULT_CONFIG, ...config }
  }, [config])

  const log = useCallback((message: string, data?: any) => {
    if (configRef.current.debug) {
      console.log(`[WebSocket] ${message}`, data)
    }
  }, [])

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

  const startHeartbeat = useCallback(() => {
    clearTimeouts()
    
    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const pingTime = qualityMonitorRef.current.startPing()
        const pingMessage: WebSocketMessage = {
          id: generateUUID(),
          type: 'ping',
          data: { timestamp: pingTime },
          timestamp: Date.now()
        }
        
        wsRef.current.send(JSON.stringify(pingMessage))
        log('Heartbeat ping sent')
      }
    }, configRef.current.heartbeatInterval)
  }, [clearTimeouts, log])

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      
      log('Message received', message)

      if (message.type === 'pong') {
        const latency = qualityMonitorRef.current.recordPong()
        setState(prev => ({ ...prev, latency }))
        return
      }

      if (message.ackId && ackManagerRef.current.resolveAck(message.ackId, message)) {
        return
      }

      setState(prev => ({ ...prev, lastMessage: message, error: null }))

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

    connectHandlersRef.current.forEach(handler => {
      try {
        handler()
      } catch (error) {
        console.error('Connect handler error:', error)
      }
    })
  }, [log, startHeartbeat])

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

    disconnectHandlersRef.current.forEach(handler => {
      try {
        handler(event)
      } catch (error) {
        console.error('Disconnect handler error:', error)
      }
    })

    if (event.code !== 1000 && event.code !== 1001) {
      scheduleReconnect()
    }
  }, [log, clearTimeouts])

  const handleError = useCallback((event: Event) => {
    log('Connection error', event)
    
    setState(prev => ({
      ...prev,
      error: 'Connection error occurred'
    }))

    errorHandlersRef.current.forEach(handler => {
      try {
        handler(event)
      } catch (error) {
        console.error('Error handler error:', error)
      }
    })
  }, [log])

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
        30000
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

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || 
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return
    }

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
      const url = new URL(config.url)
      url.searchParams.set('token', token)

      log('Connecting to WebSocket', url.toString())

      wsRef.current = new WebSocket(url.toString(), config.protocols)
      
      wsRef.current.onopen = handleOpen
      wsRef.current.onmessage = handleMessage
      wsRef.current.onclose = handleClose
      wsRef.current.onerror = handleError

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

  const send = useCallback((type: string, data: any) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      log('Cannot send: WebSocket not connected')
      return
    }

    const message: WebSocketMessage = {
      id: generateUUID(),
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

      const messageId = generateUUID()
      const message: WebSocketMessage = {
        id: messageId,
        type,
        data,
        timestamp: Date.now(),
        ackId: messageId
      }

      try {
        ackManagerRef.current.addPendingAck(messageId, resolve, reject, timeout)
        
        wsRef.current.send(JSON.stringify(message))
        log('Message sent with ack request', message)
      } catch (error) {
        reject(error)
      }
    })
  }, [log])

  const reconnect = useCallback(() => {
    log('Manual reconnect requested')
    disconnect()
    setTimeout(connect, 100)
  }, [log, disconnect, connect])

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

  useEffect(() => {
    if (isAuthenticated()) {
      connect()
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [isAuthenticated])

  useEffect(() => {
    return () => {
      clearTimeouts()
      ackManagerRef.current.clear()
      disconnect()
    }
  }, [])

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

export const useAgentUpdates = () => {
  const wsConfig = {
    url: `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/ws/agents`,
    debug: import.meta.env.DEV
  }

  return useWebSocket(wsConfig)
}

export const useChatWebSocket = () => {
  const wsConfig = {
    url: `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/ws/chat`,
    debug: import.meta.env.DEV
  }

  return useWebSocket(wsConfig)
}

export default useWebSocket
