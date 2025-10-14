// config/production.ts - Production configuration
export const productionConfig = {
  // API Configuration
  api: {
    baseURL: process.env.VITE_API_BASE_URL || 'https://api.ymera.com/v1',
    timeout: 30000,
    retryAttempts: 3,
    retryDelay: 1000,
  },

  // WebSocket Configuration
  websocket: {
    baseURL: process.env.VITE_WS_BASE_URL || 'wss://api.ymera.com/ws',
    reconnectAttempts: 5,
    heartbeatInterval: 30000,
  },

  // Security Configuration
  security: {
    tokenRefreshThreshold: 300000, // 5 minutes
    sessionTimeout: 3600000, // 1 hour
    maxLoginAttempts: 5,
    lockoutDuration: 900000, // 15 minutes
    passwordMinLength: 12,
    requireMFA: true,
  },

  // Performance Configuration
  performance: {
    enableMonitoring: true,
    metricsRetention: 86400000, // 24 hours
    performanceThresholds: {
      loadTime: 3000,
      firstContentfulPaint: 1800,
      largestContentfulPaint: 2500,
      timeToInteractive: 5000,
      apiResponseTime: 1000,
    },
    enableReporting: true,
    reportingInterval: 300000, // 5 minutes
  },

  // Feature Flags
  features: {
    realTimeUpdates: true,
    advancedAnalytics: true,
    experimentalFeatures: false,
    debugMode: false,
  },

  // UI Configuration
  ui: {
    theme: 'dark',
    animations: true,
    reducedMotion: false,
    compactMode: false,
  },

  // Logging Configuration
  logging: {
    level: 'info',
    enableRemoteLogging: true,
    logRetention: 604800000, // 7 days
  }
}

// Environment-specific overrides
export const getConfig = () => {
  const env = process.env.NODE_ENV || 'development'
  
  const baseConfig = { ...productionConfig }
  
  switch (env) {
    case 'development':
      return {
        ...baseConfig,
        api: {
          ...baseConfig.api,
          baseURL: 'http://localhost:8000/api',
        },
        websocket: {
          ...baseConfig.websocket,
          baseURL: 'ws://localhost:8000/ws',
        },
        features: {
          ...baseConfig.features,
          debugMode: true,
        },
        logging: {
          ...baseConfig.logging,
          level: 'debug',
        }
      }
    
    case 'staging':
      return {
        ...baseConfig,
        api: {
          ...baseConfig.api,
          baseURL: 'https://staging-api.ymera.com/v1',
        },
        websocket: {
          ...baseConfig.websocket,
          baseURL: 'wss://staging-api.ymera.com/ws',
        },
        features: {
          ...baseConfig.features,
          experimentalFeatures: true,
        }
      }
    
    default:
      return baseConfig
  }
}

export default getConfig()