// store/performance.ts - Enterprise performance monitoring
import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

// Performance metric types
interface PerformanceMetric {
  id: string
  name: string
  value: number
  timestamp: number
  category: 'network' | 'render' | 'user' | 'system' | 'custom'
  tags?: Record<string, string>
}

interface NavigationTiming {
  navigationStart: number
  domContentLoaded: number
  loadComplete: number
  firstPaint: number
  firstContentfulPaint: number
  largestContentfulPaint: number
  timeToInteractive: number
}

interface ResourceTiming {
  name: string
  startTime: number
  duration: number
  size: number
  type: 'script' | 'stylesheet' | 'image' | 'fetch' | 'xmlhttprequest' | 'other'
}

interface UserInteraction {
  type: 'click' | 'scroll' | 'input' | 'navigation'
  target?: string
  timestamp: number
  duration?: number
}

interface SystemMetrics {
  memoryUsage: number
  jsHeapSize: number
  domNodes: number
  eventListeners: number
  timestamp: number
}

interface PerformanceState {
  // Metrics storage
  metrics: PerformanceMetric[]
  navigationTimings: NavigationTiming | null
  resourceTimings: ResourceTiming[]
  userInteractions: UserInteraction[]
  systemMetrics: SystemMetrics[]
  
  // Performance thresholds
  thresholds: {
    loadTime: number
    firstContentfulPaint: number
    largestContentfulPaint: number
    timeToInteractive: number
    apiResponseTime: number
  }
  
  // Monitoring state
  isMonitoring: boolean
  alertsEnabled: boolean
  reportingInterval: number
  
  // Actions
  startMonitoring: () => void
  stopMonitoring: () => void
  recordMetric: (metric: Omit<PerformanceMetric, 'id' | 'timestamp'>) => void
  recordCustomMetric: (name: string, value: number, tags?: Record<string, string>) => void
  recordUserInteraction: (interaction: Omit<UserInteraction, 'timestamp'>) => void
  updateNavigationTimings: () => void
  collectResourceTimings: () => void
  collectSystemMetrics: () => void
  getMetricsByCategory: (category: PerformanceMetric['category']) => PerformanceMetric[]
  getAverageMetric: (name: string, timeWindow?: number) => number
  getPerformanceReport: () => PerformanceReport
  clearMetrics: () => void
  exportMetrics: () => string
}

interface PerformanceReport {
  summary: {
    totalMetrics: number
    timeRange: { start: number; end: number }
    averageLoadTime: number
    averageApiResponse: number
  }
  vitals: {
    firstContentfulPaint: number | null
    largestContentfulPaint: number | null
    timeToInteractive: number | null
    cumulativeLayoutShift: number | null
  }
  resources: {
    totalRequests: number
    totalSize: number
    slowestResource: ResourceTiming | null
    largestResource: ResourceTiming | null
  }
  userExperience: {
    totalInteractions: number
    averageResponseTime: number
    slowInteractions: UserInteraction[]
  }
  system: {
    averageMemoryUsage: number
    averageHeapSize: number
    peakDomNodes: number
  }
  issues: PerformanceIssue[]
}

interface PerformanceIssue {
  type: 'warning' | 'error' | 'critical'
  category: string
  message: string
  value: number
  threshold: number
  timestamp: number
  suggestions: string[]
}

// Default thresholds based on Core Web Vitals and best practices
const defaultThresholds = {
  loadTime: 3000, // 3 seconds
  firstContentfulPaint: 1800, // 1.8 seconds
  largestContentfulPaint: 2500, // 2.5 seconds
  timeToInteractive: 5000, // 5 seconds
  apiResponseTime: 1000 // 1 second
}

// Performance utilities
class PerformanceUtils {
  static measureMemoryUsage(): number {
    if ('memory' in performance) {
      const memory = (performance as any).memory
      return memory.usedJSHeapSize / 1024 / 1024 // Convert to MB
    }
    return 0
  }

  static countDomNodes(): number {
    return document.querySelectorAll('*').length
  }

  static countEventListeners(): number {
    // Approximate count - not exact due to browser limitations
    const elements = document.querySelectorAll('*')
    let count = 0
    
    // Common event types to check
    const eventTypes = ['click', 'scroll', 'input', 'mouseover', 'keydown']
    
    elements.forEach(element => {
      eventTypes.forEach(type => {
        if (element.hasAttribute(`on${type}`)) count++
      })
    })
    
    return count
  }

  static getResourceType(name: string): ResourceTiming['type'] {
    if (name.includes('.js')) return 'script'
    if (name.includes('.css')) return 'stylesheet'
    if (name.match(/\.(jpg|jpeg|png|gif|svg|webp)$/)) return 'image'
    if (name.includes('api/') || name.includes('fetch')) return 'fetch'
    if (name.includes('xhr')) return 'xmlhttprequest'
    return 'other'
  }

  static calculateCLS(): number {
    // Simplified CLS calculation
    return new Promise(resolve => {
      let cls = 0
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!(entry as any).hadRecentInput) {
            cls += (entry as any).value
          }
        }
        resolve(cls)
      }).observe({ entryTypes: ['layout-shift'] })
      
      // Timeout after 10 seconds
      setTimeout(() => resolve(cls), 10000)
    })
  }
}

export const usePerformance = create<PerformanceState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    metrics: [],
    navigationTimings: null,
    resourceTimings: [],
    userInteractions: [],
    systemMetrics: [],
    thresholds: { ...defaultThresholds },
    isMonitoring: false,
    alertsEnabled: true,
    reportingInterval: 60000, // 1 minute

    // Start monitoring
    startMonitoring: () => {
      const state = get()
      if (state.isMonitoring) return

      set({ isMonitoring: true })

      // Initial measurements
      state.updateNavigationTimings()
      state.collectResourceTimings()
      state.collectSystemMetrics()

      // Set up recurring system metrics collection
      const systemMetricsInterval = setInterval(() => {
        if (!get().isMonitoring) {
          clearInterval(systemMetricsInterval)
          return
        }
        get().collectSystemMetrics()
      }, 30000) // Every 30 seconds

      // Set up performance observer for real-time metrics
      if ('PerformanceObserver' in window) {
        // Observe navigation and paint timings
        try {
          const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
              if (entry.entryType === 'paint') {
                get().recordCustomMetric(entry.name, entry.startTime, {
                  type: 'paint'
                })
              } else if (entry.entryType === 'largest-contentful-paint') {
                get().recordCustomMetric('largest-contentful-paint', entry.startTime, {
                  type: 'vitals'
                })
              }
            }
          })
          
          observer.observe({ entryTypes: ['paint', 'largest-contentful-paint'] })
        } catch (error) {
          console.warn('Performance observer setup failed:', error)
        }
      }

      // Set up user interaction tracking
      const trackInteraction = (type: UserInteraction['type']) => (event: Event) => {
        const startTime = performance.now()
        
        // Use requestIdleCallback to measure interaction response time
        if ('requestIdleCallback' in window) {
          requestIdleCallback(() => {
            const duration = performance.now() - startTime
            get().recordUserInteraction({
              type,
              target: (event.target as Element)?.tagName || 'unknown',
              duration
            })
          })
        } else {
          get().recordUserInteraction({
            type,
            target: (event.target as Element)?.tagName || 'unknown'
          })
        }
      }

      // Add event listeners for user interactions
      document.addEventListener('click', trackInteraction('click'), { passive: true })
      document.addEventListener('scroll', trackInteraction('scroll'), { passive: true })
      document.addEventListener('input', trackInteraction('input'), { passive: true })

      console.log('Performance monitoring started')
    },

    // Stop monitoring
    stopMonitoring: () => {
      set({ isMonitoring: false })
      console.log('Performance monitoring stopped')
    },

    // Record a performance metric
    recordMetric: (metric) => {
      const newMetric: PerformanceMetric = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        ...metric
      }

      set(state => ({
        metrics: [newMetric, ...state.metrics.slice(0, 999)] // Keep last 1000 metrics
      }))

      // Check thresholds and create alerts
      const { thresholds, alertsEnabled } = get()
      if (alertsEnabled) {
        get().checkPerformanceThresholds(newMetric)
      }
    },

    // Record custom metric
    recordCustomMetric: (name, value, tags = {}) => {
      get().recordMetric({
        name,
        value,
        category: 'custom',
        tags
      })
    },

    // Record user interaction
    recordUserInteraction: (interaction) => {
      const newInteraction: UserInteraction = {
        timestamp: Date.now(),
        ...interaction
      }

      set(state => ({
        userInteractions: [newInteraction, ...state.userInteractions.slice(0, 499)] // Keep last 500
      }))
    },

    // Update navigation timings
    updateNavigationTimings: () => {
      if (!('performance' in window) || !performance.timing) return

      const timing = performance.timing
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming

      const timings: NavigationTiming = {
        navigationStart: timing.navigationStart,
        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
        loadComplete: timing.loadEventEnd - timing.navigationStart,
        firstPaint: 0,
        firstContentfulPaint: 0,
        largestContentfulPaint: 0,
        timeToInteractive: 0
      }

      // Get paint timings
      const paintEntries = performance.getEntriesByType('paint')
      for (const entry of paintEntries) {
        if (entry.name === 'first-paint') {
          timings.firstPaint = entry.startTime
        } else if (entry.name === 'first-contentful-paint') {
          timings.firstContentfulPaint = entry.startTime
        }
      }

      // Get LCP
      const lcpEntries = performance.getEntriesByType('largest-contentful-paint')
      if (lcpEntries.length > 0) {
        timings.largestContentfulPaint = lcpEntries[lcpEntries.length - 1].startTime
      }

      set({ navigationTimings: timings })

      // Record as metrics
      Object.entries(timings).forEach(([key, value]) => {
        if (value > 0) {
          get().recordMetric({
            name: key,
            value,
            category: 'render',
            tags: { type: 'navigation' }
          })
        }
      })
    },

    // Collect resource timings
    collectResourceTimings: () => {
      if (!('performance' in window)) return

      const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
      const newResourceTimings: ResourceTiming[] = []

      resources.forEach(resource => {
        const timing: ResourceTiming = {
          name: resource.name,
          startTime: resource.startTime,
          duration: resource.duration,
          size: resource.transferSize || 0,
          type: PerformanceUtils.getResourceType(resource.name)
        }

        newResourceTimings.push(timing)

        // Record as metrics
        get().recordMetric({
          name: 'resource-load-time',
          value: timing.duration,
          category: 'network',
          tags: {
            resource: timing.name,
            type: timing.type
          }
        })
      })

      set(state => ({
        resourceTimings: [...newResourceTimings, ...state.resourceTimings.slice(0, 500)]
      }))
    },

    // Collect system metrics
    collectSystemMetrics: () => {
      const metrics: SystemMetrics = {
        memoryUsage: PerformanceUtils.measureMemoryUsage(),
        jsHeapSize: ('memory' in performance) ? (performance as any).memory.usedJSHeapSize : 0,
        domNodes: PerformanceUtils.countDomNodes(),
        eventListeners: PerformanceUtils.countEventListeners(),
        timestamp: Date.now()
      }

      set(state => ({
        systemMetrics: [metrics, ...state.systemMetrics.slice(0, 99)] // Keep last 100
      }))

      // Record as performance metrics
      Object.entries(metrics).forEach(([key, value]) => {
        if (key !== 'timestamp' && typeof value === 'number') {
          get().recordMetric({
            name: `system-${key}`,
            value,
            category: 'system'
          })
        }
      })
    },

    // Get metrics by category
    getMetricsByCategory: (category) => {
      return get().metrics.filter(metric => metric.category === category)
    },

    // Get average metric value
    getAverageMetric: (name, timeWindow = 300000) => { // Default 5 minutes
      const { metrics } = get()
      const cutoffTime = Date.now() - timeWindow
      
      const relevantMetrics = metrics.filter(
        metric => metric.name === name && metric.timestamp > cutoffTime
      )

      if (relevantMetrics.length === 0) return 0

      const sum = relevantMetrics.reduce((acc, metric) => acc + metric.value, 0)
      return sum / relevantMetrics.length
    },

    // Generate comprehensive performance report
    getPerformanceReport: () => {
      const state = get()
      const { metrics, navigationTimings, resourceTimings, userInteractions, systemMetrics } = state

      const now = Date.now()
      const oneHourAgo = now - 3600000

      const recentMetrics = metrics.filter(m => m.timestamp > oneHourAgo)
      const recentInteractions = userInteractions.filter(i => i.timestamp > oneHourAgo)
      const recentSystemMetrics = systemMetrics.filter(s => s.timestamp > oneHourAgo)

      const report: PerformanceReport = {
        summary: {
          totalMetrics: recentMetrics.length,
          timeRange: { 
            start: recentMetrics.length > 0 ? Math.min(...recentMetrics.map(m => m.timestamp)) : now,
            end: now 
          },
          averageLoadTime: state.getAverageMetric('loadComplete'),
          averageApiResponse: state.getAverageMetric('api-response-time')
        },
        vitals: {
          firstContentfulPaint: navigationTimings?.firstContentfulPaint || null,
          largestContentfulPaint: navigationTimings?.largestContentfulPaint || null,
          timeToInteractive: navigationTimings?.timeToInteractive || null,
          cumulativeLayoutShift: null // Would need separate implementation
        },
        resources: {
          totalRequests: resourceTimings.length,
          totalSize: resourceTimings.reduce((sum, r) => sum + r.size, 0),
          slowestResource: resourceTimings.reduce((slowest, current) => 
            !slowest || current.duration > slowest.duration ? current : slowest, null as ResourceTiming | null
          ),
          largestResource: resourceTimings.reduce((largest, current) => 
            !largest || current.size > largest.size ? current : largest, null as ResourceTiming | null
          )
        },
        userExperience: {
          totalInteractions: recentInteractions.length,
          averageResponseTime: recentInteractions.reduce((sum, i) => sum + (i.duration || 0), 0) / Math.max(recentInteractions.length, 1),
          slowInteractions: recentInteractions.filter(i => (i.duration || 0) > 100).slice(0, 10)
        },
        system: {
          averageMemoryUsage: recentSystemMetrics.reduce((sum, s) => sum + s.memoryUsage, 0) / Math.max(recentSystemMetrics.length, 1),
          averageHeapSize: recentSystemMetrics.reduce((sum, s) => sum + s.jsHeapSize, 0) / Math.max(recentSystemMetrics.length, 1),
          peakDomNodes: Math.max(...recentSystemMetrics.map(s => s.domNodes), 0)
        },
        issues: state.detectPerformanceIssues(recentMetrics)
      }

      return report
    },

    // Clear all metrics
    clearMetrics: () => {
      set({
        metrics: [],
        resourceTimings: [],
        userInteractions: [],
        systemMetrics: []
      })
    },

    // Export metrics as JSON
    exportMetrics: () => {
      const state = get()
      const data = {
        metrics: state.metrics,
        navigationTimings: state.navigationTimings,
        resourceTimings: state.resourceTimings,
        userInteractions: state.userInteractions,
        systemMetrics: state.systemMetrics,
        exportedAt: new Date().toISOString()
      }
      return JSON.stringify(data, null, 2)
    },

    // Private helper methods
    checkPerformanceThresholds: (metric: PerformanceMetric) => {
      const { thresholds } = get()
      let issue: PerformanceIssue | null = null

      switch (metric.name) {
        case 'loadComplete':
          if (metric.value > thresholds.loadTime) {
            issue = {
              type: metric.value > thresholds.loadTime * 1.5 ? 'critical' : 'warning',
              category: 'Load Performance',
              message: `Page load time (${Math.round(metric.value)}ms) exceeds threshold`,
              value: metric.value,
              threshold: thresholds.loadTime,
              timestamp: metric.timestamp,
              suggestions: [
                'Optimize critical resources',
                'Enable resource caching',
                'Minimize JavaScript bundle size',
                'Use code splitting'
              ]
            }
          }
          break

        case 'first-contentful-paint':
          if (metric.value > thresholds.firstContentfulPaint) {
            issue = {
              type: 'warning',
              category: 'Rendering Performance',
              message: `First Contentful Paint (${Math.round(metric.value)}ms) is slow`,
              value: metric.value,
              threshold: thresholds.firstContentfulPaint,
              timestamp: metric.timestamp,
              suggestions: [
                'Optimize critical CSS',
                'Reduce render-blocking resources',
                'Use font-display: swap'
              ]
            }
          }
          break

        case 'api-response-time':
          if (metric.value > thresholds.apiResponseTime) {
            issue = {
              type: 'warning',
              category: 'Network Performance',
              message: `API response time (${Math.round(metric.value)}ms) is slow`,
              value: metric.value,
              threshold: thresholds.apiResponseTime,
              timestamp: metric.timestamp,
              suggestions: [
                'Optimize database queries',
                'Implement response caching',
                'Use CDN for static assets',
                'Consider API pagination'
              ]
            }
          }
          break
      }

      if (issue) {
        console.warn('Performance issue detected:', issue)
        // Could integrate with toast notifications or error reporting service
      }
    },

    detectPerformanceIssues: (metrics: PerformanceMetric[]): PerformanceIssue[] => {
      const issues: PerformanceIssue[] = []
      const { thresholds } = get()

      // Analyze metrics for patterns and issues
      const apiMetrics = metrics.filter(m => m.name === 'api-response-time')
      const loadMetrics = metrics.filter(m => m.name === 'loadComplete')
      const memoryMetrics = metrics.filter(m => m.name === 'system-memoryUsage')

      // Check for consistently slow API responses
      if (apiMetrics.length > 5) {
        const avgApiTime = apiMetrics.reduce((sum, m) => sum + m.value, 0) / apiMetrics.length
        if (avgApiTime > thresholds.apiResponseTime) {
          issues.push({
            type: 'warning',
            category: 'API Performance',
            message: `Average API response time is ${Math.round(avgApiTime)}ms`,
            value: avgApiTime,
            threshold: thresholds.apiResponseTime,
            timestamp: Date.now(),
            suggestions: [
              'Review server performance',
              'Implement request caching',
              'Optimize database queries'
            ]
          })
        }
      }

      // Check for memory leaks
      if (memoryMetrics.length > 10) {
        const memoryTrend = memoryMetrics.slice(-5).map(m => m.value)
        const isIncreasing = memoryTrend.every((val, i) => i === 0 || val >= memoryTrend[i - 1])
        
        if (isIncreasing && memoryTrend[memoryTrend.length - 1] > 100) { // 100MB threshold
          issues.push({
            type: 'critical',
            category: 'Memory Usage',
            message: 'Potential memory leak detected',
            value: memoryTrend[memoryTrend.length - 1],
            threshold: 100,
            timestamp: Date.now(),
            suggestions: [
              'Check for memory leaks in event listeners',
              'Review component cleanup',
              'Monitor large object retention'
            ]
          })
        }
      }

      return issues
    }
  }))
)

// Auto-start monitoring in development
if (import.meta.env.DEV) {
  setTimeout(() => {
    usePerformance.getState().startMonitoring()
  }, 1000)
}

export default usePerformance