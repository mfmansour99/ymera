// src/utils/performance-monitoring.ts
import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import { apiClient } from '../api/secure-client'

// Performance Metrics Interfaces
export interface PerformanceMetrics {
  // Core Web Vitals
  lcp?: number // Largest Contentful Paint
  fid?: number // First Input Delay  
  cls?: number // Cumulative Layout Shift
  fcp?: number // First Contentful Paint
  ttfb?: number // Time to First Byte
  
  // Custom Metrics
  timeToInteractive?: number
  totalBlockingTime?: number
  speedIndex?: number
  
  // Resource Metrics
  domContentLoaded?: number
  loadComplete?: number
  resourceCount?: number
  totalResourceSize?: number
  
  // JavaScript Metrics
  jsHeapUsed?: number
  jsHeapTotal?: number
  jsHeapLimit?: number
  
  // Network Metrics
  connectionType?: string
  effectiveType?: string
  downlink?: number
  rtt?: number
  saveData?: boolean
}

export interface UserInteractionMetric {
  id: string
  type: 'click' | 'scroll' | 'input' | 'navigation' | 'api_call'
  timestamp: number
  duration?: number
  target?: string
  metadata?: Record<string, any>
}

export interface ComponentPerformanceMetric {
  componentName: string
  renderTime: number
  mountTime?: number
  updateCount: number
  propsSize?: number
  stateSize?: number
  lastUpdate: number
}

export interface ApiPerformanceMetric {
  endpoint: string
  method: string
  duration: number
  status: number
  timestamp: number
  size?: number
  cached?: boolean
  retries?: number
}

export interface PerformanceAlert {
  id: string
  type: 'warning' | 'critical'
  metric: string
  threshold: number
  actual: number
  timestamp: number
  resolved: boolean
}

// Performance Store State
interface PerformanceState {
  // Metrics
  coreMetrics: PerformanceMetrics
  interactions: UserInteractionMetric[]
  componentMetrics: Map<string, ComponentPerformanceMetric>
  apiMetrics: ApiPerformanceMetric[]
  alerts: PerformanceAlert[]
  
  // Configuration
  isMonitoring: boolean
  sampleRate: number
  alertThresholds: Record<string, number>
  
  // Session Info
  sessionStart: number
  pageViews: number
  uniqueComponents: Set<string>
}

interface PerformanceActions {
  // Core Actions
  startMonitoring(): void
  stopMonitoring(): void
  recordMetric(metric: string, value: number, metadata?: any): void
  recordInteraction(interaction: Omit<UserInteractionMetric, 'id' | 'timestamp'>): void
  recordComponentMetric(metric: ComponentPerformanceMetric): void
  recordApiMetric(metric: ApiPerformanceMetric): void
  
  // Alerts
  checkThresholds(): void
  resolveAlert(alertId: string): void
  
  // Reporting
  getPerformanceReport(): PerformanceReport
  exportMetrics(): any
  sendReportToBackend(): Promise<void>
}

interface PerformanceReport {
  sessionDuration: number
  totalInteractions: number
  averageResponseTime: number
  performanceScore: number
  criticalIssues: PerformanceAlert[]
  recommendations: string[]
  coreVitalsGrade: 'A' | 'B' | 'C' | 'D' | 'F'
}

// Performance Thresholds (based on Google's recommendations)
const DEFAULT_THRESHOLDS = {
  lcp: 2500, // 2.5s
  fid: 100,  // 100ms
  cls: 0.1,  // 0.1
  fcp: 1800, // 1.8s
  ttfb: 800, // 800ms
  timeToInteractive: 3800, // 3.8s
  totalBlockingTime: 200, // 200ms
  jsHeapUsed: 50 * 1024 * 1024, // 50MB
  apiResponseTime: 2000, // 2s
  componentRenderTime: 16, // 16ms (60fps)
}

// Enhanced Performance Store
export const usePerformanceMonitoring = create<PerformanceState & PerformanceActions>()(
  subscribeWithSelector((set, get) => ({
    // Initial State
    coreMetrics: {},
    interactions: [],
    componentMetrics: new Map(),
    apiMetrics: [],
    alerts: [],
    isMonitoring: false,
    sampleRate: import.meta.env.PROD ? 0.1 : 1.0, // 10% sampling in production
    alertThresholds: DEFAULT_THRESHOLDS,
    sessionStart: Date.now(),
    pageViews: 1,
    uniqueComponents: new Set(),

    // Start monitoring
    startMonitoring() {
      const state = get()
      if (state.isMonitoring) return

      set({ isMonitoring: true, sessionStart: Date.now() })
      
      // Only monitor if we should sample this session
      if (Math.random() > state.sampleRate) return

      console.log('🚀 Performance monitoring started')

      // Set up core web vitals monitoring
      setupWebVitalsMonitoring()
      
      // Set up resource monitoring
      setupResourceMonitoring()
      
      // Set up interaction monitoring
      setupInteractionMonitoring()
      
      // Set up periodic reporting
      setupPeriodicReporting()
    },

    stopMonitoring() {
      set({ isMonitoring: false })
      console.log('⏹️ Performance monitoring stopped')
    },

    recordMetric(metric: string, value: number, metadata = {}) {
      set((state) => ({
        coreMetrics: {
          ...state.coreMetrics,
          [metric]: value
        }
      }))

      // Check if this metric exceeds thresholds
      get().checkThresholds()
    },

    recordInteraction(interaction) {
      const fullInteraction: UserInteractionMetric = {
        ...interaction,
        id: crypto.randomUUID(),
        timestamp: Date.now()
      }

      set((state) => ({
        interactions: [fullInteraction, ...state.interactions.slice(0, 999)] // Keep last 1000
      }))
    },

    recordComponentMetric(metric) {
      set((state) => {
        const newMetrics = new Map(state.componentMetrics)
        newMetrics.set(metric.componentName, metric)
        
        return {
          componentMetrics: newMetrics,
          uniqueComponents: new Set([...state.uniqueComponents, metric.componentName])
        }
      })

      // Alert on slow component renders
      if (metric.renderTime > get().alertThresholds.componentRenderTime) {
        const alert: PerformanceAlert = {
          id: crypto.randomUUID(),
          type: 'warning',
          metric: 'componentRenderTime',
          threshold: get().alertThresholds.componentRenderTime,
          actual: metric.renderTime,
          timestamp: Date.now(),
          resolved: false
        }

        set((state) => ({
          alerts: [alert, ...state.alerts]
        }))
      }
    },

    recordApiMetric(metric) {
      set((state) => ({
        apiMetrics: [metric, ...state.apiMetrics.slice(0, 499)] // Keep last 500
      }))

      // Alert on slow API calls
      if (metric.duration > get().alertThresholds.apiResponseTime) {
        const alert: PerformanceAlert = {
          id: crypto.randomUUID(),
          type: metric.duration > get().alertThresholds.apiResponseTime * 2 ? 'critical' : 'warning',
          metric: 'apiResponseTime',
          threshold: get().alertThresholds.apiResponseTime,
          actual: metric.duration,
          timestamp: Date.now(),
          resolved: false
        }

        set((state) => ({
          alerts: [alert, ...state.alerts]
        }))
      }
    },

    checkThresholds() {
      const { coreMetrics, alertThresholds } = get()
      const newAlerts: PerformanceAlert[] = []

      Object.entries(coreMetrics).forEach(([metric, value]) => {
        const threshold = alertThresholds[metric]
        if (threshold && value > threshold) {
          const severity = value > threshold * 2 ? 'critical' : 'warning'
          
          newAlerts.push({
            id: crypto.randomUUID(),
            type: severity,
            metric,
            threshold,
            actual: value,
            timestamp: Date.now(),
            resolved: false
          })
        }
      })

      if (newAlerts.length > 0) {
        set((state) => ({
          alerts: [...newAlerts, ...state.alerts]
        }))
      }
    },

    resolveAlert(alertId: string) {
      set((state) => ({
        alerts: state.alerts.map(alert => 
          alert.id === alertId ? { ...alert, resolved: true } : alert
        )
      }))
    },

    getPerformanceReport(): PerformanceReport {
      const state = get()
      const sessionDuration = Date.now() - state.sessionStart
      const totalInteractions = state.interactions.length
      
      // Calculate average API response time
      const apiTimes = state.apiMetrics.map(m => m.duration)
      const averageResponseTime = apiTimes.length > 0 
        ? apiTimes.reduce((a, b) => a + b) / apiTimes.length 
        : 0

      // Calculate Core Web Vitals grade
      const coreVitalsGrade = calculateCoreVitalsGrade(state.coreMetrics)
      
      // Calculate performance score (0-100)
      const performanceScore = calculatePerformanceScore(state)
      
      // Get critical issues
      const criticalIssues = state.alerts.filter(a => a.type === 'critical' && !a.resolved)
      
      // Generate recommendations
      const recommendations = generateRecommendations(state)

      return {
        sessionDuration,
        totalInteractions,
        averageResponseTime,
        performanceScore,
        criticalIssues,
        recommendations,
        coreVitalsGrade
      }
    },

    exportMetrics() {
      const state = get()
      return {
        timestamp: Date.now(),
        sessionId: crypto.randomUUID(),
        coreMetrics: state.coreMetrics,
        interactions: state.interactions.slice(0, 100), // Last 100 interactions
        componentMetrics: Object.fromEntries(state.componentMetrics),
        apiMetrics: state.apiMetrics.slice(0, 50), // Last 50 API calls
        alerts: state.alerts.filter(a => !a.resolved),
        performanceReport: get().getPerformanceReport()
      }
    },

    async sendReportToBackend() {
      if (!get().isMonitoring) return

      try {
        const report = get().exportMetrics()
        await apiClient.post('/monitoring/performance', report)
        console.log('📊 Performance report sent to backend')
      } catch (error) {
        console.error('Failed to send performance report:', error)
      }
    }
  }))
)

// Helper Functions
function calculateCoreVitalsGrade(metrics: PerformanceMetrics): 'A' | 'B' | 'C' | 'D' | 'F' {
  const { lcp = 0, fid = 0, cls = 0 } = metrics
  
  // Google's Core Web Vitals thresholds
  const lcpGood = lcp <= 2500, lcpNI = lcp <= 4000
  const fidGood = fid <= 100, fidNI = fid <= 300
  const clsGood = cls <= 0.1, clsNI = cls <= 0.25
  
  const goodCount = [lcpGood, fidGood, clsGood].filter(Boolean).length
  const needsImprovementCount = [lcpNI, fidNI, clsNI].filter(Boolean).length
  
  if (goodCount === 3) return 'A'
  if (goodCount === 2) return 'B'
  if (needsImprovementCount >= 2) return 'C'
  if (needsImprovementCount >= 1) return 'D'
  return 'F'
}

function calculatePerformanceScore(state: PerformanceState): number {
  const { coreMetrics, alerts, apiMetrics } = state
  let score = 100
  
  // Deduct for poor Core Web Vitals
  if (coreMetrics.lcp && coreMetrics.lcp > 2500) score -= 20
  if (coreMetrics.fid && coreMetrics.fid > 100) score -= 15
  if (coreMetrics.cls && coreMetrics.cls > 0.1) score -= 15
  
  // Deduct for alerts
  const criticalAlerts = alerts.filter(a => a.type === 'critical' && !a.resolved).length
  const warningAlerts = alerts.filter(a => a.type === 'warning' && !a.resolved).length
  score -= criticalAlerts * 15 + warningAlerts * 5
  
  // Deduct for slow API calls
  const slowApiCalls = apiMetrics.filter(m => m.duration > 2000).length
  score -= slowApiCalls * 2
  
  return Math.max(0, score)
}

function generateRecommendations(state: PerformanceState): string[] {
  const recommendations: string[] = []
  const { coreMetrics, alerts, componentMetrics, apiMetrics } = state
  
  // Core Web Vitals recommendations
  if (coreMetrics.lcp && coreMetrics.lcp > 2500) {
    recommendations.push('Optimize Largest Contentful Paint by reducing server response times and optimizing resource loading')
  }
  
  if (coreMetrics.fid && coreMetrics.fid > 100) {
    recommendations.push('Reduce First Input Delay by minimizing JavaScript execution time and using code splitting')
  }
  
  if (coreMetrics.cls && coreMetrics.cls > 0.1) {
    recommendations.push('Improve Cumulative Layout Shift by setting dimensions on images and avoid inserting content above existing content')
  }
  
  // Memory recommendations
  if (coreMetrics.jsHeapUsed && coreMetrics.jsHeapUsed > 30 * 1024 * 1024) {
    recommendations.push('Consider reducing memory usage by implementing virtualization for large lists and cleaning up event listeners')
  }
  
  // Component recommendations
  const slowComponents = Array.from(componentMetrics.values())
    .filter(c => c.renderTime > 16)
  if (slowComponents.length > 0) {
    recommendations.push(`Optimize slow-rendering components: ${slowComponents.map(c => c.componentName).join(', ')}`)
  }
  
  // API recommendations
  const slowApiCalls = apiMetrics.filter(m => m.duration > 2000)
  if (slowApiCalls.length > 0) {
    recommendations.push('Consider implementing caching or pagination for slow API endpoints')
  }
  
  return recommendations
}

// Web Vitals Monitoring Setup
function setupWebVitalsMonitoring() {
  // Largest Contentful Paint
  const lcpObserver = new PerformanceObserver((list) => {
    const entries = list.getEntries()
    const lastEntry = entries[entries.length - 1]
    usePerformanceMonitoring.getState().recordMetric('lcp', lastEntry.startTime)
  })
  lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] })

  // First Input Delay
  const fidObserver = new PerformanceObserver((list) => {
    const entries = list.getEntries()
    const firstEntry = entries[0]
    usePerformanceMonitoring.getState().recordMetric('fid', firstEntry.processingStart - firstEntry.startTime)
  })
  fidObserver.observe({ entryTypes: ['first-input'] })

  // Cumulative Layout Shift
  const clsObserver = new PerformanceObserver((list) => {
    let clsValue = 0
    const entries = list.getEntries()
    
    entries.forEach((entry: any) => {
      if (!entry.hadRecentInput) {
        clsValue += entry.value
      }
    })
    
    usePerformanceMonitoring.getState().recordMetric('cls', clsValue)
  })
  clsObserver.observe({ entryTypes: ['layout-shift'] })

  // First Contentful Paint
  const fcpObserver = new PerformanceObserver((list) => {
    const entries = list.getEntries()
    const fcpEntry = entries.find(entry => entry.name === 'first-contentful-paint')
    if (fcpEntry) {
      usePerformanceMonitoring.getState().recordMetric('fcp', fcpEntry.startTime)
    }
  })
  fcpObserver.observe({ entryTypes: ['paint'] })
}

function setupResourceMonitoring() {
  // Monitor navigation timing
  window.addEventListener('load', () => {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
    
    usePerformanceMonitoring.getState().recordMetric('domContentLoaded', navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart)
    usePerformanceMonitoring.getState().recordMetric('loadComplete', navigation.loadEventEnd - navigation.loadEventStart)
    usePerformanceMonitoring.getState().recordMetric('ttfb', navigation.responseStart - navigation.requestStart)
  })
  
  // Monitor memory usage
  if ('memory' in performance) {
    setInterval(() => {
      const memory = (performance as any).memory
      usePerformanceMonitoring.getState().recordMetric('jsHeapUsed', memory.usedJSHeapSize)
      usePerformanceMonitoring.getState().recordMetric('jsHeapTotal', memory.totalJSHeapSize)
    }, 30000) // Every 30 seconds
  }
}

function setupInteractionMonitoring() {
  const recordInteraction = usePerformanceMonitoring.getState().recordInteraction
  
  // Click tracking
  document.addEventListener('click', (event) => {
    const target = event.target as HTMLElement
    recordInteraction({
      type: 'click',
      target: target.tagName + (target.id ? `#${target.id}` : '') + (target.className ? `.${target.className}` : ''),
      metadata: {
        x: event.clientX,
        y: event.clientY
      }
    })
  })
  
  // Scroll tracking (throttled)
  let scrollTimeout: NodeJS.Timeout
  document.addEventListener('scroll', () => {
    clearTimeout(scrollTimeout)
    scrollTimeout = setTimeout(() => {
      recordInteraction({
        type: 'scroll',
        metadata: {
          scrollY: window.scrollY,
          scrollX: window.scrollX
        }
      })
    }, 100)
  })
}

function setupPeriodicReporting() {
  // Send report every 5 minutes in production, 30 seconds in development
  const interval = import.meta.env.PROD ? 5 * 60 * 1000 : 30 * 1000
  
  setInterval(() => {
    usePerformanceMonitoring.getState().sendReportToBackend()
  }, interval)
  
  // Send report on page unload
  window.addEventListener('beforeunload', () => {
    usePerformanceMonitoring.getState().sendReportToBackend()
  })
}

// React Performance Monitoring Hook
export const useComponentPerformance = (componentName: string) => {
  const recordComponentMetric = usePerformanceMonitoring(state => state.recordComponentMetric)
  const [renderCount, setRenderCount] = React.useState(0)
  const [mountTime] = React.useState(Date.now())
  const renderStartTime = React.useRef<number>(0)

  // Track render start
  React.useLayoutEffect(() => {
    renderStartTime.current = performance.now()
  })

  // Track render complete
  React.useEffect(() => {
    const renderTime = performance.now() - renderStartTime.current
    setRenderCount(prev => prev + 1)
    
    recordComponentMetric({
      componentName,
      renderTime,
      mountTime: renderCount === 0 ? Date.now() - mountTime : undefined,
      updateCount: renderCount,
      lastUpdate: Date.now()
    })
  })

  return {
    renderCount,
    componentName
  }
}

// API Performance Monitoring Interceptor
export const setupApiPerformanceMonitoring = () => {
  const recordApiMetric = usePerformanceMonitoring.getState().recordApiMetric
  
  // This would integrate with your API client
  const originalRequest = window.fetch
  window.fetch = async (...args) => {
    const startTime = performance.now()
    const url = args[0]?.toString() || 'unknown'
    
    try {
      const response = await originalRequest.apply(window, args)
      const endTime = performance.now()
      
      recordApiMetric({
        endpoint: url,
        method: 'GET', // Would extract from options
        duration: endTime - startTime,
        status: response.status,
        timestamp: Date.now(),
        size: parseInt(response.headers.get('content-length') || '0'),
        cached: response.headers.get('x-cache-status') === 'HIT'
      })
      
      return response
    } catch (error) {
      const endTime = performance.now()
      
      recordApiMetric({
        endpoint: url,
        method: 'GET',
        duration: endTime - startTime,
        status: 0,
        timestamp: Date.now()
      })
      
      throw error
    }
  }
}

// Performance Dashboard Component
export const PerformanceDashboard: React.FC = () => {
  const {
    coreMetrics,
    alerts,
    isMonitoring,
    getPerformanceReport
  } = usePerformanceMonitoring()
  
  const [report, setReport] = React.useState<PerformanceReport | null>(null)
  const [isExpanded, setIsExpanded] = React.useState(false)

  React.useEffect(() => {
    if (isMonitoring) {
      const interval = setInterval(() => {
        setReport(getPerformanceReport())
      }, 5000)
      
      return () => clearInterval(interval)
    }
  }, [isMonitoring, getPerformanceReport])

  if (!isMonitoring || !report) return null

  return (
    <motion.div
      className="fixed bottom-4 left-4 z-40 bg-slate-900/90 backdrop-blur-xl rounded-lg border border-slate-700 shadow-2xl max-w-sm"
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
    >
      <div
        className="p-3 cursor-pointer flex items-center justify-between hover:bg-slate-800/50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center">
          <div className={`w-3 h-3 rounded-full mr-2 ${
            report.performanceScore >= 90 ? 'bg-green-400' :
            report.performanceScore >= 70 ? 'bg-yellow-400' :
            'bg-red-400'
          }`} />
          <span className="text-sm font-medium text-slate-200">
            Performance: {report.performanceScore}/100
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        )}
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden border-t border-slate-700"
          >
            <div className="p-3 space-y-2">
              {/* Core Web Vitals */}
              <div>
                <h4 className="text-xs font-semibold text-slate-400 mb-2">Core Web Vitals</h4>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="text-center">
                    <div className="text-slate-300">LCP</div>
                    <div className={`font-mono ${
                      (coreMetrics.lcp || 0) <= 2500 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {coreMetrics.lcp ? `${Math.round(coreMetrics.lcp)}ms` : 'N/A'}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-slate-300">FID</div>
                    <div className={`font-mono ${
                      (coreMetrics.fid || 0) <= 100 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {coreMetrics.fid ? `${Math.round(coreMetrics.fid)}ms` : 'N/A'}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-slate-300">CLS</div>
                    <div className={`font-mono ${
                      (coreMetrics.cls || 0) <= 0.1 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {coreMetrics.cls ? coreMetrics.cls.toFixed(3) : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Grade and Stats */}
              <div className="flex justify-between items-center pt-2 border-t border-slate-800">
                <div>
                  <span className="text-xs text-slate-400">Grade: </span>
                  <span className={`text-sm font-bold ${
                    report.coreVitalsGrade === 'A' ? 'text-green-400' :
                    report.coreVitalsGrade === 'B' ? 'text-blue-400' :
                    report.coreVitalsGrade === 'C' ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {report.coreVitalsGrade}
                  </span>
                </div>
                <div className="text-xs text-slate-400">
                  {report.totalInteractions} interactions
                </div>
              </div>

              {/* Critical Alerts */}
              {report.criticalIssues.length > 0 && (
                <div className="pt-2 border-t border-red-900/30">
                  <h5 className="text-xs font-semibold text-red-400 mb-1">
                    Critical Issues ({report.criticalIssues.length})
                  </h5>
                  {report.criticalIssues.slice(0, 2).map(issue => (
                    <div key={issue.id} className="text-xs text-red-300 truncate">
                      {issue.metric}: {issue.actual} (limit: {issue.threshold})
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// Export utility functions
export const performanceUtils = {
  // Measure function execution time
  measureFunction: <T extends (...args: any[]) => any>(
    fn: T,
    name: string
  ): T => {
    return ((...args: Parameters<T>) => {
      const startTime = performance.now()
      const result = fn(...args)
      const endTime = performance.now()
      
      usePerformanceMonitoring.getState().recordMetric(
        `function_${name}`,
        endTime - startTime
      )
      
      return result
    }) as T
  },

  // Measure async function execution time
  measureAsyncFunction: <T extends (...args: any[]) => Promise<any>>(
    fn: T,
    name: string
  ): T => {
    return (async (...args: Parameters<T>) => {
      const startTime = performance.now()
      try {
        const result = await fn(...args)
        const endTime = performance.now()
        
        usePerformanceMonitoring.getState().recordMetric(
          `async_function_${name}`,
          endTime - startTime
        )
        
        return result
      } catch (error) {
        const endTime = performance.now()
        
        usePerformanceMonitoring.getState().recordMetric(
          `async_function_${name}_error`,
          endTime - startTime
        )
        
        throw error
      }
    }) as T
  },

  // Get current performance snapshot
  getSnapshot: () => {
    const state = usePerformanceMonitoring.getState()
    return {
      timestamp: Date.now(),
      coreMetrics: state.coreMetrics,
      alerts: state.alerts.filter(a => !a.resolved),
      performanceScore: state.getPerformanceReport().performanceScore
    }
  },

  // Check if performance is degraded
  isPerformanceDegraded: (): boolean => {
    const report = usePerformanceMonitoring.getState().getPerformanceReport()
    return report.performanceScore < 70 || report.criticalIssues.length > 0
  }
}

// Auto-start monitoring when imported
if (typeof window !== 'undefined') {
  // Start monitoring after page load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      usePerformanceMonitoring.getState().startMonitoring()
    })
  } else {
    usePerformanceMonitoring.getState().startMonitoring()
  }
}