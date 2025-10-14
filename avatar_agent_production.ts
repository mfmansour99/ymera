// src/agents/AvatarAgent.ts - Production Ready Enterprise Version
import { apiClient } from '../api/fixed_api_client';

// Safe logging utility
const logger = {
  error: (context: string, message: string, error?: any): void => {
    console.error(`[${context}]`, message, error);
  },
  warn: (context: string, message: string, data?: any): void => {
    console.warn(`[${context}]`, message, data);
  },
  info: (context: string, message: string, data?: any): void => {
    console.info(`[${context}]`, message, data);
  },
};

// UUID generation with proper fallback
const generateUUID = (): string => {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
  } catch (error) {
    logger.warn('UUID', 'crypto.randomUUID failed, using fallback');
  }

  // Fallback implementation
  const chars = '0123456789abcdef';
  const segments = [8, 4, 4, 4, 12];
  let uuid = '';

  for (const segmentLength of segments) {
    if (uuid) uuid += '-';
    for (let i = 0; i < segmentLength; i++) {
      uuid += chars[Math.floor(Math.random() * 16)];
    }
  }

  return uuid;
};

// Agent Status Types
enum AgentStatus {
  IDLE = 'idle',
  PROCESSING = 'processing',
  RUNNING = 'running',
  ERROR = 'error',
  ALERT = 'alert',
}

enum TaskPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  CRITICAL = 'critical',
}

// AI Model Configuration
const AI_MODELS = {
  VISION_ANALYSIS: 'gemini-pro-vision',
  TEXT_ANALYSIS: 'claude-3-haiku',
  CODE_REVIEW: 'claude-3-sonnet',
  PERFORMANCE_ANALYSIS: 'gpt-4',
  ACCESSIBILITY_CHECK: 'gemini-pro',
  UX_OPTIMIZATION: 'claude-3-opus',
} as const;

// Task Templates for Consistent Prompting
const TASK_TEMPLATES = {
  VIEW_OPTIMIZATION: {
    prompt: `Analyze this application screenshot for UX/UI optimization opportunities.

Current Context:
- Route: {route}
- User Role: {userRole}
- Device: {deviceType}
- Viewport: {viewport}

Focus Areas:
1. Visual hierarchy and information density
2. Accessibility compliance (WCAG 2.1 AA)
3. Performance bottlenecks (large images, heavy animations)
4. Navigation clarity and user flow
5. Mobile responsiveness issues
6. Color contrast and readability
7. Interactive element sizing and positioning

Provide specific, actionable recommendations with priority levels (Critical/High/Medium/Low).`,
    model: AI_MODELS.VISION_ANALYSIS,
  },

  NOTIFICATION_SUMMARY: {
    prompt: `Analyze and summarize these notifications for maximum clarity and actionability.

Notifications: {notifications}

Requirements:
1. Group similar notifications by category
2. Prioritize by urgency and importance
3. Suggest 3 immediate actions
4. Identify patterns or trends
5. Flag any security or critical system alerts

Format: Concise executive summary + prioritized action list.`,
    model: AI_MODELS.TEXT_ANALYSIS,
  },

  PERFORMANCE_ANALYSIS: {
    prompt: `Analyze application performance metrics and user interaction data.

Metrics:
- Load Time: {loadTime}ms
- Memory Usage: {memoryUsage}MB
- Error Rate: {errorRate}%
- User Actions: {userActions}
- Page Views: {pageViews}

Identify:
1. Performance bottlenecks
2. User experience friction points
3. Resource optimization opportunities
4. Caching improvements
5. Code splitting opportunities

Provide technical recommendations with implementation priority.`,
    model: AI_MODELS.PERFORMANCE_ANALYSIS,
  },
} as const;

// Type definitions
interface RateLimit {
  calls: number;
  lastReset: number;
  limit: number;
  window: number;
}

interface TaskMetadata {
  agentVersion: string;
  timestamp: number;
  userAgent: string;
  viewport: string;
  [key: string]: any;
}

interface TaskResult {
  success: boolean;
  result?: any;
  error?: string;
  metadata: {
    duration: number;
    model?: string;
    timestamp: number;
    taskId?: string;
    taskType?: string;
  };
}

interface AgentStatus {
  name: string;
  version: string;
  capabilities: Record<string, boolean>;
  performance: {
    totalTasks: number;
    successfulTasks: number;
    averageResponseTime: number;
    lastTaskTime: number | null;
  };
  rateLimits: Record<string, { remaining: number; resetIn: number }>;
  health: 'healthy' | 'degraded' | 'unknown';
}

interface AccessibilityIssue {
  type: string;
  count: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  elements?: any[];
  description?: string;
}

interface ViewportData {
  type: string;
  width: number;
  height: number;
  devicePixelRatio: number;
  userAgent: string;
  timestamp: number;
}

// Rate limiting utility class
class RateLimitManager {
  private limits: Map<string, RateLimit> = new Map();

  constructor(initialLimits: Record<string, RateLimit>) {
    Object.entries(initialLimits).forEach(([key, limit]) => {
      this.limits.set(key, { ...limit });
    });
  }

  check(taskType: string): boolean {
    const limit = this.limits.get(taskType);
    if (!limit) return true;

    const now = Date.now();
    // Reset if window has passed
    if (now - limit.lastReset > limit.window) {
      limit.calls = 0;
      limit.lastReset = now;
    }

    if (limit.calls >= limit.limit) {
      const resetIn = Math.ceil((limit.window - (now - limit.lastReset)) / 1000);
      throw new Error(
        `Rate limit exceeded for ${taskType}. Limit: ${limit.limit} calls per ${Math.round(limit.window / 1000)}s. Try again in ${resetIn}s.`
      );
    }

    limit.calls++;
    return true;
  }

  getRemainingAndReset(taskType: string): { remaining: number; resetIn: number } {
    const limit = this.limits.get(taskType);
    if (!limit) return { remaining: -1, resetIn: 0 };

    const now = Date.now();
    const remaining = limit.limit - limit.calls;
    const resetIn = Math.max(0, limit.window - (now - limit.lastReset));

    return { remaining, resetIn };
  }

  reset(): void {
    const now = Date.now();
    this.limits.forEach((limit) => {
      limit.calls = 0;
      limit.lastReset = now;
    });
  }
}

// Performance tracking utility
class PerformanceTracker {
  private totalTasks = 0;
  private successfulTasks = 0;
  private responseTimes: number[] = [];
  private lastTaskTime: number | null = null;

  recordSuccess(duration: number): void {
    this.totalTasks++;
    this.successfulTasks++;
    this.lastTaskTime = duration;
    this.responseTimes.push(duration);

    // Keep only last 100 measurements
    if (this.responseTimes.length > 100) {
      this.responseTimes.shift();
    }
  }

  recordFailure(): void {
    this.totalTasks++;
  }

  getStats() {
    const successRate =
      this.totalTasks > 0
        ? ((this.successfulTasks / this.totalTasks) * 100).toFixed(2)
        : 0;
    const avgTime =
      this.responseTimes.length > 0
        ? Math.round(
            this.responseTimes.reduce((a, b) => a + b, 0) / this.responseTimes.length
          )
        : 0;

    return {
      totalTasks: this.totalTasks,
      successfulTasks: this.successfulTasks,
      successRate: parseFloat(String(successRate)),
      averageResponseTime: avgTime,
      lastTaskTime: this.lastTaskTime,
    };
  }

  reset(): void {
    this.totalTasks = 0;
    this.successfulTasks = 0;
    this.responseTimes = [];
    this.lastTaskTime = null;
  }
}

/**
 * AvatarAgent - Enterprise-grade AI agent for UI/UX optimization and analysis
 * Handles concurrent requests, rate limiting, error recovery, and performance monitoring
 */
class AvatarAgent {
  readonly name: string = 'AvatarAgent';
  readonly version: string = '2.1.0';
  readonly description: string =
    'Advanced UI/UX optimization and analysis agent with enterprise capabilities';

  capabilities: Record<string, boolean>;
  private rateLimitManager: RateLimitManager;
  private performanceTracker: PerformanceTracker;
  private currentStatus: AgentStatus = AgentStatus.IDLE;
  private html2canvasLoader: Promise<any> | null = null;

  constructor() {
    this.capabilities = {
      viewOptimization: true,
      notificationSummary: true,
      performanceAnalysis: true,
      accessibilityAudit: true,
      userBehaviorAnalysis: true,
      securityScan: true,
      codeReview: false,
    };

    this.rateLimitManager = new RateLimitManager({
      viewOptimization: { calls: 0, lastReset: Date.now(), limit: 10, window: 60000 },
      notificationSummary: { calls: 0, lastReset: Date.now(), limit: 20, window: 60000 },
      performanceAnalysis: { calls: 0, lastReset: Date.now(), limit: 5, window: 300000 },
    });

    this.performanceTracker = new PerformanceTracker();

    // Auto-initialize capabilities
    this.autoInitializeCapabilities();
  }

  private autoInitializeCapabilities(): void {
    try {
      if (typeof window === 'undefined') return;

      if (typeof (window as any).html2canvas === 'function') {
        this.capabilities.advancedScreenshot = true;
      }

      if (window.performance && (performance as any).memory) {
        this.capabilities.detailedPerformanceMetrics = true;
      }

      if (window.getComputedStyle) {
        this.capabilities.accessibilityAudit = true;
      }
    } catch (error) {
      logger.warn('AvatarAgent', 'Auto-initialization failed', error);
    }
  }

  /**
   * Lazy-load html2canvas to avoid blocking initialization
   */
  private async loadHtml2Canvas(): Promise<any> {
    if (this.html2canvasLoader) {
      return this.html2canvasLoader;
    }

    this.html2canvasLoader = (async () => {
      try {
        if (typeof (window as any).html2canvas === 'function') {
          return (window as any).html2canvas;
        }

        // Attempt to dynamically import html2canvas if available
        const module = await import('html2canvas');
        return module.default;
      } catch (error) {
        logger.warn('AvatarAgent', 'Failed to load html2canvas', error);
        return null;
      }
    })();

    return this.html2canvasLoader;
  }

  /**
   * Check rate limits for task type
   */
  private checkRateLimit(taskType: string): void {
    this.rateLimitManager.check(taskType);
  }

  /**
   * Execute a task with error handling and performance tracking
   */
  private async executeTask(
    taskType: string,
    taskData: Record<string, any>,
    options: {
      allowConcurrent?: boolean;
      timeout?: number;
      retries?: number;
      priority?: TaskPriority;
      metadata?: Record<string, any>;
    } = {}
  ): Promise<TaskResult> {
    const startTime = Date.now();
    const taskId = generateUUID();

    try {
      // Validate task type
      this.checkRateLimit(taskType);

      if (!this.capabilities[taskType]) {
        throw new Error(`Capability '${taskType}' is not enabled for this agent`);
      }

      if (this.currentStatus === AgentStatus.RUNNING && !options.allowConcurrent) {
        throw new Error(`Agent busy with another task. Current status: ${this.currentStatus}`);
      }

      this.currentStatus = AgentStatus.PROCESSING;

      const template = TASK_TEMPLATES[taskType.toUpperCase() as keyof typeof TASK_TEMPLATES];
      const payload = {
        id: taskId,
        task: taskType,
        prompt: this.interpolateTemplate(template?.prompt || '', taskData),
        data: taskData,
        model: template?.model || AI_MODELS.TEXT_ANALYSIS,
        priority: options.priority || TaskPriority.NORMAL,
        metadata: {
          agentVersion: this.version,
          timestamp: Date.now(),
          userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
          viewport:
            typeof window !== 'undefined'
              ? `${window.innerWidth}x${window.innerHeight}`
              : 'unknown',
          ...options.metadata,
        },
      };

      const response = await apiClient.post('/api/agent/avatar/task', payload, {
        timeout: options.timeout || 30000,
      });

      const duration = Date.now() - startTime;
      this.performanceTracker.recordSuccess(duration);
      this.currentStatus = AgentStatus.IDLE;

      return {
        success: true,
        result: response?.result || response,
        metadata: {
          duration,
          model: payload.model,
          timestamp: Date.now(),
          taskId,
          taskType,
        },
      };
    } catch (error: any) {
      const duration = Date.now() - startTime;
      this.performanceTracker.recordFailure();

      if (error.message.includes('Rate limit')) {
        this.currentStatus = AgentStatus.ALERT;
      } else {
        this.currentStatus = AgentStatus.ERROR;
      }

      logger.error('AvatarAgent', `Task '${taskType}' failed after ${duration}ms`, error);

      return {
        success: false,
        error: error.message || 'Unknown error occurred',
        metadata: {
          duration,
          timestamp: Date.now(),
          taskId,
          taskType,
        },
      };
    }
  }

  /**
   * Interpolate template variables
   */
  private interpolateTemplate(template: string, data: Record<string, any>): string {
    if (!template || typeof template !== 'string') return '';

    return template.replace(/\{(\w+)\}/g, (match, key) => {
      return data?.[key] !== undefined ? String(data[key]) : match;
    });
  }

  /**
   * Capture viewport with fallback to metadata
   */
  private async captureViewport(): Promise<string | ViewportData | null> {
    try {
      if (typeof window === 'undefined') {
        return null;
      }

      const html2canvas = await this.loadHtml2Canvas();
      if (html2canvas && document.body) {
        const canvas = await html2canvas(document.body, {
          width: window.innerWidth,
          height: window.innerHeight,
          useCORS: true,
          allowTaint: true,
          backgroundColor: '#ffffff',
        });
        return canvas.toDataURL('image/jpeg', 0.8);
      }

      // Fallback to metadata
      return {
        type: 'viewport_metadata',
        width: window.innerWidth,
        height: window.innerHeight,
        devicePixelRatio: window.devicePixelRatio,
        userAgent: navigator.userAgent,
        timestamp: Date.now(),
      };
    } catch (error) {
      logger.warn('AvatarAgent', 'Viewport capture failed', error);
      return null;
    }
  }

  /**
   * Public method: Optimize view
   */
  async optimizeView(
    currentRoute: string,
    options: {
      userRole?: string;
      context?: Record<string, any>;
    } = {}
  ): Promise<TaskResult> {
    try {
      const viewport = await this.captureViewport();
      const userRole = options.userRole || 'user';
      const deviceType = this.detectDeviceType();

      const taskData = {
        route: currentRoute,
        viewport:
          typeof viewport === 'string'
            ? viewport
            : viewport
              ? JSON.stringify(viewport)
              : 'unavailable',
        userRole,
        deviceType,
        timestamp: Date.now(),
        ...options.context,
      };

      return await this.executeTask('viewOptimization', taskData, {
        priority: TaskPriority.HIGH,
        metadata: {
          route: currentRoute,
          userRole,
          deviceType,
        },
      });
    } catch (error: any) {
      logger.error('AvatarAgent', 'optimizeView failed', error);
      return {
        success: false,
        error: error.message || 'View optimization failed',
        metadata: {
          duration: 0,
          timestamp: Date.now(),
          taskType: 'viewOptimization',
        },
      };
    }
  }

  /**
   * Public method: Summarize notifications
   */
  async summarizeNotifications(
    notifications: any[] = [],
    options: { timeRange?: string } = {}
  ): Promise<TaskResult> {
    try {
      if (!Array.isArray(notifications) || notifications.length === 0) {
        return {
          success: true,
          result: {
            summary: 'No notifications to summarize.',
            categories: {},
            count: 0,
          },
          metadata: {
            duration: 0,
            timestamp: Date.now(),
            taskType: 'notificationSummary',
          },
        };
      }

      const unreadNotifications = notifications.filter((n) => !n.read);
      const categorized = this.categorizeNotifications(unreadNotifications);

      const taskData = {
        notifications: JSON.stringify(categorized),
        totalCount: unreadNotifications.length,
        categories: Object.keys(categorized),
        timeRange: options.timeRange || '24h',
      };

      return await this.executeTask('notificationSummary', taskData, {
        priority: TaskPriority.NORMAL,
      });
    } catch (error: any) {
      logger.error('AvatarAgent', 'summarizeNotifications failed', error);
      return {
        success: false,
        error: error.message || 'Notification summary failed',
        metadata: {
          duration: 0,
          timestamp: Date.now(),
          taskType: 'notificationSummary',
        },
      };
    }
  }

  /**
   * Public method: Analyze performance
   */
  async analyzePerformance(metrics: Record<string, any> = {}): Promise<TaskResult> {
    try {
      let memoryUsage = 0;

      if (typeof window !== 'undefined' && window.performance) {
        try {
          const perfMemory = (performance as any).memory;
          if (perfMemory?.usedJSHeapSize) {
            memoryUsage = perfMemory.usedJSHeapSize / 1024 / 1024;
          }
        } catch {
          // Memory API not available
        }
      }

      const performanceMetrics = {
        loadTime: metrics.loadTime || 0,
        memoryUsage,
        errorRate: metrics.errorRate || 0,
        userActions: metrics.userActions || [],
        pageViews: metrics.pageViews || 1,
        ...metrics,
      };

      return await this.executeTask('performanceAnalysis', performanceMetrics, {
        priority: TaskPriority.HIGH,
        timeout: 45000,
      });
    } catch (error: any) {
      logger.error('AvatarAgent', 'analyzePerformance failed', error);
      return {
        success: false,
        error: error.message || 'Performance analysis failed',
        metadata: {
          duration: 0,
          timestamp: Date.now(),
          taskType: 'performanceAnalysis',
        },
      };
    }
  }

  /**
   * Public method: Audit accessibility
   */
  async auditAccessibility(options: {
    wcagLevel?: string;
    includeColorContrast?: boolean;
    includeKeyboardNav?: boolean;
  } = {}): Promise<TaskResult> {
    try {
      const issues = await this.detectAccessibilityIssues();

      const taskData = {
        issues: JSON.stringify(issues),
        wcagLevel: options.wcagLevel || 'AA',
        includeColorContrast: options.includeColorContrast !== false,
        includeKeyboardNav: options.includeKeyboardNav !== false,
        issueCount: issues.length,
      };

      return await this.executeTask('accessibilityAudit', taskData, {
        priority: TaskPriority.NORMAL,
      });
    } catch (error: any) {
      logger.error('AvatarAgent', 'auditAccessibility failed', error);
      return {
        success: false,
        error: error.message || 'Accessibility audit failed',
        metadata: {
          duration: 0,
          timestamp: Date.now(),
          taskType: 'accessibilityAudit',
        },
      };
    }
  }

  /**
   * Detect device type based on viewport width
   */
  private detectDeviceType(): string {
    if (typeof window === 'undefined') return 'unknown';
    const width = window.innerWidth;
    if (width < 768) return 'mobile';
    if (width < 1024) return 'tablet';
    return 'desktop';
  }

  /**
   * Categorize notifications by type and priority
   */
  private categorizeNotifications(notifications: any[]): Record<string, any[]> {
    const categories: Record<string, any[]> = {
      critical: [],
      alerts: [],
      updates: [],
      messages: [],
      system: [],
    };

    notifications.forEach((notification) => {
      const type = (notification?.type || '').toLowerCase();
      const priority = (notification?.priority || 'normal').toLowerCase();

      if (priority === 'critical' || type.includes('error')) {
        categories.critical.push(notification);
      } else if (priority === 'high' || type.includes('alert')) {
        categories.alerts.push(notification);
      } else if (type.includes('update')) {
        categories.updates.push(notification);
      } else if (type.includes('message') || type.includes('chat')) {
        categories.messages.push(notification);
      } else {
        categories.system.push(notification);
      }
    });

    return categories;
  }

  /**
   * Detect accessibility issues in current DOM
   */
  private async detectAccessibilityIssues(): Promise<AccessibilityIssue[]> {
    const issues: AccessibilityIssue[] = [];

    if (typeof document === 'undefined') return issues;

    try {
      // Check for missing alt attributes on images
      const imagesWithoutAlt = document.querySelectorAll('img:not([alt])');
      if (imagesWithoutAlt.length > 0) {
        issues.push({
          type: 'missing_alt_text',
          count: imagesWithoutAlt.length,
          severity: 'high',
          description: `Found ${imagesWithoutAlt.length} images without alt text`,
          elements: Array.from(imagesWithoutAlt)
            .slice(0, 5)
            .map((img) => (img as HTMLImageElement).src),
        });
      }

      // Check for keyboard navigation issues
      const focusableElements = document.querySelectorAll(
        'button, a, input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const nonFocusable = Array.from(focusableElements).filter((el) => {
        const htmlEl = el as HTMLElement;
        return !el.hasAttribute('tabindex') && htmlEl.tabIndex < 0;
      });

      if (nonFocusable.length > 0) {
        issues.push({
          type: 'keyboard_navigation',
          count: nonFocusable.length,
          severity: 'medium',
          description: `${nonFocusable.length} interactive elements may have keyboard accessibility issues`,
        });
      }

      // Check for color contrast (basic check)
      const textElements = document.querySelectorAll('p, span, a, button, label');
      let contrastIssues = 0;

      textElements.forEach((el) => {
        const style = window.getComputedStyle(el);
        const bgColor = style.backgroundColor;
        const color = style.color;

        // Simple heuristic: both are 'rgb(0, 0, 0)' or both are white
        if (bgColor === color || (bgColor.includes('0, 0, 0') && color.includes('0, 0, 0'))) {
          contrastIssues++;
        }
      });

      if (contrastIssues > 0) {
        issues.push({
          type: 'color_contrast',
          count: Math.min(contrastIssues, 10),
          severity: 'medium',
          description: `Found potential color contrast issues in ${contrastIssues} elements`,
        });
      }
    } catch (error) {
      logger.warn('AvatarAgent', 'Accessibility detection error', error);
    }

    return issues;
  }

  /**
   * Get current agent status
   */
  getStatus(): AgentStatus {
    const stats = this.performanceTracker.getStats();
    const { remaining: viewOptRemaining, resetIn: viewOptResetIn } =
      this.rateLimitManager.getRemainingAndReset('viewOptimization');
    const { remaining: notifRemaining, resetIn: notifResetIn } =
      this.rateLimitManager.getRemainingAndReset('notificationSummary');
    const { remaining: perfRemaining, resetIn: perfResetIn } =
      this.rateLimitManager.getRemainingAndReset('performanceAnalysis');

    return {
      name: this.name,
      version: this.version,
      capabilities: this.capabilities,
      performance: {
        totalTasks: stats.totalTasks,
        successfulTasks: stats.successfulTasks,
        averageResponseTime: stats.averageResponseTime,
        lastTaskTime: stats.lastTaskTime,
      },
      rateLimits: {
        viewOptimization: { remaining: viewOptRemaining, resetIn: viewOptResetIn },
        notificationSummary: { remaining: notifRemaining, resetIn: notifResetIn },
        performanceAnalysis: { remaining: perfRemaining, resetIn: perfResetIn },
      },
      health: stats.successRate >= 90 ? 'healthy' : stats.totalTasks > 0 ? 'degraded' : 'unknown',
    };
  }

  /**
   * Update agent capabilities
   */
  updateCapabilities(capabilities: Record<string, boolean>): void {
    this.capabilities = { ...this.capabilities, ...capabilities };
    logger.info('AvatarAgent', 'Capabilities updated', this.capabilities);
  }

  /**
   * Reset agent state
   */
  reset(): void {
    this.performanceTracker.reset();
    this.rateLimitManager.reset();
    this.currentStatus = AgentStatus.IDLE;
    logger.info('AvatarAgent', 'Agent state reset');
  }
}

// Export singleton instance
export const avatarAgent = new AvatarAgent();