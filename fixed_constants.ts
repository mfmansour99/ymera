// src/utils/constants.ts
// Application-wide constants with FIXED password regex

// ============================================================================
// API CONFIGURATION
// ============================================================================

export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:3000/api',
  WS_URL: process.env.REACT_APP_WS_URL || 'ws://localhost:3000',
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
} as const;

// ============================================================================
// VALIDATION RULES - FIXED PASSWORD REGEX
// ============================================================================

export const VALIDATION_RULES = {
  USERNAME: {
    MIN_LENGTH: 3,
    MAX_LENGTH: 30,
    PATTERN: /^[a-zA-Z0-9_-]+$/,
    ERROR_MESSAGE: 'Username must be 3-30 characters and contain only letters, numbers, underscores, and hyphens',
  },
  PASSWORD: {
    MIN_LENGTH: 8,
    MAX_LENGTH: 128,
    // FIXED: Added + quantifier at the end and $ anchor
    PATTERN: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$/,
    ERROR_MESSAGE: 'Password must be at least 8 characters with uppercase, lowercase, number, and special character',
  },
  EMAIL: {
    PATTERN: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    ERROR_MESSAGE: 'Invalid email address',
  },
  AGENT_NAME: {
    MIN_LENGTH: 1,
    MAX_LENGTH: 100,
    ERROR_MESSAGE: 'Agent name must be 1-100 characters',
  },
  PROJECT_NAME: {
    MIN_LENGTH: 1,
    MAX_LENGTH: 200,
    ERROR_MESSAGE: 'Project name must be 1-200 characters',
  },
} as const;

// ============================================================================
// FILE UPLOAD CONFIGURATION
// ============================================================================

export const FILE_CONFIG = {
  MAX_FILE_SIZE: parseInt(process.env.REACT_APP_MAX_FILE_SIZE || '10485760'), // 10MB default
  MAX_TOTAL_STORAGE: 52428800, // 50MB
  ALLOWED_FILE_TYPES: [
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/webp',
    'application/pdf',
    'text/plain',
    'text/csv',
    'application/json',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  ],
  ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf', '.txt', '.csv', '.json', '.xls', '.xlsx'],
} as const;

// ============================================================================
// PAGINATION DEFAULTS
// ============================================================================

export const PAGINATION = {
  DEFAULT_PAGE: 1,
  DEFAULT_PAGE_SIZE: 10,
  PAGE_SIZE_OPTIONS: [10, 25, 50, 100],
  MAX_PAGE_SIZE: 100,
} as const;

// ============================================================================
// AGENT CONFIGURATION
// ============================================================================

export const AGENT_TYPES = [
  { id: 'code-analyzer', label: 'Code Analyzer', icon: 'Code', color: '#00f5ff' },
  { id: 'ui-designer', label: 'UI Designer', icon: 'Camera', color: '#ff00aa' },
  { id: 'backend-dev', label: 'Backend Dev', icon: 'Database', color: '#00ff88' },
  { id: 'security', label: 'Security', icon: 'Shield', color: '#ffaa00' },
  { id: 'optimizer', label: 'Optimizer', icon: 'Zap', color: '#aa00ff' },
] as const;

export const AGENT_STATUSES = [
  { id: 'idle', label: 'Idle', color: '#6b7280' },
  { id: 'thinking', label: 'Thinking', color: '#fbbf24' },
  { id: 'working', label: 'Working', color: '#10b981' },
  { id: 'completed', label: 'Completed', color: '#06b6d4' },
  { id: 'error', label: 'Error', color: '#ef4444' },
] as const;

export const AGENT_ICON_MAP = {
  Code: 'Code',
  Camera: 'Camera',
  Database: 'Database',
  Shield: 'Shield',
  Zap: 'Zap',
} as const;

// ============================================================================
// PROJECT CONFIGURATION
// ============================================================================

export const PROJECT_STATUSES = [
  { id: 'planning', label: 'Planning', color: '#8b5cf6' },
  { id: 'in_progress', label: 'In Progress', color: '#06b6d4' },
  { id: 'completed', label: 'Completed', color: '#10b981' },
  { id: 'on_hold', label: 'On Hold', color: '#fbbf24' },
  { id: 'archived', label: 'Archived', color: '#6b7280' },
] as const;

export const PROJECT_PRIORITIES = [
  { id: 'low', label: 'Low', color: '#6b7280' },
  { id: 'medium', label: 'Medium', color: '#3b82f6' },
  { id: 'high', label: 'High', color: '#f59e0b' },
  { id: 'critical', label: 'Critical', color: '#ef4444' },
] as const;

// ============================================================================
// NOTIFICATION CONFIGURATION
// ============================================================================

export const NOTIFICATION_CONFIG = {
  AUTO_DISMISS_DELAY: 5000, // 5 seconds
  MAX_VISIBLE: 5,
  POSITION: 'top-right',
} as const;

export const NOTIFICATION_TYPES = {
  INFO: 'info',
  SUCCESS: 'success',
  WARNING: 'warning',
  ERROR: 'error',
} as const;

// ============================================================================
// PERFORMANCE MODES
// ============================================================================

export const PERFORMANCE_MODES = [
  { id: 'low', label: 'Low', particleCount: 30, animationQuality: 'low' },
  { id: 'balanced', label: 'Balanced', particleCount: 50, animationQuality: 'medium' },
  { id: 'high', label: 'High', particleCount: 80, animationQuality: 'high' },
] as const;

// ============================================================================
// THEME CONFIGURATION
// ============================================================================

export const THEMES = {
  DARK: 'dark',
  LIGHT: 'light',
  AUTO: 'auto',
} as const;

export const THEME_COLORS = {
  primary: {
    50: '#ecfeff',
    100: '#cffafe',
    200: '#a5f3fc',
    300: '#67e8f9',
    400: '#22d3ee',
    500: '#06b6d4',
    600: '#0891b2',
    700: '#0e7490',
    800: '#155e75',
    900: '#164e63',
  },
} as const;

// ============================================================================
// DEBOUNCE & THROTTLE DELAYS
// ============================================================================

export const TIMING = {
  SEARCH_DEBOUNCE: 300, // ms
  AUTO_SAVE_DEBOUNCE: 1000, // ms
  SCROLL_THROTTLE: 100, // ms
  WEBSOCKET_RECONNECT: 5000, // ms
  ACTIVITY_LOG_BATCH: 2000, // ms
} as const;

// ============================================================================
// LOCAL STORAGE KEYS
// ============================================================================

export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_DATA: 'user_data',
  THEME: 'theme',
  SETTINGS: 'settings',
  LANGUAGE: 'language',
  VIEW_MODE: 'view_mode',
  FILTERS: 'filters',
  LAST_ROUTE: 'last_route',
  ACTIVITY_LOGS: 'activity_logs', // NEW: For persistence
  NOTIFICATIONS: 'notifications', // NEW: For persistence
} as const;

// ============================================================================
// ROUTE PATHS
// ============================================================================

export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  DASHBOARD: '/dashboard',
  AGENTS: '/agents',
  AGENT_DETAIL: '/agents/:id',
  PROJECTS: '/projects',
  PROJECT_DETAIL: '/projects/:id',
  PROFILE: '/profile',
  SETTINGS: '/settings',
  ACTIVITY: '/activity',
  NOT_FOUND: '/404',
} as const;

// ============================================================================
// WEBSOCKET EVENT TYPES
// ============================================================================

export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  AGENT_STATUS_CHANGE: 'agent:status_change',
  AGENT_TASK_UPDATE: 'agent:task_update',
  PROJECT_PROGRESS_UPDATE: 'project:progress_update',
  NOTIFICATION_NEW: 'notification:new',
  USER_ONLINE: 'user:online',
  USER_OFFLINE: 'user:offline',
} as const;

// ============================================================================
// ERROR CODES
// ============================================================================

export const ERROR_CODES = {
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT: 'TIMEOUT',
  FILE_TOO_LARGE: 'FILE_TOO_LARGE',
  INVALID_FILE_TYPE: 'INVALID_FILE_TYPE',
  STORAGE_LIMIT_EXCEEDED: 'STORAGE_LIMIT_EXCEEDED',
} as const;

// ============================================================================
// STATUS CODES
// ============================================================================

export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503,
} as const;

// ============================================================================
// FEATURE FLAGS
// ============================================================================

export const FEATURES = {
  WEBSOCKET_ENABLED: process.env.REACT_APP_ENABLE_WEBSOCKET !== 'false',
  FILE_UPLOAD_ENABLED: true,
  CHAT_ENABLED: process.env.REACT_APP_ENABLE_CHAT !== 'false',
  ANALYTICS_ENABLED: process.env.REACT_APP_ENABLE_ANALYTICS !== 'false',
  EXPORT_ENABLED: true,
  BULK_OPERATIONS_ENABLED: true,
  ADVANCED_SEARCH_ENABLED: process.env.REACT_APP_ENABLE_ADVANCED_SEARCH !== 'false',
} as const;

// ============================================================================
// DATE/TIME FORMATS
// ============================================================================

export const DATE_FORMATS = {
  SHORT: 'MMM d, yyyy',
  LONG: 'MMMM d, yyyy',
  WITH_TIME: 'MMM d, yyyy h:mm a',
  TIME_ONLY: 'h:mm a',
  ISO: 'yyyy-MM-dd',
  RELATIVE: 'relative', // "2 hours ago"
} as const;

// ============================================================================
// ANIMATION DURATIONS
// ============================================================================

export const ANIMATIONS = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
  PARTICLE_FRAME_RATE: 60,
  LOGO_ROTATION_SPEED: 50,
} as const;

// ============================================================================
// BREAKPOINTS (matches Tailwind defaults)
// ============================================================================

export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

// ============================================================================
// EXPORT DEFAULT CONFIG OBJECT
// ============================================================================

export const CONFIG = {
  api: API_CONFIG,
  file: FILE_CONFIG,
  pagination: PAGINATION,
  notification: NOTIFICATION_CONFIG,
  timing: TIMING,
  storage: STORAGE_KEYS,
  routes: ROUTES,
  features: FEATURES,
  dateFormats: DATE_FORMATS,
  animations: ANIMATIONS,
  breakpoints: BREAKPOINTS,
} as const;

export default CONFIG;
