export interface Settings {
  general: {
    theme: string;
    language: string;
    timezone: string;
    dateFormat: string;
    autoSave: boolean;
    compactMode: boolean;
  };
  notifications: {
    email: boolean;
    push: boolean;
    desktop: boolean;
    agentAlerts: boolean;
    projectUpdates: boolean;
    weeklyReports: boolean;
    soundEnabled: boolean;
    quietHours: {
      enabled: boolean;
      start: string;
      end: string;
    };
  };
  agents: {
    maxConcurrentTasks: number;
    autoAssignment: boolean;
    taskPriority: string;
    failureRetries: number;
    timeoutDuration: number;
    monitoring: {
      realTimeUpdates: boolean;
      performanceTracking: boolean;
      errorReporting: boolean;
    };
  };
  security: {
    twoFactorAuth: boolean;
    sessionTimeout: number;
    passwordRequirements: {
      minLength: number;
      requireNumbers: boolean;
      requireSymbols: boolean;
      requireUppercase: boolean;
    };
    ipWhitelist: string[];
    apiAccess: boolean;
    auditLogs: boolean;
  };
  integrations: {
    slack: { enabled: boolean; webhook: string };
    discord: { enabled: boolean; webhook: string };
    github: { enabled: boolean; token: string };
    jira: { enabled: boolean; url: string; token: string };
    teams: { enabled: boolean; webhook: string };
  };
  performance: {
    animationsEnabled: boolean;
    particleEffects: boolean;
    autoRefresh: boolean;
    refreshInterval: number;
    dataCaching: boolean;
    preloadImages: boolean;
    reducedMotion: boolean;
  };
}
