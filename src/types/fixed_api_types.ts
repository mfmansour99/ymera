// src/types/api.ts
// Complete API request/response type definitions with all missing types

import {
  User,
  Agent,
  Project,
  FileRecord,
  ActivityLog,
  Notification,
  AgentFilters,
  ProjectFilters,
  PaginatedResponse,
  ApiResponse,
  ActivityLogFilter,
} from './index';

// ============================================================================
// AUTH API TYPES
// ============================================================================

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponseData {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface VerifyTokenResponse {
  valid: boolean;
  user?: User;
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  newPassword: string;
}

// FIXED: Added missing ChangePasswordRequest and ChangePasswordResponse
export interface ChangePasswordRequest {
  currentPassword: string;
  newPassword: string;
}

export interface ChangePasswordResponse extends ApiResponse<{
  success: boolean;
  message?: string;
}> {}

// ============================================================================
// AGENT API TYPES
// ============================================================================

export interface GetAgentsRequest extends AgentFilters {}

export interface GetAgentsResponse extends PaginatedResponse<Agent> {}

export interface GetAgentByIdRequest {
  id: number;
}

export interface GetAgentByIdResponse extends ApiResponse<Agent> {}

export interface CreateAgentRequest {
  name: string;
  type: string;
  description?: string;
}

export interface CreateAgentResponse extends ApiResponse<Agent> {}

export interface UpdateAgentRequest {
  id: number;
  updates: {
    name?: string;
    type?: string;
    status?: string;
    description?: string;
    tasks?: number;
    efficiency?: number;
  };
}

export interface UpdateAgentResponse extends ApiResponse<Agent> {}

export interface DeleteAgentRequest {
  id: number;
}

export interface DeleteAgentResponse extends ApiResponse<{ id: number }> {}

export interface GetAgentStatsRequest {
  id: number;
  startDate?: string;
  endDate?: string;
}

export interface AgentStatsData {
  tasksCompleted: number;
  successRate: number;
  averageTime: number;
  activeTime: string;
  efficiency: number;
  cpuUsage: number[];
  memoryUsage: number[];
}

export interface GetAgentStatsResponse extends ApiResponse<AgentStatsData> {}

// ============================================================================
// PROJECT API TYPES
// ============================================================================

export interface GetProjectsRequest extends ProjectFilters {}

export interface GetProjectsResponse extends PaginatedResponse<Project> {}

export interface GetProjectByIdRequest {
  id: number;
}

export interface GetProjectByIdResponse extends ApiResponse<Project> {}

export interface CreateProjectRequest {
  name: string;
  description?: string;
  status?: string;
  team: number;
  deadline: string;
  priority?: string;
  tags?: string[];
}

export interface CreateProjectResponse extends ApiResponse<Project> {}

export interface UpdateProjectRequest {
  id: number;
  updates: {
    name?: string;
    description?: string;
    status?: string;
    progress?: number;
    team?: number;
    deadline?: string;
    priority?: string;
    tags?: string[];
  };
}

export interface UpdateProjectResponse extends ApiResponse<Project> {}

export interface DeleteProjectRequest {
  id: number;
}

export interface DeleteProjectResponse extends ApiResponse<{ id: number }> {}

export interface AssignAgentToProjectRequest {
  projectId: number;
  agentId: number;
}

export interface AssignAgentToProjectResponse extends ApiResponse<{
  projectId: number;
  agentId: number;
}> {}

// ============================================================================
// FILE API TYPES
// ============================================================================

export interface UploadFileRequest {
  file: File;
  entityType: 'agent' | 'project';
  entityId: number;
  metadata?: Record<string, unknown>; // FIXED: Changed from any to unknown
}

export interface UploadFileResponse extends ApiResponse<FileRecord> {}

export interface GetFilesRequest {
  entityType: 'agent' | 'project';
  entityId: number;
  page?: number;
  pageSize?: number;
}

export interface GetFilesResponse extends PaginatedResponse<FileRecord> {}

export interface DeleteFileRequest {
  fileId: string;
}

export interface DeleteFileResponse extends ApiResponse<{ id: string }> {}

export interface DownloadFileRequest {
  fileId: string;
}

export interface ShareFileRequest {
  fileId: string;
  fromEntityType: 'agent' | 'project';
  fromEntityId: number;
  toEntityType: 'agent' | 'project';
  toEntityId: number;
}

export interface ShareFileResponse extends ApiResponse<FileRecord> {}

// ============================================================================
// ACTIVITY LOG API TYPES
// ============================================================================

export interface GetActivityLogsRequest extends ActivityLogFilter {
  page?: number;
  pageSize?: number;
}

export interface GetActivityLogsResponse extends PaginatedResponse<ActivityLog> {}

export interface CreateActivityLogRequest {
  action: string;
  entity: string;
  entityId: string | number;
  details?: Record<string, unknown>; // FIXED: Changed from any to unknown
}

export interface CreateActivityLogResponse extends ApiResponse<ActivityLog> {}

// ============================================================================
// NOTIFICATION API TYPES
// ============================================================================

export interface GetNotificationsRequest {
  unreadOnly?: boolean;
  page?: number;
  pageSize?: number;
}

export interface GetNotificationsResponse extends PaginatedResponse<Notification> {}

export interface MarkNotificationReadRequest {
  notificationId: string;
}

export interface MarkNotificationReadResponse extends ApiResponse<Notification> {}

export interface MarkAllNotificationsReadResponse extends ApiResponse<{
  count: number;
}> {}

// ============================================================================
// USER API TYPES
// ============================================================================

export interface GetUserProfileResponse extends ApiResponse<User> {}

export interface UpdateUserProfileRequest {
  username?: string;
  email?: string;
  bio?: string;
  avatar?: string;
  preferences?: {
    theme?: string;
    language?: string;
    notifications?: boolean;
    emailNotifications?: boolean;
  };
}

export interface UpdateUserProfileResponse extends ApiResponse<User> {}

// ============================================================================
// SETTINGS API TYPES
// ============================================================================

export interface SettingsData {
  theme: string;
  animations: boolean;
  particles: boolean;
  performance: string;
  notifications: boolean;
  autoAssign: boolean;
  language: string;
}

export interface GetSettingsResponse extends ApiResponse<SettingsData> {}

export interface UpdateSettingsRequest {
  theme?: string;
  animations?: boolean;
  particles?: boolean;
  performance?: string;
  notifications?: boolean;
  autoAssign?: boolean;
  language?: string;
}

export interface UpdateSettingsResponse extends ApiResponse<SettingsData> {}

// ============================================================================
// CHAT API TYPES (Agent Interaction)
// ============================================================================

export interface SendMessageRequest {
  agentId: number;
  message: string;
  attachments?: string[];
}

export interface MessageResponseData {
  id: string;
  response: string;
  timestamp: number;
}

export interface SendMessageResponse extends ApiResponse<MessageResponseData> {}

export interface ChatHistoryItem {
  id: string;
  sender: 'user' | 'agent';
  text: string;
  timestamp: number;
}

export interface GetChatHistoryRequest {
  agentId: number;
  page?: number;
  pageSize?: number;
}

export interface GetChatHistoryResponse extends PaginatedResponse<ChatHistoryItem> {}

// ============================================================================
// EXPORT DATA API TYPES
// ============================================================================

export interface ExportDataRequest {
  type: 'agents' | 'projects' | 'all';
  format: 'json' | 'csv';
  filters?: Record<string, unknown>; // FIXED: Changed from any to unknown
}

export interface ExportDataResponseData {
  downloadUrl: string;
  expiresAt: string;
}

export interface ExportDataResponse extends ApiResponse<ExportDataResponseData> {}

// ============================================================================
// STATISTICS API TYPES
// ============================================================================

export interface DashboardStatsData {
  activeAgents: number;
  totalProjects: number;
  totalTasks: number;
  avgEfficiency: number;
  agentsByStatus: Record<string, number>;
  projectsByStatus: Record<string, number>;
  recentActivity: ActivityLog[];
}

export interface GetDashboardStatsResponse extends ApiResponse<DashboardStatsData> {}

export interface GetAnalyticsRequest {
  startDate: string;
  endDate: string;
  metrics?: string[];
}

export interface AgentPerformanceData {
  agentId: number;
  agentName: string;
  tasksCompleted: number;
  efficiency: number;
}

export interface ProjectProgressData {
  projectId: number;
  projectName: string;
  progress: number;
  status: string;
}

export interface SystemMetricsData {
  totalTasks: number;
  completedTasks: number;
  averageCompletionTime: number;
}

export interface AnalyticsData {
  agentPerformance: AgentPerformanceData[];
  projectProgress: ProjectProgressData[];
  systemMetrics: SystemMetricsData;
}

export interface GetAnalyticsResponse extends ApiResponse<AnalyticsData> {}

// ============================================================================
// WEBSOCKET API TYPES
// ============================================================================

export interface WebSocketConnectRequest {
  userId: string;
  token: string;
}

export interface WebSocketSubscribeRequest {
  events: string[];
  filters?: Record<string, unknown>; // FIXED: Changed from any to unknown
}

// ============================================================================
// BULK OPERATIONS TYPES
// ============================================================================

export interface BulkOperationFailure {
  id: number;
  reason: string;
}

export interface BulkDeleteAgentsRequest {
  agentIds: number[];
}

export interface BulkDeleteAgentsResponse extends ApiResponse<{
  deleted: number[];
  failed: BulkOperationFailure[];
}> {}

export interface BulkUpdateAgentsRequest {
  agentIds: number[];
  updates: {
    status?: string;
    efficiency?: number;
  };
}

export interface BulkUpdateAgentsResponse extends ApiResponse<{
  updated: number[];
  failed: BulkOperationFailure[];
}> {}

export interface BulkAssignAgentsRequest {
  projectId: number;
  agentIds: number[];
}

export interface BulkAssignAgentsResponse extends ApiResponse<{
  assigned: number[];
  failed: BulkOperationFailure[];
}> {}
