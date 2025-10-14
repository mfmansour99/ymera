import { Agent, Project, User } from './types';

// ============================================================================
// MOCK DATA - Complete and Production-Ready
// ============================================================================

const mockAgents: Agent[] = [
  {
    id: 'agent-001',
    name: 'CodeAnalyzer Alpha',
    type: 'code_analyzer',
    status: 'working',
    description: 'Advanced code analysis and optimization agent specializing in performance improvements and code quality metrics',
    tasksCompleted: 42,
    efficiency: '92%',
    currentPhase: 'code_analysis'
  },
  {
    id: 'agent-002',
    name: 'UIDesigner Beta',
    type: 'ui_designer',
    status: 'idle',
    description: 'Creates beautiful and functional user interfaces with focus on accessibility and user experience',
    tasksCompleted: 28,
    efficiency: '88%',
    currentPhase: null
  },
  {
    id: 'agent-003',
    name: 'BackendMaster',
    type: 'backend_developer',
    status: 'thinking',
    description: 'Builds robust backend systems and APIs with expertise in microservices architecture',
    tasksCompleted: 35,
    efficiency: '95%',
    currentPhase: 'database_design'
  },
  {
    id: 'agent-004',
    name: 'TestRunner Gamma',
    type: 'testing_agent',
    status: 'working',
    description: 'Comprehensive testing and quality assurance agent covering unit, integration, and E2E testing',
    tasksCompleted: 56,
    efficiency: '97%',
    currentPhase: 'regression_testing'
  },
  {
    id: 'agent-005',
    name: 'Optimizer Delta',
    type: 'optimization_agent',
    status: 'idle',
    description: 'Performance optimization specialist focusing on load times, bundle size, and runtime efficiency',
    tasksCompleted: 19,
    efficiency: '91%',
    currentPhase: null
  },
  {
    id: 'agent-006',
    name: 'SecurityGuard Epsilon',
    type: 'security_agent',
    status: 'working',
    description: 'Security audit specialist identifying vulnerabilities and implementing security best practices',
    tasksCompleted: 33,
    efficiency: '96%',
    currentPhase: 'security_audit'
  },
  {
    id: 'agent-007',
    name: 'DataMiner Zeta',
    type: 'data_analyst',
    status: 'thinking',
    description: 'Data analysis and insights agent providing actionable metrics and business intelligence',
    tasksCompleted: 24,
    efficiency: '89%',
    currentPhase: 'data_analysis'
  },
  {
    id: 'agent-008',
    name: 'DevOps Omega',
    type: 'devops_agent',
    status: 'completed',
    description: 'CI/CD pipeline management and deployment automation specialist',
    tasksCompleted: 47,
    efficiency: '94%',
    currentPhase: null
  }
];

const mockProjects: Project[] = [
  {
    id: 'proj-001',
    name: 'E-Commerce Platform',
    description: 'Complete e-commerce solution with payment processing, inventory management, and customer analytics',
    status: 'in_progress',
    priority: 'high',
    startDate: '2024-01-15',
    endDate: null,
    duration: '3 months',
    progress: 65,
    budget: 120000,
    spent: 78000,
    team: ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Williams'],
    technologies: ['React', 'Node.js', 'MongoDB', 'AWS', 'Stripe', 'Redis'],
    milestones: [
      { name: 'Requirements Analysis', date: '2024-01-30', completed: true },
      { name: 'UI/UX Design', date: '2024-02-15', completed: true },
      { name: 'Frontend Development', date: '2024-03-01', completed: true },
      { name: 'Backend Integration', date: '2024-03-15', completed: false },
      { name: 'Testing & Deployment', date: '2024-04-01', completed: false },
    ],
    performance: {
      tasksCompleted: 187,
      bugsFixed: 24,
      codeReviews: 62,
      efficiency: 89.5,
    },
  },
  {
    id: 'proj-002',
    name: 'Mobile Banking App',
    description: 'Cross-platform mobile banking application with biometric authentication and real-time transaction monitoring',
    status: 'completed',
    priority: 'high',
    startDate: '2023-11-01',
    endDate: '2024-02-28',
    duration: '4 months',
    progress: 100,
    budget: 150000,
    spent: 145000,
    team: ['Sarah Connor', 'Alex Murphy', 'John Connor'],
    technologies: ['Flutter', 'Dart', 'Firebase', 'Stripe', 'SQLite'],
    milestones: [
      { name: 'Requirements Gathering', date: '2023-11-15', completed: true },
      { name: 'Prototype Development', date: '2023-12-15', completed: true },
      { name: 'Core Features', date: '2024-01-15', completed: true },
      { name: 'Security Audit', date: '2024-02-01', completed: true },
      { name: 'Final Testing', date: '2024-02-20', completed: true },
    ],
    performance: {
      tasksCompleted: 247,
      bugsFixed: 32,
      codeReviews: 89,
      efficiency: 94.2,
    },
  },
  {
    id: 'proj-003',
    name: 'Healthcare Management System',
    description: 'HIPAA-compliant healthcare management platform for hospitals and clinics',
    status: 'in_progress',
    priority: 'critical',
    startDate: '2024-02-01',
    endDate: null,
    duration: '6 months',
    progress: 42,
    budget: 250000,
    spent: 105000,
    team: ['Dr. Emily Chen', 'Robert Martinez', 'Lisa Anderson', 'David Kim'],
    technologies: ['React', 'TypeScript', 'PostgreSQL', 'GraphQL', 'Docker'],
    milestones: [
      { name: 'Compliance Review', date: '2024-02-20', completed: true },
      { name: 'System Architecture', date: '2024-03-10', completed: true },
      { name: 'Patient Portal', date: '2024-04-15', completed: false },
      { name: 'Doctor Dashboard', date: '2024-05-20', completed: false },
      { name: 'Integration Testing', date: '2024-06-30', completed: false },
    ],
    performance: {
      tasksCompleted: 124,
      bugsFixed: 18,
      codeReviews: 45,
      efficiency: 87.3,
    },
  },
  {
    id: 'proj-004',
    name: 'Real-time Analytics Dashboard',
    description: 'Business intelligence dashboard with real-time data visualization and predictive analytics',
    status: 'planning',
    priority: 'medium',
    startDate: '2024-03-15',
    endDate: null,
    duration: '2 months',
    progress: 15,
    budget: 80000,
    spent: 12000,
    team: ['Michael Brown', 'Jennifer Lee'],
    technologies: ['Vue.js', 'D3.js', 'Python', 'Apache Kafka', 'ClickHouse'],
    milestones: [
      { name: 'Requirements Analysis', date: '2024-03-25', completed: true },
      { name: 'Data Pipeline Design', date: '2024-04-05', completed: false },
      { name: 'Frontend Development', date: '2024-04-20', completed: false },
      { name: 'Backend Integration', date: '2024-05-10', completed: false },
    ],
    performance: {
      tasksCompleted: 23,
      bugsFixed: 3,
      codeReviews: 8,
      efficiency: 78.5,
    },
  },
  {
    id: 'proj-005',
    name: 'Social Media Integration Platform',
    description: 'Multi-platform social media management tool with scheduling and analytics',
    status: 'on_hold',
    priority: 'low',
    startDate: '2024-01-10',
    endDate: null,
    duration: '3 months',
    progress: 28,
    budget: 95000,
    spent: 26650,
    team: ['Amanda Wilson', 'Chris Taylor'],
    technologies: ['Next.js', 'Express', 'MongoDB', 'Redis', 'OAuth2'],
    milestones: [
      { name: 'API Integration', date: '2024-01-25', completed: true },
      { name: 'Scheduling Engine', date: '2024-02-10', completed: false },
      { name: 'Analytics Module', date: '2024-03-05', completed: false },
      { name: 'User Interface', date: '2024-04-01', completed: false },
    ],
    performance: {
      tasksCompleted: 67,
      bugsFixed: 12,
      codeReviews: 29,
      efficiency: 82.1,
    },
  }
];

const mockUser: User = {
  id: 'user-001',
  name: 'Mohamed Mansour',
  email: 'mohamed@ymera.ai',
  role: 'Administrator',
  avatar: '/assets/avatar.png',
  lastLogin: '2024-03-15T14:30:00Z',
  preferences: {
    theme: 'dark',
    notifications: true,
    language: 'en'
  }
};

// ============================================================================
// API SERVICE - Complete with Error Handling
// ============================================================================

const API_DELAYS = {
  FAST: 300,
  NORMAL: 600,
  SLOW: 1000,
  ERROR: 100
} as const;

interface ApiResponse<T> {
  data: T;
  error?: string;
  timestamp: number;
  status: 'success' | 'error';
}

class ApiService {
  private requestCount = 0;
  private shouldSimulateError = false;
  private errorRate = 0; // 0-1, probability of error

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private simulateNetworkConditions(): void {
    // Simulate occasional network issues
    if (Math.random() < this.errorRate) {
      throw new Error('Network request failed');
    }
  }

  public setErrorSimulation(shouldError: boolean, rate: number = 0.1): void {
    this.shouldSimulateError = shouldError;
    this.errorRate = Math.max(0, Math.min(1, rate));
  }

  public getRequestCount(): number {
    return this.requestCount;
  }

  public resetRequestCount(): void {
    this.requestCount = 0;
  }

  // ========== AGENT METHODS ==========

  async fetchAgents(): Promise<ApiResponse<Agent[]>> {
    this.requestCount++;
    await this.delay(API_DELAYS.NORMAL);
    
    try {
      if (this.shouldSimulateError) {
        this.simulateNetworkConditions();
      }
      
      return {
        data: [...mockAgents],
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        data: [],
        error: error instanceof Error ? error.message : 'Failed to fetch agents',
        status: 'error',
        timestamp: Date.now()
      };
    }
  }

  async fetchAgentById(agentId: string): Promise<ApiResponse<Agent | null>> {
    this.requestCount++;
    await this.delay(API_DELAYS.FAST);
    
    try {
      const agent = mockAgents.find(a => a.id === agentId);
      
      if (!agent) {
        throw new Error('Agent not found');
      }
      
      return {
        data: agent,
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        data: null,
        error: error instanceof Error ? error.message : 'Failed to fetch agent',
        status: 'error',
        timestamp: Date.now()
      };
    }
  }

  async updateAgentStatus(agentId: string, status: Agent['status']): Promise<ApiResponse<Agent>> {
    this.requestCount++;
    await this.delay(API_DELAYS.NORMAL);
    
    try {
      const agentIndex = mockAgents.findIndex(a => a.id === agentId);
      
      if (agentIndex === -1) {
        throw new Error('Agent not found');
      }
      
      mockAgents[agentIndex] = {
        ...mockAgents[agentIndex],
        status
      };
      
      return {
        data: mockAgents[agentIndex],
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Failed to update agent status');
    }
  }

  async createAgent(agent: Omit<Agent, 'id' | 'tasksCompleted'>): Promise<ApiResponse<Agent>> {
    this.requestCount++;
    await this.delay(API_DELAYS.SLOW);
    
    try {
      const newAgent: Agent = {
        ...agent,
        id: `agent-${Date.now()}`,
        tasksCompleted: 0
      };
      
      mockAgents.push(newAgent);
      
      return {
        data: newAgent,
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Failed to create agent');
    }
  }

  // ========== PROJECT METHODS ==========

  async fetchProjects(): Promise<ApiResponse<Project[]>> {
    this.requestCount++;
    await this.delay(API_DELAYS.SLOW);
    
    try {
      if (this.shouldSimulateError) {
        this.simulateNetworkConditions();
      }
      
      return {
        data: [...mockProjects],
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        data: [],
        error: error instanceof Error ? error.message : 'Failed to fetch projects',
        status: 'error',
        timestamp: Date.now()
      };
    }
  }

  async fetchProjectById(projectId: string): Promise<ApiResponse<Project | null>> {
    this.requestCount++;
    await this.delay(API_DELAYS.FAST);
    
    try {
      const project = mockProjects.find(p => p.id === projectId);
      
      if (!project) {
        throw new Error('Project not found');
      }
      
      return {
        data: project,
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        data: null,
        error: error instanceof Error ? error.message : 'Failed to fetch project',
        status: 'error',
        timestamp: Date.now()
      };
    }
  }

  async createProject(project: Omit<Project, 'id' | 'progress' | 'performance'>): Promise<ApiResponse<Project>> {
    this.requestCount++;
    await this.delay(API_DELAYS.SLOW);
    
    try {
      const newProject: Project = {
        ...project,
        id: `proj-${Date.now()}`,
        progress: 0,
        performance: {
          tasksCompleted: 0,
          bugsFixed: 0,
          codeReviews: 0,
          efficiency: 0
        }
      };
      
      mockProjects.push(newProject);
      
      return {
        data: newProject,
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Failed to create project');
    }
  }

  async updateProjectProgress(projectId: string, progress: number): Promise<ApiResponse<Project>> {
    this.requestCount++;
    await this.delay(API_DELAYS.NORMAL);
    
    try {
      const projectIndex = mockProjects.findIndex(p => p.id === projectId);
      
      if (projectIndex === -1) {
        throw new Error('Project not found');
      }
      
      mockProjects[projectIndex] = {
        ...mockProjects[projectIndex],
        progress: Math.max(0, Math.min(100, progress))
      };
      
      return {
        data: mockProjects[projectIndex],
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Failed to update project progress');
    }
  }

  // ========== USER METHODS ==========

  async fetchUser(): Promise<ApiResponse<User>> {
    this.requestCount++;
    await this.delay(API_DELAYS.FAST);
    
    try {
      if (this.shouldSimulateError) {
        this.simulateNetworkConditions();
      }
      
      return {
        data: { ...mockUser },
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        data: mockUser, // Return default user on error
        error: error instanceof Error ? error.message : 'Failed to fetch user',
        status: 'error',
        timestamp: Date.now()
      };
    }
  }

  async updateUser(updates: Partial<User>): Promise<ApiResponse<User>> {
    this.requestCount++;
    await this.delay(API_DELAYS.NORMAL);
    
    try {
      Object.assign(mockUser, updates);
      
      return {
        data: { ...mockUser },
        status: 'success',
        timestamp: Date.now()
      };
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'Failed to update user');
    }
  }

  // ========== UTILITY METHODS ==========

  async healthCheck(): Promise<ApiResponse<{ healthy: boolean; version: string }>> {
    await this.delay(API_DELAYS.FAST);
    
    return {
      data: {
        healthy: true,
        version: '1.0.0'
      },
      status: 'success',
      timestamp: Date.now()
    };
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Export named functions for backward compatibility
export const fetchAgents = () => apiService.fetchAgents();
export const fetchProjects = () => apiService.fetchProjects();
export const fetchUser = () => apiService.fetchUser();
export const updateAgentStatus = (agentId: string, status: Agent['status']) => 
  apiService.updateAgentStatus(agentId, status);
export const createProject = (project: Omit<Project, 'id' | 'progress' | 'performance'>) => 
  apiService.createProject(project);