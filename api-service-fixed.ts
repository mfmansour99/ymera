import { Agent, Project, User } from './types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

// Enhanced mock data with proper types
const mockAgents: Agent[] = [
  {
    id: 'agent-001',
    name: 'CodeAnalyzer Alpha',
    type: 'code_analyzer',
    status: 'working',
    description: 'Advanced code analysis and optimization agent',
    tasksCompleted: 42,
    efficiency: '92%',
    currentPhase: 'code_analysis'
  },
  {
    id: 'agent-002',
    name: 'UIDesigner Beta',
    type: 'ui_designer',
    status: 'idle',
    description: 'Creates beautiful and functional user interfaces',
    tasksCompleted: 28,
    efficiency: '88%',
    currentPhase: null
  },
  {
    id: 'agent-003',
    name: 'BackendMaster',
    type: 'backend_developer',
    status: 'thinking',
    description