export interface Agent {
  id: string;
  name: string;
  type: 'code_analyzer' | 'ui_designer' | 'backend_developer' | 'testing_agent' | 'optimization_agent' | 'security_agent' | 'data_scientist' | 'devops_engineer';
  status: 'idle' | 'thinking' | 'working' | 'completed' | 'error';
  description: string;
  tasksCompleted: number;
  efficiency: string;
  size?: number;
  detail?: number;
  currentPhase?: string | null;
}
