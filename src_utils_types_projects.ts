export interface Milestone {
  name: string;
  date: string;
  completed: boolean;
}

export interface Performance {
  tasksCompleted: number;
  bugsFixed: number;
  codeReviews: number;
  efficiency: number;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  status: 'completed' | 'in_progress' | 'cancelled' | 'on_hold';
  priority: 'high' | 'medium' | 'low';
  startDate: string;
  endDate: string | null;
  duration: string;
  progress: number;
  budget: number;
  spent: number;
  team: string[];
  technologies: string[];
  milestones: Milestone[];
  performance: Performance;
}
