import React, { memo, useMemo } from 'react';
import { Project, Agent } from '../types';

interface PhaseDetailPanelProps {
  phase: string;
  project: Project;
  agents: Agent[];
  onClose: () => void;
}

export const PhaseDetailPanel: React.FC<PhaseDetailPanelProps> = memo(({
  phase,
  project,
  agents,
  onClose
}) => {
  // Memoize phase details to prevent recalculation
  const phaseDetails = useMemo(() => {
    const phases: Record<string, {
      title: string;
      description: string;
      tasks: string[];
      agents: string[];
      duration: string;
      status: string;
    }> = {
      planning: {
        title: 'Project Planning',
        description: 'Initial requirements gathering, scope definition, and resource allocation for the project.',
        tasks: [
          'Stakeholder interviews',
          'Requirements documentation',
          'Scope definition',
          'Resource allocation',
          'Timeline estimation'
        ],
        agents: ['project_manager', 'business_analyst'],
        duration: '2 weeks',
        status: 'completed'
      },
      design: {
        title: 'System Design',
        description: 'Architectural design, UI/UX wireframing, and technical specifications.',
        tasks: [
          'Architecture design',
          'UI/UX wireframes',
          'Database schema',
          'API specifications',
          'Design review'
        ],
        agents: ['ui_designer', 'backend_developer', 'architecture_expert'],
        duration: '3 weeks',
        status: project.status === 'completed' || project.status === 'in_progress' ? 'completed' : 'pending'
      },
      development: {
        title: 'Development Phase',
        description: 'Implementation of all features according to the design specifications.',
        tasks: [
          'Frontend development',
          'Backend services',
          'Database implementation',
          'API integration',
          'Feature testing'
        ],
        agents: ['frontend_developer', 'backend_developer', 'fullstack_developer'],
        duration: '6 weeks',
        status: project.status === 'completed' ? 'completed' : 'in_progress'
      },
      testing: {
        title: 'Testing & QA',
        description: 'Comprehensive testing including unit tests, integration tests, and user acceptance testing.',
        tasks: [
          'Unit testing',
          'Integration testing',
          'Performance testing',
          'Security testing',
          'User acceptance testing'
        ],
        agents: ['qa_engineer', 'testing_agent', 'security_expert'],
        duration: '2 weeks',
        status: project.status === 'completed' ? 'completed' : 'pending'
      },
      deployment: {
        title: 'Deployment',
        description: 'Final deployment to production environment and post-deployment monitoring.',
        tasks: [
          'Environment setup',
          'Deployment execution',
          'Monitoring setup',
          'Performance tuning',
          'Final documentation'
        ],
        agents: ['devops_engineer', 'system_admin', 'release_manager'],
        duration: '1 week',
        status: project.status === 'completed' ? 'completed' : 'pending'
      }
    };

    return phases[phase] || {
      title: phase.charAt(0).toUpperCase() + phase.slice(1),
      description: `Details about the ${phase} phase`,
      tasks: [],
      agents: [],
      duration: 'TBD',
      status: 'unknown'
    };
  }, [phase, project.status]);

  // Memoize filtered agents
  const phaseAgents = useMemo(() => 
    agents.filter(agent =>
      phaseDetails.agents.some(agentType =>
        agent.type.toLowerCase().includes(agentType.replace('_', ''))
      )
    ), [agents, phaseDetails.agents]
  );

  // Dynamic icon mapper
  const getAgentIcon = useMemo(() => (type: string): string => {
    const iconMap: Record<string, string> = {
      'code_analyzer': 'fa-code',
      'ui_designer': 'fa-paint-brush',
      'backend_developer': 'fa-server',
      'testing_agent': 'fa-vial',
      'security_expert': 'fa-shield-alt',
      'devops_engineer': 'fa-cogs',
      'frontend_developer': 'fa-laptop-code',
      'fullstack_developer': 'fa-layer-group',
      'project_manager': 'fa-tasks',
      'business_analyst': 'fa-chart-line',
      'architecture_expert': 'fa-building',
      'qa_engineer': 'fa-check-double',
      'system_admin': 'fa-server',
      'release_manager': 'fa-rocket'
    };
    return iconMap[type] || 'fa-cog';
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-accent-success';
      case 'in_progress':
        return 'text-accent-primary';
      case 'pending':
        return 'text-accent-warning';
      default:
        return 'text-secondary';
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden" role="dialog" aria-modal="true">
      <div 
        className="absolute inset-0 bg-black bg-opacity-70 backdrop-blur-sm" 
        onClick={onClose}
        aria-hidden="true"
      />

      <div className="absolute right-0 top-0 bottom-0 w-full md:w-1/2 lg:w-1/3 bg-primary-bg bg-opacity-95 backdrop-blur-lg border-l border-glass-border overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold capitalize">{phaseDetails.title}</h2>
              <p className="text-secondary capitalize">{project.name}</p>
            </div>
            <button
              onClick={onClose}
              className="glass-button w-10 h-10 flex items-center justify-center"
              aria-label="Close panel"
            >
              <i className="fas fa-times" />
            </button>
          </div>

          <div className="glass-card mb-6">
            <div className="p-4">
              <h3 className="font-medium mb-3">Phase Overview</h3>
              <p className="text-secondary mb-4">{phaseDetails.description}</p>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-secondary mb-1">Status</div>
                  <div className={`font-medium capitalize ${getStatusColor(phaseDetails.status)}`}>
                    {phaseDetails.status.replace('_', ' ')}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-secondary mb-1">Duration</div>
                  <div className="font-medium">{phaseDetails.duration}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-card mb-6">
            <div className="p-4">
              <h3 className="font-medium mb-3">Tasks</h3>
              {phaseDetails.tasks.length > 0 ? (
                <ul className="space-y-2" role="list">
                  {phaseDetails.tasks.map((task, index) => (
                    <li key={index} className="flex items-center">
                      <i className="fas fa-circle text-xs text-secondary mr-2" aria-hidden="true" />
                      <span>{task}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-secondary">No tasks defined for this phase</p>
              )}
            </div>
          </div>

          <div className="glass-card mb-6">
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium">Assigned Agents</h3>
                <span className="text-sm text-secondary">
                  {phaseAgents.length} of {phaseDetails.agents.length} types
                </span>
              </div>

              {phaseAgents.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {phaseAgents.map((agent) => (
                    <div
                      key={agent.id}
                      className="glass p-3 rounded-lg hover:bg-opacity-30 transition-all cursor-pointer"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary flex items-center justify-center">
                          <i className={`fas ${getAgentIcon(agent.type)} text-white text-xs`} />
                        </div>
                        <div className="flex-1">
                          <div className="font-medium">{agent.name}</div>
                          <div className="text-xs text-secondary capitalize">
                            {agent.type.replace('_', ' ')}
                          </div>
                        </div>
                        <div className={`text-xs ${agent.status === 'working' ? 'text-accent-primary' :
                                        agent.status === 'error' ? 'text-accent-danger' :
                                        'text-secondary'}`}>
                          {agent.status}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-secondary">No agents assigned to this phase</p>
              )}
            </div>
          </div>

          <div className="glass-card">
            <div className="p-4">
              <h3 className="font-medium mb-3">Phase Metrics</h3>

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-accent-primary font-bold text-lg">85%</div>
                  <div className="text-xs text-secondary">Completion</div>
                </div>
                <div className="text-center">
                  <div className="text-accent-success font-bold text-lg">3</div>
                  <div className="text-xs text-secondary">Tasks Done</div>
                </div>
                <div className="text-center">
                  <div className="text-accent-warning font-bold text-lg">2</div>
                  <div className="text-xs text-secondary">Pending</div>
                </div>
                <div className="text-center">
                  <div className="text-accent-danger font-bold text-lg">1</div>
                  <div className="text-xs text-secondary">Blockers</div>
                </div>
              </div>

              <div className="mt-4">
                <div className="text-sm text-secondary mb-1">Progress</div>
                <div className="w-full bg-glass rounded-full h-2">
                  <div className="bg-gradient-to-r from-accent-primary to-accent-secondary h-2 rounded-full w-5/6" />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-6 flex justify-end space-x-3">
            <button className="glass-button bg-accent-primary bg-opacity-20 border-accent-primary">
              <i className="fas fa-play mr-2" />
              Start Phase
            </button>
            <button className="glass-button bg-accent-success bg-opacity-20 border-accent-success">
              <i className="fas fa-check mr-2" />
              Complete Phase
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

PhaseDetailPanel.displayName = 'PhaseDetailPanel';