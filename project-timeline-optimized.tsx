import React, { useState, useMemo, useCallback, memo } from 'react';
import { Calendar, Clock, DollarSign, Users, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';

// ============================================================================
// TYPES
// ============================================================================

interface Project {
  id: string;
  name: string;
  description: string;
  status: 'planning' | 'in_progress' | 'completed' | 'cancelled' | 'on_hold';
  priority?: 'low' | 'medium' | 'high' | 'critical';
  startDate: string;
  endDate: string | null;
  duration: string;
  progress: number;
  budget: number;
  spent: number;
  team: string[];
  technologies?: string[];
  milestones: Array<{ name: string; date: string; completed: boolean }>;
  performance: {
    tasksCompleted: number;
    bugsFixed: number;
    codeReviews: number;
    efficiency: number;
  };
}

interface ProjectTimelineProps {
  projects: Project[];
  onProjectSelect: (project: Project) => void;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const getStatusColor = (status: Project['status']) => {
  const colors = {
    completed: '#10b981',
    in_progress: '#06b6d4',
    cancelled: '#ef4444',
    on_hold: '#fbbf24',
    planning: '#8b5cf6'
  };
  return colors[status] || '#64748b';
};

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
  }).format(amount);
};

// ============================================================================
// PROJECT CARD COMPONENT (Memoized for Performance)
// ============================================================================

const ProjectCard = memo<{
  project: Project;
  isExpanded: boolean;
  onToggle: () => void;
  index: number;
}>(({ project, isExpanded, onToggle, index }) => {
  const statusColor = useMemo(() => getStatusColor(project.status), [project.status]);
  const budgetPercentage = useMemo(() => 
    Math.min(100, (project.spent / project.budget) * 100), 
    [project.spent, project.budget]
  );

  return (
    <div style={{ position: 'relative' }}>
      {/* Timeline connector line */}
      {index > 0 && (
        <div style={{
          position: 'absolute',
          left: '32px',
          top: '-24px',
          width: '2px',
          height: '24px',
          background: 'rgba(255,255,255,0.1)'
        }} />
      )}

      <div style={{ position: 'relative', display: 'flex', alignItems: 'start', gap: '16px' }}>
        {/* Timeline dot */}
        <div style={{
          position: 'relative',
          zIndex: 10,
          minWidth: '64px',
          display: 'flex',
          justifyContent: 'center'
        }}>
          <div style={{
            width: '16px',
            height: '16px',
            borderRadius: '50%',
            border: `3px solid ${statusColor}`,
            background: '#0a0a0a',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 0 12px ${statusColor}40`
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: statusColor
            }} />
          </div>
        </div>

        {/* Project card */}
        <div
          onClick={onToggle}
          style={{
            flex: 1,
            backdropFilter: 'blur(16px)',
            background: isExpanded ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.05)',
            border: `1px solid ${isExpanded ? `${statusColor}40` : 'rgba(255,255,255,0.1)'}`,
            borderRadius: '12px',
            padding: '20px',
            cursor: 'pointer',
            transition: 'all 0.3s',
            boxShadow: isExpanded ? `0 8px 32px ${statusColor}20` : 'none'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateX(4px)';
            e.currentTarget.style.borderColor = `${statusColor}60`;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateX(0)';
            e.currentTarget.style.borderColor = isExpanded ? `${statusColor}40` : 'rgba(255,255,255,0.1)';
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ flex: 1 }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'white', marginBottom: '8px' }}>
                {project.name}
              </h3>
              {!isExpanded && (
                <p style={{ color: '#9ca3af', fontSize: '0.875rem', lineHeight: '1.5' }}>
                  {project.description.substring(0, 120)}...
                </p>
              )}
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginLeft: '16px' }}>
              <span style={{
                padding: '4px 12px',
                borderRadius: '999px',
                fontSize: '0.75rem',
                fontWeight: '600',
                background: `${statusColor}20`,
                color: statusColor,
                border: `1px solid ${statusColor}40`,
                textTransform: 'capitalize'
              }}>
                {project.status.replace('_', ' ')}
              </span>
              
              {isExpanded ? <ChevronUp size={20} color="#9ca3af" /> : <ChevronDown size={20} color="#9ca3af" />}
            </div>
          </div>

          {/* Quick stats (Always visible) */}
          {!isExpanded && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginTop: '16px' }}>
              {[
                { icon: TrendingUp, label: 'Progress', value: `${project.progress}%`, color: '#06b6d4' },
                { icon: Users, label: 'Team', value: project.team.length, color: '#8b5cf6' },
                { icon: Clock, label: 'Duration', value: project.duration, color: '#10b981' },
                { icon: DollarSign, label: 'Budget', value: `${Math.round(budgetPercentage)}%`, color: '#fbbf24' }
              ].map((stat, i) => (
                <div key={i} style={{ textAlign: 'center' }}>
                  <div style={{
                    width: '40px',
                    height: '40px',
                    margin: '0 auto 8px',
                    borderRadius: '8px',
                    background: `${stat.color}20`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <stat.icon size={20} style={{ color: stat.color }} />
                  </div>
                  <div style={{ fontSize: '1rem', fontWeight: '700', color: 'white' }}>{stat.value}</div>
                  <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>{stat.label}</div>
                </div>
              ))}
            </div>
          )}

          {/* Expanded content */}
          {isExpanded && (
            <div style={{ marginTop: '24px', paddingTop: '24px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
              {/* Full description */}
              <p style={{ color: '#d1d5db', marginBottom: '24px', lineHeight: '1.6' }}>
                {project.description}
              </p>

              {/* Detailed stats grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '24px' }}>
                {/* Timeline */}
                <div>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: 'white', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Calendar size={16} color="#06b6d4" />
                    Timeline
                  </h4>
                  <div style={{ display: 'grid', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                      <span style={{ color: '#9ca3af' }}>Started:</span>
                      <span style={{ color: 'white' }}>{formatDate(project.startDate)}</span>
                    </div>
                    {project.endDate && (
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                        <span style={{ color: '#9ca3af' }}>Completed:</span>
                        <span style={{ color: 'white' }}>{formatDate(project.endDate)}</span>
                      </div>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                      <span style={{ color: '#9ca3af' }}>Duration:</span>
                      <span style={{ color: 'white', fontWeight: '600' }}>{project.duration}</span>
                    </div>
                  </div>
                </div>

                {/* Budget */}
                <div>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: 'white', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <DollarSign size={16} color="#fbbf24" />
                    Budget
                  </h4>
                  <div style={{ display: 'grid', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                      <span style={{ color: '#9ca3af' }}>Total:</span>
                      <span style={{ color: 'white' }}>{formatCurrency(project.budget)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                      <span style={{ color: '#9ca3af' }}>Spent:</span>
                      <span style={{ color: 'white' }}>{formatCurrency(project.spent)}</span>
                    </div>
                    <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden', marginTop: '4px' }}>
                      <div style={{
                        height: '100%',
                        width: `${budgetPercentage}%`,
                        background: budgetPercentage > 90 ? '#ef4444' : budgetPercentage > 70 ? '#fbbf24' : '#10b981',
                        transition: 'width 0.3s'
                      }} />
                    </div>
                  </div>
                </div>

                {/* Team */}
                <div>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: 'white', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Users size={16} color="#8b5cf6" />
                    Team Members ({project.team.length})
                  </h4>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {project.team.map((member, i) => (
                      <div
                        key={i}
                        style={{
                          width: '36px',
                          height: '36px',
                          borderRadius: '8px',
                          background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.75rem',
                          fontWeight: '700',
                          color: 'white',
                          border: '2px solid rgba(255,255,255,0.2)',
                          cursor: 'pointer'
                        }}
                        title={member}
                      >
                        {member.split(' ').map(n => n[0]).join('')}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Progress bar */}
              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.875rem', fontWeight: '600', color: 'white' }}>Overall Progress</span>
                  <span style={{ fontSize: '0.875rem', fontWeight: '700', color: statusColor }}>{project.progress}%</span>
                </div>
                <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    width: `${project.progress}%`,
                    background: `linear-gradient(to right, ${statusColor}, ${statusColor}dd)`,
                    transition: 'width 0.5s',
                    boxShadow: `0 0 12px ${statusColor}60`
                  }} />
                </div>
              </div>

              {/* Milestones */}
              <div style={{ marginBottom: '24px' }}>
                <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: 'white', marginBottom: '12px' }}>
                  Milestones
                </h4>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {project.milestones.map((milestone, i) => (
                    <div key={i} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '12px',
                      background: 'rgba(255,255,255,0.03)',
                      borderRadius: '8px',
                      border: '1px solid rgba(255,255,255,0.05)'
                    }}>
                      <div style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        background: milestone.completed ? '#10b981' : '#6b7280',
                        boxShadow: milestone.completed ? '0 0 8px #10b98160' : 'none'
                      }} />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: '0.875rem', fontWeight: '500', color: 'white' }}>
                          {milestone.name}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>
                          {milestone.date}
                        </div>
                      </div>
                      <div style={{
                        fontSize: '0.75rem',
                        fontWeight: '600',
                        color: milestone.completed ? '#10b981' : '#6b7280'
                      }}>
                        {milestone.completed ? 'Completed' : 'Pending'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Performance metrics */}
              <div>
                <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: 'white', marginBottom: '12px' }}>
                  Performance Metrics
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px' }}>
                  {[
                    { label: 'Tasks', value: project.performance.tasksCompleted, color: '#06b6d4' },
                    { label: 'Bugs Fixed', value: project.performance.bugsFixed, color: '#10b981' },
                    { label: 'Reviews', value: project.performance.codeReviews, color: '#8b5cf6' },
                    { label: 'Efficiency', value: `${project.performance.efficiency}%`, color: '#fbbf24' }
                  ].map((metric, i) => (
                    <div key={i} style={{
                      padding: '12px',
                      background: `${metric.color}10`,
                      border: `1px solid ${metric.color}30`,
                      borderRadius: '8px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '1.5rem', fontWeight: '800', color: metric.color }}>
                        {metric.value}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: '4px' }}>
                        {metric.label}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Technologies (if available) */}
              {project.technologies && project.technologies.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', color: 'white', marginBottom: '12px' }}>
                    Technologies
                  </h4>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {project.technologies.map((tech, i) => (
                      <span
                        key={i}
                        style={{
                          padding: '4px 12px',
                          background: 'rgba(6,182,212,0.1)',
                          border: '1px solid rgba(6,182,212,0.3)',
                          borderRadius: '6px',
                          fontSize: '0.75rem',
                          color: '#06b6d4',
                          fontWeight: '500'
                        }}
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

ProjectCard.displayName = 'ProjectCard';

// ============================================================================
// PROJECT TIMELINE COMPONENT
// ============================================================================

export const ProjectTimeline: React.FC<ProjectTimelineProps> = ({ projects, onProjectSelect }) => {
  const [expandedProject, setExpandedProject] = useState<string | null>(null);

  const handleToggle = useCallback((projectId: string) => {
    setExpandedProject(prev => prev === projectId ? null : projectId);
    const project = projects.find(p => p.id === projectId);
    if (project) {
      onProjectSelect(project);
    }
  }, [projects, onProjectSelect]);

  // Sort projects by date (newest first)
  const sortedProjects = useMemo(() => 
    [...projects].sort((a, b) => 
      new Date(b.startDate).getTime() - new Date(a.startDate).getTime()
    ), [projects]
  );

  if (projects.length === 0) {
    return (
      <div style={{
        textAlign: 'center',
        padding: '4rem 2rem',
        backdropFilter: 'blur(16px)',
        background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '12px'
      }}>
        <div style={{
          width: '64px',
          height: '64px',
          margin: '0 auto 16px',
          borderRadius: '12px',
          background: 'rgba(6,182,212,0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Calendar size={32} color="#06b6d4" />
        </div>
        <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: 'white', marginBottom: '8px' }}>
          No Projects Yet
        </h3>
        <p style={{ color: '#9ca3af' }}>
          Start by creating your first project
        </p>
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      {/* Timeline vertical line */}
      <div style={{
        position: 'absolute',
        left: '32px',
        top: '0',
        bottom: '0',
        width: '2px',
        background: 'linear-gradient(to bottom, rgba(6,182,212,0.3), rgba(139,92,246,0.3), rgba(16,185,129,0.3))',
        opacity: 0.3
      }} />

      {/* Projects list */}
      <div style={{ display: 'grid', gap: '24px' }}>
        {sortedProjects.map((project, index) => (
          <ProjectCard
            key={project.id}
            project={project}
            isExpanded={expandedProject === project.id}
            onToggle={() => handleToggle(project.id)}
            index={index}
          />
        ))}
      </div>

      {/* Performance note for large datasets */}
      {projects.length > 50 && (
        <div style={{
          marginTop: '24px',
          padding: '12px 16px',
          background: 'rgba(251,191,36,0.1)',
          border: '1px solid rgba(251,191,36,0.3)',
          borderRadius: '8px',
          fontSize: '0.875rem',
          color: '#fbbf24',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <TrendingUp size={16} />
          <span>
            Showing {projects.length} projects. Consider using filters for better performance with 100+ items.
          </span>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// DEMO WITH SAMPLE DATA
// ============================================================================

export default function App() {
  const [projects] = useState<Project[]>([
    {
      id: 'proj-001',
      name: 'E-Commerce Platform',
      description: 'Complete e-commerce solution with payment processing, inventory management, and customer analytics dashboard',
      status: 'in_progress',
      priority: 'high',
      startDate: '2024-01-15',
      endDate: null,
      duration: '3 months',
      progress: 65,
      budget: 120000,
      spent: 78000,
      team: ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Williams'],
      technologies: ['React', 'Node.js', 'MongoDB', 'AWS', 'Stripe'],
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
      description: 'Cross-platform mobile banking application with biometric authentication and real-time transactions',
      status: 'completed',
      priority: 'high',
      startDate: '2023-11-01',
      endDate: '2024-02-28',
      duration: '4 months',
      progress: 100,
      budget: 150000,
      spent: 145000,
      team: ['Sarah Connor', 'Alex Murphy', 'John Connor'],
      technologies: ['Flutter', 'Dart', 'Firebase', 'Stripe'],
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
      status: 'planning',
      priority: 'critical',
      startDate: '2024-02-01',
      endDate: null,
      duration: '6 months',
      progress: 15,
      budget: 250000,
      spent: 37500,
      team: ['Dr. Emily Chen', 'Robert Martinez'],
      technologies: ['React', 'TypeScript', 'PostgreSQL', 'GraphQL'],
      milestones: [
        { name: 'Compliance Review', date: '2024-02-20', completed: true },
        { name: 'System Architecture', date: '2024-03-10', completed: false },
        { name: 'Patient Portal', date: '2024-04-15', completed: false },
        { name: 'Doctor Dashboard', date: '2024-05-20', completed: false },
      ],
      performance: {
        tasksCompleted: 23,
        bugsFixed: 3,
        codeReviews: 8,
        efficiency: 78.5,
      },
    },
  ]);

  const handleProjectSelect = (project: Project) => {
    console.log('Selected project:', project.name);
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0a0a0a',
      padding: '4rem 2rem'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ marginBottom: '3rem' }}>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: '800',
            color: 'white',
            marginBottom: '0.5rem'
          }}>
            Project Timeline
          </h1>
          <p style={{ color: '#9ca3af', fontSize: '1.125rem' }}>
            Track your projects from start to finish
          </p>
        </div>

        <ProjectTimeline
          projects={projects}
          onProjectSelect={handleProjectSelect}
        />
      </div>
    </div>
  );
}