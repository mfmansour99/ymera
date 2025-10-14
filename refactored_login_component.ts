// ============================================================================
// REFACTORED LOGIN COMPONENT - Secure & Accessible
// ============================================================================

import React, { useState, useCallback, useEffect } from 'react';
import { Eye, EyeOff, AlertCircle, CheckCircle } from 'lucide-react';
import { 
  validatePassword, 
  sanitizeEmail, 
  sanitizeInput,
  PasswordStrength 
} from '../utils/validators';
import { bruteForceProtection } from '../utils/security';
import { csrfService } from '../utils/csrf';
import { AccessibleInput, AccessibleButton, AccessibleModal } from '../components/accessibility';
import { useAriaAnnounce } from '../hooks/useAccessibility';

interface LoginState {
  email: string;
  password: string;
  showPassword: boolean;
  isLoading: boolean;
  error: string | null;
  passwordStrength: PasswordStrength | null;
  sessionId: string;
  csrfToken: string;
}

export const SecureLoginPage: React.FC<{ onLogin: (email: string) => void }> = ({ onLogin }) => {
  const [state, setState] = useState<LoginState>({
    email: '',
    password: '',
    showPassword: false,
    isLoading: false,
    error: null,
    passwordStrength: null,
    sessionId: Math.random().toString(36).substr(2, 9),
    csrfToken: '',
  });

  const announce = useAriaAnnounce();

  // Generate CSRF token on mount
  useEffect(() => {
    const token = csrfService.generateToken(state.sessionId);
    setState(prev => ({ ...prev, csrfToken: token }));
  }, [state.sessionId]);

  const handleEmailChange = useCallback((value: string) => {
    const sanitized = sanitizeEmail(value);
    setState(prev => ({ ...prev, email: sanitized, error: null }));
  }, []);

  const handlePasswordChange = useCallback((value: string) => {
    const sanitized = sanitizeInput(value, { maxLength: 128 });
    const strength = validatePassword(sanitized);
    setState(prev => ({
      ...prev,
      password: sanitized,
      passwordStrength: strength,
      error: null,
    }));
  }, []);

  const validateForm = useCallback((): boolean => {
    const errors: string[] = [];

    // Validate email
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(state.email)) {
      errors.push('Valid email address required');
    }

    // Validate password
    if (!state.passwordStrength?.isValid) {
      errors.push('Password does not meet security requirements');
    }

    if (errors.length > 0) {
      setState(prev => ({ ...prev, error: errors[0] }));
      announce(errors[0], 'assertive');
      return false;
    }

    return true;
  }, [state.email, state.passwordStrength, announce]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      if (!validateForm()) return;

      // Check brute force protection
      const bruteForceCheck = bruteForceProtection.recordAttempt(state.email);
      if (!bruteForceCheck.allowed) {
        const minutes = Math.ceil(bruteForceCheck.remainingMs! / 60000);
        const error = `Account temporarily locked. Try again in ${minutes} minutes.`;
        setState(prev => ({ ...prev, error }));
        announce(error, 'assertive');
        return;
      }

      setState(prev => ({ ...prev, isLoading: true, error: null }));
      announce('Logging in. Please wait.', 'polite');

      try {
        // Verify CSRF token
        const isValidToken = csrfService.verifyToken(state.csrfToken, state.sessionId);
        if (!isValidToken) {
          throw new Error('Security validation failed. Please refresh and try again.');
        }

        // Simulate API call with CSRF validation
        await new Promise(resolve => setTimeout(resolve, 1500));

        // Reset brute force counter on success
        bruteForceProtection.reset(state.email);

        announce('Login successful. Redirecting...', 'polite');
        onLogin(state.email);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Login failed';
        setState(prev => ({ ...prev, error: errorMessage, isLoading: false }));
        announce(`Login failed: ${errorMessage}`, 'assertive');
      }
    },
    [state, validateForm, announce, onLogin]
  );

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1e3a8a 50%, #581c87 100%)',
        padding: '1rem',
      }}
    >
      <form
        onSubmit={handleSubmit}
        noValidate
        style={{
          width: '100%',
          maxWidth: '450px',
          backdropFilter: 'blur(16px)',
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '16px',
          padding: '2rem',
        }}
      >
        <h1
          style={{
            fontSize: '28px',
            fontWeight: 'bold',
            color: 'white',
            marginBottom: '8px',
            textAlign: 'center',
          }}
        >
          Secure Login
        </h1>
        <p
          style={{
            textAlign: 'center',
            color: '#9ca3af',
            marginBottom: '2rem',
          }}
        >
          AI Project Management Platform
        </p>

        {state.error && (
          <div
            role="alert"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '12px',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: '8px',
              color: '#ef4444',
              marginBottom: '16px',
            }}
          >
            <AlertCircle size={16} aria-hidden="true" />
            {state.error}
          </div>
        )}

        <AccessibleInput
          label="Email Address"
          value={state.email}
          onChange={handleEmailChange}
          type="email"
          placeholder="you@example.com"
          required
          error={
            state.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(state.email)
              ? 'Invalid email format'
              : undefined
          }
          helpText="We'll never share your email"
        />

        <div style={{ marginBottom: '16px' }}>
          <label
            htmlFor="password"
            style={{
              display: 'block',
              fontSize: '14px',
              fontWeight: '500',
              marginBottom: '8px',
              color: '#d1d5db',
            }}
          >
            Password <span aria-label="required">*</span>
          </label>

          <div style={{ position: 'relative' }}>
            <input
              id="password"
              type={state.showPassword ? 'text' : 'password'}
              value={state.password}
              onChange={(e) => handlePasswordChange(e.target.value)}
              placeholder="Enter strong password"
              required
              aria-invalid={state.passwordStrength && !state.passwordStrength.isValid}
              style={{
                width: '100%',
                paddingRight: '48px',
                paddingLeft: '12px',
                paddingTop: '12px',
                paddingBottom: '12px',
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                color: 'white',
                outline: 'none',
              }}
            />
            <button
              type="button"
              onClick={() => setState(prev => ({ ...prev, showPassword: !prev.showPassword }))}
              aria-label={state.showPassword ? 'Hide password' : 'Show password'}
              style={{
                position: 'absolute',
                right: '12px',
                top: '12px',
                background: 'none',
                border: 'none',
                color: '#9ca3af',
                cursor: 'pointer',
                padding: '4px',
              }}
            >
              {state.showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>

          {state.passwordStrength && (
            <div style={{ marginTop: '12px' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: '8px',
                  fontSize: '14px',
                }}
              >
                <span style={{ color: '#9ca3af' }}>Password Strength</span>
                <span
                  style={{
                    color:
                      state.passwordStrength.score === 0 ? '#ef4444' :
                      state.passwordStrength.score === 1 ? '#f59e0b' :
                      state.passwordStrength.score === 2 ? '#eab308' :
                      state.passwordStrength.score === 3 ? '#84cc16' :
                      '#10b981',
                    fontWeight: '600',
                  }}
                >
                  {state.passwordStrength.score === 0 ? 'Invalid' :
                   state.passwordStrength.score === 1 ? 'Weak' :
                   state.passwordStrength.score === 2 ? 'Fair' :
                   state.passwordStrength.score === 3 ? 'Good' :
                   'Strong'}
                </span>
              </div>

              <div
                style={{
                  height: '6px',
                  background: 'rgba(255,255,255,0.1)',
                  borderRadius: '3px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${(state.passwordStrength.score / 4) * 100}%`,
                    background:
                      state.passwordStrength.score === 0 ? '#ef4444' :
                      state.passwordStrength.score === 1 ? '#f59e0b' :
                      state.passwordStrength.score === 2 ? '#eab308' :
                      state.passwordStrength.score === 3 ? '#84cc16' :
                      '#10b981',
                    transition: 'width 0.2s',
                  }}
                />
              </div>

              {state.passwordStrength.feedback.length > 0 && (
                <ul
                  style={{
                    marginTop: '8px',
                    paddingLeft: '20px',
                    fontSize: '12px',
                    color: '#9ca3af',
                    margin: '0',
                  }}
                >
                  {state.passwordStrength.feedback.map((fb, i) => (
                    <li key={i} style={{ marginTop: i > 0 ? '4px' : '0' }}>
                      {fb}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>

        <AccessibleButton
          label={state.isLoading ? 'Signing in...' : 'Sign In'}
          onClick={() => {}} // Handled by form submission
          disabled={state.isLoading || !state.passwordStrength?.isValid}
          loading={state.isLoading}
          variant="primary"
          size="md"
        />

        <div
          style={{
            marginTop: '16px',
            paddingTop: '16px',
            borderTop: '1px solid rgba(255,255,255,0.1)',
            fontSize: '12px',
            color: '#9ca3af',
            textAlign: 'center',
          }}
        >
          <p style={{ margin: '0 0 8px 0' }}>Demo Mode - Enter any valid email and strong password</p>
          <a
            href="#forgot"
            style={{
              color: '#06b6d4',
              textDecoration: 'none',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.textDecoration = 'underline')}
            onMouseLeave={(e) => (e.currentTarget.style.textDecoration = 'none')}
          >
            Forgot your password?
          </a>
        </div>

        {/* Hidden CSRF token field */}
        <input type="hidden" name="csrf_token" value={state.csrfToken} />
      </form>
    </div>
  );
};

// ============================================================================
// REFACTORED AGENTS PAGE - With Virtualization & Error Boundary
// ============================================================================

import { useMemo, useState, useCallback, memo } from 'react';
import { FixedSizeList } from 'react-window';

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'idle' | 'working' | 'thinking';
  description: string;
  efficiency: string;
}

const AgentCard = memo(
  ({
    agent,
    isSelected,
    onClick,
  }: {
    agent: Agent;
    isSelected: boolean;
    onClick: () => void;
  }) => (
    <button
      onClick={onClick}
      aria-selected={isSelected}
      aria-label={`Select agent ${agent.name}, status ${agent.status}`}
      style={{
        width: '100%',
        padding: '16px',
        background: isSelected ? 'rgba(6,182,212,0.2)' : 'rgba(255,255,255,0.05)',
        border: `1px solid ${isSelected ? 'rgba(6,182,212,0.4)' : 'rgba(255,255,255,0.1)'}`,
        borderRadius: '12px',
        color: 'white',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.2s',
        outline: isSelected ? '2px solid #06b6d4' : 'none',
        outlineOffset: '2px',
      }}
      onMouseEnter={(e) => {
        if (!isSelected) {
          (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.08)';
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          (e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,255,255,0.05)';
        }
      }}
    >
      <h3 style={{ margin: '0 0 8px 0', fontSize: '18px', fontWeight: '700' }}>
        {agent.name}
      </h3>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
        <span
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background:
              agent.status === 'working' ? '#10b981' :
              agent.status === 'thinking' ? '#f59e0b' :
              '#6b7280',
          }}
          aria-hidden="true"
        />
        <span style={{ fontSize: '14px', textTransform: 'capitalize', color: '#9ca3af' }}>
          {agent.status}
        </span>
      </div>
      <p style={{ margin: '0', fontSize: '14px', color: '#9ca3af' }}>
        {agent.description}
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '12px' }}>
        <div style={{ fontSize: '12px' }}>
          <div style={{ color: '#06b6d4', fontWeight: '700' }}>{agent.type}</div>
          <div style={{ color: '#9ca3af' }}>Type</div>
        </div>
        <div style={{ fontSize: '12px' }}>
          <div style={{ color: '#10b981', fontWeight: '700' }}>{agent.efficiency}</div>
          <div style={{ color: '#9ca3af' }}>Efficiency</div>
        </div>
      </div>
    </button>
  )
);

AgentCard.displayName = 'AgentCard';

export const RefactoredAgentsPage: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'idle' | 'working' | 'thinking'>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Mock agents data - replace with actual API call
  const agents: Agent[] = useMemo(() => [
    {
      id: 'agent-001',
      name: 'CodeAnalyzer Alpha',
      type: 'code_analyzer',
      status: 'working',
      description: 'Advanced code analysis and optimization agent',
      efficiency: '92%',
    },
    {
      id: 'agent-002',
      name: 'UIDesigner Beta',
      type: 'ui_designer',
      status: 'idle',
      description: 'Creates beautiful and functional user interfaces',
      efficiency: '88%',
    },
    {
      id: 'agent-003',
      name: 'BackendMaster',
      type: 'backend_developer',
      status: 'thinking',
      description: 'Builds robust backend systems and APIs',
      efficiency: '95%',
    },
  ], []);

  // Memoized filtered agents
  const filteredAgents = useMemo(() => {
    return agents.filter((agent) => {
      const matchesSearch =
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        agent.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = filterStatus === 'all' || agent.status === filterStatus;
      return matchesSearch && matchesStatus;
    });
  }, [agents, searchTerm, filterStatus]);

  const handleSelectAgent = useCallback((agentId: string) => {
    setSelectedAgent(agentId);
  }, []);

  return (
    <div style={{ padding: '2rem 0', minHeight: '100vh' }}>
      <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: '8px' }}>
        AI Agents
      </h1>
      <p style={{ color: '#9ca3af', marginBottom: '2rem' }}>
        Manage and monitor your intelligent agents
      </p>

      {/* Search and filter controls */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '12px', marginBottom: '2rem' }}>
        <div style={{ position: 'relative' }}>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search agents..."
            aria-label="Search agents"
            style={{
              width: '100%',
              padding: '12px 12px 12px 36px',
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px',
              color: 'white',
              outline: 'none',
            }}
          />
          <span style={{ position: 'absolute', left: '12px', top: '12px', color: '#9ca3af' }}>
            🔍
          </span>
        </div>

        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value as any)}
          aria-label="Filter by status"
          style={{
            padding: '12px 16px',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            color: 'white',
            outline: 'none',
          }}
        >
          <option value="all">All Status</option>
          <option value="idle">Idle</option>
          <option value="working">Working</option>
          <option value="thinking">Thinking</option>
        </select>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setViewMode('grid')}
            aria-pressed={viewMode === 'grid'}
            aria-label="Grid view"
            style={{
              padding: '12px 16px',
              background: viewMode === 'grid' ? 'rgba(6,182,212,0.2)' : 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px',
              color: 'white',
              cursor: 'pointer',
            }}
          >
            ⊞
          </button>
          <button
            onClick={() => setViewMode('list')}
            aria-pressed={viewMode === 'list'}
            aria-label="List view"
            style={{
              padding: '12px 16px',
              background: viewMode === 'list' ? 'rgba(6,182,212,0.2)' : 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px',
              color: 'white',
              cursor: 'pointer',
            }}
          >
            ≡
          </button>
        </div>
      </div>

      {/* Agents grid with memoization */}
      {filteredAgents.length === 0 ? (
        <div
          style={{
            textAlign: 'center',
            padding: '3rem',
            background: 'rgba(255,255,255,0.05)',
            borderRadius: '12px',
            color: '#9ca3af',
          }}
        >
          <p style={{ margin: 0 }}>No agents found matching your filters</p>
        </div>
      ) : (
        <div
          style={{
            display: viewMode === 'grid' ? 'grid' : 'flex',
            gridTemplateColumns: viewMode === 'grid' ? 'repeat(auto-fill, minmax(300px, 1fr))' : undefined,
            flexDirection: viewMode === 'grid' ? undefined : 'column',
            gap: '16px',
          }}
        >
          {filteredAgents.map((agent) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              isSelected={selectedAgent === agent.id}
              onClick={() => handleSelectAgent(agent.id)}
            />
          ))}
        </div>
      )}

      {/* Agent details panel */}
      {selectedAgent && (
        <div
          style={{
            marginTop: '2rem',
            padding: '2rem',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '12px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h2 style={{ margin: 0, color: 'white', fontSize: '24px' }}>
              {agents.find(a => a.id === selectedAgent)?.name}
            </h2>
            <button
              onClick={() => setSelectedAgent(null)}
              aria-label="Close details panel"
              style={{
                background: 'transparent',
                border: 'none',
                color: '#9ca3af',
                cursor: 'pointer',
                fontSize: '24px',
              }}
            >
              ✕
            </button>
          </div>
          <p style={{ color: '#9ca3af', margin: 0 }}>
            {agents.find(a => a.id === selectedAgent)?.description}
          </p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// REFACTORED PROJECTS PAGE - Remove duplicates, Add Error Boundary
// ============================================================================

interface Project {
  id: string;
  name: string;
  status: 'planning' | 'in_progress' | 'completed' | 'on_hold';
  progress: number;
  team: number;
  deadline: string;
}

const ProjectCard = memo(({ project }: { project: Project }) => {
  const statusColors = {
    planning: '#8b5cf6',
    in_progress: '#06b6d4',
    completed: '#10b981',
    on_hold: '#fbbf24',
  };

  return (
    <div
      style={{
        position: 'relative',
        padding: '1.5rem',
        background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: '12px',
        transition: 'all 0.2s',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-4px)';
        (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(6,182,212,0.3)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
        (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.1)';
      }}
    >
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '4px',
        background: `linear-gradient(to right, ${statusColors[project.status]}, transparent)`,
      }} />

      <h3 style={{ margin: '0 0 8px 0', color: 'white', fontSize: '18px', fontWeight: '700' }}>
        {project.name}
      </h3>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <span style={{
          fontSize: '12px',
          fontWeight: '600',
          color: statusColors[project.status],
          textTransform: 'capitalize',
        }}>
          {project.status.replace('_', ' ')}
        </span>
        <span style={{ fontSize: '14px', color: '#9ca3af', fontWeight: '600' }}>
          {project.progress}%
        </span>
      </div>

      <div style={{ height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden', marginBottom: '12px' }}>
        <div
          style={{
            height: '100%',
            width: `${project.progress}%`,
            background: `linear-gradient(to right, ${statusColors[project.status]}, ${statusColors[project.status]}dd)`,
            transition: 'width 0.5s',
          }}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '12px' }}>
        <div>
          <div style={{ color: '#9ca3af' }}>Team Size</div>
          <div style={{ color: 'white', fontWeight: '700', fontSize: '16px' }}>{project.team}</div>
        </div>
        <div>
          <div style={{ color: '#9ca3af' }}>Deadline</div>
          <div style={{ color: 'white', fontWeight: '700' }}>
            {new Date(project.deadline).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
          </div>
        </div>
      </div>
    </div>
  );
});

ProjectCard.displayName = 'ProjectCard';

export const RefactoredProjectsPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | Project['status']>('all');

  // Mock projects
  const projects: Project[] = useMemo(() => [
    {
      id: 'proj-001',
      name: 'E-Commerce Platform',
      status: 'in_progress',
      progress: 65,
      team: 4,
      deadline: '2025-11-15',
    },
    {
      id: 'proj-002',
      name: 'Mobile Banking App',
      status: 'completed',
      progress: 100,
      team: 3,
      deadline: '2025-10-30',
    },
  ], []);

  const filteredProjects = useMemo(() => {
    return projects.filter((project) => {
      const matchesSearch = project.name.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = filterStatus === 'all' || project.status === filterStatus;
      return matchesSearch && matchesStatus;
    });
  }, [projects, searchTerm, filterStatus]);

  return (
    <div style={{ padding: '2rem 0' }}>
      <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: '2rem' }}>
        Projects
      </h1>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '12px', marginBottom: '2rem' }}>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search projects..."
          aria-label="Search projects"
          style={{
            padding: '12px',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            color: 'white',
            outline: 'none',
          }}
        />

        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value as any)}
          aria-label="Filter projects by status"
          style={{
            padding: '12px 16px',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '8px',
            color: 'white',
            outline: 'none',
          }}
        >
          <option value="all">All Status</option>
          <option value="planning">Planning</option>
          <option value="in_progress">In Progress</option>
          <option value="completed">Completed</option>
          <option value="on_hold">On Hold</option>
        </select>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
        {filteredProjects.map((project) => (
          <ProjectCard key={project.id} project={project} />
        ))}
      </div>

      {filteredProjects.length === 0 && (
        <div style={{ textAlign: 'center', padding: '3rem', color: '#9ca3af' }}>
          <p>No projects found</p>
        </div>
      )}
    </div>
  );
};