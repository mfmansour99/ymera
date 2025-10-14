import React, { memo, useMemo } from 'react';
import { Agent, Project } from '../types';

interface StatusBarProps {
  agents: Agent[];
  projects: Project[];
  notifications: number;
  isLoading?: boolean;
  systemStatus?: 'operational' | 'degraded' | 'offline';
  onNotificationClick?: () => void;
}

export const StatusBar: React.FC<StatusBarProps> = memo(({
  agents,
  projects,
  notifications,
  isLoading = false,
  systemStatus = 'operational',
  onNotificationClick
}) => {
  // Memoized calculations
  const stats = useMemo(() => ({
    activeAgents: agents.filter(agent => agent.status === 'working' || agent.status === 'thinking').length,
    completedProjects: projects.filter(project => project.status === 'completed').length,
    inProgressProjects: projects.filter(project => project.status === 'in_progress').length
  }), [agents, projects]);

  const systemStatusConfig = useMemo(() => {
    const configs = {
      operational: {
        icon: 'fa-circle',
        color: 'text-agent-success',
        text: 'All systems operational',
        bgColor: 'bg-green-500'
      },
      degraded: {
        icon: 'fa-exclamation-triangle',
        color: 'text-accent-warning',
        text: 'Performance degraded',
        bgColor: 'bg-yellow-500'
      },
      offline: {
        icon: 'fa-times-circle',
        color: 'text-accent-danger',
        text: 'System offline',
        bgColor: 'bg-red-500'
      }
    };
    return configs[systemStatus];
  }, [systemStatus]);

  return (
    <div 
      className="fixed bottom-0 left-0 right-0 z-40 bg-primary-bg bg-opacity-90 backdrop-blur-md border-t border-glass-border py-3 px-4"
      role="status"
      aria-live="polite"
    >
      <div className="container">
        <div className="flex items-center justify-between">
          {/* Left side - Stats */}
          <div className="flex items-center space-x-6">
            {/* Agents Status */}
            <div className="flex items-center space-x-2">
              <i className="fas fa-robot text-accent-primary" aria-hidden="true" />
              <span className="text-sm">
                {isLoading ? (
                  <span className="inline-block">
                    <i className="fas fa-spinner fa-spin text-accent-primary" />
                  </span>
                ) : (
                  <>
                    <span className="font-medium">{stats.activeAgents}</span>
                    <span className="text-secondary"> / {agents.length}</span>
                    <span className="sr-only">active</span> Agents
                  </>
                )}
              </span>
            </div>

            {/* Divider */}
            <div className="hidden sm:block w-px h-6 bg-glass-border" aria-hidden="true" />

            {/* Projects Status */}
            <div className="flex items-center space-x-2">
              <i className="fas fa-project-diagram text-accent-secondary" aria-hidden="true" />
              <span className="text-sm">
                {isLoading ? (
                  <span className="inline-block">
                    <i className="fas fa-spinner fa-spin text-accent-secondary" />
                  </span>
                ) : (
                  <>
                    <span className="font-medium">{stats.completedProjects}</span>
                    <span className="text-secondary"> / {projects.length}</span>
                    <span className="sr-only">completed</span> Projects
                  </>
                )}
              </span>
            </div>

            {/* Divider */}
            <div className="hidden md:block w-px h-6 bg-glass-border" aria-hidden="true" />

            {/* Active Projects */}
            <div className="hidden md:flex items-center space-x-2">
              <i className="fas fa-play-circle text-accent-tertiary" aria-hidden="true" />
              <span className="text-sm">
                {isLoading ? (
                  <span className="inline-block">
                    <i className="fas fa-spinner fa-spin text-accent-tertiary" />
                  </span>
                ) : (
                  <>
                    <span className="font-medium">{stats.inProgressProjects}</span> Active
                  </>
                )}
              </span>
            </div>
          </div>

          {/* Right side - Notifications & System Status */}
          <div className="flex items-center space-x-4">
            {/* Notification Button */}
            <div className="relative">
              <button
                onClick={onNotificationClick}
                className="w-10 h-10 rounded-full glass flex items-center justify-center hover:bg-accent-primary hover:bg-opacity-20 transition-all relative focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-primary-bg"
                aria-label={notifications > 0 ? `${notifications} unread notifications` : 'No notifications'}
              >
                <i className="fas fa-bell" aria-hidden="true" />
                {notifications > 0 && (
                  <span 
                    className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-accent-danger text-white text-xs flex items-center justify-center font-bold"
                    aria-hidden="true"
                  >
                    {notifications > 99 ? '99+' : notifications}
                  </span>
                )}
              </button>
            </div>

            {/* Divider */}
            <div className="hidden sm:block w-px h-6 bg-glass-border" aria-hidden="true" />

            {/* System Status */}
            <div className="flex items-center space-x-2 text-sm">
              <div className="relative flex items-center">
                <i 
                  className={`fas ${systemStatusConfig.icon} ${systemStatusConfig.color}`}
                  aria-hidden="true"
                />
                {/* Pulse animation for active status */}
                {systemStatus === 'operational' && (
                  <span className="absolute inline-flex h-full w-full animate-ping opacity-75">
                    <span className={`inline-flex h-full w-full rounded-full ${systemStatusConfig.bgColor} opacity-75`}></span>
                  </span>
                )}
              </div>
              <span className="hidden lg:inline">{systemStatusConfig.text}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

StatusBar.displayName = 'StatusBar';