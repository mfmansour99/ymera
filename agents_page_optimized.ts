import React, { useState, useMemo, useCallback, Suspense, memo } from 'react';
import { PageHeader } from '../components/PageHeader';
import { AgentSphere, AgentsScene } from '../components';
import { Agent } from '../types/agents';
import { ErrorBoundary } from '../components/common/ErrorBoundary';

interface AgentsPageProps {
  agents: Agent[];
}

// Memoized Agent Card Component
const AgentCard = memo<{
  agent: Agent;
  isSelected: boolean;
  onSelect: () => void;
  getStatusColor: (status: Agent['status']) => string;
  getStatusBg: (status: Agent['status']) => string;
}>(({ agent, isSelected, onSelect, getStatusColor, getStatusBg }) => (
  <button
    className={`glass-card hover-lift cursor-pointer text-left w-full transition-all ${
      isSelected ? 'ring-2 ring-accent-primary' : ''
    }`}
    onClick={onSelect}
    aria-label={`View details for ${agent.name}, status: ${agent.status}`}
    aria-pressed={isSelected}
  >
    <div className="flex items-center justify-between mb-4">
      <h3 className="text-lg font-bold">{agent.name}</h3>
      <span className={`px-2 py-1 rounded text-xs ${getStatusBg(agent.status)} ${getStatusColor(agent.status)}`}>
        {agent.status.toUpperCase()}
      </span>
    </div>
    <p className="text-secondary text-sm mb-4">{agent.description.substring(0, 100)}...</p>
    <div className="grid grid-cols-2 gap-4 mb-4">
      <div>
        <div className="text-sm text-secondary">Type</div>
        <div className="font-bold text-accent">{agent.type.replace('_', ' ')}</div>
      </div>
      <div>
        <div className="text-sm text-secondary">Efficiency</div>
        <div className="font-bold text-accent-success">{agent.efficiency}</div>
      </div>
    </div>
    <div className="text-sm text-secondary">
      Tasks Completed: {agent.tasksCompleted}
    </div>
  </button>
));

AgentCard.displayName = 'AgentCard';

export const AgentsPage: React.FC<AgentsPageProps> = ({ agents }) => {
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [filterStatus, setFilterStatus] = useState<'all' | Agent['status']>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState<'3d' | 'list'>('3d');
  const [sceneError, setSceneError] = useState<string | null>(null);

  // Memoized status color functions
  const getStatusColor = useCallback((status: Agent['status']): string => {
    const colors = {
      idle: 'text-agent-idle',
      thinking: 'text-agent-thinking',
      working: 'text-agent-working',
      completed: 'text-agent-success',
      error: 'text-agent-error'
    };
    return colors[status] || 'text-secondary';
  }, []);

  const getStatusBg = useCallback((status: Agent['status']): string => {
    const backgrounds = {
      idle: 'bg-agent-idle bg-opacity-20',
      thinking: 'bg-agent-thinking bg-opacity-20',
      working: 'bg-agent-working bg-opacity-20',
      completed: 'bg-agent-success bg-opacity-20',
      error: 'bg-agent-error bg-opacity-20'
    };
    return backgrounds[status] || 'bg-glass';
  }, []);

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

  // Memoized handlers
  const handleAgentSelect = useCallback((agent: Agent) => {
    setSelectedAgent(agent);
  }, []);

  const handleAgentDeselect = useCallback(() => {
    setSelectedAgent(null);
  }, []);

  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  const handleStatusFilter = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setFilterStatus(e.target.value as Agent['status'] | 'all');
  }, []);

  const handleViewModeChange = useCallback((mode: '3d' | 'list') => {
    setViewMode(mode);
    setSceneError(null);
  }, []);

  return (
    <div className="min-h-screen pt-20 pb-20">
      <div className="container">
        <PageHeader
          title="AI Agents"
          subtitle="Manage your intelligent agents"
          actions={[
            {
              label: 'Add Agent',
              icon: 'fas fa-plus',
              onClick: () => console.log('Add new agent'),
            },
            {
              label: 'Refresh',
              icon: 'fas fa-sync-alt',
              onClick: () => console.log('Refresh agents'),
            },
          ]}
        />

        {/* Search and Filter Controls */}
        <div className="glass-card mb-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-2">
              <input
                type="text"
                placeholder="Search agents..."
                value={searchTerm}
                onChange={handleSearchChange}
                className="glass-input"
                aria-label="Search agents"
              />
            </div>
            <select
              value={filterStatus}
              onChange={handleStatusFilter}
              className="glass-input"
              aria-label="Filter by status"
            >
              <option value="all">All Status</option>
              <option value="idle">Idle</option>
              <option value="thinking">Thinking</option>
              <option value="working">Working</option>
              <option value="completed">Completed</option>
              <option value="error">Error</option>
            </select>
            <div className="flex gap-2">
              <button
                onClick={() => handleViewModeChange('3d')}
                className={`glass-button flex-1 ${viewMode === '3d' ? 'bg-accent bg-opacity-20 border-accent' : ''}`}
                aria-label="3D view"
                aria-pressed={viewMode === '3d'}
              >
                <i className="fas fa-cube" />
              </button>
              <button
                onClick={() => handleViewModeChange('list')}
                className={`glass-button flex-1 ${viewMode === 'list' ? 'bg-accent bg-opacity-20 border-accent' : ''}`}
                aria-label="List view"
                aria-pressed={viewMode === 'list'}
              >
                <i className="fas fa-list" />
              </button>
            </div>
          </div>
        </div>

        {/* 3D View with Error Boundary */}
        {viewMode === '3d' && (
          <div className="relative h-[60vh] mb-8">
            {sceneError ? (
              <div className="glass-card h-full flex items-center justify-center">
                <div className="text-center">
                  <i className="fas fa-exclamation-triangle text-6xl text-accent-danger mb-4" />
                  <h3 className="text-xl font-bold mb-2">3D Scene Error</h3>
                  <p className="text-secondary mb-6">{sceneError}</p>
                  <button
                    onClick={() => {
                      setSceneError(null);
                      window.location.reload();
                    }}
                    className="glass-button bg-accent-primary bg-opacity-20"
                  >
                    <i className="fas fa-redo mr-2" />
                    Reload Page
                  </button>
                </div>
              </div>
            ) : (
              <ErrorBoundary
                fallback={(error) => {
                  setSceneError(error.message);
                  return null;
                }}
              >
                <Suspense fallback={
                  <div className="glass-card h-full flex items-center justify-center">
                    <div className="text-center">
                      <i className="fas fa-spinner fa-spin text-4xl text-accent-primary mb-4" />
                      <p className="text-secondary">Loading 3D scene...</p>
                    </div>
                  </div>
                }>
                  <AgentsScene 
                    agents={filteredAgents} 
                    onAgentClick={handleAgentSelect} 
                  />
                </Suspense>
              </ErrorBoundary>
            )}
          </div>
        )}

        {/* List View */}
        {viewMode === 'list' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {filteredAgents.length === 0 ? (
              <div className="col-span-full glass-card p-8 text-center">
                <i className="fas fa-search text-4xl text-secondary mb-4" />
                <p className="text-secondary">No agents found matching your filters</p>
              </div>
            ) : (
              filteredAgents.map((agent) => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  isSelected={selectedAgent?.id === agent.id}
                  onSelect={() => handleAgentSelect(agent)}
                  getStatusColor={getStatusColor}
                  getStatusBg={getStatusBg}
                />
              ))
            )}
          </div>
        )}

        {/* Agent Details Panel */}
        {selectedAgent && (
          <div className="glass-card">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                <div className={`w-16 h-16 rounded-full flex items-center justify-center ${getStatusBg(selectedAgent.status)} border-2 border-glass-border`}>
                  <i className={`fas ${selectedAgent.type === 'code' ? 'fa-code' : selectedAgent.type === 'design' ? 'fa-paint-brush' : 'fa-cog'} ${getStatusColor(selectedAgent.status)} text-2xl`} />
                </div>
                <div>
                <h4 className="font-medium mb-2">Actions</h4>
                <div className="flex space-x-2">
                  <button className="glass-button bg-accent-primary bg-opacity-20 border-accent-primary">
                    <i className="fas fa-play mr-2" />
                    Start Task
                  </button>
                  <button className="glass-button bg-accent-success bg-opacity-20 border-accent-success">
                    <i className="fas fa-check mr-2" />
                    Complete Task
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};  <h2 className="text-2xl font-bold">{selectedAgent.name}</h2>
                  <p className="text-secondary">{selectedAgent.type.replace('_', ' ')} Agent</p>
                </div>
              </div>
              <button
                onClick={handleAgentDeselect}
                className="glass-button"
                aria-label="Close agent details"
              >
                <i className="fas fa-times" />
              </button>
            </div>

            {/* Agent Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div>
                <h4 className="font-medium mb-2">Status</h4>
                <div className={`flex items-center ${getStatusColor(selectedAgent.status)}`}>
                  <i className="fas fa-circle mr-2" />
                  <span className="capitalize">{selectedAgent.status.replace('_', ' ')}</span>
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">Efficiency</h4>
                <div className="text-accent-success font-bold">{selectedAgent.efficiency}</div>
              </div>
              <div>
                <h4 className="font-medium mb-2">Tasks Completed</h4>
                <div className="text-accent-primary font-bold">{selectedAgent.tasksCompleted}</div>
              </div>
            </div>

            {/* Description */}
            <div className="mb-6">
              <h4 className="font-medium mb-2">Description</h4>
              <p className="text-secondary">{selectedAgent.description}</p>
            </div>

            {/* Current Phase & Actions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2">Current Phase</h4>
                <div className="glass p-4 rounded-lg">
                  {selectedAgent.currentPhase ? (
                    <div className="flex items-center space-x-3">
                      <i className="fas fa-cog text-accent-primary" />
                      <span className="capitalize">{selectedAgent.currentPhase.replace('_', ' ')}</span>
                    </div>
                  ) : (
                    <p className="text-secondary">No active phase</p>
                  )}
                </div>
              </div>
              <div>
                