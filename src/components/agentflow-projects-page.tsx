import React, { useState, useEffect } from 'react';
import { Folder, Calendar, Users, TrendingUp, Filter, Grid, List, Eye, Clock, CheckCircle, AlertCircle } from 'lucide-react';

// ============================================================================
// PROJECT CARD COMPONENT
// ============================================================================

function ProjectCard({ project, onClick, viewMode }) {
  const statusColors = {
    planning: 'blue',
    development: 'purple',
    testing: 'yellow',
    deployed: 'green',
    paused: 'gray'
  };
  
  const color = statusColors[project.status] || 'gray';
  
  if (viewMode === 'list') {
    return (
      <div 
        onClick={onClick}
        className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-4 hover:border-cyan-400/40 transition-all cursor-pointer hover:transform hover:scale-[1.02]"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 flex-1">
            <div className={`w-12 h-12 rounded-lg bg-${color}-500/20 flex items-center justify-center`}>
              <Folder className={`w-6 h-6 text-${color}-400`} />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-200">{project.name}</h3>
              <p className="text-sm text-gray-400">{project.description}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-cyan-400">{project.progress}%</div>
              <div className="text-xs text-gray-500">Progress</div>
            </div>
            
            <div className="text-center">
              <div className="flex items-center space-x-1">
                <Users className="w-4 h-4 text-gray-400" />
                <span className="text-sm text-gray-300">{project.team.length}</span>
              </div>
              <div className="text-xs text-gray-500">Team</div>
            </div>
            
            <div className={`px-3 py-1 rounded-full text-sm bg-${color}-500/20 text-${color}-400`}>
              {project.status}
            </div>
          </div>
        </div>
        
        <div className="mt-4 h-2 bg-gray-800/50 rounded-full overflow-hidden">
          <div 
            className={`h-full bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full transition-all duration-500`}
            style={{ width: `${project.progress}%` }}
          />
        </div>
      </div>
    );
  }
  
  return (
    <div 
      onClick={onClick}
      className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6 hover:border-cyan-400/40 transition-all cursor-pointer hover:transform hover:scale-105 group"
    >
      <div className="flex items-start justify-between mb-4">
        <div className={`w-12 h-12 rounded-lg bg-${color}-500/20 flex items-center justify-center group-hover:scale-110 transition-transform`}>
          <Folder className={`w-6 h-6 text-${color}-400`} />
        </div>
        <div className={`px-3 py-1 rounded-full text-xs bg-${color}-500/20 text-${color}-400`}>
          {project.status}
        </div>
      </div>
      
      <h3 className="text-lg font-semibold text-gray-200 mb-2">{project.name}</h3>
      <p className="text-sm text-gray-400 mb-4 line-clamp-2">{project.description}</p>
      
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">Progress</span>
            <span className="text-cyan-400 font-semibold">{project.progress}%</span>
          </div>
          <div className="h-2 bg-gray-800/50 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full transition-all duration-500"
              style={{ width: `${project.progress}%` }}
            />
          </div>
        </div>
        
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-1 text-gray-400">
            <Calendar className="w-4 h-4" />
            <span>{project.deadline}</span>
          </div>
          <div className="flex items-center space-x-1 text-gray-400">
            <Users className="w-4 h-4" />
            <span>{project.team.length}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// PROJECT DETAIL MODAL
// ============================================================================

function ProjectDetailModal({ project, onClose }) {
  const phases = [
    { name: 'Analysis', status: 'completed', progress: 100 },
    { name: 'Planning', status: 'completed', progress: 100 },
    { name: 'Development', status: 'active', progress: project.progress },
    { name: 'Testing', status: 'pending', progress: 0 },
    { name: 'Deployment', status: 'pending', progress: 0 }
  ];
  
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="backdrop-blur-xl bg-gray-900/90 border border-cyan-500/20 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] overflow-auto">
        <div className="p-6 border-b border-cyan-500/20 sticky top-0 bg-gray-900/90 backdrop-blur-xl">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-3xl font-bold text-cyan-400">{project.name}</h2>
              <p className="text-gray-400 mt-1">{project.description}</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-red-500/10 rounded-lg transition-colors">
              <Eye className="w-6 h-6 text-gray-400" />
            </button>
          </div>
        </div>
        
        <div className="p-6 space-y-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-gray-800/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="w-5 h-5 text-cyan-400" />
                <span className="text-sm text-gray-400">Progress</span>
              </div>
              <div className="text-2xl font-bold text-cyan-400">{project.progress}%</div>
            </div>
            
            <div className="p-4 bg-gray-800/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Users className="w-5 h-5 text-blue-400" />
                <span className="text-sm text-gray-400">Team Size</span>
              </div>
              <div className="text-2xl font-bold text-blue-400">{project.team.length}</div>
            </div>
            
            <div className="p-4 bg-gray-800/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <Clock className="w-5 h-5 text-purple-400" />
                <span className="text-sm text-gray-400">Days Left</span>
              </div>
              <div className="text-2xl font-bold text-purple-400">{Math.floor(Math.random() * 30)}</div>
            </div>
            
            <div className="p-4 bg-gray-800/30 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-sm text-gray-400">Tasks Done</span>
              </div>
              <div className="text-2xl font-bold text-green-400">{Math.floor(project.progress * 0.5)}/50</div>
            </div>
          </div>
          
          {/* Project Phases */}
          <div>
            <h3 className="text-xl font-bold text-gray-200 mb-4">Project Phases</h3>
            <div className="space-y-3">
              {phases.map((phase, i) => (
                <div key={i} className="p-4 bg-gray-800/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      {phase.status === 'completed' && <CheckCircle className="w-5 h-5 text-green-400" />}
                      {phase.status === 'active' && <AlertCircle className="w-5 h-5 text-blue-400 animate-pulse" />}
                      {phase.status === 'pending' && <Clock className="w-5 h-5 text-gray-500" />}
                      <span className="text-gray-200 font-medium">{phase.name}</span>
                    </div>
                    <span className={`text-sm ${
                      phase.status === 'completed' ? 'text-green-400' :
                      phase.status === 'active' ? 'text-blue-400' :
                      'text-gray-500'
                    }`}>
                      {phase.progress}%
                    </span>
                  </div>
                  <div className="h-2 bg-gray-700/50 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        phase.status === 'completed' ? 'bg-green-500' :
                        phase.status === 'active' ? 'bg-blue-500' :
                        'bg-gray-600'
                      }`}
                      style={{ width: `${phase.progress}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Team Members */}
          <div>
            <h3 className="text-xl font-bold text-gray-200 mb-4">Team Members</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {project.team.map((member, i) => (
                <div key={i} className="flex items-center space-x-3 p-3 bg-gray-800/30 rounded-lg">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white font-semibold">
                    {member[0]}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-200">{member}</div>
                    <div className="text-xs text-gray-500">Developer</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Timeline */}
          <div>
            <h3 className="text-xl font-bold text-gray-200 mb-4">Recent Activity</h3>
            <div className="space-y-3">
              {[
                { action: 'Code review completed', time: '2h ago', user: 'Alpha' },
                { action: 'Feature branch merged', time: '5h ago', user: 'Beta' },
                { action: 'Testing phase started', time: '1d ago', user: 'Gamma' },
                { action: 'Design approved', time: '2d ago', user: 'Delta' }
              ].map((activity, i) => (
                <div key={i} className="flex items-start space-x-3 p-3 bg-gray-800/30 rounded-lg">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 mt-2" />
                  <div className="flex-1">
                    <p className="text-gray-300">{activity.action}</p>
                    <p className="text-xs text-gray-500 mt-1">by {activity.user} • {activity.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// PROJECTS PAGE
// ============================================================================

export default function ProjectsPage() {
  const [projects, setProjects] = useState([
    {
      id: 1,
      name: 'E-Commerce Platform',
      description: 'Full-stack e-commerce solution with AI recommendations',
      status: 'development',
      progress: 65,
      deadline: 'Dec 15, 2025',
      team: ['Alpha', 'Beta', 'Gamma', 'Delta']
    },
    {
      id: 2,
      name: 'Analytics Dashboard',
      description: 'Real-time data visualization and reporting system',
      status: 'testing',
      progress: 85,
      deadline: 'Nov 30, 2025',
      team: ['Epsilon', 'Zeta', 'Eta']
    },
    {
      id: 3,
      name: 'Mobile App',
      description: 'Cross-platform mobile application with offline support',
      status: 'planning',
      progress: 25,
      deadline: 'Jan 20, 2026',
      team: ['Theta', 'Iota']
    },
    {
      id: 4,
      name: 'API Gateway',
      description: 'Microservices API gateway with rate limiting',
      status: 'deployed',
      progress: 100,
      deadline: 'Oct 15, 2025',
      team: ['Kappa', 'Lambda', 'Mu']
    },
    {
      id: 5,
      name: 'ML Pipeline',
      description: 'Automated machine learning training pipeline',
      status: 'development',
      progress: 45,
      deadline: 'Dec 30, 2025',
      team: ['Alpha', 'Lambda']
    },
    {
      id: 6,
      name: 'CRM System',
      description: 'Customer relationship management with AI insights',
      status: 'planning',
      progress: 15,
      deadline: 'Feb 10, 2026',
      team: ['Beta', 'Gamma', 'Delta', 'Epsilon']
    }
  ]);
  
  const [selectedProject, setSelectedProject] = useState(null);
  const [viewMode, setViewMode] = useState('grid');
  const [statusFilter, setStatusFilter] = useState('all');
  
  const filteredProjects = statusFilter === 'all' 
    ? projects 
    : projects.filter(p => p.status === statusFilter);
  
  const statusCounts = {
    all: projects.length,
    planning: projects.filter(p => p.status === 'planning').length,
    development: projects.filter(p => p.status === 'development').length,
    testing: projects.filter(p => p.status === 'testing').length,
    deployed: projects.filter(p => p.status === 'deployed').length
  };
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">
          Projects
        </h1>
        <p className="text-gray-400">Track and manage your development projects</p>
      </div>
      
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <div className="flex flex-wrap gap-3">
          {Object.entries(statusCounts).map(([status, count]) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={`px-4 py-2 rounded-lg transition-all ${
                statusFilter === status
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'bg-gray-800/30 text-gray-400 border border-gray-700/30 hover:border-cyan-500/30'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)} ({count})
            </button>
          ))}
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-2 rounded-lg transition-all ${
              viewMode === 'grid' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-gray-800/30 text-gray-400 hover:text-cyan-400'
            }`}
          >
            <Grid className="w-5 h-5" />
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-2 rounded-lg transition-all ${
              viewMode === 'list' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-gray-800/30 text-gray-400 hover:text-cyan-400'
            }`}
          >
            <List className="w-5 h-5" />
          </button>
        </div>
      </div>
      
      {/* Projects Grid/List */}
      <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
        {filteredProjects.map(project => (
          <ProjectCard
            key={project.id}
            project={project}
            onClick={() => setSelectedProject(project)}
            viewMode={viewMode}
          />
        ))}
      </div>
      
      {/* Project Detail Modal */}
      {selectedProject && (
        <ProjectDetailModal
          project={selectedProject}
          onClose={() => setSelectedProject(null)}
        />
      )}
    </div>
  );
}