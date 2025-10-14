import React, { useState, useEffect, useRef } from 'react';
import { Zap, MessageCircle, Upload, Download, Settings, X, Play, Pause, BarChart3 } from 'lucide-react';

// ============================================================================
// 3D AGENT SPHERE COMPONENT
// ============================================================================

function AgentSphere({ agent, onClick, isSelected }) {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const rotationRef = useRef(0);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const size = 100;
    canvas.width = size;
    canvas.height = size;
    
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = 35;
    
    const statusColors = {
      idle: '#6b7280',
      thinking: '#f59e0b',
      working: '#3b82f6',
      completed: '#10b981',
      error: '#ef4444'
    };
    
    const animate = () => {
      ctx.clearRect(0, 0, size, size);
      
      rotationRef.current += 0.02;
      const wave = Math.sin(rotationRef.current) * 5;
      
      // Outer glow
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius + 20);
      gradient.addColorStop(0, statusColors[agent.status]);
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      ctx.arc(centerX, centerY + wave, radius + 20, 0, Math.PI * 2);
      ctx.fill();
      
      // Main sphere with gradient
      const mainGradient = ctx.createRadialGradient(
        centerX - 10, centerY - 10 + wave, 5,
        centerX, centerY + wave, radius
      );
      mainGradient.addColorStop(0, statusColors[agent.status] + 'ff');
      mainGradient.addColorStop(0.7, statusColors[agent.status] + 'aa');
      mainGradient.addColorStop(1, statusColors[agent.status] + '66');
      
      ctx.globalAlpha = 1;
      ctx.fillStyle = mainGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY + wave, radius, 0, Math.PI * 2);
      ctx.fill();
      
      // Highlight
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.beginPath();
      ctx.arc(centerX - 10, centerY - 10 + wave, 12, 0, Math.PI * 2);
      ctx.fill();
      
      // Rings if selected
      if (isSelected) {
        ctx.strokeStyle = statusColors[agent.status];
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.6;
        for (let i = 1; i <= 2; i++) {
          ctx.beginPath();
          ctx.arc(centerX, centerY + wave, radius + (i * 10), 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [agent.status, isSelected]);
  
  return (
    <div className="flex flex-col items-center group cursor-pointer" onClick={onClick}>
      <div className={`relative transition-transform duration-300 ${isSelected ? 'scale-110' : 'group-hover:scale-110'}`}>
        <canvas ref={canvasRef} className="drop-shadow-2xl" />
        {agent.status === 'working' && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          </div>
        )}
      </div>
      <div className="text-center mt-3">
        <h3 className="text-sm font-semibold text-gray-200">{agent.name}</h3>
        <p className="text-xs text-gray-400">{agent.specialty}</p>
        <span className={`inline-block mt-1 px-2 py-0.5 text-xs rounded-full ${
          agent.status === 'idle' ? 'bg-gray-500/20 text-gray-400' :
          agent.status === 'thinking' ? 'bg-yellow-500/20 text-yellow-400' :
          agent.status === 'working' ? 'bg-blue-500/20 text-blue-400' :
          agent.status === 'completed' ? 'bg-green-500/20 text-green-400' :
          'bg-red-500/20 text-red-400'
        }`}>
          {agent.status}
        </span>
      </div>
    </div>
  );
}

// ============================================================================
// AGENT CONTROL PANEL
// ============================================================================

function AgentControlPanel({ agent, onClose }) {
  const [activeTab, setActiveTab] = useState('chat');
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { sender: 'agent', text: `Hello! I'm ${agent.name}. How can I assist you today?` }
  ]);
  
  const sendMessage = () => {
    if (!message.trim()) return;
    
    setChatHistory(prev => [...prev, 
      { sender: 'user', text: message },
      { sender: 'agent', text: `Understood. Processing your request: "${message}"...` }
    ]);
    setMessage('');
  };
  
  const tabs = [
    { id: 'chat', label: 'Chat', icon: MessageCircle },
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'stats', label: 'Stats', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];
  
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="backdrop-blur-xl bg-gray-900/90 border border-cyan-500/20 rounded-2xl shadow-2xl w-full max-w-3xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-cyan-500/20 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-cyan-400">{agent.name}</h2>
              <p className="text-sm text-gray-400">{agent.specialty}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-red-500/10 rounded-lg transition-colors">
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>
        
        {/* Tabs */}
        <div className="flex border-b border-cyan-500/20 px-6">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === tab.id 
                  ? 'border-cyan-400 text-cyan-400' 
                  : 'border-transparent text-gray-400 hover:text-cyan-400'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === 'chat' && (
            <div className="space-y-4">
              <div className="h-64 overflow-y-auto space-y-3 mb-4">
                {chatHistory.map((msg, i) => (
                  <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-xs p-3 rounded-lg ${
                      msg.sender === 'user' 
                        ? 'bg-cyan-500/20 text-cyan-100' 
                        : 'bg-gray-800/50 text-gray-300'
                    }`}>
                      {msg.text}
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Type your message..."
                  className="flex-1 px-4 py-2 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400"
                />
                <button onClick={sendMessage} className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all">
                  Send
                </button>
              </div>
            </div>
          )}
          
          {activeTab === 'upload' && (
            <div className="text-center py-12">
              <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-300 mb-2">Upload Files</h3>
              <p className="text-gray-400 mb-6">Share files with {agent.name} for analysis</p>
              <button className="px-6 py-3 bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors">
                Select Files
              </button>
            </div>
          )}
          
          {activeTab === 'stats' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-800/30 rounded-lg">
                  <div className="text-2xl font-bold text-cyan-400">156</div>
                  <div className="text-sm text-gray-400">Tasks Completed</div>
                </div>
                <div className="p-4 bg-gray-800/30 rounded-lg">
                  <div className="text-2xl font-bold text-green-400">98%</div>
                  <div className="text-sm text-gray-400">Success Rate</div>
                </div>
                <div className="p-4 bg-gray-800/30 rounded-lg">
                  <div className="text-2xl font-bold text-blue-400">24.5h</div>
                  <div className="text-sm text-gray-400">Total Active Time</div>
                </div>
                <div className="p-4 bg-gray-800/30 rounded-lg">
                  <div className="text-2xl font-bold text-purple-400">4.2/5</div>
                  <div className="text-sm text-gray-400">Avg Rating</div>
                </div>
              </div>
              
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-gray-300">Recent Tasks</h3>
                {['Code optimization', 'Bug fix analysis', 'Feature implementation'].map((task, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                    <span className="text-gray-300">{task}</span>
                    <span className="text-green-400 text-sm">Completed</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {activeTab === 'settings' && (
            <div className="space-y-4">
              <div className="space-y-3">
                <h3 className="text-lg font-semibold text-gray-300 mb-4">Agent Configuration</h3>
                {[
                  { label: 'Auto-assign tasks', checked: true },
                  { label: 'Priority mode', checked: false },
                  { label: 'Notifications', checked: true }
                ].map((setting, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                    <span className="text-gray-300">{setting.label}</span>
                    <button className={`w-12 h-6 rounded-full transition-colors ${
                      setting.checked ? 'bg-cyan-500' : 'bg-gray-600'
                    }`}>
                      <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                        setting.checked ? 'translate-x-6' : 'translate-x-1'
                      }`} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// AGENTS PAGE
// ============================================================================

export default function AgentsPage() {
  const [agents, setAgents] = useState([
    { id: 1, name: 'Alpha', specialty: 'Code Analysis', status: 'working' },
    { id: 2, name: 'Beta', specialty: 'UI Design', status: 'idle' },
    { id: 3, name: 'Gamma', specialty: 'Backend Dev', status: 'thinking' },
    { id: 4, name: 'Delta', specialty: 'Testing', status: 'completed' },
    { id: 5, name: 'Epsilon', specialty: 'Optimization', status: 'idle' },
    { id: 6, name: 'Zeta', specialty: 'Security', status: 'working' },
    { id: 7, name: 'Eta', specialty: 'Data Science', status: 'thinking' },
    { id: 8, name: 'Theta', specialty: 'DevOps', status: 'idle' },
    { id: 9, name: 'Iota', specialty: 'API Design', status: 'completed' },
    { id: 10, name: 'Kappa', specialty: 'Database', status: 'working' },
    { id: 11, name: 'Lambda', specialty: 'ML Models', status: 'idle' },
    { id: 12, name: 'Mu', specialty: 'Documentation', status: 'thinking' }
  ]);
  
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [filter, setFilter] = useState('all');
  
  useEffect(() => {
    const interval = setInterval(() => {
      setAgents(prev => prev.map(agent => {
        if (Math.random() > 0.7) {
          const statuses = ['idle', 'thinking', 'working', 'completed'];
          return { ...agent, status: statuses[Math.floor(Math.random() * statuses.length)] };
        }
        return agent;
      }));
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  const filteredAgents = filter === 'all' 
    ? agents 
    : agents.filter(a => a.status === filter);
  
  const statusCounts = {
    all: agents.length,
    idle: agents.filter(a => a.status === 'idle').length,
    working: agents.filter(a => a.status === 'working').length,
    thinking: agents.filter(a => a.status === 'thinking').length,
    completed: agents.filter(a => a.status === 'completed').length
  };
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">
          AI Agents
        </h1>
        <p className="text-gray-400">Manage and interact with your specialized AI agents</p>
      </div>
      
      {/* Status Filter */}
      <div className="flex flex-wrap gap-3 mb-8">
        {Object.entries(statusCounts).map(([status, count]) => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-lg transition-all ${
              filter === status
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                : 'bg-gray-800/30 text-gray-400 border border-gray-700/30 hover:border-cyan-500/30'
            }`}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)} ({count})
          </button>
        ))}
      </div>
      
      {/* Agent Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-8">
        {filteredAgents.map(agent => (
          <AgentSphere
            key={agent.id}
            agent={agent}
            onClick={() => setSelectedAgent(agent)}
            isSelected={selectedAgent?.id === agent.id}
          />
        ))}
      </div>
      
      {/* Control Panel */}
      {selectedAgent && (
        <AgentControlPanel
          agent={selectedAgent}
          onClose={() => setSelectedAgent(null)}
        />
      )}
    </div>
  );
}