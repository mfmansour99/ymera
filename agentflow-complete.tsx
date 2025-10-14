import React, { useState, useEffect, createContext, useContext, useRef } from 'react';
import { Camera, Zap, Settings, User, LogOut, Bell, Activity, TrendingUp, MessageCircle, Upload, X, BarChart3, Folder, Calendar, Users, Grid, List, Eye, Clock, CheckCircle, AlertCircle, Mail, Award, Code, Edit2, Save, Palette, Gauge, Shield, Moon, Sun } from 'lucide-react';

// ============================================================================
// CONTEXT & STATE MANAGEMENT
// ============================================================================

const AppContext = createContext(null);

function AppProvider({ children }) {
  const [state, setState] = useState({
    user: null,
    notifications: [],
    settings: { animations: true, particles: true, performance: 'balanced' }
  });
  
  const login = (username) => {
    setState(prev => ({ ...prev, user: { id: '1', username, email: `${username}@agentflow.ai` } }));
  };
  
  const logout = () => setState(prev => ({ ...prev, user: null }));
  
  const addNotification = (notification) => {
    const id = Date.now().toString();
    setState(prev => ({ ...prev, notifications: [...prev.notifications, { ...notification, id }] }));
    setTimeout(() => setState(prev => ({ ...prev, notifications: prev.notifications.filter(n => n.id !== id) })), 5000);
  };
  
  const updateSettings = (newSettings) => {
    setState(prev => ({ ...prev, settings: { ...prev.settings, ...newSettings } }));
  };
  
  return <AppContext.Provider value={{ ...state, login, logout, addNotification, updateSettings }}>{children}</AppContext.Provider>;
}

const useApp = () => useContext(AppContext);

// ============================================================================
// PARTICLE SYSTEM
// ============================================================================

function ParticleCanvas({ enabled, performance }) {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  
  useEffect(() => {
    if (!enabled) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
    resize();
    
    const count = performance === 'high' ? 80 : performance === 'balanced' ? 50 : 30;
    particlesRef.current = Array.from({ length: count }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1,
      opacity: Math.random() * 0.5 + 0.3,
      color: ['#00f5ff', '#ff00aa', '#00ff88'][Math.floor(Math.random() * 3)]
    }));
    
    const handleMouse = (e) => { mouseRef.current = { x: e.clientX, y: e.clientY }; };
    window.addEventListener('mousemove', handleMouse);
    window.addEventListener('resize', resize);
    
    const animate = () => {
      ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      particlesRef.current.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;
        
        const dx = mouseRef.current.x - p.x;
        const dy = mouseRef.current.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < 100) {
          const force = (100 - dist) / 100;
          p.vx += (dx / dist) * force * 0.1;
          p.vy += (dy / dist) * force * 0.1;
        }
        
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;
        
        p.vx *= 0.99;
        p.vy *= 0.99;
        
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
        
        if (performance !== 'low') {
          particlesRef.current.slice(i + 1).forEach(p2 => {
            const dx = p.x - p2.x;
            const dy = p.y - p2.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 120) {
              ctx.globalAlpha = (1 - dist / 120) * 0.2;
              ctx.strokeStyle = p.color;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(p.x, p.y);
              ctx.lineTo(p2.x, p2.y);
              ctx.stroke();
            }
          });
        }
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      window.removeEventListener('mousemove', handleMouse);
      window.removeEventListener('resize', resize);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [enabled, performance]);
  
  if (!enabled) return null;
  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none" style={{ zIndex: 1 }} />;
}

// ============================================================================
// NAVIGATION
// ============================================================================

function Navigation({ currentPage, onNavigate }) {
  const { user, logout, notifications } = useApp();
  const [showNotif, setShowNotif] = useState(false);
  
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'agents', label: 'Agents', icon: Zap },
    { id: 'projects', label: 'Projects', icon: Folder },
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];
  
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-gray-900/70 border-b border-cyan-500/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">AgentFlow</span>
          </div>
          
          {user && (
            <>
              <div className="flex items-center space-x-1">
                {navItems.map(item => (
                  <button key={item.id} onClick={() => onNavigate(item.id)} className={`px-4 py-2 rounded-lg transition-all duration-200 flex items-center space-x-2 ${currentPage === item.id ? 'bg-cyan-500/20 text-cyan-400' : 'text-gray-400 hover:text-cyan-400 hover:bg-cyan-500/10'}`}>
                    <item.icon className="w-4 h-4" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </button>
                ))}
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <button onClick={() => setShowNotif(!showNotif)} className="p-2 rounded-lg hover:bg-cyan-500/10 transition-colors relative">
                    <Bell className="w-5 h-5 text-gray-400" />
                    {notifications.length > 0 && <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />}
                  </button>
                  
                  {showNotif && notifications.length > 0 && (
                    <div className="absolute right-0 mt-2 w-80 backdrop-blur-xl bg-gray-900/90 border border-cyan-500/20 rounded-lg shadow-xl p-4">
                      <h3 className="text-lg font-semibold text-cyan-400 mb-3">Notifications</h3>
                      <div className="space-y-2">
                        {notifications.map(n => (
                          <div key={n.id} className="p-3 bg-cyan-500/5 rounded-lg">
                            <p className="text-sm text-gray-300">{n.message}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                
                <button onClick={logout} className="p-2 rounded-lg hover:bg-red-500/10 transition-colors">
                  <LogOut className="w-5 h-5 text-red-400" />
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

// ============================================================================
// LOGIN PAGE
// ============================================================================

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  
  const handleSubmit = () => {
    if (!username.trim()) return;
    setLoading(true);
    setTimeout(() => { onLogin(username); setLoading(false); }, 800);
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-400 to-blue-600 mb-4">
            <Zap className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">AgentFlow</h1>
          <p className="text-gray-400">Advanced AI Project Management</p>
        </div>
        
        <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-2xl p-8">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Username</label>
              <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSubmit()} className="w-full px-4 py-3 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400 transition-colors" placeholder="Enter your username" />
            </div>
            
            <button onClick={handleSubmit} disabled={loading || !username.trim()} className="w-full py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg font-medium hover:shadow-lg hover:shadow-cyan-500/50 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed">
              {loading ? 'Signing in...' : 'Sign In'}
            </button>
          </div>
          
          <p className="mt-6 text-center text-sm text-gray-400">Demo: Enter any username to login</p>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// DASHBOARD PAGE
// ============================================================================

function DashboardPage() {
  const { addNotification } = useApp();
  
  useEffect(() => {
    addNotification({ message: 'Welcome to AgentFlow! System online and ready.' });
  }, []);
  
  const stats = [
    { label: 'Active Agents', value: '15', trend: '+12%' },
    { label: 'Projects', value: '8', trend: '+3' },
    { label: 'Tasks Done', value: '247', trend: '+45' },
    { label: 'Success Rate', value: '94%', trend: '+2%' }
  ];
  
  const activities = [
    { text: 'Agent Alpha completed code analysis', time: '2m ago' },
    { text: 'Project Beta entered testing phase', time: '15m ago' },
    { text: 'Agent Gamma optimized queries', time: '1h ago' },
    { text: 'Deployment scheduled', time: '3h ago' }
  ];
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">Dashboard</h1>
        <p className="text-gray-400">Monitor your AI agents and projects in real-time</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, i) => (
          <div key={i} className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6 hover:border-cyan-400/40 transition-all duration-200 hover:transform hover:scale-105">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-cyan-400" />
              <div className="text-right">
                <div className="text-3xl font-bold text-cyan-400">{stat.value}</div>
                <div className="text-xs text-green-400 flex items-center justify-end">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  {stat.trend}
                </div>
              </div>
            </div>
            <p className="text-gray-400 text-sm">{stat.label}</p>
          </div>
        ))}
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
          <h2 className="text-xl font-bold text-cyan-400 mb-4">Recent Activity</h2>
          <div className="space-y-3">
            {activities.map((activity, i) => (
              <div key={i} className="flex items-start space-x-3 p-3 bg-cyan-500/5 rounded-lg hover:bg-cyan-500/10 transition-colors">
                <div className="w-2 h-2 rounded-full bg-cyan-400 mt-2 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-gray-300 text-sm">{activity.text}</p>
                  <p className="text-gray-500 text-xs mt-1">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
          <h2 className="text-xl font-bold text-cyan-400 mb-4">System Status</h2>
          <div className="space-y-4">
            {[
              { label: 'API Status', value: 'Operational' },
              { label: 'Database', value: 'Healthy' },
              { label: 'Queue Processing', value: '234 tasks/min' },
              { label: 'Response Time', value: '45ms avg' }
            ].map((status, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                <span className="text-gray-300">{status.label}</span>
                <span className="text-cyan-400 font-medium">{status.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// AGENTS PAGE (Simplified 3D visualization)
// ============================================================================

function AgentSphere({ agent, onClick }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = 100;
    canvas.height = 100;
    
    const colors = { idle: '#6b7280', thinking: '#f59e0b', working: '#3b82f6', completed: '#10b981', error: '#ef4444' };
    let rotation = 0;
    
    const animate = () => {
      ctx.clearRect(0, 0, 100, 100);
      rotation += 0.02;
      const wave = Math.sin(rotation) * 5;
      
      const gradient = ctx.createRadialGradient(50, 50, 0, 50, 50, 55);
      gradient.addColorStop(0, colors[agent.status]);
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      ctx.arc(50, 50 + wave, 55, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors[agent.status];
      ctx.beginPath();
      ctx.arc(50, 50 + wave, 35, 0, Math.PI * 2);
      ctx.fill();
      
      requestAnimationFrame(animate);
    };
    
    animate();
  }, [agent.status]);
  
  return (
    <div className="flex flex-col items-center group cursor-pointer" onClick={onClick}>
      <canvas ref={canvasRef} className="drop-shadow-2xl group-hover:scale-110 transition-transform" />
      <div className="text-center mt-3">
        <h3 className="text-sm font-semibold text-gray-200">{agent.name}</h3>
        <p className="text-xs text-gray-400">{agent.specialty}</p>
        <span className={`inline-block mt-1 px-2 py-0.5 text-xs rounded-full ${agent.status === 'idle' ? 'bg-gray-500/20 text-gray-400' : agent.status === 'thinking' ? 'bg-yellow-500/20 text-yellow-400' : agent.status === 'working' ? 'bg-blue-500/20 text-blue-400' : 'bg-green-500/20 text-green-400'}`}>
          {agent.status}
        </span>
      </div>
    </div>
  );
}

function AgentsPage() {
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
  
  const [filter, setFilter] = useState('all');
  
  const filtered = filter === 'all' ? agents : agents.filter(a => a.status === filter);
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">AI Agents</h1>
        <p className="text-gray-400">Manage and interact with your specialized AI agents</p>
      </div>
      
      <div className="flex flex-wrap gap-3 mb-8">
        {['all', 'idle', 'working', 'thinking', 'completed'].map(status => (
          <button key={status} onClick={() => setFilter(status)} className={`px-4 py-2 rounded-lg transition-all ${filter === status ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' : 'bg-gray-800/30 text-gray-400 border border-gray-700/30 hover:border-cyan-500/30'}`}>
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        ))}
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-8">
        {filtered.map(agent => (
          <AgentSphere key={agent.id} agent={agent} onClick={() => alert(`Opening ${agent.name} control panel`)} />
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// PROJECTS PAGE
// ============================================================================

function ProjectCard({ project }) {
  return (
    <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6 hover:border-cyan-400/40 transition-all cursor-pointer hover:transform hover:scale-105">
      <div className="flex items-start justify-between mb-4">
        <div className="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center">
          <Folder className="w-6 h-6 text-blue-400" />
        </div>
        <div className="px-3 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400">{project.status}</div>
      </div>
      
      <h3 className="text-lg font-semibold text-gray-200 mb-2">{project.name}</h3>
      <p className="text-sm text-gray-400 mb-4">{project.description}</p>
      
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">Progress</span>
            <span className="text-cyan-400 font-semibold">{project.progress}%</span>
          </div>
          <div className="h-2 bg-gray-800/50 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full" style={{ width: `${project.progress}%` }} />
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

function ProjectsPage() {
  const projects = [
    { id: 1, name: 'E-Commerce Platform', description: 'Full-stack e-commerce solution with AI recommendations', status: 'development', progress: 65, deadline: 'Dec 15', team: ['Alpha', 'Beta', 'Gamma', 'Delta'] },
    { id: 2, name: 'Analytics Dashboard', description: 'Real-time data visualization and reporting system', status: 'testing', progress: 85, deadline: 'Nov 30', team: ['Epsilon', 'Zeta', 'Eta'] },
    { id: 3, name: 'Mobile App', description: 'Cross-platform mobile application', status: 'planning', progress: 25, deadline: 'Jan 20', team: ['Theta', 'Iota'] },
    { id: 4, name: 'API Gateway', description: 'Microservices API gateway', status: 'deployed', progress: 100, deadline: 'Oct 15', team: ['Kappa', 'Lambda', 'Mu'] }
  ];
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">Projects</h1>
        <p className="text-gray-400">Track and manage your development projects</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projects.map(project => (
          <ProjectCard key={project.id} project={project} />
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// PROFILE & SETTINGS PAGES (Simplified)
// ============================================================================

function ProfilePage() {
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-8">Profile</h1>
      <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-8">
        <div className="text-center mb-6">
          <div className="w-32 h-32 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white text-4xl font-bold mx-auto mb-4">D</div>
          <h2 className="text-2xl font-bold text-gray-200">demo_user</h2>
          <p className="text-gray-400">Senior Developer</p>
        </div>
        <div className="space-y-4">
          {[{ label: 'Projects', value: '24' }, { label: 'Tasks', value: '487' }, { label: 'Agents', value: '15' }].map((stat, i) => (
            <div key={i} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
              <span className="text-gray-300">{stat.label}</span>
              <span className="text-cyan-400 font-bold">{stat.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SettingsPage() {
  const { settings, updateSettings } = useApp();
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-8">Settings</h1>
      <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-8">
        <div className="space-y-4">
          {[
            { key: 'animations', label: 'Enable Animations' },
            { key: 'particles', label: 'Particle Effects' }
          ].map(setting => (
            <div key={setting.key} className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg">
              <span className="text-gray-200">{setting.label}</span>
              <button onClick={() => updateSettings({ [setting.key]: !settings[setting.key] })} className={`w-12 h-6 rounded-full transition-colors ${settings[setting.key] ? 'bg-cyan-500' : 'bg-gray-600'}`}>
                <div className={`w-5 h-5 bg-white rounded-full transition-transform ${settings[setting.key] ? 'translate-x-6' : 'translate-x-0.5'}`} />
              </button>
            </div>
          ))}
          
          <div>
            <label className="block text-gray-300 mb-2">Performance Mode</label>
            <div className="grid grid-cols-3 gap-3">
              {['low', 'balanced', 'high'].map(mode => (
                <button key={mode} onClick={() => updateSettings({ performance: mode })} className={`p-3 rounded-lg border transition-all ${settings.performance === mode ? 'bg-cyan-500/20 border-cyan-500/30 text-cyan-400' : 'bg-gray-800/30 border-gray-700/30 text-gray-400 hover:border-cyan-500/30'}`}>
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN APP
// ============================================================================

export default function AgentFlow() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  
  return (
    <AppProvider>
      <AppContent currentPage={currentPage} onNavigate={setCurrentPage} />
    </AppProvider>
  );
}

function AppContent({ currentPage, onNavigate }) {
  const { user, login, settings } = useApp();
  
  if (!user) {
    return (
      <>
        <ParticleCanvas enabled={settings.particles} performance={settings.performance} />
        <LoginPage onLogin={login} />
      </>
    );
  }
  
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <ParticleCanvas enabled={settings.particles} performance={settings.performance} />
      <Navigation currentPage={currentPage} onNavigate={onNavigate} />
      
      <main className="relative z-10">
        {currentPage === 'dashboard' && <DashboardPage />}
        {currentPage === 'agents' && <AgentsPage />}
        {currentPage === 'projects' && <ProjectsPage />}
        {currentPage === 'profile' && <ProfilePage />}
        {currentPage === 'settings' && <SettingsPage />}
      </main>
    </div>
  );
}