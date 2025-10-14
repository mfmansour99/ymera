import React, { useState, useEffect, useRef, createContext, useContext, useMemo, useCallback } from 'react';
import {
  Home, Cpu, Folder, Users, Settings, LogOut, Activity, Search, Plus, X, Code, 
  Camera, Database, Shield, Zap, TrendingUp, CheckCircle, AlertCircle, Bell,
  Mail, Eye, EyeOff, Save, User, BarChart3, Clock, MessageSquare, 
  Edit, Trash2, Download, Upload, Calendar, FileText
} from 'lucide-react';

// ============================================================================
// TYPES
// ============================================================================

interface UserType {
  id: string;
  username: string;
  email: string;
  role?: string;
  bio?: string;
}

interface AgentType {
  id: number;
  name: string;
  type: string;
  status: 'idle' | 'thinking' | 'working' | 'completed';
  tasks: number;
  efficiency: number;
  color: string;
  icon: string;
}

interface ProjectType {
  id: number;
  name: string;
  status: 'planning' | 'in_progress' | 'completed' | 'on_hold';
  progress: number;
  team: number;
  deadline: string;
  priority?: 'low' | 'medium' | 'high';
}

interface FileRecord {
  id: string;
  name: string;
  size: number;
  type: string;
  dataUrl: string;
  ts: number;
}

// ============================================================================
// ✅ FIXED: IN-MEMORY FILE SERVICE (NO LOCALSTORAGE)
// ============================================================================

const MAX_FILE_SIZE = 2 * 1024 * 1024; // 2MB
const MAX_STORAGE = 5 * 1024 * 1024; // 5MB

class EnhancedFileService {
  private store: { agents: Record<number, FileRecord[]>; projects: Record<number, FileRecord[]> };

  constructor() {
    this.store = { agents: {}, projects: {} };
    console.warn('📁 File Service: Using in-memory storage. Files will not persist on refresh.');
  }

  private generateId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  private toDataUrl = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      if (file.size > MAX_FILE_SIZE) {
        reject(new Error(`File size exceeds ${MAX_FILE_SIZE / 1024 / 1024}MB limit`));
        return;
      }
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result as string);
      fr.onerror = reject;
      fr.readAsDataURL(file);
    });
  };

  async uploadToAgent(agentId: number, file: File): Promise<FileRecord> {
    const id = this.generateId();
    const dataUrl = await this.toDataUrl(file);
    const rec: FileRecord = { id, name: file.name, size: file.size, type: file.type, dataUrl, ts: Date.now() };
    this.store.agents[agentId] = this.store.agents[agentId] || [];
    this.store.agents[agentId].push(rec);
    return rec;
  }

  listForAgent(agentId: number): FileRecord[] {
    return (this.store.agents[agentId] || []).slice().sort((a, b) => b.ts - a.ts);
  }

  deleteForAgent(agentId: number, fileId: string) {
    this.store.agents[agentId] = (this.store.agents[agentId] || []).filter(f => f.id !== fileId);
  }

  getTotalUsage(): number {
    return JSON.stringify(this.store).length;
  }

  getUsagePercentage(): number {
    return (this.getTotalUsage() / MAX_STORAGE) * 100;
  }
}

const fileService = new EnhancedFileService();

// ============================================================================
// APP CONTEXT
// ============================================================================

interface AppContextType {
  user: UserType | null;
  agents: AgentType[];
  projects: ProjectType[];
  notifications: string[];
  settings: { animations: boolean; particles: boolean; performance: string };
  login: (username: string) => void;
  logout: () => void;
  addNotification: (message: string) => void;
  updateSettings: (settings: Partial<AppContextType['settings']>) => void;
  updateAgent: (id: number, updates: Partial<AgentType>) => void;
  addAgent: (agent: AgentType) => void;
}

const AppContext = createContext<AppContextType | null>(null);

const useApp = () => {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within AppProvider');
  return context;
};

// ============================================================================
// APP PROVIDER
// ============================================================================

const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserType | null>(null);
  const [notifications, setNotifications] = useState<string[]>([]);
  const [settings, setSettings] = useState({ animations: true, particles: true, performance: 'balanced' });
  
  const [agents, setAgents] = useState<AgentType[]>([
    { id: 1, name: 'DataMind', type: 'code-analyzer', status: 'working', tasks: 24, efficiency: 94, color: '#00f5ff', icon: 'Code' },
    { id: 2, name: 'CodeGen', type: 'ui-designer', status: 'working', tasks: 18, efficiency: 91, color: '#ff3366', icon: 'Camera' },
    { id: 3, name: 'CloudOps', type: 'backend-dev', status: 'thinking', tasks: 12, efficiency: 88, color: '#00ff88', icon: 'Database' },
    { id: 4, name: 'SecureNet', type: 'security', status: 'working', tasks: 15, efficiency: 96, color: '#ffd700', icon: 'Shield' }
  ]);

  const [projects, setProjects] = useState<ProjectType[]>([
    { id: 1, name: 'AI Analytics', status: 'in_progress', progress: 65, team: 4, deadline: '2025-11-15', priority: 'high' },
    { id: 2, name: 'Smart Automation', status: 'in_progress', progress: 42, team: 3, deadline: '2025-11-30', priority: 'medium' },
    { id: 3, name: 'Data Pipeline', status: 'planning', progress: 15, team: 2, deadline: '2025-12-15', priority: 'low' }
  ]);

  const login = useCallback((username: string) => {
    setUser({ id: '1', username, email: `${username}@ymera.ai` });
  }, []);

  const logout = useCallback(() => setUser(null), []);

  const addNotification = useCallback((message: string) => {
    const id = Date.now().toString();
    setNotifications(prev => [...prev, message]);
    setTimeout(() => setNotifications(prev => prev.filter((_, i) => i !== 0)), 5000);
  }, []);

  const updateSettings = useCallback((newSettings: Partial<typeof settings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const updateAgent = useCallback((id: number, updates: Partial<AgentType>) => {
    setAgents(prev => prev.map(a => a.id === id ? { ...a, ...updates } : a));
  }, []);

  const addAgent = useCallback((agent: AgentType) => {
    setAgents(prev => [...prev, agent]);
  }, []);

  const value: AppContextType = {
    user, agents, projects, notifications, settings,
    login, logout, addNotification, updateSettings,
    updateAgent, addAgent
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

// ============================================================================
// UTILITIES
// ============================================================================

const formatBytes = (bytes: number): string => {
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(2)} KB`;
  return `${bytes} B`;
};

// ============================================================================
// LOGO COMPONENT
// ============================================================================

const YmeraLogo: React.FC<{ size?: number }> = React.memo(({ size = 36 }) => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setPhase(p => (p + 1) % 360), 50);
    return () => clearInterval(interval);
  }, []);

  const color1 = `hsl(${phase}, 70%, 60%)`;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <div style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox="0 0 100 100">
          <path
            d="M50 10 L90 30 L90 70 L50 90 L10 70 L10 30 Z"
            stroke={color1}
            strokeWidth="3"
            fill="none"
          />
          <circle cx="50" cy="50" r="8" fill={color1} />
        </svg>
      </div>
      <span style={{
        fontSize: 24,
        fontWeight: 700,
        background: 'linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent'
      }}>
        Ymera
      </span>
    </div>
  );
});

// ============================================================================
// PARTICLES
// ============================================================================

const Particles: React.FC<{ enabled: boolean; performance: string }> = React.memo(({ enabled, performance }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!enabled) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const count = performance === 'high' ? 80 : performance === 'balanced' ? 50 : 30;
    const particles = Array.from({ length: count }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2 + 1,
      color: ['#00f5ff', '#ff00aa', '#00ff88'][Math.floor(Math.random() * 3)]
    }));

    let frame: number;
    const animate = () => {
      ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      particles.forEach(p => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      });

      frame = requestAnimationFrame(animate);
    };
    animate();

    return () => cancelAnimationFrame(frame);
  }, [enabled, performance]);

  if (!enabled) return null;
  return <canvas ref={canvasRef} style={{ position: 'fixed', inset: 0, opacity: 0.3, pointerEvents: 'none', zIndex: 0 }} />;
});

// ============================================================================
// NAVIGATION
// ============================================================================

const Navigation: React.FC<{ currentPage: string; onNavigate: (page: string) => void }> = React.memo(({ currentPage, onNavigate }) => {
  const { user, logout, notifications } = useApp();
  const [showNotif, setShowNotif] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const navItems = useMemo(() => [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'agents', label: 'Agents', icon: Cpu },
    { id: 'projects', label: 'Projects', icon: Folder },
    { id: 'settings', label: 'Settings', icon: Settings }
  ], []);

  if (!user) return null;

  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
      backdropFilter: 'blur(20px)', backgroundColor: 'rgba(10, 10, 10, 0.95)',
      borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
    }}>
      <div style={{ maxWidth: '80rem', margin: '0 auto', padding: '0 2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '4rem' }}>
        <div onClick={() => onNavigate('dashboard')} style={{ cursor: 'pointer' }}>
          <YmeraLogo size={36} />
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          {navItems.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => onNavigate(id)}
              style={{
                padding: '0.5rem 0.75rem',
                display: 'flex', alignItems: 'center', gap: 8,
                backgroundColor: currentPage === id ? 'rgba(6,182,212,0.2)' : 'transparent',
                border: '1px solid rgba(255,255,255,0.1)',
                color: currentPage === id ? '#06b6d4' : '#9ca3af',
                borderRadius: 8, cursor: 'pointer', transition: 'all 0.2s'
              }}
            >
              <Icon size={18} />
              <span>{label}</span>
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button
            onClick={() => setShowNotif(!showNotif)}
            aria-label={`${notifications.length} notifications`}
            style={{
              padding: '0.5rem', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(255,255,255,0.05)', cursor: 'pointer', position: 'relative'
            }}
          >
            <Bell size={20} style={{ color: '#9ca3af' }} />
            {notifications.length > 0 && (
              <span style={{
                position: 'absolute', top: 4, right: 4,
                width: 8, height: 8, background: '#ef4444', borderRadius: '50%'
              }} />
            )}
          </button>

          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '0.5rem 0.75rem', borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(255,255,255,0.05)', cursor: 'pointer'
            }}
          >
            <div style={{
              width: 28, height: 28, borderRadius: 8,
              background: 'linear-gradient(to right, #06b6d4, #8b5cf6)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: 'white', fontWeight: 700, fontSize: 14
            }}>
              {user.username[0].toUpperCase()}
            </div>
            <span style={{ color: 'white' }}>{user.username}</span>
          </button>
        </div>
      </div>
    </nav>
  );
});

// ============================================================================
// LOGIN PAGE
// ============================================================================

const LoginPage: React.FC = () => {
  const { login } = useApp();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = () => {
    if (!username.trim()) {
      setError('Username is required');
      return;
    }
    setError('');
    setLoading(true);
    setTimeout(() => {
      login(username);
      setLoading(false);
    }, 800);
  };

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '1rem', background: 'linear-gradient(135deg, #0a0a0a 0%, #1e3a8a 50%, #581c87 100%)'
    }}>
      <div style={{ maxWidth: 450, width: '100%', position: 'relative', zIndex: 10 }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
            <YmeraLogo size={80} />
          </div>
          <p style={{ color: '#9ca3af', fontSize: 18 }}>AI Project Management Platform</p>
        </div>

        <div style={{
          backdropFilter: 'blur(16px)', background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: 16, padding: '2rem'
        }}>
          {error && (
            <div style={{
              padding: 12, background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8,
              color: '#ef4444', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8
            }}>
              <AlertCircle size={16} /> {error}
            </div>
          )}

          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', color: '#d1d5db', marginBottom: 8 }}>Username</label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSubmit()}
              placeholder="Enter your username"
              autoFocus
              style={{
                width: '100%', padding: 12, background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
                color: 'white', outline: 'none'
              }}
            />
          </div>

          <div style={{ marginBottom: 16, position: 'relative' }}>
            <label style={{ display: 'block', color: '#d1d5db', marginBottom: 8 }}>Password</label>
            <input
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSubmit()}
              placeholder="Enter your password"
              style={{
                width: '100%', padding: 12, paddingRight: 48,
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8,
                color: 'white', outline: 'none'
              }}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              style={{
                position: 'absolute', right: 12, top: 38,
                background: 'none', border: 'none', color: '#9ca3af', cursor: 'pointer'
              }}
            >
              {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading || !username.trim()}
            style={{
              width: '100%', padding: 12, borderRadius: 8, border: 'none',
              background: loading || !username.trim() ? '#6b7280' : 'linear-gradient(to right, #06b6d4, #3b82f6)',
              color: 'white', fontWeight: 600, cursor: loading || !username.trim() ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? 'Signing in...' : 'Enter Ymera'}
          </button>

          <p style={{ marginTop: 16, textAlign: 'center', fontSize: 14, color: '#9ca3af' }}>
            Demo Mode - Enter any username
          </p>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// DASHBOARD
// ============================================================================

const DashboardPage: React.FC = () => {
  const { agents, projects, addNotification } = useApp();

  useEffect(() => {
    addNotification('Welcome to Ymera! System online and ready.');
  }, [addNotification]);

  const stats = useMemo(() => ({
    activeAgents: agents.filter(a => a.status === 'working').length,
    totalTasks: agents.reduce((sum, a) => sum + a.tasks, 0),
    avgEfficiency: Math.round(agents.reduce((sum, a) => sum + a.efficiency, 0) / agents.length)
  }), [agents]);

  return (
    <div style={{ padding: '2rem 0' }}>
      <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>Dashboard</h1>
      <p style={{ color: '#9ca3af', marginBottom: '2rem' }}>Real-time insights into your AI operations</p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 16, marginBottom: 32 }}>
        {[
          { title: 'Active Agents', value: stats.activeAgents, icon: Cpu, color: '#00f5ff' },
          { title: 'Projects', value: projects.length, icon: Folder, color: '#ff3366' },
          { title: 'Total Tasks', value: stats.totalTasks, icon: Activity, color: '#00ff88' },
          { title: 'Avg Efficiency', value: `${stats.avgEfficiency}%`, icon: TrendingUp, color: '#ffd700' }
        ].map((stat, i) => (
          <div key={i} style={{
            backdropFilter: 'blur(16px)', background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 16
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <div style={{ padding: 12, background: `${stat.color}20`, borderRadius: 8 }}>
                <stat.icon size={24} style={{ color: stat.color }} />
              </div>
            </div>
            <div style={{ fontSize: 28, fontWeight: 800, color: 'white' }}>{stat.value}</div>
            <div style={{ color: '#9ca3af', fontSize: 14 }}>{stat.title}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
        <div style={{
          backdropFilter: 'blur(16px)', background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 16
        }}>
          <h3 style={{ fontSize: 20, fontWeight: 700, color: 'white', marginBottom: 16 }}>Active Agents</h3>
          <div style={{ display: 'grid', gap: 12 }}>
            {agents.filter(a => a.status === 'working').map(agent => (
              <div key={agent.id} style={{
                display: 'flex', alignItems: 'center', gap: 12,
                background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 12, padding: 12
              }}>
                <div style={{
                  width: 48, height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: `${agent.color}20`, borderRadius: 8, border: `1px solid ${agent.color}40`
                }}>
                  <Cpu size={24} style={{ color: agent.color }} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 700, color: 'white' }}>{agent.name}</div>
                  <div style={{ color: '#9ca3af', fontSize: 14 }}>{agent.tasks} tasks • {agent.efficiency}%</div>
                </div>
                <div style={{ width: 8, height: 8, background: '#10b981', borderRadius: 999 }} />
              </div>
            ))}
          </div>
        </div>

        <div style={{
          backdropFilter: 'blur(16px)', background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 16
        }}>
          <h3 style={{ fontSize: 20, fontWeight: 700, color: 'white', marginBottom: 16 }}>Active Projects</h3>
          <div style={{ display: 'grid', gap: 12 }}>
            {projects.filter(p => p.status === 'in_progress').map(project => (
              <div key={project.id} style={{
                background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 12, padding: 12
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontWeight: 700, color: 'white' }}>{project.name}</span>
                  <span style={{ color: '#06b6d4', fontWeight: 700 }}>{project.progress}%</span>
                </div>
                <div style={{ height: 8, background: 'rgba(255,255,255,0.08)', borderRadius: 999, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', width: `${project.progress}%`,
                    background: 'linear-gradient(to right, #06b6d4, #3b82f6)', transition: 'width 0.5s'
                  }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// AGENTS PAGE
// ============================================================================

const AgentsPage: React.FC = () => {
  const { agents } = useApp();

  return (
    <div style={{ padding: '2rem 0' }}>
      <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: 32 }}>AI Agents</h1>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
        {agents.map(agent => {
          const iconMap: Record<string, any> = { Code, Camera, Database, Shield, Zap };
          const Icon = iconMap[agent.icon] || Zap;
          
          return (
            <div key={agent.id} style={{
              position: 'relative', backdropFilter: 'blur(16px)',
              background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 12, padding: 16, cursor: 'pointer'
            }}>
              <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, height: 4,
                background: `linear-gradient(to right, ${agent.color}, transparent)`
              }} />
              
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{
                  padding: 12, background: `${agent.color}20`,
                  borderRadius: 8, border: `1px solid ${agent.color}40`
                }}>
                  <Icon size={24} style={{ color: agent.color }} />
                </div>
                <div style={{
                  padding: '4px 12px',
                  background: agent.status === 'working' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(251, 191, 36, 0.2)',
                  borderRadius: 999, fontSize: 12, fontWeight: 600,
                  color: agent.status === 'working' ? '#10b981' : '#fbbf24'
                }}>
                  {agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}
                </div>
              </div>

              <h3 style={{ fontSize: 18, fontWeight: 800, color: 'white', marginBottom: 8 }}>{agent.name}</h3>
              <p style={{ fontSize: 14, color: '#9ca3af', marginBottom: 12 }}>{agent.type.toUpperCase()}</p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 10 }}>
                <div>
                  <div style={{ fontSize: 24, fontWeight: 800, color: agent.color }}>{agent.tasks}</div>
                  <div style={{ fontSize: 12, color: '#9ca3af' }}>Tasks</div>
                </div>
                <div>
                  <div style={{ fontSize: 24, fontWeight: 800, color: agent.color }}>{agent.efficiency}%</div>
                  <div style={{ fontSize: 12, color: '#9ca3af' }}>Efficiency</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ============================================================================
// SETTINGS PAGE
// ============================================================================

const SettingsPage: React.FC = () => {
  const { settings, updateSettings } = useApp();

  return (
    <div style={{ padding: '2rem 0', maxWidth: '56rem', margin: '0 auto' }}>
      <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: 32 }}>Settings</h1>

      <div style={{
        backdropFilter: 'blur(16px)', background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 24
      }}>
        <h3 style={{ fontSize: 20, fontWeight: 700, color: 'white', marginBottom: 16 }}>
          General Settings
        </h3>

        <div style={{ display: 'grid', gap: 16 }}>
          {[
            { key: 'animations', label: 'Enable Animations', description: 'UI animations and transitions' },
            { key: 'particles', label: 'Particle Effects', description: 'Background particle system' }
          ].map(setting => (
            <div
              key={setting.key}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: 16, background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8
              }}
            >
              <div>
                <div style={{ fontWeight: 600, color: 'white', marginBottom: 4 }}>
                  {setting.label}
                </div>
                <div style={{ fontSize: 14, color: '#9ca3af' }}>
                  {setting.description}
                </div>
              </div>
              <button
                onClick={() => updateSettings({ [setting.key]: !settings[setting.key as keyof typeof settings] })}
                style={{
                  position: 'relative', width: 48, height: 24, borderRadius: 999,
                  background: settings[setting.key as keyof typeof settings] ? '#06b6d4' : '#6b7280',
                  border: 'none', cursor: 'pointer', transition: 'background 0.2s'
                }}
              >
                <div style={{
                  position: 'absolute', top: 2, left: settings[setting.key as keyof typeof settings] ? 26 : 2,
                  width: 20, height: 20, borderRadius: '50%', background: 'white',
                  transition: 'left 0.2s'
                }} />
              </button>
            </div>
          ))}

          <div>
            <label style={{ display: 'block', color: '#9ca3af', marginBottom: 8 }}>
              Performance Mode
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
              {['low', 'balanced', 'high'].map(mode => (
                <button
                  key={mode}
                  onClick={() => updateSettings({ performance: mode })}
                  style={{
                    padding: 16, borderRadius: 8,
                    background: settings.performance === mode ? 'rgba(6, 182, 212, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${settings.performance === mode ? 'rgba(6, 182, 212, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                    color: settings.performance === mode ? '#06b6d4' : 'white',
                    cursor: 'pointer', fontWeight: 600, textTransform: 'capitalize'
                  }}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  return (
    <AppProvider>
      <AppContent currentPage={currentPage} onNavigate={setCurrentPage} />
    </AppProvider>
  );
}

function AppContent({ currentPage, onNavigate }: { currentPage: string; onNavigate: (page: string) => void }) {
  const { user, settings } = useApp();

  if (!user) {
    return (
      <>
        <Particles enabled={settings.particles} performance={settings.performance} />
        <LoginPage />
      </>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0a0a0a', color: 'white' }}>
      <Particles enabled={settings.particles} performance={settings.performance} />
      <Navigation currentPage={currentPage} onNavigate={onNavigate} />

      <main style={{ paddingTop: '5rem', paddingLeft: '2rem', paddingRight: '2rem', paddingBottom: '3rem', position: 'relative', zIndex: 1 }}>
        <div style={{ maxWidth: '80rem', margin: '0 auto' }}>
          {currentPage === 'dashboard' && <DashboardPage />}
          {currentPage === 'agents' && <AgentsPage />}
          {currentPage === 'settings' && <SettingsPage />}
          {currentPage === 'projects' && (
            <div style={{ padding: '2rem 0' }}>
              <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: 32 }}>Projects</h1>
              <div style={{
                padding: 48, textAlign: 'center',
                backdropFilter: 'blur(16px)', background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12
              }}>
                <Folder size={64} style={{ color: '#06b6d4', margin: '0 auto 16px' }} />
                <p style={{ color: '#9ca3af' }}>Projects page coming soon</p>
              </div>
            </div>
          )}
          {currentPage === 'profile' && (
            <div style={{ padding: '2rem 0' }}>
              <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: 32 }}>Profile</h1>
              <div style={{
                padding: 48, textAlign: 'center',
                backdropFilter: 'blur(16px)', background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12
              }}>
                <User size={64} style={{ color: '#06b6d4', margin: '0 auto 16px' }} />
                <p style={{ color: '#9ca3af' }}>Profile page coming soon</p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}