import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { motion } from 'framer-motion';
import { Loader2, AlertTriangle, LogOut } from 'lucide-react';
import { Navigation, StatusBar } from './components';
import {
  AgentsPage,
  ProjectsPage,
  ProjectHistoryPage,
  ProfilePage,
  SettingsPage,
  LoginPage,
} from './pages';
import { fetchAgents, fetchProjects } from './utils/api';
import { Agent, Project, User } from './types';
import { ParticleSystem } from './utils/fixed-particle-system';

// Safe localStorage helper
const safeLocalStorage = {
  getItem: (key: string): string | null => {
    try {
      if (typeof window === 'undefined') return null;
      return localStorage.getItem(key);
    } catch (error) {
      console.warn(`[SafeStorage] Failed to read ${key}:`, error);
      return null;
    }
  },
  setItem: (key: string, value: string): boolean => {
    try {
      if (typeof window === 'undefined') return false;
      localStorage.setItem(key, value);
      return true;
    } catch (error) {
      console.warn(`[SafeStorage] Failed to write ${key}:`, error);
      return false;
    }
  },
  removeItem: (key: string): boolean => {
    try {
      if (typeof window === 'undefined') return false;
      localStorage.removeItem(key);
      return true;
    } catch (error) {
      console.warn(`[SafeStorage] Failed to remove ${key}:`, error);
      return false;
    }
  },
};

// Animation Manager - controls frame-based animations
class AnimationManager {
  private animationFrameId: number | null = null;
  private isRunning = false;

  start(callback: (timestamp: number) => void): void {
    if (this.isRunning) return;
    this.isRunning = true;

    const animate = (timestamp: number) => {
      callback(timestamp);
      this.animationFrameId = requestAnimationFrame(animate);
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }

  stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
      this.isRunning = false;
    }
  }

  cleanup(): void {
    this.stop();
  }
}

// Interactive Particle Effects - handles mouse-based interactions
class InteractiveParticleEffects {
  private isActive = false;
  private particles: Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    life: number;
  }> = [];

  activate(): void {
    this.isActive = true;
  }

  deactivate(): void {
    this.isActive = false;
    this.particles = [];
  }

  createParticle(x: number, y: number): void {
    if (!this.isActive) return;

    this.particles.push({
      x,
      y,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
      life: 1,
    });

    if (this.particles.length > 50) {
      this.particles.shift();
    }
  }

  update(): void {
    this.particles = this.particles
      .map((p) => ({
        ...p,
        x: p.x + p.vx,
        y: p.y + p.vy,
        vy: p.vy + 0.1,
        life: p.life - 0.02,
      }))
      .filter((p) => p.life > 0);
  }

  cleanup(): void {
    this.deactivate();
    this.particles = [];
  }
}

// Ambient Particle Background - manages ambient particle system
class AmbientParticleBackground {
  private particleSystem: ParticleSystem | null = null;
  private isInitialized = false;

  constructor() {
    try {
      if (typeof window === 'undefined') return;
      const container = document.body;
      if (!container) return;

      this.particleSystem = new ParticleSystem(container, {
        particleCount: 50,
        particleSize: { min: 1, max: 3 },
        particleColor: ['#64f4ac', '#60a5fa', '#f59e0b'],
        enableConnections: true,
        connectionDistance: 150,
        connectionOpacity: 0.3,
        enableMouse: false,
        mouseRadius: 0,
        particleLife: { min: 100, max: 300 },
        gravity: 0,
        wind: 0,
        turbulence: 0.05,
      });
      this.isInitialized = true;
    } catch (error) {
      console.warn('[ParticleBackground] Failed to initialize:', error);
      this.isInitialized = false;
    }
  }

  isReady(): boolean {
    return this.isInitialized && this.particleSystem !== null;
  }

  cleanup(): void {
    try {
      if (this.particleSystem) {
        this.particleSystem.destroy();
        this.particleSystem = null;
      }
      this.isInitialized = false;
    } catch (error) {
      console.warn('[ParticleBackground] Cleanup error:', error);
    }
  }
}

// Page Transition Animation
const pageTransitionVariants = {
  initial: {
    opacity: 0,
    y: 10,
  },
  animate: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.3,
      ease: 'easeInOut',
    },
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: {
      duration: 0.2,
      ease: 'easeInOut',
    },
  },
};

// Error Fallback Component
interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetErrorBoundary,
}) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4"
  >
    <div className="max-w-md w-full bg-slate-800/50 backdrop-blur-xl border border-red-500/30 rounded-2xl p-8 text-center shadow-2xl">
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: 'spring', stiffness: 200, damping: 15 }}
        className="mb-4"
      >
        <AlertTriangle className="w-16 h-16 text-red-500 mx-auto" />
      </motion.div>

      <h2 className="text-2xl font-bold text-white mb-2">Something Went Wrong</h2>
      <p className="text-slate-400 mb-6 text-sm line-clamp-3">{error.message}</p>

      <div className="flex gap-3">
        <motion.button
          onClick={resetErrorBoundary}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="flex-1 px-4 py-2 bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-600 hover:to-cyan-700 text-white font-medium rounded-xl transition-all shadow-lg"
        >
          Try Again
        </motion.button>
        <motion.button
          onClick={() => {
            try {
              if (typeof window !== 'undefined') {
                window.location.reload();
              }
            } catch (error) {
              console.error('Reload error:', error);
            }
          }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white font-medium rounded-xl transition-all"
        >
          Reload Page
        </motion.button>
      </div>
    </div>
  </motion.div>
);

// Loading Screen Component
const LoadingScreen: React.FC = () => (
  <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center relative overflow-hidden">
    {/* Ambient background gradient */}
    <div className="absolute inset-0 bg-gradient-to-t from-cyan-900/10 via-transparent to-transparent" />

    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="text-center relative z-10"
    >
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.1, duration: 0.5 }}
        className="mb-8"
      >
        <motion.div
          animate={{
            boxShadow: [
              '0 0 20px rgba(100, 244, 172, 0.3)',
              '0 0 40px rgba(100, 244, 172, 0.6)',
              '0 0 20px rgba(100, 244, 172, 0.3)',
            ],
          }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-cyan-400 to-green-400 rounded-full flex items-center justify-center shadow-lg"
        >
          <Loader2 className="w-10 h-10 text-slate-900 animate-spin" />
        </motion.div>
        <h1 className="text-4xl font-bold text-white mb-2">AgentFlow</h1>
        <p className="text-cyan-400 font-mono text-sm">Neural Command Center</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="w-64 h-2 bg-slate-700/50 rounded-full overflow-hidden mx-auto backdrop-blur"
      >
        <motion.div
          className="h-full bg-gradient-to-r from-cyan-400 via-green-400 to-cyan-400"
          initial={{ width: '0%' }}
          animate={{ width: '100%' }}
          transition={{ duration: 2.5, ease: 'easeInOut', repeat: Infinity }}
        />
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="text-slate-400 mt-6 font-mono text-sm"
      >
        Initializing Advanced AI Platform...
      </motion.p>
    </motion.div>
  </div>
);

// Main App Component
type PageType = 'agents' | 'projects' | 'history' | 'profile' | 'settings' | 'login';

interface AppState {
  currentPage: PageType;
  agents: Agent[];
  projects: Project[];
  user: User | null;
  isLoading: boolean;
  error: string | null;
}

const App: React.FC = () => {
  const [state, setState] = useState<AppState>({
    currentPage: 'login',
    agents: [],
    projects: [],
    user: null,
    isLoading: true,
    error: null,
  });

  const [animationManager] = useState(() => new AnimationManager());
  const [particleEffects] = useState(() => new InteractiveParticleEffects());
  const [particleBackground, setParticleBackground] = useState<AmbientParticleBackground | null>(
    null
  );

  // Initialize app data
  useEffect(() => {
    const initializeApp = async (): Promise<void> => {
      try {
        // Check for saved user
        const savedUserJson = safeLocalStorage.getItem('agentflow_user');
        let initialUser: User | null = null;

        if (savedUserJson) {
          try {
            initialUser = JSON.parse(savedUserJson);
          } catch (parseError) {
            console.error('[App] Failed to parse saved user:', parseError);
            safeLocalStorage.removeItem('agentflow_user');
          }
        }

        // Fetch data in parallel
        const [agentsData, projectsData] = await Promise.all([
          fetchAgents().catch((error) => {
            console.error('[App] Failed to fetch agents:', error);
            return [];
          }),
          fetchProjects().catch((error) => {
            console.error('[App] Failed to fetch projects:', error);
            return [];
          }),
        ]);

        setState((prev) => ({
          ...prev,
          agents: agentsData,
          projects: projectsData,
          user: initialUser,
          currentPage: initialUser ? 'agents' : 'login',
          isLoading: false,
        }));
      } catch (error) {
        console.error('[App] Initialization failed:', error);
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: 'Failed to initialize application',
        }));
      }
    };

    initializeApp();

    // Initialize ambient particles if enabled
    const particlesEnabled = safeLocalStorage.getItem('agentflow_particles_enabled') !== 'false';
    if (particlesEnabled) {
      const background = new AmbientParticleBackground();
      if (background.isReady()) {
        setParticleBackground(background);
      }
    }

    // Cleanup on unmount
    return () => {
      try {
        animationManager.cleanup();
        particleEffects.cleanup();
        particleBackground?.cleanup();
      } catch (error) {
        console.warn('[App] Cleanup error:', error);
      }
    };
  }, []);

  const handleLogin = useCallback((userData: User): void => {
    if (safeLocalStorage.setItem('agentflow_user', JSON.stringify(userData))) {
      setState((prev) => ({
        ...prev,
        user: userData,
        currentPage: 'agents',
      }));
    }
  }, []);

  const handleLogout = useCallback((): void => {
    safeLocalStorage.removeItem('agentflow_user');
    setState((prev) => ({
      ...prev,
      user: null,
      currentPage: 'login',
    }));
  }, []);

  const handleUpdateUser = useCallback((updatedData: Partial<User>): void => {
    setState((prev) => {
      if (!prev.user) return prev;

      const updatedUser = { ...prev.user, ...updatedData };
      if (safeLocalStorage.setItem('agentflow_user', JSON.stringify(updatedUser))) {
        return { ...prev, user: updatedUser };
      }
      return prev;
    });
  }, []);

  const handleNavigate = useCallback((page: PageType): void => {
    setState((prev) => ({ ...prev, currentPage: page }));

    // Smooth scroll to top
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, []);

  if (state.isLoading) {
    return <LoadingScreen />;
  }

  const renderPage = (): React.ReactNode => {
    switch (state.currentPage) {
      case 'login':
        return <LoginPage onLogin={handleLogin} />;
      case 'agents':
        return <AgentsPage agents={state.agents} />;
      case 'projects':
        return <ProjectsPage projects={state.projects} agents={state.agents} />;
      case 'history':
        return <ProjectHistoryPage />;
      case 'profile':
        return state.user ? (
          <ProfilePage user={state.user} onUpdateUser={handleUpdateUser} />
        ) : null;
      case 'settings':
        return <SettingsPage />;
      default:
        return <AgentsPage agents={state.agents} />;
    }
  };

  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onReset={() => handleNavigate('agents')}
      onError={(error) => {
        console.error('[ErrorBoundary] Caught error:', error);
      }}
    >
      <div className="app relative min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        {/* Navigation */}
        {state.user && (
          <Navigation
            currentPage={state.currentPage}
            onNavigate={handleNavigate}
            user={state.user}
          />
        )}

        {/* Main Content */}
        <motion.main
          key={state.currentPage}
          variants={pageTransitionVariants}
          initial="initial"
          animate="animate"
          exit="exit"
          className={`main-content transition-all duration-300 ${
            state.currentPage === 'login' ? 'login-page' : ''
          }`}
        >
          <Suspense fallback={<LoadingScreen />}>
            {renderPage()}
          </Suspense>
        </motion.main>

        {/* Status Bar */}
        {state.user && state.currentPage !== 'login' && (
          <StatusBar
            agents={state.agents}
            projects={state.projects}
            notifications={3}
          />
        )}

        {/* Mobile Logout Button */}
        {state.user && state.currentPage !== 'login' && (
          <motion.button
            onClick={handleLogout}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            className="fixed bottom-6 right-6 md:hidden z-40 p-4 bg-slate-800/70 backdrop-blur-lg border border-slate-700/50 rounded-full hover:bg-slate-700/70 transition-all shadow-lg"
            title="Logout"
            aria-label="Logout"
          >
            <LogOut className="w-5 h-5 text-white" />
          </motion.button>
        )}

        {/* Error State Indicator */}
        {state.error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 left-6 md:bottom-8 md:left-8 z-40 max-w-sm bg-red-500/10 border border-red-500/50 rounded-xl p-4 backdrop-blur-lg"
          >
            <p className="text-red-400 text-sm font-medium">{state.error}</p>
          </motion.div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default App;