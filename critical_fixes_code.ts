// ============================================================================
// CRITICAL FIX #1: Complete fixed-particle-system.ts with proper cleanup
// ============================================================================

import { ParticleConfig } from '../types/animation';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  opacity: number;
  life: number;
  maxLife: number;
  angle: number;
  rotationSpeed: number;
  pulse: number;
  pulseSpeed: number;
  currentSize: number;
}

export class ParticleSystem {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private particles: Particle[] = [];
  private animationId: number | null = null;
  private isRunning: boolean = false;
  private config: ParticleConfig;
  private container: HTMLElement;
  private mouse: {
    x: number;
    y: number;
    isMoving: boolean;
  };
  private mouseTimeout: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private dpr: number = 1;

  constructor(container: HTMLElement, options: Partial<ParticleConfig> = {}) {
    this.config = {
      particleCount: 100,
      particleSize: { min: 1, max: 4 },
      particleSpeed: { min: 0.5, max: 2 },
      particleColor: ['#00f5ff', '#ff00aa', '#00ff88', '#ffaa00'],
      connectionDistance: 150,
      connectionOpacity: 0.2,
      enableConnections: true,
      enableMouse: true,
      mouseRadius: 200,
      particleLife: { min: 100, max: 500 },
      gravity: 0,
      wind: 0,
      turbulence: 0.1,
      ...options,
    };

    this.mouse = {
      x: 0,
      y: 0,
      isMoving: false,
    };
    this.container = container;
    
    try {
      this.dpr = window.devicePixelRatio || 1;
    } catch {
      this.dpr = 1;
    }

    this.canvas = document.createElement('canvas');
    const ctx = this.canvas.getContext('2d', { alpha: true });
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;
    
    this.init();
  }

  private init(): void {
    this.createCanvas();
    this.setupEventListeners();
    this.createParticles();
    this.start();
  }

  private createCanvas(): void {
    this.canvas.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      pointer-events: none;
      z-index: 1;
      width: 100%;
      height: 100%;
    `;
    this.container.appendChild(this.canvas);
    this.resize();
  }

  private setupEventListeners(): void {
    this.container.addEventListener('mousemove', this.handleMouseMove);
    this.container.addEventListener('mouseleave', this.handleMouseLeave);
    
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
    });
    this.resizeObserver.observe(this.container);
    
    if (!this.resizeObserver) {
      window.addEventListener('resize', this.handleResize);
    }
  }

  private handleMouseMove = (e: MouseEvent): void => {
    try {
      const rect = this.container.getBoundingClientRect();
      this.mouse.x = e.clientX - rect.left;
      this.mouse.y = e.clientY - rect.top;
      this.mouse.isMoving = true;

      if (this.mouseTimeout) {
        clearTimeout(this.mouseTimeout);
      }
      
      this.mouseTimeout = window.setTimeout(() => {
        this.mouse.isMoving = false;
      }, 100);
    } catch (error) {
      console.error('Error handling mouse move:', error);
    }
  };

  private handleMouseLeave = (): void => {
    this.mouse.isMoving = false;
    if (this.mouseTimeout) {
      clearTimeout(this.mouseTimeout);
      this.mouseTimeout = null;
    }
  };

  private handleResize = (): void => {
    this.resize();
  };

  private resize(): void {
    try {
      const rect = this.container.getBoundingClientRect();
      this.dpr = window.devicePixelRatio || 1;
      
      this.canvas.width = rect.width * this.dpr;
      this.canvas.height = rect.height * this.dpr;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
      
      this.ctx.scale(this.dpr, this.dpr);
    } catch (error) {
      console.error('Error during resize:', error);
    }
  }

  private createParticles(): void {
    this.particles = Array.from({ length: this.config.particleCount }, () => 
      this.createParticle()
    );
  }

  private createParticle(x?: number, y?: number): Particle {
    const colors = this.config.particleColor;
    const color = colors[Math.floor(Math.random() * colors.length)];
    const size = this.config.particleSize.min + 
                 Math.random() * (this.config.particleSize.max - this.config.particleSize.min);
    const life = this.config.particleLife.min + 
                 Math.random() * (this.config.particleLife.max - this.config.particleLife.min);

    const canvasWidth = this.canvas.width / this.dpr;
    const canvasHeight = this.canvas.height / this.dpr;

    return {
      x: x !== undefined ? x : Math.random() * canvasWidth,
      y: y !== undefined ? y : Math.random() * canvasHeight,
      vx: (Math.random() - 0.5) * this.config.particleSpeed.max,
      vy: (Math.random() - 0.5) * this.config.particleSpeed.max,
      size,
      color,
      opacity: Math.random() * 0.5 + 0.3,
      life,
      maxLife: life,
      angle: Math.random() * Math.PI * 2,
      rotationSpeed: (Math.random() - 0.5) * 0.1,
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.02 + Math.random() * 0.02,
      currentSize: size,
    };
  }

  private updateParticle(particle: Particle): void {
    particle.vx += this.config.wind;
    particle.vy += this.config.gravity;
    
    particle.vx += (Math.random() - 0.5) * this.config.turbulence;
    particle.vy += (Math.random() - 0.5) * this.config.turbulence;

    if (this.config.enableMouse && this.mouse.isMoving) {
      const dx = this.mouse.x - particle.x;
      const dy = this.mouse.y - particle.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < this.config.mouseRadius && distance > 0) {
        const force = (this.config.mouseRadius - distance) / this.config.mouseRadius;
        const angle = Math.atan2(dy, dx);
        particle.vx += Math.cos(angle) * force * 0.5;
        particle.vy += Math.sin(angle) * force * 0.5;
      }
    }

    particle.vx *= 0.99;
    particle.vy *= 0.99;

    particle.x += particle.vx;
    particle.y += particle.vy;
    
    particle.angle += particle.rotationSpeed;
    particle.pulse += particle.pulseSpeed;
    particle.currentSize = particle.size + Math.sin(particle.pulse) * particle.size * 0.2;

    const canvasWidth = this.canvas.width / this.dpr;
    const canvasHeight = this.canvas.height / this.dpr;
    
    if (particle.x < 0 || particle.x > canvasWidth) {
      particle.vx *= -0.8;
      particle.x = Math.max(0, Math.min(canvasWidth, particle.x));
    }
    if (particle.y < 0 || particle.y > canvasHeight) {
      particle.vy *= -0.8;
      particle.y = Math.max(0, Math.min(canvasHeight, particle.y));
    }

    particle.life--;
    particle.opacity = Math.max(0, (particle.life / particle.maxLife) * 0.8);
    
    if (particle.life <= 0) {
      Object.assign(particle, this.createParticle());
    }
  }

  private drawParticle(particle: Particle): void {
    this.ctx.save();
    this.ctx.globalAlpha = particle.opacity;

    const gradient = this.ctx.createRadialGradient(
      particle.x, particle.y, 0,
      particle.x, particle.y, particle.currentSize * 3
    );
    gradient.addColorStop(0, particle.color);
    gradient.addColorStop(1, 'transparent');
    
    this.ctx.fillStyle = gradient;
    this.ctx.beginPath();
    this.ctx.arc(particle.x, particle.y, particle.currentSize * 3, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.fillStyle = particle.color;
    this.ctx.beginPath();
    this.ctx.arc(particle.x, particle.y, particle.currentSize, 0, Math.PI * 2);
    this.ctx.fill();
    
    this.ctx.restore();
  }

  private drawConnections(): void {
    if (!this.config.enableConnections) return;

    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const p1 = this.particles[i];
        const p2 = this.particles[j];
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < this.config.connectionDistance) {
          const opacity = (1 - distance / this.config.connectionDistance) * 
                         this.config.connectionOpacity * 
                         Math.min(p1.opacity, p2.opacity);
          
          this.ctx.save();
          this.ctx.globalAlpha = opacity;
          this.ctx.strokeStyle = p1.color;
          this.ctx.lineWidth = 1;
          this.ctx.beginPath();
          this.ctx.moveTo(p1.x, p1.y);
          this.ctx.lineTo(p2.x, p2.y);
          this.ctx.stroke();
          this.ctx.restore();
        }
      }
    }
  }

  private animate = (): void => {
    if (!this.isRunning) return;

    const canvasWidth = this.canvas.width / this.dpr;
    const canvasHeight = this.canvas.height / this.dpr;
    
    this.ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    this.particles.forEach(particle => {
      this.updateParticle(particle);
      this.drawParticle(particle);
    });
    
    this.drawConnections();
    
    this.animationId = requestAnimationFrame(this.animate);
  };

  public start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.animate();
  }

  public pause(): void {
    this.isRunning = false;
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  public resume(): void {
    if (!this.isRunning) {
      this.start();
    }
  }

  public destroy(): void {
    this.pause();
    
    try {
      this.container.removeEventListener('mousemove', this.handleMouseMove);
      this.container.removeEventListener('mouseleave', this.handleMouseLeave);
      
      if (this.resizeObserver) {
        this.resizeObserver.disconnect();
        this.resizeObserver = null;
      } else {
        window.removeEventListener('resize', this.handleResize);
      }
      
      if (this.mouseTimeout) {
        clearTimeout(this.mouseTimeout);
        this.mouseTimeout = null;
      }
      
      if (this.canvas && this.canvas.parentNode) {
        this.canvas.parentNode.removeChild(this.canvas);
      }
      
      this.particles = [];
    } catch (error) {
      console.error('Error during particle system destruction:', error);
    }
  }

  public addBurst(x: number, y: number, count: number = 10): void {
    try {
      for (let i = 0; i < count; i++) {
        const particle = this.createParticle(x, y);
        const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.3;
        const speed = 2 + Math.random() * 3;
        particle.vx = Math.cos(angle) * speed;
        particle.vy = Math.sin(angle) * speed;
        this.particles.push(particle);
      }
    } catch (error) {
      console.error('Error adding particle burst:', error);
    }
  }
}

// ============================================================================
// CRITICAL FIX #2: Enhanced Error Boundary Component
// ============================================================================

import React, { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: (error: Error) => ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ? (
        this.props.fallback(this.state.error!)
      ) : (
        <div style={{
          padding: '2rem',
          backdropFilter: 'blur(16px)',
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '12px',
          color: '#ef4444'
        }}>
          <h2>Something went wrong</h2>
          <p>{this.state.error?.message || 'An unexpected error occurred'}</p>
        </div>
      );
    }

    return this.props.children;
  }
}

// ============================================================================
// CRITICAL FIX #3: Enhanced API Service with Proper Typing & Error Handling
// ============================================================================

import { Agent, Project, User } from './types';

interface ApiResponse<T> {
  data: T;
  error?: string;
  timestamp: number;
}

const mockAgents: Agent[] = [
  {
    id: 'agent-001',
    name: 'CodeAnalyzer Alpha',
    type: 'code_analyzer',
    status: 'working',
    description: 'Advanced code analysis and optimization agent',
    tasksCompleted: 42,
    efficiency: '92%',
    currentPhase: 'code_analysis'
  },
  {
    id: 'agent-002',
    name: 'UIDesigner Beta',
    type: 'ui_designer',
    status: 'idle',
    description: 'Creates beautiful and functional user interfaces',
    tasksCompleted: 28,
    efficiency: '88%',
    currentPhase: null
  },
  {
    id: 'agent-003',
    name: 'BackendMaster',
    type: 'backend_developer',
    status: 'thinking',
    description: 'Builds robust backend systems and APIs',
    tasksCompleted: 35,
    efficiency: '95%',
    currentPhase: 'database_design'
  },
  {
    id: 'agent-004',
    name: 'TestRunner Gamma',
    type: 'testing_agent',
    status: 'working',
    description: 'Comprehensive testing and quality assurance agent',
    tasksCompleted: 56,
    efficiency: '97%',
    currentPhase: 'regression_testing'
  },
  {
    id: 'agent-005',
    name: 'Optimizer Delta',
    type: 'optimization_agent',
    status: 'idle',
    description: 'Performance optimization specialist',
    tasksCompleted: 19,
    efficiency: '91%',
    currentPhase: null
  }
];

const mockProjects: Project[] = [
  {
    id: 'proj-001',
    name: 'E-Commerce Platform',
    description: 'Complete e-commerce solution with payment processing',
    status: 'in_progress',
    priority: 'high',
    startDate: '2024-01-15',
    endDate: null,
    duration: '3 months',
    progress: 65,
    budget: 120000,
    spent: 78000,
    team: ['John Doe', 'Jane Smith', 'Mike Johnson'],
    technologies: ['React', 'Node.js', 'MongoDB', 'AWS'],
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
    description: 'Cross-platform mobile banking application',
    status: 'completed',
    priority: 'high',
    startDate: '2023-11-01',
    endDate: '2024-02-28',
    duration: '4 months',
    progress: 100,
    budget: 150000,
    spent: 145000,
    team: ['Sarah Connor', 'Alex Murphy'],
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
  }
];

const mockUser: User = {
  id: 'user-001',
  name: 'Mohamed Mansour',
  email: 'mohamed@ymera.ai',
  role: 'Administrator',
  avatar: '/assets/avatar.png',
  lastLogin: '2024-03-15T14:30:00Z',
  preferences: {
    theme: 'dark',
    notifications: true,
    language: 'en'
  }
};

const API_DELAYS = {
  FAST: 300,
  NORMAL: 600,
  SLOW: 1000
} as const;

class ApiService {
  private requestCount = 0;
  private shouldSimulateError = false;

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  public setErrorSimulation(shouldError: boolean): void {
    this.shouldSimulateError = shouldError;
  }

  async fetchAgents(): Promise<ApiResponse<Agent[]>> {
    await this.delay(API_DELAYS.NORMAL);
    
    if (this.shouldSimulateError) {
      throw new Error('Failed to fetch agents');
    }
    
    return { data: mockAgents, timestamp: Date.now() };
  }

  async fetchProjects(): Promise<ApiResponse<Project[]>> {
    await this.delay(API_DELAYS.SLOW);
    
    if (this.shouldSimulateError) {
      throw new Error('Failed to fetch projects');
    }
    
    return { data: mockProjects, timestamp: Date.now() };
  }

  async fetchUser(): Promise<ApiResponse<User>> {
    await this.delay(API_DELAYS.FAST);
    
    if (this.shouldSimulateError) {
      throw new Error('Failed to fetch user');
    }
    
    return { data: mockUser, timestamp: Date.now() };
  }

  async updateAgentStatus(agentId: string, status: Agent['status']): Promise<ApiResponse<Agent>> {
    await this.delay(API_DELAYS.NORMAL);
    
    const agent = mockAgents.find(a => a.id === agentId);
    if (!agent) {
      throw new Error('Agent not found');
    }
    
    agent.status = status;
    return { data: agent, timestamp: Date.now() };
  }

  async createProject(project: Omit<Project, 'id' | 'progress' | 'performance'>): Promise<ApiResponse<Project>> {
    await this.delay(API_DELAYS.SLOW);
    
    const newProject: Project = {
      ...project,
      id: `proj-${Date.now()}`,
      progress: 0,
      performance: {
        tasksCompleted: 0,
        bugsFixed: 0,
        codeReviews: 0,
        efficiency: 0
      }
    };
    
    mockProjects.push(newProject);
    return { data: newProject, timestamp: Date.now() };
  }
}

export const apiService = new ApiService();

// ============================================================================
// CRITICAL FIX #4: Secure Form Validation Utility
// ============================================================================

import validator from 'validator';

interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  custom?: (value: any) => boolean;
  message?: string;
}

export const validateEmail = (email: string): boolean => {
  return validator.isEmail(email);
};

export const validatePassword = (password: string): {
  isValid: boolean;
  strength: 'weak' | 'fair' | 'good' | 'strong';
  messages: string[];
} => {
  const messages: string[] = [];
  let score = 0;

  if (password.length >= 8) score++;
  else messages.push('Password must be at least 8 characters');

  if (password.length >= 12) score++;
  if (password.match(/[a-z]/)) score++;
  else messages.push('Password must contain lowercase letters');

  if (password.match(/[A-Z]/)) score++;
  else messages.push('Password must contain uppercase letters');

  if (password.match(/[0-9]/)) score++;
  else messages.push('Password must contain numbers');

  if (password.match(/[^a-zA-Z0-9]/)) score++;
  else messages.push('Password should contain special characters');

  const strengthMap = {
    0: 'weak',
    1: 'weak',
    2: 'fair',
    3: 'good',
    4: 'strong',
    5: 'strong',
    6: 'strong'
  } as const;

  return {
    isValid: score >= 3,
    strength: strengthMap[score as keyof typeof strengthMap],
    messages
  };
};

export const validateUsername = (username: string): boolean => {
  return /^[a-zA-Z0-9_-]{3,}$/.test(username);
};

export const sanitizeInput = (input: string): string => {
  return input
    .trim()
    .replace(/[<>]/g, '')
    .slice(0, 255);
};

// ============================================================================
// CRITICAL FIX #5: React Hook for File Upload with Proper Error Handling
// ============================================================================

import { useState, useCallback } from 'react';

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'application/pdf', 'text/plain'];

interface FileUploadState {
  loading: boolean;
  error: string | null;
  success: boolean;
}

export const useFileUpload = () => {
  const [state, setState] = useState<FileUploadState>({
    loading: false,
    error: null,
    success: false
  });

  const uploadFile = useCallback(async (file: File): Promise<boolean> => {
    setState({ loading: true, error: null, success: false });

    try {
      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        throw new Error(`File size exceeds ${MAX_FILE_SIZE / 1024 / 1024}MB limit`);
      }

      // Validate file type
      if (!ALLOWED_TYPES.includes(file.type)) {
        throw new Error('File type not allowed');
      }

      // Simulate upload
      await new Promise(resolve => setTimeout(resolve, 1500));

      setState({ loading: false, error: null, success: true });
      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setState({ loading: false, error: errorMessage, success: false });
      return false;
    }
  }, []);

  return { ...state, uploadFile };
};

// ============================================================================
// CRITICAL FIX #6: Secure Login Component with Validation
// ============================================================================

import React, { useState, useCallback } from 'react';
import { validateEmail, validatePassword, sanitizeInput } from '../utils/validators';

export const SecureLoginForm: React.FC = () => {
  const [formData, setFormData] = useState({ email: '', password: '' });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [attempts, setAttempts] = useState(0);
  const [isLocked, setIsLocked] = useState(false);
  const MAX_ATTEMPTS = 5;
  const LOCK_DURATION = 15 * 60 * 1000; // 15 minutes

  const validateForm = useCallback((): boolean => {
    const newErrors: Record<string, string> = {};

    if (!validateEmail(formData.email)) {
      newErrors.email = 'Please enter a valid email';
    }

    const passwordValidation = validatePassword(formData.password);
    if (!passwordValidation.isValid) {
      newErrors.password = passwordValidation.messages[0];
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();

    if (isLocked) {
      setErrors({ form: 'Too many attempts. Please try again later.' });
      return;
    }

    if (!validateForm()) return;

    try {
      // Simulate login
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: sanitizeInput(formData.email),
          password: formData.password // Never log this
        })
      });

      if (!response.ok) throw new Error('Login failed');
      setAttempts(0);
      // Handle successful login
    } catch (error) {
      const newAttempts = attempts + 1;
      setAttempts(newAttempts);

      if (newAttempts >= MAX_ATTEMPTS) {
        setIsLocked(true);
        setTimeout(() => {
          setIsLocked(false);
          setAttempts(0);
        }, LOCK_DURATION);
        setErrors({ form: 'Account locked. Try again in 15 minutes.' });
      } else {
        setErrors({ form: `Login failed. ${MAX_ATTEMPTS - newAttempts} attempts remaining.` });
      }
    }
  }, [formData, attempts, isLocked, validateForm]);

  return (
    <form onSubmit={handleSubmit} style={{ display: 'grid', gap: '1rem' }}>
      {errors.form && (
        <div style={{ color: '#ef4444', background: 'rgba(239,68,68,0.1)', padding: '1rem', borderRadius: '0.5rem' }}>
          {errors.form}
        </div>
      )}
      {/* Form fields */}
    </form>
  );
};