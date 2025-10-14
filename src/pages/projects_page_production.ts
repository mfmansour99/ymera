import React, { useState, useCallback, useMemo, useRef, Suspense, memo, ReactNode } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, Environment, Text } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import { ErrorBoundary as ReactErrorBoundary } from 'react-error-boundary';
import {
  Play,
  Pause,
  RotateCcw,
  Settings,
  Eye,
  Plus,
  Search,
  Filter,
  Calendar,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Code,
  Shield,
  Zap,
  Database,
  ArrowRight,
  X,
  Download,
  Upload,
  Edit,
  Trash2,
} from 'lucide-react';
import * as THREE from 'three';

// Project Phase Types with unique visual properties
const PHASE_TYPES = {
  ANALYSIS: {
    color: '#64f4ac',
    geometry: 'cube' as const,
    icon: Code,
    position: [0, 0, 0] as const,
    description: 'Code analysis and requirements gathering',
  },
  DESIGN: {
    color: '#60a5fa',
    geometry: 'pyramid' as const,
    icon: Eye,
    position: [3, 1, 0] as const,
    description: 'System design and architecture planning',
  },
  DEVELOPMENT: {
    color: '#f59e0b',
    geometry: 'cylinder' as const,
    icon: Settings,
    position: [6, 0, 0] as const,
    description: 'Core development and implementation',
  },
  TESTING: {
    color: '#8b5cf6',
    geometry: 'octahedron' as const,
    icon: Shield,
    position: [9, -1, 0] as const,
    description: 'Quality assurance and testing',
  },
  OPTIMIZATION: {
    color: '#ec4899',
    geometry: 'dodecahedron' as const,
    icon: Zap,
    position: [12, 0, 0] as const,
    description: 'Performance optimization and enhancement',
  },
  DEPLOYMENT: {
    color: '#10b981',
    geometry: 'icosahedron' as const,
    icon: Database,
    position: [15, 1, 0] as const,
    description: 'Deployment and final integration',
  },
} as const;

const PHASE_STATUS = {
  pending: { color: '#64748b', intensity: 0.2 },
  active: { color: '#eab308', intensity: 1.0 },
  complete: { color: '#10b981', intensity: 0.6 },
  error: { color: '#ef4444', intensity: 0.8 },
} as const;

type PhaseStatusKey = keyof typeof PHASE_STATUS;
type PhaseTypeKey = keyof typeof PHASE_TYPES;

interface Phase {
  status: PhaseStatusKey;
  completion: number;
  estimatedTime?: string;
}

interface Project {
  id: number;
  name: string;
  status: string;
  completion: number;
  description?: string;
  agents?: string[];
  phases: Partial<Record<PhaseTypeKey, Phase>>;
}

// Geometry Component Factory
const GeometryFactory = memo(
  ({
    geometry,
    isActive,
  }: {
    geometry: string;
    isActive: boolean;
  }) => {
    const size = isActive ? 1.2 : 1.0;

    switch (geometry) {
      case 'cube':
        return <boxGeometry args={[size, size, size]} />;
      case 'pyramid':
        return <coneGeometry args={[size, size * 1.5, 4]} />;
      case 'cylinder':
        return <cylinderGeometry args={[size * 0.8, size * 0.8, size * 1.2]} />;
      case 'octahedron':
        return <octahedronGeometry args={[size]} />;
      case 'dodecahedron':
        return <dodecahedronGeometry args={[size]} />;
      case 'icosahedron':
        return <icosahedronGeometry args={[size]} />;
      default:
        return <sphereGeometry args={[size]} />;
    }
  }
);
GeometryFactory.displayName = 'GeometryFactory';

// 3D Phase Component
const ProjectPhase3D = memo(
  ({
    phase,
    phaseType,
    position,
    onClick,
    isActive,
  }: {
    phase: Phase;
    phaseType: PhaseTypeKey;
    position: [number, number, number];
    onClick: () => void;
    isActive: boolean;
  }) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const particlesRef = useRef<THREE.Group>(null);
    const phaseConfig = PHASE_TYPES[phaseType];
    const statusConfig = PHASE_STATUS[phase.status];

    useFrame(({ clock }) => {
      if (meshRef.current) {
        const baseSpeed =
          phase.status === 'active'
            ? 0.02
            : phase.status === 'complete'
              ? 0.005
              : 0.001;
        meshRef.current.rotation.x += baseSpeed;
        meshRef.current.rotation.y += baseSpeed * 1.2;

        if (phase.status === 'active') {
          const pulse = Math.sin(clock.getElapsedTime() * 4) * 0.2 + 1;
          meshRef.current.scale.setScalar(pulse);
        }
      }

      if (particlesRef.current && phase.status === 'active') {
        particlesRef.current.rotation.y = clock.getElapsedTime() * 2;
        particlesRef.current.rotation.x = clock.getElapsedTime() * 1.5;
      }
    });

    return (
      <Float floatIntensity={phase.status === 'active' ? 3 : 1} speed={2}>
        <group position={position} onClick={onClick}>
          <mesh ref={meshRef}>
            <GeometryFactory geometry={phaseConfig.geometry} isActive={isActive} />
            <meshStandardMaterial
              color={statusConfig.color}
              emissive={phaseConfig.color}
              emissiveIntensity={statusConfig.intensity}
              metalness={0.7}
              roughness={0.3}
              wireframe={phase.status === 'pending'}
            />
          </mesh>

          {phase.status === 'active' && (
            <group ref={particlesRef}>
              {Array.from({ length: 8 }, (_, i) => {
                const angle = (i / 8) * Math.PI * 2;
                const radius = 2;
                return (
                  <mesh
                    key={i}
                    position={[
                      Math.cos(angle) * radius,
                      Math.sin(angle * 2) * 0.5,
                      Math.sin(angle) * radius,
                    ]}
                  >
                    <sphereGeometry args={[0.08]} />
                    <meshBasicMaterial color={phaseConfig.color} />
                  </mesh>
                );
              })}
            </group>
          )}

          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[1.8, 2, 32]} />
            <meshBasicMaterial
              color={phaseConfig.color}
              transparent
              opacity={0.3}
              side={THREE.DoubleSide}
            />
          </mesh>

          <Text
            position={[0, -2.5, 0]}
            fontSize={0.4}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
          >
            {phaseType}
          </Text>

          <Text
            position={[0, -3.2, 0]}
            fontSize={0.3}
            color={statusConfig.color}
            anchorX="center"
            anchorY="middle"
          >
            {phase.completion}%
          </Text>
        </group>
      </Float>
    );
  }
);
ProjectPhase3D.displayName = 'ProjectPhase3D';

// Connection Lines Between Phases
const PhaseConnection = memo(
  ({
    start,
    end,
    isActive,
  }: {
    start: [number, number, number];
    end: [number, number, number];
    isActive: boolean;
  }) => {
    const lineRef = useRef<THREE.Line>(null);

    useFrame(({ clock }) => {
      if (lineRef.current && isActive) {
        const opacity = Math.sin(clock.getElapsedTime() * 3) * 0.3 + 0.7;
        const material = lineRef.current.material as THREE.LineBasicMaterial;
        if (material) {
          material.opacity = opacity;
        }
      }
    });

    const points = useMemo(() => {
      const curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(...start),
        new THREE.Vector3((start[0] + end[0]) / 2, start[1] + 1, (start[2] + end[2]) / 2),
        new THREE.Vector3(...end),
      ]);
      return curve.getPoints(50);
    }, [start, end]);

    const geometry = useMemo(() => {
      return new THREE.BufferGeometry().setFromPoints(points);
    }, [points]);

    return (
      <line ref={lineRef}>
        <bufferGeometry attach="geometry" {...geometry} />
        <lineBasicMaterial
          attach="material"
          color={isActive ? '#64f4ac' : '#475569'}
          transparent
          opacity={isActive ? 0.8 : 0.3}
        />
      </line>
    );
  }
);
PhaseConnection.displayName = 'PhaseConnection';

// Project Details Panel
const ProjectDetailsPanel = memo(
  ({
    project,
    onClose,
    onAction,
  }: {
    project: Project;
    onClose: () => void;
    onAction: (action: string) => void;
  }) => {
    const [activeTab, setActiveTab] = useState('overview');

    const tabs = [
      { id: 'overview', label: 'Overview', icon: Eye },
      { id: 'phases', label: 'Phases', icon: Settings },
      { id: 'timeline', label: 'Timeline', icon: Calendar },
      { id: 'resources', label: 'Resources', icon: Database },
    ];

    return (
      <motion.div
        initial={{ opacity: 0, x: '100%' }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: '100%' }}
        className="fixed top-0 right-0 h-full w-96 bg-slate-900/95 backdrop-blur-xl border-l border-slate-700/50 shadow-2xl z-50 overflow-hidden flex flex-col"
      >
        <div className="p-6 border-b border-slate-700/50 flex-shrink-0">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-white truncate">{project.name}</h2>
            <button
              onClick={onClose}
              className="p-1 hover:bg-slate-800 rounded transition-colors flex-shrink-0"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>

          <div className="flex items-center space-x-3">
            <div
              className="px-3 py-1 rounded-full text-xs font-semibold whitespace-nowrap"
              style={{
                backgroundColor:
                  project.status === 'active'
                    ? 'rgba(34, 197, 94, 0.2)'
                    : project.status === 'completed'
                      ? 'rgba(59, 130, 246, 0.2)'
                      : project.status === 'paused'
                        ? 'rgba(234, 179, 8, 0.2)'
                        : 'rgba(71, 85, 105, 0.2)',
                color:
                  project.status === 'active'
                    ? '#4ade80'
                    : project.status === 'completed'
                      ? '#60a5fa'
                      : project.status === 'paused'
                        ? '#facc15'
                        : '#94a3b8',
              }}
            >
              {project.status.toUpperCase()}
            </div>
            <span className="text-slate-400 text-sm">{project.completion}% Complete</span>
          </div>
        </div>

        <div className="flex border-b border-slate-700/50 flex-shrink-0">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center space-x-2 p-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-cyan-400 border-b-2 border-cyan-400 bg-slate-800/50'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/30'
              }`}
            >
              <tab.icon className="w-4 h-4 flex-shrink-0" />
              <span className="hidden sm:block">{tab.label}</span>
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          <AnimatePresence mode="wait">
            {activeTab === 'overview' && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                <div>
                  <h3 className="text-lg font-semibold text-white mb-3">Project Overview</h3>
                  <p className="text-slate-400 leading-relaxed text-sm">
                    {project.description ||
                      'Advanced project management system with AI-powered optimization and real-time monitoring capabilities.'}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 rounded-xl p-4">
                    <div className="text-sm font-medium text-slate-300 mb-2">Progress</div>
                    <div className="text-2xl font-bold text-white">{project.completion}%</div>
                  </div>

                  <div className="bg-slate-800/50 rounded-xl p-4">
                    <div className="text-sm font-medium text-slate-300 mb-2">Duration</div>
                    <div className="text-2xl font-bold text-white">14d</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="font-semibold text-white text-sm">Active Agents</h4>
                  {project.agents && project.agents.length > 0 ? (
                    project.agents.map((agent) => (
                      <div
                        key={agent}
                        className="flex items-center space-x-3 p-3 bg-slate-800/30 rounded-lg"
                      >
                        <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                          <Zap className="w-4 h-4 text-white" />
                        </div>
                        <span className="text-slate-300 text-sm truncate">{agent}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-slate-500 text-sm">No agents assigned</div>
                  )}
                </div>
              </motion.div>
            )}

            {activeTab === 'phases' && (
              <motion.div
                key="phases"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                {Object.entries(project.phases || {}).map(([phaseType, phase]) => {
                  if (!phase) return null;

                  const phaseConfig = PHASE_TYPES[phaseType as PhaseTypeKey];
                  const statusConfig = PHASE_STATUS[phase.status];
                  if (!phaseConfig || !statusConfig) return null;

                  return (
                    <div key={phaseType} className="bg-slate-800/30 rounded-xl p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3 min-w-0">
                          <phaseConfig.icon
                            className="w-5 h-5 flex-shrink-0"
                            style={{ color: phaseConfig.color }}
                          />
                          <span className="font-medium text-white text-sm truncate">
                            {phaseType}
                          </span>
                        </div>
                        <div
                          className="px-2 py-1 rounded text-xs font-semibold whitespace-nowrap ml-2"
                          style={{
                            color: statusConfig.color,
                            backgroundColor: `${statusConfig.color}20`,
                          }}
                        >
                          {phase.status.toUpperCase()}
                        </div>
                      </div>

                      <p className="text-sm text-slate-400 mb-3">{phaseConfig.description}</p>

                      <div className="w-full bg-slate-700 rounded-full h-2">
                        <motion.div
                          className="h-2 rounded-full transition-all duration-500"
                          style={{
                            width: `${phase.completion}%`,
                            backgroundColor: phaseConfig.color,
                          }}
                          initial={{ width: 0 }}
                          animate={{ width: `${phase.completion}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-xs text-slate-500 mt-1">
                        <span>{phase.completion}% complete</span>
                        <span>{phase.estimatedTime || '2h'} remaining</span>
                      </div>
                    </div>
                  );
                })}
              </motion.div>
            )}

            {activeTab === 'timeline' && (
              <motion.div
                key="timeline"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                <div className="bg-slate-800/30 rounded-xl p-4">
                  <div className="text-sm font-semibold text-white mb-3">Project Timeline</div>
                  <div className="space-y-3">
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 rounded-full bg-cyan-400 mt-2 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white font-medium">Started</div>
                        <div className="text-xs text-slate-400">2 days ago</div>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 rounded-full bg-yellow-400 mt-2 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white font-medium">Current Phase</div>
                        <div className="text-xs text-slate-400">DEVELOPMENT</div>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-2 h-2 rounded-full bg-slate-500 mt-2 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white font-medium">Estimated End</div>
                        <div className="text-xs text-slate-400">In ~12 days</div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'resources' && (
              <motion.div
                key="resources"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-4"
              >
                <div className="bg-slate-800/30 rounded-xl p-4">
                  <div className="text-sm font-semibold text-white mb-3">Resource Allocation</div>
                  <div className="space-y-2 text-sm text-slate-400">
                    <div className="flex justify-between">
                      <span>CPU Usage:</span>
                      <span className="text-cyan-400 font-mono">45%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Memory Usage:</span>
                      <span className="text-cyan-400 font-mono">2.3 GB</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Agents Active:</span>
                      <span className="text-cyan-400 font-mono">{project.agents?.length || 0}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="p-6 border-t border-slate-700/50 flex-shrink-0">
          <div className="grid grid-cols-2 gap-3">
            <motion.button
              onClick={() => onAction('pause')}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex items-center justify-center space-x-2 px-4 py-2 bg-yellow-600/20 hover:bg-yellow-600/30 border border-yellow-600/50 text-yellow-400 rounded-xl transition-colors text-sm font-medium"
            >
              <Pause className="w-4 h-4" />
              <span className="hidden sm:inline">Pause</span>
            </motion.button>
            <motion.button
              onClick={() => onAction('restart')}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-600/50 text-blue-400 rounded-xl transition-colors text-sm font-medium"
            >
              <RotateCcw className="w-4 h-4" />
              <span className="hidden sm:inline">Restart</span>
            </motion.button>
          </div>
        </div>
      </motion.div>
    );
  }
);
ProjectDetailsPanel.displayName = 'ProjectDetailsPanel';

// Main Projects Page Component
const ProjectsPage = ({ projects: propsProjects, agents }: { projects?: Project[]; agents?: any[] }) => {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [activePhase, setActivePhase] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'3d' | 'list'>('3d');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');

  // Mock projects data with fallback
  const projects = useMemo<Project[]>(() => {
    if (propsProjects && propsProjects.length > 0) {
      return propsProjects;
    }

    return [
      {
        id: 1,
        name: 'E-Commerce Optimization',
        status: 'active',
        completion: 67,
        agents: ['OptimBot-02', 'SecGuard-03', 'DataFlow-05'],
        phases: {
          ANALYSIS: { status: 'complete', completion: 100 },
          DESIGN: { status: 'complete', completion: 100 },
          DEVELOPMENT: { status: 'active', completion: 85 },
          TESTING: { status: 'pending', completion: 0 },
          OPTIMIZATION: { status: 'pending', completion: 0 },
          DEPLOYMENT: { status: 'pending', completion: 0 },
        },
      },
      {
        id: 2,
        name: 'AI Security Framework',
        status: 'completed',
        completion: 100,
        agents: ['CryptGuard-08', 'CodeAnalyzer-01'],
        phases: {
          ANALYSIS: { status: 'complete', completion: 100 },
          DESIGN: { status: 'complete', completion: 100 },
          DEVELOPMENT: { status: 'complete', completion: 100 },
          TESTING: { status: 'complete', completion: 100 },
          OPTIMIZATION: { status: 'complete', completion: 100 },
          DEPLOYMENT: { status: 'complete', completion: 100 },
        },
      },
    ];
  }, [propsProjects]);

  const filteredProjects = useMemo(() => {
    return projects.filter((project) => {
      const matchesSearch = project.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesFilter = filterStatus === 'all' || project.status === filterStatus;
      return matchesSearch && matchesFilter;
    });
  }, [projects, searchQuery, filterStatus]);

  const handlePhaseClick = useCallback((project: Project, phaseType: string) => {
    setSelectedProject(project);
    setActivePhase(phaseType);
  }, []);

  const handleProjectAction = useCallback(
    (action: string) => {
      console.log(`Action: ${action} on project:`, selectedProject);
      // Add toast notification here if needed
    },
    [selectedProject]
  );

  const canvasHeight = useMemo(() => {
    if (typeof window === 'undefined') return '70vh';
    const height = window.innerHeight - 300;
    return `${Math.max(height, 400)}px`;
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 opacity-50" />

      <div className="relative z-10 p-4 sm:p-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">
              Project Visualization Center
            </h1>
            <p className="text-slate-400 text-sm sm:text-base">
              Monitor and manage project workflows in real-time
            </p>
          </div>
          <div className="flex items-center space-x-4 flex-shrink-0">
            <div className="flex bg-slate-800 rounded-xl p-1">
              <button
                onClick={() => setViewMode('3d')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  viewMode === '3d'
                    ? 'bg-cyan-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                3D
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  viewMode === 'list'
                    ? 'bg-cyan-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                List
              </button>
            </div>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center space-x-2 px-4 sm:px-6 py-2 sm:py-3 bg-gradient-to-r from-green-400 to-cyan-400 text-black font-semibold rounded-xl shadow-lg whitespace-nowrap text-sm sm:text-base"
            >
              <Plus className="w-5 h-5" />
              <span className="hidden sm:inline">New Project</span>
              <span className="sm:hidden">New</span>
            </motion.button>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-4 sm:space-y-0 sm:space-x-4 mb-8">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400 flex-shrink-0" />
            <input
              type="text"
              placeholder="Search projects..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-white placeholder-slate-400 focus:outline-none focus:border-cyan-400 transition-colors"
            />
          </div>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-400 transition-colors sm:min-w-max"
          