import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Text } from '@react-three/drei';
import { Project } from '../types';
import { PhaseSymbol } from './PhaseSymbol';

interface ProjectBuildingSceneProps {
  project: Project;
  onPhaseSelect: (phase: string) => void;
  onBack: () => void;
}

interface PhasePosition {
  name: string;
  position: [number, number, number];
  rotation: [number, number, number];
  color: string;
  size: number;
  symbol: string;
}

// WebGL Error Fallback Component
const WebGLErrorFallback: React.FC<{ error: string; onBack: () => void }> = ({ error, onBack }) => (
  <div className="glass-card h-full flex items-center justify-center">
    <div className="text-center max-w-md">
      <i className="fas fa-exclamation-triangle text-6xl text-accent-danger mb-4" />
      <h3 className="text-xl font-bold mb-2">WebGL Not Supported</h3>
      <p className="text-secondary mb-6">
        {error || 'Your browser or device does not support WebGL, which is required for 3D visualization.'}
      </p>
      <div className="space-y-2">
        <p className="text-sm text-secondary">Try:</p>
        <ul className="text-sm text-secondary text-left list-disc list-inside">
          <li>Updating your browser to the latest version</li>
          <li>Enabling hardware acceleration in browser settings</li>
          <li>Using the list view instead of 3D view</li>
        </ul>
      </div>
      <button
        onClick={onBack}
        className="glass-button bg-accent-primary bg-opacity-20 mt-6"
      >
        <i className="fas fa-arrow-left mr-2" />
        Back to List View
      </button>
    </div>
  </div>
);

// Spinning Core Component
const ProjectCore: React.FC<{ name: string }> = ({ name }) => {
  const meshRef = useRef<any>(null);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
      meshRef.current.rotation.x += 0.002;
    }
  });

  return (
    <mesh ref={meshRef} position={[0, 0.5, 0]}>
      <sphereGeometry args={[0.8, 32, 32]} />
      <meshStandardMaterial
        color="#00f5ff"
        metalness={0.8}
        roughness={0.2}
        emissive="#00f5ff"
        emissiveIntensity={0.3}
      />
      <Text
        position={[0, 0, 0.9]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
        maxWidth={2}
      >
        {name.substring(0, 3).toUpperCase()}
      </Text>
    </mesh>
  );
};

// Connection Lines Component
const ConnectionLines: React.FC<{ 
  phasePositions: PhasePosition[];
  selectedPhase: string | null;
}> = ({ phasePositions, selectedPhase }) => {
  return (
    <>
      {phasePositions.map((phase) => {
        const points = [
          [0, 0.5, 0],
          phase.position
        ];

        return (
          <line key={`line-${phase.name}`}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                array={new Float32Array(points.flat())}
                count={2}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial
              color={selectedPhase === phase.name ? phase.color : '#4a5568'}
              linewidth={2}
              transparent
              opacity={selectedPhase === phase.name ? 0.8 : 0.4}
            />
          </line>
        );
      })}
    </>
  );
};

export const ProjectBuildingScene: React.FC<ProjectBuildingSceneProps> = ({
  project,
  onPhaseSelect,
  onBack
}) => {
  const [selectedPhase, setSelectedPhase] = useState<string | null>(null);
  const [webGLError, setWebGLError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  // Check WebGL support on mount
  useEffect(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      
      if (!gl) {
        setWebGLError('WebGL is not supported by your browser or device.');
      }
    } catch (error) {
      setWebGLError('Failed to initialize WebGL context.');
    }

    // Cleanup
    return () => {
      // Clean up any WebGL resources if needed
    };
  }, []);

  // Define phase positions in 3D space
  const phasePositions: PhasePosition[] = [
    {
      name: 'planning',
      position: [0, 2, 0],
      rotation: [0, 0, 0],
      color: '#3182ce',
      size: 1.2,
      symbol: 'cube'
    },
    {
      name: 'design',
      position: [3, 0.5, 2],
      rotation: [0, Math.PI / 4, 0],
      color: '#805ad5',
      size: 1.0,
      symbol: 'dodecahedron'
    },
    {
      name: 'development',
      position: [2, 0.8, -3],
      rotation: [0, Math.PI / 3, 0],
      color: '#38a169',
      size: 1.4,
      symbol: 'octahedron'
    },
    {
      name: 'testing',
      position: [-3, 0.6, 1],
      rotation: [0, -Math.PI / 4, 0],
      color: '#e53e3e',
      size: 1.1,
      symbol: 'tetrahedron'
    },
    {
      name: 'deployment',
      position: [-2, 0.9, -2],
      rotation: [0, -Math.PI / 3, 0],
      color: '#dd6b20',
      size: 1.3,
      symbol: 'icosahedron'
    }
  ];

  const handlePhaseClick = (phase: string) => {
    setSelectedPhase(phase);
    onPhaseSelect(phase);
  };

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(amount);
  };

  // Show error fallback if WebGL is not supported
  if (webGLError) {
    return <WebGLErrorFallback error={webGLError} onBack={onBack} />;
  }

  return (
    <div ref={canvasRef} className="relative h-full">
      {/* Back Button */}
      <button
        onClick={onBack}
        className="absolute top-4 left-4 z-10 glass-button bg-primary-bg bg-opacity-80 hover:bg-opacity-100"
        aria-label="Back to projects list"
      >
        <i className="fas fa-arrow-left mr-2" />
        Back to Projects
      </button>

      {/* Project Info Card */}
      <div className="absolute top-4 right-4 z-10 glass-card p-4 max-w-xs">
        <h3 className="font-bold mb-3 text-lg">{project.name}</h3>
        <div className="space-y-2 text-sm">
          <div className="flex items-center">
            <i className="fas fa-calendar mr-2 text-secondary w-4" />
            <span>{project.duration}</span>
          </div>
          <div className="flex items-center">
            <i className="fas fa-dollar-sign mr-2 text-secondary w-4" />
            <span>{formatCurrency(project.budget)}</span>
          </div>
          <div className="flex items-center">
            <i className="fas fa-chart-line mr-2 text-secondary w-4" />
            <span>{project.progress}% Complete</span>
          </div>
        </div>
        
        {/* Progress Bar */}
        <div className="mt-4">
          <div className="w-full bg-glass rounded-full h-2">
            <div
              className="bg-gradient-to-r from-accent-primary to-accent-secondary h-2 rounded-full transition-all duration-500"
              style={{ width: `${project.progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 glass-card px-4 py-2 text-sm text-secondary">
        <i className="fas fa-mouse mr-2" />
        Click and drag to rotate • Scroll to zoom • Click phases to view details
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 8, 12], fov: 50 }}
        onCreated={({ gl }) => {
          // Enable antialiasing for better quality
          gl.antialias = true;
          // Set pixel ratio for retina displays (capped at 2 for performance)
          gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        }}
        onError={(error) => {
          console.error('Canvas error:', error);
          setWebGLError('Failed to initialize 3D scene.');
        }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} />
        
        {/* Environment */}
        <Environment preset="city" />

        {/* Camera Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          zoomSpeed={0.6}
          panSpeed={0.5}
          rotateSpeed={0.4}
          minDistance={5}
          maxDistance={20}
          maxPolarAngle={Math.PI / 1.8}
        />

        {/* Base Platform */}
        <mesh position={[0, -0.5, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <circleGeometry args={[6, 64]} />
          <meshStandardMaterial
            color="#1a202c"
            metalness={0.3}
            roughness={0.7}
            opacity={0.8}
            transparent
          />
        </mesh>

        {/* Grid Helper */}
        <gridHelper args={[12, 12, '#4a5568', '#2d3748']} position={[0, -0.45, 0]} />

        {/* Project Core */}
        <ProjectCore name={project.name} />

        {/* Connection Lines */}
        <ConnectionLines 
          phasePositions={phasePositions}
          selectedPhase={selectedPhase}
        />

        {/* Phase Symbols */}
        {phasePositions.map((phase) => (
          <PhaseSymbol
            key={phase.name}
            name={phase.name}
            position={phase.position}
            rotation={phase.rotation}
            color={phase.color}
            size={phase.size}
            symbol={phase.symbol}
            isSelected={selectedPhase === phase.name}
            onClick={() => handlePhaseClick(phase.name)}
          />
        ))}
      </Canvas>
    </div>
  );
};