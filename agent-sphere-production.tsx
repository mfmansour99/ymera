import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh } from 'three';

interface Agent {
  id: string;
  name: string;
  status: 'idle' | 'thinking' | 'working' | 'completed' | 'error';
  type: 'code_analyzer' | 'ui_designer' | 'backend_developer' | 'testing_agent' | 'optimization_agent';
}

interface AgentSphereProps {
  agent: Agent;
  position: [number, number, number];
  onClick: () => void;
}

export const AgentSphere: React.FC<AgentSphereProps> = ({ agent, position, onClick }) => {
  const meshRef = useRef<Mesh>(null);

  // Rotate the sphere with error protection
  useFrame(() => {
    if (meshRef.current) {
      try {
        meshRef.current.rotation.y += 0.005;
      } catch (error) {
        console.error('[AgentSphere] Animation error:', error);
      }
    }
  });

  // Get color based on agent status
  const getStatusColor = (): string => {
    const colorMap: Record<Agent['status'], string> = {
      idle: '#4a5568',
      thinking: '#ed8936',
      working: '#3182ce',
      completed: '#38a169',
      error: '#e53e3e'
    };
    return colorMap[agent.status] || '#666666';
  };

  // Get size based on agent type
  const getSize = (): number => {
    const sizeMap: Record<Agent['type'], number> = {
      code_analyzer: 1.2,
      ui_designer: 1.1,
      backend_developer: 1.3,
      testing_agent: 1.0,
      optimization_agent: 1.2
    };
    return sizeMap[agent.type] || 1.0;
  };

  // Handle click with error protection
  const handleClick = (e: any) => {
    try {
      e.stopPropagation();
      onClick();
    } catch (error) {
      console.error('[AgentSphere] Click handler error:', error);
    }
  };

  return (
    <mesh
      ref={meshRef}
      position={position}
      onClick={handleClick}
      onPointerOver={(e) => {
        e.stopPropagation();
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        document.body.style.cursor = 'default';
      }}
    >
      <sphereGeometry args={[getSize(), 32, 32]} />
      <meshStandardMaterial
        color={getStatusColor()}
        metalness={0.7}
        roughness={0.3}
        emissive={getStatusColor()}
        emissiveIntensity={0.2}
      />
    </mesh>
  );
};