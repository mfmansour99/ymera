import React, { useRef, useMemo, memo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import { Mesh } from 'three';

interface PhaseSymbolProps {
  name: string;
  position: [number, number, number];
  rotation: [number, number, number];
  color: string;
  size: number;
  symbol: string;
  isSelected: boolean;
  onClick: () => void;
}

// Memoized geometry selector
const useGeometry = (symbol: string, size: number) => {
  return useMemo(() => {
    switch (symbol) {
      case 'cube':
        return <boxGeometry args={[size, size, size]} />;
      case 'dodecahedron':
        return <dodecahedronGeometry args={[size * 0.7, 0]} />;
      case 'octahedron':
        return <octahedronGeometry args={[size * 0.8, 0]} />;
      case 'tetrahedron':
        return <tetrahedronGeometry args={[size * 0.9, 0]} />;
      case 'icosahedron':
        return <icosahedronGeometry args={[size * 0.75, 0]} />;
      case 'sphere':
      default:
        return <sphereGeometry args={[size * 0.8, 32, 32]} />;
    }
  }, [symbol, size]);
};

// Memoized phase name formatter
const usePhaseName = (name: string) => {
  return useMemo(() => {
    const names: Record<string, string> = {
      planning: 'Planning',
      design: 'Design',
      development: 'Development',
      testing: 'Testing',
      deployment: 'Deployment',
      requirements: 'Requirements',
      architecture: 'Architecture',
      implementation: 'Implementation',
      qa: 'QA',
      release: 'Release'
    };
    return names[name] || name.charAt(0).toUpperCase() + name.slice(1).replace('_', ' ');
  }, [name]);
};

export const PhaseSymbol: React.FC<PhaseSymbolProps> = memo(
  ({
    name,
    position,
    rotation,
    color,
    size,
    symbol,
    isSelected,
    onClick
  }) => {
    const meshRef = useRef<Mesh>(null);
    const ringRef = useRef<Mesh>(null);
    const geometry = useGeometry(symbol, size);
    const displayName = usePhaseName(name);

    // Animate only when selected for performance
    useFrame(() => {
      if (meshRef.current && isSelected) {
        meshRef.current.rotation.y += 0.01;
        meshRef.current.rotation.x += 0.005;
      }

      // Pulse the selection ring
      if (ringRef.current && isSelected) {
        const scale = 1 + Math.sin(Date.now() * 0.003) * 0.1;
        ringRef.current.scale.set(scale, scale, scale);
      }
    });

    return (
      <group position={position} rotation={rotation}>
        {/* Main Phase Mesh */}
        <mesh
          ref={meshRef}
          onClick={(e) => {
            e.stopPropagation();
            onClick();
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            document.body.style.cursor = 'pointer';
          }}
          onPointerOut={() => {
            document.body.style.cursor = 'default';
          }}
        >
          {geometry}
          <meshStandardMaterial
            color={color}
            metalness={0.7}
            roughness={0.3}
            emissive={color}
            emissiveIntensity={isSelected ? 0.5 : 0.1}
          />
        </mesh>

        {/* Phase Label */}
        <Text
          position={[0, -size - 0.5, 0]}
          fontSize={0.3}
          color={isSelected ? color : '#ffffff'}
          anchorX="center"
          anchorY="middle"
          maxWidth={2}
          outlineWidth={0.02}
          outlineColor="#000000"
        >
          {displayName}
        </Text>

        {/* Selection Indicators */}
        {isSelected && (
          <>
            {/* Selection Ring */}
            <mesh 
              ref={ringRef}
              position={[0, 0, 0]} 
              rotation={[Math.PI / 2, 0, 0]}
            >
              <ringGeometry args={[size * 1.2, size * 1.3, 64]} />
              <meshBasicMaterial
                color={color}
                side={2}
                transparent
                opacity={0.4}
              />
            </mesh>

            {/* Pulsing Sphere Effect */}
            <mesh position={[0, 0, 0]}>
              <sphereGeometry args={[size * 1.5, 32, 32]} />
              <meshBasicMaterial
                color={color}
                transparent
                opacity={0.1}
                depthWrite={false}
              />
            </mesh>

            {/* Particle Ring Effect */}
            {Array.from({ length: 8 }).map((_, i) => {
              const angle = (i / 8) * Math.PI * 2;
              const radius = size * 1.4;
              const x = Math.cos(angle) * radius;
              const z = Math.sin(angle) * radius;

              return (
                <mesh key={i} position={[x, 0, z]}>
                  <sphereGeometry args={[0.05, 16, 16]} />
                  <meshBasicMaterial
                    color={color}
                    transparent
                    opacity={0.6}
                  />
                </mesh>
              );
            })}
          </>
        )}

        {/* Hover Glow (subtle) */}
        {!isSelected && (
          <mesh position={[0, 0, 0]}>
            <sphereGeometry args={[size * 1.2, 32, 32]} />
            <meshBasicMaterial
              color={color}
              transparent
              opacity={0.05}
              depthWrite={false}
            />
          </mesh>
        )}
      </group>
    );
  },
  // Custom comparison function for better performance
  (prevProps, nextProps) => {
    return (
      prevProps.name === nextProps.name &&
      prevProps.isSelected === nextProps.isSelected &&
      prevProps.color === nextProps.color &&
      prevProps.size === nextProps.size &&
      prevProps.symbol === nextProps.symbol &&
      prevProps.position[0] === nextProps.position[0] &&
      prevProps.position[1] === nextProps.position[1] &&
      prevProps.position[2] === nextProps.position[2]
    );
  }
);

PhaseSymbol.displayName = 'PhaseSymbol';