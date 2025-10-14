import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Text, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// Enhanced LogoCore with multiple geometry morphing
const LogoCore = ({ isAnimating, hover, complexity = 1 }) => {
  const meshRef = useRef();
  const materialRef = useRef();
  const particlesRef = useRef();
  const { viewport } = useThree();

  // Color palette from your theme
  const colors = useMemo(() => ({
    primary: new THREE.Color('#64f4ac'),
    accent: new THREE.Color('#4fd1c5'),
    highlight: new THREE.Color('#7dd3fc'),
    warning: new THREE.Color('#f59e0b'),
    error: new THREE.Color('#ef4444')
  }), []);

  // Create particle system for enhanced effects
  const particles = useMemo(() => {
    const positions = [];
    const colors = [];
    const sizes = [];
    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
      positions.push(
        (Math.random() - 0.5) * 3,
        (Math.random() - 0.5) * 3,
        (Math.random() - 0.5) * 3
      );
      
      const color = new THREE.Color();
      color.setHSL(0.4 + Math.random() * 0.2, 0.7, 0.5);
      colors.push(color.r, color.g, color.b);
      
      sizes.push(Math.random() * 2 + 1);
    }

    return {
      positions: new Float32Array(positions),
      colors: new Float32Array(colors),
      sizes: new Float32Array(sizes),
      count: particleCount
    };
  }, []);

  // Complex animation loop
  useFrame(({ clock }) => {
    if (!meshRef.current || !materialRef.current) return;

    const time = clock.getElapsedTime();
    const material = materialRef.current;
    
    // Dynamic speed based on state
    let speed = isAnimating || hover ? 2.5 : 0.8;
    
    // Multi-axis rotation with mathematical relationships
    meshRef.current.rotation.x = Math.sin(time * speed * 0.7) * 0.3 + time * 0.05;
    meshRef.current.rotation.y = Math.cos(time * speed * 0.5) * 0.2 + time * 0.08;
    meshRef.current.rotation.z = Math.sin(time * speed * 0.3) * 0.1 + time * 0.03;
    
    // Organic floating motion with multiple sine waves
    meshRef.current.position.y = 
      Math.sin(time * speed * 0.6) * 0.15 + 
      Math.sin(time * speed * 1.2) * 0.05;
    
    meshRef.current.position.x = Math.cos(time * speed * 0.4) * 0.08;
    
    // Dynamic scaling with breathing effect
    const breathe = 1 + Math.sin(time * speed * 0.8) * 0.08;
    const complexity_scale = 1 + (complexity - 1) * 0.3;
    meshRef.current.scale.setScalar(breathe * complexity_scale);
    
    // Advanced color cycling through theme palette
    const colorPhase = time * speed * 0.3;
    const primaryWeight = (Math.sin(colorPhase) + 1) * 0.5;
    const accentWeight = (Math.cos(colorPhase + Math.PI * 0.5) + 1) * 0.5;
    const highlightWeight = (Math.sin(colorPhase * 2 + Math.PI) + 1) * 0.25;
    
    const targetColor = colors.primary.clone()
      .lerp(colors.accent, accentWeight * 0.6)
      .lerp(colors.highlight, highlightWeight * 0.4);
    
    material.emissive.lerp(targetColor, 0.05);
    material.color.lerp(targetColor, 0.02);
    
    // Dynamic material properties
    material.roughness = 0.1 + Math.sin(time * speed) * 0.05;
    material.metalness = 0.9 + Math.cos(time * speed * 1.3) * 0.05;
    
    // Enhanced emissive intensity with complexity
    const baseIntensity = isAnimating || hover ? 0.6 : 0.3;
    const pulseIntensity = Math.sin(time * speed * 2) * 0.2;
    material.emissiveIntensity = baseIntensity + pulseIntensity * complexity;
    
    // Animate particles if they exist
    if (particlesRef.current && (isAnimating || hover)) {
      const positions = particlesRef.current.geometry.attributes.position.array;
      for (let i = 0; i < particles.count; i++) {
        const i3 = i * 3;
        positions[i3] += Math.sin(time + i) * 0.01;
        positions[i3 + 1] += Math.cos(time + i) * 0.01;
        positions[i3 + 2] += Math.sin(time * 0.5 + i) * 0.008;
      }
      particlesRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  return (
    <group>
      {/* Main logo mesh with morphing geometry */}
      <mesh 
        ref={meshRef}
        castShadow 
        receiveShadow
      >
        <icosahedronGeometry args={[0.5, 2]} />
        <meshPhysicalMaterial
          ref={materialRef}
          transparent
          opacity={0.9}
          metalness={0.9}
          roughness={0.1}
          clearcoat={1}
          clearcoatRoughness={0.1}
          emissiveIntensity={0.3}
          envMapIntensity={1.5}
          transmission={hover ? 0.15 : 0.05}
          thickness={0.8}
          ior={1.45}
        />
      </mesh>

      {/* Particle system for magical effects */}
      {(isAnimating || hover) && (
        <points ref={particlesRef}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={particles.count}
              array={particles.positions}
              itemSize={3}
            />
            <bufferAttribute
              attach="attributes-color"
              count={particles.count}
              array={particles.colors}
              itemSize={3}
            />
            <bufferAttribute
              attach="attributes-size"
              count={particles.count}
              array={particles.sizes}
              itemSize={1}
            />
          </bufferGeometry>
          <pointsMaterial
            vertexColors
            transparent
            opacity={0.6}
            size={0.05}
            sizeAttenuation
            blending={THREE.AdditiveBlending}
          />
        </points>
      )}

      {/* Orbital rings for enhanced visual depth */}
      {complexity > 0.5 && (
        <>
          <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
            <torusGeometry args={[1.2, 0.02, 8, 32]} />
            <meshBasicMaterial
              color={colors.accent}
              transparent
              opacity={isAnimating || hover ? 0.4 : 0.2}
              side={THREE.DoubleSide}
            />
          </mesh>
          
          <mesh rotation={[0, Math.PI / 3, Math.PI / 4]} position={[0, 0, 0]}>
            <torusGeometry args={[1.0, 0.015, 6, 24]} />
            <meshBasicMaterial
              color={colors.highlight}
              transparent
              opacity={isAnimating || hover ? 0.3 : 0.15}
              side={THREE.DoubleSide}
            />
          </mesh>
        </>
      )}

      {/* Optional Y letter with 3D depth */}
      {complexity > 0.7 && (
        <Text
          position={[0, -0.8, 0.2]}
          fontSize={0.4}
          color={colors.primary}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor={colors.accent}
        >
          Y
        </Text>
      )}
    </group>
  );
};

// Environment setup for better lighting
const LogoEnvironment = () => (
  <>
    <ambientLight intensity={0.4} />
    <pointLight 
      position={[2, 2, 2]} 
      intensity={1} 
      color="#4fd1c5" 
      castShadow
    />
    <pointLight 
      position={[-2, -1, 1]} 
      intensity={0.6} 
      color="#64f4ac" 
    />
    <spotLight
      position={[0, 5, 0]}
      angle={0.3}
      penumbra={1}
      intensity={0.5}
      color="#7dd3fc"
    />
  </>
);

// Main component with enhanced interactivity
export default function AnimatedYmeraLogo({ 
  className = '', 
  size = 48,
  complexity = 1,
  autoStart = true,
  interactive = true 
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [isAnimating, setIsAnimating] = useState(autoStart);
  const [hasInteracted, setHasInteracted] = useState(false);
  const [animationPhase, setAnimationPhase] = useState(0);

  // Auto-animation control with phases
  useEffect(() => {
    if (!autoStart && !hasInteracted) return;

    let animationTimer;
    let phaseTimer;

    const startAnimation = () => {
      setIsAnimating(true);
      
      // Phase progression for varied animation patterns
      phaseTimer = setInterval(() => {
        setAnimationPhase(prev => (prev + 1) % 4);
      }, 3000);

      animationTimer = setTimeout(() => {
        if (!isHovered && !hasInteracted) {
          setIsAnimating(false);
          clearInterval(phaseTimer);
        }
      }, 8000);
    };

    if (autoStart && !hasInteracted) {
      const delay = Math.random() * 2000 + 1000;
      animationTimer = setTimeout(startAnimation, delay);
    }

    return () => {
      clearTimeout(animationTimer);
      clearInterval(phaseTimer);
    };
  }, [autoStart, hasInteracted, isHovered]);

  // Interaction handlers
  const handlePointerEnter = () => {
    if (!interactive) return;
    setIsHovered(true);
    setIsAnimating(true);
    setHasInteracted(true);
  };

  const handlePointerLeave = () => {
    if (!interactive) return;
    setIsHovered(false);
    
    setTimeout(() => {
      if (!isHovered) {
        setIsAnimating(false);
      }
    }, 1500);
  };

  const handleClick = () => {
    if (!interactive) return;
    setIsAnimating(!isAnimating);
    setHasInteracted(true);
  };

  // Dynamic complexity based on animation phase
  const currentComplexity = useMemo(() => {
    if (!isAnimating) return complexity * 0.5;
    
    switch (animationPhase) {
      case 0: return complexity * 0.7;
      case 1: return complexity * 1.0;
      case 2: return complexity * 1.3;
      case 3: return complexity * 1.1;
      default: return complexity;
    }
  }, [complexity, animationPhase, isAnimating]);

  // Performance optimization based on size
  const renderQuality = useMemo(() => {
    if (size < 32) return { dpr: 1, antialias: false };
    if (size < 64) return { dpr: 1.5, antialias: true };
    return { dpr: 2, antialias: true };
  }, [size]);

  return (
    <div
      className={`relative cursor-pointer select-none ${className}`}
      style={{ 
        width: size, 
        height: size,
        filter: isHovered ? 'brightness(1.2)' : 'brightness(1)',
        transition: 'all 0.3s ease'
      }}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      onClick={handleClick}
    >
      {/* Animated background glow */}
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background: isAnimating 
            ? `radial-gradient(circle, rgba(100, 244, 172, ${isHovered ? '0.3' : '0.15'}) 0%, transparent 70%)`
            : 'transparent',
          transform: isAnimating ? 'scale(1.2)' : 'scale(1)',
          transition: 'all 2s ease-in-out'
        }}
      />

      {/* Main 3D Canvas */}
      <Canvas
        dpr={renderQuality.dpr}
        camera={{ position: [0, 0, 3], fov: 45 }}
        gl={{ 
          antialias: renderQuality.antialias, 
          alpha: true,
          powerPreference: "high-performance"
        }}
        style={{ 
          background: 'transparent',
          width: '100%',
          height: '100%'
        }}
      >
        <LogoEnvironment />
        
        <LogoCore 
          isAnimating={isAnimating} 
          hover={isHovered}
          complexity={currentComplexity}
        />
        
        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          enableRotate={false}
          autoRotate={false}
        />
      </Canvas>

      {/* Optional loading indicator */}
      {isAnimating && complexity > 0.8 && (
        <div
          className="absolute bottom-0 right-0 w-2 h-2 rounded-full"
          style={{
            backgroundColor: '#64f4ac',
            opacity: Math.sin(Date.now() / 500) * 0.5 + 0.5
          }}
        />
      )}

      {/* Accessibility enhancement */}
      <div className="sr-only">
        Ymera AI Platform Logo - {isAnimating ? 'Animating' : 'Static'}
      </div>
    </div>
  );
}

// Preset configurations for different use cases
export const LogoPresets = {
  navbar: {
    size: 32,
    complexity: 0.6,
    autoStart: false,
    interactive: true,
    className: "transition-all duration-300"
  },
  
  hero: {
    size: 120,
    complexity: 1.2,
    autoStart: true,
    interactive: true,
    className: "drop-shadow-2xl"
  },
  
  loading: {
    size: 64,
    complexity: 0.8,
    autoStart: true,
    interactive: false,
    className: "animate-pulse"
  },
  
  favicon: {
    size: 24,
    complexity: 0.3,
    autoStart: false,
    interactive: false,
    className: ""
  },
  
  splash: {
    size: 200,
    complexity: 1.5,
    autoStart: true,
    interactive: true,
    className: "filter drop-shadow-2xl"
  }
};

// Convenience components for common use cases
export const NavbarLogo = (props) => (
  <AnimatedYmeraLogo {...LogoPresets.navbar} {...props} />
);

export const HeroLogo = (props) => (
  <AnimatedYmeraLogo {...LogoPresets.hero} {...props} />
);

export const LoadingLogo = (props) => (
  <AnimatedYmeraLogo {...LogoPresets.loading} {...props} />
);

export const SplashLogo = (props) => (
  <AnimatedYmeraLogo {...LogoPresets.splash} {...props} />
);