export interface AnimationConfig {
  amplitude?: number;
  speed?: number;
  rotationSpeed?: number;
  delay?: number;
}

export type AnimationType =
  | 'fadeInUp'
  | 'fadeInLeft'
  | 'fadeInRight'
  | 'scaleIn'
  | 'progressiveReveal';

export interface PerformanceMetrics {
  fps: number;
  memory: number;
  duration: number;
}

export interface ParticleConfig {
  particleCount: number;
  particleSize: { min: number; max: number };
  particleSpeed: { min: number; max: number };
  particleColor: string[];
  connectionDistance: number;
  connectionOpacity: number;
  enableConnections: boolean;
  enableMouse: boolean;
  mouseRadius: number;
  particleLife: { min: number; max: number };
  gravity: number;
  wind: number;
  turbulence: number;
}
