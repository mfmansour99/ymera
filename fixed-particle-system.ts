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
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { alpha: true })!;
    
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
    // Mouse movement
    this.container.addEventListener('mousemove', this.handleMouseMove);
    this.container.addEventListener('mouseleave', this.handleMouseLeave);
    
    // Window resize with ResizeObserver for better performance
    this.resizeObserver = new ResizeObserver(() => {
      this.resize();
    });
    this.resizeObserver.observe(this.container);
    
    // Fallback for older browsers
    if (!this.resizeObserver) {
      window.addEventListener('resize', this.handleResize);
    }
  }

  private handleMouseMove = (e: MouseEvent): void => {
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
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    
    this.ctx.scale(dpr, dpr);
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

    return {
      x: x !== undefined ? x : Math.random() * this.canvas.width,
      y: y !== undefined ? y : Math.random() * this.canvas.height,
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
    // Apply forces
    particle.vx += this.config.wind;
    particle.vy += this.config.gravity;
    
    // Add turbulence
    particle.vx += (Math.random() - 0.5) * this.config.turbulence;
    particle.vy += (Math.random() - 0.5) * this.config.turbulence;

    // Mouse interaction
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

    // Apply velocity damping
    particle.vx *= 0.99;
    particle.vy *= 0.99;

    // Update position
    particle.x += particle.vx;
    particle.y += particle.vy;
    
    // Update visual properties
    particle.angle += particle.rotationSpeed;
    particle.pulse += particle.pulseSpeed;
    particle.currentSize = particle.size + Math.sin(particle.pulse) * particle.size * 0.2;

    // Boundary collision with proper canvas dimensions
    const canvasWidth = this.canvas.width / (window.devicePixelRatio || 1);
    const canvasHeight = this.canvas.height / (window.devicePixelRatio || 1);
    
    if (particle.x < 0 || particle.x > canvasWidth) {
      particle.vx *= -0.8;
      particle.x = Math.max(0, Math.min(canvasWidth, particle.x));
    }
    if (particle.y < 0 || particle.y > canvasHeight) {
      particle.vy *= -0.8;
      particle.y = Math.max(0, Math.min(canvasHeight, particle.y));
    }

    // Life cycle
    particle.life--;
    particle.opacity = Math.max(0, (particle.life / particle.maxLife) * 0.8);
    
    // Respawn particle
    if (particle.life <= 0) {
      Object.assign(particle, this.createParticle());
    }
  }

  private drawParticle(particle: Particle): void {
    this.ctx.save();
    this.ctx.globalAlpha = particle.opacity;

    // Outer glow
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

    // Inner core
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

    const canvasWidth = this.canvas.width / (window.devicePixelRatio || 1);
    const canvasHeight = this.canvas.height / (window.devicePixelRatio || 1);
    
    this.ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Update and draw particles
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
    if (this.animationId) {
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
    
    // Remove event listeners
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
    
    // Remove canvas
    if (this.canvas && this.canvas.parentNode) {
      this.canvas.parentNode.removeChild(this.canvas);
    }
    
    this.particles = [];
  }

  public addBurst(x: number, y: number, count: number = 10): void {
    for (let i = 0; i < count; i++) {
      const particle = this.createParticle(x, y);
      const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5)