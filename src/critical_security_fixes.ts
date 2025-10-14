// ============================================================================
// SECURITY FIX #1: CSRF Token Service
// ============================================================================

import crypto from 'crypto';

interface CSRFToken {
  token: string;
  timestamp: number;
  signature: string;
}

class CSRFService {
  private tokens: Map<string, CSRFToken> = new Map();
  private readonly TOKEN_EXPIRY = 1000 * 60 * 60; // 1 hour
  private readonly SECRET_KEY = process.env.CSRF_SECRET || 'dev-secret-key';

  generateToken(sessionId: string): string {
    const token = crypto.randomBytes(32).toString('hex');
    const timestamp = Date.now();
    const signature = this.createSignature(token, timestamp, sessionId);

    const csrfToken: CSRFToken = { token, timestamp, signature };
    this.tokens.set(sessionId, csrfToken);

    return token;
  }

  verifyToken(token: string, sessionId: string): boolean {
    const storedToken = this.tokens.get(sessionId);
    
    if (!storedToken) {
      return false;
    }

    // Check expiry
    if (Date.now() - storedToken.timestamp > this.TOKEN_EXPIRY) {
      this.tokens.delete(sessionId);
      return false;
    }

    // Verify signature
    const expectedSignature = this.createSignature(
      storedToken.token,
      storedToken.timestamp,
      sessionId
    );

    const isValid = token === storedToken.token && 
                    expectedSignature === storedToken.signature;

    if (isValid) {
      // Invalidate token after use (regenerate on next request)
      this.tokens.delete(sessionId);
    }

    return isValid;
  }

  private createSignature(token: string, timestamp: number, sessionId: string): string {
    const data = `${token}:${timestamp}:${sessionId}`;
    return crypto
      .createHmac('sha256', this.SECRET_KEY)
      .update(data)
      .digest('hex');
  }

  clearExpiredTokens(): void {
    const now = Date.now();
    for (const [sessionId, tokenData] of this.tokens.entries()) {
      if (now - tokenData.timestamp > this.TOKEN_EXPIRY) {
        this.tokens.delete(sessionId);
      }
    }
  }
}

export const csrfService = new CSRFService();

// ============================================================================
// SECURITY FIX #2: Input Sanitization & XSS Prevention
// ============================================================================

interface SanitizeOptions {
  allowHtml?: boolean;
  maxLength?: number;
  allowSpecialChars?: boolean;
}

const DANGEROUS_PATTERNS = [
  /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
  /on\w+\s*=\s*["']?(?:javascript:)?/gi,
  /javascript:/gi,
  /vbscript:/gi,
  /onerror\s*=/gi,
  /onload\s*=/gi,
  /onclick\s*=/gi,
  /onmouseover\s*=/gi,
];

export const sanitizeInput = (
  input: string,
  options: SanitizeOptions = {}
): string => {
  const {
    allowHtml = false,
    maxLength = 500,
    allowSpecialChars = false,
  } = options;

  if (!input || typeof input !== 'string') {
    return '';
  }

  // Trim whitespace
  let sanitized = input.trim();

  // Enforce max length
  sanitized = sanitized.slice(0, maxLength);

  // Remove dangerous patterns
  DANGEROUS_PATTERNS.forEach(pattern => {
    sanitized = sanitized.replace(pattern, '');
  });

  // HTML encode if not allowing HTML
  if (!allowHtml) {
    sanitized = sanitized
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;')
      .replace(/\//g, '&#x2F;');
  }

  // Remove special characters if not allowed
  if (!allowSpecialChars) {
    sanitized = sanitized.replace(/[<>{}[\]^`|\\]/g, '');
  }

  return sanitized;
};

export const sanitizeEmail = (email: string): string => {
  return sanitizeInput(email, { maxLength: 254, allowSpecialChars: true })
    .toLowerCase()
    .trim();
};

export const sanitizeUrl = (url: string): string => {
  try {
    const parsed = new URL(url);
    // Only allow http and https protocols
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      return '';
    }
    return parsed.toString();
  } catch {
    return '';
  }
};

// ============================================================================
// SECURITY FIX #3: Enhanced Password Validation
// ============================================================================

export interface PasswordStrength {
  score: 0 | 1 | 2 | 3 | 4; // 0 = invalid, 4 = strong
  isValid: boolean;
  feedback: string[];
  entropy: number;
}

export const validatePassword = (password: string): PasswordStrength => {
  const feedback: string[] = [];
  let score = 0;

  if (!password) {
    return {
      score: 0,
      isValid: false,
      feedback: ['Password is required'],
      entropy: 0,
    };
  }

  // Length requirements
  if (password.length < 8) {
    feedback.push('Minimum 8 characters required');
  } else {
    score++;
  }

  if (password.length < 12) {
    feedback.push('12+ characters recommended for stronger password');
  }

  // Character variety
  const hasLower = /[a-z]/.test(password);
  const hasUpper = /[A-Z]/.test(password);
  const hasNumber = /\d/.test(password);
  const hasSpecial = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password);

  if (!hasLower) {
    feedback.push('Add lowercase letters');
  } else {
    score++;
  }

  if (!hasUpper) {
    feedback.push('Add uppercase letters');
  } else {
    score++;
  }

  if (!hasNumber) {
    feedback.push('Add numbers');
  } else {
    score++;
  }

  if (!hasSpecial) {
    feedback.push('Special characters recommended');
  }

  // Calculate entropy (simplified)
  let charsetSize = 0;
  if (hasLower) charsetSize += 26;
  if (hasUpper) charsetSize += 26;
  if (hasNumber) charsetSize += 10;
  if (hasSpecial) charsetSize += 32;

  const entropy = Math.log2(Math.pow(charsetSize, password.length));

  // Check for common patterns
  const commonPatterns = [
    /(.)\1{2,}/, // Repeating characters
    /123|234|345|456|567|678|789|890/, // Sequential numbers
    /abc|bcd|cde|def|efg|fgh|ghi|hij|ijk/, // Sequential letters
  ];

  const hasCommonPattern = commonPatterns.some(pattern => 
    pattern.test(password.toLowerCase())
  );

  if (hasCommonPattern) {
    feedback.push('Avoid common patterns or sequential characters');
    score = Math.max(0, score - 1);
  }

  const isValid = score >= 2 && password.length >= 8;

  return {
    score: Math.min(4, score) as 0 | 1 | 2 | 3 | 4,
    isValid,
    feedback,
    entropy,
  };
};

// ============================================================================
// SECURITY FIX #4: Rate Limiting & Brute Force Protection
// ============================================================================

interface RateLimitConfig {
  maxAttempts: number;
  windowMs: number;
  lockoutMs: number;
}

class BruteForceProtection {
  private attempts: Map<string, { count: number; timestamp: number }> = new Map();
  private locked: Map<string, number> = new Map();

  constructor(private config: RateLimitConfig) {
    // Cleanup old entries every minute
    setInterval(() => this.cleanup(), 60000);
  }

  isLocked(identifier: string): { locked: boolean; remainingMs: number } {
    const lockoutTime = this.locked.get(identifier);

    if (!lockoutTime) {
      return { locked: false, remainingMs: 0 };
    }

    const remainingMs = lockoutTime - Date.now();

    if (remainingMs <= 0) {
      this.locked.delete(identifier);
      return { locked: false, remainingMs: 0 };
    }

    return { locked: true, remainingMs };
  }

  recordAttempt(identifier: string): {
    allowed: boolean;
    attemptsRemaining: number;
    lockedUntil: number | null;
  } {
    // Check if already locked
    const lockStatus = this.isLocked(identifier);
    if (lockStatus.locked) {
      return {
        allowed: false,
        attemptsRemaining: 0,
        lockedUntil: Date.now() + lockStatus.remainingMs,
      };
    }

    const now = Date.now();
    const attempt = this.attempts.get(identifier);

    if (!attempt || now - attempt.timestamp > this.config.windowMs) {
      // First attempt or window expired
      this.attempts.set(identifier, { count: 1, timestamp: now });
      return {
        allowed: true,
        attemptsRemaining: this.config.maxAttempts - 1,
        lockedUntil: null,
      };
    }

    // Within window, increment
    attempt.count++;

    if (attempt.count >= this.config.maxAttempts) {
      const lockoutUntil = now + this.config.lockoutMs;
      this.locked.set(identifier, lockoutUntil);
      this.attempts.delete(identifier);

      return {
        allowed: false,
        attemptsRemaining: 0,
        lockedUntil: lockoutUntil,
      };
    }

    return {
      allowed: true,
      attemptsRemaining: this.config.maxAttempts - attempt.count,
      lockedUntil: null,
    };
  }

  private cleanup(): void {
    const now = Date.now();

    // Clean expired attempts
    for (const [key, value] of this.attempts.entries()) {
      if (now - value.timestamp > this.config.windowMs) {
        this.attempts.delete(key);
      }
    }

    // Clean expired lockouts
    for (const [key, lockoutTime] of this.locked.entries()) {
      if (now > lockoutTime) {
        this.locked.delete(key);
      }
    }
  }

  reset(identifier: string): void {
    this.attempts.delete(identifier);
    this.locked.delete(identifier);
  }
}

export const bruteForceProtection = new BruteForceProtection({
  maxAttempts: 5,
  windowMs: 15 * 60 * 1000, // 15 minutes
  lockoutMs: 30 * 60 * 1000, // 30 minutes
});

// ============================================================================
// PERFORMANCE FIX #1: Complete Particle System with Memoization
// ============================================================================

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

export class OptimizedParticleSystem {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private particles: Particle[] = [];
  private animationId: number | null = null;
  private isRunning: boolean = false;
  private container: HTMLElement;
  private mouse: { x: number; y: number; isMoving: boolean };
  private mouseTimeout: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private dpr: number = 1;
  private lastFrameTime: number = 0;
  private frameCount: number = 0;
  private fps: number = 60;

  constructor(
    container: HTMLElement,
    private config = {
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
      maxParticles: 500,
      enableFpsControl: true,
      targetFps: 60,
    }
  ) {
    this.mouse = { x: 0, y: 0, isMoving: false };
    this.container = container;

    try {
      this.dpr = Math.min(window.devicePixelRatio || 1, 2); // Cap at 2 for mobile
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

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
  }

  private handleMouseMove = (e: MouseEvent): void => {
    try {
      const rect = this.container.getBoundingClientRect();
      this.mouse.x = e.clientX - rect.left;
      this.mouse.y = e.clientY - rect.top;
      this.mouse.isMoving = true;

      if (this.mouseTimeout) clearTimeout(this.mouseTimeout);
      this.mouseTimeout = window.setTimeout(() => {
        this.mouse.isMoving = false;
      }, 100);
    } catch (error) {
      console.error('Mouse move handler error:', error);
    }
  };

  private handleMouseLeave = (): void => {
    this.mouse.isMoving = false;
    if (this.mouseTimeout) {
      clearTimeout(this.mouseTimeout);
      this.mouseTimeout = null;
    }
  };

  private resize(): void {
    try {
      const rect = this.container.getBoundingClientRect();
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);

      this.canvas.width = rect.width * this.dpr;
      this.canvas.height = rect.height * this.dpr;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;

      this.ctx.scale(this.dpr, this.dpr);
    } catch (error) {
      console.error('Resize error:', error);
    }
  }

  private createParticles(): void {
    const targetCount = Math.min(
      this.config.particleCount,
      this.config.maxParticles
    );
    this.particles = Array.from({ length: targetCount }, () => this.createParticle());
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
      const distanceSq = dx * dx + dy * dy;
      const radiusSq = this.config.mouseRadius * this.config.mouseRadius;

      if (distanceSq < radiusSq && distanceSq > 0) {
        const distance = Math.sqrt(distanceSq);
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
      particle.x,
      particle.y,
      0,
      particle.x,
      particle.y,
      particle.currentSize * 3
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
        const distanceSq = dx * dx + dy * dy;
        const connectionDistSq = this.config.connectionDistance * this.config.connectionDistance;

        if (distanceSq < connectionDistSq) {
          const distance = Math.sqrt(distanceSq);
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

    // FPS control
    if (this.config.enableFpsControl) {
      const now = performance.now();
      const deltaTime = now - this.lastFrameTime;
      const frameTime = 1000 / this.config.targetFps;

      if (deltaTime < frameTime) {
        this.animationId = requestAnimationFrame(this.animate);
        return;
      }

      this.lastFrameTime = now - (deltaTime % frameTime);
      this.frameCount++;
    }

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
    this.lastFrameTime = performance.now();
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
    if (!this.isRunning) this.start();
  }

  public destroy(): void {
    this.pause();

    try {
      this.container.removeEventListener('mousemove', this.handleMouseMove);
      this.container.removeEventListener('mouseleave', this.handleMouseLeave);

      if (this.resizeObserver) {
        this.resizeObserver.disconnect();
        this.resizeObserver = null;
      }

      if (this.mouseTimeout) {
        clearTimeout(this.mouseTimeout);
        this.mouseTimeout = null;
      }

      if (this.canvas?.parentNode) {
        this.canvas.parentNode.removeChild(this.canvas);
      }

      this.particles = [];
    } catch (error) {
      console.error('Error destroying particle system:', error);
    }
  }

  public addBurst(x: number, y: number, count: number = 10): void {
    try {
      if (this.particles.length >= this.config.maxParticles) {
        return;
      }

      const maxToAdd = Math.min(
        count,
        this.config.maxParticles - this.particles.length
      );

      for (let i = 0; i < maxToAdd; i++) {
        const particle = this.createParticle(x, y);
        const angle = (Math.PI * 2 * i) / maxToAdd + (Math.random() - 0.5) * 0.3;
        const speed = 2 + Math.random() * 3;
        particle.vx = Math.cos(angle) * speed;
        particle.vy = Math.sin(angle) * speed;
        this.particles.push(particle);
      }
    } catch (error) {
      console.error('Error adding particle burst:', error);
    }
  }

  public setParticleCount(count: number): void {
    const maxCount = Math.min(count, this.config.maxParticles);
    if (this.particles.length < maxCount) {
      while (this.particles.length < maxCount) {
        this.particles.push(this.createParticle());
      }
    } else if (this.particles.length > maxCount) {
      this.particles.splice(maxCount);
    }
  }

  public getStats(): { fps: number; particleCount: number; memory: number } {
    return {
      fps: this.fps,
      particleCount: this.particles.length,
      memory: performance.memory?.usedJSHeapSize || 0,
    };
  }
}