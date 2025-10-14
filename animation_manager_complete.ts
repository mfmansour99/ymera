import gsap from 'gsap';
import { AnimationConfig, AnimationType, PerformanceMetrics } from '../types/animation';

export class AnimationManager {
  private activeAnimations: Map<HTMLElement, gsap.core.Timeline>;
  private observers: Map<string, IntersectionObserver>;
  private rafCallbacks: Set<number>;
  private isRAFRunning: boolean;
  private performanceSupported: boolean;
  private prefersReducedMotion: boolean;

  constructor() {
    this.activeAnimations = new Map();
    this.observers = new Map();
    this.rafCallbacks = new Set();
    this.isRAFRunning = false;
    this.performanceSupported = 'performance' in window && !!window.performance.mark;
    this.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.init();
  }

  private init(): void {
    this.setupScrollAnimations();
    this.setupPerformanceMonitoring();
    this.setupGestureSupport();
    this.setupAccessibilityListener();
  }

  private setupAccessibilityListener(): void {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    mediaQuery.addEventListener('change', (e) => {
      this.prefersReducedMotion = e.matches;
      if (this.prefersReducedMotion) {
        this.killAllAnimations();
      }
    });
  }

  private setupScrollAnimations(): void {
    const observerOptions: IntersectionObserverInit = {
      threshold: [0.1, 0.25, 0.5, 0.75, 1.0],
      rootMargin: '0px 0px -50px 0px',
    };

    const scrollObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const element = entry.target as HTMLElement;
        const animationType = (element.dataset.animation as AnimationType) || 'fadeInUp';
        
        if (entry.isIntersecting) {
          this.triggerScrollAnimation(element, animationType, entry.intersectionRatio);
        } else if (element.dataset.animateOnce !== 'true') {
          // Reset animation if element leaves viewport and should repeat
          element.classList.remove('animate-in');
        }
      });
    }, observerOptions);

    this.observers.set('scroll', scrollObserver);
  }

  private setupPerformanceMonitoring(): void {
    if (!this.performanceSupported) return;

    // Monitor long tasks
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.duration > 50) {
              console.warn('Long animation task detected:', entry);
            }
          }
        });
        observer.observe({ entryTypes: ['measure'] });
      } catch (e) {
        console.warn('Performance monitoring not supported');
      }
    }
  }

  private setupGestureSupport(): void {
    // Add touch event support for mobile animations
    document.addEventListener('touchstart', this.handleTouch, { passive: true });
  }

  private handleTouch = (): void => {
    // Add any touch-specific animation logic here
  };

  public observeElement(element: HTMLElement): void {
    const scrollObserver = this.observers.get('scroll');
    if (scrollObserver) {
      scrollObserver.observe(element);
    }
  }

  public unobserveElement(element: HTMLElement): void {
    const scrollObserver = this.observers.get('scroll');
    if (scrollObserver) {
      scrollObserver.unobserve(element);
    }
  }

  public triggerScrollAnimation(
    element: HTMLElement,
    animationType: AnimationType,
    ratio: number,
  ): void {
    if (this.prefersReducedMotion) {
      element.classList.add('animate-instant');
      return;
    }

    // Don't re-animate if already animated
    if (element.classList.contains('animate-in')) {
      return;
    }

    element.classList.add('animate-in');
    
    this.measurePerformance(`animation-${animationType}`, () => {
      const timeline = this.createAnimation(element, animationType, ratio);
      if (timeline) {
        this.activeAnimations.set(element, timeline);
        timeline.eventCallback('onComplete', () => {
          this.activeAnimations.delete(element);
        });
      }
    });
  }

  private createAnimation(
    element: HTMLElement,
    animationType: AnimationType,
    ratio: number
  ): gsap.core.Timeline | null {
    const duration = parseFloat(element.dataset.duration || '0.8');
    const delay = parseFloat(element.dataset.delay || '0');
    const ease = element.dataset.ease || 'power2.out';

    const timeline = gsap.timeline({ delay });

    switch (animationType) {
      case 'fadeInUp':
        return timeline.fromTo(
          element,
          { opacity: 0, y: 50 },
          { opacity: 1, y: 0, duration, ease }
        );

      case 'fadeInDown':
        return timeline.fromTo(
          element,
          { opacity: 0, y: -50 },
          { opacity: 1, y: 0, duration, ease }
        );

      case 'fadeInLeft':
        return timeline.fromTo(
          element,
          { opacity: 0, x: -50 },
          { opacity: 1, x: 0, duration, ease }
        );

      case 'fadeInRight':
        return timeline.fromTo(
          element,
          { opacity: 0, x: 50 },
          { opacity: 1, x: 0, duration, ease }
        );

      case 'fadeIn':
        return timeline.fromTo(
          element,
          { opacity: 0 },
          { opacity: 1, duration, ease }
        );

      case 'scaleIn':
        return timeline.fromTo(
          element,
          { opacity: 0, scale: 0.8 },
          { opacity: 1, scale: 1, duration, ease }
        );

      case 'slideInUp':
        return timeline.fromTo(
          element,
          { opacity: 0, y: 100 },
          { opacity: 1, y: 0, duration, ease }
        );

      case 'slideInDown':
        return timeline.fromTo(
          element,
          { opacity: 0, y: -100 },
          { opacity: 1, y: 0, duration, ease }
        );

      case 'rotateIn':
        return timeline.fromTo(
          element,
          { opacity: 0, rotation: -180 },
          { opacity: 1, rotation: 0, duration, ease }
        );

      case 'zoomIn':
        return timeline.fromTo(
          element,
          { opacity: 0, scale: 0.5 },
          { opacity: 1, scale: 1, duration, ease }
        );

      case 'bounceIn':
        return timeline.fromTo(
          element,
          { opacity: 0, scale: 0.3 },
          { opacity: 1, scale: 1, duration, ease: 'bounce.out' }
        );

      case 'flipIn':
        return timeline.fromTo(
          element,
          { opacity: 0, rotationY: -90 },
          { opacity: 1, rotationY: 0, duration, ease }
        );

      default:
        console.warn(`Unknown animation type: ${animationType}`);
        return timeline.fromTo(
          element,
          { opacity: 0 },
          { opacity: 1, duration, ease }
        );
    }
  }

  public animateElement(
    element: HTMLElement,
    config: AnimationConfig
  ): gsap.core.Timeline {
    const timeline = gsap.timeline({
      delay: config.delay || 0,
      onComplete: config.onComplete,
      onStart: config.onStart,
    });

    timeline.to(element, {
      ...config.properties,
      duration: config.duration || 0.5,
      ease: config.ease || 'power2.out',
    });

    return timeline;
  }

  public staggerAnimation(
    elements: HTMLElement[],
    config: AnimationConfig
  ): gsap.core.Timeline {
    const timeline = gsap.timeline({
      delay: config.delay || 0,
    });

    timeline.fromTo(
      elements,
      config.from || {},
      {
        ...config.properties,
        duration: config.duration || 0.5,
        ease: config.ease || 'power2.out',
        stagger: config.stagger || 0.1,
        onComplete: config.onComplete,
      }
    );

    return timeline;
  }

  private measurePerformance(label: string, callback: () => void): void {
    if (!this.performanceSupported) {
      callback();
      return;
    }

    try {
      performance.mark(`${label}-start`);
      callback();
      performance.mark(`${label}-end`);
      performance.measure(label, `${label}-start`, `${label}-end`);
    } catch (error) {
      console.warn('Performance measurement failed:', error);
      callback();
    }
  }

  public getPerformanceMetrics(): PerformanceMetrics {
    if (!this.performanceSupported) {
      return {
        animationCount: this.activeAnimations.size,
        averageDuration: 0,
        longestDuration: 0,
      };
    }

    const measures = performance.getEntriesByType('measure');
    const animationMeasures = measures.filter(m => m.name.startsWith('animation-'));
    
    if (animationMeasures.length === 0) {
      return {
        animationCount: this.activeAnimations.size,
        averageDuration: 0,
        longestDuration: 0,
      };
    }

    const durations = animationMeasures.map(m => m.duration);
    const totalDuration = durations.reduce((sum, d) => sum + d, 0);
    const averageDuration = totalDuration / durations.length;
    const longestDuration = Math.max(...durations);

    return {
      animationCount: this.activeAnimations.size,
      averageDuration,
      longestDuration,
    };
  }

  public killAnimation(element: HTMLElement): void {
    const timeline = this.activeAnimations.get(element);
    if (timeline) {
      timeline.kill();
      this.activeAnimations.delete(element);
    }
    gsap.killTweensOf(element);
  }

  public killAllAnimations(): void {
    this.activeAnimations.forEach((timeline) => {
      timeline.kill();
    });
    this.activeAnimations.clear();
    gsap.killTweensOf('*');
  }

  public pauseAnimation(element: HTMLElement): void {
    const timeline = this.activeAnimations.get(element);
    if (timeline) {
      timeline.pause();
    }
  }

  public resumeAnimation(element: HTMLElement): void {
    const timeline = this.activeAnimations.get(element);
    if (timeline) {
      timeline.resume();
    }
  }

  public cleanup(): void {
    // Kill all active animations
    this.killAllAnimations();

    // Disconnect all observers
    this.observers.forEach((observer) => {
      observer.disconnect();
    });
    this.observers.clear();

    // Cancel all RAF callbacks
    this.rafCallbacks.forEach((id) => {
      cancelAnimationFrame(id);
    });
    this.rafCallbacks.clear();

    // Remove event listeners
    document.removeEventListener('touchstart', this.handleTouch);

    // Clear performance marks
    if (this.performanceSupported) {
      try {
        performance.clearMarks();
        performance.clearMeasures();
      } catch (error) {
        console.warn('Failed to clear performance data:', error);
      }
    }
  }

  public getActiveAnimationCount(): number {
    return this.activeAnimations.size;
  }

  public isAnimating(element: HTMLElement): boolean {
    return this.activeAnimations.has(element);
  }

  // Utility method for hover animations
  public createHoverAnimation(
    element: HTMLElement,
    hoverConfig: AnimationConfig,
    leaveConfig: AnimationConfig
  ): void {
    element.addEventListener('mouseenter', () => {
      if (this.prefersReducedMotion) return;
      this.animateElement(element, hoverConfig);
    });

    element.addEventListener('mouseleave', () => {
      if (this.prefersReducedMotion) return;
      this.animateElement(element, leaveConfig);
    });
  }

  // Utility method for scroll-triggered timelines
  public createScrollTimeline(
    trigger: HTMLElement,
    animations: Array<{ element: HTMLElement; config: AnimationConfig }>
  ): void {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animations.forEach(({ element, config }) => {
              this.animateElement(element, config);
            });
          }
        });
      },
      { threshold: 0.5 }
    );

    observer.observe(trigger);
    this.observers.set(`scroll-timeline-${trigger.id}`, observer);
  }
}

// Export singleton instance
export const animationManager = new AnimationManager();

// Export utility functions
export const animate = (element: HTMLElement, config: AnimationConfig) =>
  animationManager.animateElement(element, config);

export const staggerAnimate = (elements: HTMLElement[], config: AnimationConfig) =>
  animationManager.staggerAnimation(elements, config);

export const killAnimation = (element: HTMLElement) =>
  animationManager.killAnimation(element);

export const observeForAnimation = (element: HTMLElement) =>
  animationManager.observeElement(element);