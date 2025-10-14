import { gsap } from 'gsap';
import { AnimationConfig } from './types/animation';

/**
 * Animate an element with fade in and slide up effect
 * @param element Element to animate
 * @param config Animation configuration
 */
export function fadeInUp(
  element: HTMLElement,
  config: AnimationConfig = {}
): gsap.core.Tween {
  const { delay = 0, duration = 0.6, ease = 'power2.out' } = config;

  return gsap.fromTo(
    element,
    { opacity: 0, y: 20 },
    {
      opacity: 1,
      y: 0,
      duration,
      ease,
      delay,
    }
  );
}

/**
 * Animate an element with fade in and slide left effect
 * @param element Element to animate
 * @param config Animation configuration
 */
export function fadeInLeft(
  element: HTMLElement,
  config: AnimationConfig = {}
): gsap.core.Tween {
  const { delay = 0, duration = 0.6, ease = 'power2.out' } = config;

  return gsap.fromTo(
    element,
    { opacity: 0, x: -20 },
    {
      opacity: 1,
      x: 0,
      duration,
      ease,
      delay,
    }
  );
}

/**
 * Animate an element with scale in effect
 * @param element Element to animate
 * @param config Animation configuration
 */
export function scaleIn(
  element: HTMLElement,
  config: AnimationConfig = {}
): gsap.core.Tween {
  const { delay = 0, duration = 0.5, ease = 'back.out(1.7)' } = config;

  return gsap.fromTo(
    element,
    { scale: 0.8, opacity: 0 },
    {
      scale: 1,
      opacity: 1,
      duration,
      ease,
      delay,
    }
  );
}

/**
 * Animate a progressive reveal of child elements
 * @param container Container element
 * @param config Animation configuration
 */
export function progressiveReveal(
  container: HTMLElement,
  config: AnimationConfig & { stagger?: number } = {}
): gsap.core.Timeline {
  const { delay = 0, duration = 0.4, stagger = 0.1 } = config;
  const children = container.children;

  const tl = gsap.timeline({ delay });

  Array.from(children).forEach((child, index) => {
    tl.fromTo(
      child,
      { opacity: 0, y: 10 },
      {
        opacity: 1,
        y: 0,
        duration,
      },
      index * stagger
    );
  });

  return tl;
}

/**
 * Create a pulse animation
 * @param element Element to animate
 * @param config Animation configuration
 */
export function pulse(
  element: HTMLElement,
  config: AnimationConfig & { scale?: number } = {}
): gsap.core.Tween {
  const { duration = 1.5, scale = 1.05 } = config;

  return gsap.to(element, {
    scale,
    opacity: 0.9,
    duration: duration / 2,
    repeat: -1,
    yoyo: true,
    ease: 'sine.inOut',
  });
}

/**
 * Create a rotation animation
 * @param element Element to animate
 * @param config Animation configuration
 */
export function rotate(
  element: HTMLElement,
  config: AnimationConfig & { degrees?: number } = {}
): gsap.core.Tween {
  const { duration = 2, degrees = 360 } = config;

  return gsap.to(element, {
    rotation: degrees,
    duration,
    ease: 'none',
    repeat: -1,
  });
}

/**
 * Setup scroll-based animations for a container
 * @param container Container element
 * @param options Animation options
 */
export function setupScrollAnimations(
  container: HTMLElement,
  options: {
    threshold?: number | number[];
    rootMargin?: string;
    animation?: (element: HTMLElement, ratio: number) => void;
  } = {}
): IntersectionObserver {
  const {
    threshold = 0.1,
    rootMargin = '0px',
    animation = fadeInUp,
  } = options;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const element = entry.target as HTMLElement;
          animation(element, { delay: 0.1 });
          observer.unobserve(element);
        }
      });
    },
    {
      threshold,
      rootMargin,
    }
  );

  // Observe all elements with data-animate attribute
  const elements = container.querySelectorAll('[data-animate]');
  elements.forEach((element) => {
    observer.observe(element);
  });

  return observer;
}
