// ============================================================================
// ACCESSIBILITY FIX #1: Focus Trap Hook for Modals
// ============================================================================

import { useEffect, useRef, useCallback } from 'react';

export const useFocusTrap = (isOpen: boolean) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen || !containerRef.current) return;

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          lastElement.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === lastElement) {
          firstElement.focus();
          e.preventDefault();
        }
      }
    };

    // Trap initial focus
    firstElement.focus();

    container.addEventListener('keydown', handleKeyDown);
    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  return containerRef;
};

// ============================================================================
// ACCESSIBILITY FIX #2: Keyboard Navigation Hook
// ============================================================================

export interface KeyboardNavOptions {
  onEscape?: () => void;
  onEnter?: () => void;
  onArrowUp?: () => void;
  onArrowDown?: () => void;
  onArrowLeft?: () => void;
  onArrowRight?: () => void;
}

export const useKeyboardNavigation = (
  ref: React.RefObject<HTMLElement>,
  options: KeyboardNavOptions
) => {
  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          e.preventDefault();
          options.onEscape?.();
          break;
        case 'Enter':
          e.preventDefault();
          options.onEnter?.();
          break;
        case 'ArrowUp':
          e.preventDefault();
          options.onArrowUp?.();
          break;
        case 'ArrowDown':
          e.preventDefault();
          options.onArrowDown?.();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          options.onArrowLeft?.();
          break;
        case 'ArrowRight':
          e.preventDefault();
          options.onArrowRight?.();
          break;
      }
    };

    element.addEventListener('keydown', handleKeyDown);
    return () => element.removeEventListener('keydown', handleKeyDown);
  }, [ref, options]);
};

// ============================================================================
// ACCESSIBILITY FIX #3: Screen Reader Announcements
// ============================================================================

type AnnouncementRole = 'polite' | 'assertive';

export const useAriaAnnounce = () => {
  const announceRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!announceRef.current) {
      const div = document.createElement('div');
      div.setAttribute