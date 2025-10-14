import React, { useState, useCallback, useMemo } from 'react';
import { User } from '../types';

interface LoginPageProps {
  onLogin: (user: User) => void;
}

interface PasswordStrength {
  score: number;
  label: string;
  color: string;
  bgColor: string;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loginAttempts, setLoginAttempts] = useState(0);
  const [isLocked, setIsLocked] = useState(false);
  const [lockoutEndTime, setLockoutEndTime] = useState<number | null>(null);

  const MAX_ATTEMPTS = 5;
  const LOCKOUT_DURATION = 15 * 60 * 1000; // 15 minutes

  // Email validation
  const isValidEmail = useMemo(() => {
    if (!formData.email) return true; // Don't show error on empty
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(formData.email);
  }, [formData.email]);

  // Password strength calculation
  const passwordStrength = useMemo((): PasswordStrength => {
    const password = formData.password;
    if (!password) {
      return { score: 0, label: '', color: '', bgColor: '' };
    }

    let score = 0;
    const checks = {
      length: password.length >= 8,
      lowercase: /[a-z]/.test(password),
      uppercase: /[A-Z]/.test(password),
      numbers: /\d/.test(password),
      special: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)
    };

    // Calculate score
    if (checks.length) score++;
    if (checks.lowercase) score++;
    if (checks.uppercase) score++;
    if (checks.numbers) score++;
    if (checks.special) score++;

    // Additional point for length > 12
    if (password.length >= 12) score++;

    const strengthMap = [
      { score: 0, label: 'Very Weak', color: '#ef4444', bgColor: 'bg-red-500' },
      { score: 1, label: 'Weak', color: '#f59e0b', bgColor: 'bg-orange-500' },
      { score: 2, label: 'Fair', color: '#eab308', bgColor: 'bg-yellow-500' },
      { score: 3, label: 'Good', color: '#84cc16', bgColor: 'bg-lime-500' },
      { score: 4, label: 'Strong', color: '#22c55e', bgColor: 'bg-green-500' },
      { score: 5, label: 'Very Strong', color: '#10b981', bgColor: 'bg-emerald-500' },
      { score: 6, label: 'Excellent', color: '#059669', bgColor: 'bg-emerald-600' }
    ];

    return strengthMap[Math.min(score, 6)];
  }, [formData.password]);

  // Sanitize input to prevent XSS
  const sanitizeInput = useCallback((input: string): string => {
    return input
      .trim()
      .replace(/[<>]/g, '')
      .slice(0, 255);
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    
    if (name === 'email' || name === 'password') {
      const sanitizedValue = sanitizeInput(value);
      setFormData(prev => ({
        ...prev,
        [name]: sanitizedValue
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: type === 'checkbox' ? checked : value
      }));
    }
    
    // Clear error on input change
    if (error) setError(null);
  }, [sanitizeInput, error]);

  const validateForm = useCallback((): boolean => {
    // Check if locked
    if (isLocked && lockoutEndTime) {
      const remainingTime = Math.ceil((lockoutEndTime - Date.now()) / 60000);
      setError(`Account locked. Try again in ${remainingTime} minutes.`);
      return false;
    }

    // Validate email
    if (!formData.email) {
      setError('Email is required');
      return false;
    }

    if (!isValidEmail) {
      setError('Please enter a valid email address');
      return false;
    }

    // Validate password
    if (!formData.password) {
      setError('Password is required');
      return false;
    }

    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters');
      return false;
    }

    if (passwordStrength.score < 2) {
      setError('Password is too weak. Please use a stronger password.');
      return false;
    }

    return true;
  }, [formData, isValidEmail, passwordStrength, isLocked, lockoutEndTime]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsLoading(true);
    setError(null);

    try {
      // Simulate API call with CSRF token
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Simulate authentication
      if (formData.email && formData.password) {
        // Reset attempts on successful login
        setLoginAttempts(0);
        setIsLocked(false);
        setLockoutEndTime(null);

        const userData: User = {
          id: 'user-001',
          name: formData.email.split('@')[0],
          email: formData.email,
          role: 'Administrator',
          avatar: '/assets/avatar.png',
          lastLogin: new Date().toISOString(),
          preferences: {
            theme: 'dark',
            notifications: true,
            language: 'en'
          }
        };
        
        onLogin(userData);
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (err) {
      const newAttempts = loginAttempts + 1;
      setLoginAttempts(newAttempts);

      if (newAttempts >= MAX_ATTEMPTS) {
        const lockoutEnd = Date.now() + LOCKOUT_DURATION;
        setIsLocked(true);
        setLockoutEndTime(lockoutEnd);
        setError(`Too many failed attempts. Account locked for 15 minutes.`);
        
        // Auto unlock after duration
        setTimeout(() => {
          setIsLocked(false);
          setLockoutEndTime(null);
          setLoginAttempts(0);
        }, LOCKOUT_DURATION);
      } else {
        const remainingAttempts = MAX_ATTEMPTS - newAttempts;
        setError(`Invalid email or password. ${remainingAttempts} attempt${remainingAttempts !== 1 ? 's' : ''} remaining.`);
      }
    } finally {
      setIsLoading(false);
    }
  }, [formData, loginAttempts, validateForm, onLogin]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-bg to-secondary-bg p-4">
      <div className="glass-card w-full max-w-md p-8">
        <div className="text-center mb-8">
          <div className="w-16 h-16 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary mx-auto mb-4 flex items-center justify-center">
            <i className="fas fa-atom text-white text-2xl" />
          </div>
          <h1 className="text-2xl font-bold mb-1 text-gradient">AgentFlow</h1>
          <p className="text-secondary">Advanced AI Project Management</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6" noValidate>
          {error && (
            <div 
              className="glass-card bg-accent-danger bg-opacity-20 border-accent-danger p-3 rounded-lg"
              role="alert"
            >
              <div className="flex items-center space-x-2">
                <i className="fas fa-exclamation-circle" />
                <span className="text-sm">{error}</span>
              </div>
            </div>
          )}

          {/* Email Field */}
          <div>
            <label htmlFor="email" className="block text-sm font-medium mb-2">
              Email Address
            </label>
            <div className="relative">
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                className={`glass-input w-full ${!isValidEmail && formData.email ? 'border-accent-danger' : ''}`}
                placeholder="your@email.com"
                required
                autoComplete="username"
                disabled={isLoading || isLocked}
                aria-invalid={!isValidEmail && !!formData.email}
                aria-describedby={!isValidEmail && formData.email ? 'email-error' : undefined}
              />
              {formData.email && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  {isValidEmail ? (
                    <i className="fas fa-check-circle text-accent-success" />
                  ) : (
                    <i className="fas fa-times-circle text-accent-danger" />
                  )}
                </div>
              )}
            </div>
            {!isValidEmail && formData.email && (
              <p id="email-error" className="text-xs text-accent-danger mt-1">
                Please enter a valid email address
              </p>
            )}
          </div>

          {/* Password Field */}
          <div>
            <label htmlFor="password" className="block text-sm font-medium mb-2">
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                className="glass-input w-full pr-12"
                placeholder="Enter your password"
                required
                autoComplete="current-password"
                disabled={isLoading || isLocked}
                aria-describedby={formData.password ? 'password-strength' : undefined}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-secondary hover:text-primary transition-colors"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                disabled={isLoading || isLocked}
                tabIndex={0}
              >
                <i className={`fas ${showPassword ? 'fa-eye-slash' : 'fa-eye'}`} />
              </button>
            </div>

            {/* Password Strength Indicator */}
            {formData.password && (
              <div id="password-strength" className="mt-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-secondary">Password Strength</span>
                  <span className="text-xs font-medium" style={{ color: passwordStrength.color }}>
                    {passwordStrength.label}
                  </span>
                </div>
                <div className="w-full bg-glass rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-300 ${passwordStrength.bgColor}`}
                    style={{ 
                      width: `${(passwordStrength.score / 6) * 100}%`,
                      backgroundColor: passwordStrength.color
                    }}
                    role="progressbar"
                    aria-valuenow={passwordStrength.score}
                    aria-valuemin={0}
                    aria-valuemax={6}
                    aria-label={`Password strength: ${passwordStrength.label}`}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Remember Me & Forgot Password */}
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="rememberMe"
                name="rememberMe"
                checked={formData.rememberMe}
                onChange={handleInputChange}
                className="rounded text-accent-primary focus:ring-accent-primary"
                disabled={isLoading || isLocked}
              />
              <label htmlFor="rememberMe" className="ml-2 text-sm text-secondary">
                Remember me
              </label>
            </div>
            <a 
              href="#forgot" 
              className="text-sm text-accent-primary hover:underline"
              tabIndex={isLoading || isLocked ? -1 : 0}
            >
              Forgot password?
            </a>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading || isLocked || !isValidEmail || !formData.password}
            className="glass-button w-full bg-gradient-to-r from-accent-primary to-accent-secondary text-white font-medium py-3 hover:from-accent-secondary hover:to-accent-primary transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <i className="fas fa-spinner fa-spin mr-2" />
                Signing In...
              </>
            ) : isLocked ? (
              <>
                <i className="fas fa-lock mr-2" />
                Account Locked
              </>
            ) : (
              <>
                <i className="fas fa-sign-in-alt mr-2" />
                Sign In
              </>
            )}
          </button>

          {/* Sign Up Link */}
          <div className="text-center text-sm text-secondary">
            Don't have an account?{' '}
            <a 
              href="#signup" 
              className="text-accent-primary hover:underline"
              tabIndex={isLoading || isLocked ? -1 : 0}
            >
              Sign up
            </a>
          </div>
        </form>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-secondary">
          <p>© {new Date().getFullYear()} AgentFlow. All rights reserved.</p>
          <div className="mt-2 flex justify-center space-x-4">
            <a href="#privacy" className="hover:text-primary">Privacy Policy</a>
            <a href="#terms" className="hover:text-primary">Terms of Service</a>
            <a href="#support" className="hover:text-primary">Contact Support</a>
          </div>
        </div>
      </div>
    </div>
  );
};