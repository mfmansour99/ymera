import React, { useState, useCallback, useMemo, useEffect, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Save,
  RotateCcw,
  Moon,
  Bell,
  Zap,
  Shield,
  Globe,
  AlertTriangle,
  CheckCircle,
  X,
} from 'lucide-react';

// Safe localStorage helper
const safeStorage = {
  getItem: (key: string, defaultValue: any = null) => {
    try {
      if (typeof window === 'undefined') return defaultValue;
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.warn(`[Storage] Read error for ${key}:`, error);
      return defaultValue;
    }
  },
  setItem: (key: string, value: any) => {
    try {
      if (typeof window === 'undefined') return false;
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      console.warn(`[Storage] Write error for ${key}:`, error);
      return false;
    }
  },
};

// Types
interface SettingsState {
  animations: boolean;
  particles: boolean;
  notifications: boolean;
  autoAssign: boolean;
  performance: 'low' | 'balanced' | 'high';
  language: string;
  theme: 'dark' | 'light' | 'auto';
}

interface SettingOption {
  key: keyof SettingsState;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface PerformanceMode {
  id: 'low' | 'balanced' | 'high';
  label: string;
  description: string;
  particleCount: number;
}

// Constants
const PERFORMANCE_MODES: PerformanceMode[] = [
  {
    id: 'low',
    label: 'Low',
    description: 'Minimal effects',
    particleCount: 20,
  },
  {
    id: 'balanced',
    label: 'Balanced',
    description: 'Optimal balance',
    particleCount: 50,
  },
  {
    id: 'high',
    label: 'High',
    description: 'Full effects',
    particleCount: 100,
  },
];

const APPEARANCE_SETTINGS: SettingOption[] = [
  {
    key: 'animations',
    label: 'Enable Animations',
    description: 'UI animations and transitions',
    icon: Zap,
  },
  {
    key: 'particles',
    label: 'Particle Effects',
    description: 'Background particle system',
    icon: Zap,
  },
];

const NOTIFICATION_SETTINGS: SettingOption[] = [
  {
    key: 'notifications',
    label: 'Enable Notifications',
    description: 'Receive updates about agent activities',
    icon: Bell,
  },
  {
    key: 'autoAssign',
    label: 'Auto-assign Tasks',
    description: 'Automatically assign tasks to available agents',
    icon: Shield,
  },
];

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'ja', name: 'Japanese' },
  { code: 'zh', name: 'Chinese' },
];

// Default settings
const DEFAULT_SETTINGS: SettingsState = {
  animations: true,
  particles: true,
  notifications: true,
  autoAssign: true,
  performance: 'balanced',
  language: 'en',
  theme: 'dark',
};

// Toast notification component
const Toast = memo(
  ({
    message,
    type,
    onClose,
  }: {
    message: string;
    type: 'success' | 'error' | 'warning';
    onClose: () => void;
  }) => {
    useEffect(() => {
      const timer = setTimeout(onClose, 3000);
      return () => clearTimeout(timer);
    }, [onClose]);

    const colors = {
      success: { bg: 'bg-green-500/20', border: 'border-green-500/50', text: 'text-green-400' },
      error: { bg: 'bg-red-500/20', border: 'border-red-500/50', text: 'text-red-400' },
      warning: { bg: 'bg-yellow-500/20', border: 'border-yellow-500/50', text: 'text-yellow-400' },
    };

    const color = colors[type];
    const Icon = type === 'success' ? CheckCircle : AlertTriangle;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        className={`fixed bottom-6 right-6 flex items-center space-x-3 px-4 py-3 ${color.bg} border ${color.border} rounded-lg z-50`}
      >
        <Icon className={`w-5 h-5 ${color.text} flex-shrink-0`} />
        <p className={`text-sm font-medium ${color.text}`}>{message}</p>
        <button
          onClick={onClose}
          className="ml-2 text-current opacity-70 hover:opacity-100"
        >
          <X className="w-4 h-4" />
        </button>
      </motion.div>
    );
  }
);
Toast.displayName = 'Toast';

// Toggle Switch Component
const ToggleSwitch = memo(
  ({
    enabled,
    onChange,
    label,
    description,
    icon: Icon,
  }: {
    enabled: boolean;
    onChange: (value: boolean) => void;
    label: string;
    description: string;
    icon: React.ComponentType<{ className?: string }>;
  }) => (
    <motion.div
      whileHover={{ scale: 1.01 }}
      className="flex items-center justify-between p-4 bg-slate-800/50 hover:bg-slate-800/70 rounded-lg border border-slate-700/50 hover:border-slate-600/50 transition-all cursor-pointer"
      onClick={() => onChange(!enabled)}
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center flex-shrink-0">
          <Icon className="w-5 h-5 text-cyan-400" />
        </div>
        <div className="min-w-0">
          <div className="font-semibold text-white text-sm">{label}</div>
          <div className="text-xs text-slate-400 mt-1">{description}</div>
        </div>
      </div>
      <motion.button
        className={`relative w-12 h-7 rounded-full flex-shrink-0 transition-colors ${
          enabled ? 'bg-cyan-500' : 'bg-slate-600'
        }`}
        onClick={(e) => {
          e.stopPropagation();
          onChange(!enabled);
        }}
      >
        <motion.div
          className="absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full shadow-lg"
          animate={{ x: enabled ? 20 : 0 }}
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        />
      </motion.button>
    </motion.div>
  )
);
ToggleSwitch.displayName = 'ToggleSwitch';

// Settings Section Component
const SettingsSection = memo(
  ({
    title,
    icon: Icon,
    children,
    delay = 0,
  }: {
    title: string;
    icon: React.ComponentType<{ className?: string }>;
    children: React.ReactNode;
    delay?: number;
  }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-2xl p-6"
    >
      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
        <Icon className="w-6 h-6 text-cyan-400" />
        {title}
      </h3>
      <div className="space-y-4">{children}</div>
    </motion.div>
  )
);
SettingsSection.displayName = 'SettingsSection';

// Main Settings Page Component
export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SettingsState>(() =>
    safeStorage.getItem('app_settings', DEFAULT_SETTINGS)
  );

  const [hasChanges, setHasChanges] = useState(false);
  const [toasts, setToasts] = useState<Array<{ id: string; message: string; type: 'success' | 'error' | 'warning' }>>([]);

  // Handle setting changes
  const handleSettingChange = useCallback(
    (key: keyof SettingsState, value: any) => {
      setSettings((prev) => ({ ...prev, [key]: value }));
      setHasChanges(true);
    },
    []
  );

  // Add toast notification
  const addToast = useCallback((message: string, type: 'success' | 'error' | 'warning' = 'success') => {
    const id = Math.random().toString(36).substr(2, 9);
    setToasts((prev) => [...prev, { id, message, type }]);
  }, []);

  // Remove toast
  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Save settings
  const handleSave = useCallback(() => {
    if (safeStorage.setItem('app_settings', settings)) {
      addToast('Settings saved successfully', 'success');
      setHasChanges(false);
    } else {
      addToast('Failed to save settings', 'error');
    }
  }, [settings, addToast]);

  // Reset settings
  const handleReset = useCallback(() => {
    if (typeof window !== 'undefined' && window.confirm('Reset all settings to default? This cannot be undone.')) {
      setSettings(DEFAULT_SETTINGS);
      if (safeStorage.setItem('app_settings', DEFAULT_SETTINGS)) {
        addToast('Settings reset to default', 'success');
        setHasChanges(false);
      } else {
        addToast('Failed to reset settings', 'error');
      }
    }
  }, [addToast]);

  // Performance mode description
  const performanceConfig = useMemo(
    () => PERFORMANCE_MODES.find((m) => m.id === settings.performance),
    [settings.performance]
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 opacity-50" />

      <div className="relative z-10 p-4 sm:p-6 lg:p-8 max-w-5xl mx-auto">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">
            <span className="bg-gradient-to-r from-cyan-400 via-green-400 to-cyan-400 bg-clip-text text-transparent">
              Settings & Configuration
            </span>
          </h1>
          <p className="text-slate-400 text-sm sm:text-base">
            Customize your experience and system preferences
          </p>
        </motion.div>

        {/* Settings Grid */}
        <div className="space-y-6 mb-8">
          {/* Appearance Section */}
          <SettingsSection title="Appearance" icon={Moon} delay={0.1}>
            <div className="space-y-4">
              {APPEARANCE_SETTINGS.map((setting) => (
                <ToggleSwitch
                  key={setting.key}
                  enabled={settings[setting.key] as boolean}
                  onChange={(value) => handleSettingChange(setting.key, value)}
                  label={setting.label}
                  description={setting.description}
                  icon={setting.icon}
                />
              ))}

              {/* Performance Mode */}
              <div className="pt-4 border-t border-slate-700/50">
                <label className="block text-slate-300 font-semibold mb-3 text-sm">
                  Performance Mode
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {PERFORMANCE_MODES.map((mode) => (
                    <motion.button
                      key={mode.id}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleSettingChange('performance', mode.id)}
                      className={`p-4 rounded-lg font-semibold capitalize border-2 transition-all ${
                        settings.performance === mode.id
                          ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40'
                          : 'bg-slate-700/50 text-slate-300 border-slate-600/50 hover:border-slate-500/50'
                      }`}
                    >
                      <div className="text-sm mb-1">{mode.label}</div>
                      <div className="text-xs text-slate-400">{mode.particleCount} particles</div>
                    </motion.button>
                  ))}
                </div>
                {performanceConfig && (
                  <p className="text-xs text-slate-400 mt-2">{performanceConfig.description}</p>
                )}
              </div>
            </div>
          </SettingsSection>

          {/* Notifications Section */}
          <SettingsSection title="Notifications" icon={Bell} delay={0.2}>
            <div className="space-y-4">
              {NOTIFICATION_SETTINGS.map((setting) => (
                <ToggleSwitch
                  key={setting.key}
                  enabled={settings[setting.key] as boolean}
                  onChange={(value) => handleSettingChange(setting.key, value)}
                  label={setting.label}
                  description={setting.description}
                  icon={setting.icon}
                />
              ))}
            </div>
          </SettingsSection>

          {/* Language & Region Section */}
          <SettingsSection title="Language & Region" icon={Globe} delay={0.3}>
            <div className="space-y-4">
              <div>
                <label className="block text-slate-300 font-semibold mb-2 text-sm">
                  Language
                </label>
                <select
                  value={settings.language}
                  onChange={(e) => handleSettingChange('language', e.target.value)}
                  className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-cyan-400/50 transition-colors"
                >
                  {LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-300 font-semibold mb-2 text-sm">
                  Theme
                </label>
                <select
                  value={settings.theme}
                  onChange={(e) => handleSettingChange('theme', e.target.value as 'dark' | 'light' | 'auto')}
                  className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white text-sm focus:outline-none focus:border-cyan-400/50 transition-colors"
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                  <option value="auto">System Default</option>
                </select>
              </div>
            </div>
          </SettingsSection>

          {/* Danger Zone */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-red-500/10 backdrop-blur border border-red-500/30 rounded-2xl p-6"
          >
            <h3 className="text-xl font-bold text-red-400 mb-3 flex items-center gap-2">
              <AlertTriangle className="w-6 h-6" />
              Danger Zone
            </h3>
            <p className="text-slate-400 text-sm mb-4">
              These actions are irreversible. Please be certain before proceeding.
            </p>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleReset}
              className="px-6 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/40 rounded-lg transition-all font-medium text-sm"
            >
              Reset All Settings
            </motion.button>
          </motion.div>
        </div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="sticky bottom-0 p-4 bg-gradient-to-t from-slate-900 to-transparent flex items-center justify-end gap-3 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8"
        >
          {hasChanges && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-xs text-yellow-400 mr-auto flex items-center gap-2"
            >
              <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
              Unsaved changes
            </motion.div>
          )}

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleSave}
            disabled={!hasChanges}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all text-sm ${
              hasChanges
                ? 'bg-gradient-to-r from-cyan-500 to-green-500 text-white shadow-lg hover:shadow-xl'
                : 'bg-slate-700/50 text-slate-500 cursor-not-allowed'
            }`}
          >
            <Save className="w-4 h-4" />
            <span>Save Changes</span>
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => {
              setSettings(safeStorage.getItem('app_settings', DEFAULT_SETTINGS));
              setHasChanges(false);
            }}
            className="flex items-center gap-2 px-6 py-3 bg-slate-700/50 hover:bg-slate-700/70 text-slate-300 rounded-lg font-medium transition-all text-sm"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Discard</span>
          </motion.button>
        </motion.div>
      </div>

      {/* Toast Notifications */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
        <AnimatePresence>
          {toasts.map((toast) => (
            <Toast
              key={toast.id}
              message={toast.message}
              type={toast.type}
              onClose={() => removeToast(toast.id)}
            />
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default SettingsPage;