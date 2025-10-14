import React, { useState } from 'react';
import { Settings, Bell, Zap, Shield, Palette, Gauge, Save, Globe, Moon, Sun } from 'lucide-react';

// ============================================================================
// TOGGLE SWITCH COMPONENT
// ============================================================================

function ToggleSwitch({ checked, onChange, label, description }) {
  return (
    <div className="flex items-center justify-between p-4 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors">
      <div className="flex-1">
        <div className="text-gray-200 font-medium">{label}</div>
        {description && <div className="text-sm text-gray-500 mt-1">{description}</div>}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative w-12 h-6 rounded-full transition-colors ${
          checked ? 'bg-cyan-500' : 'bg-gray-600'
        }`}
      >
        <div
          className={`absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
            checked ? 'translate-x-6' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  );
}

// ============================================================================
// SETTINGS PAGE
// ============================================================================

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('general');
  const [settings, setSettings] = useState({
    // General
    theme: 'dark',
    language: 'en',
    timezone: 'UTC',
    
    // Notifications
    emailNotifications: true,
    pushNotifications: true,
    agentUpdates: true,
    projectAlerts: true,
    weeklyReports: false,
    
    // Appearance
    animations: true,
    particles: true,
    reducedMotion: false,
    compactMode: false,
    
    // Performance
    performance: 'balanced',
    autoRefresh: true,
    refreshInterval: 5,
    cacheEnabled: true,
    
    // Agents
    autoAssign: true,
    priorityMode: false,
    maxConcurrentTasks: 5,
    
    // Security
    twoFactorAuth: false,
    sessionTimeout: 30,
    loginAlerts: true
  });
  
  const updateSetting = (key, value) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };
  
  const tabs = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'appearance', label: 'Appearance', icon: Palette },
    { id: 'performance', label: 'Performance', icon: Gauge },
    { id: 'agents', label: 'Agents', icon: Zap },
    { id: 'security', label: 'Security', icon: Shield }
  ];
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">
          Settings
        </h1>
        <p className="text-gray-400">Customize your AgentFlow experience</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar Navigation */}
        <div className="lg:col-span-1">
          <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-4 space-y-2 sticky top-20">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'text-gray-400 hover:text-cyan-400 hover:bg-cyan-500/10'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
        
        {/* Settings Content */}
        <div className="lg:col-span-3">
          <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
            {/* General Settings */}
            {activeTab === 'general' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-200 mb-4">General Settings</h2>
                  <p className="text-gray-400 mb-6">Configure your basic preferences</p>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-gray-300 font-medium mb-2">Theme</label>
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { value: 'dark', label: 'Dark', icon: Moon },
                        { value: 'light', label: 'Light', icon: Sun }
                      ].map(theme => (
                        <button
                          key={theme.value}
                          onClick={() => updateSetting('theme', theme.value)}
                          className={`p-4 rounded-lg border transition-all flex items-center justify-center space-x-2 ${
                            settings.theme === theme.value
                              ? 'bg-cyan-500/20 border-cyan-500/30 text-cyan-400'
                              : 'bg-gray-800/30 border-gray-700/30 text-gray-400 hover:border-cyan-500/30'
                          }`}
                        >
                          <theme.icon className="w-5 h-5" />
                          <span>{theme.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-gray-300 font-medium mb-2">Language</label>
                    <select
                      value={settings.language}
                      onChange={(e) => updateSetting('language', e.target.value)}
                      className="w-full px-4 py-3 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                    >
                      <option value="en">English</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                      <option value="de">German</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-gray-300 font-medium mb-2">Timezone</label>
                    <select
                      value={settings.timezone}
                      onChange={(e) => updateSetting('timezone', e.target.value)}
                      className="w-full px-4 py-3 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                    >
                      <option value="UTC">UTC</option>
                      <option value="EST">EST</option>
                      <option value="PST">PST</option>
                      <option value="CET">CET</option>
                    </select>
                  </div>
                </div>
              </div>
            )}
            
            {/* Notifications */}
            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-200 mb-4">Notification Preferences</h2>
                  <p className="text-gray-400 mb-6">Control how and when you receive updates</p>
                </div>
                
                <div className="space-y-3">
                  <ToggleSwitch
                    checked={settings.emailNotifications}
                    onChange={(val) => updateSetting('emailNotifications', val)}
                    label="Email Notifications"
                    description="Receive updates via email"
                  />
                  <ToggleSwitch
                    checked={settings.pushNotifications}
                    onChange={(val) => updateSetting('pushNotifications', val)}
                    label="Push Notifications"
                    description="Browser push notifications"
                  />
                  <ToggleSwitch
                    checked={settings.agentUpdates}
                    onChange={(val) => updateSetting('agentUpdates', val)}
                    label="Agent Status Updates"
                    description="Get notified when agents complete tasks"
                  />
                  <ToggleSwitch
                    checked={settings.projectAlerts}
                    onChange={(val) => updateSetting('projectAlerts', val)}
                    label="Project Alerts"
                    description="Important project milestone notifications"
                  />
                  <ToggleSwitch
                    checked={settings.weeklyReports}
                    onChange={(val) => updateSetting('weeklyReports', val)}
                    label="Weekly Reports"
                    description="Receive weekly performance summaries"
                  />
                </div>
              </div>
            )}
            
            {/* Appearance */}
            {activeTab === 'appearance' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-200 mb-4">Appearance & Visual Effects</h2>
                  <p className="text-gray-400 mb-6">Customize the look and feel</p>
                </div>
                
                <div className="space-y-3">
                  <ToggleSwitch
                    checked={settings.animations}
                    onChange={(val) => updateSetting('animations', val)}
                    label="Enable Animations"
                    description="Smooth transitions and effects"
                  />
                  <ToggleSwitch
                    checked={settings.particles}
                    onChange={(val) => updateSetting('particles', val)}
                    label="Particle Effects"
                    description="Background particle system"
                  />
                  <ToggleSwitch
                    checked={settings.reducedMotion}
                    onChange={(val) => updateSetting('reducedMotion', val)}
                    label="Reduced Motion"
                    description="Minimize animations for accessibility"
                  />
                  <ToggleSwitch
                    checked={settings.compactMode}
                    onChange={(val) => updateSetting('compactMode', val)}
                    label="Compact Mode"
                    description="Denser UI with smaller spacing"
                  />
                </div>
              </div>
            )}
            
            {/* Performance */}
            {activeTab === 'performance' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-200 mb-4">Performance Settings</h2>
                  <p className="text-gray-400 mb-6">Optimize system performance</p>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-gray-300 font-medium mb-3">Performance Mode</label>
                    <div className="grid grid-cols-3 gap-3">
                      {['low', 'balanced', 'high'].map(mode => (
                        <button
                          key={mode}
                          onClick={() => updateSetting('performance', mode)}
                          className={`p-3 rounded-lg border transition-all ${
                            settings.performance === mode
                              ? 'bg-cyan-500/20 border-cyan-500/30 text-cyan-400'
                              : 'bg-gray-800/30 border-gray-700/30 text-gray-400 hover:border-cyan-500/30'
                          }`}
                        >
                          {mode.charAt(0).toUpperCase() + mode.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <ToggleSwitch
                    checked={settings.autoRefresh}
                    onChange={(val) => updateSetting('autoRefresh', val)}
                    label="Auto Refresh"
                    description="Automatically refresh data"
                  />
                  
                  <div>
                    <label className="block text-gray-300 font-medium mb-2">Refresh Interval (seconds)</label>
                    <input
                      type="number"
                      value={settings.refreshInterval}
                      onChange={(e) => updateSetting('refreshInterval', parseInt(e.target.value))}
                      min="1"
                      max="60"
                      className="w-full px-4 py-3 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                    />
                  </div>
                  
                  <ToggleSwitch
                    checked={settings.cacheEnabled}
                    onChange={(val) => updateSetting('cacheEnabled', val)}
                    label="Enable Caching"
                    description="Cache data for faster loading"
                  />
                </div>
              </div>
            )}
            
            {/* Agents */}
            {activeTab === 'agents' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-200 mb-4">Agent Configuration</h2>
                  <p className="text-gray-400 mb-6">Control agent behavior and automation</p>
                </div>
                
                <div className="space-y-4">
                  <ToggleSwitch
                    checked={settings.autoAssign}
                    onChange={(val) => updateSetting('autoAssign', val)}
                    label="Auto-Assign Tasks"
                    description="Automatically assign tasks to available agents"
                  />
                  <ToggleSwitch
                    checked={settings.priorityMode}
                    onChange={(val) => updateSetting('priorityMode', val)}
                    label="Priority Mode"
                    description="Agents prioritize high-importance tasks"
                  />
                  
                  <div>
                    <label className="block text-gray-300 font-medium mb-2">Max Concurrent Tasks Per Agent</label>
                    <input
                      type="number"
                      value={settings.maxConcurrentTasks}
                      onChange={(e) => updateSetting('maxConcurrentTasks', parseInt(e.target.value))}
                      min="1"
                      max="10"
                      className="w-full px-4 py-3 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                    />
                  </div>
                </div>
              </div>
            )}
            
            {/* Security */}
            {activeTab === 'security' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-2xl font-bold text-gray-200 mb-4">Security Settings</h2>
                  <p className="text-gray-400 mb-6">Protect your account and data</p>
                </div>
                
                <div className="space-y-4">
                  <ToggleSwitch
                    checked={settings.twoFactorAuth}
                    onChange={(val) => updateSetting('twoFactorAuth', val)}
                    label="Two-Factor Authentication"
                    description="Add an extra layer of security"
                  />
                  
                  <div>
                    <label className="block text-gray-300 font-medium mb-2">Session Timeout (minutes)</label>
                    <select
                      value={settings.sessionTimeout}
                      onChange={(e) => updateSetting('sessionTimeout', parseInt(e.target.value))}
                      className="w-full px-4 py-3 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white focus:outline-none focus:border-cyan-400"
                    >
                      <option value="15">15 minutes</option>
                      <option value="30">30 minutes</option>
                      <option value="60">1 hour</option>
                      <option value="120">2 hours</option>
                    </select>
                  </div>
                  
                  <ToggleSwitch
                    checked={settings.loginAlerts}
                    onChange={(val) => updateSetting('loginAlerts', val)}
                    label="Login Alerts"
                    description="Get notified of new login attempts"
                  />
                </div>
              </div>
            )}
            
            {/* Save Button */}
            <div className="mt-8 pt-6 border-t border-cyan-500/20">
              <button className="w-full py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg font-medium hover:shadow-lg hover:shadow-cyan-500/50 transition-all flex items-center justify-center space-x-2">
                <Save className="w-5 h-5" />
                <span>Save Settings</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}