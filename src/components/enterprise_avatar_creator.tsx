import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuth } from '../store/auth'
import toast from 'react-hot-toast'
import { 
  Zap, 
  Code, 
  Save, 
  RotateCcw, 
  Loader2, 
  Download,
  Upload,
  Palette,
  Settings,
  Eye,
  Sparkles,
  Grid3x3,
  Layers,
  Play,
  Pause,
  SkipBack,
  Volume2,
  VolumeX
} from 'lucide-react'

// Enhanced 3D Scene Mock with advanced visual effects
const EnhancedThreeScene = ({ config, isAnimating, onAnimationToggle }) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const sceneRef = useRef(null)
  
  const handleMouseMove = useCallback((e) => {
    if (!sceneRef.current) return
    const rect = sceneRef.current.getBoundingClientRect()
    const x = (e.clientX - rect.left - rect.width / 2) / (rect.width / 2)
    const y = (e.clientY - rect.top - rect.height / 2) / (rect.height / 2)
    setMousePosition({ x: x * 20, y: -y * 20 })
  }, [])

  const shapeStyles = useMemo(() => {
    const baseTransform = `
      rotateX(${config.shape === 'sphere' ? mousePosition.y : 45 + mousePosition.y}deg) 
      rotateY(${config.rotation + mousePosition.x}deg)
      scale(${1 + config.glowIntensity * 0.02})
    `
    
    return {
      transform: baseTransform,
      borderRadius: config.shape === 'sphere' ? '50%' : config.shape === 'pyramid' ? '20% 20% 50% 50%' : '15%',
      background: `
        radial-gradient(circle at 30% 30%, ${config.color}AA, ${config.baseColor}),
        linear-gradient(45deg, ${config.color}33, ${config.baseColor}CC)
      `,
      boxShadow: `
        0 0 ${config.glowIntensity * 15}px ${config.glowIntensity * 3}px ${config.color}66,
        inset 0 0 ${config.glowIntensity * 5}px ${config.color}33,
        0 ${config.glowIntensity * 2}px ${config.glowIntensity * 8}px rgba(0,0,0,0.3)
      `,
      transition: isAnimating ? 'transform 2s ease-in-out infinite alternate' : 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      animation: isAnimating ? `pulse-glow ${2 + config.glowIntensity * 0.2}s ease-in-out infinite alternate` : 'none',
      width: '100%',
      height: '100%',
      position: 'relative',
      overflow: 'hidden'
    }
  }, [config, mousePosition, isAnimating])

  const particleCount = Math.min(config.glowIntensity * 3, 20)
  const particles = Array.from({ length: particleCount }, (_, i) => (
    <motion.div
      key={i}
      className="absolute w-1 h-1 bg-white rounded-full opacity-70"
      style={{
        left: `${20 + (i * 60 / particleCount)}%`,
        top: `${30 + (i * 40 / particleCount)}%`,
      }}
      animate={{
        x: [0, 20, -15, 0],
        y: [0, -25, 15, 0],
        opacity: [0.7, 1, 0.3, 0.7],
      }}
      transition={{
        duration: 3 + i * 0.5,
        repeat: Infinity,
        ease: "easeInOut",
        delay: i * 0.2
      }}
    />
  ))

  return (
    <div className="flex-1 min-h-[400px] flex items-center justify-center p-6 relative">
      <style>{`
        @keyframes pulse-glow {
          0% { filter: brightness(1) saturate(1); }
          100% { filter: brightness(1.2) saturate(1.4); }
        }
      `}</style>
      
      <div 
        ref={sceneRef}
        className="relative w-64 h-64 perspective-1000 cursor-grab active:cursor-grabbing"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setMousePosition({ x: 0, y: 0 })}
      >
        <div style={shapeStyles} className="absolute inset-0 border border-white/20">
          {/* Core Energy Effect */}
          <div className="absolute inset-4 bg-gradient-to-br from-white/30 to-transparent rounded-full blur-sm" />
          
          {/* Central Icon */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <motion.div
              animate={{ rotate: isAnimating ? 360 : 0 }}
              transition={{ duration: 4, repeat: isAnimating ? Infinity : 0, ease: "linear" }}
            >
              <Zap className="w-10 h-10 text-white drop-shadow-lg" />
            </motion.div>
          </div>
          
          {/* Particle Effects */}
          {particles}
          
          {/* Grid Overlay */}
          {config.showGrid && (
            <div className="absolute inset-0 opacity-20">
              <div className="w-full h-full grid grid-cols-8 grid-rows-8 gap-px">
                {Array.from({ length: 64 }, (_, i) => (
                  <div key={i} className="border border-white/10" />
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* Ambient Light Ring */}
        <motion.div 
          className="absolute -inset-8 rounded-full"
          style={{
            background: `conic-gradient(from 0deg, transparent, ${config.color}22, transparent)`,
            filter: 'blur(4px)'
          }}
          animate={{ rotate: isAnimating ? 360 : 0 }}
          transition={{ duration: 8, repeat: isAnimating ? Infinity : 0, ease: "linear" }}
        />
      </div>

      {/* Animation Controls */}
      <motion.button
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={onAnimationToggle}
        className="absolute top-4 right-4 p-3 bg-black/50 backdrop-blur-sm rounded-full border border-white/20 text-white hover:bg-black/70 transition-colors"
      >
        {isAnimating ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
      </motion.button>
    </div>
  )
}

// Advanced Configuration Options
const AVATAR_CONFIG = {
  shapes: [
    { id: 'sphere', name: 'Sphere', icon: '○' },
    { id: 'octahedron', name: 'Octahedron', icon: '◇' },
    { id: 'pyramid', name: 'Pyramid', icon: '△' },
    { id: 'cube', name: 'Cube', icon: '□' },
    { id: 'torus', name: 'Torus', icon: '◯' }
  ],
  colorPalettes: {
    cyberpunk: [
      { name: 'Neon Pink', value: '#ff0080', base: '#4a0e2f' },
      { name: 'Electric Blue', value: '#00d4ff', base: '#0a2a3a' },
      { name: 'Acid Green', value: '#39ff14', base: '#1a4a0f' },
      { name: 'Cyber Orange', value: '#ff6600', base: '#4a2200' }
    ],
    nature: [
      { name: 'Forest Glow', value: '#64f4ac', base: '#166534' },
      { name: 'Ocean Deep', value: '#4fd1c5', base: '#15803d' },
      { name: 'Sunset Fire', value: '#f97316', base: '#7c2d12' },
      { name: 'Mountain Mist', value: '#a78bfa', base: '#4c1d95' }
    ],
    corporate: [
      { name: 'Corporate Blue', value: '#3b82f6', base: '#1e3a8a' },
      { name: 'Success Green', value: '#10b981', base: '#064e3b' },
      { name: 'Warning Amber', value: '#f59e0b', base: '#92400e' },
      { name: 'Danger Red', value: '#ef4444', base: '#7f1d1d' }
    ]
  },
  effects: [
    { id: 'particles', name: 'Particle Field', icon: Sparkles },
    { id: 'grid', name: 'Cyber Grid', icon: Grid3x3 },
    { id: 'layers', name: 'Depth Layers', icon: Layers }
  ],
  animations: [
    { id: 'rotate', name: 'Rotation' },
    { id: 'pulse', name: 'Pulse' },
    { id: 'float', name: 'Float' },
    { id: 'morph', name: 'Morph' }
  ],
  presets: [
    {
      name: 'Agent Alpha',
      config: { shape: 'sphere', color: '#64f4ac', baseColor: '#166534', glowIntensity: 7, effects: ['particles'] }
    },
    {
      name: 'Cyber Sentinel',
      config: { shape: 'octahedron', color: '#00d4ff', baseColor: '#0a2a3a', glowIntensity: 9, effects: ['grid', 'particles'] }
    },
    {
      name: 'Neural Core',
      config: { shape: 'pyramid', color: '#a78bfa', baseColor: '#4c1d95', glowIntensity: 6, effects: ['layers'] }
    }
  ],
  default: {
    shape: 'sphere',
    color: '#64f4ac',
    baseColor: '#166534',
    glowIntensity: 5,
    rotation: 0,
    effects: [],
    showGrid: false,
    animation: 'rotate'
  }
}

// Color Palette Selector Component
const ColorPaletteSelector = ({ selectedPalette, onPaletteChange, selectedColor, onColorChange }) => (
  <div className="space-y-3">
    <div className="flex gap-2">
      {Object.keys(AVATAR_CONFIG.colorPalettes).map(palette => (
        <motion.button
          key={palette}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => onPaletteChange(palette)}
          className={`px-3 py-1 rounded-lg text-xs font-medium capitalize transition-colors ${
            selectedPalette === palette
              ? 'bg-ymera-glow/20 border border-ymera-glow text-ymera-glow'
              : 'bg-slate-800/50 border border-slate-700 text-slate-300 hover:bg-slate-700/50'
          }`}
        >
          {palette}
        </motion.button>
      ))}
    </div>
    
    <div className="grid grid-cols-2 gap-2">
      {AVATAR_CONFIG.colorPalettes[selectedPalette].map(color => (
        <motion.button
          key={color.value}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => onColorChange(color.value)}
          className={`flex items-center gap-2 p-2 rounded-lg transition-all text-xs ${
            selectedColor === color.value 
              ? 'bg-slate-800 border border-ymera-glow shadow-lg' 
              : 'bg-slate-800/50 border border-slate-700 hover:bg-slate-800/70'
          }`}
        >
          <div
            className="w-4 h-4 rounded-full flex-shrink-0"
            style={{ 
              backgroundColor: color.value, 
              boxShadow: selectedColor === color.value ? `0 0 8px ${color.value}` : 'none' 
            }}
          />
          <span className="text-slate-300 font-medium">{color.name}</span>
        </motion.button>
      ))}
    </div>
  </div>
)

// Preset Selector Component
const PresetSelector = ({ onPresetSelect }) => (
  <div className="grid grid-cols-1 gap-2">
    {AVATAR_CONFIG.presets.map(preset => (
      <motion.button
        key={preset.name}
        whileHover={{ scale: 1.02, x: 4 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => onPresetSelect(preset.config)}
        className="flex items-center justify-between p-3 bg-slate-800/50 border border-slate-700 rounded-lg hover:bg-slate-800/70 transition-colors group"
      >
        <div className="flex items-center gap-3">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: preset.config.color, boxShadow: `0 0 6px ${preset.config.color}` }}
          />
          <span className="text-slate-300 font-medium">{preset.name}</span>
        </div>
        <Eye className="w-4 h-4 text-slate-500 group-hover:text-ymera-glow transition-colors" />
      </motion.button>
    ))}
  </div>
)

export default function EnterpriseAvatarCreator() {
  const navigate = useNavigate()
  const { user, setAvatarConfig, login } = useAuth()
  
  const [avatarConfig, setAvatarConfigState] = useState(AVATAR_CONFIG.default)
  const [selectedPalette, setSelectedPalette] = useState('nature')
  const [isSaving, setIsSaving] = useState(false)
  const [isAnimating, setIsAnimating] = useState(true)
  const [activeTab, setActiveTab] = useState('basic')
  const [soundEnabled, setSoundEnabled] = useState(false)

  // Advanced state management
  const [configHistory, setConfigHistory] = useState([AVATAR_CONFIG.default])
  const [historyIndex, setHistoryIndex] = useState(0)
  const [isExporting, setIsExporting] = useState(false)

  const handleConfigChange = useCallback((key, value) => {
    const newConfig = { 
      ...avatarConfig, 
      [key]: value,
      baseColor: key === 'color' 
        ? AVATAR_CONFIG.colorPalettes[selectedPalette].find(c => c.value === value)?.base || avatarConfig.baseColor
        : avatarConfig.baseColor,
    }
    
    setAvatarConfigState(newConfig)
    
    // Add to history for undo functionality
    const newHistory = configHistory.slice(0, historyIndex + 1)
    newHistory.push(newConfig)
    setConfigHistory(newHistory)
    setHistoryIndex(newHistory.length - 1)
  }, [avatarConfig, selectedPalette, configHistory, historyIndex])

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      setHistoryIndex(prev => prev - 1)
      setAvatarConfigState(configHistory[historyIndex - 1])
      toast.success('Configuration restored')
    }
  }, [historyIndex, configHistory])

  const handlePresetSelect = useCallback((presetConfig) => {
    handleConfigChange('shape', presetConfig.shape)
    handleConfigChange('color', presetConfig.color)
    handleConfigChange('glowIntensity', presetConfig.glowIntensity)
    toast.success('Preset applied successfully')
  }, [handleConfigChange])

  const handleExportConfig = useCallback(async () => {
    setIsExporting(true)
    try {
      const exportData = {
        ...avatarConfig,
        metadata: {
          created: new Date().toISOString(),
          version: '2.0.0',
          userId: user?.id
        }
      }
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `avatar-config-${Date.now()}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      toast.success('Configuration exported successfully')
    } catch (error) {
      toast.error('Failed to export configuration')
    } finally {
      setIsExporting(false)
    }
  }, [avatarConfig, user])

  const handleCreateAvatar = useCallback(async () => {
    setIsSaving(true)
    try {
      const avatarPayload = {
        ...avatarConfig,
        model_url: `/models/avatar/${avatarConfig.shape}.glb`,
        user_id: user?.id,
        created_at: new Date().toISOString(),
        version: '2.0.0'
      }
      
      // Simulate enhanced API call with progress
      const steps = ['Validating configuration', 'Generating 3D model', 'Applying effects', 'Finalizing avatar']
      for (let i = 0; i < steps.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 800))
        toast.loading(`${steps[i]}...`, { id: 'deploy-progress' })
      }
      
      const finalUser = {
        ...user,
        avatar_config: avatarPayload,
        avatar_url: `https://ui-avatars.com/api/?name=${encodeURIComponent(user?.name || 'Agent')}&background=${avatarConfig.color.slice(1)}&color=000000&size=128`
      }
      
      login?.(finalUser)
      toast.success('Your Agent Companion has been successfully deployed!', {
        icon: '🚀',
        id: 'deploy-progress'
      })
      
      // Add deployment sound effect
      if (soundEnabled) {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBSaJzfPVgyEFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBSaJzfPVgyEFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBSaJzfPVgyEFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBSaJzfPVgyEFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBSaJzfPVgyEFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBSaJzfPVgyEFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBSaJzfPVgyEFJHfH8N2QQAoUXrTp66hVFApGn+DyvWIcBQ==')
        audio.play().catch(() => {}) // Ignore autoplay restrictions
      }
      
      navigate('/')
      
    } catch (err) {
      toast.error(err?.message || 'Failed to deploy avatar. Please try again.')
    } finally {
      setIsSaving(false)
    }
  }, [avatarConfig, user, navigate, login, soundEnabled])

  const tabVariants = {
    hidden: { opacity: 0, x: 20 },
    visible: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -20 }
  }

  return (
    <>
      <Helmet>
        <title>Enterprise Avatar Creator | Ymera Agent Platform</title>
      </Helmet>
      
      <div className="min-h-screen p-8 bg-gradient-to-br from-black via-slate-900 to-black">
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12 max-w-3xl mx-auto"
        >
          <h1 className="text-6xl font-extrabold bg-gradient-to-r from-ymera-glow via-white to-ymera-accent bg-clip-text text-transparent mb-6">
            Enterprise Agent Designer
          </h1>
          <p className="text-xl text-slate-400 leading-relaxed">
            Create a sophisticated AI companion with advanced visual effects, customizable behaviors, 
            and enterprise-grade deployment options.
          </p>
        </motion.header>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 max-w-7xl mx-auto">
          
          {/* Enhanced 3D Visualizer Panel */}
          <motion.div 
            initial={{ opacity: 0, x: -50 }} 
            animate={{ opacity: 1, x: 0 }} 
            transition={{ duration: 0.8 }}
            className="xl:col-span-2 bg-gradient-to-br from-slate-900/80 to-slate-800/40 backdrop-blur-xl border border-ymera-glow/20 rounded-3xl shadow-2xl p-8 relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-ymera-glow/5 to-transparent pointer-events-none" />
            
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <Eye className="w-6 h-6 text-ymera-glow" />
                Real-time Preview
              </h2>
              
              <div className="flex items-center gap-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setSoundEnabled(!soundEnabled)}
                  className="p-2 bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 text-slate-300 hover:text-white transition-colors"
                >
                  {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                </motion.button>
                
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleUndo}
                  disabled={historyIndex === 0}
                  className="p-2 bg-slate-800/50 backdrop-blur-sm rounded-lg border border-slate-700 text-slate-300 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <SkipBack className="w-4 h-4" />
                </motion.button>
              </div>
            </div>
            
            <EnhancedThreeScene 
              config={avatarConfig} 
              isAnimating={isAnimating}
              onAnimationToggle={() => setIsAnimating(!isAnimating)}
            />
          </motion.div>

          {/* Advanced Controls Panel */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="xl:col-span-1 space-y-6"
          >
            {/* Tab Navigation */}
            <div className="bg-slate-900/80 backdrop-blur-xl border border-ymera-glow/20 rounded-2xl shadow-xl p-6">
              <div className="flex gap-1 mb-6 p-1 bg-slate-800/50 rounded-xl">
                {[
                  { id: 'basic', name: 'Basic', icon: Settings },
                  { id: 'colors', name: 'Colors', icon: Palette },
                  { id: 'presets', name: 'Presets', icon: Sparkles }
                ].map(tab => (
                  <motion.button
                    key={tab.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg font-medium text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'bg-ymera-glow/20 text-ymera-glow border border-ymera-glow/30'
                        : 'text-slate-400 hover:text-slate-300'
                    }`}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.name}
                  </motion.button>
                ))}
              </div>

              {/* Tab Content */}
              <AnimatePresence mode="wait">
                {activeTab === 'basic' && (
                  <motion.div
                    key="basic"
                    variants={tabVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                    className="space-y-6"
                  >
                    {/* Shape Selector */}
                    <div>
                      <label className="block text-sm font-semibold text-slate-300 mb-3">
                        Form Factor
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        {AVATAR_CONFIG.shapes.map(shape => (
                          <motion.button
                            key={shape.id}
                            whileHover={{ scale: 1.03, y: -2 }}
                            whileTap={{ scale: 0.97 }}
                            onClick={() => handleConfigChange('shape', shape.id)}
                            className={`p-3 rounded-xl transition-all font-medium text-sm border flex items-center gap-2 ${
                              avatarConfig.shape === shape.id
                                ? 'bg-ymera-glow/20 border-ymera-glow text-ymera-glow shadow-lg'
                                : 'bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700/50 hover:border-slate-600'
                            }`}
                          >
                            <span className="text-lg">{shape.icon}</span>
                            {shape.name}
                          </motion.button>
                        ))}
                      </div>
                    </div>

                    {/* Glow Intensity */}
                    <div>
                      <label className="flex items-center justify-between text-sm font-semibold text-slate-300 mb-3">
                        <span>Energy Intensity</span>
                        <span className="text-ymera-glow font-mono">{avatarConfig.glowIntensity}</span>
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        step="1"
                        value={avatarConfig.glowIntensity}
                        onChange={(e) => handleConfigChange('glowIntensity', parseInt(e.target.value))}
                        className="w-full h-3 bg-slate-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                      />
                      <div className="flex justify-between text-xs text-slate-500 mt-1">
                        <span>Subtle</span>
                        <span>Maximum</span>
                      </div>
                    </div>
                  </motion.div>
                )}

                {activeTab === 'colors' && (
                  <motion.div
                    key="colors"
                    variants={tabVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                  >
                    <div>
                      <label className="block text-sm font-semibold text-slate-300 mb-3">
                        Color Palette
                      </label>
                      <ColorPaletteSelector
                        selectedPalette={selectedPalette}
                        onPaletteChange={setSelectedPalette}
                        selectedColor={avatarConfig.color}
                        onColorChange={(color) => handleConfigChange('color', color)}
                      />
                    </div>
                  </motion.div>
                )}

                {activeTab === 'presets' && (
                  <motion.div
                    key="presets"
                    variants={tabVariants}
                    initial="hidden"
                    animate="visible"
                    exit="exit"
                  >
                    <div>
                      <label className="block text-sm font-semibold text-slate-300 mb-3">
                        Quick Start Templates
                      </label>
                      <PresetSelector onPresetSelect={handlePresetSelect} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Action Buttons */}
              <div className="pt-6 border-t border-slate-700/50 space-y-3 mt-8">
                <motion.button
                  type="button"
                  onClick={handleCreateAvatar}
                  disabled={isSaving}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="w-full bg-gradient-to-r from-ymera-glow to-ymera-accent hover:from-ymera-glow/90 hover:to-ymera-accent/90 text-black font-bold py-4 px-6 rounded-xl shadow-lg shadow-ymera-glow/25 transition-all duration-300 flex items-center justify-center space-x-3 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  {isSaving ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Deploying Agent...</span>
                    </>
                  ) : (
                    <>
                      <Save className="w-5 h-5" />
                      <span>Deploy Agent Companion</span>
                    </>
                  )}
                </motion.button>
                
                <div className="grid grid-cols-2 gap-3">
                  <motion.button
                    type="button"
                    onClick={handleExportConfig}
                    disabled={isExporting}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="py-2 px-4 flex items-center justify-center space-x-2 bg-slate-800/70 text-slate-300 border border-slate-600 rounded-xl hover:bg-slate-700/70 hover:border-slate-500 transition-colors disabled:opacity-50"
                  >
                    {isExporting ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4" />
                    )}
                    <span className="text-sm">Export</span>
                  </motion.button>
                  
                  <motion.button
                    type="button"
                    onClick={() => {
                      setAvatarConfigState(AVATAR_CONFIG.default)
                      setSelectedPalette('nature')
                      toast.success('Configuration reset to defaults')
                    }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="py-2 px-4 flex items-center justify-center space-x-2 bg-slate-800/70 text-slate-300 border border-slate-600 rounded-xl hover:bg-slate-700/70 hover:border-slate-500 transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span className="text-sm">Reset</span>
                  </motion.button>
                </div>
              </div>
            </div>

            {/* Configuration Summary */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Code className="w-5 h-5 text-ymera-glow" />
                Configuration Summary
              </h3>
              
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Form Factor</span>
                  <span className="text-white font-medium capitalize">{avatarConfig.shape}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Core Energy</span>
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ 
                        backgroundColor: avatarConfig.color,
                        boxShadow: `0 0 6px ${avatarConfig.color}`
                      }}
                    />
                    <span className="text-white font-mono text-xs">{avatarConfig.color}</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Intensity Level</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-ymera-glow to-ymera-accent rounded-full transition-all duration-300"
                        style={{ width: `${(avatarConfig.glowIntensity / 10) * 100}%` }}
                      />
                    </div>
                    <span className="text-white font-medium">{avatarConfig.glowIntensity}/10</span>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Animation</span>
                  <span className="text-white font-medium">{isAnimating ? 'Active' : 'Paused'}</span>
                </div>
              </div>
            </motion.div>

            {/* Tips & Help */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-gradient-to-br from-ymera-glow/10 to-ymera-accent/5 border border-ymera-glow/30 rounded-2xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-ymera-glow" />
                Pro Tips
              </h3>
              
              <ul className="space-y-2 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full bg-ymera-glow mt-2 flex-shrink-0" />
                  <span>Hover over the preview to interact with your avatar in real-time</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full bg-ymera-glow mt-2 flex-shrink-0" />
                  <span>Higher intensity levels create more dramatic visual effects</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full bg-ymera-glow mt-2 flex-shrink-0" />
                  <span>Export your configuration to share with team members</span>
                </li>
                <li className="flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full bg-ymera-glow mt-2 flex-shrink-0" />
                  <span>Use presets as starting points for custom designs</span>
                </li>
              </ul>
            </motion.div>
          </motion.div>
        </div>

        {/* Background Effects */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-ymera-glow/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-ymera-accent/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        </div>

        {/* Custom Styles */}
        <style jsx>{`
          .slider-thumb::-webkit-slider-thumb {
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #64f4ac, #10b981);
            cursor: pointer;
            border: 2px solid #000;
            box-shadow: 0 0 10px rgba(100, 244, 172, 0.5);
          }
          
          .slider-thumb::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #64f4ac, #10b981);
            cursor: pointer;
            border: 2px solid #000;
            box-shadow: 0 0 10px rgba(100, 244, 172, 0.5);
          }
          
          .perspective-1000 {
            perspective: 1000px;
          }
          
          .text-glow-strong {
            text-shadow: 0 0 20px currentColor, 0 0 40px currentColor, 0 0 60px currentColor;
          }
        `}</style>
      </div>
    </>
  )
}