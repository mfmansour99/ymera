import React, { useState, useCallback, useRef, useEffect, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, Lock, Eye, EyeOff, Shield, Zap, Terminal, Brain,
  Fingerprint, Cpu, CheckCircle, AlertTriangle, Loader2,
  ArrowRight, Sparkles, Network, Database, Chrome, Github
} from 'lucide-react';

// Enhanced 3D Logo with Neural Network Visualization
const Enhanced3DLogo = ({ stage = 'idle' }) => {
  const meshRef = useRef();
  const particlesRef = useRef();
  
  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    
    const time = clock.getElapsedTime();
    
    switch (stage) {
      case 'authenticating':
        meshRef.current.rotation.y = time * 2;
        break;
      case 'scanning':
        meshRef.current.rotation.x = Math.sin(time * 3) * 0.3;
        meshRef.current.rotation.z = Math.cos(time * 2) * 0.2;
        break;
      case 'success':
        meshRef.current.scale.setScalar(1 + Math.sin(time * 4) * 0.1);
        break;
      default:
        meshRef.current.rotation.y = time * 0.5;
    }
    
    if (particlesRef.current) {
      particlesRef.current.children.forEach((particle, index) => {
        particle.position.y = Math.sin(time * 2 + index) * 0.3;
        particle.scale.setScalar(1 + Math.sin(time * 3 + index) * 0.2);
      });
    }
  });

  const neuralNodes = React.useMemo(() => 
    Array.from({ length: 8 }, (_, i) => ({
      position: [
        Math.cos((i / 8) * Math.PI * 2) * 1.5,
        Math.sin(i * 0.5) * 0.5,
        Math.sin((i / 8) * Math.PI * 2) * 1.5
      ],
      color: stage === 'success' ? '#10b981' : stage === 'scanning' ? '#f59e0b' : '#64f4ac'
    })), [stage]
  );

  return (
    <Float rotationIntensity={0.3} floatIntensity={0.2} speed={1}>
      <group ref={meshRef}>
        <mesh>
          <dodecahedronGeometry args={[1, 1]} />
          <meshPhysicalMaterial
            color={stage === 'success' ? '#10b981' : '#64f4ac'}
            emissive={stage === 'success' ? '#10b981' : '#64f4ac'}
            emissiveIntensity={stage === 'scanning' ? 0.8 : 0.4}
            metalness={0.8}
            roughness={0.2}
            transparent
            opacity={0.9}
            wireframe={stage === 'scanning'}
          />
        </mesh>
        
        <group ref={particlesRef}>
          {neuralNodes.map((node, index) => (
            <mesh key={index} position={node.position}>
              <sphereGeometry args={[0.08]} />
              <meshBasicMaterial 
                color={node.color}
                emissive={node.color}
                emissiveIntensity={0.6}
              />
            </mesh>
          ))}
        </group>
      </group>
    </Float>
  );
};

// Security Scanner Component
const SecurityScanner = ({ stage, progress = 0 }) => {
  const scanSteps = [
    'Initializing neural interface...',
    'Analyzing biometric patterns...',
    'Verifying neural signatures...',
    'Cross-referencing security database...',
    'Finalizing authentication...'
  ];
  
  const currentStep = Math.floor((progress / 100) * scanSteps.length);
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800/40 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 mb-6"
    >
      <div className="flex items-center space-x-4 mb-4">
        <motion.div
          animate={stage === 'scanning' ? { rotate: 360 } : {}}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className={`p-3 rounded-full ${
            stage === 'success' ? 'bg-green-600' : stage === 'scanning' ? 'bg-cyan-600' : 'bg-slate-600'
          }`}
        >
          {stage === 'success' ? (
            <CheckCircle className="w-6 h-6 text-white" />
          ) : stage === 'scanning' ? (
            <Cpu className="w-6 h-6 text-white" />
          ) : (
            <Shield className="w-6 h-6 text-white" />
          )}
        </motion.div>
        
        <div className="flex-1">
          <div className="text-white font-medium mb-1">
            {stage === 'success' ? 'Neural Authentication Complete' : 
             stage === 'scanning' ? 'Security Scan in Progress' : 'Ready for Authentication'}
          </div>
          <div className="text-slate-400 text-sm">
            {stage === 'success' ? 'All security checks passed' :
             stage === 'scanning' ? scanSteps[currentStep] || scanSteps[0] : 'Awaiting biometric verification'}
          </div>
        </div>
        
        {stage === 'success' && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="flex items-center space-x-2 text-green-400"
          >
            <Shield className="w-5 h-5" />
            <span className="text-sm font-mono">VERIFIED</span>
          </motion.div>
        )}
      </div>
      
      {stage === 'scanning' && (
        <div className="w-full bg-slate-700 rounded-full h-2 mb-3">
          <motion.div 
            className="bg-gradient-to-r from-cyan-400 to-green-400 h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      )}
      
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="p-2 bg-slate-900/50 rounded-lg">
          <div className={`text-sm font-mono ${stage === 'success' ? 'text-green-400' : 'text-cyan-400'}`}>
            {stage === 'success' ? '100%' : `${Math.min(progress, 100)}%`}
          </div>
          <div className="text-xs text-slate-500">Match Rate</div>
        </div>
        <div className="p-2 bg-slate-900/50 rounded-lg">
          <div className={`text-sm font-mono ${stage === 'success' ? 'text-green-400' : 'text-cyan-400'}`}>
            {stage === 'success' ? '<1ms' : stage === 'scanning' ? '~3s' : '--'}
          </div>
          <div className="text-xs text-slate-500">Response</div>
        </div>
        <div className="p-2 bg-slate-900/50 rounded-lg">
          <div className={`text-sm font-mono ${stage === 'success' ? 'text-green-400' : 'text-cyan-400'}`}>
            {stage === 'success' ? 'AAA+' : stage === 'scanning' ? 'AA+' : 'A+'}
          </div>
          <div className="text-xs text-slate-500">Security</div>
        </div>
      </div>
    </motion.div>
  );
};

// Floating Particles Background
const FloatingParticles = ({ intensity = 1 }) => {
  const particles = React.useMemo(() => 
    Array.from({ length: Math.floor(30 * intensity) }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 3,
      duration: 3 + Math.random() * 3,
      size: 0.5 + Math.random() * 1.5,
      color: ['#64f4ac', '#60a5fa', '#f59e0b'][Math.floor(Math.random() * 3)]
    })), [intensity]
  );

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {particles.map(particle => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            backgroundColor: `${particle.color}40`
          }}
          animate={{
            y: [-20, -120],
            opacity: [0, 0.8, 0],
            scale: [0, 1, 0]
          }}
          transition={{
            duration: particle.duration,
            delay: particle.delay,
            repeat: Infinity,
            ease: "easeOut"
          }}
        />
      ))}
    </div>
  );
};

// Enhanced Input Field with Validation
const EnhancedInputField = ({ 
  icon: Icon, 
  type = 'text', 
  placeholder, 
  value, 
  onChange, 
  error, 
  label,
  showPasswordToggle = false,
  onBlur,
  disabled = false
}) => {
  const [showPassword, setShowPassword] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  
  const inputType = type === 'password' && showPassword ? 'text' : type;
  
  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-slate-300 mb-2">{label}</label>
      <div className={`relative transition-all duration-300 ${
        isFocused ? 'transform scale-[1.01]' : ''
      }`}>
        <Icon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
        <input
          type={inputType}
          value={value}
          onChange={onChange}
          onFocus={() => setIsFocused(true)}
          onBlur={(e) => {
            setIsFocused(false);
            if (onBlur) onBlur(e);
          }}
          disabled={disabled}
          placeholder={placeholder}
          className={`w-full pl-10 pr-${showPasswordToggle ? '12' : '4'} py-3 bg-slate-900/50 border rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${
            error 
              ? 'border-red-500 focus:ring-red-500/50' 
              : isFocused
                ? 'border-cyan-400 focus:ring-cyan-400/50 shadow-lg shadow-cyan-400/10'
                : 'border-slate-600 hover:border-slate-500'
          }`}
        />
        {showPasswordToggle && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            disabled={disabled}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-white transition-colors disabled:opacity-50"
          >
            {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          </button>
        )}
      </div>
      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="mt-2 text-sm text-red-400 flex items-center space-x-1"
          >
            <AlertTriangle className="w-4 h-4" />
            <span>{error}</span>
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
};

// Social Login Button
const SocialLoginButton = ({ icon: Icon, provider, onClick, disabled = false }) => (
  <motion.button
    type="button"
    onClick={onClick}
    disabled={disabled}
    whileHover={{ scale: disabled ? 1 : 1.02 }}
    whileTap={{ scale: disabled ? 1 : 0.98 }}
    className={`flex items-center justify-center space-x-3 w-full p-3 rounded-xl border transition-all duration-300 ${
      disabled 
        ? 'bg-slate-800/30 border-slate-700/30 text-slate-500 cursor-not-allowed'
        : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-700/50 hover:border-slate-600/50 text-slate-300 hover:text-white'
    }`}
  >
    <Icon size={20} />
    <span>Continue with {provider}</span>
  </motion.button>
);

// Main Advanced Login Component
const IntegratedAdvancedLogin = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [loginStep, setLoginStep] = useState('credentials');
  const [isLoading, setIsLoading] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanStage, setScanStage] = useState('idle');
  const [generalError, setGeneralError] = useState('');

  // Real-time validation
  const validateField = useCallback((field, value) => {
    const newErrors = {};
    
    if (field === 'username') {
      if (!value.trim()) {
        newErrors.username = 'Username is required';
      } else if (value.length < 3) {
        newErrors.username = 'Username must be at least 3 characters';
      } else if (!/^[a-zA-Z0-9_]+$/.test(value)) {
        newErrors.username = 'Username can only contain letters, numbers, and underscores';
      }
    }
    
    if (field === 'password') {
      if (!value.trim()) {
        newErrors.password = 'Password is required';
      } else if (value.length < 8) {
        newErrors.password = 'Password must be at least 8 characters';
      } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(value)) {
        newErrors.password = 'Password must contain uppercase, lowercase, and numbers';
      }
    }
    
    return newErrors;
  }, []);

  const validateForm = useCallback(() => {
    const usernameErrors = validateField('username', formData.username);
    const passwordErrors = validateField('password', formData.password);
    const newErrors = { ...usernameErrors, ...passwordErrors };
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData, validateField]);

  // Biometric scan effect
  useEffect(() => {
    if (loginStep === 'biometric') {
      setScanStage('scanning');
      setScanProgress(0);
      
      const interval = setInterval(() => {
        setScanProgress(prev => {
          const newProgress = prev + Math.random() * 15 + 5;
          if (newProgress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
              setScanStage('success');
              setTimeout(() => setLoginStep('success'), 1500);
            }, 500);
            return 100;
          }
          return newProgress;
        });
      }, 200);
      
      return () => clearInterval(interval);
    }
  }, [loginStep]);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    setGeneralError('');
    
    setTouched({ username: true, password: true });
    
    if (!validateForm()) return;
    
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      if (Math.random() < 0.1) {
        throw new Error('Invalid credentials. Please try again.');
      }
      
      setIsLoading(false);
      setLoginStep('biometric');
    } catch (error) {
      setIsLoading(false);
      setGeneralError(error.message);
    }
  }, [formData, validateForm]);

  const handleInputChange = useCallback((field) => (e) => {
    const value = e.target.value;
    setFormData(prev => ({ ...prev, [field]: value }));
    setGeneralError('');
    
    if (touched[field]) {
      const fieldErrors = validateField(field, value);
      setErrors(prev => ({
        ...prev,
        ...fieldErrors,
        [field]: fieldErrors[field] || undefined
      }));
    }
  }, [touched, validateField]);

  const handleBlur = useCallback((field) => () => {
    setTouched(prev => ({ ...prev, [field]: true }));
    const fieldErrors = validateField(field, formData[field]);
    setErrors(prev => ({ ...prev, ...fieldErrors }));
  }, [formData, validateField]);

  const handleSocialLogin = useCallback((provider) => {
    console.log(`Social login with ${provider}`);
    setGeneralError('Social login feature coming soon!');
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden flex items-center justify-center">
      <FloatingParticles intensity={loginStep === 'success' ? 2 : 1} />
      
      <div className="absolute inset-0 opacity-5" 
           style={{
             backgroundImage: `
               linear-gradient(rgba(100, 244, 172, 0.3) 1px, transparent 1px),
               linear-gradient(90deg, rgba(100, 244, 172, 0.3) 1px, transparent 1px)
             `,
             backgroundSize: '60px 60px'
           }} 
      />

      <AnimatePresence mode="wait">
        {loginStep === 'credentials' && (
          <motion.div
            key="credentials"
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -50, scale: 1.1 }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
            className="relative z-10 w-full max-w-md p-8"
          >
            <div className="h-32 mb-8">
              <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
                <ambientLight intensity={0.4} />
                <pointLight position={[10, 10, 10]} intensity={1} color="#64f4ac" />
                <Environment preset="night" />
                <Suspense fallback={null}>
                  <Enhanced3DLogo stage={isLoading ? 'authenticating' : 'idle'} />
                </Suspense>
                <OrbitControls enableZoom={false} enablePan={false} />
              </Canvas>
            </div>
            
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-center mb-8"
            >
              <h1 className="text-4xl font-bold text-white mb-3">
                Neural Access Portal
              </h1>
              <p className="text-slate-400 text-lg">
                Advanced AI Project Management System
              </p>
            </motion.div>

            <motion.form
              onSubmit={handleSubmit}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-slate-800/30 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-8 shadow-2xl"
              style={{ boxShadow: '0 0 60px rgba(100, 244, 172, 0.1)' }}
            >
              {generalError && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-4 p-3 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center space-x-2 text-red-400"
                >
                  <AlertTriangle className="w-5 h-5" />
                  <span className="text-sm">{generalError}</span>
                </motion.div>
              )}

              <EnhancedInputField
                icon={User}
                label="Username"
                type="text"
                placeholder="Enter your username"
                value={formData.username}
                onChange={handleInputChange('username')}
                onBlur={handleBlur('username')}
                error={touched.username ? errors.username : ''}
                disabled={isLoading}
              />

              <EnhancedInputField
                icon={Lock}
                label="Password"
                type="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleInputChange('password')}
                onBlur={handleBlur('password')}
                error={touched.password ? errors.password : ''}
                showPasswordToggle
                disabled={isLoading}
              />

              <motion.button
                type="submit"
                disabled={isLoading || Object.keys(errors).length > 0}
                whileHover={{ scale: isLoading ? 1 : 1.02 }}
                whileTap={{ scale: isLoading ? 1 : 0.98 }}
                className={`w-full py-4 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center space-x-3 ${
                  isLoading || Object.keys(errors).length > 0
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-cyan-400 to-green-400 text-slate-900 shadow-lg hover:shadow-xl'
                }`}
                style={{ 
                  boxShadow: isLoading ? 'none' : '0 0 40px rgba(100, 244, 172, 0.4)' 
                }}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Initializing Neural Interface...</span>
                  </>
                ) : (
                  <>
                    <Terminal className="w-5 h-5" />
                    <span>Access Neural Grid</span>
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </motion.button>

              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-slate-700/50" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-slate-800/50 text-slate-400">Or continue with</span>
                </div>
              </div>

              <div className="space-y-3">
                <SocialLoginButton
                  icon={Chrome}
                  provider="Google"
                  onClick={() => handleSocialLogin('google')}
                  disabled={isLoading}
                />
                <SocialLoginButton
                  icon={Github}
                  provider="GitHub"
                  onClick={() => handleSocialLogin('github')}
                  disabled={isLoading}
                />
              </div>

              <div className="mt-6 flex items-center justify-between text-sm">
                <button
                  type="button"
                  disabled={isLoading}
                  className="text-slate-400 hover:text-cyan-400 transition-colors disabled:cursor-not-allowed"
                >
                  Forgot credentials?
                </button>
                <button
                  type="button"
                  disabled={isLoading}
                  className="text-slate-400 hover:text-cyan-400 transition-colors flex items-center space-x-1 disabled:cursor-not-allowed"
                >
                  <Fingerprint className="w-4 h-4" />
                  <span>Direct biometric</span>
                </button>
              </div>
            </motion.form>
          </motion.div>
        )}

        {loginStep === 'biometric' && (
          <motion.div
            key="biometric"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.1 }}
            transition={{ duration: 0.5 }}
            className="relative z-10 w-full max-w-lg p-8"
          >
            <div className="h-32 mb-8">
              <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
                <ambientLight intensity={0.4} />
                <pointLight position={[10, 10, 10]} intensity={1} color="#64f4ac" />
                <Environment preset="night" />
                <Suspense fallback={null}>
                  <Enhanced3DLogo stage={scanStage} />
                </Suspense>
                <OrbitControls enableZoom={false} enablePan={false} />
              </Canvas>
            </div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center mb-8"
            >
              <h2 className="text-3xl font-bold text-white mb-3">
                {scanStage === 'success' ? 'Neural Patterns Verified' : 'Biometric Authentication'}
              </h2>
              <p className="text-slate-400 text-lg">
                {scanStage === 'success' 
                  ? 'Access granted to neural command interface' 
                  : 'Please remain still during neural pattern analysis'
                }
              </p>
            </motion.div>

            <SecurityScanner stage={scanStage} progress={scanProgress} />

            {scanStage === 'success' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center"
              >
                <div className="flex items-center justify-center space-x-3 text-green-400 text-lg font-semibold">
                  <Sparkles className="w-6 h-6" />
                  <span>Authentication Complete</span>
                  <Sparkles className="w-6 h-6" />
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {loginStep === 'success' && (
          <motion.div
            key="success"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.2 }}
            className="relative z-10 w-full max-w-md p-8 text-center"
          >
            <motion.div
              animate={{ 
                boxShadow: [
                  '0 0 60px rgba(100, 244, 172, 0.6)',
                  '0 0 120px rgba(100, 244, 172, 1)',
                  '0 0 60px rgba(100, 244, 172, 0.6)'
                ]
              }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-40 h-40 mx-auto mb-8 bg-gradient-to-br from-cyan-400 via-green-400 to-emerald-500 rounded-full flex items-center justify-center relative"
            >
              <CheckCircle className="w-20 h-20 text-slate-900" />
              
              {Array.from({ length: 8 }).map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-2 h-2 bg-white rounded-full"
                  initial={{ 
                    scale: 0,
                    x: 0,
                    y: 0
                  }}
                  animate={{
                    scale: [0, 1, 0],
                    x: Math.cos((i / 8) * Math.PI * 2) * 60,
                    y: Math.sin((i / 8) * Math.PI * 2) * 60
                  }}
                  transition={{
                    duration: 2,
                    delay: i * 0.1,
                    repeat: Infinity,
                    repeatDelay: 2
                  }}
                />
              ))}
            </motion.div>
            
            <h2 className="text-4xl font-bold text-white mb-4">
              Welcome to the Neural Grid
            </h2>
            <p className="text-slate-400 text-xl mb-8">
              Initializing your AI command center...
            </p>
            
            <motion.div
              animate={{ 
                opacity: [0.5, 1, 0.5],
                scale: [1, 1.05, 1]
              }}
              transition={{ duration: 2, repeat: Infinity }}
              className="text-cyan-400 font-mono text-lg"
            >
              &gt;&gt;&gt; Loading neural interface...
            </motion.div>

            <div className="mt-8 grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <Brain className="w-6 h-6 text-green-400 mx-auto mb-1" />
                <div className="text-green-400 font-mono">AI ONLINE</div>
              </div>
              <div className="text-center">
                <Database className="w-6 h-6 text-green-400 mx-auto mb-1" />
                <div className="text-green-400 font-mono">DB READY</div>
              </div>
              <div className="text-center">
                <Network className="w-6 h-6 text-green-400 mx-auto mb-1" />
                <div className="text-green-400 font-mono">NET SYNC</div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default IntegratedAdvancedLogin;