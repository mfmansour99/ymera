import React, { useState, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Terminal, User, Lock, Mail, Loader2, Eye, EyeOff, CheckCircle, XCircle, Shield, Sparkles, ArrowRight, ArrowLeft } from 'lucide-react'

// Mock auth hook for demonstration
const useAuth = () => ({
  register: async (name, email, password) => {
    await new Promise(resolve => setTimeout(resolve, 2000))
    if (email === 'test@error.com') throw new Error('Email already exists')
    return { success: true }
  },
  isLoading: false
})

// Password strength validator
const validatePassword = (password) => {
  const checks = {
    length: password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /\d/.test(password),
    special: /[!@#$%^&*(),.?":{}|<>]/.test(password)
  }
  
  const score = Object.values(checks).filter(Boolean).length
  const strength = score < 2 ? 'weak' : score < 4 ? 'medium' : 'strong'
  
  return { checks, score, strength }
}

// Email validation
const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

// Floating particles background component
const FloatingParticles = () => {
  const particles = useMemo(() => 
    Array.from({ length: 12 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 2,
      duration: Math.random() * 10 + 10
    }))
  , [])

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full bg-gradient-to-r from-green-400/20 to-cyan-400/20 blur-sm"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
          }}
          animate={{
            y: [0, -30, 0],
            x: [0, 15, 0],
            opacity: [0.3, 0.8, 0.3],
            scale: [1, 1.2, 1]
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      ))}
    </div>
  )
}

// Enhanced input component with validation
const EnhancedInput = ({ 
  icon: Icon, 
  type, 
  name, 
  placeholder, 
  value, 
  onChange, 
  error, 
  success,
  showPasswordToggle = false,
  ...props 
}) => {
  const [showPassword, setShowPassword] = useState(false)
  const [isFocused, setIsFocused] = useState(false)

  const inputType = showPasswordToggle && showPassword ? 'text' : type

  return (
    <motion.div 
      className="relative"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className={`relative transition-all duration-300 ${
        isFocused ? 'transform scale-[1.02]' : ''
      }`}>
        <Icon className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 transition-colors duration-200 ${
          error ? 'text-red-400' : success ? 'text-green-400' : isFocused ? 'text-green-400' : 'text-slate-400'
        }`} />
        
        <input
          type={inputType}
          name={name}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          className={`w-full pl-10 ${showPasswordToggle ? 'pr-12' : 'pr-4'} py-3 bg-slate-800/60 border rounded-xl 
            text-slate-100 placeholder-slate-400 backdrop-blur-sm transition-all duration-200
            focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900
            ${error 
              ? 'border-red-500 focus:ring-red-500/50' 
              : success 
                ? 'border-green-500 focus:ring-green-500/50'
                : 'border-slate-600 focus:border-green-400 focus:ring-green-400/50 hover:border-slate-500'
            }`}
          {...props}
        />
        
        {showPasswordToggle && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
          >
            {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          </button>
        )}
        
        {(error || success) && !showPasswordToggle && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            {error ? (
              <XCircle className="w-5 h-5 text-red-400" />
            ) : (
              <CheckCircle className="w-5 h-5 text-green-400" />
            )}
          </div>
        )}
      </div>
      
      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="text-red-400 text-sm mt-1 ml-1"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// Password strength indicator
const PasswordStrength = ({ password }) => {
  const validation = validatePassword(password)
  
  if (!password) return null
  
  const strengthColors = {
    weak: 'bg-red-500',
    medium: 'bg-yellow-500',
    strong: 'bg-green-500'
  }
  
  const strengthLabels = {
    weak: 'Weak',
    medium: 'Medium', 
    strong: 'Strong'
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="mt-3 space-y-2"
    >
      <div className="flex items-center space-x-2">
        <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
          <motion.div
            className={`h-full ${strengthColors[validation.strength]} transition-all duration-300`}
            initial={{ width: 0 }}
            animate={{ width: `${(validation.score / 5) * 100}%` }}
          />
        </div>
        <span className={`text-xs font-medium ${
          validation.strength === 'weak' ? 'text-red-400' :
          validation.strength === 'medium' ? 'text-yellow-400' : 'text-green-400'
        }`}>
          {strengthLabels[validation.strength]}
        </span>
      </div>
      
      <div className="grid grid-cols-2 gap-1 text-xs">
        {Object.entries(validation.checks).map(([key, passed]) => (
          <div key={key} className={`flex items-center space-x-1 ${passed ? 'text-green-400' : 'text-slate-500'}`}>
            {passed ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
            <span className="capitalize">
              {key === 'length' ? '8+ chars' : 
               key === 'uppercase' ? 'Uppercase' :
               key === 'lowercase' ? 'Lowercase' :
               key === 'number' ? 'Number' : 'Special char'}
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

// Success animation component
const SuccessAnimation = () => (
  <motion.div
    initial={{ scale: 0, rotate: -180 }}
    animate={{ scale: 1, rotate: 0 }}
    transition={{ type: "spring", damping: 15, stiffness: 300 }}
    className="flex flex-col items-center space-y-4"
  >
    <div className="relative">
      <motion.div
        className="w-24 h-24 rounded-full bg-gradient-to-r from-green-400 to-cyan-400 flex items-center justify-center"
        animate={{ 
          boxShadow: [
            "0 0 20px rgba(34, 197, 94, 0.3)",
            "0 0 40px rgba(34, 197, 94, 0.6)",
            "0 0 20px rgba(34, 197, 94, 0.3)"
          ]
        }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <CheckCircle className="w-12 h-12 text-white" />
      </motion.div>
      <motion.div
        className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-yellow-400 flex items-center justify-center"
        animate={{ rotate: 360 }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
      >
        <Sparkles className="w-4 h-4 text-yellow-800" />
      </motion.div>
    </div>
    
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
      className="text-center"
    >
      <h3 className="text-xl font-bold text-green-400 mb-2">Account Created! 🎉</h3>
      <p className="text-slate-300">Redirecting to Avatar Creator...</p>
    </motion.div>
  </motion.div>
)

export default function Register() {
  const { register, isLoading } = useAuth()
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  })
  
  const [validationErrors, setValidationErrors] = useState({})
  const [touchedFields, setTouchedFields] = useState({})
  const [step, setStep] = useState(0)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showSuccess, setShowSuccess] = useState(false)

  // Real-time validation
  const validation = useMemo(() => {
    const errors = {}
    const successes = {}
    
    if (touchedFields.name) {
      if (!formData.name.trim()) {
        errors.name = 'Name is required'
      } else if (formData.name.trim().length < 2) {
        errors.name = 'Name must be at least 2 characters'
      } else {
        successes.name = true
      }
    }
    
    if (touchedFields.email) {
      if (!formData.email) {
        errors.email = 'Email is required'
      } else if (!validateEmail(formData.email)) {
        errors.email = 'Please enter a valid email address'
      } else {
        successes.email = true
      }
    }
    
    if (touchedFields.password) {
      const passwordValidation = validatePassword(formData.password)
      if (passwordValidation.score < 3) {
        errors.password = 'Password is too weak'
      } else {
        successes.password = true
      }
    }
    
    if (touchedFields.confirmPassword) {
      if (formData.confirmPassword !== formData.password) {
        errors.confirmPassword = 'Passwords do not match'
      } else if (formData.confirmPassword) {
        successes.confirmPassword = true
      }
    }
    
    return { errors, successes }
  }, [formData, touchedFields])

  const handleChange = useCallback((e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
    setTouchedFields(prev => ({ ...prev, [name]: true }))
    
    // Clear validation errors on change
    if (validationErrors[name]) {
      setValidationErrors(prev => ({ ...prev, [name]: '' }))
    }
  }, [validationErrors])

  const handleSubmit = useCallback(async () => {
    setIsSubmitting(true)
    
    // Mark all fields as touched
    setTouchedFields({
      name: true,
      email: true,
      password: true,
      confirmPassword: true
    })
    
    // Check for validation errors
    if (Object.keys(validation.errors).length > 0) {
      setValidationErrors(validation.errors)
      setIsSubmitting(false)
      return
    }

    try {
      await register(formData.name, formData.email, formData.password)
      setShowSuccess(true)
      
      // Simulate redirect after success
      setTimeout(() => {
        alert('Redirecting to Avatar Creator...')
      }, 2000)
      
    } catch (err) {
      setValidationErrors({ submit: err.message || 'Registration failed. Please try again.' })
    } finally {
      setIsSubmitting(false)
    }
  }, [formData, validation.errors, register])

  const formSteps = [
    { title: 'Personal Info', fields: ['name', 'email'] },
    { title: 'Security', fields: ['password', 'confirmPassword'] },
    { title: 'Confirm', fields: [] }
  ]

  const canProceed = useMemo(() => {
    const currentFields = formSteps[step].fields
    return currentFields.every(field => 
      formData[field] && !validation.errors[field]
    )
  }, [step, formData, validation.errors])

  const containerVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: { 
        duration: 0.6,
        type: "spring",
        damping: 25,
        stiffness: 200
      }
    }
  }

  const stepVariants = {
    hidden: { opacity: 0, x: 100 },
    visible: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -100 }
  }

  if (showSuccess) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-900 via-black to-slate-900">
        <SuccessAnimation />
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-gradient-to-br from-slate-900 via-black to-slate-900 relative overflow-hidden">
      <FloatingParticles />
      
      {/* Animated background elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-green-400/20 blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-72 h-72 rounded-full bg-cyan-500/20 blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      </div>
      
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="relative z-10 w-full max-w-lg"
      >
        {/* Progress indicator */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {formSteps.map((stepInfo, index) => (
              <div key={index} className="flex items-center">
                <motion.div
                  className={`w-10 h-10 rounded-full border-2 flex items-center justify-center font-medium
                    ${index <= step 
                      ? 'bg-green-400 text-black border-green-400' 
                      : 'border-slate-600 text-slate-400'
                    }`}
                  animate={{ 
                    scale: index === step ? 1.1 : 1,
                    boxShadow: index === step ? '0 0 20px rgba(34, 197, 94, 0.5)' : '0 0 0px transparent'
                  }}
                >
                  {index < step ? <CheckCircle className="w-5 h-5" /> : index + 1}
                </motion.div>
                {index < formSteps.length - 1 && (
                  <div className={`w-12 h-0.5 mx-2 ${index < step ? 'bg-green-400' : 'bg-slate-600'}`} />
                )}
              </div>
            ))}
          </div>
          <h3 className="text-center text-green-400 font-medium">{formSteps[step].title}</h3>
        </div>

        <div className="bg-slate-900/80 border border-green-400/20 rounded-2xl shadow-2xl backdrop-blur-xl p-8 space-y-6">
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="relative inline-block">
              <Terminal className="w-12 h-12 mx-auto text-green-400" />
              <motion.div
                className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-gradient-to-r from-green-400 to-cyan-400"
                animate={{ rotate: 360 }}
                transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              >
                <Sparkles className="w-4 h-4 text-black m-1" />
              </motion.div>
            </div>
            
            <h1 className="mt-4 text-3xl font-bold text-slate-50">
              Create Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-cyan-400">Nexus</span> Account
            </h1>
            <p className="mt-2 text-slate-400">
              Join the next generation AI platform
            </p>
          </motion.div>

          <div className="space-y-6">
            <AnimatePresence mode="wait">
              <motion.div
                key={step}
                variants={stepVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
                className="space-y-4"
              >
                {step === 0 && (
                  <>
                    <EnhancedInput
                      icon={User}
                      type="text"
                      name="name"
                      placeholder="Full Name"
                      value={formData.name}
                      onChange={handleChange}
                      error={validation.errors.name}
                      success={validation.successes.name}
                      required
                    />
                    
                    <EnhancedInput
                      icon={Mail}
                      type="email"
                      name="email"
                      placeholder="Email Address"
                      value={formData.email}
                      onChange={handleChange}
                      error={validation.errors.email}
                      success={validation.successes.email}
                      required
                    />
                  </>
                )}
                
                {step === 1 && (
                  <>
                    <div>
                      <EnhancedInput
                        icon={Lock}
                        type="password"
                        name="password"
                        placeholder="Password"
                        value={formData.password}
                        onChange={handleChange}
                        error={validation.errors.password}
                        success={validation.successes.password}
                        showPasswordToggle={true}
                        required
                      />
                      <PasswordStrength password={formData.password} />
                    </div>
                    
                    <EnhancedInput
                      icon={Shield}
                      type="password"
                      name="confirmPassword"
                      placeholder="Confirm Password"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      error={validation.errors.confirmPassword}
                      success={validation.successes.confirmPassword}
                      showPasswordToggle={true}
                      required
                    />
                  </>
                )}
                
                {step === 2 && (
                  <div className="text-center space-y-4">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="w-20 h-20 mx-auto rounded-full bg-gradient-to-r from-green-400/20 to-cyan-400/20 flex items-center justify-center"
                    >
                      <CheckCircle className="w-10 h-10 text-green-400" />
                    </motion.div>
                    
                    <div className="space-y-2 text-slate-300">
                      <p><span className="text-slate-400">Name:</span> {formData.name}</p>
                      <p><span className="text-slate-400">Email:</span> {formData.email}</p>
                      <p className="text-sm text-slate-500">Ready to create your Agent Companion</p>
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>

            {/* Error message */}
            <AnimatePresence>
              {validationErrors.submit && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="bg-red-500/10 border border-red-500/20 rounded-lg p-3"
                >
                  <p className="text-red-400 text-sm text-center">
                    {validationErrors.submit}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Navigation buttons */}
            <div className="flex space-x-4">
              {step > 0 && (
                <motion.button
                  type="button"
                  onClick={() => setStep(step - 1)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex-1 py-3 bg-slate-800/60 text-slate-300 border border-slate-600 rounded-xl hover:bg-slate-700/60 transition-colors flex items-center justify-center space-x-2"
                >
                  <ArrowLeft className="w-4 h-4" />
                  <span>Previous</span>
                </motion.button>
              )}
              
              {step < formSteps.length - 1 ? (
                <motion.button
                  type="button"
                  onClick={() => setStep(step + 1)}
                  disabled={!canProceed}
                  whileHover={{ scale: canProceed ? 1.02 : 1 }}
                  whileTap={{ scale: canProceed ? 0.98 : 1 }}
                  className={`flex-1 py-3 rounded-xl font-medium transition-all duration-200 flex items-center justify-center space-x-2 ${
                    canProceed
                      ? 'bg-gradient-to-r from-green-400 to-cyan-400 text-black hover:shadow-lg hover:shadow-green-400/25'
                      : 'bg-slate-700/50 text-slate-500 cursor-not-allowed'
                  }`}
                >
                  <span>Next</span>
                  <ArrowRight className="w-4 h-4" />
                </motion.button>
              ) : (
                <motion.button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isSubmitting || !canProceed}
                  whileHover={{ scale: canProceed && !isSubmitting ? 1.02 : 1 }}
                  whileTap={{ scale: canProceed && !isSubmitting ? 0.98 : 1 }}
                  className="flex-1 py-3 bg-gradient-to-r from-green-400 to-cyan-400 text-black rounded-xl font-medium 
                    disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-green-400/25 
                    transition-all duration-200 flex items-center justify-center space-x-2"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Creating Account...</span>
                    </>
                  ) : (
                    <>
                      <Terminal className="w-5 h-5" />
                      <span>Create Account</span>
                    </>
                  )}
                </motion.button>
              )}
            </div>
          </div>

          <div className="text-center text-sm">
            <p className="text-slate-400">
              Already have an account?{' '}
              <button 
                onClick={() => alert('Navigate to login page')}
                className="text-green-400 hover:text-cyan-400 transition-colors font-medium hover:underline"
              >
                Sign In
              </button>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}