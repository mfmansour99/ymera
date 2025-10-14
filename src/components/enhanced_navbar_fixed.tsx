import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu, X, User, Settings, LogOut, Bell, Search, 
  Shield, Activity, Zap, Brain
} from 'lucide-react';

// Simple StatusAvatar Component
const StatusAvatar = ({ size = 'md', className = '' }) => {
  const sizes = { sm: 'w-8 h-8', md: 'w-10 h-10', lg: 'w-12 h-12' };
  
  return (
    <div className={`${sizes[size]} relative ${className}`}>
      <img
        src="https://ui-avatars.com/api/?name=User&background=64f4ac&color=000000"
        alt="Avatar"
        className="w-full h-full rounded-full object-cover"
      />
      <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-400 rounded-full ring-2 ring-black" />
    </div>
  );
};

// Notification Indicator
const NotificationIndicator = ({ count }) => {
  const [hasNew, setHasNew] = useState(false);
  
  useEffect(() => {
    if (count > 0) {
      setHasNew(true);
      const timer = setTimeout(() => setHasNew(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [count]);

  if (count === 0) return null;

  return (
    <motion.div
      className="relative"
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      <div className={`absolute -top-1 -right-1 rounded-full text-xs font-bold text-white flex items-center justify-center min-w-[18px] h-[18px] px-1 ${
        hasNew ? 'bg-red-500 animate-pulse shadow-lg shadow-red-500/50' : 'bg-red-500'
      }`}>
        {count > 99 ? '99+' : count}
      </div>
      
      {hasNew && (
        <motion.div
          className="absolute -top-1 -right-1 w-[18px] h-[18px] bg-red-500/30 rounded-full"
          animate={{ 
            scale: [1, 1.8, 1],
            opacity: [0.7, 0, 0.7]
          }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
        />
      )}
    </motion.div>
  );
};

// Smart Search Component
const SmartSearch = ({ isExpanded, onToggle }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim()) {
      console.log('Search:', query);
      setQuery('');
      onToggle();
    }
  };

  useEffect(() => {
    if (query.length > 2) {
      const mockSuggestions = [
        'Optimize dashboard layout',
        'Agent performance metrics',
        'User behavior analytics',
        'Accessibility audit results'
      ].filter(s => s.toLowerCase().includes(query.toLowerCase()));
      
      setSuggestions(mockSuggestions);
    } else {
      setSuggestions([]);
    }
  }, [query]);

  return (
    <div className="relative">
      <motion.button
        type="button"
        onClick={onToggle}
        className="p-2 rounded-full text-slate-400 hover:text-white hover:bg-slate-800/50 transition-all duration-200"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        aria-label="Smart Search"
      >
        <Search className="w-5 h-5" />
        <motion.div
          className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full opacity-60"
          animate={{ scale: [1, 1.2, 1], opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      </motion.button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            className="absolute top-full mt-2 right-0 w-96 bg-slate-900/95 backdrop-blur-xl border border-green-400/20 rounded-xl shadow-2xl z-50"
          >
            <form onSubmit={handleSearch} className="p-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search with AI assistance..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-slate-600 rounded-lg text-slate-100 placeholder-slate-400 focus:outline-none focus:border-green-400/50 focus:ring-1 focus:ring-green-400/25 transition-all"
                  autoFocus
                />
                <Brain className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-green-400/50" />
              </div>

              {suggestions.length > 0 && (
                <motion.div
                  className="mt-3 space-y-1"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <p className="text-xs text-slate-400 mb-2">AI Suggestions:</p>
                  {suggestions.map((suggestion, index) => (
                    <motion.button
                      key={suggestion}
                      type="button"
                      onClick={() => {
                        setQuery(suggestion);
                        handleSearch({ preventDefault: () => {} });
                      }}
                      className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:text-white hover:bg-slate-800/50 rounded-lg transition-colors"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <Zap className="inline w-3 h-3 mr-2 text-green-400" />
                      {suggestion}
                    </motion.button>
                  ))}
                </motion.div>
              )}
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// Navigation items
const navigation = [
  { name: 'Dashboard', href: '/', icon: Activity },
  { name: 'Agents', href: '/agents', icon: Brain },
  { name: 'Analytics', href: '/analytics', icon: Activity },
  { name: 'Settings', href: '/settings', icon: Settings }
];

export default function NavBar() {
  const [currentPath, setCurrentPath] = useState('/');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [notificationCount, setNotificationCount] = useState(3);

  // Mock user data
  const user = {
    name: 'Alex Chen',
    email: 'alex@ymera.com',
    avatar_url: 'https://ui-avatars.com/api/?name=Alex+Chen&background=64f4ac&color=000000',
    role: 'admin'
  };

  const isAuthenticated = true;
  const agentStatus = 'idle'; // idle, running, processing, alert, error

  // Scroll detection
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close menus on route change
  useEffect(() => {
    setIsMobileMenuOpen(false);
    setIsUserMenuOpen(false);
    setIsSearchExpanded(false);
  }, [currentPath]);

  // Handle navigation
  const handleNavigate = useCallback((href) => {
    setCurrentPath(href);
  }, []);

  const handleLogout = useCallback(() => {
    console.log('Logout');
    setIsUserMenuOpen(false);
  }, []);

  // Close menus on outside click
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isUserMenuOpen && !event.target.closest('#user-menu-button') && !event.target.closest('[role="menu"]')) {
        setIsUserMenuOpen(false);
      }
      if (isSearchExpanded && !event.target.closest('.search-container')) {
        setIsSearchExpanded(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isUserMenuOpen, isSearchExpanded]);

  return (
    <>
      <motion.nav 
        className={`fixed top-0 left-0 right-0 z-40 transition-all duration-300 ${
          isScrolled 
            ? 'backdrop-blur-xl bg-black/80 shadow-2xl border-b border-green-400/20' 
            : 'backdrop-blur-md bg-black/60 shadow-lg border-b border-green-400/10'
        }`}
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            
            {/* Logo and Navigation */}
            <div className="flex items-center space-x-8">
              <button 
                onClick={() => handleNavigate('/')}
                className="flex items-center space-x-2 text-2xl font-bold text-green-400 hover:text-white transition-colors"
              >
                <motion.div
                  className="w-8 h-8 bg-gradient-to-br from-green-400 to-cyan-400 rounded-lg flex items-center justify-center"
                  whileHover={{ rotate: 360, scale: 1.1 }}
                  transition={{ duration: 0.6 }}
                >
                  <Brain className="w-5 h-5 text-black" />
                </motion.div>
                <span>Ymera</span>
              </button>

              {/* Desktop Navigation */}
              <div className="hidden md:flex items-center space-x-1">
                {navigation.map((item) => (
                  <button
                    key={item.name}
                    onClick={() => handleNavigate(item.href)}
                    className={`relative px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                      currentPath === item.href
                        ? 'text-green-400 bg-green-400/10 shadow-lg'
                        : 'text-slate-300 hover:text-green-400 hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      <item.icon size={16} />
                      <span>{item.name}</span>
                    </div>
                    
                    {currentPath === item.href && (
                      <motion.div
                        className="absolute inset-0 bg-