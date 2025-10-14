import React from 'react';
import { User } from '../types';

interface NavigationProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  user: User;
}

export const Navigation: React.FC<NavigationProps> = ({ currentPage, onNavigate, user }) => {
  const navItems = [
    { id: 'agents', label: 'Agents', icon: 'fas fa-robot' },
    { id: 'projects', label: 'Projects', icon: 'fas fa-project-diagram' },
    { id: 'history', label: 'History', icon: 'fas fa-history' },
    { id: 'profile', label: 'Profile', icon: 'fas fa-user' },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-primary-bg bg-opacity-90 backdrop-blur-md border-b border-glass-border">
      <div className="container">
        <div className="flex items-center justify-between h-20">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary flex items-center justify-center">
                <i className="fas fa-atom text-white" />
              </div>
              <span className="text-xl font-bold text-gradient">AgentFlow</span>
            </div>

            <div className="hidden md:flex space-x-1">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`px-4 py-2 rounded-lg transition-all flex items-center space-x-2 ${
                    currentPage === item.id
                      ? 'bg-accent-primary bg-opacity-20 text-accent-primary'
                      : 'text-secondary hover:text-primary hover:bg-glass'
                  }`}
                >
                  <i className={item.icon} />
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button
              onClick={() => onNavigate('settings')}
              className="w-10 h-10 rounded-full glass flex items-center justify-center hover:bg-accent-primary hover:bg-opacity-20 transition-all"
              aria-label="Settings"
            >
              <i className="fas fa-cog" />
            </button>

            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary flex items-center justify-center text-white font-bold">
                {user.name.charAt(0)}
              </div>
              <div className="hidden md:block">
                <div className="font-medium">{user.name}</div>
                <div className="text-xs text-secondary">{user.role}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};
