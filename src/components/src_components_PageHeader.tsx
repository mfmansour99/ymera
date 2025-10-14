import React from 'react';

interface PageAction {
  label: string;
  icon: string;
  onClick: () => void;
  primary?: boolean;
  disabled?: boolean;
}

interface PageHeaderProps {
  title: string;
  subtitle: string;
  actions?: PageAction[];
}

export const PageHeader: React.FC<PageHeaderProps> = ({ title, subtitle, actions = [] }) => {
  return (
    <div className="mb-8">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div className="mb-6 md:mb-0">
          <h1 className="text-3xl font-bold text-gradient mb-1">{title}</h1>
          <p className="text-secondary">{subtitle}</p>
        </div>

        {actions.length > 0 && (
          <div className="flex flex-wrap gap-3">
            {actions.map((action, index) => (
              <button
                key={index}
                onClick={action.onClick}
                disabled={action.disabled}
                className={`glass-button flex items-center space-x-2 ${
                  action.primary
                    ? 'bg-accent-primary bg-opacity-20 border-accent-primary text-accent-primary'
                    : ''
                } ${action.disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-opacity-30'}`}
              >
                {action.icon && <i className={action.icon} />}
                <span>{action.label}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
