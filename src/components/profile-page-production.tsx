import React, { useState, useMemo, useCallback } from 'react';
import { User, Edit3, Save, AlertCircle, CheckCircle, X } from 'lucide-react';

// ============================================================================
// TYPES
// ============================================================================

interface UserType {
  id: string;
  name: string;
  email: string;
  role: string;
  avatar?: string;
  lastLogin: string;
  preferences?: {
    theme?: 'dark' | 'light';
    notifications?: boolean;
    language?: string;
  };
}

interface ProfilePageProps {
  user: UserType;
  onUpdateUser: (updatedData: Partial<UserType>) => void;
}

interface ValidationErrors {
  name?: string;
  email?: string;
}

// ============================================================================
// TOGGLE SWITCH COMPONENT
// ============================================================================

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
  label?: string;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = React.memo(({
  checked,
  onChange,
  disabled = false,
  label
}) => {
  const id = useMemo(() => `toggle-${Math.random().toString(36).slice(2)}`, []);

  return (
    <label
      htmlFor={id}
      style={{ 
        position: 'relative', 
        display: 'inline-flex', 
        alignItems: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1
      }}
      aria-label={label}
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => !disabled && onChange(e.target.checked)}
        disabled={disabled}
        style={{ position: 'absolute', opacity: 0, width: 0, height: 0 }}
        aria-label={label || 'Toggle switch'}
      />
      <div style={{
        width: '44px',
        height: '24px',
        backgroundColor: checked ? '#06b6d4' : 'rgba(255,255,255,0.1)',
        borderRadius: '12px',
        position: 'relative',
        transition: 'background-color 0.2s',
        border: '1px solid rgba(255,255,255,0.2)'
      }}>
        <div style={{
          position: 'absolute',
          top: '2px',
          left: checked ? '22px' : '2px',
          width: '18px',
          height: '18px',
          backgroundColor: 'white',
          borderRadius: '50%',
          transition: 'left 0.2s',
          boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
        }} />
      </div>
    </label>
  );
});

ToggleSwitch.displayName = 'ToggleSwitch';

// ============================================================================
// FORM VALIDATION
// ============================================================================

const validateForm = (formData: { name: string; email: string }): ValidationErrors => {
  const errors: ValidationErrors = {};

  if (!formData.name.trim()) {
    errors.name = 'Name is required';
  } else if (formData.name.trim().length < 2) {
    errors.name = 'Name must be at least 2 characters';
  } else if (formData.name.trim().length > 50) {
    errors.name = 'Name must be less than 50 characters';
  }

  if (!formData.email.trim()) {
    errors.email = 'Email is required';
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
    errors.email = 'Invalid email format';
  }

  return errors;
};

// ============================================================================
// PROFILE PAGE COMPONENT
// ============================================================================

export const ProfilePage: React.FC<ProfilePageProps> = ({ user, onUpdateUser }) => {
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState({
    name: user.name,
    email: user.email,
    role: user.role,
    avatar: user.avatar || ''
  });
  const [preferences, setPreferences] = useState({
    theme: user.preferences?.theme || 'dark',
    notifications: user.preferences?.notifications || false,
    language: user.preferences?.language || 'en'
  });
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (validationErrors[name as keyof ValidationErrors]) {
      setValidationErrors(prev => ({ ...prev, [name]: undefined }));
    }
  }, [validationErrors]);

  const handleSubmit = useCallback(async () => {
    const errors = validateForm(formData);
    if (Object.keys(errors).length > 0) {
      setValidationErrors(errors);
      return;
    }

    setSaveStatus('saving');

    try {
      await new Promise(resolve => setTimeout(resolve, 800));
      onUpdateUser({ ...formData, preferences });
      setSaveStatus('saved');
      setEditMode(false);
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (error) {
      setSaveStatus('error');
      console.error('Failed to update profile:', error);
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  }, [formData, preferences, onUpdateUser]);

  const handleCancel = useCallback(() => {
    setEditMode(false);
    setFormData({
      name: user.name,
      email: user.email,
      role: user.role,
      avatar: user.avatar || ''
    });
    setPreferences({
      theme: user.preferences?.theme || 'dark',
      notifications: user.preferences?.notifications || false,
      language: user.preferences?.language || 'en'
    });
    setValidationErrors({});
  }, [user]);

  return (
    <div style={{ minHeight: '100vh', paddingTop: '5rem', paddingBottom: '5rem', background: '#0a0a0a' }}>
      <div style={{ maxWidth: '80rem', margin: '0 auto', padding: '0 2rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>
            Your Profile
          </h1>
          <p style={{ color: '#9ca3af' }}>Manage your personal information and preferences</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '2rem', maxWidth: '1200px' }}>
          <div style={{
            backdropFilter: 'blur(16px)',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '16px',
            padding: '2rem'
          }}>
            {/* Header with Avatar */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '2rem', paddingBottom: '2rem', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
              <div style={{
                width: '96px',
                height: '96px',
                borderRadius: '12px',
                background: 'linear-gradient(to right, #06b6d4, #8b5cf6)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '40px',
                fontWeight: '800',
                color: 'white',
                overflow: 'hidden'
              }}>
                {formData.avatar ? (
                  <img src={formData.avatar} alt="Profile" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  formData.name.charAt(0).toUpperCase()
                )}
              </div>
              
              <div style={{ flex: 1 }}>
                <h2 style={{ fontSize: '1.5rem', fontWeight: '800', color: 'white', marginBottom: '4px' }}>
                  {formData.name}
                </h2>
                <p style={{ color: '#9ca3af' }}>{formData.email}</p>
                <p style={{ color: '#06b6d4', fontSize: '0.875rem', marginTop: '4px' }}>{formData.role}</p>
              </div>
              
              <button
                onClick={editMode ? handleSubmit : () => setEditMode(true)}
                disabled={saveStatus === 'saving'}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '12px 24px',
                  background: editMode ? 'rgba(6,182,212,0.2)' : 'rgba(255,255,255,0.1)',
                  border: `1px solid ${editMode ? 'rgba(6,182,212,0.4)' : 'rgba(255,255,255,0.2)'}`,
                  borderRadius: '8px',
                  color: editMode ? '#06b6d4' : 'white',
                  cursor: saveStatus === 'saving' ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  opacity: saveStatus === 'saving' ? 0.6 : 1
                }}
              >
                {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? (
                  <><CheckCircle size={18} />Saved!</>
                ) : editMode ? (
                  <><Save size={18} />Save</>
                ) : (
                  <><Edit3 size={18} />Edit</>
                )}
              </button>
            </div>

            {/* Form Fields */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div>
                <label style={{ display: 'block', color: '#d1d5db', marginBottom: '8px', fontSize: '0.875rem', fontWeight: '500' }}>
                  Full Name
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  disabled={!editMode}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255,255,255,0.05)',
                    border: `1px solid ${validationErrors.name ? '#ef4444' : 'rgba(255,255,255,0.1)'}`,
                    borderRadius: '8px',
                    color: 'white',
                    outline: 'none',
                    opacity: editMode ? 1 : 0.7
                  }}
                />
                {validationErrors.name && (
                  <p style={{ color: '#ef4444', fontSize: '0.75rem', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <AlertCircle size={12} />{validationErrors.name}
                  </p>
                )}
              </div>

              <div>
                <label style={{ display: 'block', color: '#d1d5db', marginBottom: '8px', fontSize: '0.875rem', fontWeight: '500' }}>
                  Email Address
                </label>
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  disabled={!editMode}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255,255,255,0.05)',
                    border: `1px solid ${validationErrors.email ? '#ef4444' : 'rgba(255,255,255,0.1)'}`,
                    borderRadius: '8px',
                    color: 'white',
                    outline: 'none',
                    opacity: editMode ? 1 : 0.7
                  }}
                />
                {validationErrors.email && (
                  <p style={{ color: '#ef4444', fontSize: '0.75rem', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <AlertCircle size={12} />{validationErrors.email}
                  </p>
                )}
              </div>

              <div>
                <label style={{ display: 'block', color: '#d1d5db', marginBottom: '8px', fontSize: '0.875rem', fontWeight: '500' }}>
                  Role
                </label>
                <select
                  name="role"
                  value={formData.role}
                  onChange={handleInputChange}
                  disabled={!editMode}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    color: 'white',
                    outline: 'none',
                    opacity: editMode ? 1 : 0.7
                  }}
                >
                  <option value="Administrator">Administrator</option>
                  <option value="Developer">Developer</option>
                  <option value="Designer">Designer</option>
                  <option value="Manager">Manager</option>
                  <option value="Viewer">Viewer</option>
                </select>
              </div>

              <div>
                <label style={{ display: 'block', color: '#d1d5db', marginBottom: '8px', fontSize: '0.875rem', fontWeight: '500' }}>
                  Last Active
                </label>
                <input
                  type="text"
                  value={new Date(user.lastLogin).toLocaleString()}
                  disabled
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.05)',
                    borderRadius: '8px',
                    color: '#9ca3af',
                    outline: 'none',
                    opacity: 0.6
                  }}
                />
              </div>
            </div>

            {/* Preferences Section */}
            <div style={{
              marginTop: '2rem',
              padding: '1.5rem',
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '12px'
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: 'white', marginBottom: '1rem' }}>
                Preferences
              </h3>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <h4 style={{ fontWeight: '600', color: 'white', marginBottom: '4px' }}>Dark Mode</h4>
                    <p style={{ fontSize: '0.875rem', color: '#9ca3af' }}>Use dark theme for better visibility</p>
                  </div>
                  <ToggleSwitch
                    checked={preferences.theme === 'dark'}
                    onChange={(value) => setPreferences(prev => ({ ...prev, theme: value ? 'dark' : 'light' }))}
                    disabled={!editMode}
                    label="Dark mode toggle"
                  />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <h4 style={{ fontWeight: '600', color: 'white', marginBottom: '4px' }}>Email Notifications</h4>
                    <p style={{ fontSize: '0.875rem', color: '#9ca3af' }}>Receive email updates and alerts</p>
                  </div>
                  <ToggleSwitch
                    checked={preferences.notifications}
                    onChange={(value) => setPreferences(prev => ({ ...prev, notifications: value }))}
                    disabled={!editMode}
                    label="Email notifications toggle"
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#d1d5db', marginBottom: '8px' }}>
                    Language
                  </label>
                  <select
                    value={preferences.language}
                    onChange={(e) => setPreferences(prev => ({ ...prev, language: e.target.value }))}
                    disabled={!editMode}
                    style={{
                      width: '100%',
                      padding: '12px',
                      background: 'rgba(255,255,255,0.05)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '8px',
                      color: 'white',
                      outline: 'none',
                      opacity: editMode ? 1 : 0.7
                    }}
                  >
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="ar">Arabic</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            {editMode && (
              <div style={{ marginTop: '1.5rem', display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
                <button
                  type="button"
                  onClick={handleCancel}
                  style={{
                    padding: '12px 24px',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    color: 'white',
                    cursor: 'pointer',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <X size={16} />Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={saveStatus === 'saving'}
                  style={{
                    padding: '12px 24px',
                    background: 'linear-gradient(to right, #06b6d4, #3b82f6)',
                    border: 'none',
                    borderRadius: '8px',
                    color: 'white',
                    cursor: saveStatus === 'saving' ? 'not-allowed' : 'pointer',
                    fontWeight: '600',
                    opacity: saveStatus === 'saving' ? 0.6 : 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <Save size={16} />{saveStatus === 'saving' ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Demo
export default function App() {
  const [user, setUser] = React.useState<UserType>({
    id: '1',
    name: 'Mohamed Mansour',
    email: 'mohamed@ymera.ai',
    role: 'Administrator',
    avatar: '',
    lastLogin: new Date().toISOString(),
    preferences: { theme: 'dark', notifications: true, language: 'en' }
  });

  return <ProfilePage user={user} onUpdateUser={(updates) => setUser(prev => ({ ...prev, ...updates }))} />;
}