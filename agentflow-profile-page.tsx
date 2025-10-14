import React, { useState } from 'react';
import { User, Mail, Calendar, Award, Code, Zap, TrendingUp, Clock, Edit2, Save, X } from 'lucide-react';

// ============================================================================
// PROFILE PAGE
// ============================================================================

export default function ProfilePage() {
  const [isEditing, setIsEditing] = useState(false);
  const [profile, setProfile] = useState({
    username: 'demo_user',
    email: 'demo_user@agentflow.ai',
    role: 'Senior Developer',
    bio: 'Passionate about AI-driven development and automation. Building the future one agent at a time.',
    joined: 'September 2024',
    skills: ['React', 'Python', 'Machine Learning', 'DevOps', 'Cloud Architecture'],
    stats: {
      projectsCompleted: 24,
      tasksCompleted: 487,
      agentsManaged: 15,
      successRate: 94
    }
  });
  
  const [editForm, setEditForm] = useState(profile);
  const [newSkill, setNewSkill] = useState('');
  
  const handleSave = () => {
    setProfile(editForm);
    setIsEditing(false);
  };
  
  const handleCancel = () => {
    setEditForm(profile);
    setIsEditing(false);
  };
  
  const addSkill = () => {
    if (newSkill.trim() && !editForm.skills.includes(newSkill.trim())) {
      setEditForm(prev => ({
        ...prev,
        skills: [...prev.skills, newSkill.trim()]
      }));
      setNewSkill('');
    }
  };
  
  const removeSkill = (skill) => {
    setEditForm(prev => ({
      ...prev,
      skills: prev.skills.filter(s => s !== skill)
    }));
  };
  
  const recentActivities = [
    { icon: Code, text: 'Completed ML Pipeline project', time: '2 hours ago', color: 'cyan' },
    { icon: Zap, text: 'Assigned 3 new agents to tasks', time: '5 hours ago', color: 'blue' },
    { icon: Award, text: 'Achieved 95% success rate milestone', time: '1 day ago', color: 'green' },
    { icon: TrendingUp, text: 'Project performance increased by 15%', time: '2 days ago', color: 'purple' }
  ];
  
  return (
    <div className="pt-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-2">
          Profile
        </h1>
        <p className="text-gray-400">Manage your account and preferences</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Profile Info */}
        <div className="lg:col-span-1 space-y-6">
          {/* Avatar & Basic Info */}
          <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
            <div className="text-center mb-6">
              <div className="w-32 h-32 rounded-full bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-white text-4xl font-bold mx-auto mb-4">
                {profile.username[0].toUpperCase()}
              </div>
              <h2 className="text-2xl font-bold text-gray-200">{profile.username}</h2>
              <p className="text-gray-400">{profile.role}</p>
            </div>
            
            <div className="space-y-3">
              <div className="flex items-center space-x-3 text-gray-300">
                <Mail className="w-5 h-5 text-cyan-400" />
                <span className="text-sm">{profile.email}</span>
              </div>
              <div className="flex items-center space-x-3 text-gray-300">
                <Calendar className="w-5 h-5 text-cyan-400" />
                <span className="text-sm">Joined {profile.joined}</span>
              </div>
            </div>
            
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="w-full mt-6 py-2 bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors flex items-center justify-center space-x-2"
            >
              {isEditing ? <X className="w-4 h-4" /> : <Edit2 className="w-4 h-4" />}
              <span>{isEditing ? 'Cancel' : 'Edit Profile'}</span>
            </button>
          </div>
          
          {/* Stats */}
          <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-200 mb-4">Statistics</h3>
            <div className="space-y-4">
              {[
                { label: 'Projects', value: profile.stats.projectsCompleted, icon: Code, color: 'cyan' },
                { label: 'Tasks', value: profile.stats.tasksCompleted, icon: Clock, color: 'blue' },
                { label: 'Agents', value: profile.stats.agentsManaged, icon: Zap, color: 'purple' },
                { label: 'Success', value: `${profile.stats.successRate}%`, icon: TrendingUp, color: 'green' }
              ].map((stat, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <stat.icon className={`w-5 h-5 text-${stat.color}-400`} />
                    <span className="text-gray-300">{stat.label}</span>
                  </div>
                  <span className={`text-lg font-bold text-${stat.color}-400`}>{stat.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* Right Column - Details */}
        <div className="lg:col-span-2 space-y-6">
          {/* Bio Section */}
          <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-200 mb-4">About</h3>
            {isEditing ? (
              <textarea
                value={editForm.bio}
                onChange={(e) => setEditForm(prev => ({ ...prev, bio: e.target.value }))}
                className="w-full px-4 py-3 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400 resize-none"
                rows="4"
              />
            ) : (
              <p className="text-gray-300">{profile.bio}</p>
            )}
          </div>
          
          {/* Skills Section */}
          <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-200 mb-4">Skills</h3>
            
            {isEditing && (
              <div className="flex space-x-2 mb-4">
                <input
                  type="text"
                  value={newSkill}
                  onChange={(e) => setNewSkill(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && addSkill()}
                  placeholder="Add new skill..."
                  className="flex-1 px-4 py-2 bg-gray-800/50 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400"
                />
                <button
                  onClick={addSkill}
                  className="px-4 py-2 bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
                >
                  Add
                </button>
              </div>
            )}
            
            <div className="flex flex-wrap gap-2">
              {(isEditing ? editForm.skills : profile.skills).map((skill, i) => (
                <div
                  key={i}
                  className="px-4 py-2 bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 rounded-lg flex items-center space-x-2"
                >
                  <span>{skill}</span>
                  {isEditing && (
                    <button
                      onClick={() => removeSkill(skill)}
                      className="hover:text-red-400 transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
          
          {/* Recent Activity */}
          <div className="backdrop-blur-xl bg-gray-900/70 border border-cyan-500/20 rounded-xl p-6">
            <h3 className="text-xl font-bold text-gray-200 mb-4">Recent Activity</h3>
            <div className="space-y-3">
              {recentActivities.map((activity, i) => (
                <div key={i} className="flex items-start space-x-3 p-3 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors">
                  <div className={`p-2 rounded-lg bg-${activity.color}-500/20`}>
                    <activity.icon className={`w-5 h-5 text-${activity.color}-400`} />
                  </div>
                  <div className="flex-1">
                    <p className="text-gray-300">{activity.text}</p>
                    <p className="text-xs text-gray-500 mt-1">{activity.time}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Save/Cancel Buttons */}
          {isEditing && (
            <div className="flex space-x-4">
              <button
                onClick={handleSave}
                className="flex-1 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg font-medium hover:shadow-lg hover:shadow-cyan-500/50 transition-all flex items-center justify-center space-x-2"
              >
                <Save className="w-5 h-5" />
                <span>Save Changes</span>
              </button>
              <button
                onClick={handleCancel}
                className="flex-1 py-3 bg-gray-800/50 border border-gray-700 text-gray-300 rounded-lg font-medium hover:bg-gray-800 transition-all"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}