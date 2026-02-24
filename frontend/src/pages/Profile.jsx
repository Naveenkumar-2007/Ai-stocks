import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { User, Mail, Shield, LogOut, Key, Sun, Moon, Bell, Globe } from 'lucide-react';

const Profile = () => {
  const { currentUser, logout } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  
  // Get user stats from localStorage
  const getUserStats = () => {
    const userEmail = currentUser?.email;
    if (!userEmail) return { predictions: 0, stocks: 0, accuracy: 0 };
    
    const stats = JSON.parse(localStorage.getItem(`userStats_${userEmail}`)) || {
      predictions: 0,
      stocksTracked: [],
      correctPredictions: 0
    };
    
    const accuracy = stats.predictions > 0 
      ? ((stats.correctPredictions / stats.predictions) * 100).toFixed(1)
      : 0;
    
    return {
      predictions: stats.predictions,
      stocks: stats.stocksTracked.length,
      accuracy: accuracy
    };
  };
  
  const stats = getUserStats();

  const handleLogout = async () => {
    try {
      setLoading(true);
      await logout();
      navigate('/');
    } catch (error) {
      console.error('Failed to log out', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdatePassword = () => {
    navigate('/change-password');
  };

  const getMemberSince = () => {
    if (currentUser?.metadata?.creationTime) {
      const date = new Date(currentUser.metadata.creationTime);
      return date.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric'
      });
    }
    return new Date().toLocaleDateString('en-US', {
      month: 'long',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const isEmailVerified = currentUser?.emailVerified || false;

  const getAuthProvider = () => {
    if (currentUser?.providerData && currentUser.providerData.length > 0) {
      const providerId = currentUser.providerData[0].providerId;
      if (providerId === 'google.com') return 'Google';
      if (providerId === 'password') return 'Email/Password';
      return providerId;
    }
    return 'Email/Password';
  };

  const getUserName = () => currentUser?.displayName || 'User';
  const getUserInitial = () => {
    if (currentUser?.displayName) {
      return currentUser.displayName.charAt(0).toUpperCase();
    }
    if (currentUser?.email) {
      return currentUser.email.charAt(0).toUpperCase();
    }
    return 'U';
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        {/* Header Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 mb-6">
          <div className="p-6 sm:p-8">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 sm:w-20 sm:h-20 bg-gradient-to-br from-cyan-500 to-cyan-600 rounded-full flex items-center justify-center text-white text-2xl sm:text-3xl font-bold shadow-lg">
                {getUserInitial()}
              </div>
              <div className="flex-1 min-w-0">
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">{getUserName()}</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">{currentUser?.email}</p>
                <div className="flex items-center gap-2 mt-2">
                  <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                    isEmailVerified 
                      ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' 
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-400'
                  }`}>
                    {isEmailVerified ? '✓ Verified' : 'Not Verified'}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    Member since {getMemberSince()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column - Settings */}
          <div className="lg:col-span-2 space-y-6">
            {/* Account Information */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center gap-2 mb-6">
                <User className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Contact Information</h2>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Name</label>
                  <p className="text-base font-semibold text-gray-900 dark:text-white">{getUserName()}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Email</label>
                  <p className="text-base text-gray-900 dark:text-white break-all">{currentUser?.email}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Provider</label>
                  <p className="text-base text-gray-900 dark:text-white">{getAuthProvider()}</p>
                </div>
              </div>
            </div>

            {/* Appearance Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center gap-2 mb-6">
                <Globe className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Appearance</h2>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-xl">
                  <div className="flex items-center gap-3">
                    {isDark ? <Moon className="w-5 h-5 text-cyan-400" /> : <Sun className="w-5 h-5 text-amber-500" />}
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">Theme</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        Current: {isDark ? 'Dark' : 'Light'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {isDark ? 'Dark' : 'Light'}
                    </span>
                    <button
                      onClick={toggleTheme}
                      className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                        isDark ? 'bg-cyan-600' : 'bg-gray-300'
                      }`}
                    >
                      <span
                        className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                          isDark ? 'translate-x-7' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Security Settings */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <div className="flex items-center gap-2 mb-6">
                <Shield className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Security</h2>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-xl">
                  <div className="flex items-center gap-3">
                    <Key className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                    <div>
                      <p className="font-medium text-gray-900 dark:text-white">Password</p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Change your password</p>
                    </div>
                  </div>
                  <button
                    onClick={handleUpdatePassword}
                    className="px-4 py-2 text-sm font-medium text-cyan-600 dark:text-cyan-400 hover:bg-cyan-50 dark:hover:bg-cyan-900/30 rounded-lg transition-colors"
                  >
                    Change
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Quick Actions */}
          <div className="space-y-6">
            {/* Account Stats */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">Account Stats</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-3 bg-gradient-to-r from-cyan-50 to-cyan-100 dark:from-cyan-900/20 dark:to-cyan-800/20 rounded-lg">
                  <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">Predictions Made</span>
                  <span className="text-xl font-bold text-cyan-600 dark:text-cyan-400">{stats.predictions}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
                  <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">Stocks Tracked</span>
                  <span className="text-xl font-bold text-blue-600 dark:text-blue-400">{stats.stocks}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
                  <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">Accuracy Rate</span>
                  <span className="text-xl font-bold text-green-600 dark:text-green-400">{stats.accuracy}%</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-4 text-center">
                Stats update as you use the platform
              </p>
            </div>

            {/* How to Use */}
            <div className="bg-gradient-to-br from-cyan-50 to-cyan-100 dark:from-cyan-900/30 dark:to-cyan-800/30 rounded-2xl border border-cyan-200 dark:border-cyan-600/30 p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-3">💡 Pro Tips</h3>
              <ul className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed space-y-2">
                <li>• Search any stock ticker to get AI-powered predictions</li>
                <li>• View technical indicators and sentiment analysis</li>
                <li>• Check daily forecasts for informed trading decisions</li>
                <li>• Track your favorite stocks for better insights</li>
              </ul>
            </div>

            {/* Sign Out */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-lg border border-red-200 dark:border-red-500/30 p-6">
              <h3 className="text-lg font-bold text-red-600 dark:text-red-400 mb-2">Sign Out</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Securely end your session.
              </p>
              <button
                onClick={handleLogout}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
              >
                <LogOut className="w-5 h-5" />
                <span>{loading ? 'Logging out...' : 'Log out'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
