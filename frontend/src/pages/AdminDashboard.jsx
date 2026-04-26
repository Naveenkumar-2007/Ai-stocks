import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { Navigate } from 'react-router-dom';

const API_BASE_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000');

function AdminDashboard() {
  const { currentUser } = useAuth();
  const { isDark } = useTheme();
  
  // Security State
  const [isLocked, setIsLocked] = useState(() => {
    return sessionStorage.getItem('adminUnlocked') !== 'true';
  });
  const [passwordInput, setPasswordInput] = useState('');
  const [lockError, setLockError] = useState('');
  const [verifying, setVerifying] = useState(false);

  // Dashboard State
  const [activeTab, setActiveTab] = useState('overview');
  const [stats, setStats] = useState(null);
  const [models, setModels] = useState([]);
  const [users, setUsers] = useState([]);
  const [chatLogs, setChatLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [scheduleTime, setScheduleTime] = useState('04:00'); // Default 4 AM IST

  useEffect(() => {
    if (!isLocked) {
      fetchAllData();
    }
  }, [isLocked]);

  const handleUnlock = async (e) => {
    e.preventDefault();
    setVerifying(true);
    setLockError('');
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: passwordInput })
      });
      const data = await res.json();
      if (data.success) {
        sessionStorage.setItem('adminUnlocked', 'true');
        setIsLocked(false);
      } else {
        setLockError(`ACCESS DENIED: ${data.error || 'Invalid Master Security Protocol.'}`);
      }
    } catch (err) {
      setLockError('Server connection failed. Is the Flask backend running?');
    } finally {
      setVerifying(false);
      setPasswordInput('');
    }
  };

  const fetchAllData = async () => {
    setLoading(true);
    await Promise.all([
      fetchStats(),
      fetchModels(),
      fetchUsers(),
      fetchChatLogs()
    ]);
    setLoading(false);
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/stats`);
      const data = await res.json();
      if (data.success) setStats(data.metrics);
    } catch (error) { console.error(error); }
  };

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/models`);
      const data = await res.json();
      if (data.success) setModels(data.models);
    } catch (error) { console.error(error); }
  };

  const fetchUsers = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/users`);
      const data = await res.json();
      if (data.success) setUsers(data.users);
    } catch (error) { console.error(error); }
  };

  const fetchChatLogs = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/chat_logs`);
      const data = await res.json();
      if (data.success) setChatLogs(data.logs);
    } catch (error) { console.error(error); }
  };

  const forceRetrain = async (ticker) => {
    if (!window.confirm(`FORCE retrain LSTM for ${ticker}? This will consume GPU/CPU resources.`)) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/force_retrain/${ticker}`, { method: 'POST' });
      const data = await res.json();
      alert(data.message);
    } catch (error) { alert("Failed to trigger retrain."); }
  };

  const triggerTuning = async () => {
    if (!window.confirm("Trigger Optuna Hyperparameter Tuning? This takes hours to complete.")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/trigger_tuning`, { method: 'POST' });
      const data = await res.json();
      alert(data.message);
    } catch (error) { alert("Failed to start tuning."); }
  };

  const trainAllStocks = async () => {
    if (!window.confirm("Train ALL active stocks right now? This will consume significant server resources!")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/admin/train_all`, { method: 'POST' });
      const data = await res.json();
      alert(data.message);
    } catch (error) { alert("Failed to trigger global training."); }
  };

  const updateScheduleTime = async () => {
    if (!scheduleTime) return;
    try {
      // Convert IST to UTC for the backend
      const [h, m] = scheduleTime.split(':').map(Number);
      let date = new Date();
      date.setUTCHours(h - 5, m - 30, 0, 0); // Subtract 5 hours 30 mins
      const utcTime = `${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}`;
      
      const res = await fetch(`${API_BASE_URL}/api/admin/set_training_time`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ time: utcTime })
      });
      const data = await res.json();
      alert(`Schedule updated to ${scheduleTime} IST (which is ${utcTime} UTC).`);
    } catch (error) { alert("Failed to update schedule."); }
  };

  const deleteUser = async (userId, email) => {
    if (!window.confirm(`PERMANENTLY DELETE user ${email || 'Anonymous'} and all their watchlists/chats?`)) return;
    try {
      await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, { method: 'DELETE' });
      fetchUsers();
    } catch (error) { alert("Failed to delete user."); }
  };

  const clearCache = async () => {
    if (!window.confirm("Clear all Redis/Local cache? This will temporarily spike API billing.")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/cache/clear`, { method: 'POST' });
      const data = await res.json();
      alert(data.message);
    } catch (error) { alert("Failed to clear cache."); }
  };

  const upgradeUser = async (userId, currentTier) => {
    const newTier = currentTier === 'free' ? 'pro' : 'free';
    if (!window.confirm(`Change user to ${newTier.toUpperCase()} tier?`)) return;
    try {
      await fetch(`${API_BASE_URL}/api/admin/users/${userId}/tier`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tier: newTier })
      });
      fetchUsers();
    } catch (error) { alert("Failed to update user."); }
  };

  if (!currentUser) {
    return <Navigate to="/login" />;
  }

  // ---- BEAUTIFUL BRANDED LOCK SCREEN UI ----
  if (isLocked) {
    return (
      <div className="container mx-auto px-4 py-24 min-h-screen flex items-center justify-center relative overflow-hidden">
        {/* Background ambient glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-green-500/20 rounded-full blur-[100px] pointer-events-none"></div>

        <div className="max-w-md w-full bg-white dark:bg-dark-card/90 backdrop-blur-2xl rounded-3xl p-8 sm:p-10 border border-gray-200 dark:border-dark-border shadow-2xl text-center relative z-10 transition-colors">
          
          <div className="flex justify-center mb-6">
            <div className="relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-green-400 to-emerald-600 rounded-2xl blur opacity-30 animate-pulse"></div>
              <img
                src={isDark ? '/assets/logo-dark.jpg' : '/assets/logo-light.jpg'}
                alt="Datavision"
                className="relative w-20 h-20 rounded-xl object-cover shadow-lg border-2 border-white/10"
              />
            </div>
          </div>
          
          <h2 className="text-3xl font-black bg-gradient-to-r from-green-600 via-emerald-500 to-green-700 dark:from-green-400 dark:via-emerald-400 dark:to-green-500 bg-clip-text text-transparent mb-2">
            Datavision Admin
          </h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mb-8 font-medium">
            Secure Infrastructure Gateway
          </p>
          
          <form onSubmit={handleUnlock} className="space-y-5">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <i className="fas fa-key text-gray-400"></i>
              </div>
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="Enter Master Password"
                className="w-full pl-11 pr-4 py-4 bg-gray-50 dark:bg-dark-elevated border-2 border-gray-200 dark:border-dark-border text-gray-900 dark:text-white rounded-xl text-lg focus:outline-none focus:border-green-500 dark:focus:border-green-400 transition-all font-medium"
                required
                autoFocus
              />
            </div>
            
            {lockError && (
              <div className="bg-red-50 dark:bg-red-500/10 border-l-4 border-red-500 p-3 rounded text-left">
                <p className="text-red-700 dark:text-red-400 text-xs font-bold">{lockError}</p>
              </div>
            )}
            
            <button 
              type="submit" 
              disabled={verifying}
              className="w-full py-4 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white font-bold rounded-xl shadow-lg hover:shadow-green-500/30 transition-all transform hover:-translate-y-1 flex justify-center items-center gap-2"
            >
              {verifying ? <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div> : "Authenticate"}
            </button>
            <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-4 font-medium">
              Protected by Enterprise-Grade Security
            </p>
          </form>
        </div>
      </div>
    );
  }

  // ---- MAIN DASHBOARD UI ----
  const tabClasses = (tabName) => `
    px-6 py-3 rounded-xl font-bold transition-all duration-300 flex items-center gap-2
    ${activeTab === tabName 
      ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white shadow-lg shadow-green-500/30' 
      : 'bg-white dark:bg-dark-card text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-dark-elevated border border-gray-200 dark:border-dark-border'
    }
  `;

  return (
    <div className="container mx-auto px-4 py-12 sm:py-24 min-h-screen">
      <div className="text-center mb-12 animate-fade-in-up">
        <div className="inline-flex items-center justify-center mb-4">
           <img src={isDark ? '/assets/logo-dark.jpg' : '/assets/logo-light.jpg'} alt="Logo" className="w-12 h-12 rounded-lg shadow-md mr-3" />
        </div>
        <h1 className="text-4xl md:text-5xl font-black mb-4 bg-gradient-to-r from-green-600 via-emerald-600 to-green-700 dark:from-green-400 dark:via-emerald-400 dark:to-green-500 bg-clip-text text-transparent pb-2">
          Enterprise Command Center
        </h1>
        <p className="text-gray-600 dark:text-gray-400 text-lg max-w-2xl mx-auto flex items-center justify-center gap-2 font-medium">
          <i className="fas fa-shield-check text-green-500"></i> Authenticated as System Administrator
        </p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap justify-center gap-3 sm:gap-4 mb-12">
        <button onClick={() => setActiveTab('overview')} className={tabClasses('overview')}>
          <i className="fas fa-server"></i> Infrastructure
        </button>
        <button onClick={() => setActiveTab('users')} className={tabClasses('users')}>
          <i className="fas fa-users"></i> User Mgmt
        </button>
        <button onClick={() => setActiveTab('mlops')} className={tabClasses('mlops')}>
          <i className="fas fa-network-wired"></i> MLOps Clusters
        </button>
        <button onClick={() => setActiveTab('chat')} className={tabClasses('chat')}>
          <i className="fas fa-shield-alt"></i> AI Auditing
        </button>
      </div>

      <div className="max-w-7xl mx-auto">
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="relative">
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-green-500 mx-auto"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                 <img src={isDark ? '/assets/logo-dark.jpg' : '/assets/logo-light.jpg'} alt="Logo" className="w-8 h-8 rounded animate-pulse" />
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-8 animate-fade-in-up">
            
            {/* 1. Infrastructure Tab */}
            {activeTab === 'overview' && stats && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                
                {/* Uptime */}
                <div className="md:col-span-2 bg-white dark:bg-dark-card rounded-2xl p-8 border border-gray-200 dark:border-dark-border shadow-lg relative overflow-hidden group">
                  <div className="absolute top-0 right-0 w-48 h-48 bg-green-500/10 rounded-full blur-3xl group-hover:bg-green-500/20 transition-colors"></div>
                  <h3 className="text-gray-500 dark:text-gray-400 font-bold mb-2 uppercase tracking-wider text-sm flex items-center gap-2">
                    <i className="fas fa-clock text-green-500"></i> Server Uptime
                  </h3>
                  <div className="text-5xl font-black text-gray-900 dark:text-white mt-4">{stats.uptime_hours}<span className="text-2xl text-gray-400 ml-1">h</span></div>
                  <p className="text-green-600 dark:text-green-400 mt-4 text-sm font-bold"><i className="fas fa-check-circle mr-1"></i> Production Grade Stability</p>
                </div>

                {/* CPU/RAM */}
                <div className="bg-white dark:bg-dark-card rounded-2xl p-6 border border-gray-200 dark:border-dark-border shadow-lg">
                  <h3 className="text-gray-500 dark:text-gray-400 font-bold mb-4 text-sm uppercase flex items-center gap-2">
                    <i className="fas fa-microchip text-emerald-500"></i> CPU Load
                  </h3>
                  <div className="flex items-end gap-2 mb-3">
                    <div className="text-4xl font-black text-gray-900 dark:text-white">{stats.cpu_usage}%</div>
                  </div>
                  <div className="w-full bg-gray-100 dark:bg-gray-800 rounded-full h-3 overflow-hidden">
                    <div className={`h-full rounded-full transition-all duration-1000 ${stats.cpu_usage > 80 ? 'bg-red-500' : 'bg-gradient-to-r from-green-400 to-emerald-500'}`} style={{ width: `${stats.cpu_usage}%` }}></div>
                  </div>
                </div>

                <div className="bg-white dark:bg-dark-card rounded-2xl p-6 border border-gray-200 dark:border-dark-border shadow-lg">
                  <h3 className="text-gray-500 dark:text-gray-400 font-bold mb-4 text-sm uppercase flex items-center gap-2">
                    <i className="fas fa-memory text-emerald-500"></i> Memory Usage
                  </h3>
                  <div className="flex items-end gap-2 mb-3">
                    <div className="text-4xl font-black text-gray-900 dark:text-white">{stats.ram_usage_percent}%</div>
                  </div>
                  <div className="w-full bg-gray-100 dark:bg-gray-800 rounded-full h-3 overflow-hidden">
                    <div className={`h-full rounded-full transition-all duration-1000 ${stats.ram_usage_percent > 85 ? 'bg-red-500' : 'bg-gradient-to-r from-green-400 to-emerald-500'}`} style={{ width: `${stats.ram_usage_percent}%` }}></div>
                  </div>
                </div>

                {/* Global Search Leaderboard */}
                <div className="col-span-1 md:col-span-4 bg-gradient-to-br from-gray-50 to-white dark:from-dark-card dark:to-dark-elevated rounded-2xl p-8 border border-gray-200 dark:border-dark-border shadow-xl mt-4">
                  <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
                    <h3 className="text-2xl font-black text-gray-900 dark:text-white flex items-center gap-3">
                      <i className="fas fa-globe-americas text-emerald-500 text-3xl"></i> Global Prediction Analytics
                    </h3>
                    <p className="text-sm font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-100 dark:bg-emerald-500/10 px-4 py-2 rounded-lg mt-3 sm:mt-0">
                      Live Tracking
                    </p>
                  </div>

                  {/* Trending Now */}
                  <h4 className="text-lg font-bold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2"><i className="fas fa-fire text-orange-500"></i> Trending Now (Last 24H)</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
                    {stats.trending_stocks && stats.trending_stocks.length > 0 ? (
                      stats.trending_stocks.map((stock, idx) => (
                        <div key={idx} className="bg-orange-50/50 dark:bg-orange-900/10 border-2 border-orange-100 dark:border-orange-500/20 rounded-xl p-5 hover:border-orange-500/50 transition-all relative group transform hover:-translate-y-1">
                          <div className={`absolute top-0 right-0 w-8 h-8 flex items-center justify-center font-black text-sm rounded-bl-xl rounded-tr-lg bg-orange-100 dark:bg-orange-500/20 text-orange-600 dark:text-orange-400`}>
                            #{idx + 1}
                          </div>
                          <p className="text-orange-500 dark:text-orange-400 text-xs font-bold uppercase tracking-wider mb-2">Surging</p>
                          <h4 className="text-2xl font-black text-gray-900 dark:text-white mb-3">{stock.ticker}</h4>
                          <p className="text-orange-600 dark:text-orange-400 font-bold text-sm bg-orange-100 dark:bg-orange-500/10 inline-block px-3 py-1 rounded-md">
                            <i className="fas fa-arrow-trend-up mr-1"></i> {stock.count} Searches
                          </p>
                        </div>
                      ))
                    ) : (
                      <div className="col-span-5 text-gray-500 dark:text-gray-400 py-4 font-medium italic">No surging stocks in the last 24 hours.</div>
                    )}
                  </div>
                  
                  {/* All Time */}
                  <h4 className="text-lg font-bold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2"><i className="fas fa-crown text-yellow-500"></i> All-Time Most Searched</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
                    {stats.top_searched_stocks_all_time && stats.top_searched_stocks_all_time.length > 0 ? (
                      stats.top_searched_stocks_all_time.map((stock, idx) => (
                        <div key={idx} className="bg-white dark:bg-dark-card border-2 border-gray-100 dark:border-dark-border rounded-xl p-5 hover:border-emerald-500/50 dark:hover:border-emerald-400/50 hover:shadow-lg transition-all relative group transform hover:-translate-y-1">
                          <div className={`absolute top-0 right-0 w-10 h-10 flex items-center justify-center font-black text-lg rounded-bl-xl rounded-tr-lg ${idx === 0 ? 'bg-yellow-400 text-yellow-900' : idx === 1 ? 'bg-gray-300 text-gray-800' : idx === 2 ? 'bg-amber-600 text-amber-100' : 'bg-gray-100 dark:bg-dark-elevated text-gray-400'}`}>
                            #{idx + 1}
                          </div>
                          <p className="text-gray-500 dark:text-gray-400 text-xs font-bold uppercase tracking-wider mb-2">Most Searched</p>
                          <h4 className="text-3xl font-black text-gray-900 dark:text-white mb-3">{stock.ticker}</h4>
                          <p className="text-emerald-600 dark:text-emerald-400 font-bold text-sm bg-emerald-50 dark:bg-emerald-500/10 inline-block px-3 py-1 rounded-md">
                            <i className="fas fa-chart-line mr-1"></i> {stock.count} Searches
                          </p>
                        </div>
                      ))
                    ) : (
                      <div className="col-span-5 text-center text-gray-500 dark:text-gray-400 py-10 border-2 border-dashed border-gray-200 dark:border-dark-border rounded-xl font-medium">
                        No prediction data logged yet.
                      </div>
                    )}
                  </div>
                </div>

                {/* Cache Clear */}
                <div className="col-span-1 md:col-span-4 bg-white dark:bg-dark-card rounded-2xl p-6 sm:p-8 border border-gray-200 dark:border-dark-border shadow-lg flex flex-col sm:flex-row justify-between items-center mt-4 gap-6">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2 mb-2">
                      <i className="fas fa-database text-red-500"></i> API Cache Management
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">Clear Redis/Local cache to force fresh data fetches from Finnhub & TwelveData APIs.</p>
                  </div>
                  <button onClick={clearCache} className="px-6 py-4 bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400 border-2 border-red-200 dark:border-red-500/30 hover:bg-red-600 hover:text-white dark:hover:text-white rounded-xl font-bold transition-all whitespace-nowrap shadow-sm hover:shadow-red-500/30">
                    <i className="fas fa-trash-alt mr-2"></i> Purge All Cache
                  </button>
                </div>
              </div>
            )}

            {/* 2. User Management Tab */}
            {activeTab === 'users' && (
              <div className="bg-white dark:bg-dark-card rounded-2xl border border-gray-200 dark:border-dark-border shadow-xl overflow-hidden">
                <div className="p-6 border-b border-gray-100 dark:border-dark-border flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-gray-50 dark:bg-dark-elevated">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                      <i className="fas fa-users text-blue-500"></i> Global User Directory
                    </h3>
                    <p className="text-sm text-gray-500 mt-1">Manage platform access and subscription tiers.</p>
                  </div>
                  <span className="bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-400 px-4 py-2 rounded-lg text-sm font-black shadow-sm">
                    {users.length} Total Users
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-white dark:bg-dark-card text-gray-500 dark:text-gray-400 text-xs uppercase tracking-widest border-b border-gray-200 dark:border-dark-border">
                        <th className="p-5 font-bold">User Email</th>
                        <th className="p-5 font-bold">Subscription</th>
                        <th className="p-5 font-bold">Activity Metrics</th>
                        <th className="p-5 font-bold text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-dark-border">
                      {users.length === 0 ? (
                        <tr><td colSpan="4" className="p-8 text-center text-gray-500 font-medium">No users found in database.</td></tr>
                      ) : users.map((user, idx) => (
                        <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-dark-elevated transition-colors">
                          <td className="p-5 font-bold text-gray-900 dark:text-white flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center text-white text-xs shadow-sm">
                              {(user.email || 'A').charAt(0).toUpperCase()}
                            </div>
                            {user.email || 'Anonymous/Guest'}
                          </td>
                          <td className="p-5">
                            <span className={`px-3 py-1 rounded-md text-xs font-black uppercase tracking-wider ${user.tier === 'pro' ? 'bg-gradient-to-r from-amber-400 to-orange-500 text-white shadow-sm' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300'}`}>
                              {user.tier} {user.tier === 'pro' && <i className="fas fa-crown ml-1 text-white"></i>}
                            </span>
                          </td>
                          <td className="p-5 text-gray-600 dark:text-gray-400 text-sm font-medium">
                            <span className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded mr-2"><i className="fas fa-eye text-blue-400 mr-1"></i> {user.watchlists}</span>
                            <span className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded"><i className="fas fa-comment text-emerald-400 mr-1"></i> {user.chats}</span>
                          </td>
                          <td className="p-5 text-right whitespace-nowrap">
                            <button 
                              onClick={() => upgradeUser(user.id, user.tier)}
                              className="text-xs px-4 py-2 bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/30 hover:bg-blue-600 hover:text-white dark:hover:text-white rounded-lg transition-all font-bold shadow-sm mr-2"
                            >
                              Toggle PRO
                            </button>
                            <button 
                              onClick={() => deleteUser(user.id, user.email)}
                              className="text-xs px-3 py-2 bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-500/30 hover:bg-red-600 hover:text-white dark:hover:text-white rounded-lg transition-all font-bold shadow-sm"
                            >
                              <i className="fas fa-trash-alt"></i>
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* 3. MLOps Tab */}
            {activeTab === 'mlops' && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Optuna Tuning */}
                  <div className="bg-gradient-to-br from-purple-600 to-indigo-700 rounded-2xl p-6 flex flex-col justify-between shadow-xl text-white">
                    <div className="mb-6">
                      <h3 className="text-2xl font-black mb-2 flex items-center gap-2">
                        <i className="fas fa-brain"></i> Deep Hyperparameter Tuning
                      </h3>
                      <p className="text-purple-100 font-medium text-sm">Trigger a massive Optuna cluster job to optimize LSTM learning rates and XGBoost depths globally across all stocks.</p>
                    </div>
                    <button onClick={triggerTuning} className="w-full py-4 bg-white text-purple-700 hover:bg-purple-50 rounded-xl font-black shadow-lg transition-transform hover:-translate-y-1 flex justify-center items-center gap-2">
                      <i className="fas fa-play"></i> Start Optuna Job
                    </button>
                  </div>

                  {/* Global Training Control */}
                  <div className="bg-gradient-to-br from-blue-600 to-cyan-700 rounded-2xl p-6 flex flex-col justify-between shadow-xl text-white">
                    <div className="mb-6">
                      <h3 className="text-2xl font-black mb-2 flex items-center gap-2">
                        <i className="fas fa-bolt"></i> Global Batch Training
                      </h3>
                      <p className="text-blue-100 font-medium text-sm mb-4">Train every single active stock in the database right now, or adjust the daily UTC schedule.</p>
                      
                      <div className="flex items-center gap-3 bg-black/20 p-3 rounded-xl border border-white/10">
                        <label className="font-bold text-sm">Daily Schedule (IST):</label>
                        <input 
                          type="time" 
                          value={scheduleTime}
                          onChange={(e) => setScheduleTime(e.target.value)}
                          className="bg-white/10 border border-white/20 text-white px-3 py-1.5 rounded-lg focus:outline-none focus:border-cyan-300 text-sm font-mono"
                        />
                        <button onClick={updateScheduleTime} className="bg-cyan-500 hover:bg-cyan-400 text-white px-4 py-1.5 rounded-lg text-sm font-bold shadow-sm transition-colors">
                          Set Time
                        </button>
                      </div>
                    </div>
                    
                    <button onClick={trainAllStocks} className="w-full py-4 bg-white text-blue-700 hover:bg-blue-50 rounded-xl font-black shadow-lg transition-transform hover:-translate-y-1 flex justify-center items-center gap-2">
                      <i className="fas fa-forward"></i> Train ALL Stocks Now
                    </button>
                  </div>
                </div>

                <div className="bg-white dark:bg-dark-card rounded-2xl border border-gray-200 dark:border-dark-border shadow-xl overflow-hidden">
                  <div className="p-6 border-b border-gray-100 dark:border-dark-border bg-gray-50 dark:bg-dark-elevated">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                      <i className="fas fa-robot text-purple-500"></i> Active Predictive Models
                    </h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-white dark:bg-dark-card text-gray-500 dark:text-gray-400 text-xs uppercase tracking-widest border-b border-gray-200 dark:border-dark-border">
                          <th className="p-5 font-bold">Ticker</th>
                          <th className="p-5 font-bold">Last Trained</th>
                          <th className="p-5 font-bold">PSI Drift Score</th>
                          <th className="p-5 font-bold">Health Status</th>
                          <th className="p-5 font-bold text-right">Manual Override</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100 dark:divide-dark-border">
                        {models.length === 0 ? (
                          <tr><td colSpan="5" className="p-8 text-center text-gray-500 font-medium">No active models found in registry.</td></tr>
                        ) : models.map((model, idx) => (
                          <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-dark-elevated transition-colors">
                            <td className="p-5 font-black text-gray-900 dark:text-white text-lg">{model.ticker}</td>
                            <td className="p-5 text-gray-600 dark:text-gray-400 font-medium">
                              {model.last_trained === 'Never' ? 'Never' : new Date(model.last_trained).toLocaleString()}
                            </td>
                            <td className="p-5 font-mono font-bold text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-500/10 px-2 py-1 inline-block rounded mt-3">
                              {model.drift_score.toFixed(4)}
                            </td>
                            <td className="p-5">
                              {model.status === 'Healthy' ? (
                                <span className="px-3 py-1 bg-green-100 dark:bg-green-500/20 text-green-700 dark:text-green-400 rounded-md text-xs font-black tracking-wider uppercase">HEALTHY</span>
                              ) : (
                                <span className="px-3 py-1 bg-red-100 dark:bg-red-500/20 text-red-700 dark:text-red-400 rounded-md text-xs font-black tracking-wider uppercase animate-pulse">DECAYING</span>
                              )}
                            </td>
                            <td className="p-5 text-right">
                              <button onClick={() => forceRetrain(model.ticker)} className="px-5 py-2 bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-500/30 hover:bg-red-600 hover:text-white dark:hover:text-white rounded-lg text-sm font-bold transition-colors shadow-sm">
                                <i className="fas fa-bolt mr-1"></i> Force Retrain
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* 4. Chat Logs Tab */}
            {activeTab === 'chat' && (
              <div className="bg-white dark:bg-dark-card rounded-2xl border border-gray-200 dark:border-dark-border shadow-xl overflow-hidden">
                <div className="p-6 border-b border-gray-100 dark:border-dark-border flex justify-between items-center bg-gray-50 dark:bg-dark-elevated">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                    <i className="fas fa-comment-dots text-emerald-500"></i> AI Safety & RAG Auditing
                  </h3>
                  <button onClick={fetchChatLogs} className="text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 font-bold transition-colors bg-emerald-50 dark:bg-emerald-500/10 px-4 py-2 rounded-lg">
                    <i className="fas fa-sync-alt mr-2"></i> Refresh Logs
                  </button>
                </div>
                <div className="p-6 space-y-5 max-h-[700px] overflow-y-auto custom-scrollbar bg-gray-50 dark:bg-dark-card">
                  {chatLogs.length === 0 ? (
                    <div className="text-center py-10 text-gray-500 font-medium">No AI conversations logged.</div>
                  ) : chatLogs.map((log, idx) => (
                    <div key={idx} className={`p-5 rounded-2xl border shadow-sm ${log.sender === 'user' ? 'bg-white dark:bg-dark-elevated border-gray-200 dark:border-dark-border ml-0 mr-12 sm:mr-24' : 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-100 dark:border-emerald-800/50 ml-12 sm:ml-24 mr-0'}`}>
                      <div className="flex justify-between items-start mb-3">
                        <span className={`font-black text-xs uppercase tracking-wider flex items-center gap-2 ${log.sender === 'user' ? 'text-gray-500 dark:text-gray-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                          {log.sender === 'user' ? <><i className="fas fa-user-circle text-lg"></i> User</> : <><i className="fas fa-robot text-lg"></i> AI Assistant</>}
                        </span>
                        <span className="text-xs font-medium text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">{new Date(log.timestamp).toLocaleString()}</span>
                      </div>
                      <p className="text-base text-gray-700 dark:text-gray-300 break-words whitespace-pre-wrap leading-relaxed font-medium">{log.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
}

export default AdminDashboard;
