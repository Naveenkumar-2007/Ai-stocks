import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Prediction from './pages/Prediction';
import About from './pages/About';
import Profile from './pages/Profile';
import Login from './pages/Login';
import Register from './pages/Register';
import ChangePassword from './pages/ChangePassword';
import ForgotPassword from './pages/ForgotPassword';
import Footer from './components/Footer';
import ProtectedRoute from './components/ProtectedRoute';
import ChatBot from './components/ChatBot';
import AdminDashboard from './pages/AdminDashboard';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import CinematicBackground from './components/CinematicBackground';
import './App.css';
import './styles/mobile.css';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <div className="min-h-screen bg-transparent transition-colors duration-300 flex flex-col relative overflow-hidden">
            <CinematicBackground />
            <Navigation />
            <main className="flex-grow">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/about" element={<About />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/forgot-password" element={<ForgotPassword />} />
                <Route
                  path="/prediction"
                  element={(
                    <ProtectedRoute>
                      <Prediction />
                    </ProtectedRoute>
                  )}
                />
                <Route
                  path="/profile"
                  element={(
                    <ProtectedRoute>
                      <Profile />
                    </ProtectedRoute>
                  )}
                />
                <Route
                  path="/change-password"
                  element={(
                    <ProtectedRoute>
                      <ChangePassword />
                    </ProtectedRoute>
                  )}
                />
                <Route
                  path="/admin"
                  element={(
                    <ProtectedRoute>
                      <AdminDashboard />
                    </ProtectedRoute>
                  )}
                />
              </Routes>
            </main>
            <Footer />
            <ChatBot />
          </div>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;