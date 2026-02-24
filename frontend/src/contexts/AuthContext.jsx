import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import {
  auth,
  googleProvider,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  signInWithPopup,
  updatePassword as firebaseUpdatePassword,
  sendPasswordResetEmail,
  updateProfile,
  onAuthStateChanged,
  initializationError
} from '../firebase/config';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

const ADMIN_EMAIL_ALLOWLIST = (process.env.REACT_APP_ADMIN_EMAILS || '')
  .split(',')
  .map((entry) => entry.trim().toLowerCase())
  .filter(Boolean);

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [claims, setClaims] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);

  const evaluateAdmin = useCallback((userClaims, user) => {
    if (!user) {
      setIsAdmin(false);
      return;
    }

    const email = user.email ? user.email.toLowerCase() : '';
    const hasAdminClaim = Boolean(userClaims?.admin);
    const allowListed = Boolean(email && ADMIN_EMAIL_ALLOWLIST.includes(email));
    setIsAdmin(hasAdminClaim || allowListed);
  }, []);

  const refreshClaims = useCallback(async (user = auth.currentUser) => {
    if (!user) {
      setClaims(null);
      setIsAdmin(false);
      return null;
    }

    try {
      const tokenResult = await user.getIdTokenResult();
      const nextClaims = tokenResult?.claims || {};
      setClaims(nextClaims);
      evaluateAdmin(nextClaims, user);
      return tokenResult;
    } catch (err) {
      console.error('Failed to refresh user claims', err);
      setClaims(null);
      setIsAdmin(false);
      return null;
    }
  }, [evaluateAdmin]);

  const signup = async (email, password, displayName) => {
    if (!auth) throw new Error(initializationError ? initializationError.message : "Firebase Auth not initialized");
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    if (displayName) {
      await updateProfile(userCredential.user, { displayName });
    }
    await refreshClaims(userCredential.user);
    return userCredential;
  };

  const login = async (email, password) => {
    if (!auth) throw new Error(initializationError ? initializationError.message : "Firebase Auth not initialized");
    const credential = await signInWithEmailAndPassword(auth, email, password);
    await refreshClaims(credential.user);
    return credential;
  };

  const loginWithGoogle = async () => {
    if (!auth) throw new Error(initializationError ? initializationError.message : "Firebase Auth not initialized");
    const credential = await signInWithPopup(auth, googleProvider);
    await refreshClaims(credential.user);
    return credential;
  };

  const logout = async () => {
    setClaims(null);
    setIsAdmin(false);
    if (!auth) return;
    await signOut(auth);
  };

  const updatePassword = (newPassword) => {
    if (!auth.currentUser) {
      throw new Error('No authenticated user');
    }
    return firebaseUpdatePassword(auth.currentUser, newPassword);
  };

  const resetPassword = (email) => sendPasswordResetEmail(auth, email);

  const updateUserProfile = (data) => {
    if (!auth.currentUser) {
      throw new Error('No authenticated user');
    }
    return updateProfile(auth.currentUser, data);
  };

  useEffect(() => {
    if (!auth) {
      console.warn("Firebase Auth not initialized - skipping auth state listener");
      setLoading(false);
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setCurrentUser(user);
      if (user) {
        await refreshClaims(user);
      } else {
        setClaims(null);
        setIsAdmin(false);
      }
      setLoading(false);
    });

    return unsubscribe;
  }, [refreshClaims]);

  const getIdToken = useCallback(async (forceRefresh = false) => {
    if (!auth.currentUser) {
      return null;
    }
    try {
      return await auth.currentUser.getIdToken(forceRefresh);
    } catch (err) {
      console.error('Failed to retrieve ID token', err);
      return null;
    }
  }, []);

  const value = {
    currentUser,
    claims,
    isAdmin,
    signup,
    login,
    loginWithGoogle,
    logout,
    updatePassword,
    resetPassword,
    updateUserProfile,
    refreshClaims,
    getIdToken
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};
