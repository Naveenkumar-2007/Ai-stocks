import { initializeApp } from 'firebase/app';
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  GoogleAuthProvider,
  signInWithPopup,
  updatePassword,
  sendPasswordResetEmail,
  updateProfile,
  onAuthStateChanged
} from 'firebase/auth';

const firebaseConfig = {
  apiKey: (process.env.REACT_APP_FIREBASE_API_KEY || '').trim(),
  authDomain: (process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || '').trim(),
  projectId: (process.env.REACT_APP_FIREBASE_PROJECT_ID || '').trim(),
  storageBucket: (process.env.REACT_APP_FIREBASE_STORAGE_BUCKET || '').trim(),
  messagingSenderId: (process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || '').trim(),
  appId: (process.env.REACT_APP_FIREBASE_APP_ID || '').trim(),
  measurementId: (process.env.REACT_APP_FIREBASE_MEASUREMENT_ID || '').trim()
};

const requiredKeys = [
  'REACT_APP_FIREBASE_API_KEY',
  'REACT_APP_FIREBASE_AUTH_DOMAIN',
  'REACT_APP_FIREBASE_PROJECT_ID',
  'REACT_APP_FIREBASE_STORAGE_BUCKET',
  'REACT_APP_FIREBASE_MESSAGING_SENDER_ID',
  'REACT_APP_FIREBASE_APP_ID'
];

const missingKeys = requiredKeys.filter((key) => !process.env[key]);

if (missingKeys.length > 0) {
  console.error('Firebase configuration error: The following environment variables are missing:', missingKeys);
  console.info('Tip: Ensure you have a .env file in the frontend/ directory with these variables defined.');
} else if (!firebaseConfig.apiKey || firebaseConfig.apiKey.length < 10) {
  console.error('Firebase configuration error: REACT_APP_FIREBASE_API_KEY appears to be invalid or too short.');
}

let app;
let auth;
let googleProvider;
let initializationError = null;

try {
  app = initializeApp(firebaseConfig);
  auth = getAuth(app);
  googleProvider = new GoogleAuthProvider();
  googleProvider.setCustomParameters({ prompt: 'select_account' });
  console.log("Firebase initialization successful");
} catch (error) {
  console.error("Firebase initialization failed:", error);
  initializationError = error;
  auth = null;
  googleProvider = null;
}

export {
  auth,
  googleProvider,
  initializationError,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
  signInWithPopup,
  updatePassword,
  sendPasswordResetEmail,
  updateProfile,
  onAuthStateChanged
};
