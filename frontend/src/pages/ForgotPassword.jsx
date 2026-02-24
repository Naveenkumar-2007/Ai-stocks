import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Mail, ArrowLeft, CheckCircle, AlertCircle, Loader } from 'lucide-react';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { resetPassword } = useAuth();

  const handleSubmit = async (event) => {
    event.preventDefault();

    const trimmedEmail = email.trim();

    if (!trimmedEmail) {
      setError('Please enter the email associated with your account.');
      return;
    }

    try {
      setError('');
      setMessage('');
      setLoading(true);
      await resetPassword(trimmedEmail);
      setMessage('Password reset email sent. Check your inbox, spam, promotions, and other folders for the reset link.');
    } catch (err) {
      let feedback = 'Failed to send reset email. Verify the address and try again.';

      if (err?.code === 'auth/user-not-found') {
        feedback = 'No account exists with that email. Please sign up or try a different address.';
      } else if (err?.code === 'auth/invalid-email') {
        feedback = 'The email address appears to be invalid. Check the spelling and try again.';
      } else if (err?.code === 'auth/too-many-requests') {
        feedback = 'Too many attempts. Please wait a moment before trying again.';
      }

      setError(feedback);
      console.error('Password reset error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-transparent relative z-10">
      <div className="max-w-md w-full space-y-8 backdrop-blur-sm">
        <div className="text-center">
          <Link to="/login" className="inline-flex items-center text-sm text-brand-600 hover:text-brand-hover">
            <ArrowLeft className="h-4 w-4 mr-2" /> Back to login
          </Link>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 border border-brand-50">
          <div className="text-center mb-6">
            <div className="mx-auto h-12 w-12 bg-gradient-to-br from-brand-500 to-brand-600 rounded-xl flex items-center justify-center">
              <Mail className="h-6 w-6 text-white" />
            </div>
            <h2 className="mt-4 text-2xl font-bold text-brand-text">Reset Password</h2>
            <p className="mt-2 text-sm text-brand-muted">
              Enter your email address and we'll send you instructions to reset your password.
            </p>
          </div>

          {error && (
            <div className="mb-4 bg-red-50 border-l-4 border-red-500 p-4 rounded">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          )}

          {message && (
            <div className="mb-4 bg-brand-50 border-l-4 border-brand-500 p-4 rounded">
              <div className="flex items-center">
                <CheckCircle className="h-5 w-5 text-brand-600 mr-2" />
                <p className="text-sm text-brand-700">{message}</p>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-brand-muted mb-2">Email address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-brand-muted" />
                </div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="block w-full pl-10 pr-3 py-3 border border-brand-100 rounded-lg placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent"
                  placeholder="you@example.com"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-brand hover:bg-brand-hover focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader className="animate-spin h-5 w-5 mr-2" />
                  Sending reset link...
                </>
              ) : (
                'Send Reset Email'
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
