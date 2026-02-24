import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col items-center gap-2">
          <div className="text-center">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
              AI Stock Price Predictions
            </h3>
            <p className="text-sm text-gray-600 dark:text-white">
              Powered by <span className="font-semibold">Datavision</span>
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              © 2026 Datavision • Advanced Machine Learning • Real-time Analytics
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;