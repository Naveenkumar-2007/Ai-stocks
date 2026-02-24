/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark theme - matching reference images
        dark: {
          bg: '#0A0E1F',      // Main dark background
          card: '#131833',     // Card background
          elevated: '#1A2038', // Elevated elements
          border: '#252B4A',   // Borders
          hover: '#2F3658',    // Hover states
        },
        // Light theme - clean and bright
        light: {
          bg: '#FFFFFF',       // Main light background
          card: '#F8F9FA',     // Card background
          elevated: '#FFFFFF', // Elevated elements
          border: '#E5E7EB',   // Borders
          hover: '#F3F4F6',    // Hover states
        },
        // Cyan/Blue brand color (matching reference)
        cyan: {
          400: '#22D3EE',
          500: '#00B8FF',      // Primary cyan
          600: '#0099FF',
          700: '#0080E6',
        },
        // Keep brand colors for backward compatibility
        brand: {
          50: '#E0F7FF',
          100: '#B9EDFF',
          200: '#7DD9FF',
          300: '#38C5FF',
          400: '#00B8FF',
          500: '#0099FF',
          600: '#0080E6',
          700: '#0066CC',
          800: '#004C99',
          900: '#003366',
          DEFAULT: '#00B8FF',
          hover: '#0099FF',
        },
        danger: {
          500: '#ef4444',
          600: '#dc2626',
        },
        success: {
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-dark': 'linear-gradient(to bottom right, #0A0E27, #131733, #1A1F3A)',
      },
      boxShadow: {
        'glow': '0 0 20px rgba(14, 165, 233, 0.3)',
        'glow-lg': '0 0 30px rgba(14, 165, 233, 0.4)',
      }
    },
  },
  plugins: [],
}