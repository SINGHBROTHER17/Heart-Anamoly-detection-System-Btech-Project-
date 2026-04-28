/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#fff0f1',
          100: '#ffe0e2',
          200: '#ffc2c6',
          500: '#e84c5a',
          600: '#d43d4b',
          700: '#be2f3b',
        },
        risk: {
          none:     '#22c55e',
          possible: '#f59e0b',
          likely:   '#f97316',
          high:     '#ef4444',
        },
        surface: '#f5f6fa',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
