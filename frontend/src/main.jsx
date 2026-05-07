import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App.jsx';
import { RecordingSessionProvider } from './context/RecordingSessionContext.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <RecordingSessionProvider>
        <App />
      </RecordingSessionProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
