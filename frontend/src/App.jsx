import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import HomePage from './pages/HomePage.jsx';
import UploadPage from './pages/UploadPage.jsx';
import ReportPage from './pages/ReportPage.jsx';
import HistoryPage from './pages/HistoryPage.jsx';
import ElectrodePage from './pages/ElectrodePage.jsx';
import ArticlesPage from './pages/ArticlesPage.jsx';
import SettingsPage from './pages/SettingsPage.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/"                element={<HomePage />} />
        <Route path="/analyze"         element={<UploadPage />} />
        <Route path="/report/:id"      element={<ReportPage />} />
        <Route path="/reports"         element={<HistoryPage />} />
        <Route path="/electrode-guide" element={<ElectrodePage />} />
        <Route path="/articles"        element={<ArticlesPage />} />
        <Route path="/settings"        element={<SettingsPage />} />
        <Route path="*"                element={<NotFound />} />
      </Route>
    </Routes>
  );
}

function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-3">
      <span className="text-5xl">🩺</span>
      <h1 className="text-xl font-bold">Page not found</h1>
      <p className="text-slate-500 text-sm">The page you're looking for doesn't exist.</p>
    </div>
  );
}
