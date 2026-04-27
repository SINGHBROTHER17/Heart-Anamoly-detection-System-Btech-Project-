import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import UploadPage from './pages/UploadPage.jsx';
import ReportPage from './pages/ReportPage.jsx';
import HistoryPage from './pages/HistoryPage.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<UploadPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/report/:id" element={<ReportPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

function NotFound() {
  return (
    <div className="text-center py-16">
      <p className="text-6xl mb-4">🩺</p>
      <h1 className="text-2xl font-bold mb-2">Page not found</h1>
      <p className="text-slate-500 dark:text-slate-400">
        The page you’re looking for doesn’t exist.
      </p>
    </div>
  );
}
