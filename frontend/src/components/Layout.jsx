import { Outlet } from 'react-router-dom';
import BottomNav from './BottomNav.jsx';
import DeviceBanner from './DeviceBanner.jsx';

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col bg-surface">
      <DeviceBanner />

      <main className="flex-1 mx-auto w-full max-w-lg page-content">
        <Outlet />
      </main>

      <BottomNav />
    </div>
  );
}
