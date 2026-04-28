import { NavLink } from 'react-router-dom';

const items = [
  {
    to: '/',
    label: 'Home',
    icon: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
        <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
      </svg>
    ),
  },
  {
    to: '/reports',
    label: 'Reports',
    icon: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 7V3.5L18.5 9H13zM8 13h8v1.5H8V13zm0 3h8v1.5H8V16zm0-6h5v1.5H8V10z" />
      </svg>
    ),
  },
  {
    to: '/electrode-guide',
    label: 'Guide',
    icon: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
        <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z" />
      </svg>
    ),
  },
  {
    to: '/settings',
    label: 'Settings',
    icon: (
      <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
        <path d="M19.14 12.936a7.16 7.16 0 0 0 .065-.936c0-.32-.027-.63-.065-.936l2.01-1.57a.48.48 0 0 0 .115-.61l-1.903-3.293a.48.48 0 0 0-.583-.21l-2.372.953a6.97 6.97 0 0 0-1.61-.933l-.36-2.523A.465.465 0 0 0 14 3h-3.8a.465.465 0 0 0-.462.397l-.36 2.523a6.97 6.97 0 0 0-1.61.933l-2.373-.953a.465.465 0 0 0-.583.21L3.01 9.403a.47.47 0 0 0 .115.611l2.01 1.57a7.23 7.23 0 0 0-.064.936c0 .32.027.628.065.936l-2.01 1.57a.48.48 0 0 0-.116.61l1.903 3.293c.12.209.376.288.583.21l2.372-.953c.5.358 1.04.65 1.61.933l.36 2.523c.05.232.256.397.462.397H14c.252 0 .44-.165.46-.397l.36-2.523a6.97 6.97 0 0 0 1.61-.933l2.373.953c.208.078.463 0 .583-.21l1.903-3.293a.47.47 0 0 0-.115-.61l-2.01-1.57zM12 15.6A3.6 3.6 0 1 1 12 8.4a3.6 3.6 0 0 1 0 7.2z" />
      </svg>
    ),
  },
];

export default function BottomNav() {
  return (
    <nav className="fixed bottom-0 inset-x-0 z-50 bg-white border-t border-slate-100 safe-area-inset-bottom">
      <div className="mx-auto max-w-lg flex">
        {items.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              [
                'flex-1 flex flex-col items-center gap-0.5 py-2.5 text-[10px] font-medium transition-colors',
                isActive ? 'text-brand-500' : 'text-slate-400',
              ].join(' ')
            }
          >
            {icon}
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
