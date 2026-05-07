import { useState } from 'react';

const STORAGE_KEY = 'ecg_user_profile';

function readProfile() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  } catch {
    return {};
  }
}

export function useUserProfile() {
  const [profile, setProfile] = useState(readProfile);

  function updateProfile(updates) {
    const next = { ...profile, ...updates };
    setProfile(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }

  return [profile, updateProfile];
}
