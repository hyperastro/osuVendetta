import { writable } from 'svelte/store';
import { browser } from '$app/environment';

// types
export type Severity = 'normal' | 'warning' | 'error';
export type Status = 'Normal' | 'Suspicious' | 'Cheating';
export interface AntiCheatResult {
  fileName: string;
  player: string;
  status: Status;
  cheatProbability: number;
}

// stores
const createPersistentStore = (key: string, startValue: boolean) => {
  const initialValue = browser ? JSON.parse(localStorage.getItem(key) || String(startValue)) : startValue;
  const store = writable(initialValue);

  if (browser) {
    store.subscribe(value => localStorage.setItem(key, String(value)));
  }

  return store;
};

export const drawerOpen = writable(true);
export const isDarkMode = createPersistentStore('darkMode', true);

// TODO (skyfly): implement replay processing logic
export const processReplay = async (file: File): Promise<AntiCheatResult> => {
  await new Promise(resolve => setTimeout(resolve, 1000));
  return {
    fileName: file.name,
    player: 'Unknown Player',
    status: ['Normal', 'Suspicious', 'Cheating'][Math.floor(Math.random() * 3)] as Status,
    cheatProbability: Math.random() * 100
  };
};