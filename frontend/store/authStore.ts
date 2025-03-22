import { create } from 'zustand';
import * as SecureStore from 'expo-secure-store';

export interface User {
  id: string;
  email: string;
  name?: string;
  profileImage?: string;
  // add additional fields as needed
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  // These functions now return Promises because they perform async secure storage operations.
  setUser: (user: User, accessToken: string) => Promise<void>;
  clearUser: () => Promise<void>;
  updateAccessToken: (accessToken: string) => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  accessToken: null,
  setUser: async (user, accessToken) => {
    // Save the access token in secure storage.
    await SecureStore.setItemAsync('accessToken', accessToken, {
      keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK,
    });
    // Update Zustand state.
    set({ user, accessToken });
  },
  updateAccessToken: async (accessToken) => {
    // Update secure storage with the new token.
    await SecureStore.setItemAsync('accessToken', accessToken, {
      keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK,
    });
    // Update Zustand state.
    set((state) => ({ ...state, accessToken }));
  },
  clearUser: async () => {
    // Remove the access token from secure storage.
    await SecureStore.deleteItemAsync('accessToken');
    // Clear Zustand state.
    set({ user: null, accessToken: null });
  },
}));
