import { refreshTokens } from './auth';
import { useAuthStore } from '@/store/authStore';
import * as SecureStore from 'expo-secure-store';

/**
 * Custom fetch wrapper that automatically tries to refresh the access token
 * if a request returns a 401 error.
 * Retrieves the access token from SecureStore.
 *
 * @param url - The URL to fetch.
 * @param options - The request options.
 * @returns A promise that resolves with the Response.
 */
export async function fetchWithAuth(url: string, options: RequestInit): Promise<Response> {
  // Retrieve the access token from SecureStore.
  let token = await SecureStore.getItemAsync('accessToken');

  // Clone and update headers.
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(options.headers as Record<string, string> || {}) };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let response = await fetch(url, { ...options, headers, credentials: 'include' });

  // If unauthorized, attempt to refresh the token.
  if (response.status === 401) {
    try {
      const refreshResponse = await refreshTokens();
      if (refreshResponse && refreshResponse.accessToken) {
        // Update secure storage and Zustand store.
        await useAuthStore.getState().updateAccessToken(refreshResponse.accessToken);
        await SecureStore.setItemAsync('accessToken', refreshResponse.accessToken, {
          keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK,
        });
        token = refreshResponse.accessToken;
        // Update the header with the new token.
        headers['Authorization'] = `Bearer ${token}`;
        // Retry the original request with updated headers.
        response = await fetch(url, { ...options, headers, credentials: 'include' });
      } else {
        await useAuthStore.getState().clearUser();
        throw new Error('Session expired. Please log in again.');
      }
    } catch (err) {
      await useAuthStore.getState().clearUser();
      throw err;
    }
  }
  
  return response;
}
