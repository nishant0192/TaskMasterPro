// frontend/app/api/notifications/notifications.ts
import { fetchWithAuth } from '@/api/fetchWithAuth';

const API_BASE_URL = process.env.API_BASE_URL || 'http://192.168.29.74:3000';

/**
 * Registers the Expo push token on the backend.
 * Input: { expoPushToken: string }
 */
export async function registerPushToken(data: { expoPushToken: string }) {
  const body = {
    id: Date.now(),
    method: 'mutation',
    params: { input: data },
    path: 'notification.registerPushToken',
  };
  const response = await fetchWithAuth(`${API_BASE_URL}/trpc/notification.registerPushToken`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
  });
  const json = await response.json();
  if (!json.result || !json.result.data) {
    throw new Error('Invalid response format');
  }
  return json.result.data;
}

/**
 * Sends a test notification to the provided Expo push token.
 * Input: { expoPushToken: string }
 */
export async function sendTestNotification(data: { expoPushToken: string }) {
  const body = {
    id: Date.now(),
    method: 'mutation',
    params: { input: data },
    path: 'notification.sendTestNotification',
  };
  const response = await fetchWithAuth(`${API_BASE_URL}/trpc/notification.sendTestNotification`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(body),
  });
  const json = await response.json();
  if (!json.result || !json.result.data) {
    throw new Error('Invalid response format');
  }
  return json.result.data;
}

export { fetchWithAuth as postRequest };
