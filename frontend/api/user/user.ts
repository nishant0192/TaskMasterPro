import { fetchWithAuth } from '@/api/auth/fetchWithAuth';

const API_BASE_URL = process.env.API_BASE_URL || 'http://192.168.29.16:3000';

/**
 * Helper to perform a request to a tRPC endpoint.
 * If method is 'query', it sends a GET request with serialized input;
 * for mutations, it sends a POST request.
 *
 * @param path - The tRPC procedure path (e.g. "user.getProfile")
 * @param input - The input data for the mutation/query.
 * @param method - The request method ("mutation" or "query").
 * @returns The parsed JSON response (unwrapped from the tRPC envelope).
 */
async function request(
  path: string,
  input: Record<string, unknown>,
  method: 'mutation' | 'query' = 'mutation'
) {
  if (method === 'query') {
    // Serialize input for GET queries.
    const params = encodeURIComponent(JSON.stringify({ input }));
    const url = `${API_BASE_URL}/trpc/${path}?id=${Date.now()}&params=${params}`;
    const response = await fetchWithAuth(url, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
    });
    const json = await response.json();
    if (!json.result || !json.result.data) {
      throw new Error('Invalid response format');
    }
    return json.result.data;
  } else {
    // For mutations, use POST.
    const body = {
      id: Date.now(),
      method,
      params: { input },
      path,
    };
    const response = await fetchWithAuth(`${API_BASE_URL}/trpc/${path}`, {
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
}

/**
 * Fetch the current user's profile.
 */
export async function getProfile() {
  return request('user.getProfile', {}, 'query');
}

/**
 * Update the current user's profile.
 * @param data - An object containing fields to update (e.g. name, profileImage)
 */
export async function updateProfile(data: { name?: string; profileImage?: string }) {
  return request('user.updateProfile', data, 'mutation');
}

/**
 * Change the current user's password.
 * @param data - An object containing currentPassword and newPassword.
 */
export async function changePassword(data: { currentPassword: string; newPassword: string }) {
  return request('user.changePassword', data, 'mutation');
}

/**
 * Log out the current user by clearing the refresh token cookie.
 */
export async function logout() {
  return request('user.logout', {}, 'mutation');
}

export { request as postRequest };
