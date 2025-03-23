import { fetchWithAuth } from '@/api/fetchWithAuth';

const API_BASE_URL = process.env.API_BASE_URL || 'http://192.168.29.16:3000';

/**
 * Helper to perform a request to a tRPC endpoint.
 * If method is 'query', it sends a GET request with serialized input;
 * for mutations, it sends a POST request.
 *
 * @param path - The tRPC procedure path (e.g. "task.createTask")
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
    // For mutations, send a POST request.
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
 * Create a new task.
 * Input: { title: string; description?: string; dueDate?: string; priority?: number }
 */
export async function createTask(data: { title: string; description?: string; dueDate?: string; priority?: number }) {
  return request('task.createTask', data, 'mutation');
}

/**
 * Retrieve all tasks for the current user.
 */
export async function getTasks() {
  return request('task.getTasks', {}, 'query');
}

/**
 * Retrieve a single task by its ID.
 * Input: { id: string }
 */
export async function getTask(data: { id: string }) {
  return request('task.getTask', data, 'query');
}

/**
 * Update a task.
 * Input: { id: string; title?: string; description?: string; dueDate?: string; priority?: number; status?: string; progress?: number; isArchived?: boolean }
 */
export async function updateTask(data: { id: string; title?: string; description?: string; dueDate?: string; priority?: number; status?: string; progress?: number; isArchived?: boolean }) {
  return request('task.updateTask', data, 'mutation');
}

/**
 * Mark a task as completed.
 * Input: { id: string }
 */
export async function completeTask(data: { id: string }) {
  return request('task.completeTask', data, 'mutation');
}

/**
 * Delete a task.
 * Input: { id: string }
 */
export async function deleteTask(data: { id: string }) {
  return request('task.deleteTask', data, 'mutation');
}

export { request as postRequest };
