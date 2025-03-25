import { fetchWithAuth } from '@/api/fetchWithAuth';

const API_BASE_URL = process.env.API_BASE_URL || 'http://192.168.29.16:3000';

/**
 * Helper to perform a request to a tRPC endpoint.
 * For queries, input is serialized into the URL; for mutations, a POST request is sent.
 */
async function request(
  path: string,
  input: Record<string, unknown>,
  method: 'mutation' | 'query' = 'mutation'
) {
  if (method === 'query') {
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

/* ----- Task Endpoints ----- */

export async function createTask(data: {
  title: string;
  description?: string;
  dueDate?: string;
  priority?: number;
}) {
  return request('task.createTask', data, 'mutation');
}

export async function getTasks(filters?: {
  status?: string;
  priority?: number;
  dueDateFrom?: string;
  dueDateTo?: string;
  search?: string;
  sortBy?: 'dueDate' | 'priority' | 'createdAt';
  sortOrder?: 'asc' | 'desc';
}) {
  return request('task.getTasks', filters ?? {}, 'query');
}

export async function getTask(data: { id: string }) {
  return request('task.getTask', data, 'query');
}

export async function updateTask(data: {
  id: string;
  title?: string;
  description?: string;
  dueDate?: string;
  priority?: number;
  status?: string;
  progress?: number;
  isArchived?: boolean;
  reminderAt?: string;
}) {
  return request('task.updateTask', data, 'mutation');
}

export async function completeTask(data: { id: string }) {
  return request('task.completeTask', data, 'mutation');
}

export async function deleteTask(data: { id: string }) {
  return request('task.deleteTask', data, 'mutation');
}

/* ----- Subtask Endpoints ----- */

export async function createSubtask(data: { taskId: string; title: string }) {
  return request('task.createSubtask', data, 'mutation');
}

export async function getSubtasks(data: { taskId: string }) {
  return request('task.getSubtasks', data, 'query');
}

export async function updateSubtask(data: { id: string; title?: string; isCompleted?: boolean }) {
  return request('task.updateSubtask', data, 'mutation');
}

export async function deleteSubtask(data: { id: string }) {
  return request('task.deleteSubtask', data, 'mutation');
}

/* ----- Comment Endpoints ----- */

export async function addComment(data: { taskId: string; content: string }) {
  return request('task.addComment', data, 'mutation');
}

export async function getComments(data: { taskId: string }) {
  return request('task.getComments', data, 'query');
}

/* ----- Search Endpoint ----- */

/**
 * Search tasks by query.
 * Input: { query: string }
 */
export async function searchTasks(data: { query: string }) {
  return request('task.searchTasks', { query: data.query }, 'query');
} 

export { request as postRequest };
