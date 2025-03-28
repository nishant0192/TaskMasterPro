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

/**
 * Create a new task.
 * Input: { title: string; description?: string; dueDate?: string; priority?: number; reminderAt?: string; subtasks?: Array<{ title: string }> }
 */
export async function createTask(data: {
  title: string;
  description?: string;
  dueDate?: string;
  priority?: number;
  reminderAt?: string;
  subtasks?: { title: string }[];
}) {
  return request('task.createTask', data, 'mutation');
}

/**
 * Retrieve all tasks for the current user.
 * Optional filters: { status?, priority?, dueDateFrom?, dueDateTo?, search?, sortBy?, sortOrder? }
 */
export async function getTasks(filters?: {
  status?: string;
  priority?: number;
  dueDateFrom?: string;
  dueDateTo?: string;
  search?: string;
  sortBy?: 'dueDate' | 'priority' | 'createdAt';
  sortOrder?: 'asc' | 'desc';
}) {
  return request('task.getTasks', filters ?? {}, 'mutation');
}

/**
 * Retrieve a single task by its ID.
 * Input: { id: string }
 */
export async function getTask(data: { id: string }) {
  return request('task.getTask', data, 'mutation');
}

/**
 * Update a task.
 * Input: { id: string; title?, description?, dueDate?, priority?, status?, progress?, isArchived?, reminderAt? }
 */
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

/* ----- Subtask Endpoints ----- */

/**
 * Create a subtask for a given task.
 * Input: { taskId: string; title: string; order?: number; reminderAt?: string }
 */
export async function createSubtask(data: {
  taskId: string;
  title: string;
  order?: number;
  reminderAt?: string;
}) {
  return request('task.createSubtask', data, 'mutation');
}

/**
 * Get all subtasks for a given task.
 * Input: { taskId: string }
 */
export async function getSubtasks(data: { taskId: string }) {
  return request('task.getSubtasks', data, 'mutation');
}

/**
 * Update a subtask.
 * Input: { id: string; title?: string; isCompleted?: boolean; order?: number; reminderAt?: string }
 */
export async function updateSubtask(data: {
  id: string;
  title?: string;
  isCompleted?: boolean;
  order?: number;
  reminderAt?: string;
}) {
  return request('task.updateSubtask', data, 'mutation');
}

/**
 * Delete a subtask.
 * Input: { id: string }
 */
export async function deleteSubtask(data: { id: string }) {
  return request('task.deleteSubtask', data, 'mutation');
}

/* ----- Reminder Endpoint ----- */

/**
 * Set or update a task's reminder.
 * Input: { id: string; reminderAt: string }
 */
export async function setReminder(data: { id: string; reminderAt: string }) {
  return request('task.setReminder', data, 'mutation');
}

/* ----- Search Endpoint ----- */

/**
 * Search tasks by keyword.
 * Input: { query: string }
 */
export async function searchTasks(data: { query: string }) {
  return request('task.searchTasks', { query: data.query }, 'mutation');
}

/* ----- Comment Endpoints ----- */

/**
 * Add a comment to a task.
 * Input: { taskId: string; content: string }
 */
export async function addComment(data: { taskId: string; content: string }) {
  return request('task.addComment', data, 'mutation');
}

/**
 * Get comments for a task.
 * Input: { taskId: string }
 */
export async function getComments(data: { taskId: string }) {
  return request('task.getComments', data, 'mutation');
}

/* ----- Activity Log Endpoint ----- */

/**
 * Get activity logs for a task.
 * Input: { taskId: string }
 */
export async function getTaskActivity(data: { taskId: string }) {
  return request('task.getTaskActivity', data, 'mutation');
}

/* ----- Attachment Endpoints ----- */

/**
 * Create a new attachment for a task.
 * Input: { taskId: string; fileName: string; fileType?: string; fileSize?: number; fileData: string }
 */
export async function createAttachment(data: {
  taskId: string;
  fileName: string;
  fileType?: string;
  fileSize?: number;
  fileData: string;
}) {
  return request('task.createAttachment', data, 'mutation');
}

/**
 * Get all attachments for a given task.
 * Input: { taskId: string }
 */
export async function getAttachments(data: { taskId: string }) {
  return request('task.getAttachments', data, 'mutation');
}

/**
 * Delete an attachment.
 * Input: { id: string }
 */
export async function deleteAttachment(data: { id: string }) {
  return request('task.deleteAttachment', data, 'mutation');
}

export { request as postRequest };
