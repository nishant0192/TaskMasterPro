import { useMutation, useQueryClient, UseMutationResult } from '@tanstack/react-query';
import {
  createTask,
  getTasks,
  getTask,
  updateTask,
  completeTask,
  deleteTask,
  createSubtask,
  getSubtasks,
  updateSubtask,
  deleteSubtask,
  setReminder,
  searchTasks,
  addComment,
  getComments,
  getTaskActivity,
  createAttachment,
  getAttachments,
  deleteAttachment,
  deleteComment,
  updateComment,
} from './task';

/* ----- Task Hooks ----- */

export function useCreateTask(): UseMutationResult<
  any,
  Error,
  {
    title: string;
    description?: string;
    dueDate?: string;
    priority?: number;
    reminderAt?: string;
    subtasks?: { title: string }[];
  },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => createTask(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useGetTasks(filters?: {
  status?: string;
  priority?: number;
  dueDateFrom?: string;
  dueDateTo?: string;
  search?: string;
  sortBy?: 'dueDate' | 'priority' | 'createdAt';
  sortOrder?: 'asc' | 'desc';
}): UseMutationResult<any, Error, void, unknown> {
  return useMutation({ mutationFn: () => getTasks(filters) });
}

export function useGetTaskById(id: string): UseMutationResult<any, Error, void, unknown> {
  return useMutation({
    mutationFn: () => getTask({ id })
  });
}

export function useUpdateTask(): UseMutationResult<
  any,
  Error,
  {
    id: string;
    title?: string;
    description?: string;
    dueDate?: string;
    priority?: number;
    status?: string;
    progress?: number;
    isArchived?: boolean;
    reminderAt?: string;
  },
  { previousTasks: any }
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => updateTask(data),
    onMutate: async (data) => {
      await queryClient.cancelQueries({ queryKey: ['tasks'] });
      const previousTasks = queryClient.getQueryData(['tasks']);
      queryClient.setQueryData(['tasks'], (old: any) => {
        if (!old) return old;
        return {
          ...old,
          tasks: old.tasks.map((task: any) =>
            task.id === data.id ? { ...task, ...data } : task
          ),
        };
      });
      return { previousTasks };
    },
    onError: (err, data, context) => {
      queryClient.setQueryData(['tasks'], context?.previousTasks);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useCompleteTask(): UseMutationResult<
  any,
  Error,
  { id: string },
  { previousTasks: any }
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => completeTask(data),
    onMutate: async (data) => {
      await queryClient.cancelQueries({ queryKey: ['tasks'] });
      const previousTasks = queryClient.getQueryData(['tasks']);
      queryClient.setQueryData(['tasks'], (old: any) => {
        if (!old) return old;
        return {
          ...old,
          tasks: old.tasks.map((task: any) =>
            task.id === data.id
              ? { ...task, status: 'DONE', progress: 100, completedAt: new Date(), isArchived: true }
              : task
          ),
        };
      });
      return { previousTasks };
    },
    onError: (err, data, context) => {
      queryClient.setQueryData(['tasks'], context?.previousTasks);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

export function useDeleteTask(): UseMutationResult<
  { success: boolean },
  Error,
  { id: string },
  { previousTasks: any }
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => {
      const res = await deleteTask(data);
      // Check if the backend returned success
      if (!res || !res.success) {
        throw new Error('Deletion failed');
      }
      return res;
    },
    onMutate: async (data) => {
      await queryClient.cancelQueries({ queryKey: ['tasks'] });
      const previousTasks = queryClient.getQueryData(['tasks']);
      queryClient.setQueryData(['tasks'], (old: any) => {
        if (!old) return old;
        return {
          ...old,
          tasks: old.tasks.filter((task: any) => task.id !== data.id),
        };
      });
      return { previousTasks };
    },
    onError: (err, data, context) => {
      queryClient.setQueryData(['tasks'], context?.previousTasks);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });
}

/* ----- Subtask Hooks ----- */

export function useCreateSubtask(): UseMutationResult<
  any,
  Error,
  { taskId: string; title: string; order?: number; reminderAt?: string },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => createSubtask(data),
    onSuccess: (_, data) => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      queryClient.invalidateQueries({ queryKey: ['subtasks', data.taskId] });
    },
  });
}

export function useGetSubtasks(taskId: string): UseMutationResult<any, Error, void, unknown> {
  return useMutation({
    mutationFn: () => getSubtasks({ taskId })
  });
}

export function useUpdateSubtask(): UseMutationResult<
  any,
  Error,
  { id: string; title?: string; isCompleted?: boolean; order?: number; reminderAt?: string },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => updateSubtask(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['subtasks'] });
    },
  });
}

export function useDeleteSubtask(): UseMutationResult<
  any,
  Error,
  { id: string },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => deleteSubtask(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['subtasks'] });
    },
  });
}

/* ----- Comment Hooks ----- */

export function useAddComment(): UseMutationResult<
  any,
  Error,
  { taskId: string; content: string },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => addComment(data),
    onSuccess: () => {
      // Invalidate or refetch comments (depending on your query key strategy)
      queryClient.invalidateQueries({ queryKey: ['comments'] });
    },
  });
}

/**
 * Hook to retrieve comments for a task.
 */
export function useGetComments(taskId: string): UseMutationResult<any, Error, void, unknown> {
  return useMutation({
    mutationFn: () => getComments({ taskId }),
  });
}

/**
 * Hook to update a comment.
 */
export function useUpdateComment(): UseMutationResult<
  any,
  Error,
  { id: string; content: string },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => updateComment(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['comments'] });
    },
  });
}

/**
 * Hook to delete a comment.
 */
export function useDeleteComment(): UseMutationResult<
  any,
  Error,
  { id: string },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => deleteComment(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['comments'] });
    },
  });
}

/* ----- Search Hook ----- */

export function useSearchTasks(query: string): UseMutationResult<any, Error, void, unknown> {
  return useMutation({
    mutationFn: () => searchTasks({ query })
  });
}

/* ----- Attachment Hooks ----- */

export function useCreateAttachment(): UseMutationResult<
  any,
  Error,
  {
    taskId: string;
    fileName: string;
    fileType?: string;
    fileSize?: number;
    fileData: string;
  },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => createAttachment(data),
    onSuccess: (_, data) => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      queryClient.invalidateQueries({ queryKey: ['attachments', data.taskId] });
    },
  });
}

export function useGetAttachments(taskId: string): UseMutationResult<any, Error, void, unknown> {
  return useMutation({
    mutationFn: () => getAttachments({ taskId })
  });
}

export function useDeleteAttachment(): UseMutationResult<
  any,
  Error,
  { id: string },
  unknown
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => deleteAttachment(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['attachments'] });
    },
  });
}

/* ----- Activity Log Hook ----- */

export function useGetTaskActivity(taskId: string): UseMutationResult<any, Error, void, unknown> {
  return useMutation({
    mutationFn: () => getTaskActivity({ taskId })
  });
}
