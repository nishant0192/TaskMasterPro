import { useQuery, useMutation, useQueryClient, UseQueryResult, UseMutationResult } from '@tanstack/react-query';
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
  addComment,
  getComments,
  searchTasks,
} from './task';

/**
 * Hook to create a new task.
 */
export function useCreateTask(): UseMutationResult<
  any,
  Error,
  { title: string; description?: string; dueDate?: string; priority?: number },
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

/**
 * Hook to fetch all tasks with optional filtering.
 */
export function useGetTasks(filters?: {
  status?: string;
  priority?: number;
  dueDateFrom?: string;
  dueDateTo?: string;
  search?: string;
  sortBy?: 'dueDate' | 'priority' | 'createdAt';
  sortOrder?: 'asc' | 'desc';
}): UseQueryResult<any, Error> {
  return useQuery({
    queryKey: ['tasks', filters],
    queryFn: () => getTasks(filters),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to fetch a single task by its ID.
 */
export function useGetTaskById(id: string): UseQueryResult<any, Error> {
  return useQuery({
    queryKey: ['task', id],
    queryFn: () => getTask({ id }),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to update a task with optimistic updates.
 */
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

/**
 * Hook to mark a task as completed.
 */
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

/**
 * Hook to delete a task.
 */
export function useDeleteTask(): UseMutationResult<
  any,
  Error,
  { id: string },
  { previousTasks: any }
> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data) => deleteTask(data),
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
  { taskId: string; title: string },
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

export function useGetSubtasks(taskId: string): UseQueryResult<any, Error> {
  return useQuery({
    queryKey: ['subtasks', taskId],
    queryFn: () => getSubtasks({ taskId }),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}

export function useUpdateSubtask(): UseMutationResult<
  any,
  Error,
  { id: string; title?: string; isCompleted?: boolean },
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
      queryClient.invalidateQueries({ queryKey: ['comments'] });
    },
  });
}

export function useGetComments(taskId: string): UseQueryResult<any, Error> {
  return useQuery({
    queryKey: ['comments', taskId],
    queryFn: () => getComments({ taskId }),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}

/* ----- Search Hook ----- */

export function useSearchTasks(query: string): UseQueryResult<any, Error> {
  return useQuery({
    queryKey: ['searchTasks', query],
    queryFn: () => searchTasks({ query }),
    enabled: Boolean(query),
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
  });
}
