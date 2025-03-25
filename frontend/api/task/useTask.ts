import { useQuery, useMutation, useQueryClient, UseQueryResult, UseMutationResult } from '@tanstack/react-query';
import { createTask, getTasks, getTask, updateTask, completeTask, deleteTask } from './task';

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
 * Hook to fetch all tasks with caching.
 */
export function useGetTasks(): UseQueryResult<any, Error> {
  const options = {
    queryKey: ['tasks'],
    queryFn: getTasks,
    staleTime: 5 * 60 * 1000,   // 5 minutes fresh
    cacheTime: 30 * 60 * 1000,  // 30 minutes cached
    refetchOnWindowFocus: false,
  };
  return useQuery(options);
}

/**
 * Hook to fetch a single task by its ID.
 */
export function useGetTaskById(id: string): UseQueryResult<any, Error> {
  const options = {
    queryKey: ['task', id],
    queryFn: () => getTask({ id }),
    staleTime: 5 * 60 * 1000,
    cacheTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
  };
  return useQuery(options);
}

/**
 * Hook to update a task with optimistic updates and query invalidation.
 */
export function useUpdateTask(): UseMutationResult<
  any,
  Error,
  { id: string; title?: string; description?: string; dueDate?: string; priority?: number; status?: string; progress?: number; isArchived?: boolean },
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
 * This sets status to "DONE", progress to 100, completedAt to now, and archives the task.
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
