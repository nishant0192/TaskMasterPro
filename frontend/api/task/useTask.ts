import { useQuery, useMutation, UseMutationResult, UseQueryResult } from '@tanstack/react-query';
import { createTask, getTasks, getTask, updateTask, completeTask, deleteTask } from './task';

/**
 * Hook to create a new task.
 */
export function useCreateTask(): UseMutationResult<any, Error, { title: string; description?: string; dueDate?: string; priority?: number }, unknown> {
  return useMutation({
    mutationFn: async (data: { title: string; description?: string; dueDate?: string; priority?: number }) => createTask(data),
  });
}

/**
 * Hook to fetch all tasks.
 */
export function useGetTasks(): UseQueryResult<any, Error> {
  return useQuery({
    queryKey: ['tasks'],
    queryFn: getTasks,
    staleTime: Infinity,
    refetchOnWindowFocus: true,
  });
}

/**
 * Hook to fetch a single task by ID.
 */
export function useGetTaskById(id: string): UseQueryResult<any, Error> {
  return useQuery({
    queryKey: ['task', id],
    queryFn: () => getTask({ id }),
    staleTime: Infinity,
    refetchOnWindowFocus: true,
  });
}

/**
 * Hook to update a task.
 */
export function useUpdateTask(): UseMutationResult<any, Error, { id: string; title?: string; description?: string; dueDate?: string; priority?: number; status?: string; progress?: number; isArchived?: boolean }, unknown> {
  return useMutation({
    mutationFn: async (data) => updateTask(data),
  });
}

/**
 * Hook to mark a task as completed.
 */
export function useCompleteTask(): UseMutationResult<any, Error, { id: string }, unknown> {
  return useMutation({
    mutationFn: async (data) => completeTask(data),
  });
}

/**
 * Hook to delete a task.
 */
export function useDeleteTask(): UseMutationResult<any, Error, { id: string }, unknown> {
  return useMutation({
    mutationFn: async (data) => deleteTask(data),
  });
}
