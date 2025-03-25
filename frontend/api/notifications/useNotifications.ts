// frontend/app/api/notifications/useNotifications.ts
import { useMutation, useQueryClient, UseMutationResult } from '@tanstack/react-query';
import { registerPushToken, sendTestNotification } from './notifications';

/**
 * Hook to register the Expo push token.
 */
export function useRegisterPushToken(): UseMutationResult<any, Error, { expoPushToken: string }, unknown> {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: { expoPushToken: string }) => registerPushToken(data),
    onSuccess: () => {
      // Optionally invalidate any notifications query if needed.
      queryClient.invalidateQueries({ queryKey: ['notifications'] });
    },
  });
}

/**
 * Hook to send a test notification.
 */
export function useTestNotification(): UseMutationResult<any, Error, { expoPushToken: string }, unknown> {
  return useMutation({
    mutationFn: async (data: { expoPushToken: string }) => sendTestNotification(data),
  });
}
