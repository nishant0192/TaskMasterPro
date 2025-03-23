import { useQuery, useMutation, UseMutationResult, UseQueryResult } from '@tanstack/react-query';
import { getProfile, updateProfile, changePassword, logout } from './user';

/**
 * Hook to fetch the current user's profile.
 * Accepts an optional options parameter to control query behavior (e.g. disable the query).
 */
export function useGetProfile(options?: { enabled?: boolean }): UseQueryResult<any, Error> {
    return useQuery({
        queryKey: ['profile'],
        queryFn: getProfile,
        staleTime: Infinity,
        refetchOnWindowFocus: true,
        enabled: options?.enabled !== false, // Enabled by default
    });
}

/**
 * Hook to update the user's profile.
 */
export function useUpdateProfile(): UseMutationResult<any, Error, { name?: string; profileImage?: string }, unknown> {
    return useMutation({
        mutationFn: async (data: { name?: string; profileImage?: string }) => updateProfile(data),
    });
}

/**
 * Hook to change the user's password.
 */
export function useChangePassword(): UseMutationResult<any, Error, { currentPassword: string; newPassword: string }, unknown> {
    return useMutation({
        mutationFn: async (data: { currentPassword: string; newPassword: string }) => changePassword(data),
    });
}

/**
 * Hook to log out the user.
 */
export function useLogout(): UseMutationResult<any, Error, void, unknown> {
    return useMutation({
        mutationFn: async () => logout(),
    });
}
