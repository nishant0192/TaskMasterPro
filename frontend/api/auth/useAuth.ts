import { useMutation, UseMutationResult } from '@tanstack/react-query';
import {
    signup,
    signin,
    forgotPassword,
    refreshTokens,
    socialSignIn,
    socialLogin,
    socialTokensRefresh,
} from './auth';

// Define input types for each mutation
export type SignupInput = { email: string; password: string; name?: string };
export type SigninInput = { email: string; password: string };
export type ForgotPasswordInput = { email: string };
export type SocialInput = { provider: string; token: string };

// Hook for signup
export function useSignup(): UseMutationResult<any, Error, SignupInput, unknown> {
    return useMutation<any, Error, SignupInput, unknown>({
        mutationFn: (data: SignupInput) => {
            return signup(data.email, data.password, data.name);
        }
    });
}
// Hook for signin
export function useSignin(): UseMutationResult<any, Error, SigninInput, unknown> {
    return useMutation<any, Error, SigninInput, unknown>({
        mutationFn: (data: SigninInput) => {
            return signin(data.email, data.password);
        }
    });
}
// Hook for forgot password
export function useForgotPassword(): UseMutationResult<any, Error, ForgotPasswordInput, unknown> {
    return useMutation<any, Error, ForgotPasswordInput, unknown>({
        mutationFn: (data: ForgotPasswordInput) => {
            return forgotPassword(data.email);
        }
    });
}
// Hook for refreshing tokens (no input)
export function useRefreshTokens(): UseMutationResult<any, Error, void, unknown> {
    return useMutation<any, Error, void, unknown>({
        mutationFn: () => refreshTokens()
    });
}
// Hook for social signin
export function useSocialSignIn(): UseMutationResult<any, Error, SocialInput, unknown> {
    return useMutation<any, Error, SocialInput, unknown>({
        mutationFn: (data: SocialInput) => {
            return socialSignIn(data.provider, data.token);
        }
    });
}
// Hook for social login
export function useSocialLogin(): UseMutationResult<any, Error, SocialInput, unknown> {
    return useMutation<any, Error, SocialInput, unknown>({
        mutationFn: (data: SocialInput) => {
            return socialLogin(data.provider, data.token);
        }
    });
}

// Hook for social tokens refresh (no input)
export function useSocialTokensRefresh(): UseMutationResult<any, Error, void, unknown> {
    return useMutation<any, Error, void, unknown>({
        mutationFn: () => socialTokensRefresh()
    });
}
