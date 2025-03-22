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
import { useAuthStore } from '@/store/authStore';

// Define input types for each mutation
export type SignupInput = { email: string; password: string; name?: string };
export type SigninInput = { email: string; password: string };
export type ForgotPasswordInput = { email: string };
export type SocialInput = { provider: string; token: string };

// Hook for signup: after successful signup, store the user data & access token.
export function useSignup(): UseMutationResult<any, Error, SignupInput, unknown> {
  const setUser = useAuthStore((state) => state.setUser);
  return useMutation({
    mutationFn: async (data: SignupInput) => {
      const response = await signup(data.email, data.password, data.name);
      // Expecting response: { accessToken, user }
      if (response && response.user && response.accessToken) {
        setUser(response.user, response.accessToken);
      }
      return response;
    },
  });
}

// Hook for signin: after successful signin, store the user data & access token.
export function useSignin(): UseMutationResult<any, Error, SigninInput, unknown> {
  const setUser = useAuthStore((state) => state.setUser);
  return useMutation({
    mutationFn: async (data: SigninInput) => {
      const response = await signin(data.email, data.password);
      if (response && response.user && response.accessToken) {
        setUser(response.user, response.accessToken);
      }
      return response;
    },
  });
}

// Hook for forgot password
export function useForgotPassword(): UseMutationResult<any, Error, ForgotPasswordInput, unknown> {
  return useMutation({
    mutationFn: async (data: ForgotPasswordInput) => forgotPassword(data.email),
  });
}

// Hook for refreshing tokens (no input)
export function useRefreshTokens(): UseMutationResult<any, Error, void, unknown> {
  const updateAccessToken = useAuthStore((state) => state.updateAccessToken);
  return useMutation({
    mutationFn: async () => {
      const response = await refreshTokens();
      if (response && response.accessToken) {
        updateAccessToken(response.accessToken);
      }
      return response;
    },
  });
}

// Hook for social signin
export function useSocialSignIn(): UseMutationResult<any, Error, SocialInput, unknown> {
  const setUser = useAuthStore((state) => state.setUser);
  return useMutation({
    mutationFn: async (data: SocialInput) => {
      const response = await socialSignIn(data.provider, data.token);
      if (response && response.user && response.accessToken) {
        setUser(response.user, response.accessToken);
      }
      return response;
    },
  });
}

// Hook for social login (alias for socialSignIn)
export function useSocialLogin(): UseMutationResult<any, Error, SocialInput, unknown> {
  const setUser = useAuthStore((state) => state.setUser);
  return useMutation({
    mutationFn: async (data: SocialInput) => {
      const response = await socialLogin(data.provider, data.token);
      if (response && response.user && response.accessToken) {
        setUser(response.user, response.accessToken);
      }
      return response;
    },
  });
}

// Hook for social tokens refresh (no input)
export function useSocialTokensRefresh(): UseMutationResult<any, Error, void, unknown> {
  const updateAccessToken = useAuthStore((state) => state.updateAccessToken);
  return useMutation({
    mutationFn: async () => {
      const response = await socialTokensRefresh();
      if (response && response.accessToken) {
        updateAccessToken(response.accessToken);
      }
      return response;
    },
  });
}
