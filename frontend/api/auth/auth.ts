// frontend/api/auth.ts

// Base URL for your backend API; adjust via environment variables if needed.
const API_BASE_URL = process.env.API_BASE_URL || 'http://192.168.29.74:3000';

/**
 * Helper to perform a POST request to a tRPC endpoint.
 * @param path - The tRPC procedure path (e.g. "auth.signup")
 * @param input - The input data for the mutation/query.
 * @param method - The request method ("mutation" or "query").
 * @returns The parsed JSON data from the tRPC response.
 */
async function postRequest(
  path: string,
  input: Record<string, unknown>,
  method: 'mutation' | 'query' = 'mutation'
) {
  const body = {
    id: Date.now(),
    method,
    params: { input }, // send input under params.input
    path,
  };

  const response = await fetch(`${API_BASE_URL}/trpc/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include', // include cookies for secure endpoints
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText);
  }

  // Unwrap the tRPC response (assumes format: { result: { data: ... } })
  const json = await response.json();
  if (!json.result || !json.result.data) {
    throw new Error('Invalid response format');
  }
  return json.result.data;
}

// SIGN UP
export async function signup(email: string, password: string, name?: string) {
  console.log('Signup initiated');
  const response = await postRequest('auth.signup', { email, password, name });
  console.log('Signup response:', response);
  return response;
}

// SIGN IN
export async function signin(email: string, password: string) {
  return postRequest('auth.signin', { email, password });
}

// FORGOT PASSWORD
export async function forgotPassword(email: string) {
  return postRequest('auth.forgotPassword', { email });
}

// REFRESH TOKENS
export async function refreshTokens() {
  return postRequest('auth.refreshTokens', {});
}

// SOCIAL SIGNIN
export async function socialSignIn(provider: string, token: string) {
  return postRequest('auth.socialSignIn', { provider, token });
}

// SOCIAL LOGIN
export async function socialLogin(provider: string, token: string) {
  return postRequest('auth.socialLogin', { provider, token });
}

// SOCIAL TOKENS REFRESH
export async function socialTokensRefresh() {
  return postRequest('auth.socialTokensRefresh', {});
}
