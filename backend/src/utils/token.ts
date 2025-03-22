import jwt, { SignOptions, Secret } from 'jsonwebtoken';

const JWT_ACCESS_SECRET: Secret = process.env.JWT_ACCESS_SECRET as string || "asdasdas";
const JWT_REFRESH_SECRET: Secret = process.env.JWT_REFRESH_SECRET as string || "asdasdas";
const ACCESS_TOKEN_EXPIRES_IN = (process.env.ACCESS_TOKEN_EXPIRES_IN || '15m') as jwt.SignOptions['expiresIn'];
const REFRESH_TOKEN_EXPIRES_IN = (process.env.REFRESH_TOKEN_EXPIRES_IN || '7d') as jwt.SignOptions['expiresIn'];

/**
 * Generates an access token signed with the JWT_ACCESS_SECRET.
 * @param user - An object containing at least an id and email.
 * @returns A signed JWT access token.
 */
export const generateAccessToken = (user: { id: string; email: string }): string => {
    const options: SignOptions = { expiresIn: ACCESS_TOKEN_EXPIRES_IN };
    return jwt.sign({ id: user.id, email: user.email }, JWT_ACCESS_SECRET, options);
};

/**
 * Generates a refresh token signed with the JWT_REFRESH_SECRET.
 * Now includes the refresh token version in the payload.
 * @param user - An object containing at least an id, email and refreshTokenVersion.
 * @returns A signed JWT refresh token.
 */
export const generateRefreshToken = (user: { id: string; email: string; refreshTokenVersion: number }): string => {
    const options: SignOptions = { expiresIn: REFRESH_TOKEN_EXPIRES_IN };
    return jwt.sign(
      { id: user.id, email: user.email, refreshTokenVersion: user.refreshTokenVersion },
      JWT_REFRESH_SECRET,
      options
    );
};

/**
 * Verifies the provided access token.
 * @param token - The JWT access token.
 * @returns The decoded token payload if valid.
 */
export const verifyAccessToken = (token: string): any => {
    return jwt.verify(token, JWT_ACCESS_SECRET);
};

/**
 * Verifies the provided refresh token.
 * @param token - The JWT refresh token.
 * @returns The decoded token payload if valid.
 */
export const verifyRefreshToken = (token: string): any => {
    return jwt.verify(token, JWT_REFRESH_SECRET);
};
