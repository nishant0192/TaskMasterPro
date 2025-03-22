import { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcrypt';
import { v4 as uuidv4 } from 'uuid';
import { generateAccessToken, generateRefreshToken, verifyRefreshToken } from '@/utils/token';
import { logger } from '@/utils/logger';
import prisma from '@/prisma/client';

/**
 * Async handler middleware that wraps controller functions
 * to automatically pass errors to Express error handling.
 */
const asyncHandler = (
    fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) => (req: Request, res: Response, next: NextFunction) =>
        Promise.resolve(fn(req, res, next)).catch(next);

// Helper: Send refresh token via a secure, HTTP‑only cookie.
const sendRefreshToken = (res: Response, token: string) => {
    res.cookie('refreshToken', token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production', // Use secure flag in production
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days in milliseconds
    });
};

interface AuthController {
    signup: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    signin: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    forgotPassword: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    refreshTokens: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    socialSignIn: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    socialLogin: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    socialTokensRefresh: (req: Request, res: Response, next: NextFunction) => Promise<any>;
}

export const authController: AuthController = {
    // SIGN UP: Create a new user account.
    signup: asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
        const { email, password, name } = req.body;
        const existingUser = await prisma.user.findUnique({ where: { email } });
        if (existingUser) {
            logger.error(`Signup failed: User with email ${email} already exists`);
            return res.status(400).json({ message: 'User already exists' });
        }
        const hashedPassword = await bcrypt.hash(password, 12);
        const newUser = await prisma.user.create({
            data: {
                email,
                name,
                passwordHash: hashedPassword,
            },
        });
        const accessToken = generateAccessToken(newUser);
        const refreshToken = generateRefreshToken(newUser);
        sendRefreshToken(res, refreshToken);
        logger.success(`Signup successful: ${email}`);
        return res.status(201).json({ accessToken });
    }),

    // SIGN IN: Authenticate user with email and password.
    signin: asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
        const { email, password } = req.body;
        const user = await prisma.user.findUnique({ where: { email } });
        if (!user || !user.passwordHash) {
            logger.error(`Signin failed: Invalid credentials for ${email}`);
            return res.status(400).json({ message: 'Invalid credentials' });
        }
        const isValid = await bcrypt.compare(password, user.passwordHash);
        if (!isValid) {
            logger.error(`Signin failed: Invalid credentials for ${email}`);
            return res.status(400).json({ message: 'Invalid credentials' });
        }
        const accessToken = generateAccessToken(user);
        const refreshToken = generateRefreshToken(user);
        sendRefreshToken(res, refreshToken);
        logger.success(`Signin successful: ${email}`);
        return res.status(200).json({ accessToken });
    }),

    // FORGOT PASSWORD: Initiate password reset process.
    forgotPassword: asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
        const { email } = req.body;
        const user = await prisma.user.findUnique({ where: { email } });
        if (!user) {
            logger.error(`ForgotPassword: User not found for ${email}`);
            return res.status(400).json({ message: 'User not found' });
        }
        const resetToken = uuidv4();
        // Update user record with reset token and expiration (ensure these fields exist in your schema)
        await prisma.user.update({
            where: { id: user.id },
            data: {
                resetPasswordToken: resetToken,
                resetPasswordExpires: new Date(Date.now() + 3600000), // 1 hour expiration
            },
        });
        // TODO: Integrate with an email service to send the reset token/link securely.
        logger.info(`ForgotPassword: Reset token generated for ${email}`);
        return res.status(200).json({ message: 'Password reset email sent' });
    }),

    // REFRESH TOKENS: Issue new access and refresh tokens using a valid refresh token.
    refreshTokens: asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
        const refreshToken = req.cookies.refreshToken;
        if (!refreshToken) {
            logger.error('RefreshTokens: Refresh token missing');
            return res.status(401).json({ message: 'Refresh token missing' });
        }
        let payload: any;
        try {
            payload = verifyRefreshToken(refreshToken);
        } catch (err) {
            logger.error('RefreshTokens: Invalid refresh token', err);
            return res.status(401).json({ message: 'Invalid refresh token' });
        }
        const user = await prisma.user.findUnique({ where: { id: payload.id } });
        if (!user) {
            logger.error(`RefreshTokens: User not found for ID ${payload.id}`);
            return res.status(401).json({ message: 'User not found' });
        }
        const newAccessToken = generateAccessToken(user);
        const newRefreshToken = generateRefreshToken(user);
        sendRefreshToken(res, newRefreshToken);
        logger.success(`RefreshTokens: Tokens refreshed for ${user.email}`);
        return res.status(200).json({ accessToken: newAccessToken });
    }),

    // SOCIAL SIGNIN / LOGIN: Handle social authentication.
    socialSignIn: asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
        const { provider, token } = req.body;
        // TODO: Verify the provided social token with the provider's API (Google, Facebook, etc.)
        // For demonstration, assume the verification returns the following:
        const socialUser = {
            email: 'social@example.com',
            name: 'Social User',
            providerId: 'social-unique-id',
        };
        let user = await prisma.user.findUnique({ where: { email: socialUser.email } });
        if (!user) {
            // Create a new social user.
            user = await prisma.user.create({
                data: {
                    email: socialUser.email,
                    name: socialUser.name,
                    oauthProvider: provider,
                    oauthProviderId: socialUser.providerId,
                },
            });
        }
        const accessToken = generateAccessToken(user);
        const refreshToken = generateRefreshToken(user);
        sendRefreshToken(res, refreshToken);
        logger.success(`SocialSignIn: Social user ${socialUser.email} signed in`);
        return res.status(200).json({ accessToken });
    }),

    // SOCIAL LOGIN: Alias for socialSignIn (or differentiate if needed).
    socialLogin: asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
        return authController.socialSignIn(req, res, next);
    }),

    // SOCIAL TOKENS REFRESH: Refresh tokens for social login users (reuse normal refresh logic).
    socialTokensRefresh: asyncHandler(async (req: Request, res: Response, next: NextFunction) => {
        return authController.refreshTokens(req, res, next);
    }),
};

export default authController;
