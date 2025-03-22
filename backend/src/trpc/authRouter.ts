import { z } from 'zod';
import { router, publicProcedure } from './context.js';
import bcrypt from 'bcrypt';
import { v4 as uuidv4 } from 'uuid';
import { generateAccessToken, generateRefreshToken, verifyRefreshToken } from '../utils/token.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';

/**
 * Helper function to handle social sign in logic.
 * Replace the stubbed verification with actual API calls to your social provider.
 */
async function handleSocialSignIn(
  input: { provider: string; token: string },
  ctx: { req: any; res: any; prisma: any }
) {
  // TODO: Replace this stub with actual social token verification.
  const socialUser = {
    email: 'social@example.com',
    name: 'Social User',
    providerId: 'social-unique-id',
  };

  let user = await ctx.prisma.user.findUnique({ where: { email: socialUser.email } });
  if (!user) {
    user = await ctx.prisma.user.create({
      data: {
        email: socialUser.email,
        name: socialUser.name,
        oauthProvider: input.provider,
        oauthProviderId: socialUser.providerId,
        lastLogin: new Date(),
        refreshTokenVersion: 0,
      },
      select: {
        id: true,
        email: true,
        name: true,
        profileImage: true,
        isActive: true,
        createdAt: true,
        updatedAt: true,
        lastLogin: true,
        refreshTokenVersion: true,
      },
    });
  } else {
    // Update lastLogin on every social sign-in.
    user = await ctx.prisma.user.update({
      where: { id: user.id },
      data: { lastLogin: new Date() },
      select: {
        id: true,
        email: true,
        name: true,
        profileImage: true,
        isActive: true,
        createdAt: true,
        updatedAt: true,
        lastLogin: true,
        refreshTokenVersion: true,
      },
    });
  }
  const accessToken = generateAccessToken(user);
  const refreshToken = generateRefreshToken(user);
  ctx.res.cookie('refreshToken', refreshToken, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    maxAge: 7 * 24 * 60 * 60 * 1000,
  });
  logger.success(`SocialSignIn: Social user ${socialUser.email} signed in`);
  return { accessToken, user };
}

export const authRouter = router({
  // SIGN UP: Create a new user account.
  signup: publicProcedure
    .input(
      z.preprocess(
        (val) => (val ? (val as any).params?.input ?? val : val),
        z.object({
          email: z.string().email(),
          password: z.string().min(6),
          name: z.string().optional(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      logger.debug(`Signup initiated for: ${input.email}`);
      const existingUser = await ctx.prisma.user.findUnique({ where: { email: input.email } });
      if (existingUser) {
        logger.error(`Signup failed: User with email ${input.email} already exists`);
        throw new Error('User already exists');
      }
      const hashedPassword = await bcrypt.hash(input.password, 12);
      // Create new user and update lastLogin.
      const newUser = await ctx.prisma.user.create({
        data: {
          email: input.email,
          name: input.name,
          passwordHash: hashedPassword,
          lastLogin: new Date(),
          refreshTokenVersion: 0,
        },
        select: {
          id: true,
          email: true,
          name: true,
          profileImage: true,
          isActive: true,
          createdAt: true,
          updatedAt: true,
          lastLogin: true,
          refreshTokenVersion: true,
        },
      });
      const accessToken = generateAccessToken(newUser);
      const refreshToken = generateRefreshToken(newUser);
      ctx.res.cookie('refreshToken', refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000,
      });
      logger.success(`Signup successful: ${input.email}`);
      return { accessToken, user: newUser };
    }),

  // SIGN IN: Authenticate user with email and password.
  signin: publicProcedure
    .input(
      z.preprocess(
        (val) => (val ? (val as any).params?.input ?? val : val),
        z.object({
          email: z.string().email(),
          password: z.string(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      logger.debug(`Signin initiated for: ${input.email}`);
      // Retrieve user along with passwordHash for verification.
      const userWithHash = await ctx.prisma.user.findUnique({ where: { email: input.email } });
      if (!userWithHash || !userWithHash.passwordHash) {
        logger.error(`Signin failed: Invalid credentials for ${input.email}`);
        throw new Error('Invalid credentials');
      }
      const isValid = await bcrypt.compare(input.password, userWithHash.passwordHash);
      if (!isValid) {
        logger.error(`Signin failed: Invalid credentials for ${input.email}`);
        throw new Error('Invalid credentials');
      }
      // Update lastLogin and get a filtered user object.
      const user = await ctx.prisma.user.update({
        where: { id: userWithHash.id },
        data: { lastLogin: new Date() },
        select: {
          id: true,
          email: true,
          name: true,
          profileImage: true,
          isActive: true,
          createdAt: true,
          updatedAt: true,
          lastLogin: true,
          refreshTokenVersion: true,
        },
      });
      const accessToken = generateAccessToken(user);
      const refreshToken = generateRefreshToken(user);
      ctx.res.cookie('refreshToken', refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000,
      });
      logger.success(`Signin successful: ${input.email}`);
      return { accessToken, user };
    }),

  // FORGOT PASSWORD: Initiate password reset process.
  forgotPassword: publicProcedure
    .input(
      z.preprocess(
        (val) => (val ? (val as any).params?.input ?? val : val),
        z.object({ email: z.string().email() })
      )
    )
    .mutation(async ({ input, ctx }) => {
      logger.debug(`ForgotPassword initiated for: ${input.email}`);
      const user = await ctx.prisma.user.findUnique({ where: { email: input.email } });
      if (!user) {
        logger.error(`ForgotPassword: User not found for ${input.email}`);
        throw new Error('User not found');
      }
      const resetToken = uuidv4();
      await ctx.prisma.user.update({
        where: { id: user.id },
        data: {
          resetPasswordToken: resetToken,
          resetPasswordExpires: new Date(Date.now() + 3600000), // 1 hour expiration
        },
      });
      // TODO: Send reset token via a secure email service.
      logger.info(`ForgotPassword: Reset token generated for ${input.email}`);
      return { message: 'Password reset email sent' };
    }),

  // REFRESH TOKENS: Issue new access and refresh tokens using a valid refresh token,
  // and perform refresh token rotation.
  refreshTokens: publicProcedure.mutation(async ({ ctx }) => {
    const refreshToken = ctx.req.cookies.refreshToken;
    logger.debug('RefreshTokens: Received refresh token');
    if (!refreshToken) {
      logger.error('RefreshTokens: Refresh token missing');
      throw new Error('Refresh token missing');
    }
    let payload: any;
    try {
      payload = verifyRefreshToken(refreshToken);
    } catch (err) {
      logger.error('RefreshTokens: Invalid refresh token', err);
      throw new Error('Invalid refresh token');
    }
    // Retrieve the user.
    const user = await ctx.prisma.user.findUnique({ where: { id: payload.id } });
    if (!user) {
      logger.error(`RefreshTokens: User not found for ID ${payload.id}`);
      throw new Error('User not found');
    }
    // Check refresh token version.
    if (user.refreshTokenVersion !== payload.refreshTokenVersion) {
      logger.error(`RefreshTokens: Refresh token version mismatch for ${user.email}`);
      throw new Error('Invalid refresh token version');
    }
    // Rotate the refresh token by incrementing the version.
    const updatedUser = await ctx.prisma.user.update({
      where: { id: user.id },
      data: {
        refreshTokenVersion: user.refreshTokenVersion + 1,
        lastLogin: new Date(), // Optionally update lastLogin here
      },
      select: {
        id: true,
        email: true,
        name: true,
        profileImage: true,
        isActive: true,
        createdAt: true,
        updatedAt: true,
        lastLogin: true,
        refreshTokenVersion: true,
      },
    });
    const newAccessToken = generateAccessToken(updatedUser);
    const newRefreshToken = generateRefreshToken(updatedUser);
    ctx.res.cookie('refreshToken', newRefreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000,
    });
    logger.success(`RefreshTokens: Tokens refreshed for ${user.email}`);
    return { accessToken: newAccessToken };
  }),

  // SOCIAL SIGNIN: Handle social authentication.
  socialSignIn: publicProcedure
    .input(
      z.preprocess(
        (val) => (val ? (val as any).params?.input ?? val : val),
        z.object({
          provider: z.string(),
          token: z.string(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      logger.debug(`SocialSignIn initiated with provider: ${input.provider}`);
      const result = await handleSocialSignIn(input, ctx);
      // Update lastLogin for social user.
      await ctx.prisma.user.update({
        where: { id: result.user.id },
        data: { lastLogin: new Date() },
      });
      return result;
    }),

  // SOCIAL LOGIN: Alias for socialSignIn.
  socialLogin: publicProcedure
    .input(
      z.preprocess(
        (val) => (val ? (val as any).params?.input ?? val : val),
        z.object({
          provider: z.string(),
          token: z.string(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      logger.debug(`SocialLogin initiated with provider: ${input.provider}`);
      const result = await handleSocialSignIn(input, ctx);
      // Update lastLogin for social user.
      await ctx.prisma.user.update({
        where: { id: result.user.id },
        data: { lastLogin: new Date() },
      });
      return result;
    }),

  // SOCIAL TOKENS REFRESH: Refresh tokens for social login users.
  socialTokensRefresh: publicProcedure.mutation(async ({ ctx }) => {
    const refreshToken = ctx.req.cookies.refreshToken;
    if (!refreshToken) {
      logger.error('SocialTokensRefresh: Refresh token missing');
      throw new Error('Refresh token missing');
    }
    let payload: any;
    try {
      payload = verifyRefreshToken(refreshToken);
    } catch (err) {
      logger.error('SocialTokensRefresh: Invalid refresh token', err);
      throw new Error('Invalid refresh token');
    }
    const user = await ctx.prisma.user.findUnique({ where: { id: payload.id } });
    if (!user) {
      logger.error(`SocialTokensRefresh: User not found for ID ${payload.id}`);
      throw new Error('User not found');
    }
    if (user.refreshTokenVersion !== payload.refreshTokenVersion) {
      logger.error(`SocialTokensRefresh: Refresh token version mismatch for ${user.email}`);
      throw new Error('Invalid refresh token version');
    }
    const updatedUser = await ctx.prisma.user.update({
      where: { id: user.id },
      data: { refreshTokenVersion: user.refreshTokenVersion + 1 },
      select: {
        id: true,
        email: true,
        name: true,
        profileImage: true,
        isActive: true,
        createdAt: true,
        updatedAt: true,
        lastLogin: true,
        refreshTokenVersion: true,
      },
    });
    const newAccessToken = generateAccessToken(updatedUser);
    const newRefreshToken = generateRefreshToken(updatedUser);
    ctx.res.cookie('refreshToken', newRefreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000,
    });
    logger.success(`SocialTokensRefresh: Tokens refreshed for ${user.email}`);
    return { accessToken: newAccessToken };
  }),
});

export default authRouter;
