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
  return { accessToken };
}

// A helper preprocessor function to unwrap input from params.input if present.
const unwrapInput = (val: unknown) => {
  if (val && typeof val === 'object' && 'params' in val) {
    const params = (val as any).params;
    if (params && typeof params === 'object' && 'input' in params) {
      return params.input;
    }
  }
  return val;
};

export const authRouter = router({
  // SIGN UP: Create a new user account with Zod validation.
  signup: publicProcedure
    .input(
      z.preprocess(unwrapInput,
        z.object({
          email: z.string().email(),
          password: z.string().min(6),
          name: z.string().optional(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      console.log("Processed signup input:", input);
      logger.debug(`Signup initiated for: ${input.email}`);

      const existingUser = await ctx.prisma.user.findUnique({ where: { email: input.email } });
      if (existingUser) {
        logger.error(`Signup failed: User with email ${input.email} already exists`);
        throw new Error('User already exists');
      }

      const hashedPassword = await bcrypt.hash(input.password, 12);
      const newUser = await ctx.prisma.user.create({
        data: {
          email: input.email,
          name: input.name,
          passwordHash: hashedPassword,
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
      return { accessToken };
    }),

  // SIGN IN: Authenticate user with email and password.
  signin: publicProcedure
    .input(
      z.preprocess(unwrapInput,
        z.object({
          email: z.string().email(),
          password: z.string(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      console.log("Processed signin input:", input);
      const { email, password } = input;
      logger.debug(`Signin initiated for: ${email}`);
      const user = await ctx.prisma.user.findUnique({ where: { email } });
      if (!user || !user.passwordHash) {
        logger.error(`Signin failed: Invalid credentials for ${email}`);
        throw new Error('Invalid credentials');
      }
      const isValid = await bcrypt.compare(password, user.passwordHash);
      if (!isValid) {
        logger.error(`Signin failed: Invalid credentials for ${email}`);
        throw new Error('Invalid credentials');
      }
      const accessToken = generateAccessToken(user);
      const refreshToken = generateRefreshToken(user);
      ctx.res.cookie('refreshToken', refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000,
      });
      logger.success(`Signin successful: ${email}`);
      return { accessToken };
    }),

  // FORGOT PASSWORD: Initiate password reset process.
  forgotPassword: publicProcedure
    .input(
      z.preprocess(unwrapInput,
        z.object({ email: z.string().email() })
      )
    )
    .mutation(async ({ input, ctx }) => {
      console.log("Processed forgotPassword input:", input);
      const { email } = input;
      logger.debug(`ForgotPassword initiated for: ${email}`);
      const user = await ctx.prisma.user.findUnique({ where: { email } });
      if (!user) {
        logger.error(`ForgotPassword: User not found for ${email}`);
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
      logger.info(`ForgotPassword: Reset token generated for ${email}`);
      return { message: 'Password reset email sent' };
    }),

  // REFRESH TOKENS: Issue new access and refresh tokens using a valid refresh token.
  refreshTokens: publicProcedure.mutation(async ({ ctx }) => {
    const refreshToken = ctx.req.cookies.refreshToken;
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
    const user = await ctx.prisma.user.findUnique({ where: { id: payload.id } });
    if (!user) {
      logger.error(`RefreshTokens: User not found for ID ${payload.id}`);
      throw new Error('User not found');
    }
    const newAccessToken = generateAccessToken(user);
    const newRefreshToken = generateRefreshToken(user);
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
      z.preprocess(unwrapInput,
        z.object({
          provider: z.string(),
          token: z.string(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      console.log("Processed socialSignIn input:", input);
      logger.debug(`SocialSignIn initiated with provider: ${input.provider}`);
      return await handleSocialSignIn(input, ctx);
    }),

  // SOCIAL LOGIN: Alias for socialSignIn.
  socialLogin: publicProcedure
    .input(
      z.preprocess(unwrapInput,
        z.object({
          provider: z.string(),
          token: z.string(),
        })
      )
    )
    .mutation(async ({ input, ctx }) => {
      console.log("Processed socialLogin input:", input);
      logger.debug(`SocialLogin initiated with provider: ${input.provider}`);
      return await handleSocialSignIn(input, ctx);
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
    const newAccessToken = generateAccessToken(user);
    const newRefreshToken = generateRefreshToken(user);
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
