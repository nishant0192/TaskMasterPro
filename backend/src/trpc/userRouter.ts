import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import bcrypt from 'bcrypt';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';

export const userRouter = router({
  /**
   * Get the current user's profile.
   * Returns only safe fields from the user record.
   */
  getProfile: protectedProcedure.query(async ({ ctx }) => {
    const userId = ctx.user.id;
    const user = await ctx.prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        name: true,
        profileImage: true,
        isActive: true,
        createdAt: true,
        updatedAt: true,
        lastLogin: true,
      },
    });
    if (!user) {
      logger.error(`getProfile: User not found for id ${userId}`);
      throw new Error('User not found');
    }
    logger.info(`getProfile: Retrieved profile for ${user.email}`);
    return { user };
  }),

  /**
   * Update the current user's profile.
   * Accepts optional name and profileImage.
   */
  updateProfile: protectedProcedure
    .input(
      z.object({
        name: z.string().optional(),
        profileImage: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.user.id;
      const updatedUser = await ctx.prisma.user.update({
        where: { id: userId },
        data: { ...input },
        select: {
          id: true,
          email: true,
          name: true,
          profileImage: true,
          isActive: true,
          createdAt: true,
          updatedAt: true,
          lastLogin: true,
        },
      });
      logger.success(`updateProfile: Updated profile for ${updatedUser.email}`);
      return { user: updatedUser };
    }),

  /**
   * Change the current user's password.
   * Requires the current password and a new password.
   */
  changePassword: protectedProcedure
    .input(
      z.object({
        currentPassword: z.string(),
        newPassword: z.string().min(6),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const userId = ctx.user.id;
      const userWithHash = await ctx.prisma.user.findUnique({
        where: { id: userId },
      });
      if (!userWithHash || !userWithHash.passwordHash) {
        logger.error(`changePassword: User not found or cannot change password for id ${userId}`);
        throw new Error('User not found or cannot change password');
      }
      const isValid = await bcrypt.compare(input.currentPassword, userWithHash.passwordHash);
      if (!isValid) {
        logger.error(`changePassword: Invalid current password for user ${userId}`);
        throw new Error('Invalid current password');
      }
      const newHash = await bcrypt.hash(input.newPassword, 12);
      const updatedUser = await ctx.prisma.user.update({
        where: { id: userId },
        data: { passwordHash: newHash },
        select: {
          id: true,
          email: true,
          name: true,
        },
      });
      logger.success(`changePassword: Password updated for ${updatedUser.email}`);
      return { user: updatedUser };
    }),

  /**
   * Logout the current user by clearing the refresh token cookie.
   */
  logout: protectedProcedure.mutation(async ({ ctx }) => {
    ctx.res.clearCookie('refreshToken', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
    });
    logger.info(`logout: User ${ctx.user.email} logged out`);
    return { message: 'Logged out successfully' };
  }),
});

export default userRouter;
