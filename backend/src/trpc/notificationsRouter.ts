// src/trpc/notificationsRouter.ts
import { Expo } from 'expo-server-sdk';
import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';

// Create a new Expo SDK client
let expo = new Expo();

// -------------------- Push Token Endpoints --------------------

// Endpoint to register/update the Expo push token for the current user.
const registerPushToken = protectedProcedure
  .input(
    z.object({
      expoPushToken: z.string().min(1, "Expo push token is required"),
    })
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: { expoPushToken: input.expoPushToken },
    });
    logger.success(`Registered push token for user ${ctx.user.email}`);
    return { success: true, user: updatedUser };
  });

// Endpoint to clear the Expo push token (e.g. on logout).
const clearPushToken = protectedProcedure.mutation(async ({ ctx }) => {
  const userId = ctx.user.id;
  const updatedUser = await prisma.user.update({
    where: { id: userId },
    data: { expoPushToken: null },
  });
  logger.success(`Cleared push token for user ${ctx.user.email}`);
  return { success: true, user: updatedUser };
});

// Endpoint to send a test notification using the Expo Server SDK.
const sendTestNotification = protectedProcedure
  .input(
    z.object({
      message: z.string().optional(),
    })
  )
  .mutation(async ({ input, ctx }) => {
    const user = await prisma.user.findUnique({ where: { id: ctx.user.id } });
    if (!user || !user.expoPushToken) {
      throw new Error("User does not have a registered push token.");
    }
    if (!Expo.isExpoPushToken(user.expoPushToken)) {
      throw new Error("Invalid Expo push token.");
    }

    // Prepare the message
    let messages = [];
    messages.push({
      to: user.expoPushToken,
      sound: 'default',
      title: 'Test Notification',
      body: input.message || "This is a test notification!",
      data: { test: true },
    });

    // The Expo push notification service accepts batches. Chunk the messages.
    let chunks = expo.chunkPushNotifications(messages);
    let tickets = [];
    for (let chunk of chunks) {
      let ticketChunk = await expo.sendPushNotificationsAsync(chunk);
      tickets.push(...ticketChunk);
    }
    logger.success(`Test notification sent to ${user.email}`);
    return { success: true, tickets };
  });

// -------------------- Export Router --------------------
export const notificationsRouter = router({
  registerPushToken,
  clearPushToken,
  sendTestNotification,
});

export default notificationsRouter;
