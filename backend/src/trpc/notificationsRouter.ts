// src/trpc/notificationsRouter.ts
import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';

// -------------------- Helper --------------------
const unwrapInput = (val: unknown) => {
  if (!val) return {};
  if (typeof val === 'object' && 'params' in val) {
    return (val as any).params.input ?? {};
  }
  return val;
};

// -------------------- Push Token Endpoints --------------------

// Endpoint to register/update the Expo push token for the current user.
const registerPushToken = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        expoPushToken: z.string().min(1, "Expo push token is required"),
      })
    )
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

// -------------------- Notification Sending --------------------

// Endpoint to send a test notification using the HTTP/2 API.
const sendTestNotification = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        message: z.string().optional(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    // Find the user and their registered Expo push token.
    const user = await prisma.user.findUnique({ where: { id: ctx.user.id } });
    if (!user || !user.expoPushToken) {
      throw new Error("User does not have a registered push token.");
    }
    const token = user.expoPushToken;
    // Prepare the push notification message.
    const messagePayload = {
      to: token,
      sound: 'default',
      title: 'Test Notification',
      body: input.message || "This is a test notification!",
      data: { test: true },
    };

    // Send a POST request directly to the HTTP/2 API endpoint.
    const response = await fetch("https://exp.host/--/api/v2/push/send", {
      method: "POST",
      headers: {
        "host": "exp.host",
        "accept": "application/json",
        "accept-encoding": "gzip, deflate",
        "content-type": "application/json",
      },
      body: JSON.stringify(messagePayload),
    });

    const dataResp = await response.json();
    logger.success(`Test notification sent to ${user.email}`);
    return { success: true, response: dataResp };
  });

// -------------------- Export Router --------------------
export const notificationsRouter = router({
  registerPushToken,
  clearPushToken,
  sendTestNotification,
});

export default notificationsRouter;
