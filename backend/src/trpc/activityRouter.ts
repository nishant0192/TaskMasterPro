import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';

// Helper to unwrap input from nested tRPC envelope.
const unwrapInput = (val: unknown) => {
  if (!val) return {};
  if (typeof val === 'object' && 'params' in val) {
    return (val as any).params.input ?? {};
  }
  return val;
};

/**
 * Create a new activity log entry.
 * The log can optionally be associated with an entity (e.g., a Task, Attachment, etc.).
 */
const createActivity = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        action: z.string().min(1, "Action is required"),
        description: z.string().optional(),
        entityType: z.string().optional(),
        entityId: z.string().optional(),
        taskId: z.string().optional(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Creating activity log for user ${ctx.user.email}: ${input.action}`);
    
    const log = await prisma.auditLog.create({
      data: {
        action: input.action,
        description: input.description,
        entityType: input.entityType,
        entityId: input.entityId,
        task: input.taskId ? { connect: { id: input.taskId } } : undefined,
        user: { connect: { id: userId } },
      },
    });
    
    logger.success(`Activity log created: ${log.id}`);
    return { log };
  });

/**
 * Fetch activity logs based on optional filters.
 * The logs are fetched for the current user and can be filtered by entityType,
 * entityId, or a related task.
 */
const getActivityLogs = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        entityType: z.string().optional(),
        entityId: z.string().optional(),
        taskId: z.string().optional(),
        skip: z.number().optional(),
        limit: z.number().optional(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Fetching activity logs for user ${ctx.user.email}`);
    
    const where: any = { userId };
    if (input.entityType) where.entityType = input.entityType;
    if (input.entityId) where.entityId = input.entityId;
    if (input.taskId) where.taskId = input.taskId;
    
    const logs = await prisma.auditLog.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      skip: input.skip,
      take: input.limit,
    });
    
    return { logs };
  });

export const activityRouter = router({
  createActivity,
  getActivityLogs,
});

export default activityRouter;
