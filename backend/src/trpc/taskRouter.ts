import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';

// Modified unwrapInput function: if the value is undefined, return an empty object.
const unwrapInput = (val: unknown) => {
  if (!val) return {};
  logger.debug(`Unwrapped input: ${JSON.stringify(val)}`);
  if (typeof val === 'object' && 'params' in val) {
    return (val as any).params.input ?? {};
  }
  return val;
};


/**
 * Create a new task.
 * Required: title.
 * Optional: description, dueDate, priority.
 * Default: status "TODO", progress 0, isArchived false.
 */
const createTask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        title: z.string().min(1, "Title is required"),
        description: z.string().optional(),
        dueDate: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date().optional()
        ),
        priority: z.number().optional(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Creating task for user ${ctx.user.email}: ${input.title}`);
    const task = await prisma.task.create({
      data: {
        title: input.title,
        description: input.description,
        dueDate: input.dueDate,
        priority: input.priority,
        status: "TODO",
        progress: 0,
        isArchived: false,
        creator: { connect: { id: userId } },
      },
    });
    logger.success(`Task created: ${task.title} (ID: ${task.id})`);
    return task;
  });

/**
 * Get all tasks for the current user.
 */
const getTasks = protectedProcedure.query(async ({ ctx }) => {
  const userId = ctx.user.id;
  logger.debug(`Fetching tasks for user ${ctx.user.email}`);
  const tasks = await prisma.task.findMany({
    where: { creatorId: userId },
    orderBy: { createdAt: 'desc' },
  });
  return { tasks };
});

/**
 * Get a single task by its ID (must belong to the current user).
 */
const getTask = protectedProcedure
  .input((input: unknown) => input) // Accept plain input without validation
  .query(async ({ input, ctx }) => {
    // Your procedure logic here
    const userId = ctx.user.id;
    const taskId = input; // Assuming input has an 'id' property

    // Fetch the task from your database
    const task = await prisma.task.findFirst({
      where: {
        id: taskId!,
        creatorId: userId,
      },
    });

    if (!task) {
      throw new Error('Task not found');
    }

    return { task };
  });



/**
 * Update a task.
 * Allows updating title, description, dueDate, priority, status, progress, and isArchived.
 */
const updateTask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
        title: z.string().optional(),
        description: z.string().optional(),
        dueDate: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date().optional()
        ),
        priority: z.number().optional(),
        status: z.string().optional(),
        progress: z.number().optional(),
        isArchived: z.boolean().optional(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Updating task ${input.id} for user ${ctx.user.email}`);

    // Build the update data with all editable fields.
    const updateData: any = {
      title: input.title,
      description: input.description,
      dueDate: input.dueDate,
      priority: input.priority,
      status: input.status,
      progress: input.progress,
      isArchived: input.isArchived,
    };

    // If the status is "DONE", set progress to 100, mark task as archived, and update completedAt.
    if (input.status === 'DONE') {
      updateData.completedAt = new Date();
      updateData.progress = 100;
      updateData.isArchived = true;
    }

    const result = await prisma.task.updateMany({
      where: {
        id: input.id,
        creatorId: userId,
      },
      data: updateData,
    });

    if (result.count === 0) {
      logger.error(`Task update failed: Task ${input.id} not found or unauthorized`);
      throw new Error('Task update failed');
    }

    logger.success(`Task updated: ${input.id}`);
    return { success: true };
  });



/**
 * Mark a task as completed.
 * Sets status to "DONE", progress to 100, and completedAt to current time.
 */
const completeTask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Completing task ${input.id} for user ${ctx.user.email}`);
    const result = await prisma.task.updateMany({
      where: {
        id: input.id,
        creatorId: userId,
      },
      data: {
        status: 'DONE',
        progress: 100,
        completedAt: new Date(),
        isArchived: true,
      },
    });
    if (result.count === 0) {
      logger.error(`Complete task failed: Task ${input.id} not found or unauthorized`);
      throw new Error('Task not found or cannot be updated');
    }
    logger.success(`Task marked as completed and archived: ${input.id}`);
    return { success: true };
  });


/**
 * Delete a task.
 */
const deleteTask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Deleting task ${input.id} for user ${ctx.user.email}`);
    const result = await prisma.task.deleteMany({
      where: {
        id: input.id,
        creatorId: userId,
      },
    });
    if (result.count === 0) {
      logger.error(`Task deletion failed: Task ${input.id} not found or unauthorized`);
      throw new Error('Task deletion failed');
    }
    logger.success(`Task deleted: ${input.id}`);
    return { success: true };
  });

export const taskRouter = router({
  createTask,
  getTasks,
  getTask,
  updateTask,
  completeTask,
  deleteTask,
});

export default taskRouter;
