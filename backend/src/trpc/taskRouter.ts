// src/trpc/taskRouter.ts
import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';

// Helper to unwrap input
const unwrapInput = (val: unknown) => {
  if (!val) return {};
  if (typeof val === 'object' && 'params' in val) {
    return (val as any).params.input ?? {};
  }
  return val;
};

/**
 * Create a new task.
 * Required: title.
 * Optional: description, dueDate, priority, subtasks, reminderAt.
 * Defaults: status "TODO", progress 0, isArchived false.
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
        reminderAt: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date().optional()
        ),
        subtasks: z.array(z.object({
          title: z.string().min(1, "Subtask title is required")
        })).optional(),
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
        reminderAt: input.reminderAt,
        status: "TODO",
        progress: 0,
        isArchived: false,
        creator: { connect: { id: userId } },
        subtasks: input.subtasks ? { create: input.subtasks } : undefined,
      },
    });
    logger.success(`Task created: ${task.title} (ID: ${task.id})`);
    return task;
  });

/**
 * Get all tasks for the current user with optional filtering and sorting.
 */
const getTasksInput = z
  .object({
    status: z.string().optional(),
    priority: z.number().optional(),
    dueDateFrom: z.string().optional(),
    dueDateTo: z.string().optional(),
    search: z.string().optional(),
    sortBy: z.string().optional(),
    sortOrder: z.enum(['asc', 'desc']).optional(),
  })
  .default({});

const getTasks = protectedProcedure
  .input((input: unknown) => input ?? {}) // Use raw input (default to empty object)
  .query(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Fetching tasks for user ${ctx.user.email} with filters: ${JSON.stringify(input)}`);

    // Cast input to any
    const rawInput = input as any;
    const where: any = { creatorId: userId };

    if (rawInput) {
      if (rawInput.status) where.status = rawInput.status;
      if (rawInput.priority !== undefined) where.priority = rawInput.priority;

      if (rawInput.dueDateFrom || rawInput.dueDateTo) {
        where.dueDate = {};
        if (rawInput.dueDateFrom) where.dueDate.gte = new Date(rawInput.dueDateFrom);
        if (rawInput.dueDateTo) where.dueDate.lte = new Date(rawInput.dueDateTo);
      }

      if (rawInput.search) {
        where.OR = [
          { title: { contains: rawInput.search, mode: 'insensitive' } },
          { description: { contains: rawInput.search, mode: 'insensitive' } },
        ];
      }
    }

    let orderBy: any = { createdAt: 'desc' };
    if (rawInput?.sortBy) {
      orderBy = { [rawInput.sortBy]: rawInput.sortOrder || 'asc' };
    }

    const tasks = await prisma.task.findMany({
      where,
      orderBy,
    });

    return { tasks };
  });


/**
 * Get a single task by its ID.
 */
const getTask = protectedProcedure
  .input((input: unknown) => input) // Accept plain input without validation
  .query(async ({ input, ctx }) => {
    // Your procedure logic here
    const userId = ctx.user.id;
    const taskId = input; // Assuming input has an 'id' property
    logger.debug(`Fetching task ${taskId} for user ${ctx.user.email}`);

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
 * Allows updating title, description, dueDate, priority, status, progress, isArchived, and reminderAt.
 */
const updateTask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
        title: z.string().optional(),
        description: z.string().optional(),
        dueDate: z.preprocess((val) => (typeof val === 'string' ? new Date(val) : val), z.date().optional()),
        priority: z.number().optional(),
        status: z.string().optional(),
        progress: z.number().optional(),
        isArchived: z.boolean().optional(),
        reminderAt: z.preprocess((val) => (typeof val === 'string' ? new Date(val) : val), z.date().optional()),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Updating task ${input.id} for user ${ctx.user.email}`);
    const updateData: any = {
      title: input.title,
      description: input.description,
      dueDate: input.dueDate,
      priority: input.priority,
      status: input.status,
      progress: input.progress,
      isArchived: input.isArchived,
      reminderAt: input.reminderAt,
    };

    if (input.status === 'DONE') {
      updateData.completedAt = new Date();
      updateData.progress = 100;
      updateData.isArchived = true;
    }

    const result = await prisma.task.updateMany({
      where: { id: input.id, creatorId: userId },
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
 * Sets status to "DONE", progress to 100, completedAt to current time, and archives the task.
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
      where: { id: input.id, creatorId: userId },
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
      where: { id: input.id, creatorId: userId },
    });
    if (result.count === 0) {
      logger.error(`Task deletion failed: Task ${input.id} not found or unauthorized`);
      throw new Error('Task deletion failed');
    }
    logger.success(`Task deleted: ${input.id}`);
    return { success: true };
  });

/**
 * Create a subtask for a given task.
 */
const createSubtask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        taskId: z.string(),
        title: z.string().min(1, "Subtask title is required"),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const subtask = await prisma.subtask.create({
      data: {
        title: input.title,
        task: { connect: { id: input.taskId } },
      },
    });
    logger.success(`Subtask created: ${subtask.title} for task ${input.taskId}`);
    return { subtask };
  });

/**
 * Get all subtasks for a given task.
 */
const getSubtasks = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        taskId: z.string(),
      })
    )
  )
  .query(async ({ input, ctx }) => {
    const subtasks = await prisma.subtask.findMany({
      where: { taskId: input.taskId },
      orderBy: { createdAt: 'asc' },
    });
    return { subtasks };
  });

/**
 * Update a subtask.
 */
const updateSubtask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
        title: z.string().optional(),
        isCompleted: z.boolean().optional(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const result = await prisma.subtask.update({
      where: { id: input.id },
      data: {
        title: input.title,
        isCompleted: input.isCompleted,
      },
    });
    logger.success(`Subtask updated: ${result.id}`);
    return { success: true, subtask: result };
  });

/**
 * Delete a subtask.
 */
const deleteSubtask = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const result = await prisma.subtask.delete({
      where: { id: input.id },
    });
    logger.success(`Subtask deleted: ${input.id}`);
    return { success: true };
  });

/**
 * Set or update a task's reminder.
 */
const setReminder = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
        reminderAt: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date()
        ),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const result = await prisma.task.updateMany({
      where: { id: input.id, creatorId: userId },
      data: { reminderAt: input.reminderAt },
    });
    if (result.count === 0) {
      logger.error(`Setting reminder failed for task ${input.id}`);
      throw new Error('Setting reminder failed');
    }
    logger.success(`Reminder set for task ${input.id} at ${input.reminderAt}`);
    return { success: true };
  });

/**
 * Search tasks by keyword.
 */
const searchTasks = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        keyword: z.string().min(1, "Keyword is required"),
      })
    )
  )
  .query(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const tasks = await prisma.task.findMany({
      where: {
        creatorId: userId,
        OR: [
          { title: { contains: input.keyword, mode: 'insensitive' } },
          { description: { contains: input.keyword, mode: 'insensitive' } },
        ],
      },
      orderBy: { createdAt: 'desc' },
    });
    return { tasks };
  });

/**
 * Add a comment to a task.
 */
const addComment = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        taskId: z.string(),
        content: z.string().min(1, "Comment cannot be empty"),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const comment = await prisma.comment.create({
      data: {
        content: input.content,
        task: { connect: { id: input.taskId } },
        author: { connect: { id: userId } },
      },
    });
    logger.success(`Comment added to task ${input.taskId} by user ${ctx.user.email}`);
    return { comment };
  });

/**
 * Get comments for a task.
 */
const getComments = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        taskId: z.string(),
      })
    )
  )
  .query(async ({ input, ctx }) => {
    const comments = await prisma.comment.findMany({
      where: { taskId: input.taskId },
      orderBy: { createdAt: 'asc' },
      include: { author: true },
    });
    return { comments };
  });

/**
 * Get activity logs for a task.
 */
const getTaskActivity = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        taskId: z.string(),
      })
    )
  )
  .query(async ({ input, ctx }) => {
    const logs = await prisma.auditLog.findMany({
      where: {
        taskId: input.taskId,
      },
      orderBy: { timestamp: 'desc' },
    });
    return { logs };
  });

export const taskRouter = router({
  createTask,
  getTasks,
  getTask,
  updateTask,
  completeTask,
  deleteTask,
  createSubtask,
  getSubtasks,
  updateSubtask,
  deleteSubtask,
  setReminder,
  searchTasks,
  addComment,
  getComments,
  getTaskActivity,
});

export default taskRouter;
