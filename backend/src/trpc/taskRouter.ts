// src/trpc/taskRouter.ts
import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';
import { uploadFile } from '../utils/upload.js';

// Helper to unwrap input from nested tRPC envelope.
const unwrapInput = (val: unknown) => {
  if (!val) return {};
  if (typeof val === 'object' && 'params' in val) {
    return (val as any).params.input ?? {};
  }
  return val;
};

/* -----------------------------------------------
   TASK PROCEDURES
----------------------------------------------- */

/**
 * Create a new task.
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
 * Get all tasks for the current user with optional filtering.
 * (Now implemented as a mutation.)
 */
const getTasks = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        status: z.string().optional(),
        priority: z.number().optional(),
        dueDateFrom: z.string().optional(),
        dueDateTo: z.string().optional(),
        search: z.string().optional(),
        sortBy: z.enum(['dueDate', 'priority', 'createdAt']).optional(),
        sortOrder: z.enum(['asc', 'desc']).optional(),
      }).default({})
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Fetching tasks for user ${ctx.user.email} with filters: ${JSON.stringify(input)}`);
    const where: any = { creatorId: userId };

    if (input.status) where.status = input.status;
    if (input.priority !== undefined) where.priority = input.priority;
    if (input.dueDateFrom || input.dueDateTo) {
      where.dueDate = {};
      if (input.dueDateFrom) where.dueDate.gte = new Date(input.dueDateFrom);
      if (input.dueDateTo) where.dueDate.lte = new Date(input.dueDateTo);
    }
    if (input.search) {
      where.OR = [
        { title: { contains: input.search, mode: 'insensitive' } },
        { description: { contains: input.search, mode: 'insensitive' } },
      ];
    }

    let orderBy: any = { createdAt: 'desc' };
    if (input.sortBy) {
      orderBy = { [input.sortBy]: input.sortOrder || 'asc' };
    }

    const tasks = await prisma.task.findMany({
      where,
      orderBy,
    });
    logger.success(`Fetched ${tasks.length} tasks for user ${ctx.user.email}`);
    return { tasks };
  });

/**
 * Get a single task by its ID.
 * (Now implemented as a mutation.)
 */
const getTask = protectedProcedure
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
    const taskId = input.id;
    logger.debug(`Fetching task ${taskId} for user ${ctx.user.email}`);

    const task = await prisma.task.findFirst({
      where: {
        id: taskId,
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

/* -----------------------------------------------
   SUBTASK PROCEDURES
----------------------------------------------- */

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
        order: z.number().optional(),
        reminderAt: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date().optional()
        ),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const subtask = await prisma.subtask.create({
      data: {
        title: input.title,
        order: input.order,
        reminderAt: input.reminderAt,
        task: { connect: { id: input.taskId } },
      },
    });
    logger.success(`Subtask created: ${subtask.title} for task ${input.taskId}`);
    return { subtask };
  });

/**
 * Get all subtasks for a given task.
 * (Now using a mutation.)
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
  .mutation(async ({ input, ctx }) => {
    const subtasks = await prisma.subtask.findMany({
      where: { taskId: input.taskId },
      orderBy: { order: 'asc' },
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
        order: z.number().optional(),
        reminderAt: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date().optional()
        ),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const result = await prisma.subtask.update({
      where: { id: input.id },
      data: {
        title: input.title,
        isCompleted: input.isCompleted,
        order: input.order,
        reminderAt: input.reminderAt,
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


/* -----------------------------------------------
   REMINDER PROCEDURE
----------------------------------------------- */

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

/* -----------------------------------------------
   SEARCH PROCEDURE
----------------------------------------------- */

/**
 * Search tasks by keyword in title or description.
 * (Converted to mutation.)
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
  .mutation(async ({ input, ctx }) => {
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

/* -----------------------------------------------
   COMMENT PROCEDURES
----------------------------------------------- */

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
 * (Converted to mutation.)
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
  .mutation(async ({ input, ctx }) => {
    const comments = await prisma.comment.findMany({
      where: { taskId: input.taskId },
      orderBy: { createdAt: 'asc' },
      include: { author: true },
    });
    return { comments };
  });

/* -----------------------------------------------
   ACTIVITY LOG PROCEDURE
----------------------------------------------- */

/**
 * Get audit logs (activity) for a task.
 * (Converted to mutation.)
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
  .mutation(async ({ input, ctx }) => {
    const logs = await prisma.auditLog.findMany({
      where: { taskId: input.taskId },
      orderBy: { timestamp: 'desc' },
    });
    return { logs };
  });

/* -----------------------------------------------
   ATTACHMENT PROCEDURES
----------------------------------------------- */

/**
 * Create a new attachment for a task.
 */
const createAttachment = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        taskId: z.string(),
        fileName: z.string(),
        fileType: z.string().optional(),
        fileSize: z.number().optional(),
        fileData: z.string().min(1, "File data is required"),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    logger.debug(`Creating attachment for task ${input.taskId}: ${input.fileName}`);
    const fileBuffer = Buffer.from(input.fileData, 'base64');
    const fileUrl = await uploadFile(fileBuffer, input.fileName);
    const attachment = await prisma.attachment.create({
      data: {
        fileName: input.fileName,
        fileType: input.fileType,
        fileSize: input.fileSize,
        fileUrl,
        task: { connect: { id: input.taskId } },
        uploadedBy: { connect: { id: ctx.user.id } },
      },
    });
    logger.success(`Attachment created for task ${input.taskId}: ${fileUrl}`);
    return { attachment };
  });

/**
 * Get all attachments for a given task.
 * (Converted to mutation.)
 */
const getAttachments = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        taskId: z.string(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const attachments = await prisma.attachment.findMany({
      where: { taskId: input.taskId },
      orderBy: { uploadedAt: 'asc' },
    });
    return { attachments };
  });

/**
 * Delete an attachment.
 */
const deleteAttachment = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const result = await prisma.attachment.delete({
      where: { id: input.id },
    });
    logger.success(`Attachment deleted: ${input.id}`);
    return { success: true };
  });

/* -----------------------------------------------
   EXPORT ROUTER
----------------------------------------------- */

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
  createAttachment,
  getAttachments,
  deleteAttachment,
});

export default taskRouter;
