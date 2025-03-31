// src/trpc/taskRouter.ts
import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';
import { uploadFile } from '../utils/upload.js';
import { scheduleTaskNotification } from '../utils/scheduleNotification.js';
import { v2 as cloudinary } from 'cloudinary';

// Helper to unwrap input from nested tRPC envelope.
const unwrapInput = (val: unknown) => {
  if (!val) return {};
  if (typeof val === 'object' && 'params' in val) {
    return (val as any).params.input ?? {};
  }
  return val;
};

/**
 * Helper function to compute differences between original and updated fields.
 * It accepts two objects and returns a summary string of the differences.
 */
function computeDiff(original: any, updates: any): string {
  const diffs: string[] = [];
  for (const key in updates) {
    if (updates.hasOwnProperty(key)) {
      const newValue = updates[key];
      const oldValue = original[key];
      // For date fields, compare their ISO strings.
      const oldValStr = oldValue instanceof Date ? oldValue.toISOString() : oldValue;
      const newValStr = newValue instanceof Date ? newValue.toISOString() : newValue;
      if (newValue !== undefined && newValue !== null && newValStr !== oldValStr) {
        diffs.push(`${key} changed from '${oldValStr}' to '${newValStr}'`);
      }
    }
  }
  return diffs.length ? diffs.join('; ') : 'No changes detected';
}

/**
 * Helper function to create an audit log entry.
 */
async function createAuditLog({
  action,
  description,
  entityType,
  entityId,
  ctx,
}: {
  action: string;
  description: string;
  entityType?: string;
  entityId?: string;
  ctx: any;
}) {
  await prisma.auditLog.create({
    data: {
      action,
      description,
      entityType,
      entityId,
      user: { connect: { id: ctx.user.id } },
      task: entityType === 'Task' && entityId ? { connect: { id: entityId } } : undefined,
    },
  });
}

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
        subtasks: z.array(
          z.object({
            title: z.string().min(1, "Subtask title is required"),
            // Each subtask may include an optional reminderAt.
            reminderAt: z.preprocess(
              (val) => (typeof val === 'string' ? new Date(val) : val),
              z.date().optional()
            ),
          })
        ).optional(),
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
    await createAuditLog({
      action: 'TASK_CREATED',
      description: `Task created with title '${task.title}'`,
      entityType: 'Task',
      entityId: task.id,
      ctx,
    });
    // Schedule notifications for task-level reminders.
    await scheduleTaskNotification(task, 'dueDate');
    await scheduleTaskNotification(task, 'reminderAt');
    return task;
  });

/**
 * Get all tasks for the current user with optional filtering.
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
      where: { id: taskId, creatorId: userId },
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
        dueDate: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date().optional()
        ),
        priority: z.number().optional(),
        status: z.string().optional(),
        progress: z.number().optional(),
        isArchived: z.boolean().optional(),
        reminderAt: z.preprocess(
          (val) => (typeof val === 'string' ? new Date(val) : val),
          z.date().optional()
        ),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    logger.debug(`Updating task ${input.id} for user ${ctx.user.email}`);

    // Fetch original task for diff comparison.
    const originalTask = await prisma.task.findFirst({
      where: { id: input.id, creatorId: userId },
    });
    if (!originalTask) {
      logger.error(`Task ${input.id} not found or unauthorized`);
      throw new Error('Task update failed');
    }

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

    const diffDescription = computeDiff(originalTask, updateData);

    const result = await prisma.task.updateMany({
      where: { id: input.id, creatorId: userId },
      data: updateData,
    });

    if (result.count === 0) {
      logger.error(`Task update failed: Task ${input.id} not found or unauthorized`);
      throw new Error('Task update failed');
    }

    logger.success(`Task updated: ${input.id}`);
    await createAuditLog({
      action: 'TASK_UPDATED',
      description: diffDescription,
      entityType: 'Task',
      entityId: input.id,
      ctx,
    });

    // Reschedule notifications if dueDate or reminderAt have changed.
    const updatedTask = await prisma.task.findUnique({ where: { id: input.id } });
    if (updatedTask) {
      await scheduleTaskNotification(updatedTask, 'dueDate');
      await scheduleTaskNotification(updatedTask, 'reminderAt');
    }

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
    await createAuditLog({
      action: 'TASK_COMPLETED',
      description: `Task marked as completed and archived.`,
      entityType: 'Task',
      entityId: input.id,
      ctx,
    });
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

    // First, find the task to ensure it exists and is owned by the user.
    const task = await prisma.task.findUnique({
      where: { id: input.id },
    });
    logger.debug(`Task found: ${task?.id}`);
    if (!task || task.creatorId !== userId) {
      logger.error(`Task deletion failed: Task ${input.id} not found or unauthorized`);
      throw new Error('Task deletion failed');
    }

    // Proceed to delete the task.
    const response = await prisma.task.delete({
      where: { id: input.id },
    });
    logger.debug(`Task deleted: ${response}`);
    logger.success(`Task deleted: ${input.id}`);

    await createAuditLog({
      action: 'TASK_DELETED',
      description: `Task deleted: ${input.id}`,
      entityType: 'Task',
      entityId: input.id,
      ctx,
    });

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
    await createAuditLog({
      action: 'SUBTASK_CREATED',
      description: `Subtask '${subtask.title}' created for task ${input.taskId}`,
      entityType: 'Subtask',
      entityId: subtask.id,
      ctx,
    });
    // If a reminder is set for the subtask, schedule its notification.
    if (subtask.reminderAt) {
      const parentTask = await prisma.task.findUnique({
        where: { id: input.taskId },
        select: { id: true, title: true, creatorId: true },
      });
      if (parentTask) {
        const subtaskReminderObj = {
          id: subtask.id,
          title: `${parentTask.title} - Subtask: ${subtask.title}`,
          creatorId: parentTask.creatorId,
          reminderAt: subtask.reminderAt,
        };
        await scheduleTaskNotification(subtaskReminderObj, 'reminderAt');
      }
    }
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
    const originalSubtask = await prisma.subtask.findUnique({
      where: { id: input.id },
    });
    if (!originalSubtask) throw new Error("Subtask not found");
    const updateData = {
      title: input.title,
      isCompleted: input.isCompleted,
      order: input.order,
      reminderAt: input.reminderAt,
    };
    const diffDescription = computeDiff(originalSubtask, updateData);
    const result = await prisma.subtask.update({
      where: { id: input.id },
      data: updateData,
    });
    logger.success(`Subtask updated: ${result.id}`);
    await createAuditLog({
      action: 'SUBTASK_UPDATED',
      description: diffDescription,
      entityType: 'Subtask',
      entityId: input.id,
      ctx,
    });
    // If reminder is set/changed, schedule notification.
    if (result.reminderAt) {
      const parentTask = await prisma.task.findUnique({
        where: { id: originalSubtask.taskId },
        select: { id: true, title: true, creatorId: true },
      });
      if (parentTask) {
        const subtaskReminderObj = {
          id: result.id,
          title: `${parentTask.title} - Subtask: ${result.title}`,
          creatorId: parentTask.creatorId,
          reminderAt: result.reminderAt,
        };
        await scheduleTaskNotification(subtaskReminderObj, 'reminderAt');
      }
    }
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
    // Optionally, retrieve subtask to cancel scheduled notification if needed.
    const subtask = await prisma.subtask.findUnique({
      where: { id: input.id },
    });
    const result = await prisma.subtask.delete({
      where: { id: input.id },
    });
    logger.success(`Subtask deleted: ${input.id}`);
    await createAuditLog({
      action: 'SUBTASK_DELETED',
      description: `Subtask deleted: ${input.id}`,
      entityType: 'Subtask',
      entityId: input.id,
      ctx,
    });
    // OPTIONAL: Cancel scheduled notification if your queue supports cancellation.
    return { success: true };
  });

/* -----------------------------------------------
   REMINDER PROCEDURE FOR TASKS
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
    await createAuditLog({
      action: 'REMINDER_SET',
      description: `Reminder set for task ${input.id} at ${input.reminderAt}`,
      entityType: 'Task',
      entityId: input.id,
      ctx,
    });
    // Reschedule notification for the new reminder time.
    const updatedTask = await prisma.task.findUnique({ where: { id: input.id } });
    if (updatedTask) {
      await scheduleTaskNotification(updatedTask, 'reminderAt');
    }
    return { success: true };
  });

/* -----------------------------------------------
   SEARCH PROCEDURE
----------------------------------------------- */

/**
 * Search tasks by keyword in title or description.
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
    // Check that the task exists before attempting to add a comment.
    const task = await prisma.task.findUnique({
      where: { id: input.taskId },
      select: { id: true },
    });
    if (!task) {
      throw new Error('Task not found for adding comment');
    }
    const userId = ctx.user.id;
    const comment = await prisma.comment.create({
      data: {
        content: input.content,
        task: { connect: { id: input.taskId } },
        author: { connect: { id: userId } },
      },
    });
    logger.success(`Comment added to task ${input.taskId} by user ${ctx.user.email}`);
    await createAuditLog({
      action: 'COMMENT_ADDED',
      description: `Comment added to task ${input.taskId}: ${input.content}`,
      entityType: 'Comment',
      entityId: comment.id,
      ctx,
    });
    return { comment };
  });

/**
 * Get all comments for a given task.
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
      include: {
        author: {
          select: {
            name: true,
            profileImage: true
          }
        }
      },
    });
    return { comments };
  });


/**
 * Update a comment.
 */
const updateComment = protectedProcedure
  .input(
    z.preprocess(
      unwrapInput,
      z.object({
        id: z.string(),
        content: z.string().min(1, "Comment cannot be empty"),
      })
    )
  )
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    // Retrieve the original comment to ensure it exists and that the current user is the author.
    const originalComment = await prisma.comment.findUnique({
      where: { id: input.id },
    });
    if (!originalComment) {
      throw new Error('Comment not found');
    }
    if (originalComment.authorId !== userId) {
      throw new Error('Unauthorized to update comment');
    }
    const diffDescription = computeDiff(originalComment, { content: input.content });
    const updatedComment = await prisma.comment.update({
      where: { id: input.id },
      data: { content: input.content },
    });
    logger.success(`Comment updated: ${input.id}`);
    await createAuditLog({
      action: 'COMMENT_UPDATED',
      description: diffDescription,
      entityType: 'Comment',
      entityId: input.id,
      ctx,
    });
    return { comment: updatedComment };
  });

/**
 * Delete a comment.
 */
const deleteComment = protectedProcedure
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
    // Retrieve the comment to verify existence and authorization.
    const comment = await prisma.comment.findUnique({
      where: { id: input.id },
    });
    if (!comment) {
      throw new Error('Comment not found');
    }
    if (comment.authorId !== userId) {
      throw new Error('Unauthorized to delete comment');
    }
    await prisma.comment.delete({
      where: { id: input.id },
    });
    logger.success(`Comment deleted: ${input.id}`);
    await createAuditLog({
      action: 'COMMENT_DELETED',
      description: `Comment deleted: ${input.id}`,
      entityType: 'Comment',
      entityId: input.id,
      ctx,
    });
    return { success: true };
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
    await createAuditLog({
      action: 'ATTACHMENT_CREATED',
      description: `Attachment '${input.fileName}' created for task ${input.taskId}`,
      entityType: 'Attachment',
      entityId: attachment.id,
      ctx,
    });
    return { attachment };
  });

/**
 * Get all attachments for a given task.
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
    // Fetch the attachment to get its file URL
    const attachment = await prisma.attachment.findUnique({
      where: { id: input.id },
      select: { fileUrl: true }, // Assuming 'fileurl' is the field that stores the Cloudinary URL
    });

    if (!attachment) {
      throw new Error(`Attachment not found: ${input.id}`);
    }

    // Extract the public_id from the file URL
    const fileUrl = attachment.fileUrl;  // e.g. https://res.cloudinary.com/dbnsrpwet/image/upload/v1743153948/IMG_20250327_134034.jpg.jpg
    // Remove the file extension and extract the correct public_id
    const publicId = fileUrl
      .split('/').slice(-1)[0] // Extract the last part of the URL (filename with extension)
      .replace(/\.[^.]+$/, ''); // Remove the file extension (.jpg)

    // Ensure the public_id includes the "attachments/" folder if applicable
    const folderedPublicId = `attachments/${publicId}`;

    // Log the public_id to ensure it's correct
    logger.info(`Attempting to delete from Cloudinary with public_id: ${folderedPublicId}`);

    // Proceed with Cloudinary delete
    try {
      const cloudinaryResult = await cloudinary.uploader.destroy(folderedPublicId);
      logger.success(`Attachment deleted from Cloudinary: ${folderedPublicId}`);

      // Log the full Cloudinary response for debugging
      logger.info(`Cloudinary response: ${JSON.stringify(cloudinaryResult)}`);

      if (cloudinaryResult.result !== 'ok') {
        logger.error(`Cloudinary deletion failed: ${JSON.stringify(cloudinaryResult)}`);
        throw new Error('Failed to delete from Cloudinary');
      }
    } catch (error) {
      logger.error('Error deleting attachment from Cloudinary:', error);
    }

    // Delete the attachment from your database
    const result = await prisma.attachment.delete({
      where: { id: input.id },
    });

    logger.success(`Attachment deleted from database: ${input.id}`);

    // Create an audit log for the deletion action
    await createAuditLog({
      action: 'ATTACHMENT_DELETED',
      description: `Attachment deleted: ${input.id}`,
      entityType: 'Attachment',
      entityId: input.id,
      ctx,
    });

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
  updateComment,
  deleteComment,
  createAttachment,
  getAttachments,
  deleteAttachment,
});

export default taskRouter;
