// src/trpc/taskRouter.ts
import { z } from 'zod';
import { router, protectedProcedure } from './context.js';
import prisma from '../prisma/client.js';
import { logger } from '../utils/logger.js';
import { uploadFile } from '../utils/upload.js';
import { scheduleTaskNotification } from '../utils/scheduleNotification.js';
import { v2 as cloudinary } from 'cloudinary';
import {
  prioritizeTasks,
  predictTaskSuccess,
  generateSubtasks,
  processNaturalLanguage,
  generateInsights,
  getBehavioralInsights,
  getAIServiceStatus,
} from './aiProcedures.js';
// Import from your audit logger
import { createAuditLog, AuditActions } from '../utils/auditLogger.js';
import { Prisma } from '@prisma/client';

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
 */
function computeDiff(original: any, updates: any): string {
  const diffs: string[] = [];
  for (const key in updates) {
    if (updates.hasOwnProperty(key)) {
      const newValue = updates[key];
      const oldValue = original[key];
      const oldValStr = oldValue instanceof Date ? oldValue.toISOString() : oldValue;
      const newValStr = newValue instanceof Date ? newValue.toISOString() : newValue;
      if (newValue !== undefined && newValue !== null && newValStr !== oldValStr) {
        diffs.push(`${key} changed from '${oldValStr}' to '${newValStr}'`);
      }
    }
  }
  return diffs.length ? diffs.join('; ') : 'No changes detected';
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
      action: AuditActions.TASK_CREATED,
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
      action: AuditActions.TASK_UPDATED,
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
      action: AuditActions.TASK_COMPLETED,
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

    // First, find the task to ensure it exists and is owned by the user
    const task = await prisma.task.findUnique({
      where: { id: input.id },
    });

    if (!task || task.creatorId !== userId) {
      logger.error(`Task deletion failed: Task ${input.id} not found or unauthorized`);
      throw new Error('Task deletion failed');
    }

    // Store task information before deletion
    const taskInfo = {
      id: task.id,
      title: task.title,
      description: task.description,
      createdAt: task.createdAt,
      dueDate: task.dueDate,
      priority: task.priority,
    };

    try {
      // Use transaction to ensure atomicity
      await prisma.$transaction(async (tx) => {
        // Delete the task first
        await tx.task.delete({
          where: { id: input.id },
        });

        // Create audit log after deletion (without task connection)
        await tx.auditLog.create({
          data: {
            action: AuditActions.TASK_DELETED,
            description: `Task deleted: "${taskInfo.title}" (ID: ${taskInfo.id})`,
            entityType: 'Task',
            entityId: taskInfo.id,
            metadata: {
              deletedTaskInfo: taskInfo,
              deletedAt: new Date().toISOString(),
              deletedBy: ctx.user.id,
            } as Prisma.InputJsonValue,
            user: { connect: { id: ctx.user.id } },
            // Don't connect to task since it's been deleted
          },
        });
      });

      logger.success(`Task deleted: ${input.id}`);
      return { success: true };

    } catch (error) {
      logger.error(`Task deletion transaction failed for ${input.id}:`, error);
      throw new Error('Failed to delete task');
    }
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
      action: AuditActions.SUBTASK_CREATED,
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
      action: AuditActions.SUBTASK_UPDATED,
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
    try {
      // Get subtask and parent task info before deletion
      const subtask = await prisma.subtask.findUnique({
        where: { id: input.id },
        include: {
          task: {
            select: {
              id: true,
              title: true,
              creatorId: true
            },
          },
        },
      });

      if (!subtask) {
        throw new Error('Subtask not found');
      }

      // Check authorization through parent task
      if (subtask.task.creatorId !== ctx.user.id) {
        throw new Error('Unauthorized to delete subtask');
      }

      const subtaskInfo = {
        id: subtask.id,
        title: subtask.title,
        taskId: subtask.taskId,
        taskTitle: subtask.task.title,
        order: subtask.order,
        isCompleted: subtask.isCompleted,
      };

      // Delete in transaction with audit log
      await prisma.$transaction(async (tx) => {
        // Delete subtask first
        await tx.subtask.delete({
          where: { id: input.id },
        });

        // Create audit log (parent task still exists, so we can connect to it)
        await tx.auditLog.create({
          data: {
            action: AuditActions.SUBTASK_DELETED,
            description: `Subtask deleted: "${subtaskInfo.title}" from task "${subtaskInfo.taskTitle}"`,
            entityType: 'Subtask',
            entityId: input.id,
            metadata: {
              deletedSubtaskInfo: subtaskInfo,
              deletedAt: new Date().toISOString(),
              deletedBy: ctx.user.id,
            } as Prisma.InputJsonValue,
            user: { connect: { id: ctx.user.id } },
            task: { connect: { id: subtask.taskId } }, // Parent task still exists
          },
        });
      });

      logger.success(`Subtask deleted: ${input.id}`);
      return { success: true };

    } catch (error) {
      logger.error(`Subtask deletion failed for ${input.id}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to delete subtask'
      );
    }
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
      action: AuditActions.REMINDER_SET,
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
      action: AuditActions.COMMENT_ADDED,
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
      action: AuditActions.COMMENT_UPDATED,
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

    try {
      // Get comment and task info before deletion
      const comment = await prisma.comment.findUnique({
        where: { id: input.id },
        include: {
          task: {
            select: {
              id: true,
              title: true
            },
          },
        },
      });

      if (!comment) {
        throw new Error('Comment not found');
      }

      if (comment.authorId !== userId) {
        throw new Error('Unauthorized to delete comment');
      }

      const commentInfo = {
        id: comment.id,
        content: comment.content,
        taskId: comment.taskId,
        taskTitle: comment.task.title,
        authorId: comment.authorId,
        createdAt: comment.createdAt,
      };

      // Delete in transaction with audit log
      await prisma.$transaction(async (tx) => {
        // Delete comment first
        await tx.comment.delete({
          where: { id: input.id },
        });

        // Create audit log (task still exists, so we can connect to it)
        await tx.auditLog.create({
          data: {
            action: AuditActions.COMMENT_DELETED,
            description: `Comment deleted from task "${commentInfo.taskTitle}"`,
            entityType: 'Comment',
            entityId: input.id,
            metadata: {
              deletedCommentInfo: {
                ...commentInfo,
                content: commentInfo.content.substring(0, 200), // Truncate for storage
              },
              deletedAt: new Date().toISOString(),
              deletedBy: ctx.user.id,
            } as Prisma.InputJsonValue,
            user: { connect: { id: ctx.user.id } },
            task: { connect: { id: comment.taskId } }, // Task still exists
          },
        });
      });

      logger.success(`Comment deleted: ${input.id}`);
      return { success: true };

    } catch (error) {
      logger.error(`Comment deletion failed for ${input.id}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to delete comment'
      );
    }
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
      action: AuditActions.ATTACHMENT_CREATED,
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
    try {
      // Fetch the attachment with task info
      const attachment = await prisma.attachment.findUnique({
        where: { id: input.id },
        include: {
          task: {
            select: {
              id: true,
              title: true,
              creatorId: true,
            },
          },
        },
      });

      if (!attachment) {
        throw new Error(`Attachment not found: ${input.id}`);
      }

      // Check authorization through parent task
      if (attachment.task.creatorId !== ctx.user.id) {
        throw new Error('Unauthorized to delete attachment');
      }

      const attachmentInfo = {
        id: attachment.id,
        fileName: attachment.fileName,
        fileType: attachment.fileType,
        fileSize: attachment.fileSize,
        fileUrl: attachment.fileUrl,
        taskId: attachment.taskId,
        taskTitle: attachment.task.title,
      };

      // Extract public_id for Cloudinary deletion
      const fileUrl = attachment.fileUrl;
      const publicId = fileUrl
        .split('/').slice(-1)[0]
        .replace(/\.[^.]+$/, '');
      const folderedPublicId = `attachments/${publicId}`;

      // Delete from Cloudinary and database in sequence
      try {
        // Delete from Cloudinary first
        const cloudinaryResult = await cloudinary.uploader.destroy(folderedPublicId);
        logger.info(`Cloudinary deletion result: ${JSON.stringify(cloudinaryResult)}`);

        // Then delete from database with audit log
        await prisma.$transaction(async (tx) => {
          // Delete attachment
          await tx.attachment.delete({
            where: { id: input.id },
          });

          // Create audit log (task still exists)
          await tx.auditLog.create({
            data: {
              action: AuditActions.ATTACHMENT_DELETED,
              description: `Attachment deleted: "${attachmentInfo.fileName}" from task "${attachmentInfo.taskTitle}"`,
              entityType: 'Attachment',
              entityId: input.id,
              metadata: {
                deletedAttachmentInfo: attachmentInfo,
                cloudinaryResult,
                deletedAt: new Date().toISOString(),
                deletedBy: ctx.user.id,
              } as Prisma.InputJsonValue,
              user: { connect: { id: ctx.user.id } },
              task: { connect: { id: attachment.taskId } }, // Task still exists
            },
          });
        });

        logger.success(`Attachment deleted from both Cloudinary and database: ${input.id}`);
        return { success: true };

      } catch (cloudinaryError) {
        logger.error('Error deleting attachment from Cloudinary:', cloudinaryError);

        // Still delete from database even if Cloudinary fails
        await prisma.$transaction(async (tx) => {
          await tx.attachment.delete({
            where: { id: input.id },
          });

          await tx.auditLog.create({
            data: {
              action: AuditActions.ATTACHMENT_DELETED,
              description: `Attachment deleted from database: "${attachmentInfo.fileName}" (Cloudinary deletion failed)`,
              entityType: 'Attachment',
              entityId: input.id,
              metadata: {
                deletedAttachmentInfo: attachmentInfo,
                cloudinaryError: String(cloudinaryError),
                deletedAt: new Date().toISOString(),
                deletedBy: ctx.user.id,
              } as Prisma.InputJsonValue,
              user: { connect: { id: ctx.user.id } },
              task: { connect: { id: attachment.taskId } },
            },
          });
        });

        logger.success(`Attachment deleted from database: ${input.id}`);
        return { success: true };
      }

    } catch (error) {
      logger.error(`Attachment deletion failed for ${input.id}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to delete attachment'
      );
    }
  });

/* -----------------------------------------------
 AI-POWERED PROCEDURES
----------------------------------------------- */

const aiPrioritizeTasks = prioritizeTasks;
const aiPredictTaskSuccess = predictTaskSuccess;
const aiGenerateSubtasks = generateSubtasks;
const aiProcessNaturalLanguage = processNaturalLanguage;
const aiGenerateInsights = generateInsights;
const aiGetBehavioralInsights = getBehavioralInsights;
const aiGetServiceStatus = getAIServiceStatus;

/**
 * Smart task scheduling using AI
 */
const aiGenerateSchedule = protectedProcedure
  .input(z.object({
    taskIds: z.array(z.string()),
    constraints: z.object({
      availableTimeSlots: z.array(z.object({
        start: z.string(),
        end: z.string(),
        date: z.string(),
      })),
      workingHours: z.object({
        start: z.string(),
        end: z.string(),
      }),
      breakPreferences: z.object({
        duration: z.number(),
        frequency: z.number(),
      }).optional(),
    }),
    preferences: z.object({
      focusTimePreference: z.enum(['morning', 'afternoon', 'evening']).optional(),
      maxContinuousWork: z.number().optional(),
      preferredTaskGrouping: z.enum(['similar', 'mixed']).optional(),
    }).optional(),
  }))
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;

    try {
      // Get tasks for scheduling
      const tasks = await ctx.prisma.task.findMany({
        where: {
          id: { in: input.taskIds },
          creatorId: userId,
          isArchived: false,
          status: { notIn: ['COMPLETED', 'CANCELLED'] },
        },
      });

      if (tasks.length === 0) {
        throw new Error('No valid tasks found for scheduling');
      }

      // Transform tasks for AI service
      const aiTasks = tasks.map(task => ({
        id: task.id,
        title: task.title,
        estimatedDuration: task.estimatedDuration || 60, // Default 1 hour
        priority: task.priority || 1,
        dueDate: task.dueDate?.toISOString(),
        dependencies: [], // Add if you have task dependencies
      }));

      // Call AI service
      const { aiIntegrationService } = await import('../services/aiIntegrationService.js');
      const result = await aiIntegrationService.generateSchedule({
        tasks: aiTasks,
        constraints: input.constraints,
        preferences: input.preferences,
      });

      // Log the operation
      await createAuditLog({
        action: AuditActions.AI_SCHEDULE_GENERATION,
        description: `AI generated schedule for ${tasks.length} tasks`,
        entityType: 'Task',
        entityId: userId,
        ctx,
      });

      return {
        success: result.success,
        data: result.data,
        fallback: result.fallback,
        error: result.error,
        metadata: {
          tasksScheduled: tasks.length,
          scheduleGenerated: result.success,
        },
      };

    } catch (error) {
      logger.error(`AI schedule generation failed for user ${userId}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to generate schedule'
      );
    }
  });

/**
 * Bulk AI operations for multiple tasks
 */
const aiBulkOptimize = protectedProcedure
  .input(z.object({
    taskIds: z.array(z.string()),
    operations: z.array(z.enum(['prioritize', 'predict', 'schedule'])),
    preferences: z.object({
      updatePriorities: z.boolean().default(true),
      generateSchedule: z.boolean().default(false),
      includeInsights: z.boolean().default(true),
    }).optional(),
  }))
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const startTime = Date.now();

    try {
      const results: any = {
        prioritization: null,
        predictions: null,
        schedule: null,
        insights: null,
        errors: [],
      };

      // Get tasks
      const tasks = await ctx.prisma.task.findMany({
        where: {
          id: { in: input.taskIds },
          creatorId: userId,
          isArchived: false,
        },
      });

      if (tasks.length === 0) {
        throw new Error('No tasks found for bulk optimization');
      }

      const { aiIntegrationService } = await import('../services/aiIntegrationService.js');

      // Run prioritization if requested
      if (input.operations.includes('prioritize')) {
        try {
          const aiTasks = tasks.map(task => ({
            id: task.id,
            title: task.title,
            description: task.description || undefined,
            dueDate: task.dueDate?.toISOString(),
            priority: task.priority || 1,
            status: task.status,
          }));

          const prioritizationResult = await aiIntegrationService.prioritizeTasks({
            tasks: aiTasks,
          });

          results.prioritization = prioritizationResult;

          // Update priorities if successful and requested
          if (prioritizationResult.success && input.preferences?.updatePriorities) {
            const updates = prioritizationResult.data.prioritizedTasks.map((task: any) =>
              ctx.prisma.task.update({
                where: { id: task.id },
                data: { priority: Math.round(task.aiPriorityScore * 10) },
              })
            );
            await ctx.prisma.$transaction(updates);
          }
        } catch (error) {
          results.errors.push(`Prioritization failed: ${error}`);
        }
      }

      // Run predictions if requested
      if (input.operations.includes('predict')) {
        try {
          const aiTasks = tasks.map(task => ({
            id: task.id,
            title: task.title,
            priority: task.priority || 1,
            estimatedDuration: task.estimatedDuration || 60,
            dueDate: task.dueDate?.toISOString(),
          }));

          const predictionResult = await aiIntegrationService.predictTaskSuccess({
            tasks: aiTasks,
          });

          results.predictions = predictionResult;
        } catch (error) {
          results.errors.push(`Predictions failed: ${error}`);
        }
      }

      // Generate insights if requested
      if (input.preferences?.includeInsights) {
        try {
          const insightsResult = await aiIntegrationService.generateInsights(userId, {
            analysisTypes: ['productivity', 'optimization'],
            includeRecommendations: true,
          });

          results.insights = insightsResult;
        } catch (error) {
          results.errors.push(`Insights failed: ${error}`);
        }
      }

      const duration = Date.now() - startTime;

      // Log bulk operation
      await createAuditLog({
        action: AuditActions.AI_BULK_OPTIMIZATION,
        description: `Bulk AI optimization on ${tasks.length} tasks: ${input.operations.join(', ')}`,
        entityType: 'Task',
        entityId: userId,
        ctx,
      });

      logger.info(`Bulk AI optimization completed in ${duration}ms for user ${userId}`);

      return {
        success: results.errors.length === 0,
        data: results,
        metadata: {
          tasksProcessed: tasks.length,
          operationsRequested: input.operations,
          operationsCompleted: input.operations.filter(op =>
            !results.errors.some((err: string) => err.toLowerCase().includes(op))
          ),
          processingTime: duration,
          errorsEncountered: results.errors.length,
        },
      };

    } catch (error) {
      logger.error(`Bulk AI optimization failed for user ${userId}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Bulk optimization failed'
      );
    }
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
  ai: router({
    prioritizeTasks: aiPrioritizeTasks,
    predictTaskSuccess: aiPredictTaskSuccess,
    generateSubtasks: aiGenerateSubtasks,
    processNaturalLanguage: aiProcessNaturalLanguage,
    generateSchedule: aiGenerateSchedule,
    generateInsights: aiGenerateInsights,
    getBehavioralInsights: aiGetBehavioralInsights,
    bulkOptimize: aiBulkOptimize,
    getServiceStatus: aiGetServiceStatus,
  }),
});

export default taskRouter;