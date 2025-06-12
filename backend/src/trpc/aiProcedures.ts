// backend/src/trpc/aiProcedures.ts - Fixed version with proper types
import { z } from 'zod';
import { protectedProcedure } from './context.js';
import { aiIntegrationService } from '../services/aiIntegrationService.js';
import { createAuditLog, AuditActions } from '../utils/auditLogger.js';
import { logger } from '../utils/logger.js';

// Type definitions for better type safety
interface PrioritizedTask {
  id: string;
  title: string;
  aiPriorityScore: number;
  reasoning?: string;
}

interface AIServiceResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  fallback?: T;
}

interface ExtractedTask {
  title: string;
  description?: string;
  priority?: number;
  dueDate?: string;
  estimatedDuration?: number;
}

interface SubtaskInfo {
  title: string;
  order?: number;
  estimatedDuration?: number;
}

// Input validation schemas
const PrioritizeTasksInput = z.object({
  taskIds: z.array(z.string()).optional(),
  context: z.object({
    userGoals: z.array(z.string()).optional(),
    timeAvailable: z.number().optional(),
    currentWorkload: z.number().optional(),
    urgentDeadlines: z.array(z.string()).optional(),
  }).optional(),
});

const PredictTaskSuccessInput = z.object({
  taskIds: z.array(z.string()).optional(),
  predictionHorizon: z.number().min(1).max(30).default(7),
  includeHistoricalData: z.boolean().default(true),
});

const GenerateSubtasksInput = z.object({
  taskId: z.string(),
  complexity: z.enum(['low', 'medium', 'high']).optional(),
  maxSubtasks: z.number().min(1).max(20).default(5),
  userPreferences: z.object({
    detailLevel: z.enum(['basic', 'detailed']).default('basic'),
    includeEstimates: z.boolean().default(false),
  }).optional(),
});

const ProcessNLPInput = z.object({
  text: z.string().min(1).max(1000),
  context: z.string().optional(),
  extractionTypes: z.array(z.enum(['tasks', 'deadlines', 'priorities', 'subtasks'])).optional(),
  autoCreateTasks: z.boolean().default(false),
});

const GenerateInsightsInput = z.object({
  timePeriod: z.object({
    start: z.string(),
    end: z.string(),
  }).optional(),
  analysisTypes: z.array(z.string()).optional(),
  includeRecommendations: z.boolean().default(true),
});

/**
 * Prioritize user's tasks using AI
 */
export const prioritizeTasks = protectedProcedure
  .input(PrioritizeTasksInput)
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const startTime = Date.now();

    try {
      // Get user's tasks
      const tasks = await ctx.prisma.task.findMany({
        where: {
          creatorId: userId,
          isArchived: false,
          ...(input.taskIds && { id: { in: input.taskIds } }),
        },
        include: {
          subtasks: {
            where: { isCompleted: false },
          },
        },
        orderBy: [
          { priority: 'desc' },
          { dueDate: 'asc' },
        ],
      });

      if (tasks.length === 0) {
        throw new Error('No tasks found to prioritize');
      }

      // Transform tasks for AI service
      const aiTasks = tasks.map(task => ({
        id: task.id,
        title: task.title,
        description: task.description ?? undefined,
        dueDate: task.dueDate?.toISOString(),
        priority: task.priority ?? 1,
        status: task.status,
        estimatedDuration: task.estimatedDuration ?? undefined,
      }));

      // Call AI service
      const result: AIServiceResponse<{ prioritizedTasks: PrioritizedTask[] }> = 
        await aiIntegrationService.prioritizeTasks({
          tasks: aiTasks,
          context: input.context,
        });

      // Update task priorities if AI was successful
      if (result.success && result.data?.prioritizedTasks) {
        const updates = result.data.prioritizedTasks.map((prioritizedTask: PrioritizedTask) =>
          ctx.prisma.task.update({
            where: { id: prioritizedTask.id },
            data: {
              priority: Math.round(prioritizedTask.aiPriorityScore * 10),
              aiPriorityScore: prioritizedTask.aiPriorityScore,
              aiLastAnalyzed: new Date(),
              updatedAt: new Date(),
            },
          })
        );

        await ctx.prisma.$transaction(updates);

        // Log successful AI operation
        await createAuditLog({
          action: AuditActions.AI_PRIORITIZATION,
          description: `AI prioritized ${tasks.length} tasks`,
          entityType: 'Task',
          entityId: userId,
          ctx,
          metadata: {
            tasksProcessed: tasks.length,
            processingTime: Date.now() - startTime,
          },
        });
      }

      const duration = Date.now() - startTime;
      logger.info(`AI prioritization completed in ${duration}ms for user ${userId}`);

      return {
        success: result.success,
        data: result.data,
        fallback: result.fallback,
        error: result.error,
        metadata: {
          tasksProcessed: tasks.length,
          processingTime: duration,
          aiServiceUsed: result.success,
        },
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      logger.error(`AI prioritization failed after ${duration}ms for user ${userId}:`, error);

      throw new Error(
        error instanceof Error ? error.message : 'Failed to prioritize tasks'
      );
    }
  });

/**
 * Predict task completion success probability
 */
export const predictTaskSuccess = protectedProcedure
  .input(PredictTaskSuccessInput)
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const startTime = Date.now();

    try {
      // Get tasks for prediction
      const tasks = await ctx.prisma.task.findMany({
        where: {
          creatorId: userId,
          isArchived: false,
          status: { notIn: ['COMPLETED', 'CANCELLED', 'DONE'] },
          ...(input.taskIds && { id: { in: input.taskIds } }),
        },
      });

      if (tasks.length === 0) {
        throw new Error('No tasks found for prediction');
      }

      // Get historical data if requested
      let historicalData = {};
      if (input.includeHistoricalData) {
        const completedTasks = await ctx.prisma.task.findMany({
          where: {
            creatorId: userId,
            status: { in: ['COMPLETED', 'DONE'] },
            completedAt: {
              gte: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000), // Last 90 days
            },
          },
        });

        // Calculate metrics
        const totalTasks = tasks.length + completedTasks.length;
        const completionRate = completedTasks.length / Math.max(1, totalTasks);

        const avgCompletionTime = completedTasks.reduce((sum, task) => {
          if (task.completedAt && task.createdAt) {
            return sum + (task.completedAt.getTime() - task.createdAt.getTime());
          }
          return sum;
        }, 0) / Math.max(1, completedTasks.length);

        historicalData = {
          completionRates: { overall: completionRate },
          averageCompletionTimes: {
            overall: avgCompletionTime / (1000 * 60 * 60), // Convert to hours
          },
          userPatterns: {
            totalTasksCompleted: completedTasks.length,
            averagePriority: completedTasks.reduce((sum, t) => sum + (t.priority ?? 1), 0) / completedTasks.length,
          },
        };
      }

      // Transform for AI service
      const aiTasks = tasks.map(task => ({
        id: task.id,
        title: task.title,
        description: task.description ?? undefined,
        dueDate: task.dueDate?.toISOString(),
        priority: task.priority ?? 1,
        estimatedDuration: task.estimatedDuration ?? undefined,
        complexity: task.complexity ?? undefined,
      }));

      // Call AI service
      const result: AIServiceResponse = await aiIntegrationService.predictTaskSuccess({
        tasks: aiTasks,
        historicalData,
        predictionHorizon: input.predictionHorizon,
      });

      const duration = Date.now() - startTime;
      logger.info(`AI prediction completed in ${duration}ms for user ${userId}`);

      return {
        success: result.success,
        data: result.data,
        fallback: result.fallback,
        error: result.error,
        metadata: {
          tasksAnalyzed: tasks.length,
          predictionHorizon: input.predictionHorizon,
          includeHistoricalData: input.includeHistoricalData,
          processingTime: duration,
        },
      };

    } catch (error) {
      logger.error(`AI prediction failed for user ${userId}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to predict task success'
      );
    }
  });

/**
 * Generate subtasks using AI
 */
export const generateSubtasks = protectedProcedure
  .input(GenerateSubtasksInput)
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const startTime = Date.now();

    try {
      // Get the main task
      const task = await ctx.prisma.task.findFirst({
        where: {
          id: input.taskId,
          creatorId: userId,
        },
        include: {
          subtasks: true,
        },
      });

      if (!task) {
        throw new Error('Task not found or access denied');
      }

      // Check if task already has subtasks
      if (task.subtasks.length > 0) {
        throw new Error('Task already has subtasks. Delete existing subtasks first if you want to regenerate.');
      }

      // Call AI service
      const result: AIServiceResponse<{ subtasks: SubtaskInfo[]; reasoning?: string }> = 
        await aiIntegrationService.generateSubtasks({
          taskTitle: task.title,
          taskDescription: task.description ?? undefined,
          complexity: input.complexity,
          maxSubtasks: input.maxSubtasks,
          userPreferences: input.userPreferences,
        });

      let createdSubtasks: any[] = [];

      // Create subtasks if AI was successful
      if (result.success && result.data?.subtasks) {
        const subtaskData = result.data.subtasks.map((subtask: SubtaskInfo, index: number) => ({
          title: subtask.title,
          taskId: task.id,
          order: subtask.order ?? index,
          isCompleted: false,
          estimatedDuration: subtask.estimatedDuration ?? null,
        }));

        createdSubtasks = await ctx.prisma.$transaction(
          subtaskData.map((data: any) => ctx.prisma.subtask.create({ data }))
        );

        // Log successful AI operation
        await createAuditLog({
          action: AuditActions.AI_SUBTASK_GENERATION,
          description: `AI generated ${createdSubtasks.length} subtasks for task ${task.title}`,
          entityType: 'Task',
          entityId: task.id,
          ctx,
          metadata: {
            subtasksCreated: createdSubtasks.length,
            complexity: input.complexity,
          },
        });
      }

      const duration = Date.now() - startTime;
      logger.info(`AI subtask generation completed in ${duration}ms for user ${userId}`);

      return {
        success: result.success,
        data: {
          subtasks: createdSubtasks,
          aiGenerated: result.success,
          reasoning: result.data?.reasoning,
        },
        fallback: result.fallback,
        error: result.error,
        metadata: {
          taskId: task.id,
          subtasksCreated: createdSubtasks.length,
          processingTime: duration,
        },
      };

    } catch (error) {
      logger.error(`AI subtask generation failed for user ${userId}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to generate subtasks'
      );
    }
  });

/**
 * Process natural language input to extract tasks
 */
export const processNaturalLanguage = protectedProcedure
  .input(ProcessNLPInput)
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const startTime = Date.now();

    try {
      // Call AI service
      const result: AIServiceResponse<{ extractedTasks: ExtractedTask[] }> = 
        await aiIntegrationService.processNaturalLanguage({
          text: input.text,
          context: input.context,
          extractionTypes: input.extractionTypes,
        });

      let createdTasks: any[] = [];

      // Auto-create tasks if requested and AI found tasks
      if (input.autoCreateTasks && result.success && result.data?.extractedTasks) {
        const taskData = result.data.extractedTasks.map((taskInfo: ExtractedTask) => ({
          title: taskInfo.title,
          description: taskInfo.description ?? null,
          priority: taskInfo.priority ?? 1,
          dueDate: taskInfo.dueDate ? new Date(taskInfo.dueDate) : null,
          creatorId: userId,
          status: 'TODO',
          estimatedDuration: taskInfo.estimatedDuration ?? null,
        }));

        createdTasks = await ctx.prisma.$transaction(
          taskData.map((data: any) => ctx.prisma.task.create({ data }))
        );

        // Log successful AI operation
        await createAuditLog({
          action: AuditActions.AI_NLP_TASK_CREATION,
          description: `AI created ${createdTasks.length} tasks from text: "${input.text.substring(0, 50)}..."`,
          entityType: 'Task',
          entityId: userId,
          ctx,
          metadata: {
            inputText: input.text,
            tasksCreated: createdTasks.length,
          },
        });
      }

      const duration = Date.now() - startTime;
      logger.info(`AI NLP processing completed in ${duration}ms for user ${userId}`);

      return {
        success: result.success,
        data: {
          ...result.data,
          createdTasks: input.autoCreateTasks ? createdTasks : [],
          autoCreated: input.autoCreateTasks,
        },
        fallback: result.fallback,
        error: result.error,
        metadata: {
          inputText: input.text.substring(0, 100),
          tasksCreated: createdTasks.length,
          processingTime: duration,
        },
      };

    } catch (error) {
      logger.error(`AI NLP processing failed for user ${userId}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to process natural language input'
      );
    }
  });

/**
 * Generate AI insights for the user
 */
export const generateInsights = protectedProcedure
  .input(GenerateInsightsInput)
  .mutation(async ({ input, ctx }) => {
    const userId = ctx.user.id;
    const startTime = Date.now();

    try {
      // Get user's task data for context
      const timePeriod = input.timePeriod ?? {
        start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), // Last 30 days
        end: new Date().toISOString(),
      };

      const tasks = await ctx.prisma.task.findMany({
        where: {
          creatorId: userId,
          createdAt: {
            gte: new Date(timePeriod.start),
            lte: new Date(timePeriod.end),
          },
        },
        include: {
          subtasks: true,
        },
      });

      // Call AI service
      const result: AIServiceResponse = await aiIntegrationService.generateInsights(userId, {
        timePeriod,
        analysisTypes: input.analysisTypes,
        includeRecommendations: input.includeRecommendations,
        taskData: {
          totalTasks: tasks.length,
          completedTasks: tasks.filter(t => t.status === 'COMPLETED' || t.status === 'DONE').length,
          overdueTasks: tasks.filter(t =>
            t.dueDate && new Date(t.dueDate) < new Date() && !['COMPLETED', 'DONE'].includes(t.status)
          ).length,
        },
      });

      const duration = Date.now() - startTime;
      logger.info(`AI insights generated in ${duration}ms for user ${userId}`);

      return {
        success: result.success,
        data: result.data,
        fallback: result.fallback,
        error: result.error,
        metadata: {
          timePeriod,
          tasksAnalyzed: tasks.length,
          processingTime: duration,
        },
      };

    } catch (error) {
      logger.error(`AI insights generation failed for user ${userId}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to generate insights'
      );
    }
  });

/**
 * Get behavioral insights for the user
 */
export const getBehavioralInsights = protectedProcedure
  .mutation(async ({ ctx }) => {
    const userId = ctx.user.id;
    const startTime = Date.now();

    try {
      // Get comprehensive user data for behavioral analysis
      const [tasks, user] = await Promise.all([
        ctx.prisma.task.findMany({
          where: { creatorId: userId },
          include: { subtasks: true },
          orderBy: { createdAt: 'desc' },
          take: 100, // Last 100 tasks for analysis
        }),
        ctx.prisma.user.findUnique({
          where: { id: userId },
          select: {
            createdAt: true,
            lastLogin: true,
            name: true,
          },
        }),
      ]);

      // Call AI service
      const result: AIServiceResponse = await aiIntegrationService.generateInsights(userId, {
        analysisTypes: ['behavioral', 'productivity', 'patterns'],
        includeRecommendations: true,
        userData: {
          accountAge: user?.createdAt ?
            Math.floor((Date.now() - user.createdAt.getTime()) / (1000 * 60 * 60 * 24)) : 0,
          lastLogin: user?.lastLogin?.toISOString(),
          taskHistory: tasks.map(task => ({
            status: task.status,
            priority: task.priority,
            createdAt: task.createdAt.toISOString(),
            completedAt: task.completedAt?.toISOString(),
            subtaskCount: task.subtasks.length,
          })),
        },
      });

      const duration = Date.now() - startTime;
      logger.info(`Behavioral insights generated in ${duration}ms for user ${userId}`);

      return {
        success: result.success,
        data: result.data,
        fallback: result.fallback,
        error: result.error,
        metadata: {
          tasksAnalyzed: tasks.length,
          accountAge: user?.createdAt ?
            Math.floor((Date.now() - user.createdAt.getTime()) / (1000 * 60 * 60 * 24)) : 0,
          processingTime: duration,
        },
      };

    } catch (error) {
      logger.error(`Behavioral insights failed for user ${userId}:`, error);
      throw new Error(
        error instanceof Error ? error.message : 'Failed to get behavioral insights'
      );
    }
  });

/**
 * Get AI service health status
 */
export const getAIServiceStatus = protectedProcedure
  .query(async () => {
    try {
      const isHealthy = await aiIntegrationService.checkHealth();

      return {
        healthy: isHealthy,
        status: isHealthy ? 'available' : 'unavailable',
        message: isHealthy ? 'AI service is running normally' : 'AI service is currently unavailable',
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      logger.error('Error checking AI service status:', error);

      return {
        healthy: false,
        status: 'error',
        message: 'Unable to check AI service status',
        timestamp: new Date().toISOString(),
      };
    }
  });