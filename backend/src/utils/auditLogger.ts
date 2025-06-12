// backend/src/utils/auditLogger.ts - Fixed version with proper Prisma types
import prisma from '../prisma/client.js';
import { logger } from './logger.js';
import { Prisma } from '@prisma/client';

/**
 * Interface for audit log creation parameters
 */
interface CreateAuditLogParams {
    action: string;
    description: string;
    entityType?: string;
    entityId?: string;
    ctx: {
        user: {
            id: string;
            email?: string;
        };
    };
    metadata?: Record<string, any>;
}

/**
 * Create an audit log entry
 */
export async function createAuditLog({
    action,
    description,
    entityType,
    entityId,
    ctx,
    metadata,
}: CreateAuditLogParams): Promise<void> {
    try {
        // Properly handle the metadata field for Prisma
        const metadataValue = metadata ? metadata as Prisma.InputJsonValue : Prisma.JsonNull;

        await prisma.auditLog.create({
            data: {
                action,
                description,
                entityType,
                entityId,
                metadata: metadataValue,
                user: { connect: { id: ctx.user.id } },
                task: entityType === 'Task' && entityId ? { connect: { id: entityId } } : undefined,
            },
        });

        logger.debug(`Audit log created: ${action} by user ${ctx.user.id}`);
    } catch (error) {
        logger.error('Failed to create audit log:', error);
        // Don't throw error to avoid breaking the main operation
    }
}

/**
 * Create an audit log entry with additional context
 */
export async function createDetailedAuditLog({
    action,
    description,
    entityType,
    entityId,
    ctx,
    metadata,
    ipAddress,
    userAgent,
}: CreateAuditLogParams & {
    ipAddress?: string;
    userAgent?: string;
}): Promise<void> {
    try {
        const metadataValue = metadata ? metadata as Prisma.InputJsonValue : Prisma.JsonNull;

        await prisma.auditLog.create({
            data: {
                action,
                description,
                entityType,
                entityId,
                metadata: metadataValue,
                ipAddress,
                userAgent,
                user: { connect: { id: ctx.user.id } },
                task: entityType === 'Task' && entityId ? { connect: { id: entityId } } : undefined,
            },
        });

        logger.debug(`Detailed audit log created: ${action} by user ${ctx.user.id}`);
    } catch (error) {
        logger.error('Failed to create detailed audit log:', error);
    }
}

/**
 * Get audit logs for a specific entity
 */
export async function getAuditLogs({
    entityType,
    entityId,
    userId,
    limit = 50,
    startDate,
    endDate,
}: {
    entityType?: string;
    entityId?: string;
    userId?: string;
    limit?: number;
    startDate?: Date;
    endDate?: Date;
}) {
    try {
        return await prisma.auditLog.findMany({
            where: {
                ...(entityType && { entityType }),
                ...(entityId && { entityId }),
                ...(userId && { userId }),
                ...(startDate || endDate ? {
                    createdAt: {
                        ...(startDate && { gte: startDate }),
                        ...(endDate && { lte: endDate }),
                    }
                } : {}),
            },
            include: {
                user: {
                    select: {
                        id: true,
                        name: true,
                        email: true,
                    },
                },
                task: {
                    select: {
                        id: true,
                        title: true,
                    },
                },
            },
            orderBy: {
                createdAt: 'desc',
            },
            take: limit,
        });
    } catch (error) {
        logger.error('Failed to get audit logs:', error);
        return [];
    }
}

/**
 * Get audit logs with aggregations
 */
export async function getAuditLogStats({
    userId,
    startDate,
    endDate,
}: {
    userId?: string;
    startDate?: Date;
    endDate?: Date;
}) {
    try {
        const stats = await prisma.auditLog.groupBy({
            by: ['action'],
            where: {
                ...(userId && { userId }),
                ...(startDate || endDate ? {
                    createdAt: {
                        ...(startDate && { gte: startDate }),
                        ...(endDate && { lte: endDate }),
                    }
                } : {}),
            },
            _count: {
                action: true,
            },
            orderBy: {
                _count: {
                    action: 'desc',
                },
            },
        });

        const totalLogs = await prisma.auditLog.count({
            where: {
                ...(userId && { userId }),
                ...(startDate || endDate ? {
                    createdAt: {
                        ...(startDate && { gte: startDate }),
                        ...(endDate && { lte: endDate }),
                    }
                } : {}),
            },
        });

        return {
            totalLogs,
            actionStats: stats.map(stat => ({
                action: stat.action,
                count: stat._count.action,
            })),
        };
    } catch (error) {
        logger.error('Failed to get audit log stats:', error);
        return {
            totalLogs: 0,
            actionStats: [],
        };
    }
}

/**
 * Clean up old audit logs (for maintenance)
 */
export async function cleanupOldAuditLogs(daysToKeep: number = 90): Promise<number> {
    try {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);

        const result = await prisma.auditLog.deleteMany({
            where: {
                createdAt: {
                    lt: cutoffDate,
                },
            },
        });

        logger.info(`Cleaned up ${result.count} old audit logs older than ${daysToKeep} days`);
        return result.count;
    } catch (error) {
        logger.error('Failed to cleanup old audit logs:', error);
        return 0;
    }
}

/**
 * Search audit logs by description or action
 */
export async function searchAuditLogs({
    searchTerm,
    userId,
    entityType,
    limit = 50,
}: {
    searchTerm: string;
    userId?: string;
    entityType?: string;
    limit?: number;
}) {
    try {
        return await prisma.auditLog.findMany({
            where: {
                ...(userId && { userId }),
                ...(entityType && { entityType }),
                OR: [
                    {
                        action: {
                            contains: searchTerm,
                            mode: 'insensitive',
                        },
                    },
                    {
                        description: {
                            contains: searchTerm,
                            mode: 'insensitive',
                        },
                    },
                ],
            },
            include: {
                user: {
                    select: {
                        id: true,
                        name: true,
                        email: true,
                    },
                },
                task: {
                    select: {
                        id: true,
                        title: true,
                    },
                },
            },
            orderBy: {
                createdAt: 'desc',
            },
            take: limit,
        });
    } catch (error) {
        logger.error('Failed to search audit logs:', error);
        return [];
    }
}

/**
 * Audit log action types for consistency
 */
export const AuditActions = {
    // Task actions
    TASK_CREATED: 'TASK_CREATED',
    TASK_UPDATED: 'TASK_UPDATED',
    TASK_DELETED: 'TASK_DELETED',
    TASK_COMPLETED: 'TASK_COMPLETED',

    // Subtask actions
    SUBTASK_CREATED: 'SUBTASK_CREATED',
    SUBTASK_UPDATED: 'SUBTASK_UPDATED',
    SUBTASK_DELETED: 'SUBTASK_DELETED',

    // AI actions
    AI_PRIORITIZATION: 'AI_PRIORITIZATION',
    AI_PREDICTION: 'AI_PREDICTION',
    AI_SUBTASK_GENERATION: 'AI_SUBTASK_GENERATION',
    AI_NLP_TASK_CREATION: 'AI_NLP_TASK_CREATION',
    AI_SCHEDULE_GENERATION: 'AI_SCHEDULE_GENERATION',
    AI_INSIGHTS_GENERATION: 'AI_INSIGHTS_GENERATION',
    AI_BULK_OPTIMIZATION: 'AI_BULK_OPTIMIZATION',

    // Comment actions
    COMMENT_ADDED: 'COMMENT_ADDED',
    COMMENT_UPDATED: 'COMMENT_UPDATED',
    COMMENT_DELETED: 'COMMENT_DELETED',

    // Attachment actions
    ATTACHMENT_CREATED: 'ATTACHMENT_CREATED',
    ATTACHMENT_DELETED: 'ATTACHMENT_DELETED',

    // Reminder actions
    REMINDER_SET: 'REMINDER_SET',
    REMINDER_TRIGGERED: 'REMINDER_TRIGGERED',

    // User actions
    USER_LOGIN: 'USER_LOGIN',
    USER_LOGOUT: 'USER_LOGOUT',
    USER_UPDATED: 'USER_UPDATED',

    // AI Configuration actions
    AI_CONFIG_UPDATED: 'AI_CONFIG_UPDATED',
    AI_FEEDBACK_SUBMITTED: 'AI_FEEDBACK_SUBMITTED',
} as const;

export type AuditAction = typeof AuditActions[keyof typeof AuditActions];

/**
 * Helper function to create audit logs for AI operations
 */
export async function createAIAuditLog({
    action,
    description,
    ctx,
    aiData,
    processingTime,
    success,
}: {
    action: AuditAction;
    description: string;
    ctx: CreateAuditLogParams['ctx'];
    aiData?: any;
    processingTime?: number;
    success?: boolean;
}) {
    await createAuditLog({
        action,
        description,
        entityType: 'AI_Operation',
        entityId: undefined,
        ctx,
        metadata: {
            aiData: aiData ? JSON.stringify(aiData) : undefined,
            processingTime,
            success,
            timestamp: new Date().toISOString(),
        },
    });
}

/**
 * Helper function to create audit logs with request context (for web requests)
 */
export async function createWebAuditLog({
    action,
    description,
    entityType,
    entityId,
    ctx,
    metadata,
    req,
}: CreateAuditLogParams & {
    req?: {
        ip?: string;
        get?: (header: string) => string | undefined;
    };
}) {
    await createDetailedAuditLog({
        action,
        description,
        entityType,
        entityId,
        ctx,
        metadata,
        ipAddress: req?.ip,
        userAgent: req?.get?.('User-Agent'),
    });
}


export async function createAuditLogForDeletion({
    action,
    description,
    entityType,
    entityId,
    ctx,
    metadata,
    skipTaskConnection = false,
}: CreateAuditLogParams & {
    skipTaskConnection?: boolean;
}): Promise<void> {
    try {
        const metadataValue = metadata ? metadata as Prisma.InputJsonValue : Prisma.JsonNull;

        // For deletions, we don't want to connect to the entity being deleted
        await prisma.auditLog.create({
            data: {
                action,
                description,
                entityType,
                entityId,
                metadata: metadataValue,
                user: { connect: { id: ctx.user.id } },
                // Only connect to task if we're not deleting it and skipTaskConnection is false
                task: (entityType === 'Task' && entityId && !skipTaskConnection) ? undefined : 
                      (entityType !== 'Task' && entityId) ? { connect: { id: entityId } } : undefined,
            },
        });

        logger.debug(`Audit log created for deletion: ${action} by user ${ctx.user.id}`);
    } catch (error) {
        logger.error('Failed to create audit log for deletion:', error);
        // Don't throw error to avoid breaking the main operation
    }
}