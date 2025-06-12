/*
  Warnings:

  - You are about to drop the column `timestamp` on the `AuditLog` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "AuditLog" DROP COLUMN "timestamp",
ADD COLUMN     "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "ipAddress" TEXT,
ADD COLUMN     "metadata" JSONB,
ADD COLUMN     "userAgent" TEXT;

-- AlterTable
ALTER TABLE "Comment" ADD COLUMN     "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- AlterTable
ALTER TABLE "Notification" ADD COLUMN     "priority" INTEGER NOT NULL DEFAULT 1,
ADD COLUMN     "readAt" TIMESTAMP(3),
ADD COLUMN     "title" TEXT,
ADD COLUMN     "type" TEXT NOT NULL DEFAULT 'info';

-- AlterTable
ALTER TABLE "Subtask" ADD COLUMN     "aiGenerated" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "estimatedDuration" INTEGER,
ALTER COLUMN "updatedAt" SET DEFAULT CURRENT_TIMESTAMP;

-- AlterTable
ALTER TABLE "Task" ADD COLUMN     "aiInsights" JSONB,
ADD COLUMN     "aiPredictions" JSONB;

-- CreateTable
CREATE TABLE "AIServiceConfig" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "enablePrioritization" BOOLEAN NOT NULL DEFAULT true,
    "enablePredictions" BOOLEAN NOT NULL DEFAULT true,
    "enableScheduling" BOOLEAN NOT NULL DEFAULT true,
    "enableNLP" BOOLEAN NOT NULL DEFAULT true,
    "enableInsights" BOOLEAN NOT NULL DEFAULT true,
    "prioritizationWeight" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "riskTolerance" TEXT NOT NULL DEFAULT 'medium',
    "insightFrequency" TEXT NOT NULL DEFAULT 'weekly',
    "learningEnabled" BOOLEAN NOT NULL DEFAULT true,
    "feedbackEnabled" BOOLEAN NOT NULL DEFAULT true,
    "cacheEnabled" BOOLEAN NOT NULL DEFAULT true,
    "cacheDuration" INTEGER NOT NULL DEFAULT 300,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AIServiceConfig_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AIFeedback" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "taskId" TEXT,
    "feedbackType" TEXT NOT NULL,
    "operation" TEXT NOT NULL,
    "rating" INTEGER NOT NULL,
    "comment" TEXT,
    "wasHelpful" BOOLEAN,
    "originalData" JSONB,
    "userAction" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AIFeedback_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProductivityMetrics" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "tasksCreated" INTEGER NOT NULL DEFAULT 0,
    "tasksCompleted" INTEGER NOT NULL DEFAULT 0,
    "tasksOverdue" INTEGER NOT NULL DEFAULT 0,
    "avgCompletionTime" DOUBLE PRECISION,
    "totalFocusTime" INTEGER NOT NULL DEFAULT 0,
    "totalBreakTime" INTEGER NOT NULL DEFAULT 0,
    "workSessions" INTEGER NOT NULL DEFAULT 0,
    "qualityScore" DOUBLE PRECISION,
    "priorityAccuracy" DOUBLE PRECISION,
    "aiSuggestionsUsed" INTEGER NOT NULL DEFAULT 0,
    "aiSuggestionsIgnored" INTEGER NOT NULL DEFAULT 0,
    "aiAccuracyScore" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ProductivityMetrics_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "AIServiceConfig_userId_key" ON "AIServiceConfig"("userId");

-- CreateIndex
CREATE INDEX "AIServiceConfig_userId_idx" ON "AIServiceConfig"("userId");

-- CreateIndex
CREATE INDEX "AIFeedback_userId_idx" ON "AIFeedback"("userId");

-- CreateIndex
CREATE INDEX "AIFeedback_taskId_idx" ON "AIFeedback"("taskId");

-- CreateIndex
CREATE INDEX "AIFeedback_feedbackType_idx" ON "AIFeedback"("feedbackType");

-- CreateIndex
CREATE INDEX "AIFeedback_rating_idx" ON "AIFeedback"("rating");

-- CreateIndex
CREATE INDEX "AIFeedback_createdAt_idx" ON "AIFeedback"("createdAt");

-- CreateIndex
CREATE INDEX "ProductivityMetrics_userId_idx" ON "ProductivityMetrics"("userId");

-- CreateIndex
CREATE INDEX "ProductivityMetrics_date_idx" ON "ProductivityMetrics"("date");

-- CreateIndex
CREATE INDEX "ProductivityMetrics_userId_date_idx" ON "ProductivityMetrics"("userId", "date");

-- CreateIndex
CREATE UNIQUE INDEX "ProductivityMetrics_userId_date_key" ON "ProductivityMetrics"("userId", "date");

-- CreateIndex
CREATE INDEX "Attachment_uploadedAt_idx" ON "Attachment"("uploadedAt");

-- CreateIndex
CREATE INDEX "AuditLog_action_idx" ON "AuditLog"("action");

-- CreateIndex
CREATE INDEX "AuditLog_createdAt_idx" ON "AuditLog"("createdAt");

-- CreateIndex
CREATE INDEX "AuditLog_userId_createdAt_idx" ON "AuditLog"("userId", "createdAt");

-- CreateIndex
CREATE INDEX "Comment_createdAt_idx" ON "Comment"("createdAt");

-- CreateIndex
CREATE INDEX "Notification_isRead_idx" ON "Notification"("isRead");

-- CreateIndex
CREATE INDEX "Notification_createdAt_idx" ON "Notification"("createdAt");

-- CreateIndex
CREATE INDEX "Notification_priority_idx" ON "Notification"("priority");

-- CreateIndex
CREATE INDEX "Subtask_isCompleted_idx" ON "Subtask"("isCompleted");

-- CreateIndex
CREATE INDEX "Subtask_order_idx" ON "Subtask"("order");

-- CreateIndex
CREATE INDEX "Subtask_aiGenerated_idx" ON "Subtask"("aiGenerated");

-- CreateIndex
CREATE INDEX "Task_estimatedDuration_idx" ON "Task"("estimatedDuration");

-- CreateIndex
CREATE INDEX "Task_complexity_idx" ON "Task"("complexity");

-- CreateIndex
CREATE INDEX "Task_isArchived_idx" ON "Task"("isArchived");

-- CreateIndex
CREATE INDEX "Task_completedAt_idx" ON "Task"("completedAt");

-- CreateIndex
CREATE INDEX "Task_creatorId_status_aiPriorityScore_idx" ON "Task"("creatorId", "status", "aiPriorityScore");

-- CreateIndex
CREATE INDEX "Task_creatorId_isArchived_dueDate_idx" ON "Task"("creatorId", "isArchived", "dueDate");

-- CreateIndex
CREATE INDEX "Task_creatorId_aiLastAnalyzed_idx" ON "Task"("creatorId", "aiLastAnalyzed");

-- AddForeignKey
ALTER TABLE "AIServiceConfig" ADD CONSTRAINT "AIServiceConfig_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AIFeedback" ADD CONSTRAINT "AIFeedback_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AIFeedback" ADD CONSTRAINT "AIFeedback_taskId_fkey" FOREIGN KEY ("taskId") REFERENCES "Task"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProductivityMetrics" ADD CONSTRAINT "ProductivityMetrics_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
