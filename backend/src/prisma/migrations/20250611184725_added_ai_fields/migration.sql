-- AlterTable
ALTER TABLE "Task" ADD COLUMN     "aiConfidence" DOUBLE PRECISION,
ADD COLUMN     "aiLastAnalyzed" TIMESTAMP(3),
ADD COLUMN     "aiPriorityScore" DOUBLE PRECISION,
ADD COLUMN     "complexity" INTEGER,
ADD COLUMN     "estimatedDuration" INTEGER;

-- CreateIndex
CREATE INDEX "Task_status_idx" ON "Task"("status");

-- CreateIndex
CREATE INDEX "Task_priority_idx" ON "Task"("priority");

-- CreateIndex
CREATE INDEX "Task_dueDate_idx" ON "Task"("dueDate");

-- CreateIndex
CREATE INDEX "Task_aiPriorityScore_idx" ON "Task"("aiPriorityScore");

-- CreateIndex
CREATE INDEX "Task_aiLastAnalyzed_idx" ON "Task"("aiLastAnalyzed");
