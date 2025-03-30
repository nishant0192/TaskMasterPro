// src/utils/scheduleNotification.ts
import prisma from '../prisma/client.js';
import notificationQueue from '../jobs/notificationQueue.js';
import { logger } from './logger.js';

/**
 * Schedule a notification for a task or subtask.
 * The notification is scheduled to fire a configurable number of minutes before the event time (e.g. dueDate or reminderAt).
 * This helper converts a stored date (e.g. "2025-03-30 05:54:09.268") to an ISO string in UTC.
 */
export async function scheduleTaskNotification(
  task: { id: string; title: string; creatorId: string; [key: string]: any },
  eventField: 'dueDate' | 'reminderAt'
): Promise<void> {
  if (!task || !task[eventField]) return;

  // Convert the stored date string to an ISO string assuming it is in UTC.
  // If the value is a string like "2025-03-30 05:54:09.268", replace the space with "T" and append "Z".
  let eventDateString: string;
  if (typeof task[eventField] === 'string') {
    eventDateString = task[eventField].replace(' ', 'T') + 'Z';
  } else {
    // If it's already a Date or timestamp, get the ISO string.
    eventDateString = new Date(task[eventField]).toISOString();
  }

  const eventTime = new Date(eventDateString).getTime();

  // Use the environment variable to determine the offset in minutes.
  // Defaults to 10 minutes if not specified.
  const offsetMinutes = Number(process.env.NOTIFICATION_OFFSET_MINUTES) || 10;
  const offsetMs = offsetMinutes * 60 * 1000;
  const notifyTime = eventTime - offsetMs;
  const delay = notifyTime - Date.now();

  if (delay <= 0) {
    // Notification time is already past; skip scheduling.
    logger.debug(`Notification time already passed for task ${task.id} (delay: ${delay} ms)`);
    return;
  }

  // Retrieve user to check for Expo push token.
  const user = await prisma.user.findUnique({ where: { id: task.creatorId } });
  if (!user || !user.expoPushToken) {
    logger.debug(`No push token for user ${task.creatorId}; skipping notification scheduling.`);
    return;
  }

  await notificationQueue.add(
    {
      expoPushToken: user.expoPushToken,
      title: 'Task Reminder',
      body: `Reminder: Your task "${task.title}" is due soon.`,
      data: { taskId: task.id, eventField },
    },
    { delay }
  );
  logger.debug(`Scheduled notification for task ${task.id} in ${delay} ms.`);
}
