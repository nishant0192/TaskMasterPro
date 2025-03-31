// jobs/notificationWorker.ts
import notificationQueue from './notificationQueue.js';
import fetch from 'node-fetch';
import { logger } from '../utils/logger.js';
import type { Job } from 'bull';

notificationQueue.process(async (job: Job) => {
  const { expoPushToken, title, body, data } = job.data as {
    expoPushToken: string;
    title: string;
    body: string;
    data?: any;
  };

  const messagePayload = {
    to: expoPushToken,
    sound: 'default',
    title,
    body,
    data,
  };

  try {
    const response = await fetch("https://exp.host/--/api/v2/push/send", {
      method: "POST",
      headers: {
        "host": "exp.host",
        "accept": "application/json",
        "accept-encoding": "gzip, deflate",
        "content-type": "application/json",
      },
      body: JSON.stringify(messagePayload),
    });
    const dataResp = await response.json();
    logger.success(`Notification sent: ${JSON.stringify(dataResp)}`);
    return dataResp;
  } catch (error) {
    logger.error("Notification sending failed", error);
    throw error;
  }
});
