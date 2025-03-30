// jobs/notificationQueue.ts
import Bull from 'bull';

const notificationQueue = new Bull('notificationQueue', {
  redis: {
    host: process.env.REDIS_HOST || '127.0.0.1',
    port: Number(process.env.REDIS_PORT) || 6379,
  },
});

export default notificationQueue;
