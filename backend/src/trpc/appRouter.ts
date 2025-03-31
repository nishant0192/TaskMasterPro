import { router } from './context.js';
import authRouter from './authRouter.js';
import userRouter from './userRouter.js';
import taskRouter from './taskRouter.js';
import notificationRouter from './notificationsRouter.js';
import activityRouter from './activityRouter.js';


export const appRouter = router({
  auth: authRouter,
  user: userRouter,
  task: taskRouter,
  notification: notificationRouter,
  activity: activityRouter,
});

export type AppRouter = typeof appRouter;
