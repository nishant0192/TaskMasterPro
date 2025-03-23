import { router } from './context.js';
import authRouter from './authRouter.js';
import userRouter from './userRouter.js';
import taskRouter from './taskRouter.js';

export const appRouter = router({
  auth: authRouter,
  user: userRouter,
  task: taskRouter,
});

export type AppRouter = typeof appRouter;
