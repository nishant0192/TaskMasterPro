import { router } from './context.js';
import authRouter from './authRouter.js';
import userRouter from './userRouter.js';

export const appRouter = router({
  auth: authRouter,
  user: userRouter,
});

export type AppRouter = typeof appRouter;
