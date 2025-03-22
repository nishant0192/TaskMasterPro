import { router } from './context.js';
import authRouter from './authRouter.js';

export const appRouter = router({
  auth: authRouter,
});

export type AppRouter = typeof appRouter;
