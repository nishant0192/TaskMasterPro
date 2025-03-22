import { initTRPC, TRPCError, inferAsyncReturnType } from '@trpc/server';
import { Request, Response } from 'express';
import prisma from '../prisma/client.js';
import { verifyAccessToken } from '../utils/token.js';

export const createContext = ({ req, res }: { req: Request; res: Response }) => ({
  req,
  res,
  prisma,
});
export type Context = inferAsyncReturnType<typeof createContext>;

const t = initTRPC.context<Context>().create();

// Middleware that requires an authorization header with a valid token.
const isAuthed = t.middleware(({ ctx, next }) => {
  const authHeader = ctx.req.headers.authorization;
  if (!authHeader) {
    throw new TRPCError({
      code: 'UNAUTHORIZED',
      message: 'Authorization header missing',
    });
  }
  const token = authHeader.split(' ')[1];
  if (!token) {
    throw new TRPCError({
      code: 'UNAUTHORIZED',
      message: 'Token missing',
    });
  }
  try {
    const user = verifyAccessToken(token);
    // Extend context with the authenticated user
    return next({
      ctx: {
        ...ctx,
        user,
      },
    });
  } catch (error) {
    throw new TRPCError({
      code: 'UNAUTHORIZED',
      message: 'Invalid token',
    });
  }
});

export const router = t.router;
export const publicProcedure = t.procedure; // Procedures accessible without token.
export const protectedProcedure = t.procedure.use(isAuthed); // Require valid token.
