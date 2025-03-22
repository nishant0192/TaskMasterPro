import { initTRPC, inferAsyncReturnType } from '@trpc/server';
import { Request, Response } from 'express';
import prisma from '../prisma/client.js';

export const createContext = ({ req, res }: { req: Request; res: Response }) => ({
  req,
  res,
  prisma,
});
export type Context = inferAsyncReturnType<typeof createContext>;

const t = initTRPC.context<Context>().create();

export const router = t.router;
export const publicProcedure = t.procedure;
