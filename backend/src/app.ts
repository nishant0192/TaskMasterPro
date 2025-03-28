// src/app.ts

import express from 'express';
import cors from 'cors';
import { createExpressMiddleware } from '@trpc/server/adapters/express';
import dotenv from 'dotenv';
import cookieParser from 'cookie-parser';
import { createContext } from './trpc/context.js';
import { appRouter } from './trpc/appRouter.js';

// Load environment variables
dotenv.config();

const app = express();

// Middlewares
app.use(cors());
app.use(express.json({ limit: '15mb' })); // Increase the payload size limit to 15MB
app.use(cookieParser());

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
  console.log('Body:', req.body);
  next();
});

app.use(
  '/trpc',
  createExpressMiddleware({
    router: appRouter,
    createContext: ({ req, res }: { req: express.Request; res: express.Response }) => createContext({ req, res }),
  })
);

// --- Health Check Endpoint ---
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK' });
});

export default app;
