// src/routes/index.ts

import { Router } from 'express';
import authRoutes from './authRoutes';
// Import additional route files as needed, for example:
// import userRoutes from './userRoutes';
// import teamRoutes from './teamRoutes';

const router = Router();

// Mount each route module on a specific path
router.use('/auth', authRoutes);
// router.use('/users', userRoutes);
// router.use('/teams', teamRoutes);

export default router;
