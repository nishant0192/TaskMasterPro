// import { Request, Response, NextFunction } from 'express';
// import { verifyAccessToken } from '@/utils/token';
// import { logger } from '@/utils/logger';

// export interface AuthenticatedRequest extends Request {
//   user?: any;
// }

// export const authMiddleware = (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction
// ) => {
//   const authHeader = req.headers.authorization;
//   if (!authHeader) {
//     logger.error('Authorization header missing');
//     return res.status(401).json({ message: 'Authorization header missing' });
//   }
//   // Expecting "Bearer <token>"
//   const token = authHeader.split(' ')[1];
//   if (!token) {
//     logger.error('Access token missing');
//     return res.status(401).json({ message: 'Access token missing' });
//   }
//   try {
//     const payload = verifyAccessToken(token);
//     req.user = payload;
//     next();
//   } catch (error) {
//     logger.error('Invalid access token', error);
//     return res.status(401).json({ message: 'Invalid access token' });
//   }
// };
