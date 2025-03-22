import express from 'express';
import authController from '@/controllers/authController';
import { authMiddleware } from '@/middlewares/authMiddleware';

const router = express.Router();

router.post('/signup', authController.signup);
router.post('/signin', authController.signin);
router.post('/forgotpassword', authController.forgotPassword);
router.post('/refreshTokens', authController.refreshTokens);
router.post('/social/signin', authController.socialSignIn);
router.post('/social/login', authController.socialLogin);
router.post('/social/refresh', authController.socialTokensRefresh);

export default router;
