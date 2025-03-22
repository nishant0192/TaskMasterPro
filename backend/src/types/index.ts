import { NextFunction } from "express";

export interface AuthController {
    signup: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    signin: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    forgotPassword: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    refreshTokens: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    socialSignIn: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    socialLogin: (req: Request, res: Response, next: NextFunction) => Promise<any>;
    socialTokensRefresh: (req: Request, res: Response, next: NextFunction) => Promise<any>;
}