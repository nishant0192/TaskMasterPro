// src/server.ts
import app from './app.js';
import dotenv from 'dotenv';
import { logger } from './utils/logger.js';

// Load environment variables (in case not loaded by app.ts)asd
dotenv.config();

const port = process.env.PORT || 3000;

app.listen(port, () => {
  logger.success(`Server is running on port ${port}`);
});
