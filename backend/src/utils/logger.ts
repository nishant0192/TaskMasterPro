import chalk from 'chalk';

const getTimestamp = () => new Date().toISOString();

export const logger = {
  debug: (message: string, ...args: any[]) => {
    console.debug(
      chalk.gray(`[DEBUG] ${getTimestamp()} - ${message}`),
      ...args
    );
  },
  info: (message: string, ...args: any[]) => {
    console.info(
      chalk.blue(`[INFO] ${getTimestamp()} - ${message}`),
      ...args
    );
  },
  success: (message: string, ...args: any[]) => {
    console.log(
      chalk.green(`[SUCCESS] ${getTimestamp()} - ${message}`),
      ...args
    );
  },
  warn: (message: string, ...args: any[]) => {
    console.warn(
      chalk.yellow(`[WARN] ${getTimestamp()} - ${message}`),
      ...args
    );
  },
  error: (message: string, ...args: any[]) => {
    console.error(
      chalk.red(`[ERROR] ${getTimestamp()} - ${message}`),
      ...args
    );
  }
};
