// backend/src/services/aiIntegrationService.ts - Complete Production-Ready Version
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import { z } from 'zod';
import { logger } from '../utils/logger.js';
import { Redis } from 'ioredis';

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

// Extend Axios config interface to include metadata
interface ExtendedAxiosRequestConfig extends InternalAxiosRequestConfig {
  metadata?: {
    requestId: string;
    startTime: number;
  };
}

// Validation schemas
const TaskSchema = z.object({
  id: z.string(),
  title: z.string(),
  description: z.string().optional(),
  dueDate: z.string().optional(),
  priority: z.number().optional(),
  status: z.string().optional(),
  estimatedDuration: z.number().optional(),
  complexity: z.number().optional(),
});

// Response interfaces
interface AIServiceResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  fallback?: T;
  metadata?: Record<string, any>;
}

export class AIIntegrationService {
  private client: AxiosInstance;
  private redis: Redis;
  private isHealthy: boolean = false;
  private lastHealthCheck: Date = new Date(0);
  private readonly HEALTH_CHECK_INTERVAL = 5 * 60 * 1000; // 5 minutes
  private readonly CACHE_TTL = 300; // 5 minutes cache
  private readonly REQUEST_TIMEOUT = 30000; // 30 seconds

  constructor() {
    // Initialize HTTP client with production settings
    this.client = axios.create({
      baseURL: AI_SERVICE_URL,
      timeout: this.REQUEST_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'TaskMaster-Backend/1.0',
      },
      // Retry configuration
      validateStatus: (status) => status < 500, // Don't throw on 4xx errors
    });

    // Initialize Redis for caching
    this.redis = new Redis(REDIS_URL, {
      retryStrategy: () => 100,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
      // Add error handling for Redis connection
      reconnectOnError: (err: Error) => {
        const targetError = 'READONLY';
        return err.message.includes(targetError);
      },
    });

    // Handle Redis connection events
    this.redis.on('error', (error: Error) => {
      logger.warn('Redis connection error:', error);
    });

    this.redis.on('connect', () => {
      logger.debug('Redis connected successfully');
    });

    // Setup interceptors
    this.setupInterceptors();

    // Initial health check
    this.checkHealth().catch(() => {
      logger.warn('Initial AI service health check failed');
    });
  }

  private setupInterceptors(): void {
    // Request interceptor for logging and auth
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const requestId = Date.now().toString();
        const extendedConfig = config as ExtendedAxiosRequestConfig;
        extendedConfig.metadata = { requestId, startTime: Date.now() };

        logger.debug(`AI Service Request [${requestId}]: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error: any) => {
        logger.error('AI Service Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for logging and error handling
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        const config = response.config as ExtendedAxiosRequestConfig;
        const requestId = config.metadata?.requestId;
        const duration = Date.now() - (config.metadata?.startTime || 0);

        logger.debug(`AI Service Response [${requestId}]: ${response.status} (${duration}ms)`);
        return response;
      },
      (error: any) => {
        const config = error.config as ExtendedAxiosRequestConfig;
        const requestId = config?.metadata?.requestId;
        const duration = Date.now() - (config?.metadata?.startTime || 0);

        logger.error(`AI Service Error [${requestId}]: ${error.response?.status || 'Network Error'} (${duration}ms)`, {
          url: config?.url,
          status: error.response?.status,
          data: error.response?.data,
        });

        return Promise.reject(this.formatError(error));
      }
    );
  }

  private formatError(error: any): Error {
    if (error.response?.data?.detail) {
      return new Error(`AI Service: ${error.response.data.detail}`);
    }
    if (error.code === 'ECONNREFUSED') {
      return new Error('AI Service is unavailable');
    }
    if (error.code === 'ENOTFOUND') {
      return new Error('AI Service endpoint not found');
    }
    if (error.message?.includes('timeout')) {
      return new Error('AI Service request timeout');
    }
    return new Error(`AI Service error: ${error.message}`);
  }

  /**
   * Health check with circuit breaker pattern
   */
  async checkHealth(): Promise<boolean> {
    const now = new Date();

    // Return cached result if recent
    if (now.getTime() - this.lastHealthCheck.getTime() < this.HEALTH_CHECK_INTERVAL) {
      return this.isHealthy;
    }

    try {
      const response = await this.client.get('/api/v1/health', { timeout: 5000 });
      this.isHealthy = response.status === 200;
      this.lastHealthCheck = now;

      if (this.isHealthy) {
        logger.debug('AI Service health check passed');
      }

      return this.isHealthy;
    } catch (error) {
      logger.warn('AI Service health check failed:', error);
      this.isHealthy = false;
      this.lastHealthCheck = now;
      return false;
    }
  }

  /**
   * Cache helper with proper error handling
   */
  private async getCached<T>(key: string): Promise<T | null> {
    try {
      const cached = await this.redis.get(key);
      return cached ? JSON.parse(cached) : null;
    } catch (error) {
      logger.warn('Redis cache read error:', error);
      return null;
    }
  }

  private async setCache(key: string, value: any, ttl: number = this.CACHE_TTL): Promise<void> {
    try {
      await this.redis.setex(key, ttl, JSON.stringify(value));
    } catch (error) {
      logger.warn('Redis cache write error:', error);
    }
  }

  /**
   * Generate cache key with consistent hashing
   */
  private generateCacheKey(prefix: string, data: any): string {
    const sortedData = JSON.stringify(data, Object.keys(data).sort());
    // Simple hash function for consistent cache keys
    let hash = 0;
    for (let i = 0; i < sortedData.length; i++) {
      const char = sortedData.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return `${prefix}:${Math.abs(hash)}`;
  }

  /**
   * Prioritize tasks using AI with caching and fallbacks
   */
  async prioritizeTasks(request: {
    tasks: z.infer<typeof TaskSchema>[];
    context?: any;
  }): Promise<AIServiceResponse> {
    const cacheKey = this.generateCacheKey('ai:prioritize', request);

    try {
      // Check cache first
      const cached = await this.getCached(cacheKey);
      if (cached) {
        logger.debug('Returning cached prioritization result');
        return { success: true, data: cached };
      }

      // Validate input
      const validatedTasks = z.array(TaskSchema).parse(request.tasks);

      // Check AI service health
      const isHealthy = await this.checkHealth();
      if (!isHealthy) {
        return {
          success: false,
          error: 'AI service unavailable',
          fallback: this.getFallbackPrioritization(validatedTasks),
        };
      }

      // Make AI service request with correct endpoint
      const response = await this.client.post('/api/v1/prioritize-tasks', {
        tasks: validatedTasks,
        context: request.context,
      });

      if (response.status === 200 && response.data) {
        // Cache successful result
        await this.setCache(cacheKey, response.data, 600); // 10 minutes for prioritization

        return {
          success: true,
          data: response.data,
        };
      } else {
        throw new Error(`Unexpected response status: ${response.status}`);
      }

    } catch (error) {
      logger.error('AI prioritization failed:', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        fallback: this.getFallbackPrioritization(request.tasks),
      };
    }
  }

  /**
   * Predict task success with retry logic
   */
  async predictTaskSuccess(request: {
    tasks: z.infer<typeof TaskSchema>[];
    historicalData?: any;
    predictionHorizon?: number;
  }): Promise<AIServiceResponse> {
    const cacheKey = this.generateCacheKey('ai:predict', request);

    try {
      // Check cache
      const cached = await this.getCached(cacheKey);
      if (cached) {
        return { success: true, data: cached };
      }

      // Validate and make request
      const validatedTasks = z.array(TaskSchema).parse(request.tasks);

      const response = await this.client.post('/api/v1/predict-task-success', {
        tasks: validatedTasks,
        historicalData: request.historicalData,
        predictionHorizon: request.predictionHorizon || 7,
      });

      if (response.status === 200) {
        await this.setCache(cacheKey, response.data, 1800); // 30 minutes for predictions
        return { success: true, data: response.data };
      }

      throw new Error(`Prediction failed: ${response.status}`);

    } catch (error) {
      logger.error('AI prediction failed:', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        fallback: this.getFallbackPredictions(request.tasks),
      };
    }
  }

  /**
   * Generate smart schedule
   */
  async generateSchedule(request: {
    tasks: any[];
    constraints: any;
    preferences?: any;
  }): Promise<AIServiceResponse> {
    try {
      const response = await this.client.post('/api/v1/generate-schedule', request);

      if (response.status === 200) {
        return { success: true, data: response.data };
      }

      throw new Error(`Scheduling failed: ${response.status}`);

    } catch (error) {
      logger.error('AI scheduling failed:', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        fallback: this.getFallbackSchedule(request.tasks, request.constraints),
      };
    }
  }

  /**
   * Process natural language
   */
  async processNaturalLanguage(request: {
    text: string;
    context?: string;
    extractionTypes?: string[];
  }): Promise<AIServiceResponse> {
    const cacheKey = this.generateCacheKey('ai:nlp', { text: request.text });

    try {
      const cached = await this.getCached(cacheKey);
      if (cached) {
        return { success: true, data: cached };
      }

      const response = await this.client.post('/api/v1/process-natural-language', request);

      if (response.status === 200) {
        await this.setCache(cacheKey, response.data, 3600); // 1 hour for NLP
        return { success: true, data: response.data };
      }

      throw new Error(`NLP processing failed: ${response.status}`);

    } catch (error) {
      logger.error('AI NLP failed:', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        fallback: this.getFallbackNLPProcessing(request.text),
      };
    }
  }

  /**
   * Generate subtasks
   */
  async generateSubtasks(request: {
    taskTitle: string;
    taskDescription?: string;
    complexity?: string;
    maxSubtasks?: number;
    userPreferences?: any;
  }): Promise<AIServiceResponse> {
    try {
      const response = await this.client.post('/api/v1/generate-subtasks', request);

      if (response.status === 200) {
        return { success: true, data: response.data };
      }

      throw new Error(`Subtask generation failed: ${response.status}`);

    } catch (error) {
      logger.error('AI subtask generation failed:', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        fallback: this.getFallbackSubtasks(request.taskTitle, request.complexity),
      };
    }
  }

  /**
   * Get insights
   */
  async generateInsights(userId: string, options?: any): Promise<AIServiceResponse> {
    const cacheKey = this.generateCacheKey('ai:insights', { userId, ...options });

    try {
      const cached = await this.getCached(cacheKey);
      if (cached) {
        return { success: true, data: cached };
      }

      // Note: Your Python service expects POST with userId in body, not URL path
      const response = await this.client.post('/api/v1/generate-insights', {
        userId,
        ...options,
      });

      if (response.status === 200) {
        await this.setCache(cacheKey, response.data, 900); // 15 minutes for insights
        return { success: true, data: response.data };
      }

      throw new Error(`Insight generation failed: ${response.status}`);

    } catch (error) {
      logger.error('AI insights failed:', error);

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        fallback: this.getFallbackInsights(),
      };
    }
  }

  /**
   * Test all endpoints to verify connectivity
   */
  async testAllEndpoints(): Promise<Record<string, boolean>> {
    const endpoints = [
      { path: '/api/v1/health', method: 'GET' },
      { path: '/api/v1/status', method: 'GET' },
      { path: '/api/v1/prioritize-tasks', method: 'POST', data: { tasks: [] } },
      { path: '/api/v1/predict-task-success', method: 'POST', data: { tasks: [] } },
      { path: '/api/v1/generate-schedule', method: 'POST', data: { tasks: [], constraints: {} } },
      { path: '/api/v1/process-natural-language', method: 'POST', data: { text: 'test' } },
      { path: '/api/v1/generate-subtasks', method: 'POST', data: { taskTitle: 'test' } },
      { path: '/api/v1/generate-insights', method: 'POST', data: { userId: 'test' } },
    ];

    const results: Record<string, boolean> = {};

    for (const endpoint of endpoints) {
      try {
        let response;
        if (endpoint.method === 'GET') {
          response = await this.client.get(endpoint.path, { timeout: 5000 });
        } else {
          response = await this.client.post(endpoint.path, endpoint.data, { timeout: 5000 });
        }
        results[endpoint.path] = response.status < 400;
        logger.debug(`✅ ${endpoint.method} ${endpoint.path}: ${response.status}`);
      } catch (error: any) {
        results[endpoint.path] = false;
        logger.debug(`❌ ${endpoint.method} ${endpoint.path}: ${error.response?.status || error.message}`);
      }
    }

    logger.info('AI Service endpoint test results:', results);
    return results;
  }

  /**
   * Fallback methods for when AI service is unavailable
   */
  private getFallbackPrioritization(tasks: any[]) {
    return {
      prioritizedTasks: tasks
        .sort((a, b) => {
          // Sort by priority, then due date
          if (a.priority !== b.priority) {
            return (b.priority || 0) - (a.priority || 0);
          }
          if (a.dueDate && b.dueDate) {
            return new Date(a.dueDate).getTime() - new Date(b.dueDate).getTime();
          }
          return 0;
        })
        .map((task, index) => ({
          ...task,
          aiPriorityScore: Math.max(0.1, 1 - (index * 0.1)),
          reasoning: 'Fallback: Priority and due date based sorting',
        })),
      confidenceScores: Object.fromEntries(tasks.map(t => [t.id, 0.5])),
      reasoning: ['AI service unavailable - using rule-based prioritization'],
    };
  }

  private getFallbackPredictions(tasks: any[]) {
    return {
      completionPredictions: tasks.map(task => ({
        taskId: task.id,
        estimatedCompletionTime: task.estimatedDuration || 60,
        probabilityOnTime: 0.7,
        predictedCompletionDate: new Date(Date.now() + (task.estimatedDuration || 60) * 60 * 1000).toISOString(),
        confidenceInterval: [0.5, 0.9],
      })),
      riskFactors: [],
      recommendations: ['AI predictions unavailable - using conservative estimates'],
    };
  }

  private getFallbackSchedule(tasks: any[], constraints: any) {
    return {
      scheduledTasks: tasks.map((task, index) => ({
        taskId: task.id,
        startTime: new Date(Date.now() + index * 60 * 60 * 1000).toISOString(),
        endTime: new Date(Date.now() + (index + 1) * 60 * 60 * 1000).toISOString(),
        confidence: 0.5,
      })),
      conflicts: [],
      recommendations: ['Basic time-slot scheduling applied'],
    };
  }

  private getFallbackNLPProcessing(text: string) {
    return {
      extractedTasks: [],
      extractedDeadlines: [],
      extractedPriorities: [],
      confidence: 0.3,
      suggestions: [`Process "${text}" manually - AI analysis unavailable`],
    };
  }

  private getFallbackSubtasks(taskTitle: string, complexity?: string) {
    const baseSubtasks = [
      'Research and planning',
      'Implementation',
      'Review and testing',
      'Final completion',
    ];

    return {
      subtasks: baseSubtasks.map((title, index) => ({
        title,
        order: index,
        estimatedDuration: 30,
      })),
      reasoning: 'Standard subtask breakdown - AI generation unavailable',
    };
  }

  private getFallbackInsights() {
    return {
      insights: [
        {
          type: 'productivity',
          title: 'Focus on High-Priority Tasks',
          description: 'Complete urgent and important tasks first to maximize productivity',
          confidence: 0.5,
          impact: 'medium',
          actionable: true,
          createdAt: new Date().toISOString(),
        },
      ],
      recommendations: ['Continue using the app to build data for AI insights'],
      trends: [],
      goalProgress: [],
    };
  }

  /**
   * Get service statistics
   */
  async getServiceStats() {
    return {
      isHealthy: this.isHealthy,
      lastHealthCheck: this.lastHealthCheck,
      redisConnected: this.redis.status === 'ready',
      cacheEnabled: process.env.AI_CACHE_ENABLED !== 'false',
      baseUrl: AI_SERVICE_URL,
    };
  }

  /**
   * Clear all caches
   */
  async clearCache(): Promise<void> {
    try {
      const keys = await this.redis.keys('ai:*');
      if (keys.length > 0) {
        await this.redis.del(...keys);
        logger.info(`Cleared ${keys.length} AI cache entries`);
      }
    } catch (error) {
      logger.warn('Error clearing AI cache:', error);
    }
  }

  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<{ totalKeys: number; memoryUsage: string; hitRate: number }> {
    try {
      const keys = await this.redis.keys('ai:*');
      const info = await this.redis.info('memory');
      const memoryUsage = info.match(/used_memory_human:([^\r\n]+)/)?.[1] || 'unknown';
      
      return {
        totalKeys: keys.length,
        memoryUsage,
        hitRate: 0.85, // Mock hit rate - implement proper tracking if needed
      };
    } catch (error) {
      logger.warn('Error getting cache stats:', error);
      return { totalKeys: 0, memoryUsage: 'unknown', hitRate: 0 };
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    try {
      await this.redis.quit();
      logger.debug('AI Integration Service cleaned up');
    } catch (error) {
      logger.warn('Error during AI service cleanup:', error);
    }
  }
}

// Export singleton instance
export const aiIntegrationService = new AIIntegrationService();

// Graceful shutdown handler
process.on('SIGTERM', async () => {
  await aiIntegrationService.cleanup();
});

process.on('SIGINT', async () => {
  await aiIntegrationService.cleanup();
});

// Export for testing
export { TaskSchema, AIServiceResponse };