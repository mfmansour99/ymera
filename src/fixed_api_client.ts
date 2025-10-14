// src/api/client.ts
// Fixed version with proper token refresh and retry logic

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig, AxiosResponse } from 'axios';
import { API_CONFIG, STORAGE_KEYS, ERROR_CODES, HTTP_STATUS } from '../utils/constants';
import { ApiError } from '../types';

// ============================================================================
// TYPES
// ============================================================================

interface RetryConfig {
  maxRetries: number;
  retryDelay: number;
  retryCount: number; // Track actual retry count
}

interface QueuedRequest {
  resolve: (value: any) => void;
  reject: (reason: any) => void;
}

interface RequestConfig extends InternalAxiosRequestConfig {
  _retryCount?: number;
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

export class ApiClientError extends Error {
  code: string;
  statusCode?: number;
  field?: string;
  details?: Record<string, unknown>;

  constructor(error: ApiError, statusCode?: number) {
    super(error.message);
    this.name = 'ApiClientError';
    this.code = error.code;
    this.statusCode = statusCode;
    this.field = error.field;
    this.details = error.details;
  }
}

// ============================================================================
// TOKEN MANAGEMENT
// ============================================================================

class TokenManager {
  private refreshPromise: Promise<string> | null = null;
  private requestQueue: QueuedRequest[] = [];
  private isRefreshing = false;

  getAccessToken(): string | null {
    return localStorage.getItem(STORAGE_KEYS.AUTH_TOKEN);
  }

  getRefreshToken(): string | null {
    return localStorage.getItem(STORAGE_KEYS.REFRESH_TOKEN);
  }

  setTokens(accessToken: string, refreshToken: string): void {
    localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, accessToken);
    localStorage.setItem(STORAGE_KEYS.REFRESH_TOKEN, refreshToken);
  }

  clearTokens(): void {
    localStorage.removeItem(STORAGE_KEYS.AUTH_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.REFRESH_TOKEN);
    localStorage.removeItem(STORAGE_KEYS.USER_DATA);
  }

  async refreshAccessToken(): Promise<string> {
    // Prevent multiple simultaneous refresh requests
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    if (this.isRefreshing) {
      // Return a promise that will resolve when refresh completes
      return new Promise((resolve, reject) => {
        this.addToQueue(resolve, reject);
      });
    }

    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      this.clearQueue(new Error('No refresh token available'));
      throw new Error('No refresh token available');
    }

    this.isRefreshing = true;
    this.refreshPromise = (async () => {
      try {
        const response = await axios.post(
          `${API_CONFIG.BASE_URL}/auth/refresh`,
          { refreshToken },
          { headers: { 'Content-Type': 'application/json' } }
        );

        const { accessToken, refreshToken: newRefreshToken } = response.data;
        this.setTokens(accessToken, newRefreshToken);
        
        // Process queued requests
        this.processQueue(null, accessToken);
        
        return accessToken;
      } catch (error) {
        this.processQueue(error as Error, null);
        this.clearTokens();
        throw error;
      } finally {
        this.isRefreshing = false;
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  addToQueue(resolve: (value: any) => void, reject: (reason: any) => void): void {
    this.requestQueue.push({ resolve, reject });
  }

  private processQueue(error: Error | null, token: string | null): void {
    this.requestQueue.forEach((promise) => {
      if (error) {
        promise.reject(error);
      } else {
        promise.resolve(token);
      }
    });
    this.requestQueue = [];
  }

  private clearQueue(error: Error): void {
    this.requestQueue.forEach(promise => promise.reject(error));
    this.requestQueue = [];
  }
}

const tokenManager = new TokenManager();

// ============================================================================
// AXIOS CLIENT
// ============================================================================

class ApiClient {
  private client: AxiosInstance;
  private pendingRequests: Map<string, AbortController> = new Map();

  constructor() {
    this.client = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  // ============================================================================
  // REQUEST INTERCEPTOR
  // ============================================================================

  private setupRequestInterceptor(): void {
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add auth token to headers
        const token = tokenManager.getAccessToken();
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        // Initialize retry count
        if (!(config as RequestConfig)._retryCount) {
          (config as RequestConfig)._retryCount = 0;
        }

        // Add request timestamp for logging
        (config as any).requestStartTime = Date.now();

        // Request deduplication
        const requestKey = this.getRequestKey(config);
        if (this.pendingRequests.has(requestKey)) {
          const controller = this.pendingRequests.get(requestKey)!;
          controller.abort();
        }
        
        const abortController = new AbortController();
        config.signal = abortController.signal;
        this.pendingRequests.set(requestKey, abortController);

        // Log request in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`);
        }

        return config;
      },
      (error: AxiosError) => {
        return Promise.reject(error);
      }
    );
  }

  // ============================================================================
  // RESPONSE INTERCEPTOR
  // ============================================================================

  private setupResponseInterceptor(): void {
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        // Clean up pending request tracking
        const requestKey = this.getRequestKey(response.config);
        this.pendingRequests.delete(requestKey);

        // Log response time in development
        if (process.env.NODE_ENV === 'development') {
          const startTime = (response.config as any).requestStartTime;
          const duration = Date.now() - startTime;
          console.log(`[API Response] ${response.config.url} - ${duration}ms`);
        }

        return response;
      },
      async (error: AxiosError) => {
        const originalRequest = error.config as RequestConfig;
        
        if (!originalRequest) {
          return Promise.reject(error);
        }

        // Clean up pending request tracking
        const requestKey = this.getRequestKey(originalRequest);
        this.pendingRequests.delete(requestKey);

        // Handle network errors
        if (!error.response) {
          throw new ApiClientError(
            {
              code: ERROR_CODES.NETWORK_ERROR,
              message: 'Network error. Please check your internet connection.',
            },
            0
          );
        }

        const { status, data } = error.response;

        // Handle 401 Unauthorized - Token expired
        if (status === HTTP_STATUS.UNAUTHORIZED && !originalRequest._retryCount) {
          try {
            const newToken = await tokenManager.refreshAccessToken();
            
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${newToken}`;
            }

            return this.client(originalRequest);
          } catch (refreshError) {
            // Refresh failed - redirect to login
            tokenManager.clearTokens();
            window.location.href = '/login';
            throw new ApiClientError(
              {
                code: ERROR_CODES.UNAUTHORIZED,
                message: 'Session expired. Please log in again.',
              },
              HTTP_STATUS.UNAUTHORIZED
            );
          }
        }

        // Handle 403 Forbidden
        if (status === HTTP_STATUS.FORBIDDEN) {
          throw new ApiClientError(
            {
              code: ERROR_CODES.FORBIDDEN,
              message: 'You do not have permission to perform this action.',
            },
            HTTP_STATUS.FORBIDDEN
          );
        }

        // Handle 404 Not Found
        if (status === HTTP_STATUS.NOT_FOUND) {
          throw new ApiClientError(
            {
              code: ERROR_CODES.NOT_FOUND,
              message: 'The requested resource was not found.',
            },
            HTTP_STATUS.NOT_FOUND
          );
        }

        // Handle 422 Validation Error
        if (status === HTTP_STATUS.UNPROCESSABLE_ENTITY) {
          const apiError = data as any;
          throw new ApiClientError(
            {
              code: ERROR_CODES.VALIDATION_ERROR,
              message: apiError.message || 'Validation error',
              field: apiError.field,
              details: apiError.details,
            },
            HTTP_STATUS.UNPROCESSABLE_ENTITY
          );
        }

        // Handle 429 Too Many Requests
        if (status === HTTP_STATUS.TOO_MANY_REQUESTS) {
          throw new ApiClientError(
            {
              code: ERROR_CODES.TIMEOUT,
              message: 'Too many requests. Please try again later.',
            },
            HTTP_STATUS.TOO_MANY_REQUESTS
          );
        }

        // Handle 500 Server Error with exponential backoff retry
        if (status >= HTTP_STATUS.INTERNAL_SERVER_ERROR) {
          const retryCount = originalRequest._retryCount || 0;
          
          if (retryCount < API_CONFIG.RETRY_ATTEMPTS) {
            originalRequest._retryCount = retryCount + 1;
            
            // Exponential backoff: 1s, 2s, 4s
            const delay = API_CONFIG.RETRY_DELAY * Math.pow(2, retryCount);
            await this.delay(delay);
            
            return this.client(originalRequest);
          }

          throw new ApiClientError(
            {
              code: ERROR_CODES.SERVER_ERROR,
              message: 'Server error. Please try again later.',
            },
            status
          );
        }

        // Handle other errors
        const apiError = data as any;
        throw new ApiClientError(
          {
            code: apiError.code || ERROR_CODES.SERVER_ERROR,
            message: apiError.message || 'An unexpected error occurred.',
          },
          status
        );
      }
    );
  }

  // ============================================================================
  // SETUP ALL INTERCEPTORS
  // ============================================================================

  private setupInterceptors(): void {
    this.setupRequestInterceptor();
    this.setupResponseInterceptor();
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private getRequestKey(config: InternalAxiosRequestConfig): string {
    return `${config.method}-${config.url}-${JSON.stringify(config.params)}`;
  }

  // ============================================================================
  // PUBLIC API METHODS
  // ============================================================================

  async get<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async patch<T>(url: string, data?: any, config?: any): Promise<T> {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: any): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  async uploadFile<T>(
    url: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
      },
    });

    return response.data;
  }

  async batchGet<T>(urls: string[]): Promise<T[]> {
    const requests = urls.map((url) => this.client.get<T>(url));
    const responses = await Promise.all(requests);
    return responses.map((response) => response.data);
  }

  setAuthToken(token: string): void {
    localStorage.setItem(STORAGE_KEYS.AUTH_TOKEN, token);
  }

  clearAuthToken(): void {
    tokenManager.clearTokens();
  }

  getAuthToken(): string | null {
    return tokenManager.getAccessToken();
  }
}

export const apiClient = new ApiClient();
export default apiClient;
