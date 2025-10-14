// src/utils/validation.ts
// Fixed Zod validation schemas with corrected regex and better type safety

import { z } from 'zod';
import { VALIDATION_RULES, FILE_CONFIG } from './constants';

// ============================================================================
// AUTH SCHEMAS
// ============================================================================

export const loginSchema = z.object({
  username: z
    .string()
    .min(VALIDATION_RULES.USERNAME.MIN_LENGTH, {
      message: `Username must be at least ${VALIDATION_RULES.USERNAME.MIN_LENGTH} characters`,
    })
    .max(VALIDATION_RULES.USERNAME.MAX_LENGTH, {
      message: `Username must be at most ${VALIDATION_RULES.USERNAME.MAX_LENGTH} characters`,
    })
    .regex(VALIDATION_RULES.USERNAME.PATTERN, {
      message: VALIDATION_RULES.USERNAME.ERROR_MESSAGE,
    }),
  password: z
    .string()
    .min(VALIDATION_RULES.PASSWORD.MIN_LENGTH, {
      message: `Password must be at least ${VALIDATION_RULES.PASSWORD.MIN_LENGTH} characters`,
    })
    .max(VALIDATION_RULES.PASSWORD.MAX_LENGTH, {
      message: `Password must be at most ${VALIDATION_RULES.PASSWORD.MAX_LENGTH} characters`,
    }),
});

export const registerSchema = loginSchema.extend({
  email: z
    .string()
    .email({ message: VALIDATION_RULES.EMAIL.ERROR_MESSAGE }),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ['confirmPassword'],
});

export const passwordResetSchema = z.object({
  email: z
    .string()
    .email({ message: VALIDATION_RULES.EMAIL.ERROR_MESSAGE }),
});

export const changePasswordSchema = z.object({
  currentPassword: z
    .string()
    .min(1, { message: 'Current password is required' }),
  newPassword: z
    .string()
    .min(VALIDATION_RULES.PASSWORD.MIN_LENGTH, {
      message: `Password must be at least ${VALIDATION_RULES.PASSWORD.MIN_LENGTH} characters`,
    })
    .regex(VALIDATION_RULES.PASSWORD.PATTERN, {
      message: VALIDATION_RULES.PASSWORD.ERROR_MESSAGE,
    }),
  confirmPassword: z.string(),
}).refine((data) => data.newPassword === data.confirmPassword, {
  message: "Passwords don't match",
  path: ['confirmPassword'],
});

// ============================================================================
// AGENT SCHEMAS
// ============================================================================

export const agentSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_RULES.AGENT_NAME.MIN_LENGTH, {
      message: VALIDATION_RULES.AGENT_NAME.ERROR_MESSAGE,
    })
    .max(VALIDATION_RULES.AGENT_NAME.MAX_LENGTH, {
      message: VALIDATION_RULES.AGENT_NAME.ERROR_MESSAGE,
    }),
  type: z.enum(['code-analyzer', 'ui-designer', 'backend-dev', 'security', 'optimizer'], {
    errorMap: () => ({ message: 'Please select a valid agent type' }),
  }),
  description: z.string().max(500, { message: 'Description must be at most 500 characters' }).optional(),
});

export const agentUpdateSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_RULES.AGENT_NAME.MIN_LENGTH)
    .max(VALIDATION_RULES.AGENT_NAME.MAX_LENGTH)
    .optional(),
  type: z.enum(['code-analyzer', 'ui-designer', 'backend-dev', 'security', 'optimizer']).optional(),
  status: z.enum(['idle', 'thinking', 'working', 'completed', 'error']).optional(),
  description: z.string().max(500).optional(),
  tasks: z.number().min(0).optional(),
  efficiency: z.number().min(0).max(100).optional(),
});

// ============================================================================
// PROJECT SCHEMAS
// ============================================================================

// FIXED: Better deadline validation that handles same-day deadlines
export const projectSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_RULES.PROJECT_NAME.MIN_LENGTH, {
      message: VALIDATION_RULES.PROJECT_NAME.ERROR_MESSAGE,
    })
    .max(VALIDATION_RULES.PROJECT_NAME.MAX_LENGTH, {
      message: VALIDATION_RULES.PROJECT_NAME.ERROR_MESSAGE,
    }),
  description: z.string().max(1000, { message: 'Description must be at most 1000 characters' }).optional(),
  status: z.enum(['planning', 'in_progress', 'completed', 'on_hold', 'archived']).optional(),
  team: z
    .number()
    .min(1, { message: 'Team must have at least 1 member' })
    .max(100, { message: 'Team size cannot exceed 100' }),
  deadline: z
    .string()
    .refine((date) => !isNaN(Date.parse(date)), { message: 'Invalid date format' })
    .refine((date) => {
      const deadlineDate = new Date(date);
      const today = new Date();
      today.setHours(0, 0, 0, 0); // Start of today
      deadlineDate.setHours(0, 0, 0, 0); // Start of deadline day
      return deadlineDate >= today; // Allow today and future dates
    }, { message: 'Deadline must be today or in the future' }),
  priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
  tags: z.array(z.string()).optional(),
});

export const projectUpdateSchema = z.object({
  name: z
    .string()
    .min(VALIDATION_RULES.PROJECT_NAME.MIN_LENGTH)
    .max(VALIDATION_RULES.PROJECT_NAME.MAX_LENGTH)
    .optional(),
  description: z.string().max(1000).optional(),
  status: z.enum(['planning', 'in_progress', 'completed', 'on_hold', 'archived']).optional(),
  progress: z.number().min(0).max(100).optional(),
  team: z.number().min(1).max(100).optional(),
  deadline: z
    .string()
    .refine((date) => !isNaN(Date.parse(date)))
    .optional(),
  priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
  tags: z.array(z.string()).optional(),
});

// ============================================================================
// FILE SCHEMAS
// ============================================================================

export const fileUploadSchema = z.object({
  file: z
    .instanceof(File)
    .refine((file) => file.size <= FILE_CONFIG.MAX_FILE_SIZE, {
      message: `File size must be less than ${FILE_CONFIG.MAX_FILE_SIZE / 1024 / 1024}MB`,
    })
    .refine(
      (file) => FILE_CONFIG.ALLOWED_FILE_TYPES.includes(file.type),
      {
        message: `File type not allowed. Allowed types: ${FILE_CONFIG.ALLOWED_EXTENSIONS.join(', ')}`,
      }
    ),
  entityType: z.enum(['agent', 'project']),
  entityId: z.number().min(1),
});

// ============================================================================
// USER PROFILE SCHEMAS
// ============================================================================

export const userProfileSchema = z.object({
  username: z
    .string()
    .min(VALIDATION_RULES.USERNAME.MIN_LENGTH)
    .max(VALIDATION_RULES.USERNAME.MAX_LENGTH)
    .regex(VALIDATION_RULES.USERNAME.PATTERN)
    .optional(),
  email: z
    .string()
    .email({ message: VALIDATION_RULES.EMAIL.ERROR_MESSAGE })
    .optional(),
  bio: z.string().max(500, { message: 'Bio must be at most 500 characters' }).optional(),
  avatar: z.string().url({ message: 'Avatar must be a valid URL' }).optional(),
});

// ============================================================================
// SETTINGS SCHEMAS
// ============================================================================

export const settingsSchema = z.object({
  theme: z.enum(['dark', 'light', 'auto']).optional(),
  animations: z.boolean().optional(),
  particles: z.boolean().optional(),
  performance: z.enum(['low', 'balanced', 'high']).optional(),
  notifications: z.boolean().optional(),
  autoAssign: z.boolean().optional(),
  language: z.string().optional(),
});

// ============================================================================
// CHAT MESSAGE SCHEMA
// ============================================================================

export const chatMessageSchema = z.object({
  message: z
    .string()
    .min(1, { message: 'Message cannot be empty' })
    .max(2000, { message: 'Message must be at most 2000 characters' }),
  agentId: z.number().min(1),
});

// ============================================================================
// VALIDATION HELPER FUNCTIONS - FIXED
// ============================================================================

/**
 * Validate data against a Zod schema with proper type narrowing
 */
export function validateData<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T; errors?: never } | { success: false; data?: never; errors: Record<string, string> } {
  const result = schema.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  const errors: Record<string, string> = {};
  result.error.errors.forEach((err) => {
    const path = err.path.join('.');
    errors[path] = err.message;
  });

  return { success: false, errors };
}

/**
 * Validate a single field
 */
export function validateField<T>(
  schema: z.ZodSchema<T>,
  fieldName: string,
  value: unknown
): string | null {
  const result = schema.safeParse({ [fieldName]: value });

  if (result.success) {
    return null;
  }

  const error = result.error.errors.find((err) => err.path[0] === fieldName);
  return error ? error.message : null;
}

/**
 * Custom email validator
 */
export function isValidEmail(email: string): boolean {
  return VALIDATION_RULES.EMAIL.PATTERN.test(email);
}

/**
 * Custom password strength checker
 */
export function checkPasswordStrength(password: string): {
  score: number;
  feedback: string[];
} {
  const feedback: string[] = [];
  let score = 0;

  if (password.length >= 8) score += 1;
  if (password.length >= 12) score += 1;
  if (/[a-z]/.test(password)) score += 1;
  if (/[A-Z]/.test(password)) score += 1;
  if (/[0-9]/.test(password)) score += 1;
  if (/[@$!%*?&#]/.test(password)) score += 1;

  if (password.length < 8) feedback.push('Use at least 8 characters');
  if (!/[a-z]/.test(password)) feedback.push('Add lowercase letters');
  if (!/[A-Z]/.test(password)) feedback.push('Add uppercase letters');
  if (!/[0-9]/.test(password)) feedback.push('Add numbers');
  if (!/[@$!%*?&#]/.test(password)) feedback.push('Add special characters');

  return { score, feedback };
}

/**
 * File size validator
 */
export function validateFileSize(file: File, maxSize: number = FILE_CONFIG.MAX_FILE_SIZE): boolean {
  return file.size <= maxSize;
}

/**
 * File type validator
 */
export function validateFileType(file: File): boolean {
  return FILE_CONFIG.ALLOWED_FILE_TYPES.includes(file.type);
}

/**
 * Get file extension
 */
export function getFileExtension(filename: string): string {
  return filename.slice(filename.lastIndexOf('.')).toLowerCase();
}

// ============================================================================
// EXPORT TYPES
// ============================================================================

export type LoginFormData = z.infer<typeof loginSchema>;
export type RegisterFormData = z.infer<typeof registerSchema>;
export type AgentFormData = z.infer<typeof agentSchema>;
export type ProjectFormData = z.infer<typeof projectSchema>;
export type UserProfileFormData = z.infer<typeof userProfileSchema>;
export type SettingsFormData = z.infer<typeof settingsSchema>;