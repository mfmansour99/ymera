import { sql } from 'drizzle-orm';
import {
  index,
  jsonb,
  pgTable,
  timestamp,
  varchar,
  text,
  boolean,
  integer,
  uuid,
  decimal,
  pgEnum,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod";

// Enums
export const userRoleEnum = pgEnum('user_role', ['admin', 'operator', 'user', 'viewer']);
export const agentStatusEnum = pgEnum('agent_status', ['active', 'inactive', 'training', 'error']);
export const taskStatusEnum = pgEnum('task_status', ['pending', 'running', 'completed', 'failed', 'cancelled']);
export const auditActionEnum = pgEnum('audit_action', ['login', 'logout', 'create', 'update', 'delete', 'execute', 'download', 'upload']);
export const fileCategoryEnum = pgEnum('file_category', ['model', 'dataset', 'config', 'output', 'log']);

// Session storage table (required for Replit Auth)
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// User storage table (required for Replit Auth)
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  role: userRoleEnum("role").default('user'),
  isActive: boolean("is_active").default(true),
  lastLoginAt: timestamp("last_login_at"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// AI Agents table
export const agents = pgTable("agents", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name", { length: 255 }).notNull(),
  description: text("description"),
  model: varchar("model", { length: 100 }).notNull(), // e.g., 'gpt-4', 'claude-3', 'gemini-pro'
  provider: varchar("provider", { length: 50 }).notNull(), // 'openai', 'anthropic', 'google'
  config: jsonb("config").notNull(), // AI model configuration
  status: agentStatusEnum("status").default('inactive'),
  ownerId: varchar("owner_id").references(() => users.id),
  totalTasks: integer("total_tasks").default(0),
  completedTasks: integer("completed_tasks").default(0),
  failedTasks: integer("failed_tasks").default(0),
  trustScore: decimal("trust_score", { precision: 5, scale: 2 }).default('500.00'),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Workflows table
export const workflows = pgTable("workflows", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name", { length: 255 }).notNull(),
  description: text("description"),
  definition: jsonb("definition").notNull(), // Workflow steps and configuration
  ownerId: varchar("owner_id").references(() => users.id),
  isActive: boolean("is_active").default(true),
  executionCount: integer("execution_count").default(0),
  lastExecutedAt: timestamp("last_executed_at"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Tasks table
export const tasks = pgTable("tasks", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  workflowId: uuid("workflow_id").references(() => workflows.id),
  agentId: uuid("agent_id").references(() => agents.id),
  name: varchar("name", { length: 255 }).notNull(),
  description: text("description"),
  input: jsonb("input"),
  output: jsonb("output"),
  status: taskStatusEnum("status").default('pending'),
  priority: integer("priority").default(5), // 1-10 scale
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  errorMessage: text("error_message"),
  executionTimeMs: integer("execution_time_ms"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Files table for file management
export const files = pgTable("files", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  filename: varchar("filename", { length: 255 }).notNull(),
  originalName: varchar("original_name", { length: 255 }).notNull(),
  mimeType: varchar("mime_type", { length: 100 }),
  size: integer("size").notNull(),
  category: fileCategoryEnum("category").default('output'),
  version: integer("version").default(1),
  parentFileId: uuid("parent_file_id").references(() => files.id), // For versioning
  ownerId: varchar("owner_id").references(() => users.id),
  taskId: uuid("task_id").references(() => tasks.id),
  metadata: jsonb("metadata"),
  isPublic: boolean("is_public").default(false),
  downloadCount: integer("download_count").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Audit logs table
export const auditLogs = pgTable("audit_logs", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  action: auditActionEnum("action").notNull(),
  resourceType: varchar("resource_type", { length: 50 }), // 'agent', 'workflow', 'task', 'file'
  resourceId: varchar("resource_id", { length: 255 }),
  details: jsonb("details"),
  ipAddress: varchar("ip_address", { length: 45 }),
  userAgent: text("user_agent"),
  createdAt: timestamp("created_at").defaultNow(),
});

// System metrics table
export const systemMetrics = pgTable("system_metrics", {
  id: uuid("id").primaryKey().default(sql`gen_random_uuid()`),
  metricName: varchar("metric_name", { length: 100 }).notNull(),
  value: decimal("value", { precision: 10, scale: 2 }).notNull(),
  tags: jsonb("tags"), // Additional metadata
  timestamp: timestamp("timestamp").defaultNow(),
});

// Relations
export const usersRelations = relations(users, ({ many }) => ({
  agents: many(agents),
  workflows: many(workflows),
  files: many(files),
  auditLogs: many(auditLogs),
}));

export const agentsRelations = relations(agents, ({ one, many }) => ({
  owner: one(users, {
    fields: [agents.ownerId],
    references: [users.id],
  }),
  tasks: many(tasks),
}));

export const workflowsRelations = relations(workflows, ({ one, many }) => ({
  owner: one(users, {
    fields: [workflows.ownerId],
    references: [users.id],
  }),
  tasks: many(tasks),
}));

export const tasksRelations = relations(tasks, ({ one, many }) => ({
  workflow: one(workflows, {
    fields: [tasks.workflowId],
    references: [workflows.id],
  }),
  agent: one(agents, {
    fields: [tasks.agentId],
    references: [agents.id],
  }),
  files: many(files),
}));

export const filesRelations = relations(files, ({ one, many }) => ({
  owner: one(users, {
    fields: [files.ownerId],
    references: [users.id],
  }),
  task: one(tasks, {
    fields: [files.taskId],
    references: [tasks.id],
  }),
  parentFile: one(files, {
    fields: [files.parentFileId],
    references: [files.id],
  }),
  versions: many(files),
}));

export const auditLogsRelations = relations(auditLogs, ({ one }) => ({
  user: one(users, {
    fields: [auditLogs.userId],
    references: [users.id],
  }),
}));

// Zod schemas for validation
export const insertUserSchema = createInsertSchema(users);
export const selectUserSchema = createSelectSchema(users);

export const insertAgentSchema = createInsertSchema(agents);
export const selectAgentSchema = createSelectSchema(agents);

export const insertWorkflowSchema = createInsertSchema(workflows);
export const selectWorkflowSchema = createSelectSchema(workflows);

export const insertTaskSchema = createInsertSchema(tasks);
export const selectTaskSchema = createSelectSchema(tasks);

export const insertFileSchema = createInsertSchema(files);
export const selectFileSchema = createSelectSchema(files);

export const insertAuditLogSchema = createInsertSchema(auditLogs);
export const selectAuditLogSchema = createSelectSchema(auditLogs);

export const insertSystemMetricSchema = createInsertSchema(systemMetrics);
export const selectSystemMetricSchema = createSelectSchema(systemMetrics);

// Types
export type UpsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;
export type Agent = typeof agents.$inferSelect;
export type InsertAgent = typeof agents.$inferInsert;
export type Workflow = typeof workflows.$inferSelect;
export type InsertWorkflow = typeof workflows.$inferInsert;
export type Task = typeof tasks.$inferSelect;
export type InsertTask = typeof tasks.$inferInsert;
export type File = typeof files.$inferSelect;
export type InsertFile = typeof files.$inferInsert;
export type AuditLog = typeof auditLogs.$inferSelect;
export type InsertAuditLog = typeof auditLogs.$inferInsert;
export type SystemMetric = typeof systemMetrics.$inferSelect;
export type InsertSystemMetric = typeof systemMetrics.$inferInsert;
