import {
  users,
  agents,
  workflows,
  tasks,
  files,
  auditLogs,
  systemMetrics,
  type User,
  type UpsertUser,
  type Agent,
  type InsertAgent,
  type Workflow,
  type InsertWorkflow,
  type Task,
  type InsertTask,
  type File,
  type InsertFile,
  type AuditLog,
  type InsertAuditLog,
  type SystemMetric,
  type InsertSystemMetric,
} from "@shared/schema";
import { db } from "./db";
import { eq, desc, and, or, like, count } from "drizzle-orm";

export interface IStorage {
  // User operations (required for Replit Auth)
  getUser(id: string): Promise<User | undefined>;
  upsertUser(user: UpsertUser): Promise<User>;
  
  // Agent operations
  getAgents(ownerId?: string): Promise<Agent[]>;
  getAgent(id: string): Promise<Agent | undefined>;
  createAgent(agent: InsertAgent): Promise<Agent>;
  updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent>;
  deleteAgent(id: string): Promise<void>;
  
  // Workflow operations
  getWorkflows(ownerId?: string): Promise<Workflow[]>;
  getWorkflow(id: string): Promise<Workflow | undefined>;
  createWorkflow(workflow: InsertWorkflow): Promise<Workflow>;
  updateWorkflow(id: string, updates: Partial<InsertWorkflow>): Promise<Workflow>;
  deleteWorkflow(id: string): Promise<void>;
  
  // Task operations
  getTasks(filters?: { workflowId?: string; agentId?: string; status?: string }): Promise<Task[]>;
  getTask(id: string): Promise<Task | undefined>;
  createTask(task: InsertTask): Promise<Task>;
  updateTask(id: string, updates: Partial<InsertTask>): Promise<Task>;
  deleteTask(id: string): Promise<void>;
  
  // File operations
  getFiles(ownerId?: string, taskId?: string): Promise<File[]>;
  getFile(id: string): Promise<File | undefined>;
  createFile(file: InsertFile): Promise<File>;
  updateFile(id: string, updates: Partial<InsertFile>): Promise<File>;
  deleteFile(id: string): Promise<void>;
  
  // Audit log operations
  createAuditLog(log: InsertAuditLog): Promise<AuditLog>;
  getAuditLogs(userId?: string, limit?: number): Promise<AuditLog[]>;
  
  // System metrics operations
  createSystemMetric(metric: InsertSystemMetric): Promise<SystemMetric>;
  getSystemMetrics(metricName?: string, limit?: number): Promise<SystemMetric[]>;
  
  // Dashboard statistics
  getDashboardStats(): Promise<{
    totalTasks: number;
    completedTasks: number;
    activeAgents: number;
    pendingTasks: number;
    totalUsers: number;
  }>;
}

export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async upsertUser(userData: UpsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .onConflictDoUpdate({
        target: users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }

  // Agent operations
  async getAgents(ownerId?: string): Promise<Agent[]> {
    const query = db.select().from(agents);
    if (ownerId) {
      return await query.where(eq(agents.ownerId, ownerId));
    }
    return await query.orderBy(desc(agents.createdAt));
  }

  async getAgent(id: string): Promise<Agent | undefined> {
    const [agent] = await db.select().from(agents).where(eq(agents.id, id));
    return agent;
  }

  async createAgent(agent: InsertAgent): Promise<Agent> {
    const [newAgent] = await db.insert(agents).values(agent).returning();
    return newAgent;
  }

  async updateAgent(id: string, updates: Partial<InsertAgent>): Promise<Agent> {
    const [updatedAgent] = await db
      .update(agents)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(agents.id, id))
      .returning();
    return updatedAgent;
  }

  async deleteAgent(id: string): Promise<void> {
    await db.delete(agents).where(eq(agents.id, id));
  }

  // Workflow operations
  async getWorkflows(ownerId?: string): Promise<Workflow[]> {
    const query = db.select().from(workflows);
    if (ownerId) {
      return await query.where(eq(workflows.ownerId, ownerId));
    }
    return await query.orderBy(desc(workflows.createdAt));
  }

  async getWorkflow(id: string): Promise<Workflow | undefined> {
    const [workflow] = await db.select().from(workflows).where(eq(workflows.id, id));
    return workflow;
  }

  async createWorkflow(workflow: InsertWorkflow): Promise<Workflow> {
    const [newWorkflow] = await db.insert(workflows).values(workflow).returning();
    return newWorkflow;
  }

  async updateWorkflow(id: string, updates: Partial<InsertWorkflow>): Promise<Workflow> {
    const [updatedWorkflow] = await db
      .update(workflows)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(workflows.id, id))
      .returning();
    return updatedWorkflow;
  }

  async deleteWorkflow(id: string): Promise<void> {
    await db.delete(workflows).where(eq(workflows.id, id));
  }

  // Task operations
  async getTasks(filters?: { workflowId?: string; agentId?: string; status?: string }): Promise<Task[]> {
    let query = db.select().from(tasks);
    
    if (filters) {
      const conditions = [];
      if (filters.workflowId) conditions.push(eq(tasks.workflowId, filters.workflowId));
      if (filters.agentId) conditions.push(eq(tasks.agentId, filters.agentId));
      if (filters.status) conditions.push(eq(tasks.status, filters.status as any));
      
      if (conditions.length > 0) {
        query = query.where(and(...conditions));
      }
    }
    
    return await query.orderBy(desc(tasks.createdAt));
  }

  async getTask(id: string): Promise<Task | undefined> {
    const [task] = await db.select().from(tasks).where(eq(tasks.id, id));
    return task;
  }

  async createTask(task: InsertTask): Promise<Task> {
    const [newTask] = await db.insert(tasks).values(task).returning();
    return newTask;
  }

  async updateTask(id: string, updates: Partial<InsertTask>): Promise<Task> {
    const [updatedTask] = await db
      .update(tasks)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(tasks.id, id))
      .returning();
    return updatedTask;
  }

  async deleteTask(id: string): Promise<void> {
    await db.delete(tasks).where(eq(tasks.id, id));
  }

  // File operations
  async getFiles(ownerId?: string, taskId?: string): Promise<File[]> {
    let query = db.select().from(files);
    
    const conditions = [];
    if (ownerId) conditions.push(eq(files.ownerId, ownerId));
    if (taskId) conditions.push(eq(files.taskId, taskId));
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
    }
    
    return await query.orderBy(desc(files.createdAt));
  }

  async getFile(id: string): Promise<File | undefined> {
    const [file] = await db.select().from(files).where(eq(files.id, id));
    return file;
  }

  async createFile(file: InsertFile): Promise<File> {
    const [newFile] = await db.insert(files).values(file).returning();
    return newFile;
  }

  async updateFile(id: string, updates: Partial<InsertFile>): Promise<File> {
    const [updatedFile] = await db
      .update(files)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(files.id, id))
      .returning();
    return updatedFile;
  }

  async deleteFile(id: string): Promise<void> {
    await db.delete(files).where(eq(files.id, id));
  }

  // Audit log operations
  async createAuditLog(log: InsertAuditLog): Promise<AuditLog> {
    const [newLog] = await db.insert(auditLogs).values(log).returning();
    return newLog;
  }

  async getAuditLogs(userId?: string, limit: number = 100): Promise<AuditLog[]> {
    let query = db.select().from(auditLogs);
    
    if (userId) {
      query = query.where(eq(auditLogs.userId, userId));
    }
    
    return await query.orderBy(desc(auditLogs.createdAt)).limit(limit);
  }

  // System metrics operations
  async createSystemMetric(metric: InsertSystemMetric): Promise<SystemMetric> {
    const [newMetric] = await db.insert(systemMetrics).values(metric).returning();
    return newMetric;
  }

  async getSystemMetrics(metricName?: string, limit: number = 1000): Promise<SystemMetric[]> {
    let query = db.select().from(systemMetrics);
    
    if (metricName) {
      query = query.where(eq(systemMetrics.metricName, metricName));
    }
    
    return await query.orderBy(desc(systemMetrics.timestamp)).limit(limit);
  }

  // Dashboard statistics
  async getDashboardStats(): Promise<{
    totalTasks: number;
    completedTasks: number;
    activeAgents: number;
    pendingTasks: number;
    totalUsers: number;
  }> {
    const [totalTasksResult] = await db.select({ count: count() }).from(tasks);
    const [completedTasksResult] = await db.select({ count: count() }).from(tasks).where(eq(tasks.status, 'completed'));
    const [activeAgentsResult] = await db.select({ count: count() }).from(agents).where(eq(agents.status, 'active'));
    const [pendingTasksResult] = await db.select({ count: count() }).from(tasks).where(eq(tasks.status, 'pending'));
    const [totalUsersResult] = await db.select({ count: count() }).from(users);

    return {
      totalTasks: totalTasksResult.count,
      completedTasks: completedTasksResult.count,
      activeAgents: activeAgentsResult.count,
      pendingTasks: pendingTasksResult.count,
      totalUsers: totalUsersResult.count,
    };
  }
}

export const storage = new DatabaseStorage();
