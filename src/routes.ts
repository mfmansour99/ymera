import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer } from "ws";
import express from "express";
import multer from "multer";
import { storage } from "./storage";
import { setupAuth, isAuthenticated } from "./replitAuth";
import { WebSocketManager } from "./services/websocketManager";
import { AIOrchestrator } from "./services/aiOrchestrator";
import { FileManager } from "./services/fileManager";
import {
  insertAgentSchema,
  insertWorkflowSchema,
  insertTaskSchema,
  insertFileSchema,
  insertAuditLogSchema,
  insertSystemMetricSchema,
} from "@shared/schema";

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 } // 100MB limit
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Auth middleware
  await setupAuth(app);

  const aiOrchestrator = new AIOrchestrator();
  const fileManager = new FileManager();

  // Add request logging middleware
  app.use((req, res, next) => {
    console.log(`${req.method} ${req.path}`, req.user ? `[User: ${req.user.claims?.sub}]` : '[Anonymous]');
    next();
  });

  // Auth routes
  app.get('/api/auth/user', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const user = await storage.getUser(userId);
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      res.json(user);
    } catch (error) {
      console.error("Error fetching user:", error);
      res.status(500).json({ message: "Failed to fetch user" });
    }
  });

  // Dashboard stats
  app.get('/api/dashboard/stats', isAuthenticated, async (req, res) => {
    try {
      const stats = await storage.getDashboardStats();
      res.json(stats);
    } catch (error) {
      console.error("Error fetching dashboard stats:", error);
      res.status(500).json({ message: "Failed to fetch dashboard stats" });
    }
  });

  // Agent routes
  app.get('/api/agents', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const agents = await storage.getAgents(userId);
      res.json(agents);
    } catch (error) {
      console.error("Error fetching agents:", error);
      res.status(500).json({ message: "Failed to fetch agents" });
    }
  });

  app.get('/api/agents/:id', isAuthenticated, async (req, res) => {
    try {
      const agent = await storage.getAgent(req.params.id);
      if (!agent) {
        return res.status(404).json({ message: "Agent not found" });
      }
      res.json(agent);
    } catch (error) {
      console.error("Error fetching agent:", error);
      res.status(500).json({ message: "Failed to fetch agent" });
    }
  });

  app.post('/api/agents', isAuthenticated, async (req: any, res) => {
    try {
      const agentData = insertAgentSchema.parse({
        ...req.body,
        ownerId: req.user.claims.sub,
      });
      
      const agent = await storage.createAgent(agentData);
      
      // Log audit event
      await storage.createAuditLog({
        userId: req.user.claims.sub,
        action: 'create',
        resourceType: 'agent',
        resourceId: agent.id,
        details: { agentName: agent.name },
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
      });

      res.status(201).json(agent);
    } catch (error) {
      console.error("Error creating agent:", error);
      res.status(400).json({ message: "Failed to create agent", error: error.message });
    }
  });

  app.patch('/api/agents/:id', isAuthenticated, async (req: any, res) => {
    try {
      const updates = insertAgentSchema.partial().parse(req.body);
      const agent = await storage.updateAgent(req.params.id, updates);
      
      await storage.createAuditLog({
        userId: req.user.claims.sub,
        action: 'update',
        resourceType: 'agent',
        resourceId: agent.id,
        details: updates,
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
      });

      res.json(agent);
    } catch (error) {
      console.error("Error updating agent:", error);
      res.status(400).json({ message: "Failed to update agent", error: error.message });
    }
  });

  app.delete('/api/agents/:id', isAuthenticated, async (req: any, res) => {
    try {
      await storage.deleteAgent(req.params.id);
      
      await storage.createAuditLog({
        userId: req.user.claims.sub,
        action: 'delete',
        resourceType: 'agent',
        resourceId: req.params.id,
        details: {},
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
      });

      res.status(204).send();
    } catch (error) {
      console.error("Error deleting agent:", error);
      res.status(500).json({ message: "Failed to delete agent" });
    }
  });

  // Workflow routes
  app.get('/api/workflows', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const workflows = await storage.getWorkflows(userId);
      res.json(workflows);
    } catch (error) {
      console.error("Error fetching workflows:", error);
      res.status(500).json({ message: "Failed to fetch workflows" });
    }
  });

  app.post('/api/workflows', isAuthenticated, async (req: any, res) => {
    try {
      const workflowData = insertWorkflowSchema.parse({
        ...req.body,
        ownerId: req.user.claims.sub,
      });
      
      const workflow = await storage.createWorkflow(workflowData);
      res.status(201).json(workflow);
    } catch (error) {
      console.error("Error creating workflow:", error);
      res.status(400).json({ message: "Failed to create workflow", error: error.message });
    }
  });

  // Execute workflow
  app.post('/api/workflows/:id/execute', isAuthenticated, async (req: any, res) => {
    try {
      const workflowId = req.params.id;
      const workflow = await storage.getWorkflow(workflowId);
      
      if (!workflow) {
        return res.status(404).json({ message: "Workflow not found" });
      }

      const result = await aiOrchestrator.executeWorkflow(workflow, req.body.input || {});
      res.json(result);
    } catch (error) {
      console.error("Error executing workflow:", error);
      res.status(500).json({ message: "Failed to execute workflow", error: error.message });
    }
  });

  // Task routes
  app.get('/api/tasks', isAuthenticated, async (req, res) => {
    try {
      const { workflowId, agentId, status } = req.query;
      const filters: any = {};
      if (workflowId) filters.workflowId = workflowId as string;
      if (agentId) filters.agentId = agentId as string;
      if (status) filters.status = status as string;

      const tasks = await storage.getTasks(filters);
      res.json(tasks);
    } catch (error) {
      console.error("Error fetching tasks:", error);
      res.status(500).json({ message: "Failed to fetch tasks" });
    }
  });

  app.post('/api/tasks', isAuthenticated, async (req: any, res) => {
    try {
      const taskData = insertTaskSchema.parse(req.body);
      const task = await storage.createTask(taskData);
      res.status(201).json(task);
    } catch (error) {
      console.error("Error creating task:", error);
      res.status(400).json({ message: "Failed to create task", error: error.message });
    }
  });

  // File upload/download routes
  app.post('/api/files/upload', isAuthenticated, upload.single('file'), async (req: any, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ message: "No file provided" });
      }

      const fileData = {
        filename: `${Date.now()}_${req.file.originalname}`,
        originalName: req.file.originalname,
        mimeType: req.file.mimetype,
        size: req.file.size,
        ownerId: req.user.claims.sub,
        category: req.body.category || 'output',
        metadata: req.body.metadata ? JSON.parse(req.body.metadata) : {},
      };

      const savedFile = await fileManager.saveFile(req.file.buffer, fileData);
      const file = await storage.createFile(savedFile);

      await storage.createAuditLog({
        userId: req.user.claims.sub,
        action: 'upload',
        resourceType: 'file',
        resourceId: file.id,
        details: { filename: file.originalName, size: file.size },
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
      });

      res.status(201).json(file);
    } catch (error) {
      console.error("Error uploading file:", error);
      res.status(500).json({ message: "Failed to upload file", error: error.message });
    }
  });

  app.get('/api/files', isAuthenticated, async (req: any, res) => {
    try {
      const userId = req.user.claims.sub;
      const files = await storage.getFiles(userId);
      res.json(files);
    } catch (error) {
      console.error("Error fetching files:", error);
      res.status(500).json({ message: "Failed to fetch files" });
    }
  });

  app.get('/api/files/:id/download', isAuthenticated, async (req: any, res) => {
    try {
      const file = await storage.getFile(req.params.id);
      if (!file) {
        return res.status(404).json({ message: "File not found" });
      }

      const fileContent = await fileManager.getFile(file.filename);
      
      await storage.createAuditLog({
        userId: req.user.claims.sub,
        action: 'download',
        resourceType: 'file',
        resourceId: file.id,
        details: { filename: file.originalName },
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
      });

      // Update download count
      await storage.updateFile(file.id, { 
        downloadCount: (file.downloadCount || 0) + 1 
      });

      res.setHeader('Content-Type', file.mimeType || 'application/octet-stream');
      res.setHeader('Content-Disposition', `attachment; filename="${file.originalName}"`);
      res.send(fileContent);
    } catch (error) {
      console.error("Error downloading file:", error);
      res.status(500).json({ message: "Failed to download file" });
    }
  });

  // System metrics routes
  app.get('/api/metrics', isAuthenticated, async (req, res) => {
    try {
      const { metric, limit } = req.query;
      const metrics = await storage.getSystemMetrics(
        metric as string,
        limit ? parseInt(limit as string) : 100
      );
      res.json(metrics);
    } catch (error) {
      console.error("Error fetching metrics:", error);
      res.status(500).json({ message: "Failed to fetch metrics" });
    }
  });

  app.post('/api/metrics', isAuthenticated, async (req: any, res) => {
    try {
      const metricData = insertSystemMetricSchema.parse(req.body);
      const metric = await storage.createSystemMetric(metricData);
      res.status(201).json(metric);
    } catch (error) {
      console.error("Error creating metric:", error);
      res.status(400).json({ message: "Failed to create metric", error: error.message });
    }
  });

  // Audit logs route
  app.get('/api/audit-logs', isAuthenticated, async (req: any, res) => {
    try {
      const { limit } = req.query;
      const logs = await storage.getAuditLogs(
        undefined,
        limit ? parseInt(limit as string) : 100
      );
      res.json(logs);
    } catch (error) {
      console.error("Error fetching audit logs:", error);
      res.status(500).json({ message: "Failed to fetch audit logs" });
    }
  });

  // AI model orchestration routes
  app.post('/api/ai/chat', isAuthenticated, async (req: any, res) => {
    try {
      const { message, model, agentId } = req.body;
      
      if (!message) {
        return res.status(400).json({ message: "Message is required" });
      }

      const response = await aiOrchestrator.processChat(message, {
        model: model || 'gpt-4',
        agentId,
        userId: req.user.claims.sub,
      });

      res.json(response);
    } catch (error) {
      console.error("Error processing chat:", error);
      res.status(500).json({ message: "Failed to process chat", error: error.message });
    }
  });

  // Create HTTP server
  const httpServer = createServer(app);

  // Setup WebSocket server on /ws path
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });
  const wsManager = new WebSocketManager(wss);

  // Start real-time metrics broadcasting
  setInterval(() => {
    wsManager.broadcastMetrics();
  }, 5000); // Every 5 seconds

  return httpServer;
}
