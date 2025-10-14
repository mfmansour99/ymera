import OpenAI from "openai";
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { storage } from "../storage";
import type { Workflow, Agent, InsertTask } from "@shared/schema";

// AI model configuration
interface AIConfig {
  model: string;
  provider: 'openai' | 'anthropic' | 'google';
  apiKey?: string;
  maxTokens?: number;
  temperature?: number;
}

interface ChatOptions {
  model?: string;
  agentId?: string;
  userId: string;
  context?: any;
}

interface WorkflowExecutionResult {
  success: boolean;
  tasks: any[];
  errors: string[];
  executionTime: number;
}

export class AIOrchestrator {
  private openai: OpenAI;
  private anthropic: Anthropic;
  private gemini: GoogleGenerativeAI;

  constructor() {
    // Initialize AI clients
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    this.anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });

    this.gemini = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");
  }

  async processChat(message: string, options: ChatOptions): Promise<any> {
    const startTime = Date.now();
    
    try {
      let agent: Agent | undefined;
      if (options.agentId) {
        agent = await storage.getAgent(options.agentId);
      }

      const config: AIConfig = {
        model: options.model || agent?.model || 'gpt-4',
        provider: this.getProviderFromModel(options.model || agent?.model || 'gpt-4'),
        maxTokens: 2048,
        temperature: 0.7,
        ...agent?.config,
      };

      let response: string;

      switch (config.provider) {
        case 'openai':
          response = await this.processOpenAIChat(message, config);
          break;
        case 'anthropic':
          response = await this.processAnthropicChat(message, config);
          break;
        case 'google':
          response = await this.processGeminiChat(message, config);
          break;
        default:
          throw new Error(`Unsupported provider: ${config.provider}`);
      }

      const executionTime = Date.now() - startTime;

      // Update agent statistics if applicable
      if (agent) {
        await storage.updateAgent(agent.id, {
          totalTasks: agent.totalTasks + 1,
          completedTasks: agent.completedTasks + 1,
        });
      }

      // Log the interaction
      await storage.createAuditLog({
        userId: options.userId,
        action: 'execute',
        resourceType: 'agent',
        resourceId: agent?.id || 'system',
        details: {
          model: config.model,
          provider: config.provider,
          executionTime,
          messageLength: message.length,
          responseLength: response.length,
        },
      });

      return {
        response,
        model: config.model,
        provider: config.provider,
        executionTime,
        agentId: agent?.id,
      };
    } catch (error) {
      console.error('Error processing chat:', error);
      
      // Update agent error count if applicable
      if (options.agentId) {
        const agent = await storage.getAgent(options.agentId);
        if (agent) {
          await storage.updateAgent(agent.id, {
            failedTasks: agent.failedTasks + 1,
          });
        }
      }

      throw error;
    }
  }

  async executeWorkflow(workflow: Workflow, input: any = {}): Promise<WorkflowExecutionResult> {
    const startTime = Date.now();
    const tasks: any[] = [];
    const errors: string[] = [];

    try {
      const definition = workflow.definition as any;
      
      if (!definition.steps || !Array.isArray(definition.steps)) {
        throw new Error('Invalid workflow definition: missing steps array');
      }

      // Execute workflow steps sequentially
      let context = { ...input };
      
      for (const step of definition.steps) {
        try {
          const stepResult = await this.executeWorkflowStep(step, context, workflow.ownerId!);
          tasks.push(stepResult);
          
          // Add step result to context for next steps
          context[`step_${step.id}_result`] = stepResult.output;
        } catch (error) {
          const errorMessage = `Step ${step.id} failed: ${error.message}`;
          errors.push(errorMessage);
          console.error(errorMessage, error);
          
          if (step.required !== false) {
            break; // Stop execution on required step failure
          }
        }
      }

      // Update workflow execution count
      await storage.updateWorkflow(workflow.id, {
        executionCount: workflow.executionCount + 1,
        lastExecutedAt: new Date(),
      });

      const executionTime = Date.now() - startTime;

      return {
        success: errors.length === 0,
        tasks,
        errors,
        executionTime,
      };
    } catch (error) {
      console.error('Error executing workflow:', error);
      errors.push(`Workflow execution failed: ${error.message}`);
      
      return {
        success: false,
        tasks,
        errors,
        executionTime: Date.now() - startTime,
      };
    }
  }

  private async executeWorkflowStep(step: any, context: any, ownerId: string): Promise<any> {
    const taskData: InsertTask = {
      workflowId: step.workflowId,
      agentId: step.agentId,
      name: step.name,
      description: step.description,
      input: { ...step.input, ...context },
      status: 'running',
      priority: step.priority || 5,
      startedAt: new Date(),
    };

    const task = await storage.createTask(taskData);

    try {
      let output: any;

      switch (step.type) {
        case 'ai_chat':
          const chatResponse = await this.processChat(step.input.message, {
            model: step.model,
            agentId: step.agentId,
            userId: ownerId,
            context,
          });
          output = chatResponse;
          break;

        case 'data_processing':
          output = await this.processData(step.input, context);
          break;

        case 'api_call':
          output = await this.makeApiCall(step.input, context);
          break;

        default:
          throw new Error(`Unknown step type: ${step.type}`);
      }

      // Update task with success
      const completedTask = await storage.updateTask(task.id, {
        status: 'completed',
        output,
        completedAt: new Date(),
        executionTimeMs: Date.now() - task.startedAt!.getTime(),
      });

      return completedTask;
    } catch (error) {
      // Update task with failure
      await storage.updateTask(task.id, {
        status: 'failed',
        errorMessage: error.message,
        completedAt: new Date(),
        executionTimeMs: Date.now() - task.startedAt!.getTime(),
      });

      throw error;
    }
  }

  private async processOpenAIChat(message: string, config: AIConfig): Promise<string> {
    // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
    const response = await this.openai.chat.completions.create({
      model: config.model.startsWith('gpt-5') ? config.model : 'gpt-5',
      messages: [{ role: 'user', content: message }],
      max_completion_tokens: config.maxTokens,
    });

    return response.choices[0].message.content || 'No response';
  }

  private async processAnthropicChat(message: string, config: AIConfig): Promise<string> {
    // claude-sonnet-4-20250514 is the newest Anthropic model
    const response = await this.anthropic.messages.create({
      model: config.model.includes('claude-sonnet-4') ? config.model : 'claude-sonnet-4-20250514',
      max_tokens: config.maxTokens || 2048,
      messages: [{ role: 'user', content: message }],
    });

    return response.content[0].type === 'text' ? response.content[0].text : 'No response';
  }

  private async processGeminiChat(message: string, config: AIConfig): Promise<string> {
    const model = this.gemini.getGenerativeModel({ 
      model: config.model.includes('gemini-2.5') ? config.model : 'gemini-2.5-flash'
    });

    const result = await model.generateContent(message);
    const response = await result.response;
    return response.text();
  }

  private getProviderFromModel(model: string): 'openai' | 'anthropic' | 'google' {
    if (model.startsWith('gpt')) return 'openai';
    if (model.startsWith('claude')) return 'anthropic';
    if (model.startsWith('gemini')) return 'google';
    return 'openai'; // Default fallback
  }

  private async processData(input: any, context: any): Promise<any> {
    // Implement data processing logic
    return { processed: true, input, context };
  }

  private async makeApiCall(input: any, context: any): Promise<any> {
    // Implement API call logic
    const { url, method = 'GET', headers = {}, body } = input;
    
    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    return await response.json();
  }
}
