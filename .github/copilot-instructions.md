# YMERA Enterprise - GitHub Copilot Instructions

This repository contains YMERA Enterprise, a production-ready AI-powered multi-agent platform designed for enterprise-scale automation and intelligent task orchestration.

## Project Overview

YMERA (Your Multi-Enterprise Resource Agent) is a comprehensive AI agent platform built with:
- **Backend**: Python 3.11+ with FastAPI, SQLAlchemy, and async/await patterns
- **Frontend**: TypeScript/React with modern UI components
- **AI Integration**: Anthropic Claude, OpenAI, and other LLM providers
- **Architecture**: Multi-agent system with learning capabilities, orchestration, and monitoring

## Code Style and Standards

### Python Code Standards
- Use Python 3.11+ features and type hints for all functions and methods
- Follow PEP 8 style guide; use `black` formatter (line length: 100)
- Use async/await patterns for I/O operations with FastAPI
- All functions must have comprehensive docstrings (Google style)
- Use `structlog` for structured logging, not `print()` statements
- Prefer dataclasses for data models with proper type annotations
- Use Pydantic v2 for API request/response models
- Handle exceptions gracefully with proper error logging and user-friendly messages

### TypeScript/React Standards
- Use TypeScript with strict mode enabled
- Prefer functional components with React Hooks
- Use proper type definitions; avoid `any` type
- Follow component naming convention: PascalCase for components, camelCase for utilities
- Keep components focused and single-purpose
- Use proper error boundaries for error handling

### Testing Requirements
- Write tests for new functionality using `pytest` for Python
- Use `pytest-asyncio` for async test cases
- Maintain test coverage above 70%
- Mock external API calls and database operations in tests
- Test both success and failure scenarios

## Architecture Guidelines

### Agent System
- Each agent inherits from `BaseAgent` class
- Agents must implement proper lifecycle management (startup, shutdown, cleanup)
- Use structured responses with `AgentResponse` dataclass
- Implement confidence scoring for agent outputs
- Support both synchronous and asynchronous operations

### API Design
- All endpoints must follow RESTful conventions
- Use proper HTTP status codes (200, 201, 400, 404, 500, etc.)
- Implement request validation using Pydantic models
- Include comprehensive error handling and logging
- Document endpoints with proper OpenAPI/Swagger annotations
- Use dependency injection for database sessions and shared resources

### Database Operations
- Use SQLAlchemy async sessions for database operations
- Always use connection pooling with proper limits
- Implement proper transaction management with rollback on errors
- Use migrations (Alembic) for schema changes
- Never hardcode database credentials; use environment variables

### Security Best Practices
- Never commit secrets, API keys, or credentials to the repository
- Use environment variables for all sensitive configuration
- Validate and sanitize all user inputs
- Implement proper authentication and authorization checks
- Use parameterized queries to prevent SQL injection
- Apply rate limiting on API endpoints
- Use HTTPS in production; implement proper CORS policies

## File Organization

### Python Modules
- Agent implementations: `*_agent.py` (e.g., `project_agent.py`, `orchestration_agent.py`)
- Data models: `*_models.py` (e.g., `agent_models.py`, `task_models.py`)
- API routes: `*_routes.py` (e.g., `project_routes.py`, `analysis_routes.py`)
- Configuration: `*_config.py` or `config_*.py`
- Utilities: `utils.py`, `helpers.py`
- Initialization: `*_init.py` for subsystem initialization

### Frontend Components
- Place React components in logical groupings
- Use clear, descriptive names for components and utilities
- Keep styling modular and maintainable

## Dependencies

### Adding New Dependencies
- Add to `requirements.txt` for core dependencies
- Consider security implications before adding new packages
- Pin versions for reproducibility
- Document why the dependency is needed
- Update `requirements_full.txt` for comprehensive installation

### AI/LLM Integration
- The platform supports multiple LLM providers (Anthropic, OpenAI, Groq, Gemini)
- Use the configured provider from `ymera_config_optimized.py`
- Implement proper error handling for API rate limits and failures
- Cache responses where appropriate to reduce API costs

## Documentation

- Update relevant documentation when changing functionality
- Keep docstrings synchronized with implementation
- Document breaking changes in commit messages
- Include usage examples for complex features
- Update API documentation for endpoint changes

## Performance and Optimization

- Use async/await for I/O-bound operations
- Implement caching strategies using Redis where appropriate
- Optimize database queries; avoid N+1 query problems
- Use connection pooling for database and external APIs
- Monitor memory usage for long-running agents
- Implement proper timeout handling for external calls

## Logging and Monitoring

- Use structured logging with `structlog` for all Python code
- Include relevant context in log messages (agent_id, task_id, etc.)
- Log errors with full stack traces
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Never log sensitive information (passwords, tokens, PII)

## Error Handling

- Always catch and handle exceptions appropriately
- Provide meaningful error messages to users
- Log detailed error information for debugging
- Use custom exception classes for domain-specific errors
- Implement proper cleanup in finally blocks or context managers
- Return appropriate HTTP status codes in API responses

## Commit and PR Guidelines

- Write clear, descriptive commit messages
- Keep commits focused on a single logical change
- Reference issue numbers in commits when applicable
- Ensure all tests pass before committing
- Run linters and formatters before committing
- Keep pull requests small and focused

## Environment Configuration

- Use `.env` files for local development configuration
- Never commit `.env` files; use `.env.example` as template
- Document all required environment variables
- Use `pydantic-settings` for configuration management
- Support multiple environments (development, staging, production)

## Special Considerations

### Multi-Agent Orchestration
- Agents can operate independently or as part of workflows
- Use the `OrchestrationAgent` for complex multi-step tasks
- Implement proper state management for long-running workflows
- Handle agent failures gracefully with retry mechanisms

### Learning and Continuous Improvement
- The system includes learning capabilities via `learning_engine_*.py`
- Store learned patterns and insights in the knowledge base
- Update agent behavior based on historical performance
- Implement feedback loops for continuous improvement

### Production Deployment
- The platform is designed for Replit and containerized deployments
- Use proper health checks and monitoring endpoints
- Implement graceful shutdown for all services
- Use proper resource limits and scaling strategies
- Monitor system metrics and agent performance

## Common Patterns

### Creating a New Agent
```python
from base_agent import BaseAgent
from agent_models import AgentResponse
from typing import Dict, Any

class MyCustomAgent(BaseAgent):
    """Agent description"""
    
    def __init__(self, agent_id: str = None):
        super().__init__(agent_id)
        # Initialize agent-specific resources
    
    async def process_task(self, task_data: Dict[str, Any]) -> AgentResponse:
        """Process task with proper error handling"""
        try:
            # Implementation
            pass
        except Exception as e:
            logger.error("Task processing failed", error=str(e))
            raise
```

### Creating API Endpoints
```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["resource"])

class RequestModel(BaseModel):
    field: str

@router.post("/endpoint")
async def endpoint(request: RequestModel):
    """Endpoint description"""
    try:
        # Implementation
        return {"status": "success"}
    except Exception as e:
        logger.error("Endpoint failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

## Resources

- FastAPI Documentation: https://fastapi.tiangolo.com/
- Pydantic V2: https://docs.pydantic.dev/latest/
- SQLAlchemy Async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- Anthropic Claude API: https://docs.anthropic.com/
- Structlog: https://www.structlog.org/

## Getting Help

When making changes to the codebase:
1. Review existing similar implementations
2. Follow established patterns in the codebase
3. Ensure changes align with the architecture
4. Test thoroughly before committing
5. Document non-obvious design decisions

---

**Note**: This platform handles enterprise-critical AI operations. Always prioritize code quality, security, and maintainability over rapid development.
