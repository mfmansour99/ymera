"""
YMERA Enterprise Project Builder Utility
Production-Ready Project Building System with Multi-Agent Integration
"""

import os
import json
import yaml
import asyncio
import tempfile
import shutil
import subprocess
import hashlib
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
import aiohttp
import git
from jinja2 import Environment, FileSystemLoader, Template
import docker
from kubernetes import client as k8s_client, config as k8s_config
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
import logging
from functools import wraps
import time
import psutil

# Core system imports (from your main architecture)
from ymera_core.config import ConfigManager
from ymera_core.database.manager import DatabaseManager
from ymera_core.logging.structured_logger import StructuredLogger
from ymera_core.cache.redis_cache import RedisCacheManager
from ymera_core.exceptions import YMERAException

# Agent system imports
from ymera_agents.orchestrator import AgentOrchestrator
from ymera_agents.learning.learning_engine import LearningEngine
from ymera_agents.communication.message_bus import MessageBus

# Service imports
from ymera_services.ai.multi_llm_manager import MultiLLMManager
from ymera_services.github.repository_analyzer import GitHubRepositoryAnalyzer
from ymera_services.code_analysis.quality_analyzer import CodeQualityAnalyzer
from ymera_services.security.vulnerability_scanner import VulnerabilityScanner
from ymera_services.deployment.pipeline_manager import DeploymentPipelineManager


class ProjectType(Enum):
    """Project types supported by the builder"""
    WEB_APPLICATION = "web_application"
    API_SERVICE = "api_service"
    MICROSERVICE = "microservice"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    MACHINE_LEARNING = "machine_learning"
    DATA_PIPELINE = "data_pipeline"
    BLOCKCHAIN = "blockchain"
    GAME = "game"
    LIBRARY = "library"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"


class BuildStage(Enum):
    """Build stages in the pipeline"""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    SCAFFOLDING = "scaffolding"
    DEVELOPMENT = "development"
    TESTING = "testing"
    SECURITY_SCAN = "security_scan"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    PACKAGING = "packaging"
    DEPLOYMENT = "deployment"
    VALIDATION = "validation"
    COMPLETION = "completion"


class BuildStatus(Enum):
    """Build status states"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ProjectConfiguration:
    """Comprehensive project configuration"""
    name: str
    project_type: ProjectType
    description: str = ""
    version: str = "1.0.0"
    
    # Technical specifications
    tech_stack: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    
    # Build configuration
    build_tools: List[str] = field(default_factory=list)
    test_frameworks: List[str] = field(default_factory=list)
    deployment_targets: List[str] = field(default_factory=list)
    
    # Quality requirements
    code_coverage_threshold: float = 0.8
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    security_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Learning configuration
    learning_enabled: bool = True
    feedback_collection: bool = True
    continuous_improvement: bool = True
    
    # Deployment configuration
    containerization: Dict[str, Any] = field(default_factory=dict)
    cloud_deployment: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildContext:
    """Build execution context"""
    project_id: str
    build_id: str
    configuration: ProjectConfiguration
    workspace_path: Path
    output_path: Path
    temp_path: Path
    stage: BuildStage = BuildStage.INITIALIZATION
    status: BuildStatus = BuildStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    error_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    agent_interactions: List[Dict[str, Any]] = field(default_factory=list)


class ProjectTemplateManager:
    """Manages project templates and scaffolding"""
    
    def __init__(self, templates_path: Path, ai_manager: MultiLLMManager):
        self.templates_path = templates_path
        self.ai_manager = ai_manager
        self.jinja_env = Environment(loader=FileSystemLoader(str(templates_path)))
        self.template_cache = {}
    
    async def initialize(self):
        """Initialize template manager"""
        self.templates_path.mkdir(parents=True, exist_ok=True)
        await self._load_builtin_templates()
        await self._generate_ai_templates()
    
    async def _load_builtin_templates(self):
        """Load built-in project templates"""
        builtin_templates = {
            ProjectType.WEB_APPLICATION: {
                "react_typescript": "templates/web/react-typescript",
                "vue_typescript": "templates/web/vue-typescript",
                "angular": "templates/web/angular",
                "nextjs": "templates/web/nextjs"
            },
            ProjectType.API_SERVICE: {
                "fastapi": "templates/api/fastapi",
                "express_typescript": "templates/api/express-typescript",
                "spring_boot": "templates/api/spring-boot",
                "django_rest": "templates/api/django-rest"
            },
            ProjectType.MICROSERVICE: {
                "kubernetes_service": "templates/microservice/k8s-service",
                "docker_compose": "templates/microservice/docker-compose",
                "serverless": "templates/microservice/serverless"
            },
            ProjectType.MACHINE_LEARNING: {
                "pytorch_project": "templates/ml/pytorch",
                "tensorflow_project": "templates/ml/tensorflow",
                "scikit_learn": "templates/ml/scikit-learn",
                "mlflow_pipeline": "templates/ml/mlflow"
            }
        }
        
        for project_type, templates in builtin_templates.items():
            for template_name, template_path in templates.items():
                full_path = self.templates_path / template_path
                full_path.mkdir(parents=True, exist_ok=True)
                await self._create_template_structure(full_path, project_type, template_name)
    
    async def _generate_ai_templates(self):
        """Generate templates using AI for specific requirements"""
        template_requests = [
            {
                "type": ProjectType.BLOCKCHAIN,
                "name": "ethereum_dapp",
                "description": "Ethereum DApp with smart contracts and React frontend"
            },
            {
                "type": ProjectType.GAME,
                "name": "unity_3d_game",
                "description": "Unity 3D game with C# scripts and asset management"
            },
            {
                "type": ProjectType.DATA_PIPELINE,
                "name": "airflow_pipeline",
                "description": "Apache Airflow data pipeline with monitoring"
            }
        ]
        
        for request in template_requests:
            await self._generate_template_with_ai(request)
    
    async def _create_template_structure(self, path: Path, project_type: ProjectType, template_name: str):
        """Create template directory structure"""
        # Common structure for all templates
        directories = [
            "src", "tests", "docs", "scripts", "config", 
            ".github/workflows", "deployment", "monitoring"
        ]
        
        for directory in directories:
            (path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create template files
        await self._create_template_files(path, project_type, template_name)
    
    async def _create_template_files(self, path: Path, project_type: ProjectType, template_name: str):
        """Create template files with proper content"""
        template_files = {
            "README.md": self._generate_readme_template(project_type, template_name),
            "pyproject.toml": self._generate_pyproject_template(project_type),
            "Dockerfile": self._generate_dockerfile_template(project_type),
            "docker-compose.yml": self._generate_docker_compose_template(project_type),
            ".gitignore": self._generate_gitignore_template(project_type),
            "requirements.txt": self._generate_requirements_template(project_type),
            "Makefile": self._generate_makefile_template(project_type),
            ".pre-commit-config.yaml": self._generate_precommit_template(),
            "pytest.ini": self._generate_pytest_template(),
            "setup.py": self._generate_setup_template(project_type)
        }
        
        for filename, content in template_files.items():
            file_path = path / filename
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(content)
    
    def _generate_readme_template(self, project_type: ProjectType, template_name: str) -> str:
        return f"""# {{{{ project_name }}}}

{{{{ project_description }}}}

## Project Type
{project_type.value} - {template_name}

## Features
{{%- for feature in features %}}
- {{{{ feature }}}}
{{%- endfor %}}

## Quick Start

### Prerequisites
{{%- for prereq in prerequisites %}}
- {{{{ prereq }}}}
{{%- endfor %}}

### Installation
```bash
# Clone the repository
git clone {{{{ repository_url }}}}
cd {{{{ project_name }}}}

# Install dependencies
{{{{ install_command }}}}

# Run development server
{{{{ dev_command }}}}
```

### Testing
```bash
{{{{ test_command }}}}
```

### Deployment
```bash
{{{{ deploy_command }}}}
```

## Architecture
{{{{ architecture_description }}}}

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License
{{{{ license }}}}
"""

    def _generate_pyproject_template(self, project_type: ProjectType) -> str:
        return """[tool.poetry]
name = "{{ project_name }}"
version = "{{ project_version }}"
description = "{{ project_description }}"
authors = ["{{ author_name }} <{{ author_email }}>"]
license = "{{ license }}"
readme = "README.md"
homepage = "{{ homepage_url }}"
repository = "{{ repository_url }}"
documentation = "{{ docs_url }}"
classifiers = [
    "Development Status :: {{ development_status }}",
    "Intended Audience :: {{ intended_audience }}",
    "License :: OSI Approved :: {{ license }}",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[tool.poetry.dependencies]
python = "^3.9"
{% for dep in dependencies -%}
{{ dep }}
{% endfor %}

[tool.poetry.group.dev.dependencies]
{% for dev_dep in dev_dependencies -%}
{{ dev_dep }}
{% endfor %}

[tool.poetry.scripts]
{{ project_name }} = "{{ project_name }}.cli:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov={{ project_name }} --cov-report=html --cov-report=term-missing"

[tool.coverage.run]
source = ["{{ project_name }}"]
omit = ["*/tests/*", "*/test_*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
"""

    def _generate_dockerfile_template(self, project_type: ProjectType) -> str:
        if project_type in [ProjectType.WEB_APPLICATION, ProjectType.API_SERVICE]:
            return """# Multi-stage build for {{ project_name }}
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only=main --no-dev

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{{ port }}/health || exit 1

# Expose port
EXPOSE {{ port }}

# Run application
CMD ["{{ start_command }}"]
"""
        return """FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    {{ system_dependencies | join(' ') }} \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy application files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*

# Expose necessary ports
EXPOSE {{ port }}

# Run application
CMD ["{{ start_command }}"]
"""

    def _generate_docker_compose_template(self, project_type: ProjectType) -> str:
        return """version: '3.8'

services:
  {{ project_name }}:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: {{ project_name }}
    restart: unless-stopped
    ports:
      - "{{ port }}:{{ port }}"
    environment:
      - NODE_ENV={{ environment }}
      - DATABASE_URL={{ database_url }}
      - REDIS_URL={{ redis_url }}
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    depends_on:
      - database
      - redis
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{{ port }}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  database:
    image: postgres:15-alpine
    container_name: {{ project_name }}-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: {{ db_name }}
      POSTGRES_USER: {{ db_user }}
      POSTGRES_PASSWORD: {{ db_password }}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    container_name: {{ project_name }}-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass {{ redis_password }}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    container_name: {{ project_name }}-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - {{ project_name }}
    networks:
      - app-network

volumes:
  postgres_data:
  redis_data:

networks:
  app-network:
    driver: bridge
"""

    def _generate_gitignore_template(self, project_type: ProjectType) -> str:
        common_ignores = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Temporary files
tmp/
temp/
.tmp/
"""

        if project_type == ProjectType.WEB_APPLICATION:
            return common_ignores + """
# Dependencies
node_modules/
jspm_packages/

# Production builds
/build
/dist
.next/
out/

# Runtime data
.npm
.eslintcache
.parcel-cache
"""

        return common_ignores

    def _generate_requirements_template(self, project_type: ProjectType) -> str:
        common_reqs = """# Core dependencies
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
sqlalchemy>=2.0.23
alembic>=1.13.0
redis>=5.0.1
celery>=5.3.4
"""

        if project_type == ProjectType.API_SERVICE:
            return common_reqs + """
# API specific
httpx>=0.25.2
python-multipart>=0.0.6
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-keycloak>=3.7.0

# Database drivers
asyncpg>=0.29.0
psycopg2-binary>=2.9.9

# Monitoring
prometheus-client>=0.19.0
opentelemetry-api>=1.21.0
"""

        return common_reqs

    def _generate_makefile_template(self, project_type: ProjectType) -> str:
        return """.PHONY: help install test lint format security build deploy clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

install: ## Install dependencies
	poetry install
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov={{ project_name }} --cov-report=html

lint: ## Run linting
	flake8 {{ project_name }} tests/
	mypy {{ project_name }}
	bandit -r {{ project_name }}

format: ## Format code
	black {{ project_name }} tests/
	isort {{ project_name }} tests/

security: ## Run security checks
	safety check
	bandit -r {{ project_name }}

build: ## Build application
	docker build -t {{ project_name }}:latest .

deploy: ## Deploy application
	docker-compose up -d

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
"""

    def _generate_precommit_template(self) -> str:
        return """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
"""

    def _generate_pytest_template(self) -> str:
        return """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --disable-warnings
    --cov={{ project_name }}
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
"""

    def _generate_setup_template(self, project_type: ProjectType) -> str:
        return """#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{{ project_name }}",
    version="{{ project_version }}",
    author="{{ author_name }}",
    author_email="{{ author_email }}",
    description="{{ project_description }}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="{{ repository_url }}",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: {{ development_status }}",
        "Intended Audience :: {{ intended_audience }}",
        "License :: OSI Approved :: {{ license }}",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pre-commit>=3.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "{{ project_name }}={{ project_name }}.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
"""

    async def get_template(self, project_type: ProjectType, template_name: str) -> Optional[Path]:
        """Get template path for project type and template name"""
        template_key = f"{project_type.value}_{template_name}"
        
        if template_key in self.template_cache:
            return self.template_cache[template_key]
        
        template_path = self.templates_path / "templates" / project_type.value.replace("_", "-") / template_name
        
        if template_path.exists():
            self.template_cache[template_key] = template_path
            return template_path
        
        return None

    async def render_template(self, template_path: Path, context: Dict[str, Any]) -> Dict[str, str]:
        """Render template files with context"""
        rendered_files = {}
        
        for file_path in template_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(template_path)
                
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                
                template = Template(content)
                rendered_content = template.render(**context)
                rendered_files[str(relative_path)] = rendered_content
        
        return rendered_files

    async def _generate_template_with_ai(self, request: Dict[str, Any]):
        """Generate template using AI assistance"""
        prompt = f"""
        Generate a comprehensive project template for:
        - Type: {request['type'].value}
        - Name: {request['name']}
        - Description: {request['description']}
        
        Create a production-ready template with:
        1. Proper project structure
        2. Configuration files
        3. Build scripts
        4. Testing setup
        5. Documentation
        6. Deployment configuration
        7. Quality assurance tools
        
        Focus on enterprise-grade standards and best practices.
        """
        
        try:
            response = await self.ai_manager.generate_completion(
                prompt=prompt,
                model_params={
                    "temperature": 0.3,
                    "max_tokens": 4000
                }
            )
            
            # Process AI response and create template
            # Implementation would parse the AI response and create actual template files
            template_path = self.templates_path / "ai_generated" / request['name']
            template_path.mkdir(parents=True, exist_ok=True)
            
            # Save AI-generated template content
            await self._save_ai_template(template_path, response.content, request)
            
        except Exception as e:
            logging.error(f"Failed to generate AI template for {request['name']}: {str(e)}")

    async def _save_ai_template(self, path: Path, content: str, request: Dict[str, Any]):
        """Save AI-generated template content"""
        # Parse AI response and create template files
        # This is a simplified implementation - in production, you'd need proper parsing
        template_info = {
            "name": request['name'],
            "type": request['type'].value,
            "description": request['description'],
            "generated_at": datetime.utcnow().isoformat(),
            "ai_generated": True,
            "content": content
        }
        
        metadata_file = path / "template_metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(template_info, indent=2))


class AgentIntegrationManager:
    """Manages integration with YMERA agent system"""
    
    def __init__(
        self,
        agent_orchestrator: AgentOrchestrator,
        learning_engine: LearningEngine,
        message_bus: MessageBus,
        logger: StructuredLogger
    ):
        self.agent_orchestrator = agent_orchestrator
        self.learning_engine = learning_engine
        self.message_bus = message_bus
        self.logger = logger
        self.active_workflows = {}
    
    async def coordinate_build_workflow(self, context: BuildContext) -> Dict[str, Any]:
        """Coordinate multi-agent build workflow"""
        workflow_id = f"build_{context.build_id}"
        
        try:
            # Define agent workflow based on project type and requirements
            workflow_plan = await self._create_workflow_plan(context)
            
            # Register workflow with orchestrator
            self.active_workflows[workflow_id] = {
                "context": context,
                "plan": workflow_plan,
                "status": "active",
                "start_time": datetime.utcnow(),
                "agent_results": {}
            }
            
            # Execute workflow stages
            results = {}
            for stage in workflow_plan['stages']:
                stage_result = await self._execute_workflow_stage(workflow_id, stage, context)
                results[stage['name']] = stage_result
                
                # Update learning engine with stage completion
                await self.learning_engine.record_workflow_stage_completion(
                    workflow_id=workflow_id,
                    stage=stage['name'],
                    result=stage_result,
                    context=context.configuration.__dict__
                )
                
                # Check for stage failure
                if stage_result.get('status') == 'failed' and stage.get('critical', False):
                    raise Exception(f"Critical stage {stage['name']} failed: {stage_result.get('error')}")
            
            # Mark workflow as completed
            self.active_workflows[workflow_id]['status'] = 'completed'
            self.active_workflows[workflow_id]['end_time'] = datetime.utcnow()
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "results": results,
                "duration": (self.active_workflows[workflow_id]['end_time'] - 
                           self.active_workflows[workflow_id]['start_time']).total_seconds()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            self.active_workflows[workflow_id]['status'] = 'failed'
            self.active_workflows[workflow_id]['error'] = str(e)
            
            # Learn from failure
            await self.learning_engine.record_workflow_failure(
                workflow_id=workflow_id,
                error=str(e),
                context=context.configuration.__dict__
            )
            
            raise
    
    async def _create_workflow_plan(self, context: BuildContext) -> Dict[str, Any]:
        """Create workflow execution plan based on project configuration"""
        project_type = context.configuration.project_type
        
        # Base workflow stages
        base_stages = [
            {
                "name": "project_analysis",
                "agent": "analysis_agent",
                "critical": True,
                "inputs": {
                    "project_config": context.configuration.__dict__,
                    "workspace_path": str(context.workspace_path)
                },
                "outputs": ["analysis_report", "recommendations"]
            },
            {
                "name": "project_planning",
                "agent": "project_management_agent",
                "critical": True,
                "inputs": {
                    "analysis_report": "project_analysis.analysis_report",
                    "requirements": context.configuration.tech_stack
                },
                "outputs": ["project_plan", "task_breakdown"]
            },
            {
                "name": "code_scaffolding",
                "agent": "enhancement_agent",
                "critical": True,
                "inputs": {
                    "project_plan": "project_planning.project_plan",
                    "template_config": context.configuration.__dict__
                },
                "outputs": ["scaffolded_code", "file_structure"]
            },
            {
                "name": "security_review",
                "agent": "security_agent",
                "critical": False,
                "inputs": {
                    "code_base": "code_scaffolding.scaffolded_code",
                    "security_requirements": context.configuration.security_requirements
                },
                "outputs": ["security_report", "vulnerability_fixes"]
            },
            {
                "name": "quality_validation",
                "agent": "validation_agent",
                "critical": True,
                "inputs": {
                    "code_base": "code_scaffolding.scaffolded_code",
                    "quality_requirements": {
                        "coverage_threshold": context.configuration.code_coverage_threshold
                    }
                },
                "outputs": ["validation_report", "quality_metrics"]
            },
            {
                "name": "documentation_generation",
                "agent": "documentation_agent",
                "critical": False,
                "inputs": {
                    "code_base": "code_scaffolding.scaffolded_code",
                    "project_plan": "project_planning.project_plan"
                },
                "outputs": ["documentation", "api_docs"]
            }
        ]
        
        # Add deployment stage if deployment targets are specified
        if context.configuration.deployment_targets:
            base_stages.append({
                "name": "deployment_preparation",
                "agent": "deployment_agent",
                "critical": False,
                "inputs": {
                    "code_base": "code_scaffolding.scaffolded_code",
                    "deployment_config": context.configuration.deployment_targets
                },
                "outputs": ["deployment_artifacts", "deployment_scripts"]
            })
        
        # Project-specific customizations
        custom_stages = await self._get_project_specific_stages(project_type, context)
        
        return {
            "workflow_id": f"build_{context.build_id}",
            "project_type": project_type.value,
            "stages": base_stages + custom_stages,
            "parallel_execution": False,  # Sequential for now
            "timeout_minutes": 60,
            "retry_count": 2
        }
    
    async def _get_project_specific_stages(self, project_type: ProjectType, context: BuildContext) -> List[Dict[str, Any]]:
        """Get project-type specific workflow stages"""
        custom_stages = []
        
        if project_type == ProjectType.MACHINE_LEARNING:
            custom_stages.extend([
                {
                    "name": "ml_model_validation",
                    "agent": "examination_agent",
                    "critical": True,
                    "inputs": {
                        "model_code": "code_scaffolding.scaffolded_code",
                        "validation_requirements": context.configuration.performance_requirements
                    },
                    "outputs": ["model_validation_report", "performance_metrics"]
                },
                {
                    "name": "data_pipeline_setup",
                    "agent": "enhancement_agent",
                    "critical": True,
                    "inputs": {
                        "ml_requirements": context.configuration.tech_stack.get("ml_frameworks", [])
                    },
                    "outputs": ["data_pipeline", "preprocessing_scripts"]
                }
            ])
        
        elif project_type == ProjectType.WEB_APPLICATION:
            custom_stages.extend([
                {
                    "name": "frontend_optimization",
                    "agent": "enhancement_agent",
                    "critical": False,
                    "inputs": {
                        "frontend_code": "code_scaffolding.scaffolded_code",
                        "performance_targets": context.configuration.performance_requirements
                    },
                    "outputs": ["optimized_frontend", "performance_report"]
                },
                {
                    "name": "accessibility_check",
                    "agent": "validation_agent",
                    "critical": False,
                    "inputs": {
                        "ui_code": "code_scaffolding.scaffolded_code"
                    },
                    "outputs": ["accessibility_report", "a11y_fixes"]
                }
            ])
        
        elif project_type == ProjectType.MICROSERVICE:
            custom_stages.extend([
                {
                    "name": "service_mesh_config",
                    "agent": "deployment_agent",
                    "critical": True,
                    "inputs": {
                        "service_code": "code_scaffolding.scaffolded_code",
                        "mesh_requirements": context.configuration.deployment_targets
                    },
                    "outputs": ["service_mesh_config", "istio_configs"]
                },
                {
                    "name": "observability_setup",
                    "agent": "monitoring_agent",
                    "critical": False,
                    "inputs": {
                        "service_config": "service_mesh_config.service_mesh_config"
                    },
                    "outputs": ["monitoring_config", "telemetry_setup"]
                }
            ])
        
        return custom_stages
    
    async def _execute_workflow_stage(self, workflow_id: str, stage: Dict[str, Any], context: BuildContext) -> Dict[str, Any]:
        """Execute a single workflow stage"""
        stage_start_time = datetime.utcnow()
        
        try:
            # Get agent for this stage
            agent = await self.agent_orchestrator.get_agent(stage['agent'])
            if not agent:
                raise Exception(f"Agent {stage['agent']} not found")
            
            # Prepare stage inputs
            stage_inputs = await self._resolve_stage_inputs(workflow_id, stage['inputs'])
            
            # Execute stage
            self.logger.info(f"Executing stage {stage['name']} with agent {stage['agent']}")
            
            result = await agent.execute_task({
                "task_type": "workflow_stage",
                "stage_name": stage['name'],
                "inputs": stage_inputs,
                "context": context.__dict__,
                "workflow_id": workflow_id
            })
            
            stage_duration = (datetime.utcnow() - stage_start_time).total_seconds()
            
            stage_result = {
                "status": "completed",
                "result": result,
                "duration": stage_duration,
                "outputs": stage.get('outputs', []),
                "agent": stage['agent']
            }
            
            # Store stage results for future stages
            self.active_workflows[workflow_id]['agent_results'][stage['name']] = stage_result
            
            self.logger.info(f"Stage {stage['name']} completed successfully in {stage_duration:.2f}s")
            
            return stage_result
            
        except Exception as e:
            stage_duration = (datetime.utcnow() - stage_start_time).total_seconds()
            
            self.logger.error(f"Stage {stage['name']} failed after {stage_duration:.2f}s: {str(e)}")
            
            return {
                "status": "failed",
                "error": str(e),
                "duration": stage_duration,
                "agent": stage['agent']
            }
    
    async def _resolve_stage_inputs(self, workflow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve stage inputs from previous stage outputs"""
        resolved_inputs = {}
        
        for input_name, input_value in inputs.items():
            if isinstance(input_value, str) and '.' in input_value:
                # This is a reference to previous stage output
                stage_name, output_key = input_value.split('.', 1)
                
                if workflow_id in self.active_workflows:
                    stage_results = self.active_workflows[workflow_id]['agent_results']
                    
                    if stage_name in stage_results:
                        stage_result = stage_results[stage_name]
                        if 'result' in stage_result and output_key in stage_result['result']:
                            resolved_inputs[input_name] = stage_result['result'][output_key]
                        else:
                            self.logger.warning(f"Output {output_key} not found in stage {stage_name}")
                            resolved_inputs[input_name] = None
                    else:
                        self.logger.warning(f"Stage {stage_name} results not found")
                        resolved_inputs[input_name] = None
                else:
                    resolved_inputs[input_name] = input_value
            else:
                resolved_inputs[input_name] = input_value
        
        return resolved_inputs
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow status"""
        if workflow_id not in self.active_workflows:
            return {"status": "not_found"}
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "status": workflow['status'],
            "start_time": workflow['start_time'].isoformat(),
            "end_time": workflow.get('end_time', {}).isoformat() if workflow.get('end_time') else None,
            "stages_completed": len(workflow['agent_results']),
            "total_stages": len(workflow['plan']['stages']),
            "current_stage": self._get_current_stage(workflow),
            "error": workflow.get('error')
        }
    
    def _get_current_stage(self, workflow: Dict[str, Any]) -> Optional[str]:
        """Get currently executing stage"""
        completed_stages = set(workflow['agent_results'].keys())
        
        for stage in workflow['plan']['stages']:
            if stage['name'] not in completed_stages:
                return stage['name']
        
        return None


class BuildMetricsCollector:
    """Collects and analyzes build metrics"""
    
    def __init__(self, cache_manager: RedisCacheManager, logger: StructuredLogger):
        self.cache_manager = cache_manager
        self.logger = logger
        self.metrics_key_prefix = "build_metrics"
    
    async def record_build_start(self, context: BuildContext):
        """Record build start metrics"""
        metrics = {
            "build_id": context.build_id,
            "project_id": context.project_id,
            "project_type": context.configuration.project_type.value,
            "start_time": datetime.utcnow().isoformat(),
            "configuration": {
                "tech_stack": context.configuration.tech_stack,
                "dependencies_count": len(context.configuration.dependencies),
                "deployment_targets": context.configuration.deployment_targets
            }
        }
        
        await self.cache_manager.set(
            f"{self.metrics_key_prefix}:start:{context.build_id}",
            json.dumps(metrics),
            ttl=3600  # 1 hour
        )
    
    async def record_stage_metrics(self, build_id: str, stage: BuildStage, duration: float, status: BuildStatus):
        """Record stage execution metrics"""
        stage_metrics = {
            "build_id": build_id,
            "stage": stage.value,
            "duration": duration,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store individual stage metrics
        await self.cache_manager.set(
            f"{self.metrics_key_prefix}:stage:{build_id}:{stage.value}",
            json.dumps(stage_metrics),
            ttl=3600
        )
        
        # Update aggregated metrics
        await self._update_aggregated_metrics(stage, duration, status)
    
    async def record_build_completion(self, context: BuildContext, success: bool, total_duration: float):
        """Record build completion metrics"""
        completion_metrics = {
            "build_id": context.build_id,
            "project_id": context.project_id,
            "success": success,
            "total_duration": total_duration,
            "end_time": datetime.utcnow().isoformat(),
            "artifacts_count": len(context.artifacts),
            "final_stage": context.stage.value,
            "error_message": context.error_message if not success else None
        }
        
        await self.cache_manager.set(
            f"{self.metrics_key_prefix}:completion:{context.build_id}",
            json.dumps(completion_metrics),
            ttl=86400  # 24 hours
        )
        
        # Update success/failure rates
        await self._update_success_metrics(context.configuration.project_type, success)
    
    async def _update_aggregated_metrics(self, stage: BuildStage, duration: float, status: BuildStatus):
        """Update aggregated stage performance metrics"""
        key = f"{self.metrics_key_prefix}:aggregated:stages"
        
        try:
            existing_data = await self.cache_manager.get(key)
            metrics = json.loads(existing_data) if existing_data else {}
        except:
            metrics = {}
        
        stage_key = stage.value
        if stage_key not in metrics:
            metrics[stage_key] = {
                "total_executions": 0,
                "total_duration": 0.0,
                "success_count": 0,
                "failure_count": 0,
                "average_duration": 0.0
            }
        
        stage_metrics = metrics[stage_key]
        stage_metrics["total_executions"] += 1
        stage_metrics["total_duration"] += duration
        
        if status == BuildStatus.SUCCESS:
            stage_metrics["success_count"] += 1
        else:
            stage_metrics["failure_count"] += 1
        
        stage_metrics["average_duration"] = stage_metrics["total_duration"] / stage_metrics["total_executions"]
        stage_metrics["success_rate"] = stage_metrics["success_count"] / stage_metrics["total_executions"]
        
        await self.cache_manager.set(key, json.dumps(metrics), ttl=86400 * 7)  # 1 week
    
    async def _update_success_metrics(self, project_type: ProjectType, success: bool):
        """Update project type success metrics"""
        key = f"{self.metrics_key_prefix}:success_rates"
        
        try:
            existing_data = await self.cache_manager.get(key)
            metrics = json.loads(existing_data) if existing_data else {}
        except:
            metrics = {}
        
        type_key = project_type.value
        if type_key not in metrics:
            metrics[type_key] = {
                "total_builds": 0,
                "successful_builds": 0,
                "failed_builds": 0,
                "success_rate": 0.0
            }
        
        type_metrics = metrics[type_key]
        type_metrics["total_builds"] += 1
        
        if success:
            type_metrics["successful_builds"] += 1
        else:
            type_metrics["failed_builds"] += 1
        
        type_metrics["success_rate"] = type_metrics["successful_builds"] / type_metrics["total_builds"]
        
        await self.cache_manager.set(key, json.dumps(metrics), ttl=86400 * 30)  # 30 days
    
    async def get_build_analytics(self, project_type: Optional[ProjectType] = None) -> Dict[str, Any]:
        """Get comprehensive build analytics"""
        analytics = {
            "stage_performance": await self._get_stage_analytics(),
            "success_rates": await self._get_success_analytics(project_type),
            "trend_analysis": await self._get_trend_analytics(),
            "resource_utilization": await self._get_resource_analytics()
        }
        
        return analytics
    
    async def _get_stage_analytics(self) -> Dict[str, Any]:
        """Get stage performance analytics"""
        key = f"{self.metrics_key_prefix}:aggregated:stages"
        data = await self.cache_manager.get(key)
        
        if not data:
            return {}
        
        return json.loads(data)
    
    async def _get_success_analytics(self, project_type: Optional[ProjectType] = None) -> Dict[str, Any]:
        """Get success rate analytics"""
        key = f"{self.metrics_key_prefix}:success_rates"
        data = await self.cache_manager.get(key)
        
        if not data:
            return {}
        
        metrics = json.loads(data)
        
        if project_type:
            return metrics.get(project_type.value, {})
        
        return metrics
    
    async def _get_trend_analytics(self) -> Dict[str, Any]:
        """Get trend analytics over time"""
        # Implementation would analyze time-series data
        # For now, return placeholder
        return {
            "build_volume_trend": "stable",
            "success_rate_trend": "improving",
            "average_duration_trend": "decreasing"
        }
    
    async def _get_resource_analytics(self) -> Dict[str, Any]:
        """Get resource utilization analytics"""
        return {
            "cpu_utilization": psutil.cpu_percent(),
            "memory_utilization": psutil.virtual_memory().percent,
            "disk_utilization": psutil.disk_usage('/').percent
        }


class ProjectBuilder:
    """Main project builder class that orchestrates the entire build process"""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        db_manager: DatabaseManager,
        ai_manager: MultiLLMManager,
        agent_orchestrator: AgentOrchestrator,
        learning_engine: LearningEngine,
        message_bus: MessageBus,
        cache_manager: RedisCacheManager,
        github_analyzer: GitHubRepositoryAnalyzer,
        code_analyzer: CodeQualityAnalyzer,
        vulnerability_scanner: VulnerabilityScanner,
        deployment_pipeline: DeploymentPipelineManager,
        logger: StructuredLogger
    ):
        self.config = config_manager
        self.db_manager = db_manager
        self.ai_manager = ai_manager
        self.agent_orchestrator = agent_orchestrator
        self.learning_engine = learning_engine
        self.message_bus = message_bus
        self.cache_manager = cache_manager
        self.github_analyzer = github_analyzer
        self.code_analyzer = code_analyzer
        self.vulnerability_scanner = vulnerability_scanner
        self.deployment_pipeline = deployment_pipeline
        self.logger = logger
        
        # Initialize components
        self.template_manager = None
        self.agent_integration = None
        self.metrics_collector = None
        self.docker_client = None
        self.active_builds = {}
    
    async def initialize(self):
        """Initialize the project builder"""
        self.logger.info("Initializing YMERA Project Builder...")
        
        # Initialize template manager
        templates_path = Path(self.config.config.get("templates_path", "./templates"))
        self.template_manager = ProjectTemplateManager(templates_path, self.ai_manager)
        await self.template_manager.initialize()
        
        # Initialize agent integration
        self.agent_integration = AgentIntegrationManager(
            self.agent_orchestrator,
            self.learning_engine,
            self.message_bus,
            self.logger
        )
        
        # Initialize metrics collector
        self.metrics_collector = BuildMetricsCollector(self.cache_manager, self.logger)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {str(e)}")
            self.docker_client = None
        
        self.logger.info("✅ YMERA Project Builder initialized successfully")
    
    async def create_project(
        self,
        configuration: ProjectConfiguration,
        template_name: Optional[str] = None,
        custom_requirements: Optional[Dict[str, Any]] = None
    ) -> BuildContext:
        """Create a new project with the specified configuration"""
        
        # Generate unique identifiers
        project_id = f"proj_{int(datetime.utcnow().timestamp())}_{hash(configuration.name) % 10000}"
        build_id = f"build_{int(datetime.utcnow().timestamp())}_{hash(project_id) % 10000}"
        
        # Create workspace directories
        workspace_path = Path(f"./workspaces/{project_id}")
        output_path = workspace_path / "output"
        temp_path = workspace_path / "temp"
        
        for path in [workspace_path, output_path, temp_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Create build context
        context = BuildContext(
            project_id=project_id,
            build_id=build_id,
            configuration=configuration,
            workspace_path=workspace_path,
            output_path=output_path,
            temp_path=temp_path,
            start_time=datetime.utcnow()
        )
        
        # Register build
        self.active_builds[build_id] = context
        
        try:
            self.logger.info(f"Starting project creation for {configuration.name} (Build ID: {build_id})")
            
            # Record build start metrics
            await self.metrics_collector.record_build_start(context)
            
            # Execute build pipeline
            await self._execute_build_pipeline(context, template_name, custom_requirements)
            
            # Mark as successful
            context.status = BuildStatus.SUCCESS
            context.end_time = datetime.utcnow()
            context.duration = context.end_time - context.start_time
            
            # Record completion metrics
            await self.metrics_collector.record_build_completion(
                context, 
                success=True, 
                total_duration=context.duration.total_seconds()
            )
            
            self.logger.info(f"Project creation completed successfully in {context.duration.total_seconds():.2f}s")
            
            return context
            
        except Exception as e:
            context.status = BuildStatus.FAILED
            context.error_message = str(e)
            context.end_time = datetime.utcnow()
            context.duration = context.end_time - context.start_time
            
            # Record failure metrics
            await self.metrics_collector.record_build_completion(
                context,
                success=False,
                total_duration=context.duration.total_seconds()
            )
            
            self.logger.error(f"Project creation failed: {str(e)}")
            raise
    
    async def _execute_build_pipeline(
        self,
        context: BuildContext,
        template_name: Optional[str],
        custom_requirements: Optional[Dict[str, Any]]
    ):
        """Execute the complete build pipeline"""
        
        pipeline_stages = [
            (BuildStage.INITIALIZATION, self._stage_initialization),
            (BuildStage.PLANNING, self._stage_planning),
            (BuildStage.SCAFFOLDING, self._stage_scaffolding),
            (BuildStage.DEVELOPMENT, self._stage_development),
            (BuildStage.TESTING, self._stage_testing),
            (BuildStage.SECURITY_SCAN, self._stage_security_scan),
            (BuildStage.OPTIMIZATION, self._stage_optimization),
            (BuildStage.DOCUMENTATION, self._stage_documentation),
            (BuildStage.PACKAGING, self._stage_packaging),
            (BuildStage.VALIDATION, self._stage_validation),
            (BuildStage.COMPLETION, self._stage_completion)
        ]
        
        for stage, stage_func in pipeline_stages:
            stage_start_time = datetime.utcnow()
            context.stage = stage
            
            try:
                self.logger.info(f"Executing stage: {stage.value}")
                await stage_func(context, template_name, custom_requirements)
                
                stage_duration = (datetime.utcnow() - stage_start_time).total_seconds()
                await self.metrics_collector.record_stage_metrics(
                    context.build_id, stage, stage_duration, BuildStatus.SUCCESS
                )
                
                self.logger.info(f"Stage {stage.value} completed in {stage_duration:.2f}s")
                
            except Exception as e:
                stage_duration = (datetime.utcnow() - stage_start_time).total_seconds()
                await self.metrics_collector.record_stage_metrics(
                    context.build_id, stage, stage_duration, BuildStatus.FAILED
                )
                
                self.logger.error(f"Stage {stage.value} failed: {str(e)}")
                raise
    
    async def _stage_initialization(
        self,
        context: BuildContext,
        template_name: Optional[str],
        custom_requirements: Optional[Dict[str, Any]]
    ):
        """Initialize the build environment"""
        
        # Create project metadata
        metadata = {
            "project_id": context.project_id,
            "build_id": context.build_id,
            "name": context.configuration.name,
            "type": context.configuration.project_type.value,
            "version": context.configuration.version,
            "created_at": datetime.utcnow().isoformat(),
            "tech_stack": context.configuration.tech_stack,
            "dependencies": context.configuration.dependencies
        }
        
        metadata_file = context.workspace_path / "project_metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        # Initialize git repository
        try:
            repo = git.Repo.init(context.workspace_path)
            repo.config_writer().set_value("user", "name", "YMERA Builder").release()
            repo.config_writer().set_value("user", "email", "builder@ymera.dev").release()
            
            # Create initial commit
            repo.index.add([str(metadata_file.name)])
            repo.index.commit("Initial project setup by YMERA")
            
            context.logs.append("Git repository initialized")
            
        except Exception as e:
            self.logger.warning(f"Git initialization failed: {str(e)}")
        
        # Create build environment
        build_env = {
            "YMERA_PROJECT_ID": context.project_id,
            "YMERA_BUILD_ID": context.build_id,
            "YMERA_WORKSPACE": str(context.workspace_path),
            "PROJECT_NAME": context.configuration.name,
            "PROJECT_TYPE": context.configuration.project_type.value
        }
        
        env_file = context.workspace_path / ".env"
        async with aiofiles.open(env_file, 'w') as f:
            for key, value in build_env.items():
                await f.write(f"{key}={value}\n")
        
        context.logs.append("Build environment initialized")
    
    async def _stage_planning(
        self,
        context: BuildContext,
        template_name: Optional[str],
        custom_requirements: Optional[Dict[str, Any]]
    ):
        """Create detailed project plan using AI and agent coordination"""
        
        # Coordinate with agent system for intelligent planning
        workflow_result = await self.agent_integration.coordinate_build_workflow(context)
        
        # Extract planning insights
        if 'project_planning' in workflow_result.get('results', {}):
            planning_result = workflow_result['results']['project_planning']
            
            if planning_result.get('status') == 'completed':
                project_plan = planning_result['result'].get('project_plan', {})
                task_breakdown = planning_result['result'].get('task_breakdown', [])
                
                # Save planning results
                plan_file = context.workspace_path /