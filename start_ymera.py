#!/usr/bin/env python3
"""
YMERA Enterprise Multi-Agent System - Replit Startup Script
Production-ready startup with comprehensive dependency checking and error recovery
"""

import os
import sys
import asyncio
import logging
import subprocess
import importlib
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YmeraDependencyChecker:
    """Comprehensive dependency checker for Replit environment"""
    
    def __init__(self):
        self.required_packages = {
            # Core FastAPI and async
            'fastapi': '0.104.0',
            'uvicorn': '0.24.0',
            'uvloop': '0.19.0',
            'websockets': '12.0',
            'aiofiles': '23.2.0',
            'aiohttp': '3.9.0',
            
            # Database
            'sqlalchemy': '2.0.23',
            'asyncpg': '0.29.0',
            'psycopg2-binary': '2.9.9',
            'alembic': '1.13.0',
            
            # Redis
            'redis': '5.0.1',
            
            # AI Services
            'openai': '1.3.0',
            'anthropic': '0.7.0',
            'groq': '0.4.0',
            'google-generativeai': '0.3.0',
            'pinecone-client': '2.2.4',
            
            # ML/AI Tools
            'numpy': '1.24.3',
            'sentence-transformers': '2.2.2',
            'scikit-learn': '1.3.0',
            'tiktoken': '0.5.0',
            
            # Security
            'cryptography': '41.0.7',
            'python-jose': '3.3.0',
            'passlib': '1.7.4',
            'bcrypt': '4.1.2',
            
            # Development Tools
            'pylint': '3.0.0',
            'bandit': '1.7.5',
            'semgrep': '1.45.0',
            'gitpython': '3.1.40',
            'pygithub': '2.1.1',
            
            # Utilities
            'pydantic': '2.5.0',
            'jinja2': '3.1.2',
            'pyyaml': '6.0.1',
            'psutil': '5.9.6',
            'prometheus-client': '0.19.0',
            'structlog': '23.2.0',
            'networkx': '3.2.1'
        }
        
        self.optional_packages = {
            'torch': '2.1.0',
            'transformers': '4.36.0',
            'tensorflow': '2.15.0'
        }
        
        self.system_dependencies = [
            'git',
            'postgresql-client'  # For Replit database connections
        ]
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major == 3 and version.minor >= 9:
            logger.info("✅ Python version is compatible")
            return True
        else:
            logger.error("❌ Python 3.9+ required")
            return False
    
    def check_replit_environment(self) -> Dict[str, bool]:
        """Check Replit-specific environment setup"""
        checks = {}
        
        # Check if running in Replit
        checks['is_replit'] = 'REPL_ID' in os.environ
        if checks['is_replit']:
            logger.info(f"✅ Running in Replit environment: {os.getenv('REPL_ID')}")
        
        # Check essential environment variables
        env_vars = [
            'REPL_SLUG', 'REPL_OWNER', 'DATABASE_URL', 
            'OPENAI_API_KEY', 'SECRET_KEY'
        ]
        
        for var in env_vars:
            checks[f'env_{var}'] = var in os.environ and os.getenv(var) != ''
            if checks[f'env_{var}']:
                logger.info(f"✅ Environment variable {var} is set")
            else:
                logger.warning(f"⚠️  Environment variable {var} is missing")
        
        return checks
    
    def install_package(self, package: str, version: str = None) -> bool:
        """Install a package using pip with Replit optimizations"""
        try:
            version_spec = f"=={version}" if version else ""
            cmd = [sys.executable, "-m", "pip", "install", f"{package}{version_spec}", "--upgrade"]
            
            logger.info(f"Installing {package}{version_spec}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {package}")
                return True
            else:
                logger.error(f"❌ Failed to install {package}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout installing {package}")
            return False
        except Exception as e:
            logger.error(f"❌ Error installing {package}: {e}")
            return False
    
    def check_package_import(self, package: str, import_name: str = None) -> bool:
        """Check if a package can be imported"""
        import_name = import_name or package
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
    
    def check_and_install_dependencies(self) -> bool:
        """Check and install all required dependencies"""
        logger.info("🔍 Checking dependencies...")
        
        missing_packages = []
        
        # Check required packages
        for package, version in self.required_packages.items():
            # Map package names to import names where different
            import_mapping = {
                'python-jose': 'jose',
                'pyyaml': 'yaml',
                'pillow': 'PIL',
                'psycopg2-binary': 'psycopg2',
                'scikit-learn': 'sklearn'
            }
            
            import_name = import_mapping.get(package, package.replace('-', '_'))
            
            if not self.check_package_import(package, import_name):
                missing_packages.append((package, version))
        
        if missing_packages:
            logger.info(f"📦 Installing {len(missing_packages)} missing packages...")
            
            for package, version in missing_packages:
                success = self.install_package(package, version)
                if not success:
                    logger.error(f"❌ Failed to install critical package: {package}")
                    return False
        
        # Install optional packages (don't fail if these don't install)
        for package, version in self.optional_packages.items():
            if not self.check_package_import(package):
                logger.info(f"📦 Installing optional package: {package}")
                self.install_package(package, version)
        
        logger.info("✅ All critical dependencies are available")
        return True
    
    def setup_replit_database(self) -> bool:
        """Setup database for Replit environment"""
        try:
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                # Create SQLite fallback for development
                database_url = "sqlite+aiosqlite:///./ymera.db"
                os.environ['DATABASE_URL'] = database_url
                logger.info("🗄️  Using SQLite fallback database")
            else:
                logger.info("🗄️  Using configured database")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def setup_redis(self) -> bool:
        """Setup Redis for Replit environment"""
        try:
            redis_url = os.getenv('REDIS_URL')
            
            if not redis_url:
                # Use in-memory fallback for development
                redis_url = "redis://localhost:6379/0"
                os.environ['REDIS_URL'] = redis_url
                logger.warning("⚠️  Redis not configured, using localhost fallback")
            else:
                logger.info("🔴 Using configured Redis")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis setup failed: {e}")
            return False

class YmeraStartup:
    """Main startup orchestrator for YMERA system"""
    
    def __init__(self):
        self.dependency_checker = YmeraDependencyChecker()
        self.startup_success = False
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def create_required_directories(self):
        """Create necessary directories for the application"""
        directories = [
            'logs',
            'data',
            'temp',
            'uploads',
            'models',
            'agents'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
    
    def setup_environment_defaults(self):
        """Setup default environment variables for Replit"""
        defaults = {
            'ENVIRONMENT': 'production',
            'DEBUG': 'false',
            'HOST': '0.0.0.0',
            'PORT': '8000',
            'WORKERS': '1',  # Replit works better with single worker
            'SECRET_KEY': 'replit-development-key-change-in-production',
            'AI_MAX_TOKENS': '4096',
            'AI_TEMPERATURE': '0.1',
            'CACHE_TTL_SECONDS': '3600',
            'MAX_CONCURRENT_AGENTS': '5',  # Lower for Replit resources
            'AGENT_TIMEOUT_SECONDS': '120'
        }
        
        for key, value in defaults.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.info(f"🔧 Set default {key}={value}")
    
    async def run_startup_checks(self) -> bool:
        """Run all startup checks and preparations"""
        logger.info("🚀 Starting YMERA Enterprise Multi-Agent System...")
        logger.info(f"📅 Startup time: {datetime.now().isoformat()}")
        
        # Basic system checks
        if not self.dependency_checker.check_python_version():
            return False
        
        # Check Replit environment
        env_checks = self.dependency_checker.check_replit_environment()
        
        # Setup environment defaults
        self.setup_environment_defaults()
        
        # Create directories
        self.create_required_directories()
        
        # Check and install dependencies
        if not self.dependency_checker.check_and_install_dependencies():
            return False
        
        # Setup database
        if not self.dependency_checker.setup_database():
            return False
        
        # Setup Redis
        if not self.dependency_checker.setup_redis():
            return False
        
        logger.info("✅ All startup checks completed successfully")
        return True
    
    def start_application(self):
        """Start the main YMERA application"""
        try:
            # Import the main application after dependencies are ready
            logger.info("📥 Importing main application...")
            
            # Add current directory to path for imports
            sys.path.insert(0, str(Path.cwd()))
            
            # Import and start the main application
            from main import app  # Your main FastAPI app
            
            import uvicorn
            
            # Configure uvicorn for Replit
            config = uvicorn.Config(
                app,
                host=os.getenv('HOST', '0.0.0.0'),
                port=int(os.getenv('PORT', 8000)),
                reload=os.getenv('DEBUG', 'false').lower() == 'true',
                workers=1,  # Single worker for Replit
                loop='uvloop',
                access_log=True,
                log_level='info'
            )
            
            server = uvicorn.Server(config)
            
            logger.info("🌟 Starting YMERA server...")
            asyncio.run(server.serve())
            
        except ImportError as e:
            logger.error(f"❌ Failed to import main application: {e}")
            logger.error("Make sure 'main.py' exists with your FastAPI app")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Failed to start application: {e}")
            sys.exit(1)

def main():
    """Main entry point"""
    startup = YmeraStartup()
    startup.setup_signal_handlers()
    
    try:
        # Run async startup checks
        success = asyncio.run(startup.run_startup_checks())
        
        if success:
            startup.startup_success = True
            logger.info("🎉 YMERA system startup completed successfully!")
            startup.start_application()
        else:
            logger.error("❌ Startup checks failed, cannot start application")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("👋 Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()