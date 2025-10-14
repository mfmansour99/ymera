"""
Production-Ready LLM Agent v2.1
Advanced LLM capabilities with RAG, multi-provider support, and enhanced reliability

Features:
- Multi-provider LLM support (OpenAI, Anthropic, with easy extensibility)
- RAG (Retrieval Augmented Generation) with Qdrant
- Conversation memory management with automatic cleanup
- Intelligent model routing based on complexity
- Production-grade error handling and recovery
- Comprehensive metrics and monitoring
- Token usage tracking and cost optimization
- Streaming response support
- Easy to add new LLM providers

Improvements in v2.1:
- Fixed imports and dependencies
- Better provider abstraction for easy extension
- Enhanced error handling
- Improved conversation management
- Better token tracking
- More robust RAG implementation
"""

import asyncio
import json
import time
import uuid
import os
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod

# Import base agent components
from enhanced_base_agent import (
    BaseAgent, AgentConfig, TaskRequest, Priority,
    AgentState, ConnectionState, run_agent
)

# Third-party imports with graceful degradation
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    EMBEDDING_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"  # For future custom providers


class MessageRole(Enum):
    """Message roles in conversations"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationMessage:
    """Individual message in a conversation"""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "token_count": self.token_count
        }


@dataclass
class ConversationMemory:
    """Conversation memory with automatic summarization"""
    conversation_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    summary: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None, token_count: int = 0):
        """Add a message to conversation"""
        msg = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
            token_count=token_count
        )
        self.messages.append(msg)
        self.total_tokens += token_count
        self.updated_at = time.time()
    
    def get_context_messages(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent messages formatted for LLM context"""
        recent = self.messages[-max_messages:]
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in recent
        ]
    
    def get_token_count(self) -> int:
        """Get total token count"""
        return sum(msg.token_count for msg in self.messages)


@dataclass
class RAGDocument:
    """Document for RAG indexing"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class LLMConfig:
    """Configuration for LLM model"""
    provider: LLMProvider
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: Optional[str] = None
    use_streaming: bool = False
    timeout: int = 60
    cost_per_1k_prompt: float = 0.0
    cost_per_1k_completion: float = 0.0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers - makes it easy to add new providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
    
    @abstractmethod
    async def initialize(self):
        """Initialize the provider client"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        """
        Generate completion
        Returns: {
            "content": str,
            "tokens_used": {"prompt": int, "completion": int, "total": int},
            "model": str
        }
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Cleanup provider resources"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""
    
    async def initialize(self):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI library not installed")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty
            ),
            timeout=config.timeout
        )
        
        return {
            "content": response.choices[0].message.content,
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            },
            "model": config.model
        }
    
    async def close(self):
        if self.client:
            await self.client.close()


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM Provider"""
    
    async def initialize(self):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic library not installed")
        
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        config: LLMConfig
    ) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        # Anthropic expects system message separately
        system_message = next(
            (m["content"] for m in messages if m["role"] == "system"),
            None
        )
        user_messages = [m for m in messages if m["role"] != "system"]
        
        response = await asyncio.wait_for(
            self.client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=user_messages,
                system=system_message
            ),
            timeout=config.timeout
        )
        
        return {
            "content": response.content[0].text,
            "tokens_used": {
                "prompt": response.usage.input_tokens,
                "completion": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens
            },
            "model": config.model
        }
    
    async def close(self):
        if self.client:
            await self.client.close()


class LLMProviderManager:
    """Manages multiple LLM providers"""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
    
    async def register_provider(self, provider_type: LLMProvider, provider: BaseLLMProvider):
        """Register a new provider"""
        await provider.initialize()
        self.providers[provider_type] = provider
    
    def get_provider(self, provider_type: LLMProvider) -> Optional[BaseLLMProvider]:
        """Get a provider by type"""
        return self.providers.get(provider_type)
    
    async def close_all(self):
        """Close all providers"""
        for provider in self.providers.values():
            await provider.close()


class LLMAgent(BaseAgent):
    """
    Production-ready LLM Agent with comprehensive features
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # LLM provider manager
        self.provider_manager = LLMProviderManager()
        
        # Vector database components
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # Model configurations with cost tracking
        self.llm_configs: Dict[str, LLMConfig] = {
            "gpt4-turbo": LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                max_tokens=4096,
                temperature=0.7,
                system_prompt="You are a helpful AI assistant in a production multi-agent system.",
                cost_per_1k_prompt=0.01,
                cost_per_1k_completion=0.03,
                timeout=60
            ),
            "gpt4": LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4",
                max_tokens=4096,
                temperature=0.7,
                system_prompt="You are a helpful AI assistant in a production multi-agent system.",
                cost_per_1k_prompt=0.03,
                cost_per_1k_completion=0.06,
                timeout=60
            ),
            "gpt35": LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                max_tokens=4096,
                temperature=0.7,
                system_prompt="You are a helpful AI assistant in a production multi-agent system.",
                cost_per_1k_prompt=0.0005,
                cost_per_1k_completion=0.0015,
                timeout=30
            ),
            "claude-opus": LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-opus-20240229",
                max_tokens=4096,
                temperature=0.7,
                system_prompt="You are a helpful AI assistant in a production multi-agent system.",
                cost_per_1k_prompt=0.015,
                cost_per_1k_completion=0.075,
                timeout=60
            ),
            "claude-sonnet": LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                temperature=0.7,
                system_prompt="You are a helpful AI assistant in a production multi-agent system.",
                cost_per_1k_prompt=0.003,
                cost_per_1k_completion=0.015,
                timeout=45
            ),
            "claude-haiku": LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                temperature=0.7,
                system_prompt="You are a helpful AI assistant in a production multi-agent system.",
                cost_per_1k_prompt=0.00025,
                cost_per_1k_completion=0.00125,
                timeout=30
            )
        }
        
        # Conversation management
        self.conversations: Dict[str, ConversationMemory] = {}
        self.max_conversation_age = int(os.getenv("MAX_CONVERSATION_AGE", "86400"))  # 24 hours
        self.max_conversation_messages = int(os.getenv("MAX_CONVERSATION_MESSAGES", "50"))
        self.max_conversation_tokens = int(os.getenv("MAX_CONVERSATION_TOKENS", "100000"))
        
        # RAG collections
        self.rag_collections = {
            "knowledge_base": "general_knowledge",
            "code_snippets": "code_examples",
            "documentation": "system_docs",
            "faq": "frequently_asked_questions"
        }
        
        # Performance tracking
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0
        }
        
        self.model_usage_stats: Dict[str, Dict[str, Any]] = {
            model: {
                "requests": 0,
                "tokens": 0,
                "errors": 0,
                "avg_latency_ms": 0,
                "total_cost": 0.0
            }
            for model in self.llm_configs
        }
        
        # Tokenizer for accurate token counting
        self.tokenizer: Optional[tiktoken.Encoding] = None
        
        # Model routing rules - can be customized
        self.model_router_rules = {
            "simple_qa": "gpt35",
            "complex_reasoning": "gpt4-turbo",
            "creative_writing": "claude-sonnet",
            "code_generation": "gpt4-turbo",
            "data_analysis": "gpt4-turbo",
            "chat_conversation": "gpt35",
            "technical_documentation": "claude-sonnet",
            "summarization": "gpt35",
            "fast_response": "claude-haiku"
        }
        
        # Circuit breaker tracking per model
        self.llm_circuit_breakers: Dict[str, Dict[str, Any]] = {}
        for model_key in self.llm_configs:
            self.llm_circuit_breakers[model_key] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
    
    async def _initialize_database(self):
        """Initialize database schema for LLM agent"""
        if not self.db_pool:
            return
        
        try:
            # Token usage tracking table
            await self._db_execute("""
                CREATE TABLE IF NOT EXISTS llm_token_usage (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255) NOT NULL,
                    model_key VARCHAR(100),
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_cost DECIMAL(10, 6) DEFAULT 0.0,
                    recorded_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await self._db_execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_usage_agent_time 
                ON llm_token_usage(agent_id, recorded_at)
            """)
            
            # Model statistics table
            await self._db_execute("""
                CREATE TABLE IF NOT EXISTS llm_model_stats (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    requests INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    avg_latency_ms FLOAT DEFAULT 0,
                    total_cost DECIMAL(10, 6) DEFAULT 0.0,
                    recorded_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            await self._db_execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_model_stats 
                ON llm_model_stats(agent_id, model_name, recorded_at)
            """)
            
            # Conversation tracking table
            await self._db_execute("""
                CREATE TABLE IF NOT EXISTS llm_conversations (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(255) UNIQUE NOT NULL,
                    agent_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    summary TEXT,
                    metadata JSONB
                )
            """)
            
            self.logger.info("LLM database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database schema: {e}")
    
    async def _setup_subscriptions(self):
        """Setup LLM agent specific subscriptions"""
        await super()._setup_subscriptions()
        
        if not self.nc:
            return
        
        # Document indexing for RAG
        await self._subscribe(
            "llm.index.document",
            self._handle_document_indexing,
            queue_group="llm-indexing"
        )
        
        # Conversation management
        await self._subscribe(
            "llm.conversation.get",
            self._handle_get_conversation,
            queue_group="llm-conversation"
        )
        
        await self._subscribe(
            "llm.conversation.clear",
            self._handle_clear_conversation,
            queue_group="llm-conversation"
        )
        
        await self._subscribe(
            "llm.conversation.summarize",
            self._handle_summarize_conversation,
            queue_group="llm-conversation"
        )
        
        # Live chat support
        await self._subscribe(
            "llm.chat.generate",
            self._handle_chat_generation,
            queue_group="llm-chat"
        )
        
        # Batch processing
        await self._subscribe(
            "llm.batch.process",
            self._handle_batch_processing,
            queue_group="llm-batch"
        )
    
    async def _start_background_tasks(self):
        """Start LLM agent background tasks"""
        await super()._start_background_tasks()
        
        # Initialize LLM providers
        await self._initialize_llm_providers()
        
        # Setup vector database
        await self._setup_vector_database()
        
        # Load initial knowledge base
        await self._load_initial_knowledge_base()
        
        # Conversation cleanup task
        task = asyncio.create_task(
            self._run_background_task(self._conversation_cleanup_loop, 3600)
        )
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        # Token usage persistence task
        task = asyncio.create_task(
            self._run_background_task(self._persist_token_usage_loop, 300)
        )
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        # Model stats persistence
        task = asyncio.create_task(
            self._run_background_task(self._persist_model_stats_loop, 600)
        )
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        self.logger.info("LLM Agent background tasks started")
    
    async def _initialize_llm_providers(self):
        """Initialize LLM provider clients"""
        try:
            # OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and OPENAI_AVAILABLE:
                provider = OpenAIProvider(openai_key)
                await self.provider_manager.register_provider(LLMProvider.OPENAI, provider)
                self.logger.info("OpenAI provider initialized")
            else:
                self.logger.warning("OpenAI not available (missing key or library)")
            
            # Anthropic
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key and ANTHROPIC_AVAILABLE:
                provider = AnthropicProvider(anthropic_key)
                await self.provider_manager.register_provider(LLMProvider.ANTHROPIC, provider)
                self.logger.info("Anthropic provider initialized")
            else:
                self.logger.warning("Anthropic not available (missing key or library)")
            
            # Initialize tokenizer
            if TIKTOKEN_AVAILABLE:
                try:
                    self.tokenizer = tiktoken.encoding_for_model("gpt-4")
                    self.logger.info("Tokenizer initialized")
                except Exception as e:
                    self.logger.warning(f"Tokenizer initialization failed: {e}")
        
        except Exception as e:
            self.logger.error(f"LLM provider initialization failed: {e}")
            raise
    
    async def _setup_vector_database(self):
        """Initialize Qdrant and embedding model"""
        if not QDRANT_AVAILABLE or not EMBEDDING_AVAILABLE:
            self.logger.warning("Vector database features disabled (missing dependencies)")
            return
        
        try:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if qdrant_api_key:
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.qdrant_client = QdrantClient(url=qdrant_url)
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            self.logger.info(f"Qdrant connected, {len(collections.collections)} collections found")
            
            # Initialize embedding model
            embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.logger.info(f"Embedding model loaded: {embedding_model_name}")
            
            # Ensure collections exist
            for alias, collection_name in self.rag_collections.items():
                try:
                    self.qdrant_client.get_collection(collection_name)
                    self.logger.info(f"Collection '{collection_name}' exists")
                except:
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                    self.logger.info(f"Collection '{collection_name}' created")
        
        except Exception as e:
            self.logger.error(f"Vector database setup failed: {e}")
            self.qdrant_client = None
            self.embedding_model = None
    
    async def _load_initial_knowledge_base(self):
        """Load initial documents into RAG system"""
        if not self.qdrant_client or not self.embedding_model:
            return
        
        initial_docs = [
            RAGDocument(
                id="doc_platform_overview",
                content="This is a production-ready multi-agent system with enterprise-grade reliability, scalability, and monitoring capabilities.",
                metadata={"source": "internal", "category": "platform", "version": "2.1"}
            ),
            RAGDocument(
                id="doc_llm_capabilities",
                content="The LLM agent supports multiple providers (OpenAI, Anthropic), RAG with Qdrant, conversation management, cost tracking, and intelligent model routing.",
                metadata={"source": "agent_docs", "category": "llm", "version": "2.1"}
            )
        ]
        
        indexed = 0
        for doc in initial_docs:
            try:
                await self._index_document(doc, "general_knowledge")
                indexed += 1
            except Exception as e:
                self.logger.error(f"Failed to index document {doc.id}: {e}")
        
        self.logger.info(f"Loaded {indexed}/{len(initial_docs)} initial documents")
    
    async def _handle_task(self, task_request: TaskRequest) -> Dict[str, Any]:
        """Handle LLM-specific tasks"""
        task_type = task_request.task_type
        payload = task_request.payload
        
        try:
            if task_type == "generate_text":
                return await self._generate_text(payload)
            
            elif task_type == "chat_completion":
                return await self._chat_completion(payload)
            
            elif task_type == "create_embedding":
                return await self._create_embedding(payload)
            
            elif task_type == "semantic_search":
                return await self._semantic_search(payload)
            
            elif task_type == "summarize_text":
                return await self._summarize_text(payload)
            
            elif task_type == "analyze_sentiment":
                return await self._analyze_sentiment(payload)
            
            elif task_type == "extract_entities":
                return await self._extract_entities(payload)
            
            else:
                return await super()._handle_task(task_request)
        
        except Exception as e:
            self.logger.error(
                f"Task execution failed: {task_type}",
                extra={"error": str(e), "task_id": task_request.task_id}
            )
            return {
                "status": "error",
                "error": str(e),
                "task_id": task_request.task_id
            }
    
    async def _generate_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using LLM"""
        prompt = payload["prompt"]
        model_key = payload.get("model_key", "gpt35")
        max_tokens = payload.get("max_tokens", 500)
        temperature = payload.get("temperature", 0.7)
        system_prompt = payload.get("system_prompt")
        
        llm_config = self.llm_configs.get(model_key)
        if not llm_config:
            raise ValueError(f"Unknown model: {model_key}")
        
        messages = []
        if system_prompt or llm_config.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or llm_config.system_prompt
            })
        messages.append({"role": "user", "content": prompt})
        
        # Create a copy of config with custom params
        custom_config = LLMConfig(
            provider=llm_config.provider,
            model=llm_config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=llm_config.system_prompt,
            timeout=llm_config.timeout,
            cost_per_1k_prompt=llm_config.cost_per_1k_prompt,
            cost_per_1k_completion=llm_config.cost_per_1k_completion
        )
        
        response = await self._call_llm_with_retry(model_key, custom_config, messages)
        
        return {
            "status": "success",
            "content": response["content"],
            "model": model_key,
            "tokens_used": response["tokens_used"],
            "cost": response["cost"]
        }
    
    async def _chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Chat completion with conversation memory and optional RAG"""
        conversation_id = payload.get("conversation_id", str(uuid.uuid4()))
        user_message = payload["message"]
        model_key = payload.get("model_key", "gpt35")
        max_tokens = payload.get("max_tokens", 500)
        temperature = payload.get("temperature", 0.7)
        use_rag = payload.get("use_rag", False)
        rag_collections = payload.get("rag_collections", ["knowledge_base"])
        
        llm_config = self.llm_configs.get(model_key)
        if not llm_config:
            raise ValueError(f"Unknown model: {model_key}")
        
        # Get or create conversation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationMemory(
                conversation_id=conversation_id
            )
        
        conversation = self.conversations[conversation_id]
        
        # Count tokens for user message
        user_token_count = self._count_tokens(user_message)
        conversation.add_message(MessageRole.USER, user_message, token_count=user_token_count)
        
        # Build messages for LLM
        messages = []
        
        # Add system prompt
        if llm_config.system_prompt:
            messages.append({
                "role": "system",
                "content": llm_config.system_prompt
            })
        
        # Add RAG context if enabled
        if use_rag and self.qdrant_client and self.embedding_model:
            rag_docs = await self._perform_rag_search(user_message, rag_collections)
            if rag_docs:
                rag_context = "\n\nRelevant Information:\n" + "\n---\n".join(
                    [f"{doc.content}" for doc in rag_docs[:3]]
                )
                messages.append({
                    "role": "system",
                    "content": f"Context from knowledge base:{rag_context}"
                })
        
        # Add conversation history
        messages.extend(conversation.get_context_messages())
        
        # Create custom config
        custom_config = LLMConfig(
            provider=llm_config.provider,
            model=llm_config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=llm_config.system_prompt,
            timeout=llm_config.timeout,
            cost_per_1k_prompt=llm_config.cost_per_1k_prompt,
            cost_per_1k_completion=llm_config.cost_per_1k_completion
        )
        
        # Call LLM
        response = await self._call_llm_with_retry(model_key, custom_config, messages)
        
        # Store assistant response
        assistant_token_count = response["tokens_used"]["completion"]
        conversation.add_message(
            MessageRole.ASSISTANT,
            response["content"],
            token_count=assistant_token_count
        )
        
        # Auto-summarize if conversation is getting long
        if (len(conversation.messages) > self.max_conversation_messages or
            conversation.get_token_count() > self.max_conversation_tokens):
            await self._summarize_conversation(conversation)
        
        return {
            "status": "success",
            "content": response["content"],
            "conversation_id": conversation_id,
            "model": model_key,
            "tokens_used": response["tokens_used"],
            "cost": response["cost"],
            "conversation_tokens": conversation.get_token_count()
        }
    
    async def _create_embedding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create embedding for text"""
        text = payload["text"]
        
        if not self.embedding_model:
            raise RuntimeError("Embedding model not available")
        
        embedding = self.embedding_model.encode(text).tolist()
        
        return {
            "status": "success",
            "embedding": embedding,
            "dimensions": len(embedding)
        }
    
    async def _semantic_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic search in vector database"""
        query = payload["query"]
        collection = payload.get("collection", "general_knowledge")
        limit = payload.get("limit", 5)
        
        if not self.qdrant_client or not self.embedding_model:
            raise RuntimeError("Vector database not available")
        
        docs = await self._perform_rag_search(query, [collection], limit)
        
        return {
            "status": "success",
            "results": [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "id": doc.id
                }
                for doc in docs
            ],
            "count": len(docs)
        }
    
    async def _summarize_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize given text"""
        text = payload["text"]
        model_key = payload.get("model_key", "gpt35")
        max_tokens = payload.get("max_tokens", 200)
        
        llm_config = self.llm_configs.get(model_key)
        if not llm_config:
            raise ValueError(f"Unknown model: {model_key}")
        
        messages = [
            {
                "role": "user",
                "content": f"Summarize the following text concisely:\n\n{text}"
            }
        ]
        
        custom_config = LLMConfig(
            provider=llm_config.provider,
            model=llm_config.model,
            max_tokens=max_tokens,
            temperature=0.3,
            timeout=llm_config.timeout,
            cost_per_1k_prompt=llm_config.cost_per_1k_prompt,
            cost_per_1k_completion=llm_config.cost_per_1k_completion
        )
        
        response = await self._call_llm_with_retry(model_key, custom_config, messages)
        
        return {
            "status": "success",
            "summary": response["content"],
            "model": model_key,
            "tokens_used": response["tokens_used"]
        }
    
    async def _analyze_sentiment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        text = payload["text"]
        model_key = payload.get("model_key", "gpt35")
        
        llm_config = self.llm_configs.get(model_key)
        if not llm_config:
            raise ValueError(f"Unknown model: {model_key}")
        
        messages = [
            {
                "role": "user",
                "content": f"Analyze the sentiment of this text. Return only one word: positive, negative, or neutral.\n\nText: {text}"
            }
        ]
        
        custom_config = LLMConfig(
            provider=llm_config.provider,
            model=llm_config.model,
            max_tokens=10,
            temperature=0.1,
            timeout=llm_config.timeout,
            cost_per_1k_prompt=llm_config.cost_per_1k_prompt,
            cost_per_1k_completion=llm_config.cost_per_1k_completion
        )
        
        response = await self._call_llm_with_retry(model_key, custom_config, messages)
        sentiment = response["content"].strip().lower()
        
        return {
            "status": "success",
            "sentiment": sentiment,
            "model": model_key,
            "tokens_used": response["tokens_used"]
        }
    
    async def _extract_entities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract named entities from text"""
        text = payload["text"]
        model_key = payload.get("model_key", "gpt35")
        
        llm_config = self.llm_configs.get(model_key)
        if not llm_config:
            raise ValueError(f"Unknown model: {model_key}")
        
        messages = [
            {
                "role": "user",
                "content": f"Extract key entities (people, organizations, locations, dates) from this text. Return as JSON list.\n\nText: {text}"
            }
        ]
        
        custom_config = LLMConfig(
            provider=llm_config.provider,
            model=llm_config.model,
            max_tokens=500,
            temperature=0.1,
            timeout=llm_config.timeout,
            cost_per_1k_prompt=llm_config.cost_per_1k_prompt,
            cost_per_1k_completion=llm_config.cost_per_1k_completion
        )
        
        response = await self._call_llm_with_retry(model_key, custom_config, messages)
        
        return {
            "status": "success",
            "entities": response["content"],
            "model": model_key,
            "tokens_used": response["tokens_used"]
        }
    
    async def _call_llm_with_retry(
        self,
        model_key: str,
        config: LLMConfig,
        messages: List[Dict[str, str]],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Call LLM API with circuit breaker and retry logic"""
        
        # Check circuit breaker
        breaker = self.llm_circuit_breakers[model_key]
        if breaker["state"] == "open":
            if breaker["last_failure"]:
                time_since_failure = time.time() - breaker["last_failure"]
                if time_since_failure < 60:  # 60 second timeout
                    raise RuntimeError(
                        f"Circuit breaker open for {model_key}, "
                        f"retry after {60 - time_since_failure:.0f}s"
                    )
                else:
                    breaker["state"] = "half-open"
                    breaker["failures"] = 0
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                # Get provider
                provider = self.provider_manager.get_provider(config.provider)
                if not provider:
                    raise RuntimeError(f"Provider {config.provider.value} not available")
                
                # Call provider
                response = await provider.generate(messages, config)
                
                latency = (time.time() - start_time) * 1000
                
                # Calculate cost
                prompt_tokens = response["tokens_used"]["prompt"]
                completion_tokens = response["tokens_used"]["completion"]
                cost = self._calculate_cost(config, prompt_tokens, completion_tokens)
                response["cost"] = cost
                
                # Update metrics
                self._update_model_metrics(model_key, response, latency, cost)
                
                # Reset circuit breaker on success
                breaker["failures"] = 0
                breaker["state"] = "closed"
                
                return response
            
            except Exception as e:
                self.logger.error(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries})",
                    extra={"model": model_key, "error": str(e)}
                )
                
                # Update circuit breaker
                breaker["failures"] += 1
                breaker["last_failure"] = time.time()
                
                if breaker["failures"] >= 5:
                    breaker["state"] = "open"
                    self.logger.warning(
                        f"Circuit breaker opened for {model_key}",
                        extra={"failures": breaker["failures"]}
                    )
                
                self.model_usage_stats[model_key]["errors"] += 1
                
                if attempt == max_retries - 1:
                    raise
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _calculate_cost(
        self,
        config: LLMConfig,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate API call cost"""
        prompt_cost = (prompt_tokens / 1000) * config.cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000) * config.cost_per_1k_completion
        return prompt_cost + completion_cost
    
    def _update_model_metrics(
        self,
        model_key: str,
        response: Dict[str, Any],
        latency_ms: float,
        cost: float
    ):
        """Update model usage metrics"""
        stats = self.model_usage_stats[model_key]
        stats["requests"] += 1
        
        tokens = response["tokens_used"]["total"]
        stats["tokens"] += tokens
        stats["total_cost"] += cost
        
        # Update average latency
        old_avg = stats["avg_latency_ms"]
        n = stats["requests"]
        stats["avg_latency_ms"] = (old_avg * (n - 1) + latency_ms) / n
        
        # Update global token usage
        self.token_usage["prompt_tokens"] += response["tokens_used"]["prompt"]
        self.token_usage["completion_tokens"] += response["tokens_used"]["completion"]
        self.token_usage["total_cost"] += cost
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Rough estimate if tokenizer not available
        return len(text.split()) * 1.3
    
    async def _index_document(self, doc: RAGDocument, collection_name: str):
        """Index document in vector database"""
        if not self.qdrant_client or not self.embedding_model:
            raise RuntimeError("Vector database not available")
        
        embedding = self.embedding_model.encode(doc.content).tolist()
        
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=doc.id,
                    vector=embedding,
                    payload={
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "indexed_at": doc.created_at
                    }
                )
            ]
        )
    
    async def _perform_rag_search(
        self,
        query: str,
        collections: List[str],
        limit: int = 3
    ) -> List[RAGDocument]:
        """Perform RAG search across collections"""
        if not self.qdrant_client or not self.embedding_model:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            all_results = []
            for collection_alias in collections:
                collection_name = self.rag_collections.get(
                    collection_alias, collection_alias
                )
                
                try:
                    results = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=limit,
                        with_payload=True
                    )
                    all_results.extend(results)
                except Exception as e:
                    self.logger.warning(
                        f"Search failed in collection {collection_name}: {e}"
                    )
            
            # Sort by score and deduplicate
            all_results.sort(key=lambda x: x.score, reverse=True)
            
            seen_ids = set()
            unique_docs = []
            
            for hit in all_results:
                doc_id = hit.payload["id"]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(
                        RAGDocument(
                            id=doc_id,
                            content=hit.payload["content"],
                            metadata=hit.payload["metadata"],
                            created_at=hit.payload.get("indexed_at", time.time())
                        )
                    )
                    
                    if len(unique_docs) >= limit:
                        break
            
            return unique_docs
        
        except Exception as e:
            self.logger.error(f"RAG search failed: {e}")
            return []
    
    async def _summarize_conversation(self, conversation: ConversationMemory):
        """Summarize conversation to reduce memory"""
        if len(conversation.messages) < 10:
            return
        
        try:
            # Keep last 5 messages, summarize the rest
            messages_to_summarize = conversation.messages[:-5]
            
            conversation_text = "\n".join([
                f"{msg.role.value}: {msg.content}"
                for msg in messages_to_summarize
            ])
            
            llm_config = self.llm_configs["gpt35"]
            messages = [
                {
                    "role": "user",
                    "content": f"Summarize this conversation concisely:\n\n{conversation_text}"
                }
            ]
            
            custom_config = LLMConfig(
                provider=llm_config.provider,
                model=llm_config.model,
                max_tokens=200,
                temperature=0.3,
                timeout=llm_config.timeout,
                cost_per_1k_prompt=llm_config.cost_per_1k_prompt,
                cost_per_1k_completion=llm_config.cost_per_1k_completion
            )
            
            response = await self._call_llm_with_retry("gpt35", custom_config, messages)
            
            conversation.summary = response["content"]
            conversation.messages = conversation.messages[-5:]
            conversation.updated_at = time.time()
            
            self.logger.info(
                "Conversation summarized",
                extra={
                    "conversation_id": conversation.conversation_id,
                    "summarized_messages": len(messages_to_summarize)
                }
            )
        
        except Exception as e:
            self.logger.error(
                f"Conversation summarization failed: {e}",
                extra={"conversation_id": conversation.conversation_id}
            )
    
    # Message handlers
    
    async def _handle_document_indexing(self, msg):
        """Handle document indexing requests"""
        try:
            data = json.loads(msg.data.decode())
            
            doc = RAGDocument(
                id=data.get("id", str(uuid.uuid4())),
                content=data["content"],
                metadata=data.get("metadata", {})
            )
            
            collection = data.get("collection", "general_knowledge")
            
            await self._index_document(doc, collection)
            
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "success",
                    "document_id": doc.id,
                    "collection": collection
                })
            
            self.logger.info(
                "Document indexed",
                extra={"doc_id": doc.id, "collection": collection}
            )
        
        except Exception as e:
            self.logger.error(f"Document indexing failed: {e}")
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "error",
                    "error": str(e)
                })
    
    async def _handle_get_conversation(self, msg):
        """Handle conversation retrieval requests"""
        try:
            data = json.loads(msg.data.decode())
            conversation_id = data["conversation_id"]
            
            conversation = self.conversations.get(conversation_id)
            
            if conversation:
                response = {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "messages": [m.to_dict() for m in conversation.messages],
                    "summary": conversation.summary,
                    "created_at": conversation.created_at,
                    "updated_at": conversation.updated_at,
                    "message_count": len(conversation.messages),
                    "total_tokens": conversation.get_token_count()
                }
            else:
                response = {
                    "status": "not_found",
                    "conversation_id": conversation_id
                }
            
            if msg.reply:
                await self._publish_response(msg.reply, response)
        
        except Exception as e:
            self.logger.error(f"Get conversation failed: {e}")
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "error",
                    "error": str(e)
                })
    
    async def _handle_clear_conversation(self, msg):
        """Handle conversation clearing requests"""
        try:
            data = json.loads(msg.data.decode())
            conversation_id = data["conversation_id"]
            
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                status = "success"
            else:
                status = "not_found"
            
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": status,
                    "conversation_id": conversation_id
                })
            
            self.logger.info(f"Conversation cleared: {conversation_id}")
        
        except Exception as e:
            self.logger.error(f"Clear conversation failed: {e}")
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "error",
                    "error": str(e)
                })
    
    async def _handle_summarize_conversation(self, msg):
        """Handle conversation summarization requests"""
        try:
            data = json.loads(msg.data.decode())
            conversation_id = data["conversation_id"]
            
            conversation = self.conversations.get(conversation_id)
            
            if conversation:
                await self._summarize_conversation(conversation)
                response = {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "summary": conversation.summary
                }
            else:
                response = {
                    "status": "not_found",
                    "conversation_id": conversation_id
                }
            
            if msg.reply:
                await self._publish_response(msg.reply, response)
        
        except Exception as e:
            self.logger.error(f"Summarize conversation failed: {e}")
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "error",
                    "error": str(e)
                })
    
    async def _handle_chat_generation(self, msg):
        """Handle live chat generation requests"""
        try:
            data = json.loads(msg.data.decode())
            
            task_request = TaskRequest(
                task_id=data.get("task_id", str(uuid.uuid4())),
                task_type="chat_completion",
                payload=data,
                correlation_id=data.get("correlation_id", str(uuid.uuid4()))
            )
            
            result = await self._chat_completion(data)
            
            if msg.reply:
                await self._publish_response(msg.reply, result)
        
        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "error",
                    "error": str(e)
                })
    
    async def _handle_batch_processing(self, msg):
        """Handle batch processing requests"""
        try:
            data = json.loads(msg.data.decode())
            batch_id = data.get("batch_id", str(uuid.uuid4()))
            tasks = data.get("tasks", [])
            
            self.logger.info(f"Processing batch {batch_id} with {len(tasks)} tasks")
            
            results = []
            for task_data in tasks:
                try:
                    task_type = task_data.get("type", "generate_text")
                    payload = task_data.get("payload", {})
                    
                    if task_type == "generate_text":
                        result = await self._generate_text(payload)
                    elif task_type == "summarize":
                        result = await self._summarize_text(payload)
                    else:
                        result = {"status": "error", "error": f"Unknown task type: {task_type}"}
                    
                    results.append({
                        "task_id": task_data.get("id"),
                        "result": result
                    })
                
                except Exception as e:
                    results.append({
                        "task_id": task_data.get("id"),
                        "result": {"status": "error", "error": str(e)}
                    })
            
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "success",
                    "batch_id": batch_id,
                    "results": results,
                    "completed": len(results),
                    "total": len(tasks)
                })
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            if msg.reply:
                await self._publish_response(msg.reply, {
                    "status": "error",
                    "error": str(e)
                })
    
    # Background tasks
    
    async def _conversation_cleanup_loop(self):
        """Clean up old conversations periodically"""
        current_time = time.time()
        to_remove = []
        
        for conv_id, conv in self.conversations.items():
            age = current_time - conv.updated_at
            if age > self.max_conversation_age:
                to_remove.append(conv_id)
        
        for conv_id in to_remove:
            del self.conversations[conv_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old conversations")
    
    async def _persist_token_usage_loop(self):
        """Persist token usage to database periodically"""
        if not self.db_pool:
            return
        
        try:
            query = """
                INSERT INTO llm_token_usage 
                (agent_id, prompt_tokens, completion_tokens, total_cost, recorded_at)
                VALUES ($1, $2, $3, $4, NOW())
            """
            
            await self._db_execute(
                query,
                self.config.agent_id,
                self.token_usage["prompt_tokens"],
                self.token_usage["completion_tokens"],
                self.token_usage["total_cost"]
            )
            
            self.logger.debug(
                "Token usage persisted",
                extra={
                    "tokens": self.token_usage["prompt_tokens"] + self.token_usage["completion_tokens"],
                    "cost": self.token_usage["total_cost"]
                }
            )
        
        except Exception as e:
            self.logger.error(f"Token usage persistence failed: {e}")
    
    async def _persist_model_stats_loop(self):
        """Persist model statistics to database"""
        if not self.db_pool:
            return
        
        try:
            for model_name, stats in self.model_usage_stats.items():
                if stats["requests"] > 0:
                    query = """
                        INSERT INTO llm_model_stats 
                        (agent_id, model_name, requests, tokens_used, errors, 
                         avg_latency_ms, total_cost, recorded_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """
                    
                    await self._db_execute(
                        query,
                        self.config.agent_id,
                        model_name,
                        stats["requests"],
                        stats["tokens"],
                        stats["errors"],
                        stats["avg_latency_ms"],
                        stats["total_cost"]
                    )
            
            self.logger.debug("Model stats persisted")
        
        except Exception as e:
            self.logger.error(f"Model stats persistence failed: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status including LLM-specific metrics"""
        base_status = await super().get_status()
        
        base_status.update({
            "llm_metrics": {
                "token_usage": self.token_usage,
                "model_usage": self.model_usage_stats,
                "active_conversations": len(self.conversations),
                "circuit_breakers": {
                    model: breaker["state"]
                    for model, breaker in self.llm_circuit_breakers.items()
                },
                "available_providers": [
                    provider.value
                    for provider in self.provider_manager.providers.keys()
                ]
            },
            "rag_status": {
                "enabled": self.qdrant_client is not None,
                "collections": list(self.rag_collections.keys()) if self.qdrant_client else []
            }
        })
        
        return base_status
    
    async def stop(self):
        """Cleanup before shutdown"""
        self.logger.info("LLM Agent shutting down...")
        
        # Persist final token usage and stats
        if self.db_pool:
            try:
                await self._persist_token_usage_loop()
                await self._persist_model_stats_loop()
            except Exception as e:
                self.logger.error(f"Final persistence failed: {e}")
        
        # Close LLM providers
        await self.provider_manager.close_all()
        
        # Close Qdrant client
        if self.qdrant_client:
            try:
                self.qdrant_client.close()
            except:
                pass
        
        await super().stop()


if __name__ == "__main__":
    import sys
    
    # Configuration from environment
    config = AgentConfig(
        agent_id=os.getenv("AGENT_ID", f"llm-agent-{uuid.uuid4().hex[:8]}"),
        name=os.getenv("AGENT_NAME", "llm_agent"),
        agent_type="llm",
        capabilities=[
            "text_generation",
            "chat_completion",
            "embeddings",
            "semantic_search",
            "summarization",
            "sentiment_analysis",
            "entity_extraction",
            "rag"
        ],
        nats_url=os.getenv("NATS_URL", "nats://localhost:4222"),
        postgres_url=os.getenv("POSTGRES_URL"),
        redis_url=os.getenv("REDIS_URL"),
        max_concurrent_tasks=int(os.getenv("MAX_CONCURRENT_TASKS", "50")),
        status_publish_interval_seconds=int(os.getenv("STATUS_PUBLISH_INTERVAL", "30")),
        heartbeat_interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL", "10")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "json"),
        log_file=os.getenv("LOG_FILE")
    )
    
    agent = LLMAgent(config)
    
    try:
        asyncio.run(run_agent(agent))
    except KeyboardInterrupt:
        print("\nShutting down LLM Agent...")
        sys.exit(0)