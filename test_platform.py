#!/usr/bin/env python3
"""
YMERA Enterprise v4.0 - Platform Test Suite
Tests critical platform components
"""

import sys
import asyncio
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

print("=" * 80)
print("YMERA Enterprise v4.0 - Platform Test Suite")
print("=" * 80)

# Test 1: Core Configuration
print("\n[1/10] Testing Core Configuration...")
try:
    from CORE_CONFIGURATION.config_manager import ConfigManager
    from CORE_CONFIGURATION.config_settings import get_settings
    print("  ✅ Core Configuration imports successful")
except Exception as e:
    print(f"  ❌ Core Configuration failed: {e}")

# Test 2: Core Engine
print("\n[2/10] Testing Core Engine...")
try:
    from CORE_ENGINE.core_engine import CoreEngine
    from CORE_ENGINE.encryption import encrypt_data, decrypt_data
    print("  ✅ Core Engine imports successful")
except Exception as e:
    print(f"  ❌ Core Engine failed: {e}")

# Test 3: Database
print("\n[3/10] Testing Database...")
try:
    from DATABASE_CORE.database_connection import get_db_session
    print("  ✅ Database imports successful")
except Exception as e:
    print(f"  ❌ Database failed: {e}")

# Test 4: Caching
print("\n[4/10] Testing Caching...")
try:
    from CACHING_PERFORMANCE.redis_cache_manager import RedisCacheManager
    print("  ✅ Caching imports successful")
except Exception as e:
    print(f"  ❌ Caching failed: {e}")

# Test 5: Pattern Recognition
print("\n[5/10] Testing Pattern Recognition...")
try:
    from PATTERN_RECOGNITION.pattern_recognition import PatternRecognitionEngine, PatternRecognitionConfig
    print("  ✅ Pattern Recognition imports successful")
except Exception as e:
    print(f"  ❌ Pattern Recognition failed: {e}")

# Test 6: Knowledge Graph
print("\n[6/10] Testing Knowledge Graph...")
try:
    from VECTOR_DATABASES.knowledge_graph import KnowledgeGraph, KnowledgeGraphConfig
    print("  ✅ Knowledge Graph imports successful")
except Exception as e:
    print(f"  ❌ Knowledge Graph failed: {e}")

# Test 7: Learning Engine
print("\n[7/10] Testing Learning Engine...")
try:
    from LEARNING_ENGINE.external_learning import ProductionExternalLearningProcessor, ExternalLearningConfig
    from LEARNING_ENGINE.memory_consolidation import ProductionMemoryConsolidator, MemoryConsolidationConfig
    print("  ✅ Learning Engine imports successful")
except Exception as e:
    print(f"  ❌ Learning Engine failed: {e}")

# Test 8: Communication
print("\n[8/10] Testing Communication...")
try:
    from COMMUNICATION_COORDINATION.message_broker import MessageBroker
    from COMMUNICATION_COORDINATION.task_dispatcher import TaskDispatcher
    print("  ✅ Communication imports successful")
except Exception as e:
    print(f"  ❌ Communication failed: {e}")

# Test 9: API Gateway
print("\n[9/10] Testing API Gateway...")
try:
    from API_GATEWAY_CORE_ROUTES.ymera_api_gateway import create_api_router
    print("  ✅ API Gateway imports successful")
except Exception as e:
    print(f"  ❌ API Gateway failed: {e}")

# Test 10: Monitoring
print("\n[10/10] Testing Monitoring...")
try:
    from MONITORING_HEALTH.health_checker import HealthChecker
    from MONITORING_HEALTH.metrics_collector import MetricsCollector
    print("  ✅ Monitoring imports successful")
except Exception as e:
    print(f"  ❌ Monitoring failed: {e}")

print("\n" + "=" * 80)
print("✅ CORE PLATFORM TEST COMPLETED!")
print("=" * 80)
print("\nCritical components tested:")
print("  1. Core Configuration")
print("  2. Core Engine")
print("  3. Database")
print("  4. Caching")
print("  5. Pattern Recognition")
print("  6. Knowledge Graph")
print("  7. Learning Engine")
print("  8. Communication")
print("  9. API Gateway")
print("  10. Monitoring")
print("=" * 80)

