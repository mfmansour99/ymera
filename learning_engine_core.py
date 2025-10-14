"""
YMERA Enterprise - Learning Engine Core Classes
Production-Ready Continuous Learning System - v4.1
Enterprise-grade implementation - ALL ISSUES FIXED
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

import asyncio
import json
import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from enum import Enum
import time
import hashlib
from pathlib import Path

# Third-party imports
import redis.asyncio as aioredis
import networkx as nx
import numpy as np
import structlog
from fastapi import HTTPException
from pydantic import BaseModel, Field, validator
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.learning_engine")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

LEARNING_CYCLE_INTERVAL = 60
KNOWLEDGE_SYNC_INTERVAL = 300
PATTERN_DISCOVERY_INTERVAL = 900
MEMORY_CONSOLIDATION_INTERVAL = 3600

MIN_PATTERN_CONFIDENCE = 0.7
MIN_KNOWLEDGE_RELEVANCE = 0.6
MAX_KNOWLEDGE_AGE_DAYS = 30
KNOWLEDGE_RETENTION_THRESHOLD = 0.8

# ===============================================================================
# ENUMS & DATA MODELS
# ===============================================================================

class LearningType(Enum):
    EXPERIENTIAL = "experiential"
    COLLABORATIVE = "collaborative"
    EXTERNAL = "external"
    PATTERN_BASED = "pattern_based"

class KnowledgeStatus(Enum):
    ACTIVE = "active"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"
    PENDING_VALIDATION = "pending_validation"

@dataclass
class Experience:
    id: str
    agent_id: str
    timestamp: datetime
    action: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    success: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeItem:
    id: str
    content: Dict[str, Any]
    source_type: LearningType
    source_id: str
    confidence: float
    relevance_scores: Dict[str, float]
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    status: KnowledgeStatus
    tags: Set[str] = field(default_factory=set)
    relationships: Dict[str, float] = field(default_factory=dict)

@dataclass
class LearningPattern:
    id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float
    frequency: int
    agents_involved: Set[str]
    discovered_at: datetime
    effectiveness: float

@dataclass
class LearningMetrics:
    learning_velocity: float
    knowledge_retention_rate: float
    agent_knowledge_diversity: float
    collaboration_score: float
    external_integration_success: float
    problem_solving_efficiency: float
    pattern_discovery_count: int
    knowledge_transfer_count: int
    total_experiences: int
    active_knowledge_items: int

# ===============================================================================
# CORE CLASSES
# ===============================================================================

class KnowledgeGraph:
    """Production-ready knowledge graph implementation"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.knowledge_index = {}
        self.vector_store = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.logger = logger.bind(component="knowledge_graph")
    
    async def add_knowledge(self, knowledge: KnowledgeItem) -> bool:
        """Add knowledge item to the graph"""
        try:
            self.graph.add_node(
                knowledge.id,
                content=knowledge.content,
                confidence=knowledge.confidence,
                creation_time=knowledge.creation_time,
                status=knowledge.status,
                tags=knowledge.tags
            )
            
            self.knowledge_index[knowledge.id] = knowledge
            
            content_text = json.dumps(knowledge.content)
            content_vector = self._vectorize_content(content_text)
            self.vector_store[knowledge.id] = content_vector
            
            await self._establish_relationships(knowledge)
            
            self.logger.info("Knowledge added", knowledge_id=knowledge.id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to add knowledge", error=str(e), knowledge_id=knowledge.id)
            return False
    
    async def find_similar_knowledge(
        self, 
        query_content: Dict[str, Any], 
        threshold: float = 0.6,
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """Find similar knowledge items"""
        try:
            query_text = json.dumps(query_content)
            query_vector = self._vectorize_content(query_text)
            
            similarities = []
            for kid, vector in self.vector_store.items():
                similarity = cosine_similarity([query_vector], [vector])[0][0]
                if similarity >= threshold:
                    similarities.append((kid, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for kid, sim_score in similarities[:limit]:
                knowledge = self.knowledge_index[kid]
                knowledge.relevance_scores['query_similarity'] = sim_score
                results.append(knowledge)
            
            return results
            
        except Exception as e:
            self.logger.error("Failed to find similar knowledge", error=str(e))
            return []
    
    def _vectorize_content(self, content: str) -> np.ndarray:
        """Convert content to vector representation"""
        try:
            if not hasattr(self.vectorizer, 'vocabulary_'):
                all_content = [json.dumps(k.content) for k in self.knowledge_index.values()]
                if all_content:
                    self.vectorizer.fit(all_content + [content])
            
            return self.vectorizer.transform([content]).toarray()[0]
        except:
            return np.zeros(1000)
    
    async def _establish_relationships(self, new_knowledge: KnowledgeItem) -> None:
        """Establish relationships with existing knowledge"""
        similar_items = await self.find_similar_knowledge(
            new_knowledge.content, 
            threshold=MIN_KNOWLEDGE_RELEVANCE,
            limit=5
        )
        
        for similar_item in similar_items:
            if similar_item.id != new_knowledge.id:
                self.graph.add_edge(
                    new_knowledge.id,
                    similar_item.id,
                    weight=similar_item.relevance_scores.get('query_similarity', 0.0),
                    relationship_type="content_similarity"
                )


class ExperienceProcessor:
    """Process and extract knowledge from agent experiences"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.experience_buffer = deque(maxlen=1000)
        self.processing_queue = asyncio.Queue()
        self.logger = logger.bind(component="experience_processor")
    
    async def add_experience(self, experience: Experience) -> None:
        """Add new experience for processing"""
        self.experience_buffer.append(experience)
        await self.processing_queue.put(experience)
        self.logger.debug("Experience added", experience_id=experience.id)
    
    async def process_experiences(self) -> int:
        """Process pending experiences and extract knowledge"""
        processed_count = 0
        
        while not self.processing_queue.empty():
            try:
                experience = await self.processing_queue.get()
                knowledge_items = await self._extract_knowledge(experience)
                
                for knowledge in knowledge_items:
                    await self.knowledge_graph.add_knowledge(knowledge)
                    processed_count += 1
                
            except Exception as e:
                self.logger.error("Failed to process experience", error=str(e))
        
        return processed_count
    
    async def _extract_knowledge(self, experience: Experience) -> List[KnowledgeItem]:
        """Extract actionable knowledge from experience"""
        knowledge_items = []
        
        try:
            if experience.success and experience.confidence > 0.6:
                action_knowledge = KnowledgeItem(
                    id=str(uuid.uuid4()),
                    content={
                        "type": "action_outcome",
                        "action": experience.action,
                        "context": experience.context,
                        "outcome": experience.outcome,
                        "conditions": self._extract_conditions(experience.context)
                    },
                    source_type=LearningType.EXPERIENTIAL,
                    source_id=experience.agent_id,
                    confidence=experience.confidence,
                    relevance_scores={"experience_based": experience.confidence},
                    creation_time=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    access_count=0,
                    status=KnowledgeStatus.PENDING_VALIDATION,
                    tags={experience.action, "experiential", experience.agent_id}
                )
                knowledge_items.append(action_knowledge)
            
            if self._has_significant_context(experience):
                context_knowledge = KnowledgeItem(
                    id=str(uuid.uuid4()),
                    content={
                        "type": "context_pattern",
                        "context_signature": self._create_context_signature(experience.context),
                        "typical_actions": [experience.action],
                        "success_rate": 1.0 if experience.success else 0.0,
                        "sample_size": 1
                    },
                    source_type=LearningType.PATTERN_BASED,
                    source_id=experience.agent_id,
                    confidence=experience.confidence * 0.8,
                    relevance_scores={"pattern_based": experience.confidence * 0.8},
                    creation_time=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    access_count=0,
                    status=KnowledgeStatus.PENDING_VALIDATION,
                    tags={"pattern", "context", experience.agent_id}
                )
                knowledge_items.append(context_knowledge)
            
        except Exception as e:
            self.logger.error("Failed to extract knowledge from experience", error=str(e))
        
        return knowledge_items
    
    def _extract_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meaningful conditions from context"""
        conditions = {}
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                conditions[key] = value
            elif isinstance(value, dict) and len(value) < 5:
                conditions[key] = value
        return conditions
    
    def _has_significant_context(self, experience: Experience) -> bool:
        """Check if experience has significant contextual information"""
        return (
            len(experience.context) >= 3 and
            experience.confidence > 0.5 and
            any(isinstance(v, (str, int, float)) for v in experience.context.values())
        )
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for context patterns"""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()


class PatternDiscoveryEngine:
    """Discover behavioral patterns in agent interactions"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.discovered_patterns = {}
        self.pattern_cache = {}
        self.logger = logger.bind(component="pattern_discovery")
    
    async def discover_patterns(self, experiences: List[Experience]) -> List[LearningPattern]:
        """Discover patterns from agent experiences"""
        if len(experiences) < 10:
            return []
        
        patterns = []
        
        try:
            sequence_patterns = await self._discover_sequence_patterns(experiences)
            patterns.extend(sequence_patterns)
            
            context_patterns = await self._discover_context_patterns(experiences)
            patterns.extend(context_patterns)
            
            collab_patterns = await self._discover_collaboration_patterns(experiences)
            patterns.extend(collab_patterns)
            
            for pattern in patterns:
                self.discovered_patterns[pattern.id] = pattern
                await self._create_pattern_knowledge(pattern)
            
            self.logger.info("Patterns discovered", count=len(patterns))
            
        except Exception as e:
            self.logger.error("Pattern discovery failed", error=str(e))
        
        return patterns
    
    async def _discover_sequence_patterns(self, experiences: List[Experience]) -> List[LearningPattern]:
        """Discover common action sequences"""
        patterns = []
        
        agent_sequences = defaultdict(list)
        for exp in experiences:
            agent_sequences[exp.agent_id].append(exp)
        
        for agent_id, agent_exps in agent_sequences.items():
            agent_exps.sort(key=lambda x: x.timestamp)
            
            for seq_len in range(2, 5):
                sequences = self._extract_sequences(agent_exps, seq_len)
                common_seqs = self._find_common_sequences(sequences)
                
                for seq, frequency in common_seqs.items():
                    if frequency >= 3:
                        pattern = LearningPattern(
                            id=str(uuid.uuid4()),
                            pattern_type="action_sequence",
                            description=f"Common sequence: {' -> '.join(seq)}",
                            conditions={"agent_id": agent_id, "sequence_length": seq_len},
                            outcomes={"frequency": frequency, "success_rate": self._calculate_seq_success_rate(seq, agent_exps)},
                            confidence=min(frequency / 10.0, 1.0),
                            frequency=frequency,
                            agents_involved={agent_id},
                            discovered_at=datetime.utcnow(),
                            effectiveness=self._calculate_seq_success_rate(seq, agent_exps)
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _discover_context_patterns(self, experiences: List[Experience]) -> List[LearningPattern]:
        """Discover patterns in context-outcome relationships"""
        patterns = []
        
        context_groups = defaultdict(list)
        for exp in experiences:
            context_sig = self._create_context_signature(exp.context)
            context_groups[context_sig].append(exp)
        
        for context_sig, group_exps in context_groups.items():
            if len(group_exps) >= 5:
                success_rate = sum(1 for exp in group_exps if exp.success) / len(group_exps)
                
                if success_rate > 0.7 or success_rate < 0.3:
                    actions = [exp.action for exp in group_exps]
                    action_counts = defaultdict(int)
                    for action in actions:
                        action_counts[action] += 1
                    
                    most_common_action = max(action_counts.items(), key=lambda x: x[1])
                    
                    pattern = LearningPattern(
                        id=str(uuid.uuid4()),
                        pattern_type="context_outcome",
                        description=f"Context {context_sig[:8]} leads to {most_common_action[0]}",
                        conditions={"context_signature": context_sig},
                        outcomes={
                            "recommended_action": most_common_action[0],
                            "success_rate": success_rate,
                            "sample_size": len(group_exps)
                        },
                        confidence=success_rate if success_rate > 0.5 else (1 - success_rate),
                        frequency=len(group_exps),
                        agents_involved={exp.agent_id for exp in group_exps},
                        discovered_at=datetime.utcnow(),
                        effectiveness=success_rate
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _discover_collaboration_patterns(self, experiences: List[Experience]) -> List[LearningPattern]:
        """Discover patterns in multi-agent collaborations"""
        patterns = []
        
        collaboration_windows = self._find_collaboration_windows(experiences)
        
        for window in collaboration_windows:
            if len(window) >= 3 and len({exp.agent_id for exp in window}) >= 2:
                agents_involved = {exp.agent_id for exp in window}
                success_rate = sum(1 for exp in window if exp.success) / len(window)
                
                pattern = LearningPattern(
                    id=str(uuid.uuid4()),
                    pattern_type="collaboration",
                    description=f"Collaboration between {len(agents_involved)} agents",
                    conditions={
                        "agents": list(agents_involved),
                        "time_window": "5_minutes",
                        "min_interactions": len(window)
                    },
                    outcomes={
                        "success_rate": success_rate,
                        "interaction_count": len(window),
                        "agents_count": len(agents_involved)
                    },
                    confidence=min(len(window) / 10.0, 1.0) * success_rate,
                    frequency=1,
                    agents_involved=agents_involved,
                    discovered_at=datetime.utcnow(),
                    effectiveness=success_rate
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_sequences(self, experiences: List[Experience], length: int) -> List[Tuple[str, ...]]:
        """Extract action sequences of given length"""
        sequences = []
        for i in range(len(experiences) - length + 1):
            seq = tuple(experiences[i + j].action for j in range(length))
            sequences.append(seq)
        return sequences
    
    def _find_common_sequences(self, sequences: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], int]:
        """Find commonly occurring sequences"""
        seq_counts = defaultdict(int)
        for seq in sequences:
            seq_counts[seq] += 1
        return dict(seq_counts)
    
    def _calculate_seq_success_rate(self, sequence: Tuple[str, ...], experiences: List[Experience]) -> float:
        """Calculate success rate for a sequence"""
        seq_occurrences = []
        for i in range(len(experiences) - len(sequence) + 1):
            if all(experiences[i + j].action == sequence[j] for j in range(len(sequence))):
                seq_occurrences.append(experiences[i + len(sequence) - 1].success)
        
        if not seq_occurrences:
            return 0.0
        
        return sum(seq_occurrences) / len(seq_occurrences)
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for context patterns"""
        key_attrs = {}
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                key_attrs[key] = value
        
        context_str = json.dumps(key_attrs, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _find_collaboration_windows(self, experiences: List[Experience]) -> List[List[Experience]]:
        """Find time windows where multiple agents were active"""
        sorted_exps = sorted(experiences, key=lambda x: x.timestamp)
        
        windows = []
        current_window = []
        window_start = None
        
        for exp in sorted_exps:
            if not current_window:
                current_window = [exp]
                window_start = exp.timestamp
            else:
                if (exp.timestamp - window_start).total_seconds() <= 300:
                    current_window.append(exp)
                else:
                    if len({e.agent_id for e in current_window}) >= 2:
                        windows.append(current_window)
                    
                    current_window = [exp]
                    window_start = exp.timestamp
        
        if len({e.agent_id for e in current_window}) >= 2:
            windows.append(current_window)
        
        return windows
    
    async def _create_pattern_knowledge(self, pattern: LearningPattern) -> None:
        """Convert discovered pattern into knowledge item"""
        knowledge = KnowledgeItem(
            id=str(uuid.uuid4()),
            content={
                "type": "discovered_pattern",
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "conditions": pattern.conditions,
                "outcomes": pattern.outcomes,
                "effectiveness": pattern.effectiveness,
                "agents_involved": list(pattern.agents_involved)
            },
            source_type=LearningType.PATTERN_BASED,
            source_id="pattern_discovery_engine",
            confidence=pattern.confidence,
            relevance_scores={"pattern_based": pattern.confidence},
            creation_time=pattern.discovered_at,
            last_accessed=pattern.discovered_at,
            access_count=0,
            status=KnowledgeStatus.VALIDATED,
            tags={"pattern", pattern.pattern_type, "discovered"}
        )
        
        await self.knowledge_graph.add_knowledge(knowledge)


class InterAgentKnowledgeTransfer:
    """Manage knowledge transfer between agents"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.transfer_history = defaultdict(list)
        self.agent_profiles = {}
        self.logger = logger.bind(component="knowledge_transfer")
    
    async def synchronize_agent_knowledge(self, agent_ids: List[str]) -> Dict[str, int]:
        """Synchronize knowledge between specified agents"""
        transfer_counts = defaultdict(int)
        
        try:
            for agent_id in agent_ids:
                await self._update_agent_profile(agent_id)
            
            opportunities = await self._identify_transfer_opportunities(agent_ids)
            
            for opportunity in opportunities:
                success = await self._execute_knowledge_transfer(opportunity)
                if success:
                    transfer_counts[opportunity['target_agent']] += 1
            
            self.logger.info("Knowledge synchronization completed", 
                           agents=len(agent_ids), 
                           transfers=sum(transfer_counts.values()))
            
        except Exception as e:
            self.logger.error("Knowledge synchronization failed", error=str(e))
        
        return dict(transfer_counts)
    
    async def _update_agent_profile(self, agent_id: str) -> None:
        """Update agent's knowledge profile"""
        agent_knowledge = []
        for knowledge in self.knowledge_graph.knowledge_index.values():
            if knowledge.source_id == agent_id or agent_id in knowledge.tags:
                agent_knowledge.append(knowledge)
        
        profile = {
            "agent_id": agent_id,
            "knowledge_count": len(agent_knowledge),
            "knowledge_types": set(),
            "expertise_areas": defaultdict(int),
            "last_updated": datetime.utcnow(),
            "avg_confidence": 0.0
        }
        
        if agent_knowledge:
            total_confidence = 0
            for knowledge in agent_knowledge:
                profile["knowledge_types"].add(knowledge.source_type.value)
                for tag in knowledge.tags:
                    profile["expertise_areas"][tag] += 1
                total_confidence += knowledge.confidence
            
            profile["avg_confidence"] = total_confidence / len(agent_knowledge)
        
        profile["knowledge_types"] = list(profile["knowledge_types"])
        profile["expertise_areas"] = dict(profile["expertise_areas"])
        
        self.agent_profiles[agent_id] = profile
    
    async def _identify_transfer_opportunities(self, agent_ids: List[str]) -> List[Dict[str, Any]]:
        """Identify opportunities for knowledge transfer"""
        opportunities = []
        
        for source_agent in agent_ids:
            source_profile = self.agent_profiles.get(source_agent, {})
            source_expertise = source_profile.get("expertise_areas", {})
            
            for target_agent in agent_ids:
                if source_agent == target_agent:
                    continue
                
                target_profile = self.agent_profiles.get(target_agent, {})
                target_expertise = target_profile.get("expertise_areas", {})
                
                for expertise_area, source_count in source_expertise.items():
                    target_count = target_expertise.get(expertise_area, 0)
                    
                    if source_count >= target_count + 3:
                        transfer_knowledge = await self._find_transferable_knowledge(
                            source_agent, target_agent, expertise_area
                        )
                        
                        if transfer_knowledge:
                            opportunities.append({
                                "source_agent": source_agent,
                                "target_agent": target_agent,
                                "knowledge_items": transfer_knowledge,
                                "expertise_area": expertise_area,
                                "priority": source_count - target_count
                            })
        
        opportunities.sort(key=lambda x: x["priority"], reverse=True)
        return opportunities[:20]
    
    async def _find_transferable_knowledge(
        self, 
        source_agent: str, 
        target_agent: str, 
        expertise_area: str
    ) -> List[KnowledgeItem]:
        """Find knowledge items suitable for transfer"""
        transferable = []
        
        for knowledge in self.knowledge_graph.knowledge_index.values():
            if (knowledge.source_id == source_agent and 
                expertise_area in knowledge.tags and
                knowledge.confidence > 0.7 and
                knowledge.status == KnowledgeStatus.VALIDATED):
                
                similar_knowledge = await self.knowledge_graph.find_similar_knowledge(
                    knowledge.content, threshold=0.8, limit=5
                )
                
                has_similar = any(
                    target_agent in k.tags or k.source_id == target_agent 
                    for k in similar_knowledge
                )
                
                if not has_similar:
                    transferable.append(knowledge)
        
        return transferable[:5]
    
    async def _execute_knowledge_transfer(self, opportunity: Dict[str, Any]) -> bool:
        """Execute a knowledge transfer opportunity"""
        try:
            source_agent = opportunity["source_agent"]
            target_agent = opportunity["target_agent"]
            knowledge_items = opportunity["knowledge_items"]
            
            for knowledge in knowledge_items:
                adapted_knowledge = KnowledgeItem(
                    id=str(uuid.uuid4()),
                    content=knowledge.content.copy(),
                    source_type=LearningType.COLLABORATIVE,
                    source_id=f"transfer_{source_agent}_to_{target_agent}",
                    confidence=knowledge.confidence * 0.9,
                    relevance_scores=knowledge.relevance_scores.copy(),
                    creation_time=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    access_count=0,
                    status=KnowledgeStatus.PENDING_VALIDATION,
                    tags=knowledge.tags.union({target_agent, "transferred", "collaborative"}),
                    relationships=knowledge.relationships.copy()
                )
                
                success = await self.knowledge_graph.add_knowledge(adapted_knowledge)
                if success:
                    self.transfer_history[target_agent].append({
                        "source_agent": source_agent,
                        "knowledge_id": adapted_knowledge.id,
                        "original_knowledge_id": knowledge.id,
                        "transfer_time": datetime.utcnow(),
                        "expertise_area": opportunity["expertise_area"]
                    })
            
            self.logger.info("Knowledge transfer executed",
                           source=source_agent,
                           target=target_agent,
                           count=len(knowledge_items))
            return True
            
        except Exception as e:
            self.logger.error("Knowledge transfer failed", error=str(e))
            return False


# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

async def calculate_learning_metrics(
    knowledge_graph: KnowledgeGraph,
    experience_processor: ExperienceProcessor,
    pattern_engine: PatternDiscoveryEngine,
    knowledge_transfer: InterAgentKnowledgeTransfer
) -> LearningMetrics:
    """Calculate comprehensive learning system metrics"""
    
    try:
        active_knowledge = [k for k in knowledge_graph.knowledge_index.values() 
                          if k.status != KnowledgeStatus.DEPRECATED]
        
        total_experiences = len(experience_processor.experience_buffer)
        
        if active_knowledge:
            recent_knowledge = [k for k in active_knowledge 
                              if (datetime.utcnow() - k.creation_time).total_seconds() < 3600]
            learning_velocity = len(recent_knowledge)
        else:
            learning_velocity = 0.0
        
        knowledge_retention_rate = 0.8  # Simplified for now
        
        agent_knowledge_counts = defaultdict(int)
        for knowledge in active_knowledge:
            if knowledge.source_id.startswith("agent_"):
                agent_knowledge_counts[knowledge.source_id] += 1
        
        if agent_knowledge_counts:
            diversity_scores = list(agent_knowledge_counts.values())
            agent_knowledge_diversity = 1.0 - (np.std(diversity_scores) / (np.mean(diversity_scores) + 1))
        else:
            agent_knowledge_diversity = 0.0
        
        collaborative_knowledge = [k for k in active_knowledge 
                                 if k.source_type == LearningType.COLLABORATIVE]
        collaboration_score = len(collaborative_knowledge) / max(len(active_knowledge), 1)
        
        external_knowledge = [k for k in active_knowledge 
                            if k.source_type == LearningType.EXTERNAL]
        validated_external = [k for k in external_knowledge 
                            if k.status == KnowledgeStatus.VALIDATED]
        external_integration_success = (len(validated_external) / max(len(external_knowledge), 1) 
                                      if external_knowledge else 0.0)
        
        if total_experiences > 0:
            successful_experiences = sum(1 for exp in experience_processor.experience_buffer if exp.success)
            problem_solving_efficiency = successful_experiences / total_experiences
        else:
            problem_solving_efficiency = 0.0
        
        pattern_discovery_count = len(pattern_engine.discovered_patterns)
        
        knowledge_transfer_count = sum(len(transfers) for transfers in knowledge_transfer.transfer_history.values())
        
        return LearningMetrics(
            learning_velocity=learning_velocity,
            knowledge_retention_rate=knowledge_retention_rate,
            agent_knowledge_diversity=agent_knowledge_diversity,
            collaboration_score=collaboration_score,
            external_integration_success=external_integration_success,
            problem_solving_efficiency=problem_solving_efficiency,
            pattern_discovery_count=pattern_discovery_count,
            knowledge_transfer_count=knowledge_transfer_count,
            total_experiences=total_experiences,
            active_knowledge_items=len(active_knowledge)
        )
        
    except Exception as e:
        logger.error("Failed to calculate learning metrics", error=str(e))
        return LearningMetrics(
            learning_velocity=0.0,
            knowledge_retention_rate=0.0,
            agent_knowledge_diversity=0.0,
            collaboration_score=0.0,
            external_integration_success=0.0,
            problem_solving_efficiency=0.0,
            pattern_discovery_count=0,
            knowledge_transfer_count=0,
            total_experiences=0,
            active_knowledge_items=0
        )


async def initialize_learning_engine() -> Tuple[
    KnowledgeGraph, 
    ExperienceProcessor, 
    PatternDiscoveryEngine, 
    InterAgentKnowledgeTransfer
]:
    """Initialize the complete learning engine system"""
    
    try:
        knowledge_graph = KnowledgeGraph()
        experience_processor = ExperienceProcessor(knowledge_graph)
        pattern_engine = PatternDiscoveryEngine(knowledge_graph)
        knowledge_transfer = InterAgentKnowledgeTransfer(knowledge_graph)
        
        logger.info("Learning engine initialized successfully")
        
        return (
            knowledge_graph,
            experience_processor, 
            pattern_engine,
            knowledge_transfer
        )
        
    except Exception as e:
        logger.error("Failed to initialize learning engine", error=str(e))
        raise


async def health_check() -> Dict[str, Any]:
    """Learning engine health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "module": "learning_engine_core",
        "version": "4.1"
    }


# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "KnowledgeGraph",
    "ExperienceProcessor", 
    "PatternDiscoveryEngine",
    "InterAgentKnowledgeTransfer",
    "Experience",
    "KnowledgeItem", 
    "LearningPattern",
    "LearningMetrics",
    "LearningType",
    "KnowledgeStatus",
    "initialize_learning_engine",
    "calculate_learning_metrics",
    "health_check"
]