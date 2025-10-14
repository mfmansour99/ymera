"""
YMERA Enterprise - Pattern Recognition Engine
Production-Ready Behavioral Pattern Discovery System - v4.0
Fixed and deployment-ready version
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from itertools import combinations
from enum import Enum

# Third-party imports
try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis  # Fallback for older versions

import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ymera.learning_engine.pattern_recognition")

# ===============================================================================
# CONFIGURATION
# ===============================================================================

class Settings:
    """Configuration settings for the pattern recognition engine"""
    REDIS_URL: str = "redis://localhost:6379"
    DB_CONNECTION_STRING: str = ""
    ENCRYPTION_KEY: str = ""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Create default settings instance
settings = Settings()

# ===============================================================================
# CONSTANTS
# ===============================================================================

MIN_PATTERN_INSTANCES = 3
MIN_PATTERN_CONFIDENCE = 0.6
MAX_PATTERN_COMPLEXITY = 10
PATTERN_DISCOVERY_INTERVAL = 15 * 60  # 15 minutes
TEMPORAL_WINDOW_SIZE = 3600  # 1 hour
SIMILARITY_THRESHOLD = 0.8
MAX_PATTERNS_PER_TYPE = 100

PATTERN_TYPES = {
    "temporal": "Time-based behavioral patterns",
    "sequential": "Action sequence patterns", 
    "collaborative": "Agent collaboration patterns",
    "performance": "Performance trend patterns",
    "error": "Error occurrence patterns",
    "communication": "Communication patterns",
    "resource": "Resource usage patterns",
    "learning": "Learning efficiency patterns"
}

# ===============================================================================
# DATA MODELS
# ===============================================================================

@dataclass
class PatternRecognitionConfig:
    """Configuration for pattern recognition engine"""
    enabled: bool = True
    min_pattern_instances: int = MIN_PATTERN_INSTANCES
    min_pattern_confidence: float = MIN_PATTERN_CONFIDENCE
    max_pattern_complexity: int = MAX_PATTERN_COMPLEXITY
    discovery_interval: int = PATTERN_DISCOVERY_INTERVAL
    temporal_window_size: int = TEMPORAL_WINDOW_SIZE
    similarity_threshold: float = SIMILARITY_THRESHOLD
    max_patterns_per_type: int = MAX_PATTERNS_PER_TYPE
    enable_real_time_discovery: bool = True
    enable_advanced_analytics: bool = True
    pattern_retention_days: int = 90
    redis_url: str = "redis://localhost:6379"

@dataclass
class PatternInstance:
    """Single instance of a pattern occurrence"""
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

@dataclass
class DiscoveredPattern:
    """Represents a discovered behavioral pattern"""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str = ""
    name: str = ""
    description: str = ""
    instances: List[PatternInstance] = field(default_factory=list)
    confidence: float = 0.0
    significance: float = 0.0
    frequency: int = 0
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    agents_involved: Set[str] = field(default_factory=set)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    predictive_value: float = 0.0
    optimization_potential: float = 0.0

class PatternAnalysisRequest(BaseModel):
    """Request for pattern analysis"""
    events: List[Dict[str, Any]]
    analysis_type: str = "all"
    time_window: Optional[int] = None
    agent_filter: Optional[List[str]] = None
    min_confidence: float = 0.6
    include_predictions: bool = False

class PatternSearchResult(BaseModel):
    """Pattern search result"""
    pattern_id: str
    pattern_type: str
    name: str
    description: str
    confidence: float
    significance: float
    frequency: int
    last_seen: datetime
    instances_count: int
    predictive_value: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# ===============================================================================
# DECORATORS
# ===============================================================================

def track_performance(func):
    """Decorator to track function performance"""
    async def wrapper(*args, **kwargs):
        start_time = datetime.utcnow()
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
            raise
    return wrapper

# ===============================================================================
# ANALYZERS
# ===============================================================================

class BasePatternAnalyzer(ABC):
    """Abstract base class for pattern analyzers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
    
    @abstractmethod
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze events for patterns"""
        pass
    
    @abstractmethod
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate discovered pattern"""
        pass

class TemporalPatternAnalyzer(BasePatternAnalyzer):
    """Analyzes temporal patterns in agent behavior"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze temporal patterns in events"""
        patterns = []
        
        try:
            time_windows = self._group_events_by_time(events)
            
            periodic_patterns = await self._find_periodic_patterns(time_windows)
            patterns.extend(periodic_patterns)
            
            burst_patterns = await self._find_burst_patterns(events)
            patterns.extend(burst_patterns)
            
            trend_patterns = await self._find_trend_patterns(events)
            patterns.extend(trend_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Temporal pattern analysis failed: {str(e)}")
            return []
    
    def _group_events_by_time(self, events: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Group events by time windows"""
        time_windows = defaultdict(list)
        window_size = self.config.get("temporal_window_size", TEMPORAL_WINDOW_SIZE)
        
        for event in events:
            timestamp = event.get("timestamp", datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            
            window_id = int(timestamp.timestamp()) // window_size
            time_windows[window_id].append(event)
        
        return time_windows
    
    async def _find_periodic_patterns(self, time_windows: Dict[int, List[Dict[str, Any]]]) -> List[DiscoveredPattern]:
        """Find periodic patterns in time windows"""
        patterns = []
        
        try:
            window_counts = {window_id: len(events) for window_id, events in time_windows.items()}
            
            if len(window_counts) < 3:
                return patterns
            
            sorted_windows = sorted(window_counts.keys())
            counts = [window_counts[w] for w in sorted_windows]
            
            if len(counts) >= 8:
                fft_result = np.fft.fft(counts)
                frequencies = np.fft.fftfreq(len(counts))
                
                power_spectrum = np.abs(fft_result)
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                period = 1.0 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else 0
                
                if period >= 2:
                    confidence = power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
                    
                    if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                        pattern = DiscoveredPattern(
                            pattern_type="temporal",
                            name=f"Periodic Activity Pattern (Period: {period:.1f} windows)",
                            description=f"Recurring activity pattern with period of {period:.1f} time windows",
                            confidence=float(confidence),
                            significance=float(confidence * len(counts) / max(counts)),
                            frequency=len([c for c in counts if c > np.mean(counts)]),
                            context_requirements={"period": period, "min_activity": float(np.mean(counts))}
                        )
                        
                        for window_id, events in time_windows.items():
                            if len(events) > np.mean(counts):
                                pattern.instances.append(PatternInstance(
                                    timestamp=datetime.fromtimestamp(window_id * self.config.get("temporal_window_size", TEMPORAL_WINDOW_SIZE)),
                                    data={"event_count": len(events), "window_id": window_id},
                                    confidence=float(confidence)
                                ))
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find periodic patterns: {str(e)}")
            return []
    
    async def _find_burst_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find burst patterns in events"""
        patterns = []
        
        try:
            agent_timelines = defaultdict(list)
            
            for event in events:
                agent_id = event.get("agent_id", "unknown")
                timestamp = event.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                agent_timelines[agent_id].append(timestamp)
            
            for agent_id, timestamps in agent_timelines.items():
                if len(timestamps) < 5:
                    continue
                
                timestamps.sort()
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                
                if not intervals:
                    continue
                
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                if std_interval > 0:
                    burst_threshold = mean_interval - 2 * std_interval
                    burst_intervals = [i for i in intervals if i < burst_threshold and i > 0]
                    
                    if len(burst_intervals) >= 3:
                        confidence = len(burst_intervals) / len(intervals)
                        
                        if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                            pattern = DiscoveredPattern(
                                pattern_type="temporal",
                                name=f"Activity Burst Pattern - Agent {agent_id}",
                                description=f"Agent shows burst activity patterns with {len(burst_intervals)} burst intervals",
                                confidence=confidence,
                                significance=confidence * (mean_interval / (burst_threshold + 1)),
                                frequency=len(burst_intervals),
                                agents_involved={agent_id}
                            )
                            
                            for i, interval in enumerate(intervals):
                                if interval < burst_threshold and interval > 0:
                                    pattern.instances.append(PatternInstance(
                                        timestamp=timestamps[i],
                                        agent_id=agent_id,
                                        data={"interval": interval, "burst_intensity": mean_interval / interval},
                                        confidence=confidence
                                    ))
                            
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find burst patterns: {str(e)}")
            return []
    
    async def _find_trend_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find trend patterns in events"""
        patterns = []
        
        try:
            event_types = defaultdict(list)
            
            for event in events:
                event_type = event.get("event_type", "unknown")
                timestamp = event.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                event_types[event_type].append(timestamp)
            
            for event_type, timestamps in event_types.items():
                if len(timestamps) < 5:
                    continue
                
                timestamps.sort()
                
                start_time = timestamps[0]
                end_time = timestamps[-1]
                hours = int((end_time - start_time).total_seconds() / 3600) + 1
                
                if hours < 3:
                    continue
                
                hourly_counts = [0] * hours
                for ts in timestamps:
                    hour_idx = int((ts - start_time).total_seconds() / 3600)
                    if 0 <= hour_idx < hours:
                        hourly_counts[hour_idx] += 1
                
                x = np.arange(hours)
                y = np.array(hourly_counts)
                
                if np.sum(y) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    if abs(r_value) > 0.5 and p_value < 0.05:
                        trend_type = "increasing" if slope > 0 else "decreasing"
                        confidence = abs(r_value)
                        
                        pattern = DiscoveredPattern(
                            pattern_type="temporal",
                            name=f"Activity Trend: {event_type} ({trend_type})",
                            description=f"Event type {event_type} shows a {trend_type} trend over time",
                            confidence=confidence,
                            significance=confidence * len(timestamps),
                            frequency=len(timestamps),
                            context_requirements={
                                "event_type": event_type,
                                "trend_type": trend_type,
                                "r_value": r_value
                            }
                        )
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find trend patterns: {str(e)}")
            return []
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate temporal pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Temporal pattern validation failed: {str(e)}")
            return False

class SequentialPatternAnalyzer(BasePatternAnalyzer):
    """Analyzes sequential patterns in agent actions"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze sequential patterns in events"""
        patterns = []
        
        try:
            agent_sequences = defaultdict(list)
            
            for event in events:
                agent_id = event.get("agent_id")
                action = event.get("action", event.get("event_type"))
                timestamp = event.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                if agent_id and action:
                    agent_sequences[agent_id].append((timestamp, action, event))
            
            for agent_id, sequence in agent_sequences.items():
                sequence.sort(key=lambda x: x[0])  # Sort by timestamp
                
                # Find frequent subsequences
                frequent_subsequences = self._find_frequent_subsequences([s[1] for s in sequence])
                
                for sub_seq, count in frequent_subsequences.items():
                    if count >= self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                        confidence = count / len(sequence)
                        
                        sequence_display = " -> ".join(sub_seq)
                        pattern = DiscoveredPattern(
                            pattern_type="sequential",
                            name=f"Frequent Sequence: {agent_id} - {sequence_display}",
                            description=f"Agent {agent_id} frequently performs sequence: {sequence_display}",
                            confidence=confidence,
                            significance=confidence * count,
                            frequency=count,
                            agents_involved={agent_id},
                            context_requirements={
                                "sequence": list(sub_seq),
                                "agent_id": agent_id
                            }
                        )
                        
                        # Find instances of this subsequence
                        actions_only = [s[1] for s in sequence]
                        for i in range(len(actions_only) - len(sub_seq) + 1):
                            if tuple(actions_only[i:i+len(sub_seq)]) == sub_seq:
                                # Use the event corresponding to the start of the sequence
                                original_event = sequence[i][2]
                                pattern.instances.append(PatternInstance(
                                    timestamp=original_event.get("timestamp", datetime.utcnow()),
                                    agent_id=agent_id,
                                    data={
                                        "sequence_start_event": original_event,
                                        "matched_sequence": list(sub_seq)
                                    },
                                    confidence=confidence
                                ))
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Sequential pattern analysis failed: {str(e)}")
            return []
    
    def _find_frequent_subsequences(self, sequence: List[str], max_len: int = 3) -> Dict[Tuple[str, ...], int]:
        """Find frequent subsequences in a given sequence"""
        frequent_subsequences = defaultdict(int)
        
        for length in range(1, max_len + 1):
            for i in range(len(sequence) - length + 1):
                sub_seq = tuple(sequence[i:i+length])
                frequent_subsequences[sub_seq] += 1
                
        return frequent_subsequences
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate sequential pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sequential pattern validation failed: {str(e)}")
            return False

class CollaborativePatternAnalyzer(BasePatternAnalyzer):
    """Analyzes collaborative patterns between agents"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze collaborative patterns in events"""
        patterns = []
        
        try:
            collaboration_networks = await self._find_collaboration_networks(events)
            patterns.extend(collaboration_networks)
            
            synchronization_patterns = await self._find_synchronization_patterns(events)
            patterns.extend(synchronization_patterns)
            
            role_patterns = await self._find_role_patterns(events)
            patterns.extend(role_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Collaborative pattern analysis failed: {str(e)}")
            return []
    
    def _is_interaction_event(self, event: Dict[str, Any]) -> bool:
        """Check if event represents an interaction"""
        interaction_indicators = [
            "collaboration", "communication", "handoff", "sync",
            "share", "assist", "coordinate", "team"
        ]
        
        event_type = event.get("event_type", "").lower()
        action = event.get("action", "").lower()
        
        # Check if event involves multiple agents
        if "participants" in event.get("data", {}):
            return True
        
        if "target_agent" in event.get("data", {}):
            return True
        
        # Check for interaction keywords
        return any(indicator in event_type or indicator in action 
                  for indicator in interaction_indicators)
    
    async def _find_collaboration_networks(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find collaboration network patterns"""
        patterns = []
        
        try:
            # Build collaboration graph
            collaboration_pairs = defaultdict(int)
            agent_interactions = defaultdict(set)
            
            for event in events:
                agent_id = event.get("agent_id")
                data = event.get("data", {})
                
                # Extract collaboration partners
                partners = []
                if "participants" in data:
                    partners = [p for p in data["participants"] if p != agent_id]
                elif "target_agent" in data:
                    partners = [data["target_agent"]]
                
                for partner in partners:
                    if agent_id and partner:
                        pair = tuple(sorted([agent_id, partner]))
                        collaboration_pairs[pair] += 1
                        agent_interactions[agent_id].add(partner)
                        agent_interactions[partner].add(agent_id)
            
            # Find frequent collaboration pairs
            total_interactions = sum(collaboration_pairs.values())
            
            for (agent1, agent2), count in collaboration_pairs.items():
                if count >= 3:  # Minimum collaboration instances
                    frequency = count / total_interactions
                    
                    if frequency > 0.1:  # At least 10% of interactions
                        pattern = DiscoveredPattern(
                            pattern_type="collaborative",
                            name=f"Frequent Collaboration: {agent1} ↔ {agent2}",
                            description=f"Agents {agent1} and {agent2} collaborate frequently ({count} interactions)",
                            confidence=min(1.0, frequency * 5),
                            significance=frequency * count,
                            frequency=count,
                            agents_involved={agent1, agent2},
                            context_requirements={
                                "collaboration_pair": [agent1, agent2],
                                "min_interactions": 3
                            }
                        )
                        
                        # Find collaboration instances
                        for event in events:
                            event_agent = event.get("agent_id")
                            data = event.get("data", {})
                            
                            partners = []
                            if "participants" in data:
                                partners = data["participants"]
                            elif "target_agent" in data:
                                partners = [data["target_agent"]]
                            
                            if (event_agent == agent1 and agent2 in partners) or \
                               (event_agent == agent2 and agent1 in partners):
                                pattern.instances.append(PatternInstance(
                                    timestamp=event.get("timestamp", datetime.utcnow()),
                                    agent_id=event_agent,
                                    data={
                                        "collaboration_type": event.get("event_type"),
                                        "partners": partners,
                                        "context": data
                                    },
                                    confidence=frequency
                                ))
                        
                        patterns.append(pattern)
            
            # Find collaboration hubs (agents with many connections)
            for agent_id, partners in agent_interactions.items():
                if len(partners) >= 3:  # Connected to at least 3 other agents
                    hub_strength = len(partners) / len(agent_interactions)
                    
                    pattern = DiscoveredPattern(
                        pattern_type="collaborative",
                        name=f"Collaboration Hub: {agent_id}",
                        description=f"Agent {agent_id} acts as collaboration hub with {len(partners)} connections",
                        confidence=min(1.0, hub_strength * 3),
                        significance=hub_strength * len(partners),
                        frequency=len(partners),
                        agents_involved={agent_id},
                        context_requirements={
                            "hub_agent": agent_id,
                            "min_connections": 3
                        }
                    )
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error("Failed to find collaboration networks", error=str(e))
            return []
    
    async def _find_synchronization_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find synchronization patterns between agents"""
        patterns = []
        
        try:
            # Group events by time windows (5-minute windows)
            window_size = 300  # 5 minutes
            time_windows = defaultdict(lambda: defaultdict(list))
            
            for event in events:
                timestamp = event.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                window_id = int(timestamp.timestamp()) // window_size
                agent_id = event.get("agent_id", "unknown")
                time_windows[window_id][agent_id].append(event)
            
            # Find synchronization patterns
            sync_groups = []
            
            for window_id, agent_events in time_windows.items():
                if len(agent_events) >= 2:  # At least 2 agents active
                    # Check if agents are performing similar actions
                    agent_actions = {}
                    for agent_id, events_list in agent_events.items():
                        actions = [e.get("action", e.get("event_type", "")) for e in events_list]
                        agent_actions[agent_id] = actions
                    
                    # Find agents with similar action patterns
                    agents = list(agent_actions.keys())
                    for i in range(len(agents)):
                        for j in range(i + 1, len(agents)):
                            agent1, agent2 = agents[i], agents[j]
                            actions1 = set(agent_actions[agent1])
                            actions2 = set(agent_actions[agent2])
                            
                            # Calculate action similarity
                            if actions1 and actions2:
                                similarity = len(actions1 & actions2) / len(actions1 | actions2)
                                
                                if similarity > 0.5:  # 50% similarity threshold
                                    sync_groups.append({
                                        "window_id": window_id,
                                        "agents": [agent1, agent2],
                                        "similarity": similarity,
                                        "common_actions": list(actions1 & actions2)
                                    })
            
            # Group synchronization instances by agent pairs
            agent_pair_syncs = defaultdict(list)
            
            for sync in sync_groups:
                pair = tuple(sorted(sync["agents"]))
                agent_pair_syncs[pair].append(sync)
            
            # Create patterns for frequent synchronization
            for (agent1, agent2), syncs in agent_pair_syncs.items():
                if len(syncs) >= 3:  # At least 3 synchronization instances
                    avg_similarity = sum(s["similarity"] for s in syncs) / len(syncs)
                    
                    pattern = DiscoveredPattern(
                        pattern_type="collaborative",
                        name=f"Synchronization Pattern: {agent1} ⟷ {agent2}",
                        description=f"Agents {agent1} and {agent2} show synchronized behavior ({len(syncs)} instances)",
                        confidence=avg_similarity,
                        significance=avg_similarity * len(syncs),
                        frequency=len(syncs),
                        agents_involved={agent1, agent2},
                        context_requirements={
                            "sync_agents": [agent1, agent2],
                            "min_similarity": 0.5
                        }
                    )
                    
                    # Add synchronization instances
                    for sync in syncs:
                        pattern.instances.append(PatternInstance(
                            timestamp=datetime.fromtimestamp(sync["window_id"] * window_size),
                            data={
                                "agents": sync["agents"],
                                "similarity": sync["similarity"],
                                "common_actions": sync["common_actions"]
                            },
                            confidence=sync["similarity"]
                        ))
                    
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error("Failed to find synchronization patterns", error=str(e))
            return []
    
    async def _find_role_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find role-based interaction patterns"""
        patterns = []
        
        try:
            agent_roles = defaultdict(lambda: {"initiator": 0, "responder": 0, "coordinator": 0})
            
            for event in events:
                agent_id = event.get("agent_id")
                event_type = event.get("event_type", "").lower()
                data = event.get("data", {})
                
                if not agent_id:
                    continue
                
                # Determine role based on event characteristics
                if any(keyword in event_type for keyword in ["initiate", "start", "create", "lead"]):
                    agent_roles[agent_id]["initiator"] += 1
                elif any(keyword in event_type for keyword in ["respond", "reply", "acknowledge", "follow"]):
                    agent_roles[agent_id]["responder"] += 1
                elif any(keyword in event_type for keyword in ["coordinate", "organize", "manage", "assign"]):
                    agent_roles[agent_id]["coordinator"] += 1
                
                # Check for coordination indicators in data
                if "task_assignment" in data or "resource_allocation" in data:
                    agent_roles[agent_id]["coordinator"] += 1
            
            # Identify dominant roles
            for agent_id, roles in agent_roles.items():
                total_actions = sum(roles.values())
                
                if total_actions >= 5:  # Minimum actions for role analysis
                    dominant_role = max(roles.keys(), key=lambda k: roles[k])
                    role_strength = roles[dominant_role] / total_actions
                    
                    if role_strength > 0.6:  # 60% of actions in one role
                        pattern = DiscoveredPattern(
                            pattern_type="collaborative",
                            name=f"Role Pattern: {agent_id} as {dominant_role.title()}",
                            description=f"Agent {agent_id} primarily acts as {dominant_role} ({role_strength:.1%} of interactions)",
                            confidence=role_strength,
                            significance=role_strength * total_actions,
                            frequency=roles[dominant_role],
                            agents_involved={agent_id},
                            context_requirements={
                                "agent_role": dominant_role,
                                "min_role_strength": 0.6
                            }
                        )
                        
                        # Find role instances
                        for event in events:
                            if event.get("agent_id") == agent_id:
                                event_type = event.get("event_type", "").lower()
                                
                                # Check if event matches the dominant role
                                role_match = False
                                if dominant_role == "initiator" and any(kw in event_type for kw in ["initiate", "start", "create", "lead"]):
                                    role_match = True
                                elif dominant_role == "responder" and any(kw in event_type for kw in ["respond", "reply", "acknowledge", "follow"]):
                                    role_match = True
                                elif dominant_role == "coordinator" and any(kw in event_type for kw in ["coordinate", "organize", "manage", "assign"]):
                                    role_match = True
                                
                                if role_match:
                                    pattern.instances.append(PatternInstance(
                                        timestamp=event.get("timestamp", datetime.utcnow()),
                                        agent_id=agent_id,
                                        data={
                                            "role": dominant_role,
                                            "event_type": event_type,
                                            "role_strength": role_strength
                                        },
                                        confidence=role_strength
                                    ))
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error("Failed to find role patterns", error=str(e))
            return []
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate collaborative pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Collaborative pattern validation failed", error=str(e))
            return False

class PerformancePatternAnalyzer(BasePatternAnalyzer):
    """Analyzes performance-related patterns"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze performance patterns in events"""
        patterns = []
        
        try:
            latency_patterns = await self._find_latency_patterns(events)
            patterns.extend(latency_patterns)
            
            throughput_patterns = await self._find_throughput_patterns(events)
            patterns.extend(throughput_patterns)
            
            resource_patterns = await self._find_resource_usage_patterns(events)
            patterns.extend(resource_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Performance pattern analysis failed: {str(e)}")
            return []
    
    async def _find_latency_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find latency-related patterns"""
        patterns = []
        
        try:
            action_latencies = defaultdict(list)
            
            for event in events:
                action = event.get("action", event.get("event_type"))
                latency = event.get("data", {}).get("latency")
                
                if action and isinstance(latency, (int, float)):
                    action_latencies[action].append(latency)
            
            for action, latencies in action_latencies.items():
                if len(latencies) < 5:
                    continue
                
                mean_latency = np.mean(latencies)
                std_latency = np.std(latencies)
                
                if std_latency > 0:
                    # Identify unusually high latencies (outliers)
                    outlier_threshold = mean_latency + 2 * std_latency
                    high_latency_instances = [l for l in latencies if l > outlier_threshold]
                    
                    if len(high_latency_instances) >= 2:  # At least 2 high latency instances
                        confidence = len(high_latency_instances) / len(latencies)
                        
                        if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                            pattern = DiscoveredPattern(
                                pattern_type="performance",
                                name=f"High Latency Pattern: {action}",
                                description=f"Action {action} shows unusually high latencies (avg: {mean_latency:.2f}s)",
                                confidence=confidence,
                                significance=confidence * mean_latency,
                                frequency=len(high_latency_instances),
                                context_requirements={
                                    "action": action,
                                    "mean_latency": mean_latency,
                                    "outlier_threshold": outlier_threshold
                                }
                            )
                            
                            for event in events:
                                if event.get("action", event.get("event_type")) == action and \
                                   event.get("data", {}).get("latency", 0) > outlier_threshold:
                                    pattern.instances.append(PatternInstance(
                                        timestamp=event.get("timestamp", datetime.utcnow()),
                                        agent_id=event.get("agent_id"),
                                        data={
                                            "latency": event["data"]["latency"],
                                            "action": action
                                        },
                                        confidence=confidence
                                    ))
                            
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find latency patterns: {str(e)}")
            return []
    
    async def _find_throughput_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find throughput-related patterns"""
        patterns = []
        
        try:
            # Group events by time windows (e.g., 1-hour windows)
            window_size = self.config.get("temporal_window_size", TEMPORAL_WINDOW_SIZE) # Default 1 hour
            time_windows = defaultdict(lambda: defaultdict(int)) # {window_id: {agent_id: count}}
            
            for event in events:
                timestamp = event.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                window_id = int(timestamp.timestamp()) // window_size
                agent_id = event.get("agent_id", "unknown")
                time_windows[window_id][agent_id] += 1
            
            for agent_id in set(event.get("agent_id", "unknown") for event in events):
                throughput_data = [] # List of (timestamp, count) for a specific agent
                for window_id in sorted(time_windows.keys()):
                    timestamp = datetime.fromtimestamp(window_id * window_size)
                    count = time_windows[window_id].get(agent_id, 0)
                    throughput_data.append((timestamp, count))
                
                if len(throughput_data) < 5:
                    continue
                
                # Detect significant drops or spikes in throughput
                counts = np.array([item[1] for item in throughput_data])
                if len(counts) > 1:
                    diffs = np.diff(counts)
                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs)
                    
                    if std_diff > 0:
                        # Significant drop
                        drop_threshold = mean_diff - 2 * std_diff
                        for i, d in enumerate(diffs):
                            if d < drop_threshold:
                                confidence = abs(d / mean_diff) if mean_diff != 0 else 1.0
                                if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                                    pattern = DiscoveredPattern(
                                        pattern_type="performance",
                                        name=f"Throughput Drop: Agent {agent_id}",
                                        description=f"Agent {agent_id} experienced a significant throughput drop",
                                        confidence=confidence,
                                        significance=confidence * abs(d),
                                        frequency=1,
                                        agents_involved={agent_id},
                                        context_requirements={
                                            "agent_id": agent_id,
                                            "drop_magnitude": d,
                                            "timestamp": throughput_data[i+1][0].isoformat()
                                        }
                                    )
                                    pattern.instances.append(PatternInstance(
                                        timestamp=throughput_data[i+1][0],
                                        agent_id=agent_id,
                                        data={
                                            "previous_throughput": throughput_data[i][1],
                                            "current_throughput": throughput_data[i+1][1],
                                            "change": d
                                        },
                                        confidence=confidence
                                    ))
                                    patterns.append(pattern)
                        
                        # Significant spike
                        spike_threshold = mean_diff + 2 * std_diff
                        for i, d in enumerate(diffs):
                            if d > spike_threshold:
                                confidence = abs(d / mean_diff) if mean_diff != 0 else 1.0
                                if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                                    pattern = DiscoveredPattern(
                                        pattern_type="performance",
                                        name=f"Throughput Spike: Agent {agent_id}",
                                        description=f"Agent {agent_id} experienced a significant throughput spike",
                                        confidence=confidence,
                                        significance=confidence * d,
                                        frequency=1,
                                        agents_involved={agent_id},
                                        context_requirements={
                                            "agent_id": agent_id,
                                            "spike_magnitude": d,
                                            "timestamp": throughput_data[i+1][0].isoformat()
                                        }
                                    )
                                    pattern.instances.append(PatternInstance(
                                        timestamp=throughput_data[i+1][0],
                                        agent_id=agent_id,
                                        data={
                                            "previous_throughput": throughput_data[i][1],
                                            "current_throughput": throughput_data[i+1][1],
                                            "change": d
                                        },
                                        confidence=confidence
                                    ))
                                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find throughput patterns: {str(e)}")
            return []
    
    async def _find_resource_usage_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Find resource usage patterns"""
        patterns = []
        
        try:
            resource_metrics = defaultdict(lambda: defaultdict(list)) # {agent_id: {metric_name: [value]}}
            
            for event in events:
                agent_id = event.get("agent_id", "unknown")
                data = event.get("data", {})
                
                cpu_usage = data.get("cpu_usage")
                memory_usage = data.get("memory_usage")
                
                if isinstance(cpu_usage, (int, float)):
                    resource_metrics[agent_id]["cpu_usage"].append(cpu_usage)
                if isinstance(memory_usage, (int, float)):
                    resource_metrics[agent_id]["memory_usage"].append(memory_usage)
            
            for agent_id, metrics in resource_metrics.items():
                for metric_name, values in metrics.items():
                    if len(values) < 5:
                        continue
                    
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    
                    if std_value > 0:
                        # High usage patterns
                        high_threshold = mean_value + 2 * std_value
                        high_usage_instances = [v for v in values if v > high_threshold]
                        
                        if len(high_usage_instances) >= 2:
                            confidence = len(high_usage_instances) / len(values)
                            if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                                pattern = DiscoveredPattern(
                                    pattern_type="resource",
                                    name=f"High {metric_name.replace('_', ' ').title()}: Agent {agent_id}",
                                    description=f"Agent {agent_id} shows consistently high {metric_name.replace('_', ' ')} (avg: {mean_value:.2f})",
                                    confidence=confidence,
                                    significance=confidence * mean_value,
                                    frequency=len(high_usage_instances),
                                    agents_involved={agent_id},
                                    context_requirements={
                                        "agent_id": agent_id,
                                        "metric": metric_name,
                                        "mean_value": mean_value,
                                        "high_threshold": high_threshold
                                    }
                                )
                                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find resource usage patterns: {str(e)}")
            return []
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate performance pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Performance pattern validation failed: {str(e)}")
            return False

class ErrorPatternAnalyzer(BasePatternAnalyzer):
    """Analyzes error occurrence patterns"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze error patterns in events"""
        patterns = []
        
        try:
            error_counts = defaultdict(lambda: defaultdict(int)) # {error_type: {agent_id: count}}
            error_timestamps = defaultdict(list) # {error_type: [timestamp]}
            
            for event in events:
                error_type = event.get("data", {}).get("error_type")
                agent_id = event.get("agent_id", "unknown")
                timestamp = event.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                if error_type:
                    error_counts[error_type][agent_id] += 1
                    error_timestamps[error_type].append(timestamp)
            
            # Find frequent error types
            for error_type, agent_counts in error_counts.items():
                total_errors = sum(agent_counts.values())
                if total_errors >= self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                    confidence = total_errors / len(events) # Proportion of all events that are this error
                    if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                        pattern = DiscoveredPattern(
                            pattern_type="error",
                            name=f"Frequent Error: {error_type}",
                            description=f"Error type {error_type} occurs frequently",
                            confidence=confidence,
                            significance=confidence * total_errors,
                            frequency=total_errors,
                            agents_involved=set(agent_counts.keys()),
                            context_requirements={"error_type": error_type}
                        )
                        
                        for event in events:
                            if event.get("data", {}).get("error_type") == error_type:
                                pattern.instances.append(PatternInstance(
                                    timestamp=event.get("timestamp", datetime.utcnow()),
                                    agent_id=event.get("agent_id"),
                                    data=event.get("data", {}),
                                    confidence=confidence
                                ))
                        patterns.append(pattern)
            
            # Find error correlation (e.g., multiple agents failing with same error)
            for error_type, agent_counts in error_counts.items():
                if len(agent_counts) >= 2: # Error affecting multiple agents
                    confidence = len(agent_counts) / len(set(e.get("agent_id") for e in events if e.get("data", {}).get("error_type") == error_type))
                    if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                        affected_agents_str = ", ".join(agent_counts.keys())
                        pattern = DiscoveredPattern(
                            pattern_type="error",
                            name=f"Correlated Error: {error_type}",
                            description=f"Error type {error_type} affects multiple agents: {affected_agents_str}",
                            confidence=confidence,
                            significance=confidence * len(agent_counts),
                            frequency=len(agent_counts),
                            agents_involved=set(agent_counts.keys()),
                            context_requirements={
                                "error_type": error_type,
                                "affected_agents": list(agent_counts.keys())
                            }
                        )
                        patterns.append(pattern)
            
            # Find error bursts (sudden increase in errors)
            for error_type, timestamps in error_timestamps.items():
                if len(timestamps) < 5:
                    continue
                
                timestamps.sort()
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                
                if not intervals:
                    continue
                
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                if std_interval > 0:
                    burst_threshold = mean_interval - 2 * std_interval
                    burst_intervals = [i for i in intervals if i < burst_threshold and i > 0]
                    
                    if len(burst_intervals) >= 2:
                        confidence = len(burst_intervals) / len(intervals)
                        if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                            pattern = DiscoveredPattern(
                                pattern_type="error",
                                name=f"Error Burst: {error_type}",
                                description=f"Error type {error_type} shows burst occurrences",
                                confidence=confidence,
                                significance=confidence * len(burst_intervals),
                                frequency=len(burst_intervals),
                                context_requirements={
                                    "error_type": error_type,
                                    "burst_intervals_count": len(burst_intervals)
                                }
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find error patterns: {str(e)}")
            return []
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate error pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error pattern validation failed: {str(e)}")
            return False

class CommunicationPatternAnalyzer(BasePatternAnalyzer):
    """Analyzes communication patterns between agents"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze communication patterns in events"""
        patterns = []
        
        try:
            communication_pairs = defaultdict(int)
            agent_communication_frequency = defaultdict(int)
            
            for event in events:
                if event.get("event_type") == "communication":
                    sender = event.get("agent_id")
                    receiver = event.get("data", {}).get("receiver_id")
                    
                    if sender and receiver:
                        pair = tuple(sorted([sender, receiver]))
                        communication_pairs[pair] += 1
                        agent_communication_frequency[sender] += 1
                        agent_communication_frequency[receiver] += 1
            
            # Find frequent communication pairs
            for (agent1, agent2), count in communication_pairs.items():
                if count >= self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                    confidence = count / sum(communication_pairs.values()) if sum(communication_pairs.values()) > 0 else 0
                    if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                        pattern = DiscoveredPattern(
                            pattern_type="communication",
                            name=f"Frequent Communication: {agent1} ↔ {agent2}",
                            description=f"Agents {agent1} and {agent2} communicate frequently ({count} times)",
                            confidence=confidence,
                            significance=confidence * count,
                            frequency=count,
                            agents_involved={agent1, agent2},
                            context_requirements={
                                "communication_pair": [agent1, agent2],
                                "min_communications": self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES)
                            }
                        )
                        patterns.append(pattern)
            
            # Find communication hubs
            for agent_id, freq in agent_communication_frequency.items():
                if freq >= self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES) * 2: # Agent communicates a lot
                    confidence = freq / sum(agent_communication_frequency.values()) if sum(agent_communication_frequency.values()) > 0 else 0
                    if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                        pattern = DiscoveredPattern(
                            pattern_type="communication",
                            name=f"Communication Hub: {agent_id}",
                            description=f"Agent {agent_id} acts as a communication hub ({freq} communications)",
                            confidence=confidence,
                            significance=confidence * freq,
                            frequency=freq,
                            agents_involved={agent_id},
                            context_requirements={
                                "communication_hub_agent": agent_id,
                                "min_frequency": self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES) * 2
                            }
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find communication patterns: {str(e)}")
            return []
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate communication pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Communication pattern validation failed: {str(e)}")
            return False

class ResourceUsagePatternAnalyzer(BasePatternAnalyzer):
    """Analyzes resource usage patterns"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze resource usage patterns in events"""
        patterns = []
        
        try:
            resource_metrics = defaultdict(lambda: defaultdict(list)) # {agent_id: {metric_name: [value]}}
            
            for event in events:
                agent_id = event.get("agent_id", "unknown")
                data = event.get("data", {})
                
                cpu_usage = data.get("cpu_usage")
                memory_usage = data.get("memory_usage")
                disk_io = data.get("disk_io")
                network_io = data.get("network_io")
                
                if isinstance(cpu_usage, (int, float)):
                    resource_metrics[agent_id]["cpu_usage"].append(cpu_usage)
                if isinstance(memory_usage, (int, float)):
                    resource_metrics[agent_id]["memory_usage"].append(memory_usage)
                if isinstance(disk_io, (int, float)):
                    resource_metrics[agent_id]["disk_io"].append(disk_io)
                if isinstance(network_io, (int, float)):
                    resource_metrics[agent_id]["network_io"].append(network_io)
            
            for agent_id, metrics in resource_metrics.items():
                for metric_name, values in metrics.items():
                    if len(values) < 5:
                        continue
                    
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    
                    if std_value > 0:
                        # High usage patterns
                        high_threshold = mean_value + 2 * std_value
                        high_usage_instances = [v for v in values if v > high_threshold]
                        
                        if len(high_usage_instances) >= 2:
                            confidence = len(high_usage_instances) / len(values)
                            if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                                pattern = DiscoveredPattern(
                                    pattern_type="resource",
                                    name=f"High {metric_name.replace('_', ' ').title()}: Agent {agent_id}",
                                    description=f"Agent {agent_id} shows consistently high {metric_name.replace('_', ' ')} (avg: {mean_value:.2f})",
                                    confidence=confidence,
                                    significance=confidence * mean_value,
                                    frequency=len(high_usage_instances),
                                    agents_involved={agent_id},
                                    context_requirements={
                                        "agent_id": agent_id,
                                        "metric": metric_name,
                                        "mean_value": mean_value,
                                        "high_threshold": high_threshold
                                    }
                                )
                                patterns.append(pattern)
                        
                        # Low usage patterns (potential underutilization)
                        low_threshold = mean_value - 2 * std_value
                        low_usage_instances = [v for v in values if v < low_threshold]
                        
                        if len(low_usage_instances) >= 2:
                            confidence = len(low_usage_instances) / len(values)
                            if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                                pattern = DiscoveredPattern(
                                    pattern_type="resource",
                                    name=f"Low {metric_name.replace('_', ' ').title()}: Agent {agent_id}",
                                    description=f"Agent {agent_id} shows consistently low {metric_name.replace('_', ' ')} (avg: {mean_value:.2f}) - potential underutilization",
                                    confidence=confidence,
                                    significance=confidence * mean_value,
                                    frequency=len(low_usage_instances),
                                    agents_involved={agent_id},
                                    context_requirements={
                                        "agent_id": agent_id,
                                        "metric": metric_name,
                                        "mean_value": mean_value,
                                        "low_threshold": low_threshold
                                    }
                                )
                                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find resource usage patterns: {str(e)}")
            return []
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate resource usage pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource usage pattern validation failed: {str(e)}")
            return False

class LearningEfficiencyPatternAnalyzer(BasePatternAnalyzer):
    """Analyzes learning efficiency patterns"""
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """Analyze learning efficiency patterns in events"""
        patterns = []
        
        try:
            learning_metrics = defaultdict(lambda: defaultdict(list)) # {agent_id: {metric_name: [value]}}
            
            for event in events:
                agent_id = event.get("agent_id", "unknown")
                data = event.get("data", {})
                
                task_completion_time = data.get("task_completion_time")
                error_rate = data.get("error_rate")
                
                if isinstance(task_completion_time, (int, float)):
                    learning_metrics[agent_id]["task_completion_time"].append(task_completion_time)
                if isinstance(error_rate, (int, float)):
                    learning_metrics[agent_id]["error_rate"].append(error_rate)
            
            for agent_id, metrics in learning_metrics.items():
                for metric_name, values in metrics.items():
                    if len(values) < 5:
                        continue
                    
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    
                    if std_value > 0:
                        # Improvement patterns (e.g., decreasing task completion time or error rate)
                        if metric_name in ["task_completion_time", "error_rate"]:
                            # Check for a significant downward trend
                            x = np.arange(len(values))
                            y = np.array(values)
                            
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            
                            if r_value < -0.5 and p_value < 0.05: # Significant negative correlation
                                confidence = abs(r_value)
                                if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                                    pattern = DiscoveredPattern(
                                        pattern_type="learning",
                                        name=f"Improved {metric_name.replace('_', ' ').title()}: Agent {agent_id}",
                                        description=f"Agent {agent_id} shows improved {metric_name.replace('_', ' ')} (trend: {slope:.2f})",
                                        confidence=confidence,
                                        significance=confidence * abs(slope),
                                        frequency=len(values),
                                        agents_involved={agent_id},
                                        context_requirements={
                                            "agent_id": agent_id,
                                            "metric": metric_name,
                                            "trend_slope": slope,
                                            "r_value": r_value
                                        }
                                    )
                                    patterns.append(pattern)
                        
                        # Degradation patterns (e.g., increasing task completion time or error rate)
                        if metric_name in ["task_completion_time", "error_rate"]:
                            # Check for a significant upward trend
                            x = np.arange(len(values))
                            y = np.array(values)
                            
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            
                            if r_value > 0.5 and p_value < 0.05: # Significant positive correlation
                                confidence = abs(r_value)
                                if confidence > self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                                    pattern = DiscoveredPattern(
                                        pattern_type="learning",
                                        name=f"Degraded {metric_name.replace('_', ' ').title()}: Agent {agent_id}",
                                        description=f"Agent {agent_id} shows degraded {metric_name.replace('_', ' ')} (trend: {slope:.2f})",
                                        confidence=confidence,
                                        significance=confidence * slope,
                                        frequency=len(values),
                                        agents_involved={agent_id},
                                        context_requirements={
                                            "agent_id": agent_id,
                                            "metric": metric_name,
                                            "trend_slope": slope,
                                            "r_value": r_value
                                        }
                                    )
                                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to find learning efficiency patterns: {str(e)}")
            return []
    
    async def validate_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Validate learning efficiency pattern"""
        try:
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Learning efficiency pattern validation failed: {str(e)}")
            return False

# ===============================================================================
# PATTERN RECOGNITION ENGINE
# ===============================================================================

class PatternRecognitionEngine:
    """
    Production-ready pattern recognition engine for behavioral analysis.
    
    Discovers patterns in agent interactions, performance, errors, and more.
    Utilizes various analytical techniques including statistical analysis, 
    machine learning (clustering), and temporal analysis.
    """
    
    def __init__(self, config: Optional[PatternRecognitionConfig] = None):
        self.config = config or PatternRecognitionConfig()
        self.logger = logger
        self.analyzers: Dict[str, BasePatternAnalyzer] = {
            "temporal": TemporalPatternAnalyzer(asdict(self.config)),
            "sequential": SequentialPatternAnalyzer(asdict(self.config)),
            "collaborative": CollaborativePatternAnalyzer(asdict(self.config)),
            "performance": PerformancePatternAnalyzer(asdict(self.config)),
            "error": ErrorPatternAnalyzer(asdict(self.config)),
            "communication": CommunicationPatternAnalyzer(asdict(self.config)),
            "resource": ResourceUsagePatternAnalyzer(asdict(self.config)),
            "learning": LearningEfficiencyPatternAnalyzer(asdict(self.config))
        }
        self.discovered_patterns: Dict[str, DiscoveredPattern] = {}
        self.redis = None
        
    async def _init_redis(self):
        """Initialize Redis connection"""
        if not self.redis:
            self.redis = aioredis.from_url(self.config.redis_url, decode_responses=True)
            self.logger.info(f"Connected to Redis at {self.config.redis_url}")

    async def _close_redis(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None
            self.logger.info("Closed Redis connection")

    async def _store_pattern(self, pattern: DiscoveredPattern):
        """Store a discovered pattern in Redis"""
        await self._init_redis()
        pattern_json = json.dumps(asdict(pattern), default=str)
        await self.redis.set(f"pattern:{pattern.pattern_id}", pattern_json)
        # Set expiration for patterns
        await self.redis.expire(f"pattern:{pattern.pattern_id}", self.config.pattern_retention_days * 24 * 3600)
        self.logger.info(f"Stored pattern {pattern.pattern_id} ({pattern.name})")

    async def _load_pattern(self, pattern_id: str) -> Optional[DiscoveredPattern]:
        """Load a pattern from Redis"""
        await self._init_redis()
        pattern_json = await self.redis.get(f"pattern:{pattern_id}")
        if pattern_json:
            pattern_data = json.loads(pattern_json)
            # Convert datetime strings back to datetime objects
            if "discovered_at" in pattern_data: 
                pattern_data["discovered_at"] = datetime.fromisoformat(pattern_data["discovered_at"])
            if "last_seen" in pattern_data: 
                pattern_data["last_seen"] = datetime.fromisoformat(pattern_data["last_seen"])
            # Convert agents_involved list back to set
            if "agents_involved" in pattern_data: 
                pattern_data["agents_involved"] = set(pattern_data["agents_involved"])
            # Convert instances data
            if "instances" in pattern_data:
                pattern_data["instances"] = [
                    PatternInstance(
                        instance_id=inst["instance_id"],
                        timestamp=datetime.fromisoformat(inst["timestamp"]),
                        agent_id=inst["agent_id"],
                        data=inst["data"],
                        context=inst["context"],
                        confidence=inst["confidence"]
                    ) for inst in pattern_data["instances"]]
            return DiscoveredPattern(**pattern_data)
        return None

    async def _update_pattern(self, pattern: DiscoveredPattern):
        """Update an existing pattern in Redis"""
        pattern.last_seen = datetime.utcnow()
        await self._store_pattern(pattern)

    async def _delete_pattern(self, pattern_id: str):
        """Delete a pattern from Redis"""
        await self._init_redis()
        await self.redis.delete(f"pattern:{pattern_id}")
        self.logger.info(f"Deleted pattern {pattern_id}")

    @track_performance
    async def discover_patterns(self, events: List[Dict[str, Any]], analysis_type: str = "all") -> List[DiscoveredPattern]:
        """Discover behavioral patterns from a list of events"""
        if not self.config.enabled:
            self.logger.info("Pattern recognition engine is disabled.")
            return []

        new_patterns = []
        tasks = []

        for analyzer_name, analyzer_instance in self.analyzers.items():
            if analysis_type == "all" or analyzer_name == analysis_type:
                tasks.append(analyzer_instance.analyze_events(events))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Error during pattern discovery: {res}")
                continue
            for pattern in res:
                if await self._validate_and_deduplicate_pattern(pattern):
                    new_patterns.append(pattern)
                    await self._store_pattern(pattern)

        return new_patterns

    async def _validate_and_deduplicate_pattern(self, new_pattern: DiscoveredPattern) -> bool:
        """Validate a new pattern and check for duplicates"""
        if not self.config.enabled:
            return False

        analyzer = self.analyzers.get(new_pattern.pattern_type)
        if not analyzer or not await analyzer.validate_pattern(new_pattern):
            return False

        # Simple deduplication: check if a similar pattern already exists
        # This can be made more sophisticated with similarity metrics
        for existing_pattern_id, existing_pattern in self.discovered_patterns.items():
            if existing_pattern.pattern_type == new_pattern.pattern_type and \
               existing_pattern.name == new_pattern.name:
                # Update existing pattern instead of adding duplicate
                existing_pattern.last_seen = datetime.utcnow()
                existing_pattern.frequency += new_pattern.frequency
                existing_pattern.instances.extend(new_pattern.instances)
                await self._update_pattern(existing_pattern)
                return False
        
        self.discovered_patterns[new_pattern.pattern_id] = new_pattern
        return True

    @track_performance
    async def get_patterns(self, pattern_type: Optional[str] = None, agent_id: Optional[str] = None, 
                           min_confidence: float = 0.0) -> List[PatternSearchResult]:
        """Retrieve discovered patterns based on criteria"""
        await self._init_redis()
        all_pattern_keys = await self.redis.keys("pattern:*")
        
        results = []
        for key in all_pattern_keys:
            pattern = await self._load_pattern(key.split(":")[1])
            if pattern and pattern.confidence >= min_confidence:
                if pattern_type and pattern.pattern_type != pattern_type:
                    continue
                if agent_id and agent_id not in pattern.agents_involved:
                    continue
                
                results.append(PatternSearchResult(
                    pattern_id=pattern.pattern_id,
                    pattern_type=pattern.pattern_type,
                    name=pattern.name,
                    description=pattern.description,
                    confidence=pattern.confidence,
                    significance=pattern.significance,
                    frequency=pattern.frequency,
                    last_seen=pattern.last_seen,
                    instances_count=len(pattern.instances),
                    predictive_value=pattern.predictive_value
                ))
        
        return results

    @track_performance
    async def predict_next_action(self, agent_id: str, recent_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict the next likely action for an agent based on patterns"""
        # This is a simplified example. A real prediction engine would be more complex.
        
        relevant_patterns = await self.get_patterns(agent_id=agent_id, min_confidence=0.7)
        
        if not relevant_patterns:
            return {"prediction": "No strong patterns found", "confidence": 0.0}
            
        # For simplicity, we'll just pick the most frequent action from sequential patterns
        action_counts = defaultdict(int)
        for pattern_result in relevant_patterns:
            if pattern_result.pattern_type == "sequential":
                pattern = await self._load_pattern(pattern_result.pattern_id)
                if pattern and pattern.context_requirements and "sequence" in pattern.context_requirements:
                    sequence = pattern.context_requirements["sequence"]
                    if len(sequence) > 1:
                        action_counts[sequence[-1]] += pattern.frequency # Count the last action in sequence
        
        if action_counts:
            predicted_action = max(action_counts, key=action_counts.get)
            total_frequency = sum(action_counts.values())
            confidence = action_counts[predicted_action] / total_frequency
            return {"prediction": predicted_action, "confidence": confidence}
        
        return {"prediction": "No sequential patterns for prediction", "confidence": 0.0}

    @track_performance
    async def identify_anomalies(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify anomalous events that deviate from established patterns"""
        anomalies = []
        
        # This is a placeholder. Real anomaly detection would involve:
        # 1. Training models on normal behavior patterns.
        # 2. Scoring new events against these models.
        # 3. Flagging events with low scores as anomalies.
        
        # For now, a simple rule: flag events that don't match any strong patterns
        for event in events:
            is_anomalous = True
            for pattern_id, pattern in self.discovered_patterns.items():
                # Very basic check: if event type matches a pattern type, consider it not anomalous
                if event.get("event_type") == pattern.pattern_type and pattern.confidence > 0.8:
                    is_anomalous = False
                    break
            
            if is_anomalous:
                anomalies.append({"event": event, "reason": "No strong matching pattern found"})
                
        return anomalies

    async def shutdown(self):
        """Gracefully shut down the engine"""
        await self._close_redis()
        self.logger.info("Pattern Recognition Engine shut down.")

# ===============================================================================
# MAIN EXECUTION (for testing/demonstration)
# ===============================================================================

async def main():
    config = PatternRecognitionConfig(
        min_pattern_instances=2,
        min_pattern_confidence=0.5,
        temporal_window_size=60 # 1 minute for testing
    )
    engine = PatternRecognitionEngine(config)

    # Example Events
    test_events = [
        {"agent_id": "agent1", "event_type": "task_start", "timestamp": datetime.utcnow() - timedelta(minutes=10), "data": {"task_id": "T1"}},
        {"agent_id": "agent2", "event_type": "task_start", "timestamp": datetime.utcnow() - timedelta(minutes=9), "data": {"task_id": "T2"}},
        {"agent_id": "agent1", "event_type": "communication", "timestamp": datetime.utcnow() - timedelta(minutes=8), "data": {"receiver_id": "agent2", "message": "Hello"}},
        {"agent_id": "agent2", "event_type": "communication", "timestamp": datetime.utcnow() - timedelta(minutes=7), "data": {"receiver_id": "agent1", "message": "Hi"}},
        {"agent_id": "agent1", "event_type": "task_complete", "timestamp": datetime.utcnow() - timedelta(minutes=6), "data": {"task_id": "T1", "latency": 5.0}},
        {"agent_id": "agent3", "event_type": "task_start", "timestamp": datetime.utcnow() - timedelta(minutes=5), "data": {"task_id": "T3"}},
        {"agent_id": "agent1", "event_type": "communication", "timestamp": datetime.utcnow() - timedelta(minutes=4), "data": {"receiver_id": "agent2", "message": "Follow up"}},
        {"agent_id": "agent2", "event_type": "communication", "timestamp": datetime.utcnow() - timedelta(minutes=3), "data": {"receiver_id": "agent1", "message": "Got it"}},
        {"agent_id": "agent1", "event_type": "task_start", "timestamp": datetime.utcnow() - timedelta(minutes=2), "data": {"task_id": "T4"}},
        {"agent_id": "agent1", "event_type": "task_complete", "timestamp": datetime.utcnow() - timedelta(minutes=1), "data": {"task_id": "T4", "latency": 12.0}},
        {"agent_id": "agent1", "event_type": "error", "timestamp": datetime.utcnow() - timedelta(seconds=30), "data": {"error_type": "API_ERROR", "code": 500}},
        {"agent_id": "agent2", "event_type": "error", "timestamp": datetime.utcnow() - timedelta(seconds=25), "data": {"error_type": "API_ERROR", "code": 500}},
        {"agent_id": "agent1", "event_type": "resource_usage", "timestamp": datetime.utcnow() - timedelta(seconds=20), "data": {"cpu_usage": 90, "memory_usage": 80}},
        {"agent_id": "agent1", "event_type": "resource_usage", "timestamp": datetime.utcnow() - timedelta(seconds=15), "data": {"cpu_usage": 95, "memory_usage": 85}},
        {"agent_id": "agent1", "event_type": "task_start", "timestamp": datetime.utcnow() - timedelta(seconds=10), "data": {"task_id": "T5"}},
        {"agent_id": "agent1", "event_type": "task_complete", "timestamp": datetime.utcnow() - timedelta(seconds=5), "data": {"task_id": "T5", "latency": 3.0}},
    ]

    print("\n--- Discovering Patterns ---")
    discovered_patterns = await engine.discover_patterns(test_events)
    for pattern in discovered_patterns:
        print(f"Discovered: {pattern.name} (Type: {pattern.pattern_type}, Confidence: {pattern.confidence:.2f})")

    print("\n--- Retrieving Patterns ---")
    retrieved_patterns = await engine.get_patterns(min_confidence=0.5)
    for pattern in retrieved_patterns:
        print(f"Retrieved: {pattern.name} (Type: {pattern.pattern_type}, Frequency: {pattern.frequency})")

    print("\n--- Predicting Next Action for agent1 ---")
    prediction = await engine.predict_next_action("agent1", test_events[-3:])
    print(f"Prediction for agent1: {prediction}")

    print("\n--- Identifying Anomalies ---")
    anomalies = await engine.identify_anomalies(test_events)
    if anomalies:
        for anomaly in anomalies:
            print(f"Anomaly: {anomaly['event']['event_type']} by {anomaly['event']['agent_id']} - {anomaly['reason']}")
    else:
        print("No anomalies identified.")

    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

