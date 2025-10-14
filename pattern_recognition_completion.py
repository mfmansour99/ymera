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
                    frequency = count / total_interactions if total_interactions > 0 else 0
                    
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
                    hub_strength = len(partners) / len(agent_interactions) if len(agent_interactions) > 0 else 0
                    
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
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
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
            # Analyze interaction directions and types
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
            # Check minimum instances
            if len(pattern.instances) < self.config.get("min_pattern_instances", MIN_PATTERN_INSTANCES):
                return False
            
            # Check confidence threshold
            if pattern.confidence < self.config.get("min_pattern_confidence", MIN_PATTERN_CONFIDENCE):
                return False
            
            # Check agent involvement
            if len(pattern.agents_involved) < 1:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error("Collaborative pattern validation failed", error=str(e))
            return False

class PatternRecognitionEngine:
    """
    Production-ready pattern recognition engine for behavioral analysis.
    
    Discovers patterns in agent behavior, interactions, and system performance
    using multiple analysis strategies and machine learning techniques.
    """
    
    def __init__(
        self,
        config: PatternRecognitionConfig,
        knowledge_graph,
        metrics_collector
    ):
        self.config = config
        self.knowledge_graph = knowledge_graph
        self.metrics_collector = metrics_collector
        self.logger = logger.bind(component="pattern_recognition_engine")
        
        # Initialize analyzers
        self._analyzers = {
            "temporal": TemporalPatternAnalyzer(config.__dict__),
            "sequential": SequentialPatternAnalyzer(config.__dict__),
            "collaborative": CollaborativePatternAnalyzer(config.__dict__)
        }
        
        # Pattern storage
        self._discovered_patterns = {}
        self._pattern_cache = {}
        
        # Performance tracking
        self._analysis_performance = []
        self._discovery_stats = defaultdict(int)
        
        # Health status
        self._health_status = "unknown"
        self._is_initialized = False
    
    async def _initialize_resources(self) -> None:
        """Initialize pattern recognition engine resources"""
        try:
            self.logger.info("Initializing pattern recognition engine")
            
            # Initialize Redis for pattern caching
            self._redis_client = await aioredis.from_url(
                settings.REDIS_URL,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Load existing patterns from knowledge graph
            await self._load_existing_patterns()
            
            # Start real-time discovery if enabled
            if self.config.enable_real_time_discovery:
                asyncio.create_task(self._real_time_discovery_loop())
            
            self._is_initialized = True
            self._health_status = "healthy"
            
            self.logger.info(
                "Pattern recognition engine initialized successfully",
                analyzers_count=len(self._analyzers),
                patterns_loaded=len(self._discovered_patterns)
            )
            
        except Exception as e:
            self._health_status = "unhealthy"
            self.logger.error("Failed to initialize pattern recognition engine", error=str(e))
            raise
    
    async def _load_existing_patterns(self) -> None:
        """Load existing patterns from knowledge graph"""
        try:
            from .knowledge_graph import KnowledgeQuery
            
            query = KnowledgeQuery(
                query_type="exact",
                query_data={"content": {"type": "pattern"}},
                max_results=1000,
                include_connections=False
            )
            
            results = await self.knowledge_graph.query_knowledge(query)
            
            for result in results:
                pattern_data = result.content
                if "pattern_id" in pattern_data:
                    self._discovered_patterns[pattern_data["pattern_id"]] = pattern_data
            
            self.logger.info(f"Loaded {len(results)} existing patterns")
            
        except Exception as e:
            self.logger.warning("Failed to load existing patterns", error=str(e))
    
    @track_performance
    async def analyze_event_patterns(self, events: List[Dict[str, Any]]) -> List[DiscoveredPattern]:
        """
        Analyze events for behavioral patterns.
        
        Args:
            events: List of events to analyze
            
        Returns:
            List of discovered patterns
        """
        try:
            analysis_start = datetime.utcnow()
            
            self.logger.debug(
                "Starting pattern analysis",
                events_count=len(events)
            )
            
            all_patterns = []
            
            # Run all analyzers
            for analyzer_name, analyzer in self._analyzers.items():
                try:
                    patterns = await analyzer.analyze_events(events)
                    
                    # Validate patterns
                    valid_patterns = []
                    for pattern in patterns:
                        if await analyzer.validate_pattern(pattern):
                            valid_patterns.append(pattern)
                            self._discovery_stats[analyzer_name] += 1
                    
                    all_patterns.extend(valid_patterns)
                    
                    self.logger.debug(
                        "Analyzer completed",
                        analyzer=analyzer_name,
                        patterns_found=len(valid_patterns)
                    )
                    
                except Exception as e:
                    self.logger.error(
                        "Analyzer failed",
                        analyzer=analyzer_name,
                        error=str(e)
                    )
            
            # Deduplicate and merge similar patterns
            unique_patterns = await self._deduplicate_patterns(all_patterns)
            
            # Store patterns in knowledge graph
            for pattern in unique_patterns:
                await self._store_pattern(pattern)
            
            # Update metrics
            analysis_duration = (datetime.utcnow() - analysis_start).total_seconds()
            self._analysis_performance.append({
                "duration": analysis_duration,
                "events_analyzed": len(events),
                "patterns_found": len(unique_patterns),
                "timestamp": analysis_start
            })
            
            # Keep only last 100 analysis records
            if len(self._analysis_performance) > 100:
                self._analysis_performance.pop(0)
            
            await self.metrics_collector.update_metrics({
                "patterns_discovered": len(unique_patterns),
                "pattern_analysis_duration": analysis_duration,
                "events_analyzed": len(events)
            })
            
            self.logger.info(
                "Pattern analysis completed",
                events_analyzed=len(events),
                patterns_discovered=len(unique_patterns),
                duration=analysis_duration
            )
            
            return unique_patterns
            
        except Exception as e:
            self.logger.error("Pattern analysis failed", error=str(e))
            raise PatternRecognitionError(f"Pattern analysis failed: {str(e)}")
    
    async def discover_new_patterns(self) -> List[DiscoveredPattern]:
        """
        Discover new patterns from recent knowledge graph data.
        
        Returns:
            List of newly discovered patterns
        """
        try:
            self.logger.info("Starting new pattern discovery")
            
            # Get recent events from knowledge graph
            from .knowledge_graph import KnowledgeQuery
            
            recent_time = datetime.utcnow() - timedelta(hours=1)
            
            query = KnowledgeQuery(
                query_type="temporal",
                query_data={"start_time": recent_time},
                max_results=1000,
                include_connections=True
            )
            
            results = await self.knowledge_graph.query_knowledge(query)
            
            # Convert knowledge results to events
            events = []
            for result in results:
                event = {
                    "event_id": result.node_id,
                    "timestamp": result.created_at,
                    "event_type": result.node_type,
                    "agent_id": result.content.get("agent_id"),
                    "data": result.content,
                    "confidence": result.confidence
                }
                events.append(event)
            
            # Analyze for patterns
            new_patterns = await self.analyze_event_patterns(events)
            
            # Filter out patterns we already know
            truly_new_patterns = []
            for pattern in new_patterns:
                if not await self._is_known_pattern(pattern):
                    truly_new_patterns.append(pattern)
                    self._discovered_patterns[pattern.pattern_id] = pattern
            
            self.logger.info(
                "New pattern discovery completed",
                events_analyzed=len(events),
                new_patterns=len(truly_new_patterns)
            )
            
            return truly_new_patterns
            
        except Exception as e:
            self.logger.error("New pattern discovery failed", error=str(e))
            return []
    
    async def _deduplicate_patterns(self, patterns: List[DiscoveredPattern]) -> List[DiscoveredPattern]:
        """Remove duplicate and similar patterns"""
        unique_patterns = []
        
        try:
            for pattern in patterns:
                is_duplicate = False
                
                for existing in unique_patterns:
                    similarity = await self._calculate_pattern_similarity(pattern, existing)
                    
                    if similarity > self.config.similarity_threshold:
                        # Merge patterns by keeping the one with higher confidence
                        if pattern.confidence > existing.confidence:
                            unique_patterns.remove(existing)
                            unique_patterns.append(pattern)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_patterns.append(pattern)
            
            return unique_patterns
            
        except Exception as e:
            self.logger.error("Pattern deduplication failed", error=str(e))
            return patterns  # Return original patterns if deduplication fails
    
    async def _calculate_pattern_similarity(self, pattern1: DiscoveredPattern, pattern2: DiscoveredPattern) -> float:
        """Calculate similarity between two patterns"""
        try:
            # Different pattern types are not similar
            if pattern1.pattern_type != pattern2.pattern_type:
                return 0.0
            
            # Compare agents involved
            agents1 = pattern1.agents_involved
            agents2 = pattern2.agents_involved
            
            if agents1 and agents2:
                agent_similarity = len(agents1 & agents2) / len(agents1 | agents2)
            else:
                agent_similarity = 0.5  # Neutral if no agent info
            
            # Compare context requirements
            context1 = pattern1.context_requirements
            context2 = pattern2.context_requirements
            
            context_similarity = 0.0
            if context1 and context2:
                common_keys = set(context1.keys()) & set(context2.keys())
                if common_keys:
                    matches = sum(1 for key in common_keys if context1[key] == context2[key])
                    context_similarity = matches / len(common_keys)
            
            # Compare name similarity (simple word overlap)
            name1_words = set(pattern1.name.lower().split())
            name2_words = set(pattern2.name.lower().split())
            
            if name1_words and name2_words:
                name_similarity = len(name1_words & name2_words) / len(name1_words | name2_words)
            else:
                name_similarity = 0.0
            
            # Weighted average
            total_similarity = (
                agent_similarity * 0.4 +
                context_similarity * 0.4 +
                name_similarity * 0.2
            )
            
            return total_similarity
            
        except Exception as e:
            self.logger.error("Pattern similarity calculation failed", error=str(e))
            return 0.0
    
    async def _store_pattern(self, pattern: DiscoveredPattern) -> None:
        """Store pattern in knowledge graph"""
        try:
            from .knowledge_graph import KnowledgeItem
            
            pattern_data = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "name": pattern.name,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "significance": pattern.significance,
                "frequency": pattern.frequency,
                "discovered_at": pattern.discovered_at.isoformat(),
                "last_seen": pattern.last_seen.isoformat(),
                "agents_involved": list(pattern.agents_involved),
                "context_requirements": pattern.context_requirements,
                "predictive_value": pattern.predictive_value,
                "optimization_potential": pattern.optimization_potential,
                "instances_count": len(pattern.instances)
            }
            
            knowledge_item = KnowledgeItem(
                content=pattern_data,
                node_type="pattern",
                confidence=pattern.confidence,
                source="pattern_recognition",
                tags=["pattern", pattern.pattern_type]
            )
            
            await self.knowledge_graph.add_knowledge_item(knowledge_item)
            
        except Exception as e:
            self.logger.error("Failed to store pattern", pattern_id=pattern.pattern_id, error=str(e))
    
    async def _is_known_pattern(self, pattern: DiscoveredPattern) -> bool:
        """Check if pattern is already known"""
        try:
            for known_pattern in self._discovered_patterns.values():
                if isinstance(known_pattern, dict):
                    # Convert dict to DiscoveredPattern for comparison
                    known_pattern_obj = DiscoveredPattern(
                        pattern_id=known_pattern.get("pattern_id", ""),
                        pattern_type=known_pattern.get("pattern_type", ""),
                        name=known_pattern.get("name", ""),
                        description=known_pattern.get("description", ""),
                        confidence=known_pattern.get("confidence", 0.0),
                        agents_involved=set(known_pattern.get("agents_involved", [])),
                        context_requirements=known_pattern.get("context_requirements", {})
                    )
                else:
                    known_pattern_obj = known_pattern
                
                similarity = await self._calculate_pattern_similarity(pattern, known_pattern_obj)
                if similarity > self.config.similarity_threshold:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error("Known pattern check failed", error=str(e))
            return False
    
    async def _real_time_discovery_loop(self) -> None:
        """Real-time pattern discovery loop"""
        self.logger.info("Starting real-time pattern discovery loop")
        
        while True:
            try:
                await asyncio.sleep(self.config.discovery_interval)
                
                # Discover new patterns
                new_patterns = await self.discover_new_patterns()
                
                if new_patterns:
                    self.logger.info(f"Discovered {len(new_patterns)} new patterns in real-time")
                
            except Exception as e:
                self.logger.error("Error in real-time discovery loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def get_pattern_analytics(self) -> Dict[str, Any]:
        """Get pattern recognition analytics"""
        try:
            analytics = {
                "total_patterns": len(self._discovered_patterns),
                "patterns_by_type": defaultdict(int),
                "discovery_stats": dict(self._discovery_stats),
                "performance": {
                    "avg_analysis_time": 0.0,
                    "total_analyses": len(self._analysis_performance),
                    "total_events_analyzed": sum(a["events_analyzed"] for a in self._analysis_performance)
                },
                "pattern_quality": {
                    "avg_confidence": 0.0,
                    "avg_significance": 0.0,
                    "high_confidence_patterns": 0
                }
            }
            
            # Calculate performance metrics
            if self._analysis_performance:
                analytics["performance"]["avg_analysis_time"] = sum(
                    a["duration"] for a in self._analysis_performance
                ) / len(self._analysis_performance)
            
            # Analyze pattern quality
            if self._discovered_patterns:
                confidences = []
                significances = []
                
                for pattern in self._discovered_patterns.values():
                    if isinstance(pattern, dict):
                        pattern_type = pattern.get("pattern_type", "unknown")
                        confidence = pattern.get("confidence", 0.0)
                        significance = pattern.get("significance", 0.0)
                    else:
                        pattern_type = pattern.pattern_type
                        confidence = pattern.confidence
                        significance = pattern.significance
                    
                    analytics["patterns_by_type"][pattern_type] += 1
                    confidences.append(confidence)
                    significances.append(significance)
                    
                    if confidence > 0.8:
                        analytics["pattern_quality"]["high_confidence_patterns"] += 1
                
                if confidences:
                    analytics["pattern_quality"]["avg_confidence"] = sum(confidences) / len(confidences)
                if significances:
                    analytics["pattern_quality"]["avg_significance"] = sum(significances) / len(significances)
            
            return analytics
            
        except Exception as e:
            self.logger.error("Failed to get pattern analytics", error=str(e))
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Pattern recognition engine health check"""
        try:
            # Test Redis connection
            redis_healthy = False
            try:
                await self._redis_client.ping()
                redis_healthy = True
            except Exception:
                pass
            
            return {
                "status": self._health_status,
                "initialized": self._is_initialized,
                "analyzers_count": len(self._analyzers),
                "patterns_discovered": len(self._discovered_patterns),
                "redis_healthy": redis_healthy,
                "analysis_performance_samples": len(self._analysis_performance),
                "real_time_discovery": self.config.enable_real_time_discovery,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self) -> None:
        """Cleanup pattern recognition resources"""
        try:
            self.logger.info("Cleaning up pattern recognition engine resources")
            
            # Close Redis connection
            if hasattr(self, '_