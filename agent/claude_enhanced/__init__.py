"""
Claude-Enhanced Multi-Agent Operations System

This module implements Claude's agent skills into the three-tier agent architecture:
- Strategic Agent (MIP) + Claude reasoning
- Tactical Agent (LP) + Claude planning
- Operational Agent (DP) + Claude optimization

Key Features:
1. Tool Use: Each agent has specialized OR tools + Claude tool calling
2. Extended Thinking: Deep multi-step reasoning
3. Agent Collaboration: Intelligent constraint propagation
4. Adaptive Decision Making: Context-aware optimization
"""

from .enhanced_agents import (
    ClaudeStrategicAgent,
    ClaudeTacticalAgent,
    ClaudeOperationalAgent
)
from .intelligent_orchestrator import IntelligentOrchestrator
from .agent_skills import AgentSkillsToolkit

__all__ = [
    'ClaudeStrategicAgent',
    'ClaudeTacticalAgent', 
    'ClaudeOperationalAgent',
    'IntelligentOrchestrator',
    'AgentSkillsToolkit'
]

