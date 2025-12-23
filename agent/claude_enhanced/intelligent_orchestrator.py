"""
Intelligent Orchestrator for Claude-Enhanced Multi-Agent System

Coordinates the three-tier agent architecture with intelligent decision making:
- Strategic (MIP) -> Tactical (LP) -> Operational (DP)
- Constraint propagation between agents
- Collaborative reasoning
- Performance monitoring
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import os
import time

from .enhanced_agents import (
    ClaudeStrategicAgent,
    ClaudeTacticalAgent,
    ClaudeOperationalAgent,
    AgentResponse
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.orchestrator.definitions import (
    OperationsContext, StrategicDecision, TacticalDecision, OperationalDecision
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for the entire pipeline execution"""
    total_time: float = 0
    strategic_time: float = 0
    tactical_time: float = 0
    operational_time: float = 0
    total_cost: float = 0
    total_value: float = 0
    success: bool = False
    tool_calls_count: int = 0
    reasoning_generated: bool = False


@dataclass
class CollaborationResult:
    """Result from agent collaboration"""
    strategic_decision: Optional[StrategicDecision] = None
    tactical_decision: Optional[TacticalDecision] = None
    operational_decision: Optional[OperationalDecision] = None
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    reasoning_summary: str = ""
    agent_logs: List[Dict] = field(default_factory=list)


class IntelligentOrchestrator:
    """
    Intelligent Multi-Agent Orchestrator
    
    Coordinates the three-tier agent system with:
    1. Claude-enhanced reasoning at each layer
    2. Intelligent constraint propagation
    3. Collaborative decision making
    4. Performance monitoring and logging
    """
    
    def __init__(self, api_key: Optional[str] = None, use_reasoning: bool = True):
        """
        Initialize the orchestrator with Claude-enhanced agents.
        
        Args:
            api_key: Anthropic API key for Claude reasoning
            use_reasoning: Whether to use Claude for enhanced reasoning
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_reasoning = use_reasoning and self.api_key is not None
        
        # Initialize enhanced agents
        self.strategic_agent = ClaudeStrategicAgent(api_key=self.api_key)
        self.tactical_agent = ClaudeTacticalAgent(api_key=self.api_key)
        self.operational_agent = ClaudeOperationalAgent(api_key=self.api_key)
        
        # Pipeline state
        self.last_result: Optional[CollaborationResult] = None
        self.execution_history = []
        
        logger.info(f"Orchestrator initialized. Claude reasoning: {'enabled' if self.use_reasoning else 'disabled'}")
    
    def run_pipeline(self, context: OperationsContext, 
                    use_reasoning: Optional[bool] = None) -> OperationsContext:
        """
        Execute the full operations pipeline with intelligent coordination.
        
        Strategic (MIP) -> Tactical (LP) -> Operational (DP)
        
        Args:
            context: Operations context with input data
            use_reasoning: Override for Claude reasoning (None = use default)
            
        Returns:
            Updated context with all decisions populated
        """
        use_reasoning = use_reasoning if use_reasoning is not None else self.use_reasoning
        
        logger.info("=" * 60)
        logger.info("🚀 Starting Intelligent Multi-Agent Pipeline")
        logger.info(f"   Claude Reasoning: {'Enabled' if use_reasoning else 'Disabled'}")
        logger.info("=" * 60)
        
        start_time = time.time()
        metrics = PipelineMetrics()
        reasoning_parts = []
        agent_logs = []
        
        # Phase 1: Strategic Decision (MIP)
        logger.info("\n📊 Phase 1: Strategic Planning (Facility Location)")
        phase_start = time.time()
        
        strategic_response = self.strategic_agent.act(
            fixed_costs=context.potential_facilities_costs,
            transport_costs=context.transport_costs_full,
            demand=context.customer_demands,
            capacity=context.potential_facilities_capacities,
            use_reasoning=use_reasoning
        )
        
        metrics.strategic_time = time.time() - phase_start
        metrics.tool_calls_count += len(strategic_response.tool_calls)
        agent_logs.append({
            "phase": "strategic",
            "success": strategic_response.success,
            "tool_calls": strategic_response.tool_calls,
            "message": strategic_response.message
        })
        
        if strategic_response.reasoning:
            reasoning_parts.append(f"**Strategic Decision:**\n{strategic_response.reasoning}")
            metrics.reasoning_generated = True
        
        if not strategic_response.success:
            logger.error(f"❌ Strategic Phase failed: {strategic_response.message}")
            context.strategic_plan = StrategicDecision([], np.array([]), np.array([]), 0.0, is_feasible=False)
            metrics.success = False
            self.last_result = CollaborationResult(
                strategic_decision=context.strategic_plan,
                metrics=metrics,
                reasoning_summary="\n\n".join(reasoning_parts),
                agent_logs=agent_logs
            )
            return context
        
        # Convert strategic response to decision
        open_facilities = strategic_response.decision['open_facilities']
        context.strategic_plan = StrategicDecision(
            open_facilities_indices=open_facilities,
            facility_capacities=context.potential_facilities_capacities,
            transport_costs_matrix=context.transport_costs_full,
            total_fixed_cost=strategic_response.metrics.get('total_cost', 0),
            is_feasible=True
        )
        
        logger.info(f"   ✅ Open facilities: {open_facilities}")
        logger.info(f"   📝 Tool calls: {strategic_response.tool_calls}")
        
        # Phase 2: Tactical Decision (LP)
        logger.info("\n🚚 Phase 2: Tactical Logistics (Flow Optimization)")
        phase_start = time.time()
        
        # Constraint propagation: Filter to open facilities only
        strat_plan = context.strategic_plan
        active_supply = strat_plan.facility_capacities[open_facilities]
        active_costs = strat_plan.transport_costs_matrix[open_facilities]
        
        tactical_response = self.tactical_agent.act(
            supply=active_supply,
            demand=context.customer_demands,
            costs=active_costs,
            open_facility_indices=open_facilities,
            use_reasoning=use_reasoning
        )
        
        metrics.tactical_time = time.time() - phase_start
        metrics.tool_calls_count += len(tactical_response.tool_calls)
        agent_logs.append({
            "phase": "tactical",
            "success": tactical_response.success,
            "tool_calls": tactical_response.tool_calls,
            "message": tactical_response.message
        })
        
        if tactical_response.reasoning:
            reasoning_parts.append(f"**Tactical Decision:**\n{tactical_response.reasoning}")
        
        if not tactical_response.success:
            logger.error(f"❌ Tactical Phase failed: {tactical_response.message}")
            context.tactical_plan = TacticalDecision(np.array([]), {}, 0.0, is_feasible=False)
            metrics.success = False
            self.last_result = CollaborationResult(
                strategic_decision=context.strategic_plan,
                tactical_decision=context.tactical_plan,
                metrics=metrics,
                reasoning_summary="\n\n".join(reasoning_parts),
                agent_logs=agent_logs
            )
            return context
        
        # Convert tactical response to decision
        allocation = tactical_response.decision['transport_matrix']
        facility_volumes = tactical_response.decision['facility_inbound_volumes']
        
        # Convert facility volumes keys to int if they're strings
        if facility_volumes:
            facility_volumes = {int(k): v for k, v in facility_volumes.items()}
        
        context.tactical_plan = TacticalDecision(
            transport_allocation=allocation,
            facility_inbound_volumes=facility_volumes,
            total_transport_cost=tactical_response.metrics.get('total_cost', 0),
            is_feasible=True
        )
        
        logger.info(f"   ✅ Transport cost: ${tactical_response.metrics.get('total_cost', 0):.2f}")
        logger.info(f"   📝 Tool calls: {tactical_response.tool_calls}")
        
        # Phase 3: Operational Decision (DP)
        logger.info("\n📦 Phase 3: Operational Optimization (Inventory Mix)")
        phase_start = time.time()
        
        # Constraint propagation: Use facility volumes as capacity constraints
        operational_response = self.operational_agent.act(
            mode='multi_facility',
            facility_capacities=facility_volumes,
            values=context.item_values,
            weights=context.item_weights,
            use_reasoning=use_reasoning
        )
        
        metrics.operational_time = time.time() - phase_start
        metrics.tool_calls_count += len(operational_response.tool_calls)
        agent_logs.append({
            "phase": "operational",
            "success": operational_response.success,
            "tool_calls": operational_response.tool_calls,
            "message": operational_response.message
        })
        
        if operational_response.reasoning:
            reasoning_parts.append(f"**Operational Decision:**\n{operational_response.reasoning}")
        
        if operational_response.success:
            facility_plans = operational_response.decision.get('facility_inventory_plans', {})
            context.operational_plan = OperationalDecision(
                facility_inventory_plans=facility_plans,
                total_inventory_value=operational_response.metrics.get('total_value', 0),
                is_feasible=True
            )
            metrics.total_value = operational_response.metrics.get('total_value', 0)
            logger.info(f"   ✅ Total inventory value: ${metrics.total_value:.2f}")
        else:
            logger.warning(f"   ⚠️ Operational phase: {operational_response.message}")
            context.operational_plan = OperationalDecision({}, 0, is_feasible=False)
        
        logger.info(f"   📝 Tool calls: {operational_response.tool_calls}")
        
        # Pipeline complete
        metrics.total_time = time.time() - start_time
        metrics.total_cost = (
            context.strategic_plan.total_fixed_cost + 
            context.tactical_plan.total_transport_cost
        )
        metrics.success = True
        
        # Generate summary
        reasoning_summary = "\n\n".join(reasoning_parts)
        if not reasoning_summary:
            reasoning_summary = self._generate_summary(context, metrics)
        
        self.last_result = CollaborationResult(
            strategic_decision=context.strategic_plan,
            tactical_decision=context.tactical_plan,
            operational_decision=context.operational_plan,
            metrics=metrics,
            reasoning_summary=reasoning_summary,
            agent_logs=agent_logs
        )
        
        self.execution_history.append(self.last_result)
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("✅ Pipeline Completed Successfully!")
        logger.info(f"   Total Time: {metrics.total_time:.2f}s")
        logger.info(f"   Total Cost: ${metrics.total_cost:.2f}")
        logger.info(f"   Total Value: ${metrics.total_value:.2f}")
        logger.info(f"   Tool Calls: {metrics.tool_calls_count}")
        logger.info("=" * 60)
        
        return context
    
    def _generate_summary(self, context: OperationsContext, metrics: PipelineMetrics) -> str:
        """Generate a summary when Claude reasoning is not used"""
        return f"""## Pipeline Execution Summary

**Strategic Phase:**
- Opened {len(context.strategic_plan.open_facilities_indices)} facilities
- Facilities: {context.strategic_plan.open_facilities_indices}
- Fixed cost: ${context.strategic_plan.total_fixed_cost:.2f}

**Tactical Phase:**
- Optimized transportation flow
- Transport cost: ${context.tactical_plan.total_transport_cost:.2f}
- Facilities served: {list(context.tactical_plan.facility_inbound_volumes.keys())}

**Operational Phase:**
- Inventory value: ${context.operational_plan.total_inventory_value:.2f}
- Facilities optimized: {len(context.operational_plan.facility_inventory_plans)}

**Performance:**
- Total execution time: {metrics.total_time:.2f}s
- Tool calls: {metrics.tool_calls_count}
"""
    
    def get_collaboration_result(self) -> Optional[CollaborationResult]:
        """Get the last collaboration result"""
        return self.last_result
    
    def get_reasoning(self) -> str:
        """Get combined reasoning from all agents"""
        if not self.last_result:
            return ""
        return self.last_result.reasoning_summary
    
    def compare_with_baseline(self, context: OperationsContext) -> Dict[str, Any]:
        """
        Compare Claude-enhanced results with baseline (non-reasoning) results.
        """
        # Run with reasoning
        context_enhanced = OperationsContext(
            potential_facilities_costs=context.potential_facilities_costs.copy(),
            potential_facilities_capacities=context.potential_facilities_capacities.copy(),
            customer_demands=context.customer_demands.copy(),
            transport_costs_full=context.transport_costs_full.copy(),
            item_values=context.item_values.copy(),
            item_weights=context.item_weights.copy()
        )
        self.run_pipeline(context_enhanced, use_reasoning=True)
        enhanced_result = self.last_result
        
        # Run without reasoning
        context_baseline = OperationsContext(
            potential_facilities_costs=context.potential_facilities_costs.copy(),
            potential_facilities_capacities=context.potential_facilities_capacities.copy(),
            customer_demands=context.customer_demands.copy(),
            transport_costs_full=context.transport_costs_full.copy(),
            item_values=context.item_values.copy(),
            item_weights=context.item_weights.copy()
        )
        self.run_pipeline(context_baseline, use_reasoning=False)
        baseline_result = self.last_result
        
        return {
            "enhanced": {
                "total_cost": enhanced_result.metrics.total_cost,
                "total_value": enhanced_result.metrics.total_value,
                "time": enhanced_result.metrics.total_time,
                "tool_calls": enhanced_result.metrics.tool_calls_count
            },
            "baseline": {
                "total_cost": baseline_result.metrics.total_cost,
                "total_value": baseline_result.metrics.total_value,
                "time": baseline_result.metrics.total_time,
                "tool_calls": baseline_result.metrics.tool_calls_count
            },
            "improvement": {
                "cost_reduction": baseline_result.metrics.total_cost - enhanced_result.metrics.total_cost,
                "value_increase": enhanced_result.metrics.total_value - baseline_result.metrics.total_value,
                "time_difference": enhanced_result.metrics.total_time - baseline_result.metrics.total_time
            }
        }


if __name__ == "__main__":
    # Demo the intelligent orchestrator
    print("🤖 Intelligent Multi-Agent Orchestrator Demo")
    print("=" * 60)
    
    # Create test context
    context = OperationsContext(
        potential_facilities_costs=np.array([100.0, 100.0, 100.0]),
        potential_facilities_capacities=np.array([200.0, 200.0, 200.0]),
        customer_demands=np.array([50.0, 50.0, 50.0, 50.0, 50.0]),
        transport_costs_full=np.array([
            [2, 2, 10, 10, 10],
            [10, 10, 2, 2, 10],
            [10, 10, 10, 10, 2]
        ], dtype=float),
        item_values=[60.0, 100.0, 120.0, 80.0, 90.0, 50.0, 70.0, 110.0],
        item_weights=[10, 20, 30, 15, 25, 8, 12, 22]
    )
    
    # Run orchestrator (without API key for demo)
    orchestrator = IntelligentOrchestrator(use_reasoning=False)
    result_context = orchestrator.run_pipeline(context)
    
    # Show results
    print("\n📊 Final Results:")
    print(f"   Open Facilities: {result_context.strategic_plan.open_facilities_indices}")
    print(f"   Total Fixed Cost: ${result_context.strategic_plan.total_fixed_cost:.2f}")
    print(f"   Total Transport Cost: ${result_context.tactical_plan.total_transport_cost:.2f}")
    print(f"   Total Inventory Value: ${result_context.operational_plan.total_inventory_value:.2f}")
    
    # Show reasoning
    print("\n📝 Reasoning Summary:")
    print(orchestrator.get_reasoning())

