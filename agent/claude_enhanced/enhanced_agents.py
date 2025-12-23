"""
Claude-Enhanced Operations Agents

Implements the three-tier agent architecture with Claude agent skills:
- ClaudeStrategicAgent (MIP + AI reasoning)
- ClaudeTacticalAgent (LP + AI planning)
- ClaudeOperationalAgent (DP + AI optimization)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import json
import os

from .agent_skills import (
    AgentSkillsToolkit,
    StrategicAgentSkills,
    TacticalAgentSkills,
    OperationalAgentSkills,
    ToolResult
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.orchestrator.definitions import (
    StrategicDecision, TacticalDecision, OperationalDecision
)

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standard response from an enhanced agent"""
    success: bool
    decision: Any
    metrics: Dict[str, float]
    message: str
    reasoning: str = ""
    tool_calls: List[str] = None
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class ClaudeEnhancedAgent:
    """Base class for Claude-enhanced agents"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.skills: AgentSkillsToolkit = None
        self.last_reasoning = ""
        self.execution_log = []
        
    def _log_execution(self, action: str, result: Any):
        """Log agent execution for debugging and learning"""
        self.execution_log.append({
            "agent": self.name,
            "action": action,
            "result": result,
            "success": result.success if hasattr(result, 'success') else True
        })
    
    def get_reasoning(self) -> str:
        """Get the last reasoning from the agent"""
        return self.last_reasoning


class ClaudeStrategicAgent(ClaudeEnhancedAgent):
    """
    Strategic Agent: Decides on Facility Locations using MIP + Claude reasoning.
    
    Enhanced with Claude agent skills:
    - Analyzes cost structures before optimization
    - Evaluates multiple scenarios
    - Assesses strategic risks
    - Provides reasoning for decisions
    """
    
    def __init__(self, name: str = "Claude_Strategic_Agent", api_key: Optional[str] = None):
        super().__init__(name, api_key)
        self.skills = StrategicAgentSkills(api_key=self.api_key)
        
    def act(self, fixed_costs: np.ndarray, transport_costs: np.ndarray,
            demand: np.ndarray, capacity: np.ndarray,
            use_reasoning: bool = True) -> AgentResponse:
        """
        Decide which facilities to open using MIP optimization + Claude reasoning.
        
        Args:
            fixed_costs: Fixed cost for each potential facility
            transport_costs: Transport cost matrix [facility, customer]
            demand: Customer demands
            capacity: Facility capacities
            use_reasoning: Whether to use Claude for enhanced reasoning
            
        Returns:
            AgentResponse with facility location decision
        """
        logger.info(f"[{self.name}] Starting strategic decision making...")
        
        # Prepare context
        context = {
            "fixed_costs": fixed_costs.tolist(),
            "transport_costs": transport_costs.tolist(),
            "demand": demand.tolist(),
            "capacity": capacity.tolist()
        }
        
        tool_calls = []
        reasoning_text = ""
        
        if use_reasoning and self.skills.client:
            # Use Claude for enhanced reasoning
            prompt = """Analyze this facility location problem and find the optimal solution.

Consider:
1. The trade-off between fixed facility costs and transportation costs
2. Capacity constraints and demand requirements
3. Strategic risks of the recommended configuration

Please use the available tools systematically to find the optimal facility locations."""

            reasoning_text, tool_results = self.skills.reason_with_tools(prompt, context)
            tool_calls = [tr.tool_name for tr in tool_results]
            
            # Extract optimization result from tool results
            for tr in tool_results:
                if tr.tool_name == "optimize_facility_selection" and tr.success:
                    if tr.result.get("success"):
                        open_facilities = tr.result["open_facilities"]
                        total_cost = tr.result["total_cost"]
                        
                        self.last_reasoning = reasoning_text
                        self._log_execution("strategic_decision", tr)
                        
                        return AgentResponse(
                            success=True,
                            decision={
                                'open_facilities': open_facilities,
                                'allocation_plan': None  # To be filled by tactical agent
                            },
                            metrics={'total_cost': total_cost},
                            message=f"Optimal locations found: {open_facilities}",
                            reasoning=reasoning_text,
                            tool_calls=tool_calls
                        )
        
        # Fallback: Direct optimization
        logger.info(f"[{self.name}] Using direct MIP optimization...")
        
        opt_result = self.skills.execute_tool("optimize_facility_selection", {
            "fixed_costs": context["fixed_costs"],
            "transport_costs": context["transport_costs"],
            "demand": context["demand"],
            "capacity": context["capacity"]
        })
        
        tool_calls.append(opt_result.tool_name)
        
        if opt_result.success and opt_result.result.get("success"):
            open_facilities = opt_result.result["open_facilities"]
            total_cost = opt_result.result["total_cost"]
            
            # Run risk assessment
            risk_result = self.skills.execute_tool("assess_strategic_risk", {
                "open_facilities": open_facilities,
                "capacity": context["capacity"],
                "demand": context["demand"],
                "demand_uncertainty": 0.2
            })
            tool_calls.append(risk_result.tool_name)
            
            risk_info = risk_result.result if risk_result.success else {}
            
            self.last_reasoning = f"Direct optimization: Selected facilities {open_facilities}. " + \
                                  f"Risk level: {risk_info.get('risk_level', 'unknown')}"
            
            return AgentResponse(
                success=True,
                decision={
                    'open_facilities': open_facilities,
                    'allocation_plan': None,
                    'risk_assessment': risk_info
                },
                metrics={
                    'total_cost': total_cost,
                    'num_facilities': len(open_facilities),
                    'capacity_utilization': risk_info.get('capacity_utilization', 0)
                },
                message=f"Optimal locations found: {open_facilities}",
                reasoning=self.last_reasoning,
                tool_calls=tool_calls
            )
        else:
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message=f"Optimization failed: {opt_result.metadata.get('error', 'Unknown error')}",
                reasoning="Strategic optimization failed",
                tool_calls=tool_calls
            )


class ClaudeTacticalAgent(ClaudeEnhancedAgent):
    """
    Tactical Agent: Decides on Logistics/Transportation using LP + Claude planning.
    
    Enhanced with Claude agent skills:
    - Analyzes supply-demand balance
    - Optimizes transportation flow
    - Evaluates route efficiency
    - Provides planning recommendations
    """
    
    def __init__(self, name: str = "Claude_Tactical_Agent", api_key: Optional[str] = None):
        super().__init__(name, api_key)
        self.skills = TacticalAgentSkills(api_key=self.api_key)
        
    def act(self, supply: np.ndarray, demand: np.ndarray, costs: np.ndarray,
            open_facility_indices: List[int] = None,
            use_reasoning: bool = True) -> AgentResponse:
        """
        Decide on transport quantities using LP optimization + Claude planning.
        
        Args:
            supply: Supply at each source (open facilities)
            demand: Demand at each destination (customers)
            costs: Transport cost matrix [source, destination]
            open_facility_indices: Original indices of open facilities (for constraint propagation)
            use_reasoning: Whether to use Claude for enhanced reasoning
            
        Returns:
            AgentResponse with transportation decision
        """
        logger.info(f"[{self.name}] Starting tactical planning...")
        
        context = {
            "supply": supply.tolist(),
            "demand": demand.tolist(),
            "costs": costs.tolist(),
            "open_facility_indices": open_facility_indices or list(range(len(supply)))
        }
        
        tool_calls = []
        reasoning_text = ""
        
        if use_reasoning and self.skills.client:
            prompt = """Analyze this transportation problem and find the optimal flow.

Consider:
1. The balance between supply and demand
2. Cost efficiency of different routes
3. Requirements for downstream operational planning

Please use the available tools to optimize the transportation flow."""

            reasoning_text, tool_results = self.skills.reason_with_tools(prompt, context)
            tool_calls = [tr.tool_name for tr in tool_results]
            
            for tr in tool_results:
                if tr.tool_name == "optimize_transportation" and tr.success:
                    if tr.result.get("success"):
                        allocation = np.array(tr.result["allocation"])
                        total_cost = tr.result["total_cost"]
                        
                        # Calculate facility requirements
                        req_result = self.skills.execute_tool("calculate_facility_requirements", {
                            "allocation": tr.result["allocation"],
                            "facility_indices": context["open_facility_indices"]
                        })
                        
                        facility_volumes = req_result.result.get("facility_requirements", {}) if req_result.success else {}
                        
                        self.last_reasoning = reasoning_text
                        
                        return AgentResponse(
                            success=True,
                            decision={
                                'transport_matrix': allocation,
                                'facility_inbound_volumes': facility_volumes
                            },
                            metrics={
                                'total_cost': total_cost,
                                'active_routes': tr.result.get("active_routes", 0)
                            },
                            message="Optimal transport flow found.",
                            reasoning=reasoning_text,
                            tool_calls=tool_calls
                        )
        
        # Fallback: Direct optimization
        logger.info(f"[{self.name}] Using direct LP optimization...")
        
        # First analyze supply-demand
        analysis = self.skills.execute_tool("analyze_supply_demand", {
            "supply": context["supply"],
            "demand": context["demand"]
        })
        tool_calls.append(analysis.tool_name)
        
        if analysis.success and not analysis.result.get("is_feasible"):
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message="Infeasible: Supply cannot meet demand",
                reasoning=analysis.result.get("reasoning", ""),
                tool_calls=tool_calls
            )
        
        # Optimize transportation
        opt_result = self.skills.execute_tool("optimize_transportation", {
            "supply": context["supply"],
            "demand": context["demand"],
            "costs": context["costs"]
        })
        tool_calls.append(opt_result.tool_name)
        
        if opt_result.success and opt_result.result.get("success"):
            allocation = np.array(opt_result.result["allocation"])
            total_cost = opt_result.result["total_cost"]
            
            # Calculate facility requirements
            req_result = self.skills.execute_tool("calculate_facility_requirements", {
                "allocation": opt_result.result["allocation"],
                "facility_indices": context["open_facility_indices"]
            })
            tool_calls.append(req_result.tool_name)
            
            facility_volumes = req_result.result.get("facility_requirements", {}) if req_result.success else {}
            
            # Evaluate efficiency
            eff_result = self.skills.execute_tool("evaluate_route_efficiency", {
                "allocation": opt_result.result["allocation"],
                "costs": context["costs"]
            })
            tool_calls.append(eff_result.tool_name)
            
            self.last_reasoning = f"LP optimization successful. {eff_result.result.get('reasoning', '')}" if eff_result.success else "LP optimization successful."
            
            return AgentResponse(
                success=True,
                decision={
                    'transport_matrix': allocation,
                    'facility_inbound_volumes': facility_volumes
                },
                metrics={
                    'total_cost': total_cost,
                    'active_routes': opt_result.result.get("active_routes", 0),
                    'avg_cost_per_unit': eff_result.result.get("average_cost_per_unit", 0) if eff_result.success else 0
                },
                message="Optimal transport flow found.",
                reasoning=self.last_reasoning,
                tool_calls=tool_calls
            )
        else:
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message=f"Optimization failed: {opt_result.metadata.get('error', 'Unknown')}",
                reasoning="Tactical optimization failed",
                tool_calls=tool_calls
            )


class ClaudeOperationalAgent(ClaudeEnhancedAgent):
    """
    Operational Agent: Decides on Inventory Mix using DP + Claude optimization.
    
    Enhanced with Claude agent skills:
    - Analyzes item value-weight trade-offs
    - Optimizes inventory mix per facility
    - Evaluates plan quality
    - Provides execution recommendations
    """
    
    def __init__(self, name: str = "Claude_Operational_Agent", api_key: Optional[str] = None):
        super().__init__(name, api_key)
        self.skills = OperationalAgentSkills(api_key=self.api_key)
        
    def act(self, mode: str, use_reasoning: bool = True, **kwargs) -> AgentResponse:
        """
        Execute operational optimization based on mode.
        
        Args:
            mode: 'inventory' for knapsack, 'path' for shortest path
            use_reasoning: Whether to use Claude for enhanced reasoning
            **kwargs: Mode-specific parameters
            
        Returns:
            AgentResponse with operational decision
        """
        if mode == 'inventory':
            return self.optimize_inventory(
                kwargs.get('values', []),
                kwargs.get('weights', []),
                kwargs.get('capacity', 0),
                use_reasoning=use_reasoning
            )
        elif mode == 'multi_facility':
            return self.optimize_multi_facility(
                kwargs.get('facility_capacities', {}),
                kwargs.get('values', []),
                kwargs.get('weights', []),
                use_reasoning=use_reasoning
            )
        else:
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message=f"Unknown mode: {mode}",
                reasoning=""
            )
    
    def optimize_inventory(self, values: List[float], weights: List[int], capacity: int,
                           use_reasoning: bool = True) -> AgentResponse:
        """
        Optimize inventory mix for a single facility.
        """
        logger.info(f"[{self.name}] Starting inventory optimization...")
        
        context = {
            "values": values,
            "weights": weights,
            "capacity": capacity
        }
        
        tool_calls = []
        
        if use_reasoning and self.skills.client:
            prompt = """Optimize the inventory mix for maximum value within capacity.

Consider:
1. Value-weight trade-offs of different items
2. Capacity constraints
3. Quality of the final solution

Use the available tools to find the optimal inventory mix."""

            reasoning_text, tool_results = self.skills.reason_with_tools(prompt, context)
            tool_calls = [tr.tool_name for tr in tool_results]
            
            for tr in tool_results:
                if tr.tool_name == "optimize_inventory_mix" and tr.success:
                    if tr.result.get("success"):
                        self.last_reasoning = reasoning_text
                        
                        return AgentResponse(
                            success=True,
                            decision={'selected_items': tr.result["selected_items"]},
                            metrics={
                                'total_value': tr.result["total_value"],
                                'total_weight': tr.result["total_weight"],
                                'capacity_used': tr.result["capacity_used"]
                            },
                            message=f"Optimal mix: {len(tr.result['selected_items'])} items",
                            reasoning=reasoning_text,
                            tool_calls=tool_calls
                        )
        
        # Fallback: Direct DP
        logger.info(f"[{self.name}] Using direct DP optimization...")
        
        # Analyze items first
        analysis = self.skills.execute_tool("analyze_inventory_items", {
            "values": values,
            "weights": weights
        })
        tool_calls.append(analysis.tool_name)
        
        # Run optimization
        opt_result = self.skills.execute_tool("optimize_inventory_mix", {
            "values": values,
            "weights": weights,
            "capacity": capacity
        })
        tool_calls.append(opt_result.tool_name)
        
        if opt_result.success and opt_result.result.get("success"):
            # Evaluate the plan
            eval_result = self.skills.execute_tool("evaluate_inventory_plan", {
                "selected_items": opt_result.result["selected_items"],
                "values": values,
                "weights": weights,
                "capacity": capacity
            })
            tool_calls.append(eval_result.tool_name)
            
            rating = eval_result.result.get("rating", "unknown") if eval_result.success else "unknown"
            
            self.last_reasoning = f"DP optimization: Selected {len(opt_result.result['selected_items'])} items. Rating: {rating}"
            
            return AgentResponse(
                success=True,
                decision={'selected_items': opt_result.result["selected_items"]},
                metrics={
                    'total_value': opt_result.result["total_value"],
                    'total_weight': opt_result.result["total_weight"],
                    'capacity_used': opt_result.result["capacity_used"]
                },
                message=f"Optimal inventory mix found. Rating: {rating}",
                reasoning=self.last_reasoning,
                tool_calls=tool_calls
            )
        else:
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message=f"Optimization failed: {opt_result.metadata.get('error', 'Unknown')}",
                reasoning="Operational optimization failed",
                tool_calls=tool_calls
            )
    
    def optimize_multi_facility(self, facility_capacities: Dict[int, float],
                                values: List[float], weights: List[int],
                                use_reasoning: bool = True) -> AgentResponse:
        """
        Optimize inventory across multiple facilities.
        """
        logger.info(f"[{self.name}] Optimizing inventory for {len(facility_capacities)} facilities...")
        
        context = {
            "facility_capacities": {str(k): v for k, v in facility_capacities.items()},
            "values": values,
            "weights": weights
        }
        
        tool_calls = []
        
        # Use multi-facility optimization
        opt_result = self.skills.execute_tool("multi_facility_optimization", {
            "facility_capacities": context["facility_capacities"],
            "values": values,
            "weights": weights
        })
        tool_calls.append(opt_result.tool_name)
        
        if opt_result.success:
            facility_plans = opt_result.result.get("facility_plans", {})
            total_value = opt_result.result.get("total_value_all_facilities", 0)
            
            # Convert back to int keys
            facility_inventory_plans = {}
            for k, v in facility_plans.items():
                facility_inventory_plans[int(k)] = v.get("selected_items", [])
            
            self.last_reasoning = opt_result.result.get("reasoning", "Multi-facility optimization complete")
            
            return AgentResponse(
                success=True,
                decision={'facility_inventory_plans': facility_inventory_plans},
                metrics={
                    'total_value': total_value,
                    'facilities_optimized': len(facility_plans)
                },
                message=f"Optimized {len(facility_plans)} facilities",
                reasoning=self.last_reasoning,
                tool_calls=tool_calls
            )
        else:
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message=f"Multi-facility optimization failed",
                reasoning="",
                tool_calls=tool_calls
            )


if __name__ == "__main__":
    # Test the enhanced agents
    print("🤖 Testing Claude-Enhanced Agents")
    print("=" * 60)
    
    # Test Strategic Agent
    print("\n1️⃣ Testing Strategic Agent...")
    strategic = ClaudeStrategicAgent()
    
    fixed_costs = np.array([100, 100, 100])
    transport_costs = np.array([
        [2, 2, 10, 10, 10],
        [10, 10, 2, 2, 10],
        [10, 10, 10, 10, 2]
    ])
    demand = np.array([50, 50, 50, 50, 50])
    capacity = np.array([200, 200, 200])
    
    result = strategic.act(fixed_costs, transport_costs, demand, capacity, use_reasoning=False)
    print(f"   Success: {result.success}")
    print(f"   Decision: {result.decision}")
    print(f"   Reasoning: {result.reasoning[:100]}..." if result.reasoning else "")
    
    # Test Tactical Agent
    print("\n2️⃣ Testing Tactical Agent...")
    tactical = ClaudeTacticalAgent()
    
    if result.success:
        open_indices = result.decision['open_facilities']
        active_supply = capacity[open_indices]
        active_costs = transport_costs[open_indices]
        
        tact_result = tactical.act(active_supply, demand, active_costs, open_indices, use_reasoning=False)
        print(f"   Success: {tact_result.success}")
        print(f"   Total Cost: {tact_result.metrics.get('total_cost', 'N/A')}")
    
    # Test Operational Agent
    print("\n3️⃣ Testing Operational Agent...")
    operational = ClaudeOperationalAgent()
    
    values = [60, 100, 120, 80, 90]
    weights = [10, 20, 30, 15, 25]
    capacity = 50
    
    op_result = operational.act('inventory', values=values, weights=weights, capacity=capacity, use_reasoning=False)
    print(f"   Success: {op_result.success}")
    print(f"   Selected Items: {op_result.decision}")
    print(f"   Total Value: {op_result.metrics.get('total_value', 'N/A')}")
    
    print("\n✅ All agents tested!")

