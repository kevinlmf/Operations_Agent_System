"""
Agent Skills Toolkit

Defines specialized tools for each agent tier in the operations system.
These tools encapsulate OR domain knowledge and can be called by Claude.
"""

import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import logging

logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, bool):
        return obj
    else:
        return obj


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    success: bool
    result: Any
    reasoning: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentSkillsToolkit:
    """
    Base toolkit providing common functionality for all agent tiers.
    Each agent tier extends this with specialized tools.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_claude: bool = True):
        self.api_key = api_key
        self.use_claude = use_claude and api_key is not None
        self.client = None
        self._init_client()
        
        # Tool execution history for learning
        self.execution_history = []
        
    def _init_client(self):
        """Initialize Claude client if API key available"""
        if self.use_claude and self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Claude client: {e}")
                self.client = None
                
    def execute_tool(self, tool_name: str, tool_input: Dict) -> ToolResult:
        """Execute a tool - to be overridden by subclasses"""
        raise NotImplementedError
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions in Claude API format - to be overridden"""
        raise NotImplementedError
    
    def reason_with_tools(self, prompt: str, context: Dict) -> Tuple[str, List[ToolResult]]:
        """
        Use Claude to reason about a problem and execute tools.
        Returns reasoning text and list of tool results.
        """
        if not self.client:
            # Fallback: execute tools based on simple heuristics
            return self._fallback_reasoning(prompt, context)
            
        system_prompt = self._get_system_prompt()
        tools = self.get_tool_definitions()
        
        messages = [{"role": "user", "content": self._format_prompt(prompt, context)}]
        tool_results = []
        reasoning_text = ""
        
        max_iterations = 5
        for _ in range(max_iterations):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=system_prompt,
                    tools=tools,
                    messages=messages
                )
                
                # Extract text blocks
                for block in response.content:
                    if hasattr(block, 'text'):
                        reasoning_text += block.text + "\n"
                
                # Check for tool uses
                tool_uses = [block for block in response.content if block.type == "tool_use"]
                
                if not tool_uses:
                    break
                    
                # Execute tools
                tool_outputs = []
                for tool_use in tool_uses:
                    result = self.execute_tool(tool_use.name, tool_use.input)
                    tool_results.append(result)
                    tool_outputs.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result.result if result.success else {"error": result.metadata.get("error", "Unknown")})
                    })
                
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_outputs})
                
            except Exception as e:
                logger.error(f"Claude reasoning failed: {e}")
                return self._fallback_reasoning(prompt, context)
        
        return reasoning_text, tool_results
    
    def _get_system_prompt(self) -> str:
        """Get system prompt - to be overridden by subclasses"""
        return "You are an expert operations research agent."
    
    def _format_prompt(self, prompt: str, context: Dict) -> str:
        """Format user prompt with context"""
        context_str = json.dumps(context, indent=2, default=str)
        return f"{prompt}\n\nContext:\n{context_str}"
    
    def _fallback_reasoning(self, prompt: str, context: Dict) -> Tuple[str, List[ToolResult]]:
        """Fallback when Claude is not available"""
        return "Fallback mode - using heuristic decision making", []


class StrategicAgentSkills(AgentSkillsToolkit):
    """
    Skills for Strategic Agent (MIP-based facility location decisions).
    
    Tools:
    - analyze_facility_costs: Analyze fixed vs variable costs
    - evaluate_location_scenarios: Evaluate different facility combinations
    - optimize_facility_selection: Run MIP optimization
    - assess_strategic_risk: Risk assessment for facility decisions
    """
    
    def get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "analyze_facility_costs",
                "description": "Analyze facility costs to understand the trade-offs between fixed costs and transportation costs. Use this before optimization to understand the cost structure.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fixed_costs": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Fixed cost for each potential facility"
                        },
                        "transport_costs": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Transport cost matrix [facility][customer]"
                        },
                        "demand": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Customer demands"
                        }
                    },
                    "required": ["fixed_costs", "transport_costs", "demand"]
                }
            },
            {
                "name": "evaluate_location_scenarios",
                "description": "Evaluate different facility location scenarios without full optimization. Useful for quick what-if analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario_facilities": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of facility indices to open in this scenario"
                        },
                        "fixed_costs": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Fixed costs array"
                        },
                        "transport_costs": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Transport cost matrix"
                        },
                        "demand": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Customer demands"
                        },
                        "capacity": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Facility capacities"
                        }
                    },
                    "required": ["scenario_facilities", "fixed_costs", "transport_costs", "demand", "capacity"]
                }
            },
            {
                "name": "optimize_facility_selection",
                "description": "Run Mixed Integer Programming to find optimal facility locations. This is the main optimization tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fixed_costs": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Fixed cost for each facility"
                        },
                        "transport_costs": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Transport cost matrix"
                        },
                        "demand": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Customer demands"
                        },
                        "capacity": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Facility capacities"
                        }
                    },
                    "required": ["fixed_costs", "transport_costs", "demand", "capacity"]
                }
            },
            {
                "name": "assess_strategic_risk",
                "description": "Assess strategic risks of facility decisions including demand uncertainty and capacity utilization.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "open_facilities": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Indices of open facilities"
                        },
                        "capacity": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Facility capacities"
                        },
                        "demand": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Customer demands"
                        },
                        "demand_uncertainty": {
                            "type": "number",
                            "description": "Coefficient of variation for demand uncertainty"
                        }
                    },
                    "required": ["open_facilities", "capacity", "demand"]
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, tool_input: Dict) -> ToolResult:
        try:
            if tool_name == "analyze_facility_costs":
                result = self._analyze_facility_costs(**tool_input)
            elif tool_name == "evaluate_location_scenarios":
                result = self._evaluate_location_scenarios(**tool_input)
            elif tool_name == "optimize_facility_selection":
                result = self._optimize_facility_selection(**tool_input)
            elif tool_name == "assess_strategic_risk":
                result = self._assess_strategic_risk(**tool_input)
            else:
                return ToolResult(tool_name, False, None, metadata={"error": f"Unknown tool: {tool_name}"})
            
            # Convert numpy types to native Python types for JSON serialization
            result = convert_to_serializable(result)
            return ToolResult(tool_name, True, result, reasoning=result.get("reasoning", ""))
        except Exception as e:
            return ToolResult(tool_name, False, None, metadata={"error": str(e)})
    
    def _analyze_facility_costs(self, fixed_costs: List[float], transport_costs: List[List[float]], 
                                 demand: List[float]) -> Dict:
        """Analyze cost structure"""
        fixed_costs = np.array(fixed_costs)
        transport_costs = np.array(transport_costs)
        demand = np.array(demand)
        
        total_demand = np.sum(demand)
        
        # Calculate potential transport cost for each facility if it serves all demand
        potential_transport = []
        for i, costs in enumerate(transport_costs):
            weighted_cost = np.sum(np.array(costs) * demand)
            potential_transport.append(weighted_cost)
        
        # Cost efficiency score (lower is better)
        efficiency_scores = []
        for i in range(len(fixed_costs)):
            total_cost = fixed_costs[i] + potential_transport[i]
            efficiency = total_cost / total_demand if total_demand > 0 else float('inf')
            efficiency_scores.append(round(efficiency, 3))
        
        # Recommend facilities
        ranked = sorted(range(len(efficiency_scores)), key=lambda x: efficiency_scores[x])
        
        return {
            "total_demand": round(total_demand, 2),
            "fixed_costs_summary": {
                "min": round(float(np.min(fixed_costs)), 2),
                "max": round(float(np.max(fixed_costs)), 2),
                "mean": round(float(np.mean(fixed_costs)), 2)
            },
            "transport_costs_per_facility": [round(x, 2) for x in potential_transport],
            "efficiency_scores": efficiency_scores,
            "recommended_ranking": ranked[:3],
            "reasoning": f"Facilities ranked by cost efficiency. Top 3: {ranked[:3]}"
        }
    
    def _evaluate_location_scenarios(self, scenario_facilities: List[int], fixed_costs: List[float],
                                     transport_costs: List[List[float]], demand: List[float],
                                     capacity: List[float]) -> Dict:
        """Evaluate a specific facility scenario"""
        fixed_costs = np.array(fixed_costs)
        transport_costs = np.array(transport_costs)
        demand = np.array(demand)
        capacity = np.array(capacity)
        
        # Check feasibility
        total_capacity = np.sum(capacity[scenario_facilities])
        total_demand = np.sum(demand)
        is_feasible = total_capacity >= total_demand
        
        # Calculate fixed cost
        total_fixed = np.sum(fixed_costs[scenario_facilities])
        
        # Estimate transport cost (simplified: assign each customer to nearest open facility)
        total_transport = 0
        assignments = []
        for j, d in enumerate(demand):
            min_cost = float('inf')
            best_facility = -1
            for i in scenario_facilities:
                if transport_costs[i][j] < min_cost:
                    min_cost = transport_costs[i][j]
                    best_facility = i
            total_transport += min_cost * d
            assignments.append({"customer": j, "facility": best_facility, "cost": round(min_cost * d, 2)})
        
        return {
            "scenario": scenario_facilities,
            "is_feasible": is_feasible,
            "total_capacity": round(total_capacity, 2),
            "total_demand": round(total_demand, 2),
            "capacity_utilization": round(total_demand / total_capacity, 3) if total_capacity > 0 else 0,
            "total_fixed_cost": round(total_fixed, 2),
            "estimated_transport_cost": round(total_transport, 2),
            "total_cost": round(total_fixed + total_transport, 2),
            "assignments": assignments[:5],  # First 5 for brevity
            "reasoning": f"Scenario {'feasible' if is_feasible else 'infeasible'}. Total cost: {total_fixed + total_transport:.2f}"
        }
    
    def _optimize_facility_selection(self, fixed_costs: List[float], transport_costs: List[List[float]],
                                     demand: List[float], capacity: List[float]) -> Dict:
        """Run MIP optimization"""
        fixed_costs = np.array(fixed_costs)
        transport_costs = np.array(transport_costs)
        demand = np.array(demand)
        capacity = np.array(capacity)
        
        m = len(fixed_costs)
        n = len(demand)
        num_vars = m + m * n
        
        # Objective
        c = np.concatenate([fixed_costs, transport_costs.flatten()])
        
        # Demand constraints
        A_eq = np.zeros((n, num_vars))
        b_eq = demand.copy()
        for j in range(n):
            for i in range(m):
                A_eq[j, m + i*n + j] = 1
        
        # Capacity constraints
        A_ub = np.zeros((m, num_vars))
        b_ub = np.zeros(m)
        for i in range(m):
            A_ub[i, i] = -capacity[i]
            for j in range(n):
                A_ub[i, m + i*n + j] = 1
        
        A = np.vstack([A_eq, A_ub])
        lb = np.concatenate([b_eq, np.full(m, -np.inf)])
        ub = np.concatenate([b_eq, b_ub])
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.zeros(num_vars)
        integrality[:m] = 1
        
        lb_vars = np.zeros(num_vars)
        ub_vars = np.concatenate([np.ones(m), np.full(m*n, np.inf)])
        bounds = Bounds(lb_vars, ub_vars)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.success:
                y = res.x[:m]
                x = res.x[m:].reshape(m, n)
                open_facilities = [i for i, val in enumerate(y) if val > 0.5]
                
                return {
                    "success": True,
                    "open_facilities": open_facilities,
                    "total_cost": round(res.fun, 2),
                    "allocation_summary": {
                        "num_open": len(open_facilities),
                        "total_flow": round(np.sum(x), 2)
                    },
                    "reasoning": f"Optimal solution found. Open facilities: {open_facilities}, Total cost: {res.fun:.2f}"
                }
            else:
                return {
                    "success": False,
                    "error": res.message,
                    "reasoning": f"Optimization failed: {res.message}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "reasoning": f"Optimization error: {str(e)}"
            }
    
    def _assess_strategic_risk(self, open_facilities: List[int], capacity: List[float],
                               demand: List[float], demand_uncertainty: float = 0.2) -> Dict:
        """Assess strategic risks"""
        capacity = np.array(capacity)
        demand = np.array(demand)
        
        total_capacity = np.sum(capacity[open_facilities])
        total_demand = np.sum(demand)
        
        # Capacity buffer
        capacity_buffer = (total_capacity - total_demand) / total_demand if total_demand > 0 else 0
        
        # Probability of demand exceeding capacity (normal approximation)
        from scipy import stats
        demand_std = total_demand * demand_uncertainty
        if demand_std > 0:
            z_score = (total_capacity - total_demand) / demand_std
            prob_exceed = 1 - stats.norm.cdf(z_score)
        else:
            prob_exceed = 0 if total_capacity >= total_demand else 1
        
        # Risk level
        if prob_exceed < 0.05:
            risk_level = "low"
        elif prob_exceed < 0.15:
            risk_level = "medium"
        elif prob_exceed < 0.30:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Single point of failure risk
        if len(open_facilities) == 1:
            spof_risk = "high"
        elif len(open_facilities) == 2:
            spof_risk = "medium"
        else:
            spof_risk = "low"
        
        return {
            "capacity_utilization": round(total_demand / total_capacity, 3) if total_capacity > 0 else float('inf'),
            "capacity_buffer": round(capacity_buffer, 3),
            "probability_capacity_exceeded": round(prob_exceed, 4),
            "risk_level": risk_level,
            "single_point_of_failure_risk": spof_risk,
            "num_facilities": len(open_facilities),
            "recommendations": self._get_risk_recommendations(risk_level, spof_risk, capacity_buffer),
            "reasoning": f"Risk level: {risk_level}. Capacity buffer: {capacity_buffer*100:.1f}%"
        }
    
    def _get_risk_recommendations(self, risk_level: str, spof_risk: str, capacity_buffer: float) -> List[str]:
        recommendations = []
        if risk_level in ["high", "critical"]:
            recommendations.append("Consider opening additional facilities or expanding capacity")
        if spof_risk == "high":
            recommendations.append("Single facility creates high dependency risk - consider backup facility")
        if capacity_buffer < 0.1:
            recommendations.append("Low capacity buffer - demand spikes may cause stockouts")
        if not recommendations:
            recommendations.append("Current configuration has acceptable risk profile")
        return recommendations
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Strategic Operations Agent specializing in facility location decisions.

Your role:
1. Analyze facility cost structures to understand trade-offs
2. Evaluate different facility scenarios
3. Use MIP optimization to find optimal facility locations
4. Assess strategic risks of decisions

Always think step-by-step:
1. First, analyze the cost structure
2. Then, evaluate promising scenarios
3. Run optimization to find the best solution
4. Finally, assess risks of the recommended solution

Use the available tools systematically and provide clear reasoning for your decisions."""

    def _fallback_reasoning(self, prompt: str, context: Dict) -> Tuple[str, List[ToolResult]]:
        """Fallback logic for strategic decisions"""
        results = []
        
        # Run analysis
        if 'fixed_costs' in context and 'transport_costs' in context and 'demand' in context:
            analysis = self.execute_tool("analyze_facility_costs", {
                "fixed_costs": context['fixed_costs'],
                "transport_costs": context['transport_costs'],
                "demand": context['demand']
            })
            results.append(analysis)
            
            # Run optimization
            if 'capacity' in context:
                opt_result = self.execute_tool("optimize_facility_selection", {
                    "fixed_costs": context['fixed_costs'],
                    "transport_costs": context['transport_costs'],
                    "demand": context['demand'],
                    "capacity": context['capacity']
                })
                results.append(opt_result)
        
        return "Fallback: Executed strategic analysis and optimization", results


class TacticalAgentSkills(AgentSkillsToolkit):
    """
    Skills for Tactical Agent (LP-based logistics optimization).
    
    Tools:
    - analyze_supply_demand: Analyze supply-demand balance
    - optimize_transportation: Run LP transportation optimization
    - evaluate_route_efficiency: Evaluate transportation route efficiency
    - rebalance_flow: Adjust flow allocation
    """
    
    def get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "analyze_supply_demand",
                "description": "Analyze the balance between supply and demand. Essential first step before optimization.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "supply": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Supply at each source (active facilities)"
                        },
                        "demand": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Demand at each destination (customers)"
                        }
                    },
                    "required": ["supply", "demand"]
                }
            },
            {
                "name": "optimize_transportation",
                "description": "Run Linear Programming to optimize transportation flow from sources to destinations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "supply": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Supply at each source"
                        },
                        "demand": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Demand at each destination"
                        },
                        "costs": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Transportation cost matrix [source][destination]"
                        }
                    },
                    "required": ["supply", "demand", "costs"]
                }
            },
            {
                "name": "evaluate_route_efficiency",
                "description": "Evaluate the efficiency of transportation routes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "allocation": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Current flow allocation matrix"
                        },
                        "costs": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Cost matrix"
                        }
                    },
                    "required": ["allocation", "costs"]
                }
            },
            {
                "name": "calculate_facility_requirements",
                "description": "Calculate inbound volume requirements for each facility.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "allocation": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "Flow allocation matrix"
                        },
                        "facility_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Original facility indices"
                        }
                    },
                    "required": ["allocation", "facility_indices"]
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, tool_input: Dict) -> ToolResult:
        try:
            if tool_name == "analyze_supply_demand":
                result = self._analyze_supply_demand(**tool_input)
            elif tool_name == "optimize_transportation":
                result = self._optimize_transportation(**tool_input)
            elif tool_name == "evaluate_route_efficiency":
                result = self._evaluate_route_efficiency(**tool_input)
            elif tool_name == "calculate_facility_requirements":
                result = self._calculate_facility_requirements(**tool_input)
            else:
                return ToolResult(tool_name, False, None, metadata={"error": f"Unknown tool: {tool_name}"})
            
            # Convert numpy types to native Python types for JSON serialization
            result = convert_to_serializable(result)
            return ToolResult(tool_name, True, result, reasoning=result.get("reasoning", ""))
        except Exception as e:
            return ToolResult(tool_name, False, None, metadata={"error": str(e)})
    
    def _analyze_supply_demand(self, supply: List[float], demand: List[float]) -> Dict:
        """Analyze supply-demand balance"""
        supply = np.array(supply)
        demand = np.array(demand)
        
        total_supply = np.sum(supply)
        total_demand = np.sum(demand)
        balance = total_supply - total_demand
        
        is_balanced = abs(balance) < 0.01 * total_demand
        is_feasible = total_supply >= total_demand
        
        return {
            "total_supply": round(total_supply, 2),
            "total_demand": round(total_demand, 2),
            "balance": round(balance, 2),
            "is_balanced": is_balanced,
            "is_feasible": is_feasible,
            "supply_utilization": round(total_demand / total_supply, 3) if total_supply > 0 else 0,
            "supply_distribution": [round(s / total_supply, 3) for s in supply] if total_supply > 0 else [],
            "demand_distribution": [round(d / total_demand, 3) for d in demand] if total_demand > 0 else [],
            "reasoning": f"{'Balanced' if is_balanced else 'Unbalanced'} system. Supply excess: {balance:.2f}"
        }
    
    def _optimize_transportation(self, supply: List[float], demand: List[float],
                                 costs: List[List[float]]) -> Dict:
        """Run LP transportation optimization"""
        supply = np.array(supply)
        demand = np.array(demand)
        costs = np.array(costs)
        
        m, n = len(supply), len(demand)
        
        if costs.shape != (m, n):
            return {"success": False, "error": f"Cost matrix shape mismatch"}
        
        # Flatten costs for objective
        c = costs.flatten()
        
        # Supply constraints (<=)
        A_ub = np.zeros((m, m * n))
        b_ub = supply.copy()
        for i in range(m):
            A_ub[i, i*n:(i+1)*n] = 1
        
        # Demand constraints (==)
        A_eq = np.zeros((n, m * n))
        b_eq = demand.copy()
        for j in range(n):
            for i in range(m):
                A_eq[j, i*n + j] = 1
        
        bounds = [(0, None) for _ in range(m * n)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                allocation = res.x.reshape(m, n)
                
                return {
                    "success": True,
                    "total_cost": round(res.fun, 2),
                    "allocation": [[round(x, 2) for x in row] for row in allocation],
                    "active_routes": int(np.sum(allocation > 0.01)),
                    "reasoning": f"Optimal transportation plan found. Cost: {res.fun:.2f}"
                }
            else:
                return {
                    "success": False,
                    "error": res.message,
                    "reasoning": f"Optimization failed: {res.message}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "reasoning": f"Optimization error: {str(e)}"
            }
    
    def _evaluate_route_efficiency(self, allocation: List[List[float]], 
                                   costs: List[List[float]]) -> Dict:
        """Evaluate route efficiency"""
        allocation = np.array(allocation)
        costs = np.array(costs)
        
        # Calculate weighted average cost
        total_flow = np.sum(allocation)
        total_cost = np.sum(allocation * costs)
        avg_cost = total_cost / total_flow if total_flow > 0 else 0
        
        # Find most/least efficient routes
        efficiency = []
        for i in range(allocation.shape[0]):
            for j in range(allocation.shape[1]):
                if allocation[i, j] > 0.01:
                    efficiency.append({
                        "route": f"{i}->{j}",
                        "flow": round(allocation[i, j], 2),
                        "cost": round(costs[i, j], 2),
                        "total_cost": round(allocation[i, j] * costs[i, j], 2)
                    })
        
        efficiency.sort(key=lambda x: x["cost"])
        
        return {
            "total_flow": round(total_flow, 2),
            "total_cost": round(total_cost, 2),
            "average_cost_per_unit": round(avg_cost, 3),
            "num_active_routes": len(efficiency),
            "most_efficient_routes": efficiency[:3] if efficiency else [],
            "least_efficient_routes": efficiency[-3:] if efficiency else [],
            "reasoning": f"Average transport cost: {avg_cost:.3f} per unit"
        }
    
    def _calculate_facility_requirements(self, allocation: List[List[float]],
                                         facility_indices: List[int]) -> Dict:
        """Calculate inbound volumes for facilities"""
        allocation = np.array(allocation)
        
        # Sum outflows for each facility (row sum)
        outflows = np.sum(allocation, axis=1)
        
        requirements = {}
        for idx, local_idx in enumerate(range(len(facility_indices))):
            if local_idx < len(outflows):
                requirements[facility_indices[local_idx]] = round(outflows[local_idx], 2)
        
        return {
            "facility_requirements": requirements,
            "total_volume": round(np.sum(outflows), 2),
            "reasoning": f"Calculated requirements for {len(requirements)} facilities"
        }
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Tactical Operations Agent specializing in logistics and transportation optimization.

Your role:
1. Analyze supply-demand balance given the strategic constraints
2. Optimize transportation flow using Linear Programming
3. Evaluate route efficiency
4. Calculate facility requirements for operational planning

You operate within constraints set by the Strategic Agent (which facilities are open).

Always think step-by-step:
1. First, analyze the supply-demand balance
2. Then, optimize transportation
3. Evaluate the efficiency of the solution
4. Calculate requirements for the Operational Agent

Use the available tools systematically and ensure your decisions respect upstream constraints."""

    def _fallback_reasoning(self, prompt: str, context: Dict) -> Tuple[str, List[ToolResult]]:
        """Fallback logic for tactical decisions"""
        results = []
        
        if 'supply' in context and 'demand' in context:
            # Analyze supply-demand
            analysis = self.execute_tool("analyze_supply_demand", {
                "supply": context['supply'],
                "demand": context['demand']
            })
            results.append(analysis)
            
            # Optimize if costs available
            if 'costs' in context:
                opt_result = self.execute_tool("optimize_transportation", {
                    "supply": context['supply'],
                    "demand": context['demand'],
                    "costs": context['costs']
                })
                results.append(opt_result)
        
        return "Fallback: Executed tactical analysis and optimization", results


class OperationalAgentSkills(AgentSkillsToolkit):
    """
    Skills for Operational Agent (DP-based inventory optimization).
    
    Tools:
    - analyze_inventory_items: Analyze item values and weights
    - optimize_inventory_mix: Run DP knapsack optimization
    - evaluate_inventory_plan: Evaluate inventory plan quality
    - optimize_path: Find optimal path through cost grid
    """
    
    def get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "name": "analyze_inventory_items",
                "description": "Analyze inventory items to understand value-weight trade-offs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Value of each item"
                        },
                        "weights": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Weight/space of each item"
                        }
                    },
                    "required": ["values", "weights"]
                }
            },
            {
                "name": "optimize_inventory_mix",
                "description": "Run Dynamic Programming knapsack to find optimal inventory mix for given capacity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Value of each item"
                        },
                        "weights": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Weight of each item"
                        },
                        "capacity": {
                            "type": "integer",
                            "description": "Maximum capacity/budget"
                        }
                    },
                    "required": ["values", "weights", "capacity"]
                }
            },
            {
                "name": "evaluate_inventory_plan",
                "description": "Evaluate the quality of an inventory selection.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "selected_items": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Indices of selected items"
                        },
                        "values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "All item values"
                        },
                        "weights": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "All item weights"
                        },
                        "capacity": {
                            "type": "integer",
                            "description": "Capacity constraint"
                        }
                    },
                    "required": ["selected_items", "values", "weights", "capacity"]
                }
            },
            {
                "name": "multi_facility_optimization",
                "description": "Optimize inventory across multiple facilities with different capacities.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "facility_capacities": {
                            "type": "object",
                            "description": "Dict mapping facility_id to capacity"
                        },
                        "values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Item values"
                        },
                        "weights": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Item weights"
                        }
                    },
                    "required": ["facility_capacities", "values", "weights"]
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, tool_input: Dict) -> ToolResult:
        try:
            if tool_name == "analyze_inventory_items":
                result = self._analyze_inventory_items(**tool_input)
            elif tool_name == "optimize_inventory_mix":
                result = self._optimize_inventory_mix(**tool_input)
            elif tool_name == "evaluate_inventory_plan":
                result = self._evaluate_inventory_plan(**tool_input)
            elif tool_name == "multi_facility_optimization":
                result = self._multi_facility_optimization(**tool_input)
            else:
                return ToolResult(tool_name, False, None, metadata={"error": f"Unknown tool: {tool_name}"})
            
            # Convert numpy types to native Python types for JSON serialization
            result = convert_to_serializable(result)
            return ToolResult(tool_name, True, result, reasoning=result.get("reasoning", ""))
        except Exception as e:
            return ToolResult(tool_name, False, None, metadata={"error": str(e)})
    
    def _analyze_inventory_items(self, values: List[float], weights: List[int]) -> Dict:
        """Analyze item characteristics"""
        values = np.array(values)
        weights = np.array(weights)
        
        # Value density (value per unit weight)
        densities = values / np.maximum(weights, 1)
        
        # Rank by density
        ranked = sorted(range(len(densities)), key=lambda x: densities[x], reverse=True)
        
        return {
            "num_items": len(values),
            "total_value": round(np.sum(values), 2),
            "total_weight": int(np.sum(weights)),
            "value_stats": {
                "min": round(float(np.min(values)), 2),
                "max": round(float(np.max(values)), 2),
                "mean": round(float(np.mean(values)), 2)
            },
            "weight_stats": {
                "min": int(np.min(weights)),
                "max": int(np.max(weights)),
                "mean": round(float(np.mean(weights)), 2)
            },
            "density_ranking": ranked[:5],
            "top_items_by_density": [{"item": i, "density": round(densities[i], 3)} for i in ranked[:5]],
            "reasoning": f"Analyzed {len(values)} items. Best value density items: {ranked[:3]}"
        }
    
    def _optimize_inventory_mix(self, values: List[float], weights: List[int], capacity: int) -> Dict:
        """Run knapsack DP"""
        n = len(values)
        
        # DP table
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Backtrack to find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i - 1)
                w -= weights[i-1]
        
        selected.reverse()
        
        total_value = dp[n][capacity]
        total_weight = sum(weights[i] for i in selected)
        
        return {
            "success": True,
            "selected_items": selected,
            "total_value": round(total_value, 2),
            "total_weight": total_weight,
            "capacity_used": round(total_weight / capacity, 3) if capacity > 0 else 0,
            "num_items_selected": len(selected),
            "reasoning": f"Optimal mix: {len(selected)} items, value {total_value:.2f}, using {total_weight}/{capacity} capacity"
        }
    
    def _evaluate_inventory_plan(self, selected_items: List[int], values: List[float],
                                 weights: List[int], capacity: int) -> Dict:
        """Evaluate inventory plan quality"""
        values = np.array(values)
        weights = np.array(weights)
        
        selected_values = values[selected_items]
        selected_weights = weights[selected_items]
        
        total_value = np.sum(selected_values)
        total_weight = np.sum(selected_weights)
        
        # Check feasibility
        is_feasible = total_weight <= capacity
        
        # Compare to theoretical maximum (greedy approximation)
        densities = values / np.maximum(weights, 1)
        ranked = sorted(range(len(densities)), key=lambda x: densities[x], reverse=True)
        
        greedy_value = 0
        greedy_weight = 0
        for i in ranked:
            if greedy_weight + weights[i] <= capacity:
                greedy_value += values[i]
                greedy_weight += weights[i]
        
        efficiency = total_value / greedy_value if greedy_value > 0 else 0
        
        return {
            "is_feasible": is_feasible,
            "total_value": round(total_value, 2),
            "total_weight": int(total_weight),
            "capacity_used": round(total_weight / capacity, 3) if capacity > 0 else 0,
            "greedy_benchmark": round(greedy_value, 2),
            "efficiency_vs_greedy": round(efficiency, 3),
            "rating": "optimal" if efficiency > 0.99 else ("good" if efficiency > 0.9 else "suboptimal"),
            "reasoning": f"Plan uses {total_weight}/{capacity} capacity for value {total_value:.2f}"
        }
    
    def _multi_facility_optimization(self, facility_capacities: Dict[str, float],
                                     values: List[float], weights: List[int]) -> Dict:
        """Optimize inventory across multiple facilities"""
        results = {}
        total_value = 0
        
        for facility_id, capacity in facility_capacities.items():
            cap = int(capacity)
            if cap <= 0:
                continue
            
            opt = self._optimize_inventory_mix(values, weights, cap)
            
            if opt["success"]:
                results[facility_id] = {
                    "selected_items": opt["selected_items"],
                    "total_value": opt["total_value"],
                    "total_weight": opt["total_weight"],
                    "capacity_used": opt["capacity_used"]
                }
                total_value += opt["total_value"]
        
        return {
            "facility_plans": results,
            "total_value_all_facilities": round(total_value, 2),
            "num_facilities_optimized": len(results),
            "reasoning": f"Optimized {len(results)} facilities for total value {total_value:.2f}"
        }
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Operational Agent specializing in inventory optimization and resource allocation.

Your role:
1. Analyze inventory items to understand value-weight trade-offs
2. Optimize inventory mix using Dynamic Programming
3. Evaluate inventory plans for quality
4. Handle multi-facility optimization

You operate within constraints set by the Tactical Agent (facility capacity requirements).

Always think step-by-step:
1. First, analyze the items available
2. Then, optimize the inventory mix for given capacity
3. Evaluate the quality of your solution
4. Provide clear execution recommendations

Use the available tools systematically and ensure efficient resource utilization."""

    def _fallback_reasoning(self, prompt: str, context: Dict) -> Tuple[str, List[ToolResult]]:
        """Fallback logic for operational decisions"""
        results = []
        
        if 'values' in context and 'weights' in context:
            # Analyze items
            analysis = self.execute_tool("analyze_inventory_items", {
                "values": context['values'],
                "weights": context['weights']
            })
            results.append(analysis)
            
            # Optimize if capacity available
            if 'capacity' in context:
                opt_result = self.execute_tool("optimize_inventory_mix", {
                    "values": context['values'],
                    "weights": context['weights'],
                    "capacity": context['capacity']
                })
                results.append(opt_result)
        
        return "Fallback: Executed operational analysis and optimization", results

