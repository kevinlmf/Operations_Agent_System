"""
Claude Agent for Inventory Optimization

Implements Claude's latest agentic capabilities:
1. Tool Use - Execute specialized inventory optimization tools
2. Extended Thinking - Deep reasoning for complex decisions
3. Multi-step Planning - Agentic decision workflows
4. Context Awareness - Adapts to dynamic scenarios
"""

import numpy as np
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import InventoryMethod, MethodCategory, InventoryState, InventoryAction


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    success: bool
    result: Any
    metadata: Dict[str, Any]


class InventoryOptimizationTools:
    """
    Specialized tools for inventory optimization that Claude agent can use.
    Each tool encapsulates domain knowledge and optimization algorithms.
    """
    
    def __init__(self, holding_cost: float = 2.0, ordering_cost: float = 50.0,
                 stockout_cost: float = 10.0, lead_time: int = 1,
                 service_level: float = 0.95):
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.lead_time = lead_time
        self.service_level = service_level
        
        # Store historical analysis results
        self.analysis_cache = {}
    
    def get_tool_definitions(self) -> List[Dict]:
        """Return tool definitions in Claude API format"""
        return [
            {
                "name": "calculate_eoq",
                "description": "Calculate Economic Order Quantity (EOQ) - the optimal order quantity that minimizes total inventory costs. Use this for baseline ordering decisions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "annual_demand": {
                            "type": "number",
                            "description": "Expected annual demand"
                        },
                        "demand_std": {
                            "type": "number",
                            "description": "Standard deviation of demand (optional, for adjustment)"
                        }
                    },
                    "required": ["annual_demand"]
                }
            },
            {
                "name": "calculate_safety_stock",
                "description": "Calculate optimal safety stock level considering demand variability and service level requirements. Essential for handling uncertainty.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "demand_std": {
                            "type": "number",
                            "description": "Standard deviation of demand"
                        },
                        "lead_time": {
                            "type": "integer",
                            "description": "Lead time in periods"
                        },
                        "service_level": {
                            "type": "number",
                            "description": "Target service level (0-1)"
                        }
                    },
                    "required": ["demand_std"]
                }
            },
            {
                "name": "forecast_demand",
                "description": "Forecast future demand using time series analysis. Considers trends, seasonality, and recent patterns.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "historical_demand": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Recent demand history"
                        },
                        "horizon": {
                            "type": "integer",
                            "description": "Number of periods to forecast"
                        },
                        "include_seasonality": {
                            "type": "boolean",
                            "description": "Whether to consider seasonal patterns"
                        }
                    },
                    "required": ["historical_demand", "horizon"]
                }
            },
            {
                "name": "analyze_demand_pattern",
                "description": "Analyze demand patterns to detect trends, seasonality, and anomalies. Provides insights for decision making.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "demand_history": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Historical demand data"
                        }
                    },
                    "required": ["demand_history"]
                }
            },
            {
                "name": "calculate_reorder_point",
                "description": "Calculate the reorder point - inventory level at which a new order should be placed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "average_demand": {
                            "type": "number",
                            "description": "Average demand per period"
                        },
                        "lead_time": {
                            "type": "integer",
                            "description": "Lead time in periods"
                        },
                        "safety_stock": {
                            "type": "number",
                            "description": "Safety stock level"
                        }
                    },
                    "required": ["average_demand", "lead_time", "safety_stock"]
                }
            },
            {
                "name": "multi_period_optimization",
                "description": "Perform multi-period inventory optimization using dynamic programming approach. Best for complex planning horizons.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "current_inventory": {
                            "type": "number",
                            "description": "Current inventory level"
                        },
                        "demand_forecast": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Forecasted demand for planning horizon"
                        },
                        "outstanding_orders": {
                            "type": "number",
                            "description": "Orders already placed but not received"
                        }
                    },
                    "required": ["current_inventory", "demand_forecast"]
                }
            },
            {
                "name": "risk_assessment",
                "description": "Assess inventory risk including stockout probability and cost exposure. Use for risk-adjusted decisions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "current_inventory": {
                            "type": "number",
                            "description": "Current inventory level"
                        },
                        "expected_demand": {
                            "type": "number",
                            "description": "Expected demand"
                        },
                        "demand_std": {
                            "type": "number",
                            "description": "Demand standard deviation"
                        },
                        "lead_time": {
                            "type": "integer",
                            "description": "Lead time for orders"
                        }
                    },
                    "required": ["current_inventory", "expected_demand", "demand_std"]
                }
            },
            {
                "name": "recommend_order",
                "description": "Generate final order recommendation synthesizing all analysis. This is the primary decision tool.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "current_inventory": {
                            "type": "number",
                            "description": "Current inventory level"
                        },
                        "eoq": {
                            "type": "number",
                            "description": "Calculated EOQ"
                        },
                        "reorder_point": {
                            "type": "number",
                            "description": "Reorder point"
                        },
                        "safety_stock": {
                            "type": "number",
                            "description": "Safety stock level"
                        },
                        "demand_forecast": {
                            "type": "number",
                            "description": "Forecasted demand"
                        },
                        "risk_level": {
                            "type": "string",
                            "description": "Risk assessment result"
                        }
                    },
                    "required": ["current_inventory", "eoq", "reorder_point"]
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, tool_input: Dict) -> ToolResult:
        """Execute a tool and return the result"""
        try:
            if tool_name == "calculate_eoq":
                result = self._calculate_eoq(**tool_input)
            elif tool_name == "calculate_safety_stock":
                result = self._calculate_safety_stock(**tool_input)
            elif tool_name == "forecast_demand":
                result = self._forecast_demand(**tool_input)
            elif tool_name == "analyze_demand_pattern":
                result = self._analyze_demand_pattern(**tool_input)
            elif tool_name == "calculate_reorder_point":
                result = self._calculate_reorder_point(**tool_input)
            elif tool_name == "multi_period_optimization":
                result = self._multi_period_optimization(**tool_input)
            elif tool_name == "risk_assessment":
                result = self._risk_assessment(**tool_input)
            elif tool_name == "recommend_order":
                result = self._recommend_order(**tool_input)
            else:
                return ToolResult(tool_name, False, None, {"error": f"Unknown tool: {tool_name}"})
            
            return ToolResult(tool_name, True, result, {})
        except Exception as e:
            return ToolResult(tool_name, False, None, {"error": str(e)})
    
    def _calculate_eoq(self, annual_demand: float, demand_std: float = None) -> Dict:
        """Calculate Economic Order Quantity"""
        # Classic EOQ formula: sqrt(2 * D * K / h)
        eoq = np.sqrt(2 * annual_demand * self.ordering_cost / self.holding_cost)
        
        # Adjust for uncertainty if demand_std provided
        uncertainty_adjustment = 1.0
        if demand_std and demand_std > 0:
            cv = demand_std / (annual_demand / 365) if annual_demand > 0 else 0
            uncertainty_adjustment = 1 + 0.1 * cv  # Increase order size under uncertainty
        
        adjusted_eoq = eoq * uncertainty_adjustment
        
        return {
            "eoq": round(eoq, 2),
            "adjusted_eoq": round(adjusted_eoq, 2),
            "annual_orders": round(annual_demand / eoq, 1) if eoq > 0 else 0,
            "order_cycle_days": round(365 * eoq / annual_demand, 1) if annual_demand > 0 else 0,
            "uncertainty_adjustment": round(uncertainty_adjustment, 3)
        }
    
    def _calculate_safety_stock(self, demand_std: float, lead_time: int = None,
                                service_level: float = None) -> Dict:
        """Calculate safety stock level"""
        from scipy import stats
        
        lead_time = lead_time or self.lead_time
        service_level = service_level or self.service_level
        
        # Z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        # Safety stock formula: z * σ * sqrt(L)
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        
        return {
            "safety_stock": round(max(0, safety_stock), 2),
            "z_score": round(z_score, 3),
            "lead_time": lead_time,
            "service_level": service_level,
            "coverage_days": round(safety_stock / (demand_std + 0.01), 1)
        }
    
    def _forecast_demand(self, historical_demand: List[float], horizon: int,
                        include_seasonality: bool = True) -> Dict:
        """Forecast future demand"""
        demand = np.array(historical_demand)
        
        # Simple exponential smoothing with trend
        alpha = 0.3  # Smoothing factor
        beta = 0.1   # Trend factor
        
        # Initialize
        level = demand[0]
        trend = np.mean(np.diff(demand[:min(7, len(demand))])) if len(demand) > 1 else 0
        
        # Fit model
        for i in range(1, len(demand)):
            prev_level = level
            level = alpha * demand[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        
        # Generate forecasts
        forecasts = []
        for h in range(1, horizon + 1):
            forecast = level + h * trend
            
            # Add seasonality if requested and enough data
            if include_seasonality and len(demand) >= 14:
                # Simple weekly seasonality
                week_day = (len(demand) + h - 1) % 7
                seasonal_factors = []
                for d in range(len(demand)):
                    if d % 7 == week_day:
                        seasonal_factors.append(demand[d])
                if seasonal_factors:
                    seasonal_adjustment = np.mean(seasonal_factors) / np.mean(demand) if np.mean(demand) > 0 else 1
                    forecast *= seasonal_adjustment
            
            forecasts.append(max(0, forecast))
        
        return {
            "forecasts": [round(f, 2) for f in forecasts],
            "mean_forecast": round(np.mean(forecasts), 2),
            "trend": round(trend, 3),
            "level": round(level, 2),
            "forecast_std": round(np.std(forecasts) if len(forecasts) > 1 else 0, 2)
        }
    
    def _analyze_demand_pattern(self, demand_history: List[float]) -> Dict:
        """Analyze demand patterns"""
        demand = np.array(demand_history)
        
        # Basic statistics
        mean_demand = np.mean(demand)
        std_demand = np.std(demand)
        cv = std_demand / mean_demand if mean_demand > 0 else 0
        
        # Trend detection (linear regression)
        x = np.arange(len(demand))
        if len(demand) >= 3:
            slope, intercept = np.polyfit(x, demand, 1)
            trend_strength = abs(slope) * len(demand) / (std_demand + 0.01)
            trend_direction = "increasing" if slope > 0.5 else ("decreasing" if slope < -0.5 else "stable")
        else:
            slope = 0
            trend_strength = 0
            trend_direction = "insufficient_data"
        
        # Seasonality detection (autocorrelation at lag 7)
        seasonality_score = 0
        if len(demand) >= 14:
            for lag in [7, 14]:
                if len(demand) > lag:
                    corr = np.corrcoef(demand[:-lag], demand[lag:])[0, 1]
                    seasonality_score = max(seasonality_score, abs(corr) if not np.isnan(corr) else 0)
        
        has_seasonality = seasonality_score > 0.3
        
        # Volatility analysis
        if len(demand) >= 7:
            rolling_std = []
            for i in range(7, len(demand)):
                rolling_std.append(np.std(demand[i-7:i]))
            volatility_trend = "increasing" if len(rolling_std) > 1 and rolling_std[-1] > rolling_std[0] else "stable"
        else:
            volatility_trend = "insufficient_data"
        
        # Anomaly detection (simple z-score)
        z_scores = (demand - mean_demand) / (std_demand + 0.01)
        anomalies = np.where(np.abs(z_scores) > 2.5)[0].tolist()
        
        return {
            "mean_demand": round(mean_demand, 2),
            "std_demand": round(std_demand, 2),
            "cv": round(cv, 3),
            "trend_direction": trend_direction,
            "trend_strength": round(trend_strength, 3),
            "has_seasonality": has_seasonality,
            "seasonality_score": round(seasonality_score, 3),
            "volatility_trend": volatility_trend,
            "anomaly_periods": anomalies[:5],  # Top 5 anomalies
            "demand_pattern": self._classify_demand_pattern(cv, trend_strength, has_seasonality)
        }
    
    def _classify_demand_pattern(self, cv: float, trend_strength: float, has_seasonality: bool) -> str:
        """Classify demand pattern type"""
        if cv < 0.2:
            base = "stable"
        elif cv < 0.5:
            base = "variable"
        else:
            base = "highly_variable"
        
        modifiers = []
        if trend_strength > 0.5:
            modifiers.append("trending")
        if has_seasonality:
            modifiers.append("seasonal")
        
        if modifiers:
            return f"{base}_{'_'.join(modifiers)}"
        return base
    
    def _calculate_reorder_point(self, average_demand: float, lead_time: int,
                                 safety_stock: float) -> Dict:
        """Calculate reorder point"""
        reorder_point = average_demand * lead_time + safety_stock
        
        return {
            "reorder_point": round(reorder_point, 2),
            "lead_time_demand": round(average_demand * lead_time, 2),
            "safety_stock_component": round(safety_stock, 2),
            "days_of_supply_at_rop": round(reorder_point / average_demand, 1) if average_demand > 0 else 0
        }
    
    def _multi_period_optimization(self, current_inventory: float,
                                   demand_forecast: List[float],
                                   outstanding_orders: float = 0) -> Dict:
        """Multi-period optimization using simplified dynamic programming"""
        horizon = len(demand_forecast)
        
        # State: inventory level
        # Decision: order quantity
        
        # Backward induction
        best_orders = []
        projected_inventory = [current_inventory + outstanding_orders]
        total_cost = 0
        
        for t in range(horizon):
            inv = projected_inventory[-1]
            demand = demand_forecast[t]
            
            # Simple heuristic: order to bring inventory to target level
            target = demand * 1.5  # Target 1.5x expected demand
            order_qty = max(0, target - inv + demand)
            
            # Calculate period cost
            ending_inv = inv + order_qty - demand
            holding_cost = self.holding_cost * max(0, ending_inv)
            stockout_cost = self.stockout_cost * max(0, -ending_inv)
            ordering_cost = self.ordering_cost if order_qty > 0 else 0
            
            period_cost = holding_cost + stockout_cost + ordering_cost
            total_cost += period_cost
            
            best_orders.append(round(order_qty, 2))
            projected_inventory.append(max(0, ending_inv))
        
        return {
            "recommended_orders": best_orders,
            "first_period_order": round(best_orders[0] if best_orders else 0, 2),
            "total_expected_cost": round(total_cost, 2),
            "avg_inventory": round(np.mean(projected_inventory), 2),
            "min_inventory": round(min(projected_inventory), 2),
            "planning_horizon": horizon
        }
    
    def _risk_assessment(self, current_inventory: float, expected_demand: float,
                        demand_std: float, lead_time: int = None) -> Dict:
        """Assess inventory risk"""
        from scipy import stats
        
        lead_time = lead_time or self.lead_time
        
        # Stockout probability
        total_expected = expected_demand * lead_time
        total_std = demand_std * np.sqrt(lead_time)
        
        if total_std > 0:
            z_score = (current_inventory - total_expected) / total_std
            stockout_prob = 1 - stats.norm.cdf(z_score)
        else:
            stockout_prob = 0 if current_inventory >= total_expected else 1
        
        # Expected stockout quantity
        expected_stockout = max(0, total_expected - current_inventory)
        
        # Cost exposure
        stockout_cost_exposure = expected_stockout * self.stockout_cost * stockout_prob
        holding_cost_exposure = current_inventory * self.holding_cost
        
        # Risk level classification
        if stockout_prob < 0.05:
            risk_level = "low"
        elif stockout_prob < 0.15:
            risk_level = "medium"
        elif stockout_prob < 0.3:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Days of supply
        days_of_supply = current_inventory / expected_demand if expected_demand > 0 else float('inf')
        
        return {
            "stockout_probability": round(stockout_prob, 4),
            "risk_level": risk_level,
            "expected_stockout_qty": round(expected_stockout, 2),
            "stockout_cost_exposure": round(stockout_cost_exposure, 2),
            "holding_cost_exposure": round(holding_cost_exposure, 2),
            "total_cost_exposure": round(stockout_cost_exposure + holding_cost_exposure, 2),
            "days_of_supply": round(days_of_supply, 1),
            "coverage_ratio": round(current_inventory / total_expected, 2) if total_expected > 0 else float('inf')
        }
    
    def _recommend_order(self, current_inventory: float, eoq: float, reorder_point: float,
                        safety_stock: float = 0, demand_forecast: float = 0,
                        risk_level: str = "medium") -> Dict:
        """Generate final order recommendation"""
        
        # Decision logic
        should_order = current_inventory <= reorder_point
        
        # Base order quantity
        if should_order:
            base_qty = eoq
            
            # Adjust for risk
            risk_multipliers = {
                "low": 1.0,
                "medium": 1.1,
                "high": 1.25,
                "critical": 1.5
            }
            adjusted_qty = base_qty * risk_multipliers.get(risk_level, 1.0)
            
            # Ensure we reach safety stock
            min_order = max(0, safety_stock + demand_forecast - current_inventory)
            order_quantity = max(adjusted_qty, min_order)
        else:
            order_quantity = 0
        
        # Order urgency
        if current_inventory < safety_stock:
            urgency = "immediate"
        elif current_inventory < reorder_point:
            urgency = "soon"
        else:
            urgency = "not_needed"
        
        return {
            "order_quantity": round(order_quantity, 2),
            "should_order": should_order,
            "urgency": urgency,
            "order_reason": self._get_order_reason(current_inventory, reorder_point, safety_stock, risk_level),
            "target_inventory_after_order": round(current_inventory + order_quantity, 2),
            "confidence": self._calculate_confidence(risk_level, demand_forecast)
        }
    
    def _get_order_reason(self, current_inv: float, rop: float, ss: float, risk: str) -> str:
        """Generate explanation for order decision"""
        if current_inv < ss:
            return "Inventory below safety stock - urgent replenishment needed"
        elif current_inv < rop:
            return "Inventory below reorder point - standard replenishment"
        elif risk in ["high", "critical"]:
            return "Preemptive order due to high demand risk"
        else:
            return "Inventory sufficient - no order needed"
    
    def _calculate_confidence(self, risk_level: str, demand_forecast: float) -> float:
        """Calculate confidence in recommendation"""
        base_confidence = {
            "low": 0.95,
            "medium": 0.85,
            "high": 0.70,
            "critical": 0.55
        }
        return base_confidence.get(risk_level, 0.80)


class ClaudeAgentInventoryMethod(InventoryMethod):
    """
    Claude Agent-based Inventory Management Method
    
    Uses Claude's agentic capabilities:
    1. Tool Use - Executes specialized inventory optimization tools
    2. Extended Thinking - Deep reasoning for complex decisions
    3. Multi-step Planning - Agentic decision workflows
    4. Context Awareness - Adapts to dynamic scenarios
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514",
                 holding_cost: float = 2.0,
                 ordering_cost: float = 50.0,
                 stockout_cost: float = 10.0,
                 lead_time: int = 1,
                 service_level: float = 0.95,
                 use_extended_thinking: bool = True,
                 max_tool_iterations: int = 5,
                 fallback_mode: bool = True):
        """
        Initialize Claude Agent for inventory management.
        
        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Claude model to use
            holding_cost: Per-unit holding cost per period
            ordering_cost: Fixed cost per order
            stockout_cost: Per-unit stockout cost
            lead_time: Order lead time in periods
            service_level: Target service level (0-1)
            use_extended_thinking: Enable extended thinking for complex decisions
            max_tool_iterations: Maximum tool use iterations per decision
            fallback_mode: If True, use local tools when API unavailable
        """
        super().__init__("Claude_Agent", MethodCategory.HYBRID)
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.use_extended_thinking = use_extended_thinking
        self.max_tool_iterations = max_tool_iterations
        self.fallback_mode = fallback_mode
        
        # Initialize tools
        self.tools = InventoryOptimizationTools(
            holding_cost=holding_cost,
            ordering_cost=ordering_cost,
            stockout_cost=stockout_cost,
            lead_time=lead_time,
            service_level=service_level
        )
        
        # Initialize Claude client
        self.client = None
        self._init_client()
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.demand_stats = {}
        
    def _init_client(self):
        """Initialize Anthropic client"""
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️ anthropic package not installed. Using fallback mode.")
                self.client = None
            except Exception as e:
                print(f"⚠️ Could not initialize Anthropic client: {e}")
                self.client = None
        else:
            if not self.fallback_mode:
                print("⚠️ No API key provided. Using fallback mode.")
    
    def fit(self,
            demand_history: np.ndarray,
            external_features: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Fit the agent on historical demand data.
        
        Args:
            demand_history: Historical demand time series
            external_features: Optional external factors
        """
        # Analyze demand patterns
        demand_list = demand_history.tolist() if isinstance(demand_history, np.ndarray) else list(demand_history)
        
        pattern_result = self.tools.execute_tool(
            "analyze_demand_pattern",
            {"demand_history": demand_list}
        )
        
        if pattern_result.success:
            self.demand_stats = pattern_result.result
        else:
            self.demand_stats = {
                "mean_demand": np.mean(demand_history),
                "std_demand": np.std(demand_history),
                "cv": np.std(demand_history) / np.mean(demand_history) if np.mean(demand_history) > 0 else 0,
                "trend_direction": "stable",
                "has_seasonality": False,
                "demand_pattern": "unknown"
            }
        
        self._is_fitted = True
        
    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        Predict future demand using agent's forecasting tools.
        
        Args:
            current_state: Current inventory state
            horizon: Number of periods to predict
            
        Returns:
            Predicted demand for horizon periods
        """
        if not self._is_fitted:
            raise ValueError("Agent must be fitted before prediction")
        
        # Get historical demand
        demand_list = current_state.demand_history.tolist() if isinstance(
            current_state.demand_history, np.ndarray
        ) else list(current_state.demand_history)
        
        # Use forecasting tool
        forecast_result = self.tools.execute_tool(
            "forecast_demand",
            {
                "historical_demand": demand_list,
                "horizon": horizon,
                "include_seasonality": self.demand_stats.get("has_seasonality", False)
            }
        )
        
        if forecast_result.success:
            return np.array(forecast_result.result["forecasts"][:horizon])
        else:
            # Fallback: simple mean
            return np.full(horizon, np.mean(demand_list))
    
    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend inventory action using agentic reasoning.
        
        This method implements the full agentic workflow:
        1. Analyze current situation
        2. Execute relevant tools
        3. Synthesize insights
        4. Generate recommendation
        
        Args:
            current_state: Current inventory state
            
        Returns:
            Recommended inventory action
        """
        if not self._is_fitted:
            raise ValueError("Agent must be fitted before recommendation")
        
        # Try Claude API first, fallback to local tools if unavailable
        if self.client and self.api_key:
            try:
                return self._recommend_with_claude(current_state)
            except Exception as e:
                if self.fallback_mode:
                    print(f"⚠️ Claude API failed ({e}), using fallback")
                    return self._recommend_with_local_tools(current_state)
                raise
        else:
            return self._recommend_with_local_tools(current_state)
    
    def _recommend_with_claude(self, current_state: InventoryState) -> InventoryAction:
        """Use Claude API for recommendation with tool use"""
        import anthropic
        
        # Prepare context
        demand_list = current_state.demand_history.tolist() if isinstance(
            current_state.demand_history, np.ndarray
        ) else list(current_state.demand_history)
        
        context = {
            "current_inventory": current_state.inventory_level,
            "outstanding_orders": current_state.outstanding_orders,
            "recent_demand": demand_list[-7:] if len(demand_list) >= 7 else demand_list,
            "time_step": current_state.time_step,
            "demand_pattern": self.demand_stats.get("demand_pattern", "unknown"),
            "mean_demand": self.demand_stats.get("mean_demand", np.mean(demand_list)),
            "std_demand": self.demand_stats.get("std_demand", np.std(demand_list)),
            "trend": self.demand_stats.get("trend_direction", "stable")
        }
        
        system_prompt = """You are an expert inventory optimization agent. Your goal is to make optimal ordering decisions that minimize total costs while maintaining service levels.

You have access to specialized inventory optimization tools. Use them systematically:
1. First, analyze the demand pattern if not cached
2. Calculate safety stock and EOQ
3. Assess risk based on current inventory
4. Make a final recommendation

Think step-by-step and use tools to gather data before making decisions. Be quantitative and precise."""

        user_message = f"""Current inventory situation:
- Current inventory level: {context['current_inventory']} units
- Outstanding orders: {context['outstanding_orders']} units
- Recent demand (last 7 periods): {context['recent_demand']}
- Demand pattern: {context['demand_pattern']}
- Mean demand: {context['mean_demand']:.2f}
- Demand std: {context['std_demand']:.2f}
- Trend: {context['trend']}

Please analyze this situation and recommend an order quantity. Use the available tools to calculate EOQ, safety stock, reorder point, and assess risk before making your final recommendation."""

        messages = [{"role": "user", "content": user_message}]
        tools = self.tools.get_tool_definitions()
        
        # Agentic loop
        for iteration in range(self.max_tool_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=messages
            )
            
            # Check if model wants to use tools
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            
            if not tool_uses:
                # No more tools needed, extract recommendation
                text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
                return self._extract_recommendation_from_text(" ".join(text_blocks), context)
            
            # Execute tools
            tool_results = []
            for tool_use in tool_uses:
                result = self.tools.execute_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result.result if result.success else {"error": result.metadata.get("error", "Unknown error")})
                })
            
            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        
        # Fallback if max iterations reached
        return self._recommend_with_local_tools(current_state)
    
    def _extract_recommendation_from_text(self, text: str, context: Dict) -> InventoryAction:
        """Extract order recommendation from Claude's response"""
        import re
        
        # Try to find order quantity in text
        patterns = [
            r"order[:\s]+(\d+(?:\.\d+)?)",
            r"recommend[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*units",
            r"quantity[:\s]+(\d+(?:\.\d+)?)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                order_qty = float(match.group(1))
                return InventoryAction(
                    order_quantity=order_qty,
                    forecast=context.get("mean_demand", 50.0)
                )
        
        # Fallback to local tools if can't parse
        return self._recommend_with_local_tools_from_context(context)
    
    def _recommend_with_local_tools(self, current_state: InventoryState) -> InventoryAction:
        """Use local tools for recommendation (fallback mode)"""
        demand_list = current_state.demand_history.tolist() if isinstance(
            current_state.demand_history, np.ndarray
        ) else list(current_state.demand_history)
        
        context = {
            "current_inventory": current_state.inventory_level,
            "outstanding_orders": current_state.outstanding_orders,
            "mean_demand": self.demand_stats.get("mean_demand", np.mean(demand_list)),
            "std_demand": self.demand_stats.get("std_demand", np.std(demand_list))
        }
        
        return self._recommend_with_local_tools_from_context(context)
    
    def _recommend_with_local_tools_from_context(self, context: Dict) -> InventoryAction:
        """Execute local tool chain for recommendation"""
        mean_demand = context.get("mean_demand", 50.0)
        std_demand = context.get("std_demand", 10.0)
        current_inv = context.get("current_inventory", 0)
        
        # Step 1: Calculate EOQ
        annual_demand = mean_demand * 365
        eoq_result = self.tools.execute_tool(
            "calculate_eoq",
            {"annual_demand": annual_demand, "demand_std": std_demand}
        )
        eoq = eoq_result.result["adjusted_eoq"] if eoq_result.success else mean_demand * 7
        
        # Step 2: Calculate safety stock
        ss_result = self.tools.execute_tool(
            "calculate_safety_stock",
            {"demand_std": std_demand}
        )
        safety_stock = ss_result.result["safety_stock"] if ss_result.success else std_demand * 2
        
        # Step 3: Calculate reorder point
        rop_result = self.tools.execute_tool(
            "calculate_reorder_point",
            {"average_demand": mean_demand, "lead_time": self.tools.lead_time, "safety_stock": safety_stock}
        )
        reorder_point = rop_result.result["reorder_point"] if rop_result.success else mean_demand + safety_stock
        
        # Step 4: Risk assessment
        risk_result = self.tools.execute_tool(
            "risk_assessment",
            {
                "current_inventory": current_inv,
                "expected_demand": mean_demand,
                "demand_std": std_demand
            }
        )
        risk_level = risk_result.result["risk_level"] if risk_result.success else "medium"
        
        # Step 5: Final recommendation
        rec_result = self.tools.execute_tool(
            "recommend_order",
            {
                "current_inventory": current_inv,
                "eoq": eoq,
                "reorder_point": reorder_point,
                "safety_stock": safety_stock,
                "demand_forecast": mean_demand,
                "risk_level": risk_level
            }
        )
        
        if rec_result.success:
            order_qty = rec_result.result["order_quantity"]
        else:
            # Final fallback
            order_qty = max(0, reorder_point - current_inv + eoq) if current_inv < reorder_point else 0
        
        return InventoryAction(
            order_quantity=order_qty,
            forecast=mean_demand,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            expected_cost=self.tools.ordering_cost if order_qty > 0 else 0
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get agent parameters"""
        return {
            "model": self.model,
            "use_extended_thinking": self.use_extended_thinking,
            "max_tool_iterations": self.max_tool_iterations,
            "fallback_mode": self.fallback_mode,
            "holding_cost": self.tools.holding_cost,
            "ordering_cost": self.tools.ordering_cost,
            "stockout_cost": self.tools.stockout_cost,
            "lead_time": self.tools.lead_time,
            "service_level": self.tools.service_level,
            "demand_stats": self.demand_stats,
            "api_available": self.client is not None
        }


if __name__ == "__main__":
    # Example usage
    print("🤖 Claude Agent Inventory Method - Example Usage")
    print("=" * 60)
    
    # Generate sample demand
    np.random.seed(42)
    t = np.arange(365)
    base = 50 + 10 * np.sin(2 * np.pi * t / 365) + 5 * np.sin(2 * np.pi * t / 7)
    demand = np.maximum(base + np.random.normal(0, 8, 365), 0)
    
    print(f"Sample demand - Mean: {np.mean(demand):.2f}, Std: {np.std(demand):.2f}")
    
    # Initialize agent (fallback mode without API key)
    agent = ClaudeAgentInventoryMethod(
        api_key=None,  # Will use fallback mode
        fallback_mode=True,
        holding_cost=2.0,
        ordering_cost=50.0,
        stockout_cost=10.0,
        service_level=0.95
    )
    
    # Fit on historical data
    print("\n📊 Fitting agent on historical data...")
    agent.fit(demand)
    print(f"Demand pattern: {agent.demand_stats.get('demand_pattern', 'unknown')}")
    print(f"Trend: {agent.demand_stats.get('trend_direction', 'unknown')}")
    print(f"Has seasonality: {agent.demand_stats.get('has_seasonality', False)}")
    
    # Create test state
    test_state = InventoryState(
        inventory_level=75.0,
        outstanding_orders=25.0,
        demand_history=demand[-30:],
        time_step=365
    )
    
    # Get recommendation
    print("\n🎯 Getting recommendation...")
    action = agent.recommend_action(test_state)
    print(f"Recommended order quantity: {action.order_quantity:.2f}")
    print(f"Forecast: {action.forecast:.2f}" if action.forecast else "")
    print(f"Reorder point: {action.reorder_point:.2f}" if action.reorder_point else "")
    print(f"Safety stock: {action.safety_stock:.2f}" if action.safety_stock else "")
    
    # Predict demand
    print("\n📈 Demand forecast...")
    forecast = agent.predict_demand(test_state, horizon=7)
    print(f"7-day forecast: {forecast}")
    
    print("\n✅ Example completed!")

