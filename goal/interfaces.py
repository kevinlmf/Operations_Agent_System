"""
Core interfaces for the inventory optimization system

This module defines the base interfaces that all methods must implement,
ensuring consistency and interoperability across different approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from enum import Enum


class MethodCategory(Enum):
    """Categories of inventory optimization methods"""
    TRADITIONAL = "traditional"
    MACHINE_LEARNING = "ml"
    REINFORCEMENT_LEARNING = "rl"
    HYBRID = "hybrid"


@dataclass
class InventoryState:
    """Represents the current state of inventory system"""
    inventory_level: float          # Current inventory on hand
    outstanding_orders: float       # Orders in transit
    demand_history: np.ndarray     # Historical demand data
    time_step: int                 # Current time period
    external_features: Optional[Dict[str, float]] = None  # Weather, promotions, etc.


@dataclass
class InventoryAction:
    """Represents an inventory management action"""
    order_quantity: float          # How much to order
    order_timing: Optional[int] = None  # When to place order (for advanced methods)
    forecast: Optional[float] = None  # Demand forecast (for ML methods)
    expected_cost: Optional[float] = None  # Expected cost (for traditional methods)
    reorder_point: Optional[float] = None  # Reorder point (for traditional methods)
    safety_stock: Optional[float] = None  # Safety stock level (for traditional methods)


@dataclass
class InventoryResult:
    """Results from inventory management simulation"""
    total_cost: float
    holding_cost: float
    stockout_cost: float
    ordering_cost: float
    service_level: float
    inventory_turnover: float
    forecast_accuracy: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


class InventoryMethod(ABC):
    """Base interface for all inventory optimization methods"""

    def __init__(self, method_name: str, category: MethodCategory):
        self._method_name = method_name
        self._category = category
        self._is_fitted = False
        self._parameters = {}

    @property
    def method_name(self) -> str:
        """Return method name for identification"""
        return self._method_name

    @property
    def category(self) -> MethodCategory:
        """Return method category"""
        return self._category

    @property
    def is_fitted(self) -> bool:
        """Check if method has been trained/fitted"""
        return self._is_fitted

    @abstractmethod
    def fit(self,
            demand_history: np.ndarray,
            external_features: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Train/fit the method on historical data

        Args:
            demand_history: Historical demand time series
            external_features: Optional external factors
            **kwargs: Method-specific parameters
        """
        pass

    @abstractmethod
    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        Predict future demand

        Args:
            current_state: Current inventory state
            horizon: Number of periods to predict ahead

        Returns:
            Predicted demand for next `horizon` periods
        """
        pass

    @abstractmethod
    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend inventory action given current state

        Args:
            current_state: Current inventory system state

        Returns:
            Recommended action (order quantity, timing)
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters/hyperparameters"""
        pass

    def set_parameters(self, **params) -> None:
        """Set method parameters"""
        self._parameters.update(params)

    def get_info(self) -> Dict[str, Any]:
        """Get method information"""
        return {
            'name': self.method_name,
            'category': self.category.value,
            'is_fitted': self.is_fitted,
            'parameters': self.get_parameters()
        }

    def decide(self, current_state: InventoryState) -> InventoryAction:
        """Alias for recommend_action for backward compatibility"""
        return self.recommend_action(current_state)


class TraditionalMethod(InventoryMethod):
    """Base class for traditional inventory methods"""

    def __init__(self, method_name: str):
        super().__init__(method_name, MethodCategory.TRADITIONAL)


class MLMethod(InventoryMethod):
    """Base class for machine learning methods"""

    def __init__(self, method_name: str):
        super().__init__(method_name, MethodCategory.MACHINE_LEARNING)
        self.model = None

    @abstractmethod
    def build_model(self) -> Any:
        """Build the ML model architecture"""
        pass

    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ML model"""
        pass


class RLMethod(InventoryMethod):
    """Base class for reinforcement learning methods"""

    def __init__(self, method_name: str):
        super().__init__(method_name, MethodCategory.REINFORCEMENT_LEARNING)
        self.agent = None
        self.environment = None

    @abstractmethod
    def build_agent(self) -> Any:
        """Build the RL agent"""
        pass

    @abstractmethod
    def train_agent(self, num_episodes: int) -> None:
        """Train the RL agent"""
        pass


class InventoryEvaluator:
    """Unified evaluation framework for all inventory methods"""

    def __init__(self,
                 holding_cost: float = 2.0,
                 stockout_cost: float = 10.0,
                 ordering_cost: float = 50.0):
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost

    def evaluate_method(self,
                       method: InventoryMethod,
                       test_states: List[InventoryState],
                       true_demands: np.ndarray,
                       **kwargs) -> InventoryResult:
        """
        Evaluate method performance on test data

        Args:
            method: The inventory method to evaluate
            test_states: List of test states
            true_demands: Actual demand realizations
            **kwargs: Additional evaluation parameters

        Returns:
            InventoryResult with performance metrics
        """
        if not method.is_fitted:
            raise ValueError(f"Method {method.method_name} must be fitted before evaluation")

        total_costs = []
        service_levels = []
        inventory_levels = []
        forecast_errors = []

        for i, (state, actual_demand) in enumerate(zip(test_states, true_demands)):
            # Get method recommendation
            action = method.recommend_action(state)
            predicted_demand = method.predict_demand(state, horizon=1)[0]

            # Simulate inventory dynamics
            cost_breakdown, service_level, final_inventory = self._simulate_period(
                state, action, actual_demand
            )

            total_costs.append(cost_breakdown)
            service_levels.append(service_level)
            inventory_levels.append(final_inventory)

            # Forecast accuracy
            forecast_error = abs(predicted_demand - actual_demand)
            forecast_errors.append(forecast_error)

        # Aggregate results
        return InventoryResult(
            total_cost=np.mean([c[0] for c in total_costs]),
            holding_cost=np.mean([c[1] for c in total_costs]),
            stockout_cost=np.mean([c[2] for c in total_costs]),
            ordering_cost=np.mean([c[3] for c in total_costs]),
            service_level=np.mean(service_levels),
            inventory_turnover=self._calculate_inventory_turnover(inventory_levels, true_demands),
            forecast_accuracy=1.0 - (np.mean(forecast_errors) / np.mean(true_demands))
        )

    def _simulate_period(self,
                        state: InventoryState,
                        action: InventoryAction,
                        actual_demand: float) -> Tuple[Tuple[float, float, float, float], float, float]:
        """Simulate one period of inventory dynamics"""

        # Calculate costs
        holding_cost = self.holding_cost * max(0, state.inventory_level)
        stockout_cost = self.stockout_cost * max(0, actual_demand - state.inventory_level)
        ordering_cost = self.ordering_cost * (1 if action.order_quantity > 0 else 0)

        total_cost = holding_cost + stockout_cost + ordering_cost

        # Service level (1 if no stockout, 0 otherwise)
        service_level = 1.0 if state.inventory_level >= actual_demand else 0.0

        # Final inventory after demand
        final_inventory = max(0, state.inventory_level - actual_demand)

        # Return cost breakdown as tuple: (total_cost, holding_cost, stockout_cost, ordering_cost)
        cost_breakdown = (total_cost, holding_cost, stockout_cost, ordering_cost)

        return (cost_breakdown, service_level, final_inventory)

    def _calculate_inventory_turnover(self,
                                    inventory_levels: List[float],
                                    demands: np.ndarray) -> float:
        """Calculate inventory turnover ratio"""
        total_sales = np.sum(demands)
        avg_inventory = np.mean(inventory_levels)
        return total_sales / avg_inventory if avg_inventory > 0 else 0.0


# Backward compatibility alias
BaseMethod = InventoryMethod


class ComparisonFramework:
    """Framework for comparing multiple inventory methods"""

    def __init__(self, evaluator: InventoryEvaluator):
        self.evaluator = evaluator
        self.results = {}

    def add_method(self, method: InventoryMethod) -> None:
        """Add a method to the comparison"""
        if not method.is_fitted:
            print(f"Warning: Method {method.method_name} is not fitted")

        self.methods = getattr(self, 'methods', [])
        self.methods.append(method)

    def run_comparison(self,
                      test_states: List[InventoryState],
                      true_demands: np.ndarray) -> Dict[str, InventoryResult]:
        """Run comparison across all methods"""

        results = {}

        for method in getattr(self, 'methods', []):
            print(f"Evaluating {method.method_name}...")

            try:
                result = self.evaluator.evaluate_method(method, test_states, true_demands)
                results[method.method_name] = result
                print(f"✅ {method.method_name} completed")

            except Exception as e:
                print(f"❌ {method.method_name} failed: {e}")
                continue

        self.results = results
        return results

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary comparison table"""

        if not self.results:
            return {}

        summary = {}

        for method_name, result in self.results.items():
            summary[method_name] = {
                'Total Cost': result.total_cost,
                'Service Level': result.service_level,
                'Inventory Turnover': result.inventory_turnover,
                'Forecast Accuracy': result.forecast_accuracy or 0.0
            }

        return summary

    def get_best_method(self, metric: str = 'total_cost') -> Tuple[str, float]:
        """Get best performing method by specified metric"""

        if not self.results:
            return None, None

        if metric == 'total_cost':
            best_method = min(self.results.items(), key=lambda x: x[1].total_cost)
        elif metric == 'service_level':
            best_method = max(self.results.items(), key=lambda x: x[1].service_level)
        elif metric == 'forecast_accuracy':
            best_method = max(self.results.items(), key=lambda x: x[1].forecast_accuracy or 0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return best_method[0], getattr(best_method[1], metric)