"""
Economic Order Quantity (EOQ) Method

Classic inventory optimization model that determines the optimal order quantity
that minimizes total inventory cost assuming constant demand rate.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import TraditionalMethod, InventoryState, InventoryAction


class EOQMethod(TraditionalMethod):
    """
    Economic Order Quantity implementation

    The EOQ model finds the optimal order quantity that minimizes:
    - Holding costs (inventory carrying costs)
    - Ordering costs (fixed cost per order)

    Assumptions:
    - Constant demand rate
    - Fixed lead time
    - No stockouts allowed
    - Instantaneous delivery
    """

    def __init__(self,
                 holding_cost: float = 2.0,
                 ordering_cost: float = 50.0,
                 service_level: float = 0.95,
                 lead_time: int = 7,
                 unit_cost: float = None):
        """
        Initialize EOQ method

        Args:
            holding_cost: Annual holding cost per unit (h)
            ordering_cost: Fixed cost per order (K)
            service_level: Target service level (for safety stock)
            lead_time: Lead time in periods
            unit_cost: Cost per unit (optional, for cost calculations)
        """
        super().__init__("EOQ")
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.service_level = service_level
        self.lead_time = lead_time
        self.unit_cost = unit_cost

        # Fitted parameters
        self.demand_rate = None
        self.demand_std = None
        self.optimal_order_quantity = None
        self.reorder_point = None
        self.safety_stock = None

    def fit(self,
            demand_history: np.ndarray,
            external_features: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Fit EOQ parameters based on historical demand

        Args:
            demand_history: Historical demand time series
            external_features: Not used for EOQ
            **kwargs: Additional parameters
        """
        if len(demand_history) == 0:
            raise ValueError("Demand history cannot be empty")

        # Estimate demand parameters（确保std非负）
        self.demand_rate = max(np.mean(demand_history), 0.1)  # 确保均值至少为0.1
        self.demand_std = max(np.std(demand_history), 0.01)  # 确保std至少为0.01，避免scale < 0错误

        # Calculate optimal order quantity using EOQ formula
        # Q* = sqrt(2 * D * K / h)
        self.optimal_order_quantity = np.sqrt(
            2 * self.demand_rate * self.ordering_cost / self.holding_cost
        )

        # Calculate safety stock and reorder point
        # Safety stock = Z * σ * sqrt(L)
        # where Z is the service level z-score
        from scipy.stats import norm
        z_score = norm.ppf(self.service_level)
        # 确保demand_std非负且合理
        safe_std = max(self.demand_std, 0.01)
        self.safety_stock = z_score * safe_std * np.sqrt(self.lead_time)

        # Reorder point = Lead time demand + Safety stock
        self.reorder_point = self.demand_rate * self.lead_time + self.safety_stock

        self._is_fitted = True

        # Store parameters
        self._parameters = {
            'holding_cost': self.holding_cost,
            'ordering_cost': self.ordering_cost,
            'service_level': self.service_level,
            'lead_time': self.lead_time,
            'unit_cost': self.unit_cost,
            'demand_rate': self.demand_rate,
            'demand_std': self.demand_std,
            'optimal_order_quantity': self.optimal_order_quantity,
            'reorder_point': self.reorder_point,
            'safety_stock': self.safety_stock
        }

    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        Predict future demand (constant rate assumption)

        Args:
            current_state: Current inventory state
            horizon: Number of periods ahead to predict

        Returns:
            Predicted demand for next horizon periods
        """
        if not self.is_fitted:
            raise ValueError("EOQ method must be fitted before prediction")

        # EOQ assumes constant demand rate
        return np.full(horizon, self.demand_rate)

    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend inventory action using (Q,r) policy

        The EOQ uses a fixed quantity, variable timing policy:
        - When inventory position drops to reorder point r, order quantity Q
        - Otherwise, don't order

        Args:
            current_state: Current inventory system state

        Returns:
            Recommended inventory action
        """
        if not self.is_fitted:
            raise ValueError("EOQ method must be fitted before recommendation")

        # Calculate inventory position (on-hand + on-order)
        inventory_position = (current_state.inventory_level +
                            current_state.outstanding_orders)

        # Decision rule: Order if inventory position <= reorder point
        if inventory_position <= self.reorder_point:
            order_quantity = self.optimal_order_quantity
        else:
            order_quantity = 0.0

        # Calculate expected cost
        expected_cost = (
            self.holding_cost * (inventory_position + order_quantity) / 2 +
            (self.ordering_cost if order_quantity > 0 else 0)
        )

        return InventoryAction(
            order_quantity=order_quantity,
            reorder_point=self.reorder_point,
            safety_stock=self.safety_stock,
            expected_cost=expected_cost
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Get EOQ method parameters"""
        return self._parameters.copy() if hasattr(self, '_parameters') else {}

    def get_policy_description(self) -> str:
        """Get human-readable description of the policy"""
        if not self.is_fitted:
            return "EOQ method (not fitted)"

        return (f"EOQ Policy: Order {self.optimal_order_quantity:.1f} units "
                f"when inventory position ≤ {self.reorder_point:.1f} "
                f"(Safety stock: {self.safety_stock:.1f})")


class EOQWithBackorders(EOQMethod):
    """
    EOQ variant that allows backorders (stockouts)

    This variant considers the cost of stockouts and allows intentional
    stockouts when it's economically optimal.
    """

    def __init__(self,
                 holding_cost: float = 2.0,
                 ordering_cost: float = 50.0,
                 backorder_cost: float = 5.0,
                 **kwargs):
        """
        Initialize EOQ with backorders

        Args:
            holding_cost: Holding cost per unit per period
            ordering_cost: Fixed cost per order
            backorder_cost: Cost per unit backordered per period
        """
        super().__init__(holding_cost, ordering_cost, **kwargs)
        self.backorder_cost = backorder_cost
        self._method_name = "EOQ_with_Backorders"

    def fit(self, demand_history: np.ndarray, **kwargs) -> None:
        """Fit EOQ with backorders model"""
        super().fit(demand_history, **kwargs)

        # Adjust optimal order quantity for backorders
        # Q* = sqrt(2*D*K/h * (h+b)/b)
        # where b is backorder cost
        backorder_factor = ((self.holding_cost + self.backorder_cost) /
                           self.backorder_cost)

        self.optimal_order_quantity = (
            self.optimal_order_quantity * np.sqrt(backorder_factor)
        )

        # Adjust reorder point (can be negative with backorders)
        self.reorder_point = (self.demand_rate * self.lead_time -
                             self.optimal_order_quantity *
                             self.backorder_cost /
                             (self.holding_cost + self.backorder_cost))

        # Update parameters
        self._parameters.update({
            'backorder_cost': self.backorder_cost,
            'optimal_order_quantity': self.optimal_order_quantity,
            'reorder_point': self.reorder_point
        })


# Helper functions for EOQ analysis
def calculate_eoq_costs(demand_rate: float,
                       order_quantity: float,
                       holding_cost: float,
                       ordering_cost: float) -> Dict[str, float]:
    """
    Calculate total costs for given EOQ parameters

    Returns:
        Dictionary with cost breakdown
    """
    # Average inventory = Q/2 (assuming no safety stock)
    avg_inventory = order_quantity / 2

    # Annual holding cost = h * Q/2
    annual_holding_cost = holding_cost * avg_inventory

    # Annual ordering cost = K * D/Q
    annual_ordering_cost = ordering_cost * demand_rate / order_quantity

    total_cost = annual_holding_cost + annual_ordering_cost

    return {
        'total_cost': total_cost,
        'holding_cost': annual_holding_cost,
        'ordering_cost': annual_ordering_cost,
        'average_inventory': avg_inventory,
        'order_frequency': demand_rate / order_quantity
    }


def sensitivity_analysis(demand_rate: float,
                        holding_cost: float,
                        ordering_cost: float,
                        quantity_range: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Perform sensitivity analysis around optimal EOQ

    Args:
        demand_rate: Annual demand rate
        holding_cost: Holding cost per unit per year
        ordering_cost: Fixed cost per order
        quantity_range: Range of order quantities to analyze

    Returns:
        Dictionary with cost components for each quantity
    """
    results = {
        'quantities': quantity_range,
        'total_costs': np.zeros_like(quantity_range),
        'holding_costs': np.zeros_like(quantity_range),
        'ordering_costs': np.zeros_like(quantity_range)
    }

    for i, q in enumerate(quantity_range):
        costs = calculate_eoq_costs(demand_rate, q, holding_cost, ordering_cost)
        results['total_costs'][i] = costs['total_cost']
        results['holding_costs'][i] = costs['holding_cost']
        results['ordering_costs'][i] = costs['ordering_cost']

    return results


if __name__ == "__main__":
    # Example usage and testing
    print("🏪 EOQ Method - Example Usage")
    print("=" * 40)

    # Generate sample demand data
    np.random.seed(42)
    demand_history = np.random.normal(50, 10, 365)  # Daily demand for a year
    demand_history = np.maximum(demand_history, 0)  # Ensure non-negative

    print(f"Sample demand statistics:")
    print(f"  Mean: {np.mean(demand_history):.2f}")
    print(f"  Std: {np.std(demand_history):.2f}")

    # Initialize and fit EOQ method
    eoq = EOQMethod(holding_cost=2.0, ordering_cost=50.0)
    eoq.fit(demand_history)

    print(f"\nEOQ Results:")
    print(f"  {eoq.get_policy_description()}")

    # Test recommendation
    current_state = InventoryState(
        inventory_level=30.0,
        outstanding_orders=0.0,
        demand_history=demand_history[-30:],  # Last 30 days
        time_step=365
    )

    action = eoq.recommend_action(current_state)
    print(f"\nRecommendation for current inventory {current_state.inventory_level}:")
    print(f"  Order quantity: {action.order_quantity:.1f}")

    # Cost analysis
    params = eoq.get_parameters()
    costs = calculate_eoq_costs(
        params['demand_rate'],
        params['optimal_order_quantity'],
        params['holding_cost'],
        params['ordering_cost']
    )

    print(f"\nCost Analysis:")
    print(f"  Total annual cost: ${costs['total_cost']:.2f}")
    print(f"  Holding cost: ${costs['holding_cost']:.2f}")
    print(f"  Ordering cost: ${costs['ordering_cost']:.2f}")
    print(f"  Order frequency: {costs['order_frequency']:.1f} times/year")