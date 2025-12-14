"""
(s,S) Policy Implementation

Two-parameter inventory control policy where:
- s: reorder point (when to order)
- S: order-up-to level (how much to stock)

When inventory position drops to or below s, order enough to bring
inventory position up to S.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path
from scipy import stats, optimize

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.interfaces import TraditionalMethod, InventoryState, InventoryAction


class sSPolicyMethod(TraditionalMethod):
    """
    (s,S) Policy inventory management method

    Optimal two-parameter policy for periodic review systems with:
    - Fixed ordering cost
    - Linear holding and shortage costs
    - Stochastic demand
    """

    def __init__(self,
                 holding_cost: float = 2.0,
                 shortage_cost: float = None,
                 ordering_cost: float = 50.0,
                 lead_time: int = 7,
                 review_period: int = 1,
                 optimization_method: str = "analytical",
                 stockout_cost: float = None,
                 min_order_quantity: float = None):
        """
        Initialize (s,S) policy method

        Args:
            holding_cost: Holding cost per unit per period
            shortage_cost: Shortage cost per unit per period (same as stockout_cost)
            ordering_cost: Fixed cost per order
            lead_time: Lead time in periods
            review_period: Review period in periods
            optimization_method: 'analytical' or 'numerical' optimization
            stockout_cost: Alias for shortage_cost (for backward compatibility)
            min_order_quantity: Minimum order quantity (optional, for display)
        """
        super().__init__("s_S_Policy")
        self.holding_cost = holding_cost
        # Support both shortage_cost and stockout_cost parameter names
        if shortage_cost is None and stockout_cost is not None:
            self.shortage_cost = stockout_cost
        elif shortage_cost is not None:
            self.shortage_cost = shortage_cost
        else:
            self.shortage_cost = 10.0  # default
        self.ordering_cost = ordering_cost
        self.lead_time = lead_time
        self.review_period = review_period
        self.optimization_method = optimization_method
        self.min_order_quantity = min_order_quantity

        # Fitted parameters
        self.demand_mean = None
        self.demand_std = None
        self.s = None  # reorder point
        self.S = None  # order-up-to level
        self.protection_period = lead_time + review_period

    def fit(self,
            demand_history: np.ndarray,
            external_features: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Fit (s,S) policy parameters

        Args:
            demand_history: Historical demand time series
            external_features: Not used for (s,S) policy
            **kwargs: Additional parameters
        """
        if len(demand_history) == 0:
            raise ValueError("Demand history cannot be empty")

        # Estimate demand parameters
        self.demand_mean = np.mean(demand_history)
        self.demand_std = np.std(demand_history)

        # Optimize (s,S) parameters
        if self.optimization_method == "analytical":
            self.s, self.S = self._optimize_analytical()
        elif self.optimization_method == "numerical":
            self.s, self.S = self._optimize_numerical()
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

        self._is_fitted = True

        # Store parameters
        self._parameters = {
            'holding_cost': self.holding_cost,
            'shortage_cost': self.shortage_cost,
            'ordering_cost': self.ordering_cost,
            'lead_time': self.lead_time,
            'review_period': self.review_period,
            'protection_period': self.protection_period,
            'demand_mean': self.demand_mean,
            'demand_std': self.demand_std,
            's_reorder_point': self.s,
            'S_order_up_to': self.S,
            'optimization_method': self.optimization_method
        }

    def _optimize_analytical(self) -> Tuple[float, float]:
        """
        Analytical approximation for (s,S) optimization

        Uses approximations from inventory theory literature
        """
        # Expected demand during protection period
        mu_L = self.demand_mean * self.protection_period
        sigma_L = self.demand_std * np.sqrt(self.protection_period)

        # Approximate optimal S using EOQ-like formula
        # S ≈ mu_L + safety_stock
        h = self.holding_cost
        p = self.shortage_cost
        K = self.ordering_cost

        # Service level approximation
        service_level = p / (p + h)
        z_alpha = stats.norm.ppf(service_level)
        safety_stock = z_alpha * sigma_L

        S = mu_L + safety_stock

        # Approximate optimal s
        # For high service levels, s ≈ S - EOQ/2
        # where EOQ is modified for stochastic demand
        D_annual = self.demand_mean * 365  # Annualized demand
        eoq_approx = np.sqrt(2 * D_annual * K / h)

        # Scale EOQ to review period
        eoq_period = eoq_approx * self.review_period / 365

        s = S - eoq_period / 2

        # Ensure s is reasonable
        s = max(s, mu_L - 2 * sigma_L)  # At least 2 std devs below mean
        s = min(s, S - 1)  # At least 1 unit below S

        return float(s), float(S)

    def _optimize_numerical(self) -> Tuple[float, float]:
        """
        Numerical optimization for (s,S) parameters

        Uses scipy.optimize to minimize expected total cost
        """
        mu_L = self.demand_mean * self.protection_period
        sigma_L = self.demand_std * np.sqrt(self.protection_period)

        # Initial guess using analytical method
        s_init, S_init = self._optimize_analytical()

        def total_cost(params):
            s, S = params
            if s >= S:  # Invalid policy
                return 1e6

            return self._expected_total_cost(s, S, mu_L, sigma_L)

        # Bounds: s and S should be positive, S > s
        bounds = [(0, mu_L + 3*sigma_L), (1, mu_L + 4*sigma_L)]

        # Constraint: S > s
        constraints = {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.1}

        result = optimize.minimize(
            total_cost,
            [s_init, S_init],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return float(result.x[0]), float(result.x[1])
        else:
            # Fall back to analytical solution
            return self._optimize_analytical()

    def _expected_total_cost(self, s: float, S: float, mu_L: float, sigma_L: float) -> float:
        """
        Calculate expected total cost for given (s,S) policy

        Includes holding cost, shortage cost, and ordering cost
        """
        # Expected holding cost
        expected_inventory = (S + s) / 2
        holding_cost = self.holding_cost * expected_inventory

        # Expected shortage cost (approximation)
        # This is a simplified calculation - exact calculation is complex
        shortage_prob = 1 - stats.norm.cdf(s, mu_L, sigma_L)
        expected_shortage = sigma_L * stats.norm.pdf(stats.norm.ppf(1 - shortage_prob))
        shortage_cost = self.shortage_cost * expected_shortage * shortage_prob

        # Expected ordering cost
        # Order frequency depends on demand rate and order quantity
        avg_order_quantity = S - s
        order_frequency = self.demand_mean / avg_order_quantity if avg_order_quantity > 0 else 0
        ordering_cost = self.ordering_cost * order_frequency

        return holding_cost + shortage_cost + ordering_cost

    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        Predict future demand (uses fitted mean)

        Args:
            current_state: Current inventory state
            horizon: Number of periods ahead to predict

        Returns:
            Predicted demand for next horizon periods
        """
        if not self.is_fitted:
            raise ValueError("(s,S) policy must be fitted before prediction")

        return np.full(horizon, self.demand_mean)

    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend inventory action using (s,S) policy

        Policy rule:
        - If inventory position <= s: order up to S
        - Otherwise: don't order

        Args:
            current_state: Current inventory system state

        Returns:
            Recommended inventory action
        """
        if not self.is_fitted:
            raise ValueError("(s,S) policy must be fitted before recommendation")

        # Calculate inventory position
        inventory_position = (current_state.inventory_level +
                            current_state.outstanding_orders)

        # (s,S) policy decision rule
        if inventory_position <= self.s:
            order_quantity = max(0, self.S - inventory_position)
        else:
            order_quantity = 0.0

        return InventoryAction(order_quantity=order_quantity)

    def get_parameters(self) -> Dict[str, Any]:
        """Get (s,S) policy parameters"""
        return self._parameters.copy() if hasattr(self, '_parameters') else {}

    def get_policy_description(self) -> str:
        """Get human-readable description of the policy"""
        if not self.is_fitted:
            return "(s,S) policy (not fitted)"

        return (f"(s,S) Policy: s={self.s:.1f}, S={self.S:.1f} "
                f"(Order up to {self.S:.1f} when inventory ≤ {self.s:.1f})")

    def calculate_policy_metrics(self) -> Dict[str, float]:
        """Calculate key policy performance metrics"""
        if not self.is_fitted:
            raise ValueError("Policy must be fitted first")

        # Average order quantity
        avg_order_quantity = self.S - self.s

        # Average inventory level
        avg_inventory = (self.S + self.s) / 2

        # Order frequency (orders per time unit)
        order_frequency = self.demand_mean / avg_order_quantity

        # Inventory turnover
        inventory_turnover = self.demand_mean / avg_inventory

        # Safety stock
        expected_demand = self.demand_mean * self.protection_period
        safety_stock = self.s - expected_demand

        return {
            'average_order_quantity': avg_order_quantity,
            'average_inventory': avg_inventory,
            'order_frequency': order_frequency,
            'inventory_turnover': inventory_turnover,
            'safety_stock': safety_stock,
            'fill_rate_estimate': self._estimate_fill_rate()
        }

    def _estimate_fill_rate(self) -> float:
        """Estimate fill rate (service level) of the policy"""
        mu_L = self.demand_mean * self.protection_period
        sigma_L = self.demand_std * np.sqrt(self.protection_period)

        # Probability of no stockout
        fill_rate = stats.norm.cdf(self.s, mu_L, sigma_L)
        return max(0, min(1, fill_rate))


class AdaptivesSPolicy(sSPolicyMethod):
    """
    Adaptive (s,S) policy that adjusts parameters based on recent performance

    Monitors actual performance and adjusts s and S parameters to maintain
    target service levels and cost efficiency.
    """

    def __init__(self,
                 target_service_level: float = 0.95,
                 adaptation_period: int = 30,
                 **kwargs):
        """
        Initialize adaptive (s,S) policy

        Args:
            target_service_level: Target service level to maintain
            adaptation_period: How often to recalibrate parameters
        """
        super().__init__(**kwargs)
        self.target_service_level = target_service_level
        self.adaptation_period = adaptation_period
        self.last_adaptation = 0
        self._method_name = "Adaptive_s_S_Policy"

    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend action with adaptive parameter adjustment
        """
        if not self.is_fitted:
            raise ValueError("Adaptive (s,S) policy must be fitted before recommendation")

        # Check if it's time to adapt parameters
        if (current_state.time_step - self.last_adaptation >= self.adaptation_period and
            len(current_state.demand_history) >= self.adaptation_period):

            self._adapt_parameters(current_state.demand_history)
            self.last_adaptation = current_state.time_step

        # Use standard (s,S) recommendation
        return super().recommend_action(current_state)

    def _adapt_parameters(self, recent_demand: np.ndarray):
        """
        Adapt s and S parameters based on recent performance

        Args:
            recent_demand: Recent demand history
        """
        # Use only recent demand for parameter updates
        recent_window = recent_demand[-self.adaptation_period:]

        # Update demand estimates
        recent_mean = np.mean(recent_window)
        recent_std = np.std(recent_window)

        # Smooth the updates
        alpha = 0.2  # Smoothing parameter
        self.demand_mean = alpha * recent_mean + (1 - alpha) * self.demand_mean
        self.demand_std = alpha * recent_std + (1 - alpha) * self.demand_std

        # Re-optimize parameters with updated demand estimates
        self.s, self.S = self._optimize_analytical()


def compare_s_S_methods(demand_history: np.ndarray,
                       cost_params: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare different (s,S) optimization approaches

    Args:
        demand_history: Historical demand data
        cost_params: Dictionary with cost parameters

    Returns:
        Comparison results for different methods
    """
    methods = ['analytical', 'numerical']
    results = {}

    for method in methods:
        policy = sSPolicyMethod(
            holding_cost=cost_params.get('holding_cost', 2.0),
            shortage_cost=cost_params.get('shortage_cost', 10.0),
            ordering_cost=cost_params.get('ordering_cost', 50.0),
            optimization_method=method
        )

        try:
            policy.fit(demand_history)
            params = policy.get_parameters()
            metrics = policy.calculate_policy_metrics()

            results[method] = {
                's': params['s_reorder_point'],
                'S': params['S_order_up_to'],
                'metrics': metrics,
                'description': policy.get_policy_description()
            }

        except Exception as e:
            results[method] = {'error': str(e)}

    return results


if __name__ == "__main__":
    # Example usage and testing
    print("📊 (s,S) Policy Method - Example Usage")
    print("=" * 40)

    # Generate sample demand data
    np.random.seed(42)
    demand_history = np.random.gamma(2, 25, 365)  # Realistic demand pattern

    print(f"Sample demand statistics:")
    print(f"  Mean: {np.mean(demand_history):.2f}")
    print(f"  Std: {np.std(demand_history):.2f}")
    print(f"  CV: {np.std(demand_history)/np.mean(demand_history):.3f}")

    # Test (s,S) policy
    cost_params = {
        'holding_cost': 2.0,
        'shortage_cost': 10.0,
        'ordering_cost': 50.0
    }

    print(f"\n📈 (s,S) Policy Optimization:")
    comparison = compare_s_S_methods(demand_history, cost_params)

    for method, results in comparison.items():
        if 'error' not in results:
            print(f"\n{method.upper()} Method:")
            print(f"  {results['description']}")
            print(f"  Avg Order Qty: {results['metrics']['average_order_quantity']:.1f}")
            print(f"  Avg Inventory: {results['metrics']['average_inventory']:.1f}")
            print(f"  Order Frequency: {results['metrics']['order_frequency']:.2f}/period")
            print(f"  Estimated Fill Rate: {results['metrics']['fill_rate_estimate']:.1%}")

    # Test adaptive policy
    print(f"\n🔄 Adaptive (s,S) Policy:")
    adaptive_policy = AdaptivesSPolicy(target_service_level=0.95)
    adaptive_policy.fit(demand_history[:300])

    # Simulate some time steps
    for t in range(300, 310):
        current_state = InventoryState(
            inventory_level=50.0,
            outstanding_orders=20.0,
            demand_history=demand_history[max(0, t-30):t],
            time_step=t
        )

        action = adaptive_policy.recommend_action(current_state)
        if action.order_quantity > 0:
            print(f"  Time {t}: Order {action.order_quantity:.1f} units")

    # Final policy state
    final_params = adaptive_policy.get_parameters()
    print(f"\n  Final policy: s={final_params['s_reorder_point']:.1f}, "
          f"S={final_params['S_order_up_to']:.1f}")