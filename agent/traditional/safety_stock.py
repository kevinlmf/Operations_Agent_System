"""
Safety Stock Methods

Various approaches to calculate safety stock to buffer against demand uncertainty
while maintaining target service levels.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
from pathlib import Path
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import TraditionalMethod, InventoryState, InventoryAction


class SafetyStockMethod(TraditionalMethod):
    """
    Safety Stock inventory management method

    Uses statistical models to determine appropriate buffer stock levels
    to achieve target service levels under demand uncertainty.
    """

    def __init__(self,
                 service_level: float = 0.95,
                 lead_time: int = 7,
                 review_period: int = 1,
                 method: str = "normal",
                 holding_cost: float = None,
                 ordering_cost: float = None):
        """
        Initialize Safety Stock method

        Args:
            service_level: Target service level (e.g., 0.95 = 95%)
            lead_time: Lead time in periods
            review_period: Review period (how often to check inventory)
            method: Statistical method ('normal', 'poisson', 'empirical')
            holding_cost: Cost to hold one unit for one period (optional)
            ordering_cost: Fixed cost per order (optional)
        """
        super().__init__("SafetyStock")
        self.service_level = service_level
        self.lead_time = lead_time
        self.review_period = review_period
        self.method = method
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost

        # Fitted parameters
        self.demand_mean = None
        self.demand_std = None
        self.safety_stock = None
        self.reorder_point = None
        self.target_stock_level = None

    def fit(self,
            demand_history: np.ndarray,
            external_features: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Fit safety stock parameters based on historical demand

        Args:
            demand_history: Historical demand time series
            external_features: Not used for safety stock
            **kwargs: Additional parameters
        """
        if len(demand_history) == 0:
            raise ValueError("Demand history cannot be empty")

        # Estimate demand parameters（确保std非负）
        self.demand_mean = max(np.mean(demand_history), 0.1)  # 确保均值至少为0.1
        self.demand_std = max(np.std(demand_history), 0.01)  # 确保std至少为0.01，避免scale < 0错误

        # Calculate safety stock based on chosen method
        if self.method == "normal":
            self.safety_stock = self._calculate_normal_safety_stock()
        elif self.method == "poisson":
            self.safety_stock = self._calculate_poisson_safety_stock()
        elif self.method == "empirical":
            self.safety_stock = self._calculate_empirical_safety_stock(demand_history)
        elif self.method == "adaptive":
            self.safety_stock = self._calculate_normal_safety_stock()  # Start with normal
        else:
            raise ValueError(f"Unknown safety stock method: {self.method}")

        # Calculate reorder point and target stock level
        self.reorder_point = (self.demand_mean * self.lead_time +
                             self.safety_stock)

        # For periodic review: target = demand during (L+R) + safety stock
        self.target_stock_level = (self.demand_mean *
                                  (self.lead_time + self.review_period) +
                                  self.safety_stock)

        self._is_fitted = True

        # Store parameters
        self._parameters = {
            'service_level': self.service_level,
            'lead_time': self.lead_time,
            'review_period': self.review_period,
            'method': self.method,
            'holding_cost': self.holding_cost,
            'ordering_cost': self.ordering_cost,
            'demand_mean': self.demand_mean,
            'demand_std': self.demand_std,
            'safety_stock': self.safety_stock,
            'reorder_point': self.reorder_point,
            'target_stock_level': self.target_stock_level
        }

    def _calculate_normal_safety_stock(self) -> float:
        """Calculate safety stock assuming normal demand distribution"""
        z_score = stats.norm.ppf(self.service_level)

        # Safety stock = Z * σ * sqrt(L + R)
        # where L is lead time, R is review period
        protection_period = self.lead_time + self.review_period
        
        # 确保demand_std非负且合理
        safe_std = max(self.demand_std, 0.01)
        safety_stock = z_score * safe_std * np.sqrt(protection_period)

        return max(0, safety_stock)

    def _calculate_poisson_safety_stock(self) -> float:
        """Calculate safety stock assuming Poisson demand distribution"""
        # For Poisson distribution, mean = variance
        lambda_param = self.demand_mean
        protection_period = self.lead_time + self.review_period

        # Expected demand during protection period
        expected_demand = lambda_param * protection_period

        # Find the demand level that achieves target service level
        target_demand = stats.poisson.ppf(self.service_level, expected_demand)

        safety_stock = max(0, target_demand - expected_demand)
        return safety_stock

    def _calculate_empirical_safety_stock(self, demand_history: np.ndarray) -> float:
        """Calculate safety stock using empirical demand distribution"""
        if len(demand_history) < self.lead_time + self.review_period:
            # Fall back to normal method if insufficient data
            return self._calculate_normal_safety_stock()

        # Calculate rolling demand over protection period
        protection_period = self.lead_time + self.review_period
        rolling_demands = []

        for i in range(len(demand_history) - protection_period + 1):
            period_demand = np.sum(demand_history[i:i + protection_period])
            rolling_demands.append(period_demand)

        rolling_demands = np.array(rolling_demands)

        # Safety stock = Service level quantile - mean demand
        target_demand = np.quantile(rolling_demands, self.service_level)
        mean_period_demand = np.mean(rolling_demands)

        safety_stock = max(0, target_demand - mean_period_demand)
        return safety_stock

    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        Predict future demand (uses historical average)

        Args:
            current_state: Current inventory state
            horizon: Number of periods ahead to predict

        Returns:
            Predicted demand for next horizon periods
        """
        if not self.is_fitted:
            raise ValueError("Safety Stock method must be fitted before prediction")

        return np.full(horizon, self.demand_mean)

    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend inventory action using safety stock policy

        Uses (s,S) policy where:
        - s = reorder point
        - S = target stock level

        Args:
            current_state: Current inventory system state

        Returns:
            Recommended inventory action
        """
        if not self.is_fitted:
            raise ValueError("Safety Stock method must be fitted before recommendation")

        # Calculate inventory position
        inventory_position = (current_state.inventory_level +
                            current_state.outstanding_orders)

        # (s,S) policy: if inventory position <= s, order up to S
        if inventory_position <= self.reorder_point:
            order_quantity = max(0, self.target_stock_level - inventory_position)
        else:
            order_quantity = 0.0

        return InventoryAction(
            order_quantity=order_quantity,
            reorder_point=self.reorder_point,
            safety_stock=self.safety_stock
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Get Safety Stock method parameters"""
        return self._parameters.copy() if hasattr(self, '_parameters') else {}

    def calculate_service_level_achieved(self, test_demands: np.ndarray) -> float:
        """
        Calculate actual service level achieved with current safety stock

        Args:
            test_demands: Test demand data

        Returns:
            Achieved service level (fraction of periods with no stockout)
        """
        if not self.is_fitted:
            raise ValueError("Method must be fitted first")

        stockout_events = 0
        total_periods = len(test_demands) - self.lead_time - self.review_period + 1

        for i in range(total_periods):
            period_demand = np.sum(test_demands[i:i + self.lead_time + self.review_period])
            expected_demand = self.demand_mean * (self.lead_time + self.review_period)

            if period_demand > expected_demand + self.safety_stock:
                stockout_events += 1

        return 1.0 - (stockout_events / total_periods) if total_periods > 0 else 1.0


class AdaptiveSafetyStock(SafetyStockMethod):
    """
    Adaptive Safety Stock that adjusts based on recent demand patterns

    This variant updates safety stock levels based on rolling windows
    of recent demand to adapt to changing patterns.
    """

    def __init__(self,
                 service_level: float = 0.95,
                 lead_time: int = 7,
                 review_period: int = 1,
                 window_size: int = 30,
                 adaptation_rate: float = 0.1):
        """
        Initialize Adaptive Safety Stock method

        Args:
            service_level: Target service level
            lead_time: Lead time in periods
            review_period: Review period
            window_size: Size of rolling window for adaptation
            adaptation_rate: Rate of adaptation (0-1)
        """
        super().__init__(service_level, lead_time, review_period, method="adaptive")
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self._method_name = "AdaptiveSafetyStock"

    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend action with adaptive safety stock

        Updates safety stock based on recent demand patterns
        """
        if not self.is_fitted:
            raise ValueError("Adaptive Safety Stock method must be fitted before recommendation")

        # Update safety stock based on recent demand
        if len(current_state.demand_history) >= self.window_size:
            recent_demand = current_state.demand_history[-self.window_size:]
            recent_std = np.std(recent_demand)

            # Adapt safety stock based on recent volatility
            z_score = stats.norm.ppf(self.service_level)
            protection_period = self.lead_time + self.review_period
            new_safety_stock = z_score * recent_std * np.sqrt(protection_period)

            # Smooth adaptation
            self.safety_stock = ((1 - self.adaptation_rate) * self.safety_stock +
                               self.adaptation_rate * new_safety_stock)

            # Update reorder point and target stock level
            recent_mean = np.mean(recent_demand)
            self.reorder_point = recent_mean * self.lead_time + self.safety_stock
            self.target_stock_level = (recent_mean * (self.lead_time + self.review_period) +
                                     self.safety_stock)

        # Use standard recommendation logic
        return super().recommend_action(current_state)


def compare_safety_stock_methods(demand_history: np.ndarray,
                               service_level: float = 0.95,
                               lead_time: int = 7) -> Dict[str, Dict[str, float]]:
    """
    Compare different safety stock calculation methods

    Args:
        demand_history: Historical demand data
        service_level: Target service level
        lead_time: Lead time in periods

    Returns:
        Dictionary with results for each method
    """
    methods = ['normal', 'poisson', 'empirical']
    results = {}

    for method in methods:
        ss_method = SafetyStockMethod(
            service_level=service_level,
            lead_time=lead_time,
            method=method
        )

        try:
            ss_method.fit(demand_history)
            params = ss_method.get_parameters()

            results[method] = {
                'safety_stock': params['safety_stock'],
                'reorder_point': params['reorder_point'],
                'target_stock_level': params['target_stock_level']
            }
        except Exception as e:
            results[method] = {'error': str(e)}

    return results


if __name__ == "__main__":
    # Example usage and testing
    print("🛡️ Safety Stock Method - Example Usage")
    print("=" * 40)

    # Generate sample demand data with some variability
    np.random.seed(42)
    base_demand = 50
    demand_history = np.random.gamma(2, base_demand/2, 365)  # Gamma distribution for realistic demand

    print(f"Sample demand statistics:")
    print(f"  Mean: {np.mean(demand_history):.2f}")
    print(f"  Std: {np.std(demand_history):.2f}")
    print(f"  CV: {np.std(demand_history)/np.mean(demand_history):.3f}")

    # Compare different safety stock methods
    print(f"\n📊 Safety Stock Method Comparison:")
    comparison = compare_safety_stock_methods(demand_history)

    for method, results in comparison.items():
        if 'error' not in results:
            print(f"\n{method.upper()} Method:")
            print(f"  Safety Stock: {results['safety_stock']:.1f}")
            print(f"  Reorder Point: {results['reorder_point']:.1f}")
            print(f"  Target Level: {results['target_stock_level']:.1f}")

    # Test adaptive method
    print(f"\n🔄 Adaptive Safety Stock:")
    adaptive_method = AdaptiveSafetyStock()
    adaptive_method.fit(demand_history)

    current_state = InventoryState(
        inventory_level=40.0,
        outstanding_orders=0.0,
        demand_history=demand_history[-30:],
        time_step=365
    )

    action = adaptive_method.recommend_action(current_state)
    print(f"  Current inventory: {current_state.inventory_level}")
    print(f"  Recommended order: {action.order_quantity:.1f}")

    # Service level analysis
    normal_method = SafetyStockMethod(method="normal")
    normal_method.fit(demand_history[:300])  # Fit on first 300 days

    # Test on remaining data
    achieved_sl = normal_method.calculate_service_level_achieved(demand_history[300:])
    target_sl = normal_method.service_level

    print(f"\n📈 Service Level Analysis:")
    print(f"  Target Service Level: {target_sl:.1%}")
    print(f"  Achieved Service Level: {achieved_sl:.1%}")
    print(f"  Difference: {achieved_sl - target_sl:+.1%}")