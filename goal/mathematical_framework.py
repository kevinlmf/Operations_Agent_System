"""
Mathematical Framework for Inventory Optimization Problem

This module defines the core mathematical components of the inventory management problem:
- State space representation
- Cost function formulation
- Demand process modeling
- Service level constraints
"""

import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Callable
import chex


class InventoryState(NamedTuple):
    """State representation at time t"""
    inventory_level: float      # I_t: Current inventory on hand
    outstanding_orders: float   # O_t: In-transit inventory (ordered but not delivered)
    demand_history: chex.Array  # H_t: Historical demand [D_{t-w}, ..., D_{t-1}]
    time_step: int             # Current time period


class InventoryParams(NamedTuple):
    """Problem parameters"""
    holding_cost: float        # h: Holding cost per unit per period
    stockout_cost: float       # p: Penalty cost per unit short
    ordering_cost: float       # K: Fixed cost per order
    unit_cost: float          # c: Variable cost per unit ordered
    lead_time: int            # L: Periods between order and delivery
    service_level: float      # α: Minimum service level (e.g., 0.95)
    max_order_qty: float      # Maximum order quantity constraint


class DemandParams(NamedTuple):
    """Demand process parameters"""
    base_demand: float         # Base demand level
    seasonality_amplitude: float  # Seasonal variation amplitude
    trend_rate: float         # Trend component
    noise_std: float          # Random noise standard deviation
    promotion_prob: float     # Probability of promotional spike
    promotion_multiplier: float  # Demand multiplier during promotions


def calculate_period_cost(
    state: InventoryState,
    action: float,
    demand: float,
    params: InventoryParams
) -> float:
    """
    Calculate total cost for current period

    C_t = h·max(I_t, 0) + p·max(-I_t, 0) + K·𝟙(a_t > 0) + c·a_t
    """
    # Update inventory after demand realization
    new_inventory = state.inventory_level - demand

    # Holding cost (positive inventory)
    holding_cost = params.holding_cost * jnp.maximum(new_inventory, 0)

    # Stockout cost (negative inventory = backorders)
    stockout_cost = params.stockout_cost * jnp.maximum(-new_inventory, 0)

    # Fixed ordering cost (if order placed)
    fixed_cost = params.ordering_cost * (action > 0).astype(float)

    # Variable ordering cost
    variable_cost = params.unit_cost * action

    total_cost = holding_cost + stockout_cost + fixed_cost + variable_cost

    return total_cost


def generate_demand(
    key: chex.PRNGKey,
    time_step: int,
    params: DemandParams
) -> float:
    """
    Generate demand for time t following a complex process:
    D_t = base + seasonality + trend + promotion + noise
    """
    # Seasonal component (12-period cycle for monthly data)
    seasonal_component = params.seasonality_amplitude * jnp.sin(2 * jnp.pi * time_step / 12)

    # Trend component
    trend_component = params.trend_rate * time_step

    # Random noise
    noise_key, promotion_key = random.split(key)
    noise = random.normal(noise_key) * params.noise_std

    # Promotional spike (random occurrence)
    promotion_indicator = random.bernoulli(promotion_key, params.promotion_prob).astype(float)
    promotion_effect = (params.promotion_multiplier - 1) * promotion_indicator

    # Combine components
    demand = (params.base_demand +
             seasonal_component +
             trend_component +
             noise) * (1 + promotion_effect)

    # Ensure non-negative demand
    demand = jnp.maximum(demand, 0.0)

    return demand


def update_state(
    state: InventoryState,
    action: float,
    demand: float,
    delivered_orders: float,
    params: InventoryParams
) -> InventoryState:
    """Update state after action and demand realization"""

    # Update inventory: previous level + deliveries - demand
    new_inventory = state.inventory_level + delivered_orders - demand

    # Update outstanding orders (remove delivered, add new order)
    new_outstanding = state.outstanding_orders - delivered_orders + action

    # Update demand history (rolling window)
    new_demand_history = jnp.roll(state.demand_history, shift=-1)
    new_demand_history = new_demand_history.at[-1].set(demand)

    return InventoryState(
        inventory_level=new_inventory,
        outstanding_orders=new_outstanding,
        demand_history=new_demand_history,
        time_step=state.time_step + 1
    )


def calculate_service_level(inventory_levels: chex.Array) -> float:
    """Calculate service level as fraction of periods without stockout"""
    no_stockout = (inventory_levels >= 0).astype(float)
    return jnp.mean(no_stockout)


def is_feasible_action(action: float, state: InventoryState, params: InventoryParams) -> bool:
    """Check if action satisfies constraints"""
    return (action >= 0) and (action <= params.max_order_qty)


class InventoryEnvironment:
    """JAX-compatible inventory management environment"""

    def __init__(self, inv_params: InventoryParams, demand_params: DemandParams):
        self.inv_params = inv_params
        self.demand_params = demand_params

    def reset(self, key: chex.PRNGKey, history_length: int = 12) -> InventoryState:
        """Reset environment to initial state"""
        # Initialize with some base inventory and empty order pipeline
        initial_inventory = self.demand_params.base_demand * 2  # 2 periods of base demand
        initial_outstanding = 0.0

        # Initialize demand history with base demand
        initial_history = jnp.full((history_length,), self.demand_params.base_demand)

        return InventoryState(
            inventory_level=initial_inventory,
            outstanding_orders=initial_outstanding,
            demand_history=initial_history,
            time_step=0
        )

    def step(self, key: chex.PRNGKey, state: InventoryState, action: float):
        """Execute one step of the environment"""

        # Generate demand for current period
        demand = generate_demand(key, state.time_step, self.demand_params)

        # Determine deliveries (orders placed L periods ago)
        # For simplicity, assume orders arrive exactly after lead_time
        if state.time_step >= self.inv_params.lead_time:
            # In practice, would need to track individual order delivery times
            delivered = state.outstanding_orders / self.inv_params.lead_time
        else:
            delivered = 0.0

        # Calculate cost for this period
        cost = calculate_period_cost(state, action, demand, self.inv_params)

        # Update state
        next_state = update_state(state, action, demand, delivered, self.inv_params)

        # Additional info
        info = {
            'demand': demand,
            'delivered': delivered,
            'service_level': float(next_state.inventory_level >= 0),
            'inventory_level': next_state.inventory_level
        }

        return next_state, cost, info