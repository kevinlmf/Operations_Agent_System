import gym
from gym import spaces
import numpy as np
import logging
from typing import Optional, Dict, List

from agent.orchestrator.orchestrator import OperationsOrchestrator
from agent.orchestrator.definitions import OperationsContext

logger = logging.getLogger(__name__)

class OperationsEnv(gym.Env):
    """
    RL Environment for Multi-Agent Operations.
    The RL agent acts as a 'Manager' setting the Safety Stock policy.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(OperationsEnv, self).__init__()
        
        # --- Action Space ---
        # Discrete levels of Safety Stock Factor
        # 0: 1.0x (Just-in-Time)
        # 1: 1.2x (Conservative)
        # 2: 1.5x (Hoarding)
        self.action_space = spaces.Discrete(3)
        self.action_map = {0: 1.0, 1: 1.2, 2: 1.5}
        
        # --- Observation Space ---
        # 1. Current Inventory Level (Normalized 0-1)
        # 2. Recent Demand Mean (Normalized)
        # 3. Recent Demand Std Dev (Normalized)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # --- Internal State ---
        self.orchestrator = OperationsOrchestrator()
        self.current_inventory = 0.0
        self.max_capacity = 1500.0 # Sum of all facility capacities
        self.demand_history = []
        
        # Simulation Parameters
        self.base_demand_mean = 250
        self.base_demand_std = 50
        self.unit_revenue = 10.0
        self.holding_cost_per_unit = 1.0
        self.stockout_penalty_per_unit = 5.0
        
        # Context Template (Static data)
        self.template_context = self._create_template_context()

    def _create_template_context(self) -> OperationsContext:
        """Creates the static parts of the context (costs, capacities, items)."""
        # 3 Potential Warehouses
        fixed_costs = np.array([1000, 1200, 800])
        capacities = np.array([500, 500, 500])
        
        # Transport costs (3x5)
        transport_costs = np.array([
            [2, 4, 5, 2, 1],
            [3, 1, 6, 3, 2],
            [1, 3, 2, 1, 5]
        ])
        
        # Inventory Items (Simplified: 1 generic item type for RL demo)
        # We treat "items" as generic units for the RL high-level view
        item_values = [10.0] * 10
        item_weights = [1] * 10 
        
        return OperationsContext(
            potential_facilities_costs=fixed_costs,
            potential_facilities_capacities=capacities,
            customer_demands=np.zeros(5), # Placeholder
            transport_costs_full=transport_costs,
            item_values=item_values,
            item_weights=item_weights
        )

    def reset(self):
        self.current_inventory = 500.0 # Start with some stock
        self.demand_history = [self.base_demand_mean] * 5
        return self._get_observation()

    def _get_observation(self):
        inv_ratio = self.current_inventory / self.max_capacity
        
        recent_demand = np.array(self.demand_history[-5:])
        demand_mean = np.mean(recent_demand) / 500.0 # Normalize
        demand_std = np.std(recent_demand) / 100.0 # Normalize
        
        return np.array([inv_ratio, demand_mean, demand_std], dtype=np.float32)

    def step(self, action):
        # 1. Parse Action
        safety_factor = self.action_map[action]
        
        # 2. Generate REAL Demand (Stochastic)
        real_total_demand = max(0, np.random.normal(self.base_demand_mean, self.base_demand_std))
        self.demand_history.append(real_total_demand)
        
        # Distribute real demand across 5 customers (randomly)
        customer_proportions = np.random.dirichlet(np.ones(5))
        real_customer_demands = real_total_demand * customer_proportions
        
        # 3. Prepare Context for OR Agents (Planning Phase)
        # The Agents see the "Forecast" (which we simulate as Real * SafetyFactor for simplicity here,
        # or we could say Forecast is Mean, and SafetyFactor adjusts the target).
        # Let's say: Target_Supply = Forecast_Mean * Safety_Factor
        target_supply_total = self.base_demand_mean * safety_factor
        
        # We tell the OR agents to satisfy this Target Supply
        # We do this by setting the "Demand" in the context to this target.
        # The LP agent will try to ship this amount.
        planning_demands = target_supply_total * customer_proportions
        
        context = self.template_context
        context.customer_demands = planning_demands
        
        # 4. Execute OR Pipeline
        result_context = self.orchestrator.run_pipeline(context)
        
        # 5. Calculate Outcomes
        if not result_context.is_complete():
            # Penalty for failure
            return self._get_observation(), -1000.0, True, {"msg": "Pipeline Failed"}
            
        # How much was actually shipped/stocked?
        # Sum of inbound volumes to facilities
        total_shipped = sum(result_context.tactical_plan.facility_inbound_volumes.values())
        
        # Update Inventory
        self.current_inventory += total_shipped
        
        # Fulfill Real Demand
        sales = min(self.current_inventory, real_total_demand)
        unmet_demand = real_total_demand - sales
        self.current_inventory -= sales
        
        # 6. Calculate Reward (Profit)
        revenue = sales * self.unit_revenue
        
        # Costs from OR Agents
        fixed_cost = result_context.strategic_plan.total_fixed_cost
        transport_cost = result_context.tactical_plan.total_transport_cost
        
        # Operational Costs
        holding_cost = self.current_inventory * self.holding_cost_per_unit
        penalty_cost = unmet_demand * self.stockout_penalty_per_unit
        
        # Total Reward
        # We scale down fixed cost for the step reward to make it learnable (amortize it)
        # or assume fixed cost is per-step (rent).
        reward = revenue - (fixed_cost + transport_cost + holding_cost + penalty_cost)
        
        done = False # Infinite horizon for now
        info = {
            "real_demand": real_total_demand,
            "shipped": total_shipped,
            "sales": sales,
            "inventory": self.current_inventory,
            "revenue": revenue,
            "costs": fixed_cost + transport_cost + holding_cost + penalty_cost
        }
        
        return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        print(f"Inv: {self.current_inventory:.2f} | Last Demand: {self.demand_history[-1]:.2f}")
