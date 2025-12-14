import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass

from src.or_optimization.linear_programming import LogisticsScheduler
from src.or_optimization.dynamic_programming import DynamicProgrammingSolver
from src.or_optimization.mixed_integer_programming import FacilityLocationSolver

logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Standard response from an OR Agent"""
    success: bool
    decision: Any
    metrics: Dict[str, float]
    message: str

class ORAgent:
    """Base class for OR Agents"""
    def __init__(self, name: str):
        self.name = name

    def act(self, *args, **kwargs) -> AgentResponse:
        raise NotImplementedError

class MIPAgent(ORAgent):
    """
    Strategic Agent: Decides on Facility Locations.
    Uses Mixed Integer Programming.
    """
    def __init__(self, name: str = "MIP_Strategic_Agent"):
        super().__init__(name)
        self.solver = FacilityLocationSolver()

    def act(self, fixed_costs: np.ndarray, transport_costs: np.ndarray, 
            demand: np.ndarray, capacity: np.ndarray) -> AgentResponse:
        """
        Decide which facilities to open.
        """
        logger.info(f"[{self.name}] Optimizing facility locations...")
        result = self.solver.optimize_locations(fixed_costs, transport_costs, demand, capacity)
        
        if result['success']:
            return AgentResponse(
                success=True,
                decision={
                    'open_facilities': result['open_facilities'],
                    'allocation_plan': result['allocation']
                },
                metrics={'total_cost': result['total_cost']},
                message="Optimal locations found."
            )
        else:
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message=f"Optimization failed: {result['status']}"
            )

class LPAgent(ORAgent):
    """
    Tactical Agent: Decides on Logistics/Transportation.
    Uses Linear Programming.
    """
    def __init__(self, name: str = "LP_Tactical_Agent"):
        super().__init__(name)
        self.solver = LogisticsScheduler()

    def act(self, supply: np.ndarray, demand: np.ndarray, costs: np.ndarray) -> AgentResponse:
        """
        Decide on transport quantities.
        """
        logger.info(f"[{self.name}] Optimizing logistics flow...")
        result = self.solver.optimize_transportation(supply, demand, costs)
        
        if result['success']:
            return AgentResponse(
                success=True,
                decision={'transport_matrix': result['allocation']},
                metrics={'total_cost': result['total_cost']},
                message="Optimal transport flow found."
            )
        else:
            return AgentResponse(
                success=False,
                decision=None,
                metrics={},
                message=f"Optimization failed: {result['status']}"
            )

class DPAgent(ORAgent):
    """
    Operational Agent: Decides on Inventory Mix or Path.
    Uses Dynamic Programming.
    """
    def __init__(self, name: str = "DP_Operational_Agent"):
        super().__init__(name)
        self.solver = DynamicProgrammingSolver()

    def optimize_inventory(self, values: List[float], weights: List[int], capacity: int) -> AgentResponse:
        """
        Decide on inventory mix (Knapsack).
        """
        logger.info(f"[{self.name}] Optimizing inventory mix...")
        max_val, items = self.solver.inventory_knapsack(values, weights, capacity)
        
        return AgentResponse(
            success=True,
            decision={'selected_items': items},
            metrics={'total_value': max_val},
            message="Optimal inventory mix found."
        )

    def optimize_path(self, grid: np.ndarray) -> AgentResponse:
        """
        Decide on optimal path.
        """
        logger.info(f"[{self.name}] Optimizing path...")
        cost, path = self.solver.min_cost_path(grid)
        
        return AgentResponse(
            success=True,
            decision={'path': path},
            metrics={'min_cost': cost},
            message="Optimal path found."
        )
    
    def act(self, mode: str, **kwargs) -> AgentResponse:
        if mode == 'inventory':
            return self.optimize_inventory(kwargs['values'], kwargs['weights'], kwargs['capacity'])
        elif mode == 'path':
            return self.optimize_path(kwargs['grid'])
        else:
            return AgentResponse(False, None, {}, f"Unknown mode: {mode}")
