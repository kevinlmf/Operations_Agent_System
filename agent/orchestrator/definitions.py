from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class StrategicDecision:
    """
    Output from the Strategic (MIP) Agent.
    Acts as a constraint for the Tactical (LP) Agent.
    """
    open_facilities_indices: List[int]
    facility_capacities: np.ndarray  # Capacity of opened facilities
    transport_costs_matrix: np.ndarray # Full cost matrix, to be filtered
    
    # Metadata
    total_fixed_cost: float
    is_feasible: bool = False

@dataclass
class TacticalDecision:
    """
    Output from the Tactical (LP) Agent.
    Acts as a constraint for the Operational (DP) Agent.
    """
    # Flow matrix: [Facility_i -> Customer_j]
    transport_allocation: np.ndarray 
    
    # Aggregated constraints for DP
    # e.g., Total volume allocated to Facility_i that needs to be stocked
    facility_inbound_volumes: Dict[int, float] 
    
    # Metadata
    total_transport_cost: float
    is_feasible: bool = False

@dataclass
class OperationalDecision:
    """
    Output from the Operational (DP) Agent.
    Final execution details.
    """
    # Map of Facility_ID -> List of Selected Item Indices
    facility_inventory_plans: Dict[int, List[int]]
    
    # Metadata
    total_inventory_value: float
    is_feasible: bool = False

@dataclass
class OperationsContext:
    """
    Shared Blackboard / Context for the Multi-Agent System.
    """
    # Initial Inputs
    potential_facilities_costs: np.ndarray
    potential_facilities_capacities: np.ndarray
    customer_demands: np.ndarray
    transport_costs_full: np.ndarray
    
    # Operational Data (Items available to stock)
    item_values: List[float]
    item_weights: List[int]
    
    # Agent Decisions (Populated sequentially)
    strategic_plan: Optional[StrategicDecision] = None
    tactical_plan: Optional[TacticalDecision] = None
    operational_plan: Optional[OperationalDecision] = None
    
    def is_complete(self) -> bool:
        return (self.strategic_plan is not None and 
                self.tactical_plan is not None and 
                self.operational_plan is not None)
