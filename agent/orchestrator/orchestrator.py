import logging
import numpy as np
from typing import Dict, List, Any

from agent.or_agents import MIPAgent, LPAgent, DPAgent
from agent.orchestrator.definitions import (
    OperationsContext, StrategicDecision, TacticalDecision, OperationalDecision
)

logger = logging.getLogger(__name__)

class OperationsOrchestrator:
    """
    Orchestrates the Multi-Agent Operations flow.
    Ensures constraints are propagated top-down:
    Strategic (MIP) -> Tactical (LP) -> Operational (DP).
    """
    
    def __init__(self):
        self.mip_agent = MIPAgent("Strategic_Agent")
        self.lp_agent = LPAgent("Tactical_Agent")
        self.dp_agent = DPAgent("Operational_Agent")
        
    def run_pipeline(self, context: OperationsContext) -> OperationsContext:
        """
        Executes the full operations pipeline.
        """
        logger.info("Starting Multi-Agent Operations Pipeline...")
        
        # --- Phase 1: Strategic (MIP) ---
        context.strategic_plan = self._run_strategic_phase(context)
        if not context.strategic_plan.is_feasible:
            logger.error("Strategic Phase failed. Aborting pipeline.")
            return context
            
        # --- Phase 2: Tactical (LP) ---
        context.tactical_plan = self._run_tactical_phase(context)
        if not context.tactical_plan.is_feasible:
            logger.error("Tactical Phase failed. Aborting pipeline.")
            return context
            
        # --- Phase 3: Operational (DP) ---
        context.operational_plan = self._run_operational_phase(context)
        
        logger.info("Pipeline Completed Successfully.")
        return context

    def _run_strategic_phase(self, context: OperationsContext) -> StrategicDecision:
        logger.info(">>> Phase 1: Strategic Planning (Facility Location)")
        
        response = self.mip_agent.act(
            fixed_costs=context.potential_facilities_costs,
            transport_costs=context.transport_costs_full,
            demand=context.customer_demands,
            capacity=context.potential_facilities_capacities
        )
        
        if not response.success:
            return StrategicDecision([], np.array([]), np.array([]), 0.0, is_feasible=False)
            
        return StrategicDecision(
            open_facilities_indices=response.decision['open_facilities'],
            facility_capacities=context.potential_facilities_capacities,
            transport_costs_matrix=context.transport_costs_full,
            total_fixed_cost=response.metrics['total_cost'],
            is_feasible=True
        )

    def _run_tactical_phase(self, context: OperationsContext) -> TacticalDecision:
        logger.info(">>> Phase 2: Tactical Logistics (Flow Optimization)")
        
        strat_plan = context.strategic_plan
        open_indices = strat_plan.open_facilities_indices
        
        # CONSTRAINT PROPAGATION:
        # 1. Filter Supply: Only open facilities can supply.
        active_supply = strat_plan.facility_capacities[open_indices]
        
        # 2. Filter Costs: Only rows corresponding to open facilities.
        active_costs = strat_plan.transport_costs_matrix[open_indices]
        
        response = self.lp_agent.act(
            supply=active_supply,
            demand=context.customer_demands,
            costs=active_costs
        )
        
        if not response.success:
            return TacticalDecision(np.array([]), {}, 0.0, is_feasible=False)
            
        # Calculate inbound volumes for each OPEN facility
        # Allocation matrix shape: (Num_Open_Facilities, Num_Customers)
        allocation_matrix = response.decision['transport_matrix']
        
        # Sum across customers (rows) to get total outflow/inbound requirement for each facility
        # Axis 1 = sum over columns (customers)
        facility_outflows = np.sum(allocation_matrix, axis=1)
        
        # Map back to original facility indices
        facility_inbound_volumes = {}
        for idx, local_idx in enumerate(range(len(open_indices))):
            original_id = open_indices[local_idx]
            facility_inbound_volumes[original_id] = facility_outflows[local_idx]
            
        return TacticalDecision(
            transport_allocation=allocation_matrix,
            facility_inbound_volumes=facility_inbound_volumes,
            total_transport_cost=response.metrics['total_cost'],
            is_feasible=True
        )

    def _run_operational_phase(self, context: OperationsContext) -> OperationalDecision:
        logger.info(">>> Phase 3: Operational Optimization (Inventory Mix)")
        
        tactical_plan = context.tactical_plan
        facility_inventory_plans = {}
        total_value = 0.0
        
        # Iterate over each ACTIVE facility that has flow allocated
        for facility_id, required_volume in tactical_plan.facility_inbound_volumes.items():
            if required_volume <= 0:
                continue
                
            logger.info(f"Optimizing inventory for Facility {facility_id} (Budget/Capacity: {required_volume})")
            
            # CONSTRAINT PROPAGATION:
            # The 'capacity' for the Knapsack problem is limited by the 
            # logistical flow allocated to this facility (or we could use it as a target).
            # Here we assume the flow represents the storage budget we are filling.
            
            response = self.dp_agent.act(
                mode='inventory',
                values=context.item_values,
                weights=context.item_weights,
                capacity=int(required_volume) # Knapsack usually requires int capacity
            )
            
            if response.success:
                facility_inventory_plans[facility_id] = response.decision['selected_items']
                total_value += response.metrics['total_value']
            else:
                logger.warning(f"Inventory optimization failed for Facility {facility_id}")
                
        return OperationalDecision(
            facility_inventory_plans=facility_inventory_plans,
            total_inventory_value=total_value,
            is_feasible=True
        )
