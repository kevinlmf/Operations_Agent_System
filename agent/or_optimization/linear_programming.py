import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LogisticsScheduler:
    """
    Solves logistics scheduling problems using Linear Programming.
    Example: Minimize transportation costs from warehouses to customers.
    """
    
    def __init__(self):
        pass

    def optimize_transportation(
        self, 
        supply: np.ndarray, 
        demand: np.ndarray, 
        costs: np.ndarray
    ) -> Dict:
        """
        Optimize transportation from M warehouses to N customers.
        
        Args:
            supply: Array of shape (M,) representing supply at each warehouse.
            demand: Array of shape (N,) representing demand at each customer.
            costs: Matrix of shape (M, N) representing unit transport cost.
            
        Returns:
            Dictionary containing:
            - status: Optimization status
            - total_cost: Optimized total cost
            - allocation: Matrix (M, N) of units to transport
        """
        # Number of warehouses (M) and customers (N)
        m = len(supply)
        n = len(demand)
        
        if costs.shape != (m, n):
            raise ValueError(f"Costs matrix shape {costs.shape} must match ({m}, {n})")
            
        if np.sum(supply) < np.sum(demand):
            logger.warning("Total supply is less than total demand. Problem may be infeasible.")

        # Flatten the cost matrix for the objective function c^T * x
        # x is a vector of size M*N, representing flow from i to j flattened
        c = costs.flatten()
        
        # Equality constraints: Supply constraints (sum over j for each i = supply[i])
        # We typically treat supply as an upper bound (<=), but for standard transportation 
        # where we want to move goods, we often use equality if supply == demand, 
        # or inequality if supply >= demand. 
        # Let's assume standard: sum(x_ij) <= supply[i] for each warehouse i
        A_ub = np.zeros((m, m * n))
        b_ub = np.zeros(m)
        
        for i in range(m):
            # For warehouse i, sum of flows to all customers j
            # Indices in flattened x: i*n to (i+1)*n
            A_ub[i, i*n : (i+1)*n] = 1
            b_ub[i] = supply[i]
            
        # Equality constraints: Demand constraints (sum over i for each j = demand[j])
        # We must meet demand: sum(x_ij) == demand[j] for each customer j
        A_eq = np.zeros((n, m * n))
        b_eq = np.zeros(n)
        
        for j in range(n):
            # For customer j, sum of flows from all warehouses i
            # Indices: j, j+n, j+2n, ...
            for i in range(m):
                A_eq[j, i*n + j] = 1
            b_eq[j] = demand[j]
            
        # Bounds: x_ij >= 0
        bounds = [(0, None) for _ in range(m * n)]
        
        # Solve LP
        # Minimize c^T * x
        # Subject to:
        # A_ub * x <= b_ub (Supply constraints)
        # A_eq * x == b_eq (Demand constraints)
        
        res = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            A_eq=A_eq, 
            b_eq=b_eq, 
            bounds=bounds, 
            method='highs'
        )
        
        result = {
            "success": res.success,
            "status": res.message,
            "total_cost": res.fun if res.success else None,
            "allocation": res.x.reshape(m, n) if res.success else None
        }
        
        return result

def demo_logistics():
    """
    Simple demo for the LogisticsScheduler.
    """
    # 2 Warehouses
    supply = np.array([100, 200])
    
    # 3 Customers
    demand = np.array([50, 100, 150])
    
    # Cost matrix (2x3)
    costs = np.array([
        [2, 4, 5],  # Warehouse 0 costs to Cust 0, 1, 2
        [3, 1, 6]   # Warehouse 1 costs to Cust 0, 1, 2
    ])
    
    scheduler = LogisticsScheduler()
    result = scheduler.optimize_transportation(supply, demand, costs)
    
    print("Logistics Optimization Result:")
    print(f"Success: {result['success']}")
    print(f"Total Cost: {result['total_cost']}")
    print("Allocation Matrix:")
    print(result['allocation'])

if __name__ == "__main__":
    demo_logistics()
