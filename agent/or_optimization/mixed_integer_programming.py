import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FacilityLocationSolver:
    """
    Solves Facility Location problems using Mixed Integer Programming (MIP).
    Goal: Minimize total cost (Fixed Facility Cost + Transportation Cost).
    """
    
    def __init__(self):
        pass

    def optimize_locations(
        self, 
        fixed_costs: np.ndarray, 
        transport_costs: np.ndarray, 
        demand: np.ndarray, 
        capacity: np.ndarray
    ) -> Dict:
        """
        Select optimal facility locations to minimize costs while meeting demand.
        
        Args:
            fixed_costs: Array of shape (M,) - Cost to open each facility.
            transport_costs: Matrix of shape (M, N) - Unit cost to ship from i to j.
            demand: Array of shape (N,) - Demand of each customer.
            capacity: Array of shape (M,) - Capacity of each facility.
            
        Returns:
            Dictionary containing:
            - status: Optimization status
            - total_cost: Optimized total cost
            - open_facilities: List of indices of opened facilities
            - allocation: Matrix (M, N) of units transported
        """
        m = len(fixed_costs)
        n = len(demand)
        
        if transport_costs.shape != (m, n):
            raise ValueError(f"Transport costs shape {transport_costs.shape} must match ({m}, {n})")
            
        # Decision variables:
        # y_i: Binary (1 if facility i is open, 0 otherwise). Size M.
        # x_ij: Continuous (Amount shipped from i to j). Size M*N.
        # Total vars = M + M*N
        # Vector structure: [y_0, ..., y_{m-1}, x_00, ..., x_{m-1,n-1}]
        
        num_vars = m + m * n
        
        # Objective function: Minimize sum(f_i * y_i) + sum(c_ij * x_ij)
        c = np.concatenate([fixed_costs, transport_costs.flatten()])
        
        # Constraints
        
        # 1. Demand constraints: sum(x_ij) for all i == demand[j] for each j
        # We need N constraints.
        A_eq = np.zeros((n, num_vars))
        b_eq = np.zeros(n)
        
        for j in range(n):
            # For customer j, sum x_ij over all i
            # x_ij indices start at m. x_ij is at index m + i*n + j
            for i in range(m):
                idx = m + i*n + j
                A_eq[j, idx] = 1
            b_eq[j] = demand[j]
            
        # 2. Capacity constraints: sum(x_ij) for all j <= capacity[i] * y_i
        # Rearranged: sum(x_ij) - capacity[i] * y_i <= 0
        # We need M constraints.
        A_ub = np.zeros((m, num_vars))
        b_ub = np.zeros(m)
        
        for i in range(m):
            # Term -capacity[i] * y_i
            A_ub[i, i] = -capacity[i]
            # Term sum(x_ij) over all j
            for j in range(n):
                idx = m + i*n + j
                A_ub[i, idx] = 1
            b_ub[i] = 0
            
        # Combine constraints for scipy.optimize.milp
        # milp uses LinearConstraint(A, lb, ub)
        
        # Equality constraints (Demand): lb = ub = demand
        # Inequality constraints (Capacity): -inf <= A_ub * x <= 0
        
        A = np.vstack([A_eq, A_ub])
        lb = np.concatenate([b_eq, np.full(m, -np.inf)])
        ub = np.concatenate([b_eq, b_ub])
        
        constraints = LinearConstraint(A, lb, ub)
        
        # Integrality: 1 for integer/binary, 0 for continuous
        # y_i are binary (1), x_ij are continuous (0)
        integrality = np.zeros(num_vars)
        integrality[:m] = 1 
        
        # Bounds:
        # y_i in [0, 1]
        # x_ij in [0, inf]
        lb_vars = np.zeros(num_vars)
        ub_vars = np.concatenate([np.ones(m), np.full(m*n, np.inf)])
        
        bounds = Bounds(lb_vars, ub_vars)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            y = res.x[:m]
            x = res.x[m:].reshape(m, n)
            
            # Round y to nearest integer (0 or 1) due to floating point tolerance
            open_facilities = [i for i, val in enumerate(y) if val > 0.5]
            
            return {
                "success": True,
                "status": res.message,
                "total_cost": res.fun,
                "open_facilities": open_facilities,
                "allocation": x
            }
        else:
            return {
                "success": False,
                "status": res.message,
                "total_cost": None,
                "open_facilities": [],
                "allocation": None
            }

def demo_mip():
    """
    Simple demo for FacilityLocationSolver.
    """
    # 3 Potential Facility Locations
    fixed_costs = np.array([100, 100, 100])
    capacity = np.array([500, 500, 500])
    
    # 5 Customers
    demand = np.array([50, 50, 50, 50, 50]) # Total 250
    
    # Transport costs (3x5)
    # Facility 0 is cheap for Cust 0, 1
    # Facility 1 is cheap for Cust 2, 3
    # Facility 2 is cheap for Cust 4
    transport_costs = np.array([
        [2, 2, 10, 10, 10],
        [10, 10, 2, 2, 10],
        [10, 10, 10, 10, 2]
    ])
    
    solver = FacilityLocationSolver()
    result = solver.optimize_locations(fixed_costs, transport_costs, demand, capacity)
    
    print("Facility Location Optimization Result:")
    print(f"Success: {result['success']}")
    print(f"Total Cost: {result['total_cost']}")
    print(f"Open Facilities: {result['open_facilities']}")
    print("Allocation Matrix:")
    print(result['allocation'])

if __name__ == "__main__":
    demo_mip()
