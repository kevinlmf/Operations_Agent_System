import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DynamicProgrammingSolver:
    """
    Solves problems using Dynamic Programming.
    Includes:
    1. Grid Path Planning (Min Cost Path)
    2. Inventory Knapsack (Maximize value given capacity)
    """
    
    def __init__(self):
        pass

    def min_cost_path(self, grid: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Finds the minimum cost path from top-left (0,0) to bottom-right (m-1, n-1)
        in a grid where values represent cost to traverse that cell.
        Moves allowed: Right and Down.
        
        Args:
            grid: 2D numpy array of costs.
            
        Returns:
            Tuple containing:
            - min_cost: Minimum cost to reach bottom-right.
            - path: List of coordinates [(0,0), ..., (m-1, n-1)]
        """
        m, n = grid.shape
        if m == 0 or n == 0:
            return 0.0, []

        # dp[i][j] stores min cost to reach cell (i, j)
        dp = np.zeros((m, n))
        
        # Initialize starting point
        dp[0][0] = grid[0][0]
        
        # Initialize first column (can only come from above)
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
            
        # Initialize first row (can only come from left)
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
            
        # Fill DP table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
                
        min_cost = dp[m-1][n-1]
        
        # Backtrack to find path
        path = []
        i, j = m-1, n-1
        path.append((i, j))
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                if dp[i-1][j] < dp[i][j-1]:
                    i -= 1
                else:
                    j -= 1
            path.append((i, j))
            
        return min_cost, path[::-1]

    def inventory_knapsack(
        self, 
        values: List[float], 
        weights: List[int], 
        capacity: int
    ) -> Tuple[float, List[int]]:
        """
        Solves the 0/1 Knapsack problem for inventory selection.
        Maximize total value of selected items without exceeding capacity.
        
        Args:
            values: List of values for each item.
            weights: List of weights (costs/sizes) for each item.
            capacity: Maximum total weight allowed.
            
        Returns:
            Tuple containing:
            - max_value: Maximum value achievable.
            - selected_indices: List of indices of selected items.
        """
        n = len(values)
        if len(weights) != n:
            raise ValueError("Values and weights must have same length")
            
        # dp[i][w] = max value using first i items with capacity w
        # Rows: 0 to n (items)
        # Cols: 0 to capacity
        dp = np.zeros((n + 1, capacity + 1))
        
        for i in range(1, n + 1):
            val = values[i-1]
            wt = weights[i-1]
            for w in range(capacity + 1):
                if wt <= w:
                    # Max of (excluding item, including item)
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt] + val)
                else:
                    dp[i][w] = dp[i-1][w]
                    
        max_value = dp[n][capacity]
        
        # Backtrack to find items
        selected_indices = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_indices.append(i-1)
                w -= weights[i-1]
                
        return max_value, selected_indices[::-1]

def demo_dp():
    """
    Simple demo for DynamicProgrammingSolver.
    """
    solver = DynamicProgrammingSolver()
    
    print("--- DP: Min Cost Path ---")
    grid = np.array([
        [1, 3, 1],
        [1, 5, 1],
        [4, 2, 1]
    ])
    cost, path = solver.min_cost_path(grid)
    print(f"Grid:\n{grid}")
    print(f"Min Cost: {cost}")
    print(f"Path: {path}")
    
    print("\n--- DP: Inventory Knapsack ---")
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    max_val, items = solver.inventory_knapsack(values, weights, capacity)
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Capacity: {capacity}")
    print(f"Max Value: {max_val}")
    print(f"Selected Items (indices): {items}")

if __name__ == "__main__":
    demo_dp()
