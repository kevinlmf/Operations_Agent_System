"""
Multi-Agent Hierarchical Reinforcement Learning for Inventory Management

Architecture:
- CEO Agent: High-level strategic decisions (budget allocation, priorities)
- Department Manager Agents: Department-specific inventory management
- Coordination Agent: Coordinates between departments (resource sharing, conflict resolution)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from collections import deque
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.interfaces import RLMethod


class AgentRole(Enum):
    """Agent roles in the hierarchy"""
    CEO = "ceo"  # 大老板 - Strategic decisions
    MANAGER = "manager"  # 部门经理 - Operational decisions
    COORDINATOR = "coordinator"  # 协调员 - Inter-department coordination


@dataclass
class DepartmentState:
    """State of a single department"""
    department_id: str
    inventory_level: float
    outstanding_orders: List[float]
    demand_history: List[float]
    budget_allocation: float
    performance_metrics: Dict[str, float]


@dataclass
class GlobalState:
    """Global state for CEO and Coordination agents"""
    department_states: List[DepartmentState]
    total_budget: float
    time_step: int
    global_metrics: Dict[str, float]


class CEOAgent(nn.Module):
    """
    CEO Agent - 大老板
    Makes high-level strategic decisions:
    - Budget allocation across departments
    - Priority setting
    - Strategic goals
    """
    hidden_sizes: Tuple[int, ...] = (128, 64)
    num_departments: int = 3

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes]
        # Output: budget allocation for each department (normalized)
        self.budget_layer = nn.Dense(self.num_departments)
        # Output: priority weights for each department
        self.priority_layer = nn.Dense(self.num_departments)

    def __call__(self, global_state_features: jnp.ndarray):
        """
        Forward pass of CEO network
        
        Args:
            global_state_features: [batch_size, feature_dim]
                Features: department performances, total budget, time features
        
        Returns:
            budget_allocation: [batch_size, num_departments] - Budget allocation (softmax)
            priorities: [batch_size, num_departments] - Priority weights
        """
        x = global_state_features
        for layer in self.layers:
            x = nn.relu(layer(x))
        
        # Budget allocation (softmax to ensure sum = 1)
        budget_logits = self.budget_layer(x)
        budget_allocation = nn.softmax(budget_logits)
        
        # Priority weights (sigmoid, independent)
        priority_logits = self.priority_layer(x)
        priorities = nn.sigmoid(priority_logits)
        
        return budget_allocation, priorities


class DepartmentManagerAgent(nn.Module):
    """
    Department Manager Agent - 部门经理
    Makes operational decisions for a specific department:
    - Order quantities
    - Inventory targets
    - Department-specific optimization
    """
    hidden_sizes: Tuple[int, ...] = (128, 64)
    num_actions: int = 21  # Discretized order quantities

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes]
        self.output_layer = nn.Dense(self.num_actions)

    def __call__(self, department_state_features: jnp.ndarray):
        """
        Forward pass of Department Manager network
        
        Args:
            department_state_features: [batch_size, feature_dim]
                Features: inventory, demand, budget, CEO priorities
        
        Returns:
            q_values: [batch_size, num_actions] - Q-values for each action
        """
        x = department_state_features
        for layer in self.layers:
            x = nn.relu(layer(x))
        
        q_values = self.output_layer(x)
        return q_values


class CoordinationAgent(nn.Module):
    """
    Coordination Agent - 协调员
    Coordinates between departments:
    - Resource sharing decisions
    - Conflict resolution
    - Cross-department optimization
    """
    hidden_sizes: Tuple[int, ...] = (128, 64)
    num_departments: int = 3

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes]
        # Output: resource transfer matrix [from_dept, to_dept]
        self.transfer_layer = nn.Dense(self.num_departments * self.num_departments)
        # Output: coordination actions (e.g., joint ordering)
        self.coordination_layer = nn.Dense(self.num_departments)

    def __call__(self, coordination_state_features: jnp.ndarray):
        """
        Forward pass of Coordination network
        
        Args:
            coordination_state_features: [batch_size, feature_dim]
                Features: all department states, conflicts, opportunities
        
        Returns:
            transfer_matrix: [batch_size, num_departments, num_departments] - Resource transfers
            coordination_actions: [batch_size, num_departments] - Coordination signals
        """
        x = coordination_state_features
        for layer in self.layers:
            x = nn.relu(layer(x))
        
        # Resource transfer matrix (reshaped)
        transfer_logits = self.transfer_layer(x)
        transfer_matrix = nn.sigmoid(
            transfer_logits.reshape(-1, self.num_departments, self.num_departments)
        )
        
        # Coordination actions
        coordination_logits = self.coordination_layer(x)
        coordination_actions = nn.sigmoid(coordination_logits)
        
        return transfer_matrix, coordination_actions


class MultiDepartmentEnvironment:
    """
    Multi-Department Inventory Environment
    Simulates multiple departments with shared resources and coordination needs
    """
    
    def __init__(
        self,
        num_departments: int = 3,
        department_names: Optional[List[str]] = None,
        total_budget: float = 10000.0,
        holding_costs: Optional[List[float]] = None,
        stockout_costs: Optional[List[float]] = None,
        ordering_costs: Optional[List[float]] = None,
        max_inventories: Optional[List[int]] = None,
        lead_times: Optional[List[int]] = None,
        demand_generators: Optional[List[callable]] = None,
    ):
        """
        Initialize multi-department environment
        
        Args:
            num_departments: Number of departments
            department_names: Names of departments (e.g., ["Electronics", "Clothing", "Food"])
            total_budget: Total budget available
            holding_costs: Holding cost per department
            stockout_costs: Stockout cost per department
            ordering_costs: Ordering cost per department
            max_inventories: Max inventory per department
            lead_times: Lead time per department
            demand_generators: Demand generator function per department
        """
        self.num_departments = num_departments
        self.department_names = department_names or [f"Dept_{i}" for i in range(num_departments)]
        self.total_budget = total_budget
        
        # Department-specific parameters
        self.holding_costs = holding_costs or [2.0] * num_departments
        self.stockout_costs = stockout_costs or [10.0] * num_departments
        self.ordering_costs = ordering_costs or [50.0] * num_departments
        self.max_inventories = max_inventories or [200] * num_departments
        self.lead_times = lead_times or [1] * num_departments
        
        # Default demand generators
        if demand_generators is None:
            self.demand_generators = [self._default_demand_generator] * num_departments
        else:
            self.demand_generators = demand_generators
        
        # Initialize department states
        self.department_states = []
        for i in range(num_departments):
            self.department_states.append(DepartmentState(
                department_id=self.department_names[i],
                inventory_level=0.0,
                outstanding_orders=[0.0] * self.lead_times[i],
                demand_history=deque(maxlen=30),
                budget_allocation=total_budget / num_departments,  # Equal initial allocation
                performance_metrics={}
            ))
        
        self.time_step = 0
        self.global_metrics = {
            'total_cost': 0.0,
            'total_service_level': 0.0,
            'budget_utilization': 0.0
        }
    
    def _default_demand_generator(self, time_step: int, dept_id: int = 0) -> int:
        """Generate demand with seasonal pattern"""
        base_demand = 20 + dept_id * 5
        seasonal = 10 * np.sin(2 * np.pi * time_step / 365)
        noise = np.random.poisson(5)
        return max(0, int(base_demand + seasonal + noise))
    
    def reset(self, rng: jax.random.PRNGKey) -> GlobalState:
        """Reset environment to initial state"""
        self.time_step = 0
        
        for i, dept_state in enumerate(self.department_states):
            dept_state.inventory_level = 0.0
            dept_state.outstanding_orders = [0.0] * self.lead_times[i]
            dept_state.demand_history = deque(maxlen=30)
            dept_state.budget_allocation = self.total_budget / self.num_departments
            dept_state.performance_metrics = {
                'total_cost': 0.0,
                'service_level': 1.0,
                'orders_placed': 0
            }
        
        self.global_metrics = {
            'total_cost': 0.0,
            'total_service_level': 0.0,
            'budget_utilization': 0.0
        }
        
        return self._get_global_state()
    
    def step(
        self,
        ceo_actions: Dict[str, jnp.ndarray],  # budget_allocation, priorities
        manager_actions: List[int],  # Order quantities per department
        coordination_actions: Optional[Dict[str, jnp.ndarray]] = None  # Resource transfers
    ) -> Tuple[GlobalState, Dict[str, float], bool]:
        """
        Execute one step in the multi-department environment
        
        Args:
            ceo_actions: CEO decisions (budget_allocation, priorities)
            manager_actions: Department manager decisions (order quantities)
            coordination_actions: Coordination agent decisions (resource transfers)
        
        Returns:
            next_global_state, rewards_dict, done
        """
        # 1. Apply CEO budget allocation
        budget_allocation = ceo_actions.get('budget_allocation', 
                                           jnp.ones(self.num_departments) / self.num_departments)
        priorities = ceo_actions.get('priorities', jnp.ones(self.num_departments))
        
        for i, dept_state in enumerate(self.department_states):
            dept_state.budget_allocation = float(budget_allocation[i]) * self.total_budget
        
        # 2. Apply coordination actions (resource sharing)
        if coordination_actions:
            transfer_matrix = coordination_actions.get('transfer_matrix', None)
            if transfer_matrix is not None:
                # Apply resource transfers between departments
                self._apply_resource_transfers(transfer_matrix)
        
        # 3. Execute department manager actions
        department_rewards = []
        department_costs = []
        
        for i, (dept_state, order_qty) in enumerate(zip(self.department_states, manager_actions)):
            # Generate demand
            demand = self.demand_generators[i](self.time_step, i)
            dept_state.demand_history.append(demand)
            
            # Process deliveries
            delivered = dept_state.outstanding_orders.pop(0) if dept_state.outstanding_orders else 0.0
            dept_state.inventory_level += delivered
            
            # Satisfy demand
            sales = min(dept_state.inventory_level, demand)
            dept_state.inventory_level -= sales
            stockout = demand - sales
            
            # Calculate costs
            holding_cost = self.holding_costs[i] * max(0, dept_state.inventory_level)
            stockout_cost = self.stockout_costs[i] * stockout
            ordering_cost = self.ordering_costs[i] * (1 if order_qty > 0 else 0)
            
            total_cost = holding_cost + stockout_cost + ordering_cost
            department_costs.append(total_cost)
            
            # Place order (constrained by budget)
            if order_qty > 0:
                max_order = min(
                    order_qty,
                    self.max_inventories[i] - dept_state.inventory_level - sum(dept_state.outstanding_orders),
                    dept_state.budget_allocation / (self.ordering_costs[i] + 1e-6)  # Budget constraint
                )
                order_qty = max(0, int(max_order))
            
            dept_state.outstanding_orders.append(float(order_qty))
            
            # Update performance metrics
            dept_state.performance_metrics['total_cost'] += total_cost
            dept_state.performance_metrics['service_level'] = (
                dept_state.performance_metrics.get('service_level', 1.0) * 0.9 + 
                (1.0 if stockout == 0 else 0.0) * 0.1
            )
            dept_state.performance_metrics['orders_placed'] += 1
            
            # Reward (negative cost, weighted by priority)
            reward = -total_cost * float(priorities[i])
            department_rewards.append(reward)
        
        # 4. Update global metrics
        total_cost = sum(department_costs)
        avg_service_level = np.mean([d.performance_metrics['service_level'] 
                                     for d in self.department_states])
        budget_utilization = sum([d.budget_allocation for d in self.department_states]) / self.total_budget
        
        self.global_metrics['total_cost'] += total_cost
        self.global_metrics['total_service_level'] = avg_service_level
        self.global_metrics['budget_utilization'] = budget_utilization
        
        # 5. Prepare rewards (scaled to prevent extreme values)
        # Scale rewards to reasonable range (-100 to 100)
        reward_scale = 0.01  # Scale down costs
        service_bonus_scale = 10.0  # Scale down service level bonus
        
        rewards_dict = {
            'ceo_reward': -total_cost * reward_scale,  # CEO cares about total cost
            'coordinator_reward': -total_cost * reward_scale + service_bonus_scale * avg_service_level,  # Coordinator balances cost and service
            'department_rewards': [r * reward_scale for r in department_rewards],  # Scale department rewards
            'global_reward': -total_cost * reward_scale + service_bonus_scale * 0.5 * avg_service_level
        }
        
        # 6. Check if done
        self.time_step += 1
        done = self.time_step >= 365  # One year episodes
        
        return self._get_global_state(), rewards_dict, done
    
    def _apply_resource_transfers(self, transfer_matrix: jnp.ndarray):
        """Apply resource transfers between departments"""
        # transfer_matrix[i, j] = amount to transfer from dept i to dept j
        transfers = np.array(transfer_matrix)
        
        for i in range(self.num_departments):
            for j in range(self.num_departments):
                if i != j and transfers[i, j] > 0:
                    # Transfer inventory from dept i to dept j
                    transfer_amount = min(
                        transfers[i, j],
                        self.department_states[i].inventory_level
                    )
                    self.department_states[i].inventory_level -= transfer_amount
                    self.department_states[j].inventory_level += transfer_amount
    
    def _get_global_state(self) -> GlobalState:
        """Get current global state"""
        return GlobalState(
            department_states=self.department_states.copy(),
            total_budget=self.total_budget,
            time_step=self.time_step,
            global_metrics=self.global_metrics.copy()
        )
    
    def get_ceo_state_features(self) -> jnp.ndarray:
        """Extract features for CEO agent"""
        features = []
        
        # Department performance metrics
        for dept_state in self.department_states:
            features.extend([
                dept_state.performance_metrics.get('total_cost', 0.0) / 1000.0,  # Normalized
                dept_state.performance_metrics.get('service_level', 1.0),
                dept_state.budget_allocation / self.total_budget,  # Budget ratio
                dept_state.inventory_level / self.max_inventories[0]  # Normalized inventory
            ])
        
        # Global metrics
        features.extend([
            self.global_metrics['total_cost'] / 10000.0,
            self.global_metrics['total_service_level'],
            self.global_metrics['budget_utilization'],
            self.time_step / 365.0  # Time feature
        ])
        
        return jnp.array(features, dtype=jnp.float32)
    
    def get_department_state_features(self, dept_id: int) -> jnp.ndarray:
        """Extract features for department manager agent"""
        dept_state = self.department_states[dept_id]
        
        # Inventory features
        inv_norm = dept_state.inventory_level / self.max_inventories[dept_id]
        orders_norm = np.array(dept_state.outstanding_orders) / self.max_inventories[dept_id]
        
        # Demand features
        recent_demand = list(dept_state.demand_history)[-7:] if dept_state.demand_history else [0]
        demand_mean = np.mean(recent_demand) / 100.0
        demand_std = np.std(recent_demand) / 100.0 if len(recent_demand) > 1 else 0.0
        
        # Budget and priority features
        budget_ratio = dept_state.budget_allocation / self.total_budget
        
        # Time features
        day_of_year = (self.time_step % 365) / 365.0
        
        features = jnp.array([
            inv_norm,
            *orders_norm,
            demand_mean,
            demand_std,
            budget_ratio,
            day_of_year
        ], dtype=jnp.float32)
        
        return features
    
    def get_coordination_state_features(self) -> jnp.ndarray:
        """Extract features for coordination agent"""
        features = []
        
        # All department states
        for dept_state in self.department_states:
            features.extend([
                dept_state.inventory_level / self.max_inventories[0],
                dept_state.performance_metrics.get('service_level', 1.0),
                dept_state.budget_allocation / self.total_budget
            ])
        
        # Cross-department metrics (conflicts, opportunities)
        # For example: inventory imbalance
        inventories = [d.inventory_level for d in self.department_states]
        avg_inv = np.mean(inventories)
        inv_std = np.std(inventories) / (avg_inv + 1e-6)  # Coefficient of variation
        
        features.extend([
            inv_std,  # Inventory imbalance
            self.time_step / 365.0
        ])
        
        return jnp.array(features, dtype=jnp.float32)


class MultiAgentExperience(NamedTuple):
    """Experience tuple for multi-agent RL"""
    ceo_state: jnp.ndarray
    ceo_action: Dict[str, jnp.ndarray]
    manager_states: List[jnp.ndarray]
    manager_actions: List[int]
    coordinator_state: jnp.ndarray
    coordinator_action: Optional[Dict[str, jnp.ndarray]]
    rewards: Dict[str, float]
    next_ceo_state: jnp.ndarray
    next_manager_states: List[jnp.ndarray]
    next_coordinator_state: jnp.ndarray
    done: bool


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent system"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: MultiAgentExperience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int, rng: jax.random.PRNGKey) -> List[MultiAgentExperience]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = random.choice(rng, len(self.buffer), (batch_size,), replace=False)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self):
        return len(self.buffer)

