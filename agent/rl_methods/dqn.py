"""
Deep Q-Network (DQN) for Inventory Management

Implements DQN reinforcement learning agent using JAX/Flax for dynamic
inventory control with experience replay and target networks.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any, Optional, Tuple, NamedTuple
from collections import deque
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import RLMethod, InventoryState, InventoryAction


class Experience(NamedTuple):
    """Experience tuple for replay buffer"""
    state: jnp.ndarray
    action: int
    reward: float
    next_state: jnp.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, ...]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Sample random indices
        indices = random.choice(rng, len(self.buffer), (batch_size,), replace=False)

        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]

        states = jnp.stack([exp.state for exp in batch])
        actions = jnp.array([exp.action for exp in batch])
        rewards = jnp.array([exp.reward for exp in batch])
        next_states = jnp.stack([exp.next_state for exp in batch])
        dones = jnp.array([exp.done for exp in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Deep Q-Network architecture"""

    hidden_sizes: Tuple[int, ...] = (256, 256)
    num_actions: int = 21  # 0 to 20 units (discretized action space)

    def setup(self):
        self.layers = [nn.Dense(size) for size in self.hidden_sizes]
        self.output_layer = nn.Dense(self.num_actions)

    def __call__(self, x):
        """
        Forward pass of Q-network

        Args:
            x: State representation [batch_size, state_dim]

        Returns:
            Q-values for all actions [batch_size, num_actions]
        """
        for layer in self.layers:
            x = nn.relu(layer(x))

        q_values = self.output_layer(x)
        return q_values


class InventoryEnvironment:
    """Inventory management environment for RL training"""

    def __init__(self,
                 holding_cost: float = 2.0,
                 stockout_cost: float = 10.0,
                 ordering_cost: float = 50.0,
                 max_inventory: int = 200,
                 lead_time: int = 1,
                 demand_generator: Optional[callable] = None):
        """
        Initialize inventory environment

        Args:
            holding_cost: Cost per unit held per period
            stockout_cost: Cost per unit short per period
            ordering_cost: Fixed cost per order
            max_inventory: Maximum inventory capacity
            lead_time: Order lead time
            demand_generator: Function to generate demand
        """
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost
        self.max_inventory = max_inventory
        self.lead_time = lead_time

        # Default demand generator (Poisson with seasonal pattern)
        if demand_generator is None:
            self.demand_generator = self._default_demand_generator
        else:
            self.demand_generator = demand_generator

        # State variables
        self.inventory_level = 0
        self.outstanding_orders = [0] * lead_time  # Orders in pipeline
        self.time_step = 0
        self.demand_history = deque(maxlen=30)

        # Initialize demand history
        for _ in range(30):
            self.demand_history.append(self.demand_generator(0))

    def _default_demand_generator(self, time_step: int) -> int:
        """Generate demand with seasonal pattern"""
        # Base demand with seasonal variation
        base_demand = 50
        seasonal = 10 * np.sin(2 * np.pi * time_step / 365.25)
        weekly = 5 * np.sin(2 * np.pi * time_step / 7)

        mean_demand = base_demand + seasonal + weekly
        demand = np.random.poisson(max(1, mean_demand))

        return int(demand)

    def reset(self, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Reset environment to initial state"""
        self.inventory_level = 100  # Start with some inventory
        self.outstanding_orders = [0] * self.lead_time
        self.time_step = 0

        # Reset demand history
        self.demand_history.clear()
        for _ in range(30):
            self.demand_history.append(self.demand_generator(0))

        return self._get_state()

    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool]:
        """
        Execute one step in the environment

        Args:
            action: Order quantity (discretized)

        Returns:
            next_state, reward, done
        """
        # Convert action to order quantity
        order_quantity = int(action)

        # Generate demand for current period
        demand = self.demand_generator(self.time_step)
        self.demand_history.append(demand)

        # Process deliveries (orders placed lead_time ago)
        delivered = self.outstanding_orders.pop(0)
        self.inventory_level += delivered

        # Satisfy demand
        sales = min(self.inventory_level, demand)
        self.inventory_level -= sales
        stockout = demand - sales

        # Calculate costs
        holding_cost = self.holding_cost * max(0, self.inventory_level)
        stockout_cost = self.stockout_cost * stockout
        ordering_cost = self.ordering_cost * (1 if order_quantity > 0 else 0)

        total_cost = holding_cost + stockout_cost + ordering_cost

        # Place new order
        if order_quantity > 0:
            # Ensure inventory doesn't exceed capacity
            max_order = self.max_inventory - self.inventory_level - sum(self.outstanding_orders)
            order_quantity = max(0, min(order_quantity, max_order))

        self.outstanding_orders.append(order_quantity)

        # Reward is negative cost (we want to minimize cost)
        reward = -total_cost

        # Check if episode is done (arbitrary episode length)
        self.time_step += 1
        done = self.time_step >= 365  # One year episodes

        next_state = self._get_state()

        return next_state, reward, done

    def _get_state(self) -> jnp.ndarray:
        """Get current state representation"""
        # State includes:
        # - Current inventory level (normalized)
        # - Outstanding orders (normalized)
        # - Recent demand history statistics
        # - Time features (seasonality)

        # Normalize inventory and orders
        inv_norm = self.inventory_level / self.max_inventory
        orders_norm = np.array(self.outstanding_orders) / self.max_inventory

        # Demand statistics
        recent_demand = list(self.demand_history)[-7:]  # Last week
        demand_mean = np.mean(recent_demand) / 100.0  # Normalize
        demand_std = np.std(recent_demand) / 100.0

        # Time features
        day_of_year = (self.time_step % 365) / 365.0
        day_of_week = (self.time_step % 7) / 7.0

        # Combine state features
        state = jnp.array([
            inv_norm,
            *orders_norm,
            demand_mean,
            demand_std,
            day_of_year,
            day_of_week
        ], dtype=jnp.float32)

        return state


class DQNInventoryMethod(RLMethod):
    """DQN-based inventory management method"""

    def __init__(self,
                 state_dim: int = 6,
                 num_actions: int = 21,
                 hidden_sizes: Tuple[int, ...] = (256, 256),
                 learning_rate: float = 0.001,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 10,
                 gamma: float = 0.99):
        """
        Initialize DQN inventory method

        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate for Q-network
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            memory_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            gamma: Discount factor
        """
        super().__init__("DQN")
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        # Networks and training state
        self.q_network = None
        self.target_network = None
        self.state = None
        self.target_params = None
        self.replay_buffer = ReplayBuffer(memory_size)

        # Training tracking
        self.training_step = 0
        self.episode_rewards = []

        # Environment
        self.environment = None

    def build_agent(self) -> QNetwork:
        """Build Q-network architecture"""
        return QNetwork(
            hidden_sizes=self.hidden_sizes,
            num_actions=self.num_actions
        )

    def fit(self,
            demand_history: np.ndarray,
            external_features: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Fit DQN agent using historical demand for environment setup

        Args:
            demand_history: Historical demand for creating realistic environment
            external_features: Not used for DQN
        """
        # Create demand generator from historical data
        def demand_generator(time_step):
            # Use historical demand pattern with some randomness
            base_idx = time_step % len(demand_history)
            base_demand = demand_history[base_idx]
            noise = np.random.normal(0, 0.1 * base_demand)
            return max(1, int(base_demand + noise))

        # Create environment
        self.environment = InventoryEnvironment(demand_generator=demand_generator)

        # Build networks
        self.q_network = self.build_agent()

        # Initialize networks
        rng = random.PRNGKey(42)
        dummy_state = jnp.zeros((1, self.state_dim))
        params = self.q_network.init(rng, dummy_state)

        # Create optimizer
        optimizer = optax.adam(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.q_network.apply,
            params=params,
            tx=optimizer
        )

        # Initialize target network
        self.target_params = params

        self._is_fitted = True

    def train_agent(self, num_episodes: int, fast_mode: bool = False) -> None:
        """
        Train DQN agent for specified number of episodes

        Args:
            num_episodes: Number of training episodes
            fast_mode: If True, use faster training with fewer steps per episode
        """
        if not self.is_fitted:
            raise ValueError("Agent must be initialized before training")

        rng = random.PRNGKey(42)
        max_steps_per_episode = 50 if fast_mode else 200  # 限制每episode步数以加快训练

        for episode in range(num_episodes):
            # Reset environment
            rng, reset_rng = random.split(rng)
            state = self.environment.reset(reset_rng)
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                # Select action using epsilon-greedy policy
                rng, action_rng = random.split(rng)
                action = self._select_action(state, action_rng)

                # Execute action
                next_state, reward, done = self.environment.step(action)

                # Store experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                self.replay_buffer.push(experience)

                # Train on batch (批量训练以提高效率)
                if len(self.replay_buffer) >= self.batch_size:
                    # 每4步训练一次，而不是每步都训练
                    if steps % 4 == 0:
                        rng, train_rng = random.split(rng)
                        self._train_step(train_rng)

                state = next_state
                episode_reward += reward
                self.training_step += 1
                steps += 1

            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.target_params = self.state.params

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Track episode reward
            self.episode_rewards.append(episode_reward)

            # Print progress (减少打印频率)
            print_interval = max(1, num_episodes // 5)  # 只打印5次
            if episode % print_interval == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                print(f"Episode {episode}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")

    def _select_action(self, state: jnp.ndarray, rng: jax.random.PRNGKey) -> int:
        """Select action using epsilon-greedy policy"""
        if random.uniform(rng) < self.epsilon:
            # Random action
            action = random.randint(rng, (), 0, self.num_actions)
        else:
            # Greedy action
            state_batch = jnp.expand_dims(state, 0)
            q_values = self.q_network.apply(self.state.params, state_batch)
            action = jnp.argmax(q_values[0])

        return int(action)

    def _train_step(self, rng: jax.random.PRNGKey):
        """Single training step with experience replay"""
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, rng
        )

        @jit
        def loss_and_grad_fn(params, target_params, states, actions, rewards, next_states, dones):
            def loss_fn(params):
                # Current Q-values
                q_values = self.q_network.apply(params, states)
                current_q = q_values[jnp.arange(len(actions)), actions]

                # Target Q-values
                next_q_values = self.q_network.apply(target_params, next_states)
                next_q_max = jnp.max(next_q_values, axis=1)

                # Compute target
                target_q = rewards + self.gamma * next_q_max * (1 - dones)

                # MSE loss
                loss = jnp.mean((current_q - target_q) ** 2)
                return loss

            return jax.value_and_grad(loss_fn)(params)

        # Update Q-network
        loss, grads = loss_and_grad_fn(self.state.params, self.target_params, states, actions, rewards, next_states, dones)
        self.state = self.state.apply_gradients(grads=grads)

        return loss

    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        DQN doesn't explicitly predict demand, return mean from history

        Args:
            current_state: Current inventory state
            horizon: Number of periods ahead

        Returns:
            Simple demand prediction based on history
        """
        if len(current_state.demand_history) > 0:
            mean_demand = np.mean(current_state.demand_history[-7:])  # Use recent week
        else:
            mean_demand = 50.0  # Default

        return np.full(horizon, mean_demand)

    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend inventory action using trained DQN policy

        Args:
            current_state: Current inventory state

        Returns:
            Recommended inventory action
        """
        if not self.is_fitted:
            raise ValueError("DQN agent must be trained before recommendation")

        # Convert inventory state to DQN state representation
        state_vector = self._convert_state(current_state)

        # Get Q-values and select best action
        state_batch = jnp.expand_dims(state_vector, 0)
        q_values = self.q_network.apply(self.state.params, state_batch)
        action_idx = jnp.argmax(q_values[0])

        # Convert action index to order quantity
        order_quantity = float(action_idx)

        return InventoryAction(order_quantity=order_quantity)

    def _convert_state(self, inventory_state: InventoryState) -> jnp.ndarray:
        """Convert InventoryState to DQN state representation"""
        # Normalize inventory level
        inv_norm = inventory_state.inventory_level / 200.0  # Assume max 200

        # Outstanding orders - ensure we always have the correct number of features
        # The DQN environment uses lead_time=1, so we need to match that with a single outstanding order
        # But to match the 8-dimensional state, we need to pad to match the expected lead time
        if hasattr(inventory_state, 'outstanding_orders_list'):
            # Multiple outstanding orders (as array)
            orders_list = inventory_state.outstanding_orders_list
        else:
            # Single outstanding order value - convert to list with lead_time=1
            orders_list = [inventory_state.outstanding_orders]

        # Ensure we have exactly 1 outstanding order (lead time = 1) as expected by the environment
        if len(orders_list) == 1:
            orders_norm = np.array(orders_list) / 200.0
        else:
            # Take only the first order if multiple, or pad if none
            orders_norm = np.array([orders_list[0] if orders_list else 0.0]) / 200.0

        # Demand statistics
        if len(inventory_state.demand_history) > 0:
            recent_demand = inventory_state.demand_history[-7:]
            demand_mean = np.mean(recent_demand) / 100.0
            demand_std = np.std(recent_demand) / 100.0
        else:
            demand_mean = 0.5
            demand_std = 0.1

        # Time features
        day_of_year = (inventory_state.time_step % 365) / 365.0
        day_of_week = (inventory_state.time_step % 7) / 7.0

        # Combine state features to match environment's 6-dimensional state
        # Format: [inv_norm, order_1, demand_mean, demand_std, day_of_year, day_of_week]
        # This gives us exactly 6 features to match the environment
        state = jnp.array([
            inv_norm,                    # 1: inventory level
            orders_norm[0],              # 2: outstanding orders (lead time = 1)
            demand_mean,                 # 3: recent demand mean
            demand_std,                  # 4: recent demand std
            day_of_year,                 # 5: seasonal feature
            day_of_week                  # 6: weekly feature
        ], dtype=jnp.float32)

        return state

    def get_parameters(self) -> Dict[str, Any]:
        """Get DQN method parameters"""
        return {
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'hidden_sizes': self.hidden_sizes,
            'learning_rate': self.learning_rate,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'gamma': self.gamma,
            'current_epsilon': self.epsilon,
            'training_steps': self.training_step,
            'episodes_trained': len(self.episode_rewards)
        }


if __name__ == "__main__":
    # Example usage and testing
    print("🎮 DQN Inventory Management - Example Usage")
    print("=" * 50)

    # Generate sample demand data
    np.random.seed(42)
    demand_history = np.random.gamma(2, 25, 365)

    print(f"Sample demand statistics:")
    print(f"  Mean: {np.mean(demand_history):.2f}")
    print(f"  Std: {np.std(demand_history):.2f}")

    # Initialize DQN agent
    dqn_agent = DQNInventoryMethod(
        state_dim=6,
        num_actions=21,
        hidden_sizes=(128, 128),
        epsilon_decay=0.99,
        memory_size=5000
    )

    print(f"\n🔧 Initializing DQN agent...")
    try:
        # Fit agent (initialize environment and networks)
        dqn_agent.fit(demand_history)

        print(f"✅ Agent initialized")
        print(f"🎯 Training for 50 episodes...")

        # Train agent (reduced episodes for demo)
        dqn_agent.train_agent(num_episodes=50)

        # Test recommendation
        current_state = InventoryState(
            inventory_level=75.0,
            outstanding_orders=25.0,
            demand_history=demand_history[-30:],
            time_step=365
        )

        action = dqn_agent.recommend_action(current_state)
        demand_pred = dqn_agent.predict_demand(current_state, horizon=7)

        print(f"\n📊 Test Results:")
        print(f"  Current inventory: {current_state.inventory_level}")
        print(f"  Outstanding orders: {current_state.outstanding_orders}")
        print(f"  DQN recommendation: Order {action.order_quantity:.0f} units")
        print(f"  Demand prediction: {demand_pred}")

        # Training statistics
        params = dqn_agent.get_parameters()
        print(f"\n📈 Training Statistics:")
        print(f"  Episodes trained: {params['episodes_trained']}")
        print(f"  Training steps: {params['training_steps']}")
        print(f"  Final epsilon: {params['current_epsilon']:.3f}")

        if len(dqn_agent.episode_rewards) > 0:
            print(f"  Average reward (last 10): {np.mean(dqn_agent.episode_rewards[-10:]):.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ DQN testing completed!")