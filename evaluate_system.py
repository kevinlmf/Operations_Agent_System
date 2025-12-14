"""
Basic System Evaluation

Compare traditional and RL methods for inventory optimization.
"""

import numpy as np
import pandas as pd

from evaluation.comparison.evaluator import EnhancedInventoryEvaluator
from agent.traditional.eoq import EOQMethod
from agent.rl_methods.dqn import DQNInventoryMethod
from goal.interfaces import InventoryState


def generate_demand_data(days, mean=50, std=10, seasonality=True):
    """Generate synthetic demand data"""
    np.random.seed(42)
    base_demand = np.random.normal(mean, std, days)
    
    if seasonality:
        t = np.arange(days)
        seasonal = 10 * np.sin(2 * np.pi * t / 365.25)
        weekly = 5 * np.sin(2 * np.pi * t / 7)
        base_demand += seasonal + weekly
        
    return np.maximum(base_demand, 0).astype(int)


def main():
    print("🚀 Starting System Evaluation...")
    print("=" * 50)

    # 1. Data Generation
    print("\n1️⃣ Generating Test Data...")
    train_days = 365
    test_days = 90
    total_days = train_days + test_days
    
    full_demand = generate_demand_data(total_days)
    train_demand = full_demand[:train_days]
    test_demand = full_demand[train_days:]
    
    print(f"  Training data: {len(train_demand)} days")
    print(f"  Test data: {len(test_demand)} days")
    print(f"  Mean demand: {np.mean(full_demand):.1f}")

    # Create test states
    test_states = []
    current_inv = 100
    for i in range(len(test_demand)):
        state = InventoryState(
            inventory_level=current_inv,
            outstanding_orders=0,
            demand_history=train_demand[-30:],
            time_step=train_days + i
        )
        test_states.append(state)

    # 2. Method Initialization & Training
    methods = {}

    # Traditional Method: EOQ
    print("\n2️⃣ Setting up EOQ Method...")
    eoq = EOQMethod(holding_cost=2.0, ordering_cost=50.0, lead_time=1)
    eoq.fit(train_demand)
    methods['EOQ'] = eoq
    print(f"  EOQ fitted: Q*={eoq.optimal_order_quantity:.1f}, r={eoq.reorder_point:.1f}")

    # RL Method: DQN
    print("\n3️⃣ Setting up & Training DQN Agent...")
    dqn = DQNInventoryMethod(
        state_dim=6,
        num_actions=21,
        hidden_sizes=(64, 64),
        learning_rate=0.001,
        memory_size=10000,
        batch_size=32
    )
    
    print("  Training DQN (this may take a moment)...")
    dqn.fit(train_demand)
    dqn.train_agent(num_episodes=10)
    methods['DQN'] = dqn
    print("  DQN training completed")

    # 3. Evaluation
    print("\n4️⃣ Running Comparative Evaluation...")
    evaluator = EnhancedInventoryEvaluator(
        holding_cost=2.0,
        stockout_cost=10.0,
        ordering_cost=50.0
    )

    results_df = evaluator.compare_methods(methods, test_states, test_demand)

    # 4. Reporting
    print("\n" + "=" * 50)
    evaluator.print_comparison_summary(results_df)
    
    # Save results
    results_path = "results/evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to {results_path}")

    # Create plots
    try:
        plot_path = "results/evaluation_plots.png"
        evaluator.create_performance_plots(results_df, save_path=plot_path)
    except Exception as e:
        print(f"Could not create plots: {e}")


if __name__ == "__main__":
    main()
