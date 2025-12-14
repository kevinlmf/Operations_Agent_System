"""
Net Benefit Optimization Evaluation

Objective: Find optimal inventory method maximizing Net Benefit = Revenue - Total Cost
Constraint: Total Cost <= Cost Constraint
"""

import numpy as np
import pandas as pd

from evaluation.comparison.net_benefit_optimizer import NetBenefitOptimizer
from evaluation.comparison.evaluator import EnhancedInventoryEvaluator
from evaluation.comparison.parameter_optimizer import ParameterOptimizer
from agent.traditional.eoq import EOQMethod
from agent.traditional.safety_stock import SafetyStockMethod
from agent.rl_methods.dqn import DQNInventoryMethod
from agent.ml_methods.lstm import LSTMInventoryMethod
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
    print("🚀 Net Benefit Optimization Evaluation")
    print("=" * 70)
    print("Objective: Maximize Net Benefit = Revenue - Total Cost")
    print("=" * 70)

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
    print("\n2️⃣ Setting up Methods...")
    methods = {}

    # Traditional Method: EOQ
    print("  📊 Traditional Method: EOQ...")
    try:
        eoq = EOQMethod(holding_cost=2.0, ordering_cost=50.0, lead_time=1)
        eoq.fit(train_demand)
        methods['EOQ'] = eoq
        print(f"    ✅ EOQ fitted: Q*={eoq.optimal_order_quantity:.1f}")
    except Exception as e:
        print(f"    ❌ EOQ failed: {e}")

    # Traditional Method: Safety Stock
    print("  📊 Traditional Method: Safety Stock...")
    try:
        safety_stock = SafetyStockMethod(service_level=0.95, method='normal')
        safety_stock.fit(train_demand)
        methods['Safety_Stock'] = safety_stock
        print(f"    ✅ Safety Stock fitted")
    except Exception as e:
        print(f"    ❌ Safety Stock failed: {e}")

    # ML Method: LSTM
    print("  🤖 ML Method: LSTM...")
    try:
        import jax
        devices = jax.devices()
        device_type = devices[0].device_kind if devices else "CPU"
        print(f"    🔧 JAX Device: {device_type}")
        
        lstm = LSTMInventoryMethod(
            sequence_length=30,
            hidden_size=32,
            num_layers=1,
            epochs=10,
            batch_size=64
        )
        lstm.fit(train_demand)
        methods['LSTM'] = lstm
        print(f"    ✅ LSTM trained")
    except Exception as e:
        print(f"    ❌ LSTM failed: {e}")

    # RL Method: DQN
    print("  🎮 RL Method: DQN...")
    try:
        import jax
        devices = jax.devices()
        device_type = devices[0].device_kind if devices else "CPU"
        print(f"    🔧 JAX Device: {device_type}")
        
        dqn = DQNInventoryMethod(
            state_dim=6,
            num_actions=21,
            hidden_sizes=(32, 32),
            learning_rate=0.001,
            memory_size=5000,
            batch_size=64
        )
        dqn.fit(train_demand)
        dqn.train_agent(num_episodes=10, fast_mode=True)
        methods['DQN'] = dqn
        print(f"    ✅ DQN trained")
    except Exception as e:
        print(f"    ❌ DQN failed: {e}")

    if not methods:
        print("\n❌ No methods available for evaluation")
        return

    # 3. Net Benefit Evaluation
    print(f"\n3️⃣ Running Net Benefit Analysis...")
    print(f"   Comparing {len(methods)} methods: {list(methods.keys())}")
    
    optimizer = NetBenefitOptimizer(
        unit_price=20.0,
        unit_cost=10.0,
        holding_cost=2.0,
        stockout_cost=10.0,
        ordering_cost=50.0,
        cost_constraint=None,
        periods_per_year=365
    )

    results_df = optimizer.compare_methods_net_benefit(
        methods, test_states, test_demand, num_periods=len(test_demand)
    )

    # 4. Find Optimal Method
    print(f"\n4️⃣ Finding Optimal Method...")
    try:
        best_method_name, best_result = optimizer.find_optimal_method(
            methods, test_states, test_demand, num_periods=len(test_demand)
        )
        print(f"   🏆 Best Method: {best_method_name}")
        print(f"   Net Benefit: ${best_result.net_benefit:,.2f}")
        print(f"   ROI: {best_result.roi:.1f}%")
    except Exception as e:
        print(f"   ⚠️  Could not determine optimal method: {e}")

    # 5. Reporting
    print("\n" + "=" * 70)
    optimizer.print_comparison_summary(results_df)
    
    # Save results
    results_path = "results/net_benefit_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Net Benefit Comparison Analysis', fontsize=16)
        
        ax1 = axes[0, 0]
        methods_list = results_df['method_name'].tolist()
        net_benefits = results_df['net_benefit'].tolist()
        colors = ['green' if b > 0 else 'red' for b in net_benefits]
        ax1.barh(methods_list, net_benefits, color=colors)
        ax1.set_xlabel('Net Benefit ($)')
        ax1.set_title('Net Benefit by Method')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        costs = results_df['total_cost'].tolist()
        ax2.barh(methods_list, costs, color='orange')
        ax2.set_xlabel('Total Cost ($)')
        ax2.set_title('Total Cost by Method')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        rois = results_df['roi'].tolist()
        ax3.barh(methods_list, rois, color='blue')
        ax3.set_xlabel('ROI (%)')
        ax3.set_title('Return on Investment')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ratios = results_df['cost_benefit_ratio'].tolist()
        ax4.barh(methods_list, ratios, color='purple')
        ax4.set_xlabel('Cost-Benefit Ratio')
        ax4.set_title('Revenue / Total Cost')
        ax4.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='Break-even')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "results/net_benefit_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

    print("\n" + "=" * 70)
    print("✅ Net Benefit Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
