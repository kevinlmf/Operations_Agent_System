"""
Basic System Evaluation

Compare traditional, RL, and Claude Agent methods for inventory optimization.
Includes Claude Agent Skills for intelligent decision making.
"""

import numpy as np
import pandas as pd
import os

from evaluation.comparison.evaluator import EnhancedInventoryEvaluator
from agent.traditional.eoq import EOQMethod
from agent.traditional.safety_stock import SafetyStockMethod
from agent.rl_methods.dqn import DQNInventoryMethod
from agent.claude_agent import ClaudeAgentInventoryMethod
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
    try:
        eoq = EOQMethod(holding_cost=2.0, ordering_cost=50.0, lead_time=1)
        eoq.fit(train_demand)
        methods['EOQ'] = eoq
        print(f"  ✅ EOQ fitted: Q*={eoq.optimal_order_quantity:.1f}, r={eoq.reorder_point:.1f}")
    except Exception as e:
        print(f"  ❌ EOQ failed: {e}")

    # Traditional Method: Safety Stock
    print("\n3️⃣ Setting up Safety Stock Method...")
    try:
        safety_stock = SafetyStockMethod(service_level=0.95, method='normal')
        safety_stock.fit(train_demand)
        methods['Safety_Stock'] = safety_stock
        print(f"  ✅ Safety Stock fitted")
    except Exception as e:
        print(f"  ❌ Safety Stock failed: {e}")

    # RL Method: DQN
    print("\n4️⃣ Setting up & Training DQN Agent...")
    try:
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
        dqn.train_agent(num_episodes=10, fast_mode=True)
        methods['DQN'] = dqn
        print("  ✅ DQN training completed")
    except Exception as e:
        print(f"  ❌ DQN failed: {e}")

    # Claude Agent Method (Agentic AI with Tool Use)
    print("\n5️⃣ Setting up Claude Agent (Agentic AI)...")
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        claude_agent = ClaudeAgentInventoryMethod(
            api_key=api_key,
            model="claude-sonnet-4-20250514",
            holding_cost=2.0,
            ordering_cost=50.0,
            stockout_cost=10.0,
            lead_time=1,
            service_level=0.95,
            use_extended_thinking=True,
            max_tool_iterations=5,
            fallback_mode=True
        )
        claude_agent.fit(train_demand)
        methods['Claude_Agent'] = claude_agent
        
        params = claude_agent.get_parameters()
        api_status = "API Connected" if params.get("api_available") else "Fallback Mode"
        print(f"  ✅ Claude Agent initialized - {api_status}")
        
        # Show demand analysis
        demand_stats = params.get('demand_stats', {})
        print(f"     📊 Demand Pattern: {demand_stats.get('demand_pattern', 'unknown')}")
        print(f"     📈 Trend: {demand_stats.get('trend_direction', 'unknown')}")
        print(f"     🌊 Seasonality: {demand_stats.get('has_seasonality', False)}")
    except Exception as e:
        print(f"  ❌ Claude Agent failed: {e}")

    if not methods:
        print("\n❌ No methods available for evaluation")
        return

    # 6. Evaluation
    print(f"\n6️⃣ Running Comparative Evaluation...")
    print(f"   Comparing {len(methods)} methods: {list(methods.keys())}")
    
    evaluator = EnhancedInventoryEvaluator(
        holding_cost=2.0,
        stockout_cost=10.0,
        ordering_cost=50.0
    )

    results_df = evaluator.compare_methods(methods, test_states, test_demand)

    # 7. Reporting
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    evaluator.print_comparison_summary(results_df)
    
    # Highlight Claude Agent performance
    if 'Claude_Agent' in methods:
        print("\n🤖 Claude Agent Analysis:")
        claude_result = results_df[results_df['method_name'] == 'Claude_Agent']
        if not claude_result.empty:
            row = claude_result.iloc[0]
            print(f"   Total Cost: ${row.get('total_cost', 0):,.2f}")
            print(f"   Service Level: {row.get('service_level', 0)*100:.1f}%")
            print(f"   Forecast Accuracy: {row.get('forecast_accuracy', 0)*100:.1f}%")
            
            # Compare with best traditional method
            trad_methods = results_df[results_df['method_name'].isin(['EOQ', 'Safety_Stock'])]
            if not trad_methods.empty:
                best_trad = trad_methods.loc[trad_methods['total_cost'].idxmin()]
                cost_diff = row.get('total_cost', 0) - best_trad.get('total_cost', 0)
                if cost_diff < 0:
                    print(f"   ✅ Beats {best_trad['method_name']} by ${abs(cost_diff):,.2f}")
                else:
                    print(f"   📊 Within ${cost_diff:,.2f} of {best_trad['method_name']}")
    
    # Save results
    results_path = "results/evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to {results_path}")

    # Create plots
    try:
        plot_path = "results/evaluation_plots.png"
        evaluator.create_performance_plots(results_df, save_path=plot_path)
        print(f"📊 Plots saved to {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")
    
    print("\n" + "=" * 60)
    print("✅ System Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
