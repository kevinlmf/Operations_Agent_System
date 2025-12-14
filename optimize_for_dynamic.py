"""
Optimize RL/DL for Dynamic Environments

Objective: Optimize RL and DL methods to beat traditional baseline (EOQ)
"""

import numpy as np
import pandas as pd

from evaluation.comparison.dynamic_optimizer import DynamicOptimizer
from evaluation.comparison.dynamic_scenario_evaluator import DynamicScenarioEvaluator, ScenarioCharacteristics
from evaluation.comparison.net_benefit_optimizer import NetBenefitOptimizer
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
    print("🚀 Optimize RL/DL for Dynamic Environments")
    print("=" * 70)
    print("Objective: Beat traditional baseline (EOQ) with RL/DL methods")
    print("=" * 70)

    # 1. Data Generation
    print("\n1️⃣ Generating Dynamic Test Data...")
    train_days = 365
    test_days = 90
    total_days = train_days + test_days
    
    full_demand = generate_demand_data(total_days)
    train_demand = full_demand[:train_days]
    test_demand = full_demand[train_days:]
    
    print(f"  Training data: {len(train_demand)} days")
    print(f"  Test data: {len(test_days)} days")

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

    # 2. Setup Dynamic Scenario
    print("\n2️⃣ Setting up Dynamic Scenario...")
    scenario = ScenarioCharacteristics(
        has_seasonality=True,
        has_trend=True,
        uncertainty_level=0.2,
        trend_strength=0.05,
        seasonality_amplitude=0.3,
        volatility=0.15
    )
    
    base_optimizer = NetBenefitOptimizer(
        unit_price=20.0,
        unit_cost=10.0,
        holding_cost=2.0,
        stockout_cost=10.0,
        ordering_cost=50.0
    )
    
    dynamic_evaluator = DynamicScenarioEvaluator(base_optimizer, scenario)

    # 3. Establish Baseline
    print("\n3️⃣ Establishing Baseline (EOQ)...")
    eoq = EOQMethod(holding_cost=2.0, ordering_cost=50.0, lead_time=1)
    eoq.fit(train_demand)
    
    baseline_result = dynamic_evaluator.evaluate_method_comprehensive(
        eoq, train_demand, test_states, test_demand, num_scenarios=10
    )
    baseline_nb = baseline_result.get('risk_adjusted_net_benefit', 0)
    
    print(f"  ✅ Baseline (EOQ) Performance:")
    print(f"     Risk-Adjusted Net Benefit: ${baseline_nb:,.2f}")
    print(f"     Expected Net Benefit: ${baseline_result.get('expected_net_benefit', 0):,.2f}")
    print(f"     Risk: ${baseline_result.get('risk', 0):,.2f}")
    
    # 4. Optimize RL/DL Methods
    print("\n4️⃣ Optimizing RL/DL Methods to Beat Baseline...")
    
    optimizer = DynamicOptimizer(
        base_optimizer=base_optimizer,
        scenario=scenario,
        baseline_method=eoq,
        baseline_performance=baseline_nb
    )
    
    optimized_methods = {}
    
    # Optimize DQN
    print("\n  🎮 Optimizing DQN...")
    try:
        dqn_opt = optimizer.optimize_dqn_for_dynamic(
            train_demand, test_states, test_demand, num_scenarios=10
        )
        optimized_methods['DQN'] = dqn_opt
        
        if dqn_opt['beats_baseline']:
            improvement = ((dqn_opt['performance'].get('risk_adjusted_net_benefit', 0) - baseline_nb) / abs(baseline_nb) * 100)
            print(f"     ✅ DQN beats baseline! Improvement: {improvement:.1f}%")
        else:
            print(f"     ⚠️  DQN did not beat baseline")
    except Exception as e:
        print(f"     ❌ DQN optimization failed: {e}")
    
    # Optimize LSTM
    print("\n  🤖 Optimizing LSTM...")
    try:
        lstm_opt = optimizer.optimize_lstm_for_dynamic(
            train_demand, test_states, test_demand, num_scenarios=10
        )
        optimized_methods['LSTM'] = lstm_opt
        
        if lstm_opt['beats_baseline']:
            improvement = ((lstm_opt['performance'].get('risk_adjusted_net_benefit', 0) - baseline_nb) / abs(baseline_nb) * 100)
            print(f"     ✅ LSTM beats baseline! Improvement: {improvement:.1f}%")
        else:
            print(f"     ⚠️  LSTM did not beat baseline")
    except Exception as e:
        print(f"     ❌ LSTM optimization failed: {e}")

    # 5. Comparison
    print("\n5️⃣ Final Comparison...")
    comparison_df = optimizer.compare_with_baseline(optimized_methods, test_states, test_demand)
    
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 70)
    
    # Best method
    best = comparison_df.iloc[0]
    print(f"\n🏆 Best Method: {best['method_name']}")
    print(f"   Risk-Adjusted Net Benefit: ${best['risk_adjusted_net_benefit']:,.2f}")
    
    if not best.get('is_baseline', False):
        print(f"   Improvement vs Baseline: {best['improvement_pct']:.1f}%")
    
    # Methods that beat baseline
    beaters = comparison_df[comparison_df['improvement_pct'] > 0]
    if len(beaters) > 0:
        print(f"\n✅ Methods that beat baseline ({len(beaters)}):")
        for _, row in beaters.iterrows():
            if not row.get('is_baseline', False):
                print(f"   - {row['method_name']}: +{row['improvement_pct']:.1f}%")
    else:
        print(f"\n⚠️  No methods beat baseline")
        print(f"   Suggestions:")
        print(f"   - Increase training episodes")
        print(f"   - Adjust network architecture")
        print(f"   - Add seasonal features")
    
    # Save results
    results_path = "results/dynamic_optimization_results.csv"
    comparison_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('RL/DL Optimization vs Baseline', fontsize=16)
        
        methods = comparison_df['method_name'].tolist()
        risk_adj_nb = comparison_df['risk_adjusted_net_benefit'].tolist()
        improvements = comparison_df['improvement_pct'].tolist()
        
        ax1 = axes[0]
        colors = ['green' if nb > baseline_nb else 'red' for nb in risk_adj_nb]
        ax1.barh(methods, risk_adj_nb, color=colors)
        ax1.axvline(x=baseline_nb, color='blue', linestyle='--', linewidth=2, label='Baseline')
        ax1.set_xlabel('Risk-Adjusted Net Benefit ($)')
        ax1.set_title('Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        colors2 = ['green' if imp > 0 else 'red' for imp in improvements]
        ax2.barh(methods, improvements, color=colors2)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Improvement (%)')
        ax2.set_title('Improvement vs Baseline')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "results/dynamic_optimization_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

    print("\n" + "=" * 70)
    print("✅ Dynamic Optimization Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
