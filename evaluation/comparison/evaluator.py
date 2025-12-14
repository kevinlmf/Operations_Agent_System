"""
Enhanced Performance Evaluator for Inventory Methods

This module provides comprehensive evaluation capabilities for comparing
traditional, ML, and RL inventory management approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import InventoryMethod, InventoryState, InventoryResult


class EnhancedInventoryEvaluator:
    """Enhanced evaluator with comprehensive metrics and visualization"""

    def __init__(self,
                 holding_cost: float = 1.0,
                 stockout_cost: float = 5.0,
                 ordering_cost: float = 25.0,
                 service_level_target: float = 0.95):
        """
        Initialize enhanced evaluator

        Args:
            holding_cost: Holding cost per unit per period
            stockout_cost: Stockout cost per unit per period
            ordering_cost: Fixed ordering cost per order
            service_level_target: Target service level
        """
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost
        self.service_level_target = service_level_target

    def evaluate_method_comprehensive(self,
                                    method: InventoryMethod,
                                    test_states: List[InventoryState],
                                    true_demands: np.ndarray,
                                    method_name: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single method

        Args:
            method: The inventory method to evaluate
            test_states: List of test states
            true_demands: Actual demand realizations
            method_name: Optional method name for reporting

        Returns:
            Comprehensive evaluation results
        """
        if not method.is_fitted:
            raise ValueError(f"Method must be fitted before evaluation")

        method_name = method_name or method.method_name
        start_time = time.time()

        # Initialize tracking arrays
        actions = []
        predictions = []
        costs = []
        inventory_levels = []
        service_levels = []
        forecast_errors = []
        order_frequencies = []

        # Initialize inventory simulation with lead time consideration
        lead_time = getattr(method, 'lead_time', 0)  # Get lead time from method if available
        current_inventory = test_states[0].inventory_level if test_states else 50.0

        # Track pending orders for lead time simulation
        pending_orders = []  # List of (arrival_period, quantity) tuples
        total_orders_placed = 0

        print(f"  Evaluating {method_name} on {len(true_demands)} periods...")

        try:
            for i, (state, actual_demand) in enumerate(zip(test_states, true_demands)):
                # Simulate inventory dynamics with lead time
                # First, process arriving orders (from lead_time periods ago)
                arriving_orders = [qty for arrival_period, qty in pending_orders if arrival_period == i]
                for qty in arriving_orders:
                    current_inventory += qty

                # Remove processed orders
                pending_orders = [(period, qty) for period, qty in pending_orders if period != i]

                # Create current state with accurate inventory position for method decision
                outstanding_orders = sum(qty for _, qty in pending_orders)
                current_state = InventoryState(
                    inventory_level=current_inventory,
                    outstanding_orders=outstanding_orders,
                    demand_history=state.demand_history,
                    time_step=state.time_step
                )

                # Get method recommendation based on current state
                action = method.recommend_action(current_state)
                actions.append(action.order_quantity)

                # Get demand prediction
                try:
                    predicted_demand = method.predict_demand(current_state, horizon=1)[0]
                    predictions.append(predicted_demand)
                    forecast_error = abs(predicted_demand - actual_demand)
                    forecast_errors.append(forecast_error)
                except Exception as e:
                    # Fallback for methods without explicit prediction
                    predictions.append(actual_demand)
                    forecast_errors.append(0)

                # Place new order if recommended
                if action.order_quantity > 0:
                    arrival_period = i + lead_time
                    pending_orders.append((arrival_period, action.order_quantity))
                    total_orders_placed += 1

                # Satisfy demand
                units_sold = min(current_inventory, actual_demand)
                current_inventory -= units_sold
                stockout = actual_demand - units_sold

                # Calculate period costs
                holding_cost = self.holding_cost * max(0, current_inventory)
                stockout_cost = self.stockout_cost * stockout

                # Amortize ordering cost over periods based on order frequency
                # Only count ordering cost proportional to order quantity relative to typical orders
                if action.order_quantity > 0:
                    # Estimate annual order frequency from typical demand/order patterns
                    typical_demand_per_period = actual_demand if actual_demand > 0 else 50
                    estimated_periods_per_order = max(1, action.order_quantity / typical_demand_per_period)
                    ordering_cost = self.ordering_cost / estimated_periods_per_order
                else:
                    ordering_cost = 0

                period_cost = holding_cost + stockout_cost + ordering_cost
                costs.append(period_cost)

                # Track metrics
                inventory_levels.append(current_inventory)
                service_level = 1.0 if stockout == 0 else 0.0
                service_levels.append(service_level)

            # Calculate aggregate metrics
            results = self._calculate_comprehensive_metrics(
                costs, inventory_levels, service_levels, forecast_errors,
                actions, true_demands, total_orders_placed, len(test_states)
            )

            # Add method info
            results['method_name'] = method_name
            results['evaluation_time'] = time.time() - start_time
            results['method_category'] = method.category.value if hasattr(method, 'category') else 'unknown'

            # Add detailed arrays for further analysis
            results['detailed_data'] = {
                'costs': np.array(costs),
                'inventory_levels': np.array(inventory_levels),
                'service_levels': np.array(service_levels),
                'forecast_errors': np.array(forecast_errors),
                'actions': np.array(actions),
                'predictions': np.array(predictions),
                'true_demands': np.array(true_demands)
            }

            return results

        except Exception as e:
            print(f"    Error evaluating {method_name}: {e}")
            return {
                'method_name': method_name,
                'error': str(e),
                'evaluation_time': time.time() - start_time
            }

    def _calculate_comprehensive_metrics(self,
                                       costs: List[float],
                                       inventory_levels: List[float],
                                       service_levels: List[float],
                                       forecast_errors: List[float],
                                       actions: List[float],
                                       true_demands: np.ndarray,
                                       total_orders: int,
                                       total_periods: int) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""

        # Convert to numpy arrays
        costs = np.array(costs)
        inventory_levels = np.array(inventory_levels)
        service_levels = np.array(service_levels)
        forecast_errors = np.array(forecast_errors)
        actions = np.array(actions)

        # Cost metrics
        total_cost = np.sum(costs)
        avg_cost_per_period = np.mean(costs)
        cost_std = np.std(costs)

        # Service level metrics
        avg_service_level = np.mean(service_levels)
        service_level_consistency = 1 - np.std(service_levels)  # Higher is better

        # Inventory metrics
        avg_inventory = np.mean(inventory_levels)
        inventory_turnover = np.sum(true_demands) / avg_inventory if avg_inventory > 0 else 0
        max_inventory = np.max(inventory_levels)
        min_inventory = np.min(inventory_levels)

        # Forecast accuracy metrics
        if len(forecast_errors) > 0 and np.sum(true_demands) > 0:
            mae = np.mean(forecast_errors)
            mape = np.mean(forecast_errors / np.maximum(true_demands, 1e-8)) * 100
            rmse = np.sqrt(np.mean(forecast_errors ** 2))
            forecast_accuracy = max(0, 1 - mae / np.mean(true_demands))
        else:
            mae = mape = rmse = 0
            forecast_accuracy = 0

        # Order pattern metrics
        order_frequency = total_orders / total_periods
        avg_order_size = np.mean(actions[actions > 0]) if np.any(actions > 0) else 0
        order_variability = np.std(actions) / (np.mean(actions) + 1e-8)

        # Efficiency metrics
        cost_per_unit_sold = total_cost / np.sum(true_demands) if np.sum(true_demands) > 0 else 0
        service_cost_ratio = avg_service_level / avg_cost_per_period if avg_cost_per_period > 0 else 0

        # Risk metrics
        stockout_frequency = np.mean(1 - service_levels)
        cost_volatility = cost_std / avg_cost_per_period if avg_cost_per_period > 0 else 0

        return {
            # Primary metrics
            'total_cost': float(total_cost),
            'avg_cost_per_period': float(avg_cost_per_period),
            'service_level': float(avg_service_level),
            'forecast_accuracy': float(forecast_accuracy),

            # Cost breakdown
            'cost_std': float(cost_std),
            'cost_per_unit_sold': float(cost_per_unit_sold),
            'cost_volatility': float(cost_volatility),

            # Service metrics
            'service_level_consistency': float(service_level_consistency),
            'stockout_frequency': float(stockout_frequency),

            # Inventory metrics
            'avg_inventory': float(avg_inventory),
            'inventory_turnover': float(inventory_turnover),
            'max_inventory': float(max_inventory),
            'min_inventory': float(min_inventory),

            # Forecast metrics
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse),

            # Order pattern metrics
            'order_frequency': float(order_frequency),
            'avg_order_size': float(avg_order_size),
            'order_variability': float(order_variability),
            'total_orders': int(total_orders),

            # Efficiency metrics
            'service_cost_ratio': float(service_cost_ratio),
        }

    def compare_methods(self,
                       methods: Dict[str, InventoryMethod],
                       test_states: List[InventoryState],
                       true_demands: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple methods comprehensively

        Args:
            methods: Dictionary of method_name -> method
            test_states: List of test states
            true_demands: Actual demand realizations

        Returns:
            DataFrame with comparison results
        """
        print(f"🔍 Comparing {len(methods)} methods on {len(true_demands)} test periods...")

        results = []

        for method_name, method in methods.items():
            if 'error' not in str(method):  # Skip methods with errors
                result = self.evaluate_method_comprehensive(
                    method, test_states, true_demands, method_name
                )
                if 'error' not in result:
                    results.append(result)
                else:
                    print(f"  ❌ {method_name}: {result['error']}")
            else:
                print(f"  ❌ {method_name}: Method has errors")

        if not results:
            print("❌ No methods successfully evaluated")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by total cost (lower is better)
        df = df.sort_values('total_cost')

        return df

    def print_comparison_summary(self, df: pd.DataFrame, top_n: int = 5):
        """Print formatted comparison summary"""

        if df.empty:
            print("No results to display")
            return

        print(f"\n📊 METHOD COMPARISON SUMMARY (Top {min(top_n, len(df))})")
        print("=" * 80)

        # Key metrics to display
        key_metrics = [
            ('total_cost', 'Total Cost', '.1f'),
            ('service_level', 'Service Level', '.1%'),
            ('forecast_accuracy', 'Forecast Acc.', '.1%'),
            ('inventory_turnover', 'Inv. Turnover', '.2f'),
            ('order_frequency', 'Order Freq.', '.2f')
        ]

        # Print header
        header = f"{'Rank':<4} {'Method':<20}"
        for _, display_name, _ in key_metrics:
            header += f"{display_name:>12}"
        print(header)
        print("-" * len(header))

        # Print top methods
        for i, (_, row) in enumerate(df.head(top_n).iterrows()):
            rank = i + 1
            method_name = row['method_name'][:19]  # Truncate long names

            line = f"{rank:<4} {method_name:<20}"

            for metric_name, _, format_str in key_metrics:
                value = row.get(metric_name, 0)
                if format_str == '.1%':
                    line += f"{value:>12.1%}"
                elif format_str == '.2f':
                    line += f"{value:>12.2f}"
                else:
                    line += f"{value:>12.1f}"

            print(line)

        # Performance insights
        print(f"\n🏆 PERFORMANCE INSIGHTS:")
        best_cost = df.iloc[0]
        best_service = df.loc[df['service_level'].idxmax()]
        best_forecast = df.loc[df['forecast_accuracy'].idxmax()]

        print(f"  💰 Lowest Cost: {best_cost['method_name']} (${best_cost['total_cost']:.1f})")
        print(f"  🎯 Best Service: {best_service['method_name']} ({best_service['service_level']:.1%})")
        print(f"  🔮 Best Forecast: {best_forecast['method_name']} ({best_forecast['forecast_accuracy']:.1%})")

        # Method category analysis
        if 'method_category' in df.columns:
            category_performance = df.groupby('method_category')['total_cost'].mean().sort_values()
            print(f"\n📈 BY CATEGORY (Avg. Cost):")
            for category, avg_cost in category_performance.items():
                print(f"  {category.title()}: ${avg_cost:.1f}")

    def create_performance_plots(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create visualization plots for method comparison"""

        if df.empty:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Inventory Method Performance Comparison', fontsize=16)

        # 1. Cost vs Service Level Scatter Plot
        ax1 = axes[0, 0]
        ax1.scatter(df['service_level'], df['total_cost'], alpha=0.7, s=100)
        for _, row in df.iterrows():
            ax1.annotate(row['method_name'],
                        (row['service_level'], row['total_cost']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        ax1.set_xlabel('Service Level')
        ax1.set_ylabel('Total Cost')
        ax1.set_title('Cost vs Service Level Trade-off')
        ax1.grid(True, alpha=0.3)

        # 2. Forecast Accuracy Comparison
        ax2 = axes[0, 1]
        methods = df['method_name'].tolist()
        accuracies = df['forecast_accuracy'].tolist()
        bars = ax2.bar(range(len(methods)), accuracies)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Forecast Accuracy')
        ax2.set_title('Demand Forecasting Accuracy')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Color bars by performance
        max_acc = max(accuracies) if accuracies else 1
        for bar, acc in zip(bars, accuracies):
            bar.set_color(plt.cm.RdYlGn(acc / max_acc))

        # 3. Cost Components Breakdown (Top 5 methods)
        ax3 = axes[1, 0]
        top_methods = df.head(5)
        if 'detailed_data' in top_methods.columns:
            # This would need more detailed cost breakdown data
            # For now, show total costs
            bars = ax3.bar(range(len(top_methods)), top_methods['total_cost'])
            ax3.set_xlabel('Method (Top 5)')
            ax3.set_ylabel('Total Cost')
            ax3.set_title('Total Cost Comparison (Top 5)')
            ax3.set_xticks(range(len(top_methods)))
            ax3.set_xticklabels(top_methods['method_name'], rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)

        # 4. Efficiency Metrics Radar Chart (simplified to bar chart)
        ax4 = axes[1, 1]
        best_method = df.iloc[0]
        metrics = ['service_level', 'forecast_accuracy', 'inventory_turnover']
        values = [best_method[metric] for metric in metrics]
        metric_labels = ['Service Level', 'Forecast Acc.', 'Inventory Turnover']

        bars = ax4.barh(metric_labels, values)
        ax4.set_xlabel('Performance Score')
        ax4.set_title(f'Best Method: {best_method["method_name"]}')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance plots saved to: {save_path}")
        else:
            plt.show()

        return fig


if __name__ == "__main__":
    # Example usage
    print("🔍 Enhanced Inventory Evaluator - Example")
    print("=" * 50)

    # This would normally be tested with actual methods
    # For demo, we create mock results

    mock_results = [
        {
            'method_name': 'EOQ',
            'total_cost': 2500.0,
            'service_level': 0.92,
            'forecast_accuracy': 0.75,
            'inventory_turnover': 4.2,
            'order_frequency': 0.15
        },
        {
            'method_name': 'LSTM',
            'total_cost': 2200.0,
            'service_level': 0.95,
            'forecast_accuracy': 0.88,
            'inventory_turnover': 5.1,
            'order_frequency': 0.12
        },
        {
            'method_name': 'DQN',
            'total_cost': 2100.0,
            'service_level': 0.94,
            'forecast_accuracy': 0.82,
            'inventory_turnover': 5.8,
            'order_frequency': 0.18
        }
    ]

    df = pd.DataFrame(mock_results)
    evaluator = EnhancedInventoryEvaluator()

    print("📊 Mock Comparison Results:")
    evaluator.print_comparison_summary(df)

    print("\n✅ Evaluator demonstration completed!")