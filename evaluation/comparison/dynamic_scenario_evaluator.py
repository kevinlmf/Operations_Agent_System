"""
Dynamic Scenario Evaluator - 处理复杂动态场景

考虑因素：
1. 季节性 (Seasonality)
2. 趋势 (Trends)
3. 不确定性 (Uncertainty)
4. 动态规划多期决策
5. 时间序列预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import InventoryMethod, InventoryState
from evaluation.comparison.net_benefit_optimizer import NetBenefitOptimizer, NetBenefitResult


@dataclass
class ScenarioCharacteristics:
    """场景特征"""
    has_seasonality: bool = True
    has_trend: bool = True
    uncertainty_level: float = 0.2  # 需求不确定性系数 (CV)
    trend_strength: float = 0.05  # 趋势强度
    seasonality_amplitude: float = 0.3  # 季节性幅度
    volatility: float = 0.15  # 波动性


class DynamicScenarioEvaluator:
    """
    动态场景评估器
    
    在复杂动态场景下评估方法性能，考虑：
    - 季节性变化
    - 趋势变化
    - 需求不确定性
    - 多期动态决策
    """
    
    def __init__(self,
                 base_optimizer: NetBenefitOptimizer,
                 scenario: ScenarioCharacteristics = None):
        """
        初始化动态场景评估器
        
        Args:
            base_optimizer: 基础Net Benefit优化器
            scenario: 场景特征
        """
        self.base_optimizer = base_optimizer
        self.scenario = scenario or ScenarioCharacteristics()
    
    def generate_dynamic_demand(self,
                               base_demand: np.ndarray,
                               periods: int,
                               seed: int = 42) -> np.ndarray:
        """
        生成动态需求（考虑季节性、趋势、不确定性）
        
        Args:
            base_demand: 基础需求
            periods: 期数
            seed: 随机种子
            
        Returns:
            动态需求数组
        """
        np.random.seed(seed)
        t = np.arange(periods, dtype=float)
        
        # 基础需求（确保为正且合理）
        base_mean = max(np.mean(base_demand), 1.0)
        base_std = max(np.std(base_demand), 0.01)
        
        # 基础需求：使用原始数据或生成
        if len(base_demand) >= periods:
            demand = base_demand[:periods].astype(float)
        else:
            # 重复并截取
            demand = np.tile(base_demand, (periods // len(base_demand) + 1))[:periods].astype(float)
        
        # 确保基础需求为正
        demand = np.maximum(demand, 0.1 * base_mean)
        
        # 1. 添加趋势（乘法模型：更符合现实）
        if self.scenario.has_trend:
            # 趋势因子：1 + trend_strength * (t / periods)
            trend_factor = 1.0 + self.scenario.trend_strength * (t / max(periods, 1))
            demand = demand * trend_factor
        
        # 2. 添加季节性（乘法模型：季节性影响是比例性的）
        if self.scenario.has_seasonality:
            # 年度季节性因子：1 ± amplitude
            annual_factor = 1.0 + self.scenario.seasonality_amplitude * np.sin(2 * np.pi * t / 365.25)
            # 周季节性因子
            weekly_factor = 1.0 + 0.1 * np.sin(2 * np.pi * t / 7)
            demand = demand * annual_factor * weekly_factor
        
        # 3. 添加不确定性（使用对数正态分布确保为正）
        # CV方法：不确定性是比例性的
        cv = max(self.scenario.uncertainty_level, 0.01)  # 确保CV至少为0.01
        # 对数正态分布：log(X) ~ N(log(μ) - 0.5*σ², σ²)
        # 其中σ = CV
        log_std = cv
        uncertainty_factor = np.exp(np.random.normal(0, log_std, periods))
        demand = demand * uncertainty_factor
        
        # 4. 添加短期波动性（使用对数正态分布）
        volatility_cv = max(self.scenario.volatility, 0.01)
        volatility_factor = np.exp(np.random.normal(0, volatility_cv, periods))
        demand = demand * volatility_factor
        
        # 5. 需求平滑（避免极端波动，更符合现实）
        # 使用移动平均平滑，但保留一些波动性
        window = min(3, max(1, periods // 10))
        if window > 1 and periods > window:
            # 使用pandas的rolling更安全
            try:
                import pandas as pd
                demand_series = pd.Series(demand)
                smoothed = demand_series.rolling(window=window, center=True, min_periods=1).mean().values
                # 混合：70%平滑 + 30%原始
                demand = 0.7 * smoothed + 0.3 * demand
            except:
                # 回退到numpy
                kernel = np.ones(window) / window
                smoothed = np.convolve(demand, kernel, mode='same')
                demand = 0.7 * smoothed + 0.3 * demand
        
        # 6. 确保非负且合理范围
        # 最小值：基础均值的5%，最大值：基础均值的10倍
        min_demand = max(0.05 * base_mean, 0.5)
        max_demand = 10.0 * base_mean
        demand = np.clip(demand, min_demand, max_demand)
        
        return demand
    
    def evaluate_with_dynamic_programming(self,
                                        method: InventoryMethod,
                                        initial_state: InventoryState,
                                        demand_scenarios: List[np.ndarray],
                                        horizon: int = 30) -> Dict[str, Any]:
        """
        使用动态规划评估多期决策
        
        Args:
            method: 库存管理方法
            initial_state: 初始状态
            demand_scenarios: 需求场景列表（考虑不确定性）
            horizon: 规划期数
            
        Returns:
            评估结果
        """
        # DP状态：库存水平
        # DP决策：订货量
        # DP目标：最小化期望成本（或最大化期望Net Benefit）
        
        results = []
        
        for scenario_idx, demand_scenario in enumerate(demand_scenarios):
            state = initial_state
            total_cost = 0.0
            total_revenue = 0.0
            inventory_levels = [state.inventory_level]
            
            for t in range(min(horizon, len(demand_scenario))):
                # 获取方法推荐
                try:
                    action = method.recommend_action(state)
                    order_qty = action.order_quantity if hasattr(action, 'order_quantity') else 0
                except:
                    order_qty = 0
                
                # 更新库存
                available_inv = state.inventory_level + order_qty
                
                # 满足需求
                demand = demand_scenario[t]
                units_sold = min(available_inv, demand)
                stockout = max(0, demand - available_inv)
                
                # 计算成本和收入
                holding_cost = self.base_optimizer.holding_cost * max(0, available_inv - units_sold)
                stockout_cost = self.base_optimizer.stockout_cost * stockout
                ordering_cost = self.base_optimizer.ordering_cost * (1 if order_qty > 0 else 0)
                
                period_cost = holding_cost + stockout_cost + ordering_cost
                period_revenue = units_sold * self.base_optimizer.unit_price
                
                total_cost += period_cost
                total_revenue += period_revenue
                
                # 更新状态
                new_inventory = max(0, available_inv - units_sold)
                state = InventoryState(
                    inventory_level=new_inventory,
                    outstanding_orders=0,
                    demand_history=np.append(state.demand_history[-29:], demand),
                    time_step=state.time_step + 1
                )
                inventory_levels.append(new_inventory)
            
            results.append({
                'scenario': scenario_idx,
                'total_cost': total_cost,
                'total_revenue': total_revenue,
                'net_benefit': total_revenue - total_cost,
                'inventory_levels': inventory_levels
            })
        
        # 计算期望值
        expected_cost = np.mean([r['total_cost'] for r in results])
        expected_revenue = np.mean([r['total_revenue'] for r in results])
        expected_net_benefit = np.mean([r['net_benefit'] for r in results])
        
        # 计算风险指标（标准差）
        net_benefits = [r['net_benefit'] for r in results]
        risk = np.std(net_benefits)
        
        return {
            'expected_cost': expected_cost,
            'expected_revenue': expected_revenue,
            'expected_net_benefit': expected_net_benefit,
            'risk': risk,
            'scenarios': results
        }
    
    def evaluate_with_time_series_forecasting(self,
                                             method: InventoryMethod,
                                             historical_demand: np.ndarray,
                                             test_demands: np.ndarray,
                                             test_states: List[InventoryState]) -> Dict[str, Any]:
        """
        使用时间序列预测评估
        
        考虑：
        - ARIMA模型
        - 季节性分解
        - 趋势检测
        - 预测区间（不确定性量化）
        """
        # 尝试导入statsmodels，如果没有则使用简单方法
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.tsa.stattools import adfuller
            has_statsmodels = True
        except ImportError:
            has_statsmodels = False
        
        # 1. 时间序列分解
        try:
            # 转换为pandas Series
            ts = pd.Series(historical_demand)
            
            if has_statsmodels and len(ts) >= 24:  # 需要至少2个周期
                # 使用statsmodels进行季节性分解
                decomposition = seasonal_decompose(
                    ts,
                    model='additive',
                    period=min(7, len(ts) // 2)  # 周季节性
                )
                trend = decomposition.trend.dropna().values
                seasonal = decomposition.seasonal.dropna().values
                residual = decomposition.resid.dropna().values
                
                # 平稳性检验
                is_stationary = False
                try:
                    adf_result = adfuller(historical_demand)
                    is_stationary = adf_result[1] < 0.05  # p-value < 0.05表示平稳
                except:
                    is_stationary = True
            else:
                # 简单方法：移动平均作为趋势
                window = min(7, len(historical_demand) // 4)
                trend = pd.Series(historical_demand).rolling(window=window, center=True).mean().fillna(
                    np.mean(historical_demand)
                ).values
                
                # 季节性：使用周期性模式
                if len(historical_demand) >= 14:
                    period = 7  # 周周期
                    seasonal_pattern = []
                    for i in range(len(historical_demand)):
                        seasonal_pattern.append(
                            np.mean([historical_demand[j] for j in range(i % period, len(historical_demand), period)]) - np.mean(historical_demand)
                        )
                    seasonal = np.array(seasonal_pattern)
                else:
                    seasonal = np.zeros_like(historical_demand)
                
                residual = historical_demand - trend - seasonal
                is_stationary = True
            
            # 2. 趋势检测
            if len(trend) >= 30:
                trend_slope = np.mean(np.diff(trend[-30:]))
                trend_direction = 'up' if trend_slope > 0 else ('down' if trend_slope < 0 else 'stable')
                trend_strength = abs(trend_slope) / np.mean(trend) if np.mean(trend) > 0 else 0
            else:
                trend_direction = 'stable'
                trend_strength = 0
            
            # 3. 不确定性量化（确保std非负）
            residual_std = max(np.std(residual), 0.01)  # 确保至少有一个最小值
            hist_mean = max(np.mean(historical_demand), 0.1)  # 确保均值至少为0.1
            cv = residual_std / hist_mean
            
        except Exception as e:
            # 回退到简单统计
            trend = historical_demand
            seasonal = np.zeros_like(historical_demand)
            residual = historical_demand - np.mean(historical_demand)
            trend_direction = 'stable'
            trend_strength = 0
            is_stationary = True
            demand_std = max(np.std(historical_demand), 0.01)  # 确保std非负
            hist_mean = max(np.mean(historical_demand), 0.1)  # 确保均值至少为0.1
            cv = demand_std / hist_mean
        
        # 5. 评估方法在动态场景下的表现
        results = []
        
        for i, (state, true_demand) in enumerate(zip(test_states, test_demands)):
            # 获取预测
            try:
                forecast = method.predict_demand(state, horizon=1)
                predicted_demand = forecast[0] if len(forecast) > 0 else np.mean(historical_demand)
            except:
                predicted_demand = np.mean(historical_demand)
            
            # 计算预测误差
            forecast_error = abs(predicted_demand - true_demand)
            forecast_accuracy = 1 - (forecast_error / true_demand) if true_demand > 0 else 0
            
            # 考虑不确定性的决策
            # 使用预测区间（预测值 ± 不确定性）
            uncertainty_adjusted_demand = predicted_demand * (1 + cv)
            
            results.append({
                'period': i,
                'true_demand': true_demand,
                'predicted_demand': predicted_demand,
                'uncertainty_adjusted_demand': uncertainty_adjusted_demand,
                'forecast_error': forecast_error,
                'forecast_accuracy': forecast_accuracy
            })
        
        avg_forecast_accuracy = np.mean([r['forecast_accuracy'] for r in results])
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'is_stationary': is_stationary,
            'uncertainty_level': cv,
            'avg_forecast_accuracy': avg_forecast_accuracy,
            'forecast_results': results,
            'decomposition': {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual
            }
        }
    
    def evaluate_method_comprehensive(self,
                                     method: InventoryMethod,
                                     historical_demand: np.ndarray,
                                     test_states: List[InventoryState],
                                     test_demands: np.ndarray,
                                     num_scenarios: int = 10) -> Dict[str, Any]:
        """
        综合评估方法在动态场景下的表现
        
        Args:
            method: 库存管理方法
            historical_demand: 历史需求
            test_states: 测试状态
            test_demands: 测试需求
            num_scenarios: 蒙特卡洛场景数
            
        Returns:
            综合评估结果
        """
        # 1. 生成动态需求场景（考虑不确定性）
        demand_scenarios = []
        for i in range(num_scenarios):
            scenario_demand = self.generate_dynamic_demand(
                test_demands,
                len(test_demands),
                seed=42 + i
            )
            demand_scenarios.append(scenario_demand)
        
        # 2. 动态规划评估
        initial_state = test_states[0] if test_states else InventoryState(
            inventory_level=100,
            outstanding_orders=0,
            demand_history=historical_demand[-30:],
            time_step=0
        )
        
        dp_results = self.evaluate_with_dynamic_programming(
            method,
            initial_state,
            demand_scenarios,
            horizon=len(test_demands)
        )
        
        # 3. 时间序列预测评估
        ts_results = self.evaluate_with_time_series_forecasting(
            method,
            historical_demand,
            test_demands,
            test_states
        )
        
        # 4. 基础Net Benefit评估
        base_result = self.base_optimizer.evaluate_method_net_benefit(
            method,
            test_states,
            test_demands,
            len(test_demands),
            method.method_name if hasattr(method, 'method_name') else 'Unknown'
        )
        
        # 5. 综合结果
        return {
            'method_name': method.method_name if hasattr(method, 'method_name') else 'Unknown',
            'base_net_benefit': base_result.net_benefit,
            'expected_net_benefit': dp_results['expected_net_benefit'],
            'risk': dp_results['risk'],
            'risk_adjusted_net_benefit': dp_results['expected_net_benefit'] - 0.5 * dp_results['risk'],  # 风险调整
            'trend_direction': ts_results['trend_direction'],
            'trend_strength': ts_results['trend_strength'],
            'uncertainty_level': ts_results['uncertainty_level'],
            'forecast_accuracy': ts_results['avg_forecast_accuracy'],
            'is_stationary': ts_results['is_stationary'],
            'dp_results': dp_results,
            'ts_results': ts_results,
            'base_result': base_result
        }
    
    def compare_methods_dynamic(self,
                               methods: Dict[str, InventoryMethod],
                               historical_demand: np.ndarray,
                               test_states: List[InventoryState],
                               test_demands: np.ndarray,
                               num_scenarios: int = 10) -> pd.DataFrame:
        """
        对比所有方法在动态场景下的表现
        
        Returns:
            DataFrame包含所有方法的动态评估结果
        """
        results = []
        
        for method_name, method in methods.items():
            try:
                result = self.evaluate_method_comprehensive(
                    method,
                    historical_demand,
                    test_states,
                    test_demands,
                    num_scenarios
                )
                results.append(result)
            except Exception as e:
                print(f"❌ 评估 {method_name} 失败: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # 按风险调整后的Net Benefit排序
        if 'risk_adjusted_net_benefit' in df.columns:
            df = df.sort_values('risk_adjusted_net_benefit', ascending=False)
        
        return df

