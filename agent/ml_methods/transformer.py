"""
Transformer-based Demand Forecasting for Inventory Management

Implements Transformer architecture using JAX/Flax for time series forecasting
with multi-head attention and positional encoding.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any, Optional, Tuple, Sequence
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.interfaces import MLMethod, InventoryState, InventoryAction


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""

    d_model: int
    max_len: int = 5000

    def setup(self):
        # Create positional encoding matrix
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, np.newaxis]

        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32) *
            -(math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = jnp.array(pe)

    def __call__(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    d_model: int
    num_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads

        self.query_proj = nn.Dense(self.d_model)
        self.key_proj = nn.Dense(self.d_model)
        self.value_proj = nn.Dense(self.d_model)
        self.output_proj = nn.Dense(self.d_model)

        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, mask=None, training=False):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            training: Whether in training mode
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        q = self.query_proj(x)  # [batch_size, seq_len, d_model]
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(q, k, v, mask)

        # Transpose back and reshape
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_len, d_model)

        # Output projection
        output = self.output_proj(attention_output)
        output = self.dropout(output, deterministic=not training)

        return output

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """Compute scaled dot-product attention"""
        # Compute attention scores
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        scores = scores / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        # Apply softmax
        attention_weights = nn.softmax(scores, axis=-1)

        # Apply attention to values
        output = jnp.matmul(attention_weights, v)

        return output


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""

    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    def setup(self):
        self.self_attention = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )

        self.dense1 = nn.Dense(self.d_ff)
        self.dense2 = nn.Dense(self.d_model)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()

    def __call__(self, x, mask=None, training=False):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            training: Whether in training mode
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, mask, training)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.dense1(x)
        ff_output = nn.relu(ff_output)
        ff_output = self.dropout1(ff_output, deterministic=not training)
        ff_output = self.dense2(ff_output)
        ff_output = self.dropout2(ff_output, deterministic=not training)
        x = self.layer_norm2(x + ff_output)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for time series forecasting"""

    num_layers: int = 6
    d_model: int = 128
    num_heads: int = 8
    d_ff: int = 512
    dropout_rate: float = 0.1
    max_seq_len: int = 1000

    def setup(self):
        self.input_projection = nn.Dense(self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_len)

        self.transformer_blocks = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]

        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, mask=None, training=False):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            training: Whether in training mode
        """
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x, deterministic=not training)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask, training)

        return x


class TimeSeriesTransformer(nn.Module):
    """Complete transformer model for time series forecasting"""

    seq_len: int = 50
    num_layers: int = 4
    d_model: int = 128
    num_heads: int = 8
    d_ff: int = 512
    dropout_rate: float = 0.1
    output_size: int = 1

    def setup(self):
        self.encoder = TransformerEncoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            max_seq_len=self.seq_len
        )

        # Output layers
        self.global_pool = lambda x: jnp.mean(x, axis=1)  # Global average pooling
        self.dense1 = nn.Dense(self.d_model // 2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.dense2 = nn.Dense(self.output_size)

    def __call__(self, x, training=False):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            training: Whether in training mode
        """
        # Encode sequence
        encoded = self.encoder(x, training=training)

        # Global pooling to get fixed-size representation
        pooled = self.global_pool(encoded)

        # Generate output with proper dropout handling
        output = self.dense1(pooled)
        output = nn.relu(output)
        output = self.dropout(output, deterministic=not training)
        output = self.dense2(output)

        return output


class TransformerInventoryMethod(MLMethod):
    """Transformer-based inventory method with demand forecasting"""

    def __init__(self,
                 sequence_length: int = 50,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 512,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.0001,
                 epochs: int = 150,
                 batch_size: int = 32,
                 forecast_horizon: int = 7):
        """
        Initialize Transformer inventory method

        Args:
            sequence_length: Length of input sequences
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            forecast_horizon: Number of periods to forecast ahead
        """
        super().__init__("Transformer")
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.forecast_horizon = forecast_horizon

        # Model components
        self.model = None
        self.params = None
        self.state = None
        self.scaler_mean = None
        self.scaler_std = None

        # Safety stock parameters
        self.safety_stock = 0.0
        self.reorder_point = 0.0

    def build_model(self) -> nn.Module:
        """Build transformer model architecture"""
        return TimeSeriesTransformer(
            seq_len=self.sequence_length,
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            output_size=1
        )

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences for training"""
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])

        return np.array(X), np.array(y)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using z-score normalization"""
        if self.scaler_mean is None:
            self.scaler_mean = np.mean(data)
            self.scaler_std = np.std(data)

        return (data - self.scaler_mean) / (self.scaler_std + 1e-8)

    def _denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale"""
        return data * self.scaler_std + self.scaler_mean

    def fit(self,
            demand_history: np.ndarray,
            external_features: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """
        Train Transformer model on demand history

        Args:
            demand_history: Historical demand time series
            external_features: Optional external features (not used yet)
        """
        if len(demand_history) < self.sequence_length + 10:
            raise ValueError(f"Need at least {self.sequence_length + 10} data points for training")

        # Normalize data
        normalized_data = self._normalize_data(demand_history.astype(np.float32))

        # Create sequences
        X, y = self._create_sequences(normalized_data)

        # Add feature dimension if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        # Build model
        self.model = self.build_model()

        # Initialize parameters
        rng = random.PRNGKey(42)
        sample_input = X[:1]  # Use first sample for initialization

        params = self.model.init(rng, sample_input, training=False)

        # Create optimizer with warmup
        warmup_steps = min(1000, self.epochs * len(X) // self.batch_size // 4)  # Adjust warmup based on total steps
        total_steps = self.epochs * len(X) // self.batch_size
        decay_steps = max(total_steps - warmup_steps, 1)  # Ensure positive decay_steps
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=self.learning_rate * 0.1
        )
        optimizer = optax.adamw(scheduler, weight_decay=1e-4)

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )

        # Train model
        self._train_model(X, y, rng)

        # Calculate safety stock from prediction errors
        predictions = []
        for i in range(0, len(X), 10):  # Sample every 10th for efficiency
            pred = self._predict_single(X[i:i+1])
            predictions.append(pred[0])

        predictions = np.array(predictions)
        actuals = y[::10]

        # Denormalize predictions and actuals
        predictions_denorm = self._denormalize_data(predictions)
        actuals_denorm = self._denormalize_data(actuals)

        # Calculate residuals
        residuals = actuals_denorm - predictions_denorm
        residual_std = np.std(residuals)

        # Set safety stock (2 standard deviations)
        self.safety_stock = 2.0 * residual_std
        self.reorder_point = np.mean(demand_history) * 7 + self.safety_stock

        self._is_fitted = True

    def _train_model(self, X: np.ndarray, y: np.ndarray, rng: jax.random.PRNGKey):
        """Train the Transformer model"""

        @jit
        def train_step(state, batch_x, batch_y, dropout_rng):
            """Single training step"""
            def loss_fn(params):
                predictions = state.apply_fn(
                    params, batch_x, training=True, rngs={'dropout': dropout_rng}
                )

                # Reshape for loss calculation
                predictions = predictions.reshape(-1)
                targets = batch_y.reshape(-1)

                # Mean squared error with L2 regularization
                mse_loss = jnp.mean((predictions - targets) ** 2)

                return mse_loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        # Training loop
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        print(f"Training Transformer: {n_samples} samples, {n_batches} batches per epoch")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            rng, dropout_rng = random.split(rng)

            # Shuffle data
            indices = random.permutation(rng, n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                batch_x = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                dropout_rng, step_rng = random.split(dropout_rng)
                self.state, batch_loss = train_step(self.state, batch_x, batch_y, step_rng)
                epoch_loss += batch_loss

            # Print progress
            if epoch % max(1, self.epochs // 10) == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}")

    def _predict_single(self, x: np.ndarray) -> np.ndarray:
        """Make prediction for single input"""
        pred = self.state.apply_fn(self.state.params, x, training=False)
        return np.array(pred)

    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        Predict future demand using Transformer model

        Args:
            current_state: Current inventory state
            horizon: Number of periods to predict ahead

        Returns:
            Predicted demand for next horizon periods
        """
        if not self.is_fitted:
            raise ValueError("Transformer model must be fitted before prediction")

        # Use recent demand history
        recent_demand = current_state.demand_history[-self.sequence_length:]

        if len(recent_demand) < self.sequence_length:
            # Pad with mean if insufficient history
            mean_demand = np.mean(recent_demand) if len(recent_demand) > 0 else 50.0
            padded = np.full(self.sequence_length, mean_demand)
            padded[-len(recent_demand):] = recent_demand
            recent_demand = padded

        # Normalize input
        normalized_input = self._normalize_data(recent_demand.astype(np.float32))

        # Reshape for model input
        model_input = normalized_input.reshape(1, self.sequence_length, 1)

        predictions = []
        current_input = model_input

        # Multi-step prediction
        for _ in range(horizon):
            pred = self._predict_single(current_input)
            predictions.append(pred[0, 0])

            # Update input for next prediction (slide window)
            current_input = jnp.concatenate([
                current_input[:, 1:, :],
                pred.reshape(1, 1, 1)
            ], axis=1)

        # Denormalize predictions
        predictions = np.array(predictions)
        predictions_denorm = self._denormalize_data(predictions)

        return np.maximum(predictions_denorm, 0.0)  # Ensure non-negative

    def recommend_action(self, current_state: InventoryState) -> InventoryAction:
        """
        Recommend inventory action based on Transformer demand forecast

        Args:
            current_state: Current inventory state

        Returns:
            Recommended inventory action
        """
        if not self.is_fitted:
            raise ValueError("Transformer model must be fitted before recommendation")

        # Predict demand for forecast horizon
        demand_forecast = self.predict_demand(current_state, self.forecast_horizon)
        expected_demand = np.sum(demand_forecast)

        # Calculate inventory position
        inventory_position = current_state.inventory_level + current_state.outstanding_orders

        # Reorder point policy with ML forecast
        if inventory_position <= self.reorder_point:
            order_quantity = max(0, expected_demand + self.safety_stock - inventory_position)
        else:
            order_quantity = 0.0

        return InventoryAction(order_quantity=order_quantity)

    def get_parameters(self) -> Dict[str, Any]:
        """Get Transformer method parameters"""
        return {
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'forecast_horizon': self.forecast_horizon,
            'safety_stock': float(self.safety_stock),
            'reorder_point': float(self.reorder_point),
            'scaler_mean': float(self.scaler_mean) if self.scaler_mean is not None else None,
            'scaler_std': float(self.scaler_std) if self.scaler_std is not None else None
        }

    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model implementation for MLMethod interface"""
        # This is handled by the fit method
        pass


if __name__ == "__main__":
    # Example usage and testing
    print("🚀 Transformer Demand Forecasting - Example Usage")
    print("=" * 50)

    # Generate more complex sample demand data
    np.random.seed(42)
    n_days = 500  # Need more data for transformer
    t = np.arange(n_days)

    # Create realistic demand pattern with multiple seasonalities
    annual = 15 * np.sin(2 * np.pi * t / 365.25)    # Annual seasonality
    weekly = 8 * np.sin(2 * np.pi * t / 7)          # Weekly pattern
    monthly = 5 * np.sin(2 * np.pi * t / 30.44)     # Monthly pattern
    trend = 0.01 * t                                 # Slight upward trend
    noise = np.random.normal(0, 3, n_days)          # Random noise

    # Add some promotional spikes
    promotions = np.random.binomial(1, 0.05, n_days) * 20

    demand_history = 60 + annual + weekly + monthly + trend + noise + promotions
    demand_history = np.maximum(demand_history, 0)  # Ensure non-negative

    print(f"Sample demand statistics:")
    print(f"  Mean: {np.mean(demand_history):.2f}")
    print(f"  Std: {np.std(demand_history):.2f}")
    print(f"  Min/Max: {np.min(demand_history):.1f}/{np.max(demand_history):.1f}")

    # Split data
    train_size = int(0.8 * len(demand_history))
    train_data = demand_history[:train_size]
    test_data = demand_history[train_size:]

    print(f"\n🔧 Training Transformer...")
    model = TransformerInventoryMethod(
        sequence_length=50,
        d_model=64,      # Smaller model for faster training
        num_heads=4,
        num_layers=3,
        epochs=30,       # Fewer epochs for demo
        batch_size=16
    )

    try:
        # Fit model
        model.fit(train_data)

        # Test prediction
        current_state = InventoryState(
            inventory_level=50.0,
            outstanding_orders=15.0,
            demand_history=train_data[-100:],
            time_step=train_size
        )

        # Get forecast
        forecast = model.predict_demand(current_state, horizon=7)
        action = model.recommend_action(current_state)

        print(f"  ✅ Training completed")
        print(f"  📊 7-day forecast: {forecast}")
        print(f"  📦 Recommended order: {action.order_quantity:.1f}")

        # Calculate forecast accuracy on test data
        if len(test_data) >= 7:
            actual_week = test_data[:7]
            forecast_error = np.mean(np.abs(forecast - actual_week))
            mape = np.mean(np.abs((forecast - actual_week) / actual_week)) * 100
            print(f"  🎯 MAE on test week: {forecast_error:.2f}")
            print(f"  🎯 MAPE on test week: {mape:.2f}%")

        # Show model parameters
        params = model.get_parameters()
        print(f"  🔧 Model: {params['d_model']}D, {params['num_layers']} layers, {params['num_heads']} heads")
        print(f"  📈 Safety stock: {params['safety_stock']:.1f}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    print(f"\n✅ Transformer testing completed!")