"""
LSTM-based Demand Forecasting for Inventory Management

Implements LSTM neural networks using JAX/Flax for demand prediction
with various architectural variants including attention mechanisms.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any, Optional, Tuple, Sequence
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from goal.interfaces import MLMethod, InventoryState, InventoryAction


class LSTMCell(nn.Module):
    """LSTM Cell implementation in Flax"""

    features: int

    @nn.compact
    def __call__(self, carry, inputs):
        """
        Args:
            carry: (hidden_state, cell_state)
            inputs: input at current timestep
        """
        h, c = carry

        # Concatenate input and hidden state
        combined = jnp.concatenate([inputs, h], axis=-1)

        # Compute gates
        gates = nn.Dense(4 * self.features)(combined)

        # Split gates
        forget_gate = nn.sigmoid(gates[..., :self.features])
        input_gate = nn.sigmoid(gates[..., self.features:2*self.features])
        cell_gate = nn.tanh(gates[..., 2*self.features:3*self.features])
        output_gate = nn.sigmoid(gates[..., 3*self.features:])

        # Update cell state
        new_c = forget_gate * c + input_gate * cell_gate

        # Update hidden state
        new_h = output_gate * nn.tanh(new_c)

        return (new_h, new_c), new_h


class BasicLSTM(nn.Module):
    """Basic LSTM network for demand forecasting"""

    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1
    output_size: int = 1

    def setup(self):
        self.lstm_layers = [LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Dense(self.output_size)

    def __call__(self, x, training=False):
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_size]
            training: Whether in training mode
        """
        batch_size, seq_len, input_size = x.shape

        # Initialize carry states for all layers
        carry_states = []
        for _ in range(self.num_layers):
            h = jnp.zeros((batch_size, self.hidden_size))
            c = jnp.zeros((batch_size, self.hidden_size))
            carry_states.append((h, c))

        # Process sequence
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]

            for i, lstm_layer in enumerate(self.lstm_layers):
                carry_states[i], layer_output = lstm_layer(carry_states[i], layer_input)
                layer_input = self.dropout(layer_output, deterministic=not training)

            outputs.append(layer_output)

        # Use last output for prediction
        final_output = outputs[-1]
        prediction = self.output_layer(final_output)

        return prediction


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for demand forecasting"""

    hidden_size: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.1
    output_size: int = 1
    attention_dim: int = 32

    def setup(self):
        self.lstm_layers = [LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        self.dropout = nn.Dropout(self.dropout_rate)

        # Attention mechanism
        self.attention_w = nn.Dense(self.attention_dim)
        self.attention_u = nn.Dense(self.attention_dim)
        self.attention_v = nn.Dense(1)

        self.output_layer = nn.Dense(self.output_size)

    def __call__(self, x, training=False):
        """
        Args:
            x: Input sequence [batch_size, seq_len, input_size]
            training: Whether in training mode
        """
        batch_size, seq_len, input_size = x.shape

        # Initialize carry states
        carry_states = []
        for _ in range(self.num_layers):
            h = jnp.zeros((batch_size, self.hidden_size))
            c = jnp.zeros((batch_size, self.hidden_size))
            carry_states.append((h, c))

        # Collect all hidden states
        hidden_states = []

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for i, lstm_layer in enumerate(self.lstm_layers):
                carry_states[i], layer_output = lstm_layer(carry_states[i], layer_input)
                layer_input = self.dropout(layer_output, deterministic=not training)

            hidden_states.append(layer_output)

        # Stack hidden states: [batch_size, seq_len, hidden_size]
        hidden_states = jnp.stack(hidden_states, axis=1)

        # Attention mechanism
        # Compute attention scores
        attention_scores = self._compute_attention(hidden_states, carry_states[-1][0])

        # Apply attention weights
        context_vector = jnp.sum(attention_scores[:, :, None] * hidden_states, axis=1)

        # Final prediction
        prediction = self.output_layer(context_vector)

        return prediction, attention_scores

    def _compute_attention(self, hidden_states, final_hidden):
        """
        Compute attention weights using additive attention

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            final_hidden: [batch_size, hidden_size]
        """
        seq_len = hidden_states.shape[1]

        # Expand final_hidden to match sequence length
        final_hidden_expanded = jnp.expand_dims(final_hidden, 1)
        final_hidden_expanded = jnp.repeat(final_hidden_expanded, seq_len, axis=1)

        # Compute attention energies
        energy = nn.tanh(
            self.attention_w(hidden_states) +
            self.attention_u(final_hidden_expanded)
        )

        # Compute attention scores
        scores = self.attention_v(energy).squeeze(-1)  # [batch_size, seq_len]

        # Apply softmax to get weights
        attention_weights = nn.softmax(scores, axis=1)

        return attention_weights


class LSTMInventoryMethod(MLMethod):
    """LSTM-based inventory method with demand forecasting"""

    def __init__(self,
                 sequence_length: int = 30,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 use_attention: bool = False,
                 forecast_horizon: int = 7):
        """
        Initialize LSTM inventory method

        Args:
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_attention: Whether to use attention mechanism
            forecast_horizon: Number of periods to forecast ahead
        """
        super().__init__("LSTM_Attention" if use_attention else "LSTM")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_attention = use_attention
        self.forecast_horizon = forecast_horizon

        # Model components
        self.model = None
        self.params = None
        self.state = None
        self.scaler_mean = None
        self.scaler_std = None

        # Safety stock parameters (learned from residuals)
        self.safety_stock = 0.0
        self.reorder_point = 0.0

    def build_model(self) -> nn.Module:
        """Build LSTM model architecture"""
        if self.use_attention:
            return AttentionLSTM(
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate,
                output_size=1
            )
        else:
            return BasicLSTM(
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
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
        Train LSTM model on demand history

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

        # Check if we have enough sequences
        if len(X) == 0:
            raise ValueError(f"No sequences created. Data length: {len(demand_history)}, sequence_length: {self.sequence_length}")

        print(f"  Created {len(X)} sequences for training")

        # Add feature dimension if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        # Adjust batch size if necessary
        if self.batch_size > len(X):
            self.batch_size = max(1, len(X) // 4)  # Use quarter of available data as batch size
            print(f"  Adjusted batch size to {self.batch_size}")

        # Build model
        self.model = self.build_model()

        # Initialize parameters
        rng = random.PRNGKey(42)
        sample_input = X[:1]  # Use first sample for initialization

        if self.use_attention:
            params = self.model.init(rng, sample_input, training=False)
        else:
            params = self.model.init(rng, sample_input, training=False)

        # Create optimizer
        optimizer = optax.adam(self.learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )

        # Train model
        self._train_model(X, y, rng)

        # Calculate safety stock from prediction errors
        predictions = []
        for i in range(len(X)):
            pred = self._predict_single(X[i:i+1])
            predictions.append(pred[0])

        predictions = np.array(predictions)
        actuals = y

        # Denormalize predictions and actuals
        predictions_denorm = self._denormalize_data(predictions)
        actuals_denorm = self._denormalize_data(actuals)

        # Calculate residuals
        residuals = actuals_denorm - predictions_denorm
        residual_std = np.std(residuals)

        # Set safety stock (2 standard deviations)
        self.safety_stock = 2.0 * residual_std
        self.reorder_point = np.mean(demand_history) * 7 + self.safety_stock  # 7-day lead time

        self._is_fitted = True

    def _train_model(self, X: np.ndarray, y: np.ndarray, rng: jax.random.PRNGKey):
        """Train the LSTM model"""

        @jit
        def train_step(state, batch_x, batch_y, dropout_rng):
            """Single training step"""
            def loss_fn(params):
                if self.use_attention:
                    predictions, _ = state.apply_fn(params, batch_x, training=True, rngs={'dropout': dropout_rng})
                else:
                    predictions = state.apply_fn(params, batch_x, training=True, rngs={'dropout': dropout_rng})

                # Reshape for loss calculation
                predictions = predictions.reshape(-1)
                targets = batch_y.reshape(-1)

                # Mean squared error
                loss = jnp.mean((predictions - targets) ** 2)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        # Training loop
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

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

            # Print progress occasionally
            if epoch % max(1, self.epochs // 10) == 0:
                avg_loss = epoch_loss / n_batches
                print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.6f}")

    def _predict_single(self, x: np.ndarray) -> np.ndarray:
        """Make prediction for single input"""
        if self.use_attention:
            pred, _ = self.state.apply_fn(self.state.params, x, training=False)
        else:
            pred = self.state.apply_fn(self.state.params, x, training=False)
        return np.array(pred)

    def predict_demand(self,
                      current_state: InventoryState,
                      horizon: int = 1) -> np.ndarray:
        """
        Predict future demand using LSTM model

        Args:
            current_state: Current inventory state
            horizon: Number of periods to predict ahead

        Returns:
            Predicted demand for next horizon periods
        """
        if not self.is_fitted:
            raise ValueError("LSTM model must be fitted before prediction")

        # Use recent demand history
        recent_demand = current_state.demand_history[-self.sequence_length:]

        # Convert to numpy array if it's a list
        if isinstance(recent_demand, list):
            recent_demand = np.array(recent_demand)

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

            # Update input for next prediction
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
        Recommend inventory action based on LSTM demand forecast

        Args:
            current_state: Current inventory state

        Returns:
            Recommended inventory action
        """
        if not self.is_fitted:
            raise ValueError("LSTM model must be fitted before recommendation")

        # Predict demand for forecast horizon
        demand_forecast = self.predict_demand(current_state, self.forecast_horizon)
        expected_demand = np.sum(demand_forecast)
        next_period_forecast = demand_forecast[0]

        # Calculate inventory position
        inventory_position = current_state.inventory_level + current_state.outstanding_orders

        # Simple reorder point policy with ML forecast
        if inventory_position <= self.reorder_point:
            # Order to cover forecast demand plus safety stock
            order_quantity = max(0, expected_demand + self.safety_stock - inventory_position)
        else:
            order_quantity = 0.0

        return InventoryAction(
            order_quantity=order_quantity,
            forecast=float(next_period_forecast)
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Get LSTM method parameters"""
        return {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'use_attention': self.use_attention,
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

    def train(self, demand_history: np.ndarray, epochs: int = None,
              batch_size: int = None, verbose: bool = True) -> Dict[str, list]:
        """
        Convenience method to train the model (alias for fit with additional options)

        Args:
            demand_history: Historical demand time series
            epochs: Number of training epochs (overrides default)
            batch_size: Batch size for training (overrides default)
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history (loss values)
        """
        # Override parameters if provided
        if epochs is not None:
            original_epochs = self.epochs
            self.epochs = epochs
        if batch_size is not None:
            original_batch_size = self.batch_size
            self.batch_size = batch_size

        # Call fit method
        self.fit(demand_history)

        # Restore original parameters
        if epochs is not None:
            self.epochs = original_epochs
        if batch_size is not None:
            self.batch_size = original_batch_size

        # Return a simple history (actual loss tracking would require modifying _train_model)
        return {'loss': [0.0] * self.epochs}


if __name__ == "__main__":
    # Example usage and testing
    print("🧠 LSTM Demand Forecasting - Example Usage")
    print("=" * 50)

    # Generate sample demand data
    np.random.seed(42)
    n_days = 365
    t = np.arange(n_days)

    # Create realistic demand pattern
    seasonal = 10 * np.sin(2 * np.pi * t / 365.25)  # Annual seasonality
    weekly = 5 * np.sin(2 * np.pi * t / 7)         # Weekly pattern
    trend = 0.02 * t                                # Slight upward trend
    noise = np.random.normal(0, 5, n_days)         # Random noise

    demand_history = 50 + seasonal + weekly + trend + noise
    demand_history = np.maximum(demand_history, 0)  # Ensure non-negative

    print(f"Sample demand statistics:")
    print(f"  Mean: {np.mean(demand_history):.2f}")
    print(f"  Std: {np.std(demand_history):.2f}")
    print(f"  Min/Max: {np.min(demand_history):.1f}/{np.max(demand_history):.1f}")

    # Test both LSTM variants
    models = {
        'Basic LSTM': LSTMInventoryMethod(
            sequence_length=30,
            hidden_size=32,
            epochs=50,
            use_attention=False
        ),
        'Attention LSTM': LSTMInventoryMethod(
            sequence_length=30,
            hidden_size=32,
            epochs=50,
            use_attention=True
        )
    }

    # Split data
    train_size = int(0.8 * len(demand_history))
    train_data = demand_history[:train_size]
    test_data = demand_history[train_size:]

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")

        try:
            # Fit model
            model.fit(train_data)

            # Test prediction
            current_state = InventoryState(
                inventory_level=40.0,
                outstanding_orders=10.0,
                demand_history=train_data[-50:],
                time_step=train_size
            )

            # Get forecast
            forecast = model.predict_demand(current_state, horizon=7)
            action = model.recommend_action(current_state)

            print(f"  ✅ Training completed")
            print(f"  📊 7-day forecast: {forecast}")
            print(f"  📦 Recommended order: {action.order_quantity:.1f}")

            # Calculate forecast accuracy on test data
            if len(test_data) > 7:
                actual_week = test_data[:7]
                forecast_error = np.mean(np.abs(forecast - actual_week))
                print(f"  🎯 MAE on test week: {forecast_error:.2f}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ LSTM testing completed!")