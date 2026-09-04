"""Dueling DQN component used by the OSAF-SORS model."""

from __future__ import annotations

from torch import Tensor, nn


class D3QN(nn.Module):
    """Dueling DQN with value/advantage aggregation."""

    def __init__(self, state_size: int, action_size: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if state_size < 1 or action_size < 1:
            raise ValueError("state_size and action_size must be positive")
        self.state_size = state_size
        self.action_size = action_size
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size),
        )
        self.apply(self._initialise_linear_layers)

    @staticmethod
    def _initialise_linear_layers(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: Tensor) -> Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
