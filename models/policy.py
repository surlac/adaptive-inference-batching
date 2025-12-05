"""
Neural network policy model for RL agents.
"""
import torch
import torch.nn as nn
from typing import Tuple

class PolicyNetwork(nn.Module):
    """Neural network policy for the REINFORCE and PPO agents."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions (batch sizes)
            hidden_dim: Number of units in hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        # Attention-enhanced architecture
        self.proj = nn.Linear(state_dim, 4)  # Project input to compatible dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=4,  # Adjusted to be divisible by num_heads=2
            num_heads=2,
            dropout=0.1
        )
        self.policy_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            tuple: (action_probs, state_value)
        """
        x = self.proj(x)
        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)
        
        # Policy head
        logits = self.policy_head(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Value head (baseline)
        value = self.value_head(x)
        
        return probs, value.squeeze(-1)
