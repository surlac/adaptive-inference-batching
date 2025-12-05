"""
Implementation of REINFORCE policy gradient algorithm for adaptive batching.
Uses a neural network policy with baseline for variance reduction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from typing import Dict, Tuple

from models.policy import PolicyNetwork

class REINFORCEAgent:
    """
    Standard REINFORCE (Monte Carlo Policy Gradient) implementation.
    Includes baseline subtraction for variance reduction, but currently relies on
    external state value estimation or simple rewards.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        gamma: float = 0.95,        # Discount factor (shorter horizon for stability)
        lr: float = 3e-4,           # Karpathy constant
        entropy_coef: float = 0.01, # Encourage exploration
        max_grad_norm: float = 0.5, # Prevent gradient explosion
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the REINFORCE agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions (batch sizes)
            hidden_dim: Number of units in hidden layers
            gamma: Discount factor
            lr: Learning rate
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Trajectory buffer
        self.reset_buffer()
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.masks = []
        
        # For tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Select an action using the current policy.
        
        Args:
            state: Current state
            deterministic: If True, select the action with highest probability
            
        Returns:
            tuple: (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.policy_net(state_tensor)
            
        if deterministic:
            action = torch.argmax(probs).item()
            log_prob = torch.log(probs[0, action] + 1e-10)
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
        
        return action, log_prob, value.squeeze()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ) -> None:
        """
        Store a transition in the trajectory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of the action
            value: State value estimate
            done: Whether the episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(1.0 - float(done))
    
    def compute_returns(self, next_value: torch.Tensor) -> torch.Tensor:
        """
        Compute the discounted returns for each step in the trajectory.
        
        Args:
            next_value: Value estimate for the next state
            
        Returns:
            torch.Tensor: Tensor of discounted returns
        """
        returns = []
        R = next_value
        
        # Compute returns in reverse order
        for r, mask in zip(reversed(self.rewards), reversed(self.masks)):
            R = r + self.gamma * R * mask
            returns.insert(0, R)
        
        return torch.tensor(returns, device=self.device)
    
    def update(self) -> Dict[str, float]:
        """
        Update the policy using the current trajectory.
        
        Returns:
            dict: Dictionary of training statistics
        """
        if not self.rewards:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach()
        old_values = torch.stack(self.values).detach()
        
        # Compute advantages and returns
        with torch.no_grad():
            next_value = self.policy_net(states[-1].unsqueeze(0))[1].squeeze()
        
        returns = self.compute_returns(next_value)
        advantages = returns - old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current policy and value predictions
        probs, values = self.policy_net(states)
        dist = Categorical(probs)
        
        # Compute policy loss
        new_log_probs = dist.log_prob(actions)
        ratio = (new_log_probs - old_log_probs).exp()
        policy_loss1 = ratio * advantages
        policy_loss2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        # Compute value loss
        value_loss = 0.5 * (returns - values).pow(2).mean()
        
        # Compute entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.masks = []
        
        # Return statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
            'avg_return': returns.mean().item(),
            'avg_value': old_values.mean().item(),
        }
        
        return stats
    
    def save(self, path: str) -> None:
        """Save the model to a file."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_net.to(self.device)


