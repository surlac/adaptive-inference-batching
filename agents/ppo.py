"""
Implementation of Proximal Policy Optimization (PPO) algorithm for adaptive batching.
Uses Generalized Advantage Estimation (GAE) and clipped surrogate objective for stable training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple
from models.policy import PolicyNetwork

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for adaptive batching.
    Implements a reliable policy gradient method that avoids destructive large updates
    using a clipped surrogate objective.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,      # Standard capacity for low-dim state (9 features)
        lr: float = 3e-4,           # Standard "safe" learning rate (Karpathy constant)
        gamma: float = 0.99,        # High discount for long-horizon optimization
        gae_lambda: float = 0.95,   # Standard Generalized Advantage Estimation value
        clip_ratio: float = 0.2,    # Standard PPO Clipping (Schulman et al. 2017)
        entropy_coef: float = 0.01, # Small bonus to encourage exploration
        value_coef: float = 0.5,    # Scales value loss relative to policy loss
        target_kl: float = 0.01,    # Target KL divergence for early stopping
        update_epochs: int = 10,    # Number of updates per batch
        batch_size: int = 64,       # Minibatch size for updates
        max_grad_norm: float = 0.5, # Gradient clipping to prevent explosion
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        
        # Initialize policy network (shared backbone for actor/critic)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.masks = []
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
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
            
        return action, log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.masks.append(1.0 - float(done))
        
    def compute_gae(self, next_value):
        values = self.values + [next_value]
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            
        return returns, advantages
        
    def update(self):
        if not self.states:
            return {}
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        # Get next value for GAE
        with torch.no_grad():
            _, next_val = self.policy_net(states[-1].unsqueeze(0))
            next_value = next_val.item()
            
        returns, advantages = self.compute_gae(next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update Loop
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        stats = {
            'policy_loss': [], 'value_loss': [], 'entropy': [], 'kl': [], 'clip_frac': []
        }
        
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Forward pass
                probs, values = self.policy_net(mb_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Policy Loss
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = 0.5 * (mb_returns - values.squeeze()).pow(2).mean()
                
                # Total Loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Stats
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
                    clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.item())
                stats['kl'].append(approx_kl)
                stats['clip_frac'].append(clip_frac)
                
            # Early stopping based on KL
            if np.mean(stats['kl']) > 1.5 * self.target_kl:
                break
                
        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.masks = []
        
        return {k: np.mean(v) for k, v in stats.items()}

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
