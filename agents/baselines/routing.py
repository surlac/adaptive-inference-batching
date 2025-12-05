"""
Routing baseline agents for multi-GPU scenarios.
"""
import numpy as np

class RoundRobinRoutingAgent:
    """
    Round-robin routing agent for multi-GPU scenarios.
    """
    def __init__(self, num_gpus=2):
        self.next_gpu = 0
        self.name = "round_robin_routing"
        
    def get_action(self, state, **kwargs):
        # 0=Wait, 1=GPU0, 2=GPU1
        # If queue < 8, Wait (Action 0)
        # Else, send to next GPU
        
        # State[0] is normalized queue length (len/100)
        queue_len = state[0] * 100
        if queue_len < 8:
            return 0
            
        action = self.next_gpu + 1 # 0->1, 1->2
        self.next_gpu = (self.next_gpu + 1) % 2
        return action

class RandomRoutingAgent:
    """Randomly routes to available GPUs."""
    def __init__(self, num_gpus=2):
        self.num_gpus = num_gpus
        
    def get_action(self, state, **kwargs):
        # 0=Wait
        queue_len = state[0] * 100
        if queue_len < 8:
            return 0
        return np.random.randint(1, self.num_gpus + 1)

class ShortestQueueRoutingAgent:
    """Routes to GPU with fewer pending requests (approximated by availability in this partial state)."""
    # Note: The provided state doesn't explicit show per-GPU queue depth, only Busy status.
    # But in the RoutingScenario, we know we are pushing to GPU clusters.
    # As a proxy for this simplified environment, we route to the first non-busy GPU, or random if both busy.
    
    def __init__(self, num_gpus=2):
        self.num_gpus = num_gpus
        
    def get_action(self, state, **kwargs):
        # State: [Q, Time, Type, Busy0, Busy1, ...]
        queue_len = state[0] * 100
        if queue_len < 8:
            return 0
            
        # Check Busy Status
        busy_0 = state[3]
        busy_1 = state[4]
        
        if not busy_0 and not busy_1:
            return 1 # GPU 0
        elif not busy_0:
            return 1
        elif not busy_1:
            return 2
        else:
            return np.random.randint(1, 3)
