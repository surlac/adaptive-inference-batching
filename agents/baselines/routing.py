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
        # 0=Wait
        # Action 1..32 = GPU0, Batch 1..32
        # Action 33..64 = GPU1, Batch 1..32
        
        # Target Batch Size: Always try to schedule max (32)
        # The simulator will handle if queue < 32
        batch_size = 32
        
        if self.next_gpu == 0:
            action = batch_size # 32 -> GPU0, Batch 32
        else:
            action = 32 + batch_size # 64 -> GPU1, Batch 32
            
        self.next_gpu = (self.next_gpu + 1) % 2
        return action

class RandomRoutingAgent:
    """Randomly routes to available GPUs."""
    def __init__(self, num_gpus=2):
        self.num_gpus = num_gpus
        
    def get_action(self, state, **kwargs):
        target_gpu = np.random.randint(0, self.num_gpus)
        batch_size = 32
        
        if target_gpu == 0:
            return batch_size
        else:
            return 32 + batch_size

class ShortestQueueRoutingAgent:
    """Routes to GPU with fewer pending requests (approximated by availability in this partial state)."""
    
    def __init__(self, num_gpus=2):
        self.num_gpus = num_gpus
        
    def get_action(self, state, **kwargs):
        # Check Busy Status (Lower is better/free-er)
        # state[8] is GPU0 busy time (normalized)
        # state[9] is GPU1 busy time (normalized)
        
        busy_0 = state[8]
        busy_1 = state[9]
        
        batch_size = 32
        
        if busy_0 <= busy_1:
            return batch_size # GPU 0
        else:
            return 32 + batch_size # GPU 1
