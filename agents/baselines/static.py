
"""
Static batching baseline agent implementation.
"""
import numpy as np

class StaticBatchingAgent:
    """
    Baseline agent that always uses a fixed batch size.
    """
    
    def __init__(self, batch_size: int = 8):
        """
        Initialize the static batching agent.
        
        Args:
            batch_size: Fixed batch size to use
        """
        self.batch_size = batch_size
        self.name = f"static_batch_{batch_size}"
    
    def get_action(self, _state: np.ndarray, **kwargs) -> int:
        """
        Always return the fixed batch size.
        
        Args:
            state: Current state (ignored)
            
        Returns:
            int: Fixed batch size
        """
        return self.batch_size
