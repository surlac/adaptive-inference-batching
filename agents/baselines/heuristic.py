
"""
Heuristic-based baseline agents (Timeout, Adaptive, Greedy Latency).
"""
import numpy as np

class TimeoutBatchingAgent:
    """
    Baseline agent that batches requests until a timeout is reached or max batch size.
    """
    
    def __init__(self, max_batch_size: int = 12, timeout: float = 5.0, max_queue_length: int = 100):
        """
        Initialize the timeout batching agent.
        
        Args:
            max_batch_size: Maximum batch size
            timeout: Maximum time to wait before batching (ms)
            max_queue_length: Maximum queue length (for denormalizing state)
        """
        self.max_batch_size = max_batch_size
        self.timeout = timeout / 1000.0  # Convert to seconds
        self.last_batch_time = 0.0
        self.min_queue_to_batch = 1  # Batch when at least 1 request
        self.max_queue_length = max_queue_length
        self.name = f"timeout_{int(timeout)}ms"
    
    def get_action(
        self, 
        state: np.ndarray, 
        _current_time: float = 0.0, 
        queue_length: int = 0,
        **kwargs
    ) -> int:
        """
        Decide whether to batch based on timeout and queue length.
        """
        # Extract and denormalize queue_length from state if not provided
        if queue_length == 0 and len(state) > 0:
            queue_length = int(state[0] * self.max_queue_length)  # Denormalize
        
        if queue_length == 0:
            return 0
        
        # Batch if: queue is getting full OR we have enough requests
        if queue_length >= self.max_batch_size:
            # Queue is full, batch immediately
            return min(queue_length, self.max_batch_size)
        elif queue_length >= self.min_queue_to_batch:
            # We have some requests, batch them to avoid timeout
            return min(queue_length, self.max_batch_size)
        
        return 0

class AdaptiveThresholdAgent:
    """
    Adaptive batching agent that adjusts batch size based on queue length.
    """
    
    def __init__(
        self, 
        max_batch_size: int = 16, 
        min_batch_size: int = 4,
        queue_threshold_low: int = 3,
        queue_threshold_high: int = 10,
        max_queue_length: int = 100,
        # Added kwargs to capture extra args (like queue_threshold/step_size) from legacy calls
        **kwargs 
    ):
        """
        Initialize the adaptive threshold agent.
        """
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.queue_threshold_low = queue_threshold_low
        self.queue_threshold_high = queue_threshold_high
        self.max_queue_length = max_queue_length
        self.name = "adaptive_threshold"
    
    def get_action(
        self, 
        state: np.ndarray, 
        queue_length: int = 0,
        **kwargs
    ) -> int:
        """
        Adjust batch size adaptively based on queue length using linear interpolation.
        """
        # Extract and denormalize queue_length from state if not provided
        if queue_length == 0 and len(state) > 0:
            queue_length = int(state[0] * self.max_queue_length)  # Denormalize
        
        if queue_length == 0:
            return 0
        
        # Linear interpolation between min and max batch size based on queue
        if queue_length <= self.queue_threshold_low:
            batch_size = self.min_batch_size
        elif queue_length >= self.queue_threshold_high:
            batch_size = self.max_batch_size
        else:
            # Interpolate
            ratio = (queue_length - self.queue_threshold_low) / (self.queue_threshold_high - self.queue_threshold_low)
            batch_size = int(self.min_batch_size + ratio * (self.max_batch_size - self.min_batch_size))
        
        return min(batch_size, queue_length)

class GreedyLatencyAgent:
    """
    Greedy agent that tries to minimize latency by batching when it estimates
    that not batching would increase latency too much.
    """
    
    def __init__(
        self, 
        max_batch_size: int = 32,
        latency_threshold: float = 50.0,  # ms
        processing_rate: float = 10.0,    # req/ms
    ):
        """
        Initialize the greedy latency agent.
        """
        self.max_batch_size = max_batch_size
        self.latency_threshold = latency_threshold
        self.processing_rate = processing_rate
        self.name = f"greedy_latency_{int(latency_threshold)}ms"
    
    def get_action(
        self, 
        _state: np.ndarray, 
        queue_length: int = 0,
        **kwargs
    ) -> int:
        """
        Decide batch size based on estimated latency.
        """
        if queue_length == 0:
            return 0
        
        # Estimate processing time for different batch sizes
        best_batch_size = 1
        best_latency = float('inf')
        
        for batch_size in range(1, min(queue_length, self.max_batch_size) + 1):
            # Simple model: processing time increases with batch size
            # and we want to minimize max latency
            processing_time = batch_size / self.processing_rate  # ms
            estimated_latency = processing_time + (queue_length / batch_size) * processing_time
            
            if estimated_latency < best_latency:
                best_latency = estimated_latency
                best_batch_size = batch_size
        
        return best_batch_size
