import numpy as np
import heapq
from typing import List, Dict, Optional, Tuple
from .structures import Request, Batch, RequestStatus

class ComputeCluster:
    """
    Manages GPU resources and batch processing simulation.
    """
    def __init__(
        self,
        num_gpus: int,
        gpu_memory_mb: float,
        batch_overhead_ms: float,
        device_delay_params: Dict[str, tuple],
        seed: Optional[int] = None
    ):
        self.num_gpus = num_gpus
        self.gpu_memory_mb = gpu_memory_mb
        self.batch_overhead_ms = batch_overhead_ms
        self.device_delay_params = device_delay_params
        self.rng = np.random.RandomState(seed)
        
        # State
        self.gpu_busy_until = np.zeros(num_gpus)
        self.active_batches = [] # Heap of (end_time, batch)
        self.next_batch_id = 0
        
    def reset(self):
        self.gpu_busy_until = np.zeros(self.num_gpus)
        self.active_batches = []
        self.next_batch_id = 0

    def schedule_batch(
        self, 
        batch_size: int, 
        gpu_id: int, 
        queue: List[Request], 
        current_time: float
    ) -> bool:
        """
        Attempts to schedule a batch on the specified GPU.
        Returns True if successful, False otherwise.
        """
        if not queue or batch_size <= 0: return False
        if gpu_id >= self.num_gpus: return False
        
        # Determine start time (available when GPU is free)
        start_time = max(current_time, self.gpu_busy_until[gpu_id])
        
        # Select requests respecting memory constraints
        batch_requests = []
        total_memory = 0.0
        
        # Select requests from the queue that fit within GPU memory limits (First-Fit).
        for _ in range(min(batch_size, len(queue))):
            if not queue: break
            
            # Look at first item
            req = queue[0]
            
            # Simple greedy: if it fits, take it.
            if total_memory + req.input_size <= self.gpu_memory_mb:
                # Remove from queue
                r = queue.pop(0)
                batch_requests.append(r)
                total_memory += r.input_size
            else:
                # Stop adding requests if the next one doesn't fit (Head-of-Line blocking memory constraint).
                # Yes, it breaks on first non-fit.
                break
                
        if not batch_requests:
            return False
            
        # Create Batch
        batch = Batch(batch_id=self.next_batch_id, requests=batch_requests, gpu_id=gpu_id)
        self.next_batch_id += 1
        
        batch.start_time = start_time
        for req in batch.requests:
            req.status = RequestStatus.PROCESSING
            req.start_time = current_time # Request is considered 'scheduled' now
            
            req.batch_id = batch.batch_id
            
        # Calc processing time
        max_processing_time = 0.0
        total_data_size = 0.0
        
        for req in batch.requests:
            mean_delay, std_delay = self.device_delay_params[req.model_type]
            processing_time = max(1.0, self.rng.normal(mean_delay, std_delay))
            max_processing_time = max(max_processing_time, processing_time)
            total_data_size += req.input_size
            
        batch_size_penalty = 1.0 + (0.01 * len(batch_requests))
        memory_transfer_time = (total_data_size / 1000.0) * 0.1
        
        total_time_ms = (
            self.batch_overhead_ms + 
            max_processing_time * batch_size_penalty +
            memory_transfer_time
        )
        
        batch.end_time = start_time + (total_time_ms / 1000.0)
        self.gpu_busy_until[gpu_id] = batch.end_time
        heapq.heappush(self.active_batches, (batch.end_time, batch))
        
        return True

    def process_completed_batches(self, current_time: float) -> Tuple[List[Batch], List[Request]]:
        """
        Returns (completed_batches, completed_requests)
        """
        completed_batches = []
        completed_requests = []
        
        while self.active_batches and self.active_batches[0][0] <= current_time:
            end_time, batch = heapq.heappop(self.active_batches)
            
            for req in batch.requests:
                req.status = RequestStatus.COMPLETED
                req.end_time = end_time
                completed_requests.append(req)
                
            completed_batches.append(batch)
            
        return completed_batches, completed_requests
