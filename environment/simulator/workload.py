import numpy as np
import random
from typing import List, Dict, Optional
from .structures import Request

class WorkloadGenerator:
    """
    Handles generation of inference requests based on workload patterns.
    """
    def __init__(
        self, 
        request_rate: float, 
        time_window_ms: float, 
        max_queue_length: int,
        request_size_types: List[Dict[str, int]],
        device_delay_params: Dict[str, tuple],
        seed: Optional[int] = None
    ):
        self.rng = np.random.RandomState(seed)
        self.initial_request_rate = request_rate
        self.request_rate = request_rate
        self.time_window_ms = time_window_ms
        self.max_queue_length = max_queue_length
        self.request_size_types = request_size_types
        self.device_delay_params = device_delay_params
        
        # State
        self.current_step = 0
        self.time_of_day = 0.0
        self.current_request = self.request_size_types[0]
        self.next_request_id = 0
        
    def reset(self):
        self.current_step = 0
        self.time_of_day = 0.0
        self.request_rate = self.initial_request_rate
        self.next_request_id = 0
        self.current_request = random.choice(self.request_size_types)

    def generate(self, current_time: float, queue: List[Request], metrics: Dict) -> None:
        """
        Generate new inference requests and add them to the queue.
        """
        # Real-world workload patterns (can be overridden by scenarios)
        self.time_of_day = (self.current_step % 10000) / 10000
        # Base sinusoidal pattern
        self.request_rate = 10 + 40 * (0.5 + 0.5 * np.sin(2 * np.pi * self.time_of_day))
        
        avg_requests = self.request_rate * (self.time_window_ms / 1000.0)
        num_new_requests = self.rng.poisson(avg_requests)
        
        for _ in range(num_new_requests):
            if len(queue) >= self.max_queue_length: break
                
            self.current_request = random.choice(self.request_size_types)
            model_type = self.rng.choice(list(self.device_delay_params.keys()))
            input_size = self.current_request['memory_mb']
            max_latency = self.rng.uniform(50.0, 200.0)
            
            request = Request(
                request_id=self.next_request_id,
                arrival_time=current_time,
                model_type=model_type,
                input_size=input_size,
                max_latency=max_latency
            )
            
            queue.append(request)
            self.next_request_id += 1
            metrics['total_requests'] += 1
        
        self.current_step += 1
