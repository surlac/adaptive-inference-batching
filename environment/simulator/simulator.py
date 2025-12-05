import numpy as np

from typing import List, Dict, Tuple, Optional
from .structures import RequestStatus
from .metrics import MetricsTracker
from .workload import WorkloadGenerator
from .compute import ComputeCluster

class Simulator:
    """
    Gym-style environment for simulating ML inference batching with reinforcement learning.
    """
    def __init__(
        self,
        max_batch_size: int = 32,
        max_queue_length: int = 100,
        time_window_ms: float = 10.0,
        max_wait_time_ms: float = 100.0,
        device_delay_params: Dict[str, Tuple[float, float]] = None,
        request_rate: float = 10.0,
        gpu_memory_mb: float = 8192.0,
        batch_overhead_ms: float = 2.0,
        num_gpus: int = 1,
        seed: Optional[int] = None,
        request_size_types: List[Dict[str, int]] = None
    ):
        self.max_batch_size = max_batch_size
        self.max_queue_length = max_queue_length
        self.time_window_ms = time_window_ms
        self.max_wait_time_ms = max_wait_time_ms
        self.num_gpus = num_gpus
        self.seed = seed
        
        self.device_delay_params = device_delay_params or {
            "resnet50": (10.0, 2.0),
            "bert-base": (15.0, 3.0),
            "gpt2": (25.0, 5.0),
        }
        
        self.request_size_types = request_size_types or [
            {'memory_mb': 50, 'processing_ms': 2},
            {'memory_mb': 150, 'processing_ms': 5},
            {'memory_mb': 300, 'processing_ms': 10}
        ]

        # Components
        self.tracker = MetricsTracker()
        self.workload = WorkloadGenerator(
            request_rate=request_rate,
            time_window_ms=time_window_ms,
            max_queue_length=max_queue_length,
            request_size_types=self.request_size_types,
            device_delay_params=self.device_delay_params,
            seed=seed
        )
        self.compute = ComputeCluster(
            num_gpus=num_gpus,
            gpu_memory_mb=gpu_memory_mb,
            batch_overhead_ms=batch_overhead_ms,
            device_delay_params=self.device_delay_params,
            seed=seed
        )
        
        # Initialize state via reset (DRY)
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_time = 0.0
        self.queue = []
        self.completed_batches = []
        self.completed_requests = []
        self.timed_out_requests = []
        self.current_step = 0
        self.current_revenue_rate = 0.0
        self.avg_processing_time = 0.0

        # Reset Components
        self.tracker.reset()
        self.metrics = self.tracker.metrics
        self.workload.reset()
        self.compute.reset()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_time += self.time_window_ms / 1000.0
        
        # Update Compute Cluster (Process completions)
        new_batches, new_reqs = self.compute.process_completed_batches(self.current_time)
        self.completed_batches.extend(new_batches)
        self.completed_requests.extend(new_reqs)
        
        # Check for Timeouts
        self._process_timeouts()
        
        # Ingest new workload
        self.workload.generate(self.current_time, self.queue, self.metrics)
        self.current_step += 1
        
        # Schedule compute actions
        if action > 0:
            if self.num_gpus == 1:
                gpu_id = 0
                batch_size = action
            else:
                if action == 0:
                    batch_size = 0
                    gpu_id = 0
                else:
                    adjusted_action = action - 1
                    batch_size = (adjusted_action % self.max_batch_size) + 1
                    gpu_id = adjusted_action // self.max_batch_size
            
            if batch_size > 0:
                self.compute.schedule_batch(batch_size, gpu_id, self.queue, self.current_time)
        
        # Calculate metrics and current reward
        self.metrics = self.tracker.update(
            self.current_time, 
            self.completed_requests, 
            self.timed_out_requests, 
            self.completed_batches,
            len(self.queue),
            len(self.compute.active_batches)
        )
        
        num_completed = len([r for r in self.completed_requests if r.end_time >= self.current_time - self.time_window_ms/1000.0])
        num_timeouts = len([r for r in self.timed_out_requests if r.end_time and r.end_time >= self.current_time - self.time_window_ms/1000.0])
        throughput = num_completed
        latency = self.metrics['avg_latency']
        
        reward = self._calculate_reward(throughput, num_timeouts, latency)
        done = False
        next_state = self._get_state()
        
        return next_state, reward, done, self.metrics.copy()
    
    def _get_state(self) -> np.ndarray:
        time_since_last_batch = (
            self.current_time - max([b.end_time for b in self.completed_batches], default=0.0)
            if self.completed_batches else self.current_time
        )
        
        fast_reqs = 0
        slow_reqs = 0
        for i in range(min(len(self.queue), 32)):
            if self.queue[i].model_type == "resnet50": fast_reqs += 1
            else: slow_reqs += 1
                
        state = [
            len(self.queue) / self.max_queue_length,
            time_since_last_batch / self.time_window_ms,
            self.workload.current_request['memory_mb'] / 500,
            self.workload.time_of_day,
            self.current_revenue_rate / 100,
            self.avg_processing_time / 20,
            fast_reqs / 32.0,
            slow_reqs / 32.0
        ]
        
        for i in range(self.num_gpus):
            busy_time = max(0.0, self.compute.gpu_busy_until[i] - self.current_time)
            state.append(min(busy_time / 0.1, 1.0))
        
        return np.array(state, dtype=np.float32)

    def _process_timeouts(self) -> None:
        i = 0
        while i < len(self.queue):
            req = self.queue[i]
            if self.current_time - req.arrival_time > (self.max_wait_time_ms / 1000.0):
                req.status = RequestStatus.TIMED_OUT
                req.end_time = self.current_time
                self.timed_out_requests.append(req)
                self.queue.pop(i)
                self.metrics['timed_out_requests'] += 1
            else:
                i += 1

    def _calculate_reward(self, throughput: int, num_timeouts: int, latency: float) -> float:
        """
        Calculate the reward function for the RL agent.
        
        NOTE: This is the default reward function. Specific scenarios (e.g., HeterogeneousScenario)
        may override this method to implement custom objectives (e.g., weighted latency).
        
        Formula:
            Reward = (throughput - 5 * num_timeouts) / 2.7 - latency / 100
            
        Components:
            - Throughput: Number of requests completed in the last step.
            - Timeouts: Heavy penalty (5x) for dropped requests.
            - Latency: Penalty for average latency (scaled down by 100).
            - Scaling (2.7): Normalization factor to keep rewards in reasonable range.
            
        Args:
            throughput: Number of completed requests
            num_timeouts: Number of timed out requests
            latency: Average latency in ms
            
        Returns:
            float: Calculated reward
        """
        # Reward = Throughput (penalize timeouts) - Latency Penalty
        return (throughput - 5 * num_timeouts) / 2.7 - latency / 100
