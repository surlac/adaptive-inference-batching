from collections import defaultdict
from typing import Dict, List, Any
from .structures import Request, Batch

class MetricsTracker:
    """
    Handles tracking and updating of environment metrics.
    """
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'timed_out_requests': 0,
            'avg_latency': 0.0,
            'throughput': 0.0,
            'avg_batch_size': 0.0,
            'utilization': 0.0,
            'queue_length': 0,
            'active_batches': 0,
        }
        self.metrics_history = defaultdict(list)
        self.last_update_time = 0.0

    def reset(self):
        """Reset metrics to initial state."""
        self.metrics = {
             'total_requests': 0,
            'completed_requests': 0,
            'timed_out_requests': 0,
            'avg_latency': 0.0,
            'throughput': 0.0,
            'avg_batch_size': 0.0,
            'utilization': 0.0,
             'queue_length': 0,
            'active_batches': 0,
        }
        self.metrics_history = defaultdict(list)
        self.last_update_time = 0.0
    
    def update(
        self,
        current_time: float,
        completed_requests: List[Request],
        timed_out_requests: List[Request],
        completed_batches: List[Batch],
        queue_length: int,
        active_batches_count: int
    ) -> Dict[str, Any]:
        """
        Update performance metrics based on current state.
        """
        time_delta = current_time - self.last_update_time
        if time_delta <= 0:
            return self.metrics.copy()
            
        # Check recent activity (last 1 second)
        recent_completed = [r for r in completed_requests if r.end_time >= current_time - 1.0]
        recent_batches = [b for b in completed_batches if b.end_time >= current_time - 1.0]
        
        if completed_requests:
            avg_latency = sum(
                req.latency for req in completed_requests[-100:]
            ) / len(completed_requests[-100:])
            
            throughput = len(recent_completed)
            
            avg_batch_size = (
                sum(batch.size for batch in completed_batches[-10:]) /
                len(completed_batches[-10:]) if completed_batches[-10:] else 0
            )
            
            # Simple utilization approximation
            processing_time = sum(
                b.end_time - b.start_time for b in recent_batches
            )
            utilization = min(processing_time / 1.0, 1.0)
        else:
            avg_latency = 0.0
            throughput = 0.0
            avg_batch_size = 0.0
            utilization = 0.0
            
        self.metrics.update({
            'completed_requests': len(completed_requests),
            'timed_out_requests': len(timed_out_requests),
            'avg_latency': avg_latency,
            'throughput': throughput,
            'avg_batch_size': avg_batch_size,
            'utilization': utilization,
            'queue_length': queue_length,
            'active_batches': active_batches_count,
        })
        
        # Record history
        for metric, value in self.metrics.items():
            self.metrics_history[metric].append((current_time, value))
            
        self.last_update_time = current_time
        return self.metrics.copy()
