from .static import StaticBatchingAgent
from .heuristic import TimeoutBatchingAgent, AdaptiveThresholdAgent, GreedyLatencyAgent
from .routing import RoundRobinRoutingAgent

def create_baseline_agents(max_batch_size: int = 32) -> list:
    """
    Create a list of baseline agents for comparison.
    """
    return [
        StaticBatchingAgent(batch_size=1),
        StaticBatchingAgent(batch_size=4),
        StaticBatchingAgent(batch_size=8),
        StaticBatchingAgent(batch_size=16),
        TimeoutBatchingAgent(max_batch_size=max_batch_size, timeout=5.0),
        TimeoutBatchingAgent(max_batch_size=max_batch_size, timeout=10.0),
        TimeoutBatchingAgent(max_batch_size=max_batch_size, timeout=20.0),
        AdaptiveThresholdAgent(
            max_batch_size=max_batch_size,
        ),
        GreedyLatencyAgent(
            max_batch_size=max_batch_size,
            latency_threshold=50.0
        ),
        GreedyLatencyAgent(
            max_batch_size=max_batch_size,
            latency_threshold=100.0
        ),
    ]
