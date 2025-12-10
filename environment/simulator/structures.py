from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"

@dataclass
class Request:
    """Represents an inference request in the system."""
    request_id: int
    arrival_time: float
    model_type: str
    input_size: int
    max_latency: float
    status: RequestStatus = RequestStatus.QUEUED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    batch_id: Optional[int] = None
    
    @property
    def latency(self) -> Optional[float]:
        if self.end_time is not None and self.start_time is not None:
            return self.end_time - self.arrival_time
        return None

@dataclass
class Batch:
    """Represents a batch of inference requests."""
    batch_id: int
    requests: List[Request] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    gpu_id: Optional[int] = None
    
    @property
    def size(self) -> int:
        return len(self.requests)
    
    @property
    def total_input_size(self) -> int:
        return sum(req.input_size for req in self.requests)
