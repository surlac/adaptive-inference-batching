import types
import functools
import numpy as np
from environment.simulator import Request
from .base import Scenario

class GammaTraceScenario(Scenario):
    """
    Simulates realistic Gamma-distributed traffic logic.
    Based on Microsoft Azure trace analysis from:
    "BurstGPT: Efficient and Reliable Serving for Large Language Models"
    (https://arxiv.org/abs/2401.09670)
    """
    def __init__(self, shape=0.1, scale=100.0):
        self.shape = shape
        self.scale = scale

    def apply(self, env):
        # We attach state to the workload instance since that's what we are patching
        env.workload.gamma_shape = self.shape
        env.workload.gamma_scale = self.scale
        env.workload.current_rate = self.shape * self.scale
        
        bound_func = functools.partial(self._generate_requests_gamma_impl)
        env.workload.generate = types.MethodType(bound_func, env.workload)
        return env

    def _generate_requests_gamma_impl(self, workload_instance, current_time, queue, metrics):
        if workload_instance.current_step % 100 == 0:
            workload_instance.current_rate = workload_instance.rng.gamma(workload_instance.gamma_shape, workload_instance.gamma_scale)
            workload_instance.current_rate = min(workload_instance.current_rate, 500.0)
            
        workload_instance.request_rate = workload_instance.current_rate
        avg_requests = workload_instance.request_rate * (workload_instance.time_window_ms / 1000.0)
        num_new_requests = workload_instance.rng.poisson(avg_requests)
        
        for _ in range(num_new_requests):
            if len(queue) >= workload_instance.max_queue_length: break
            
            workload_instance.current_request = workload_instance.request_size_types[0]
            model_type = workload_instance.rng.choice(list(workload_instance.device_delay_params.keys()))
            input_size = workload_instance.current_request['memory_mb']
            max_latency = workload_instance.rng.lognormal(mean=np.log(100), sigma=0.5)
            
            req = Request(
                workload_instance.next_request_id, 
                current_time, 
                model_type, 
                input_size, 
                max_latency
            )
            queue.append(req)
            workload_instance.next_request_id += 1
            metrics['total_requests'] += 1
        workload_instance.current_step += 1
