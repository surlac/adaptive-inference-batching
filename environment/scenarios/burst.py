import types
import functools
from environment.simulator import Request
from .base import Scenario

class BurstScenario(Scenario):
    """
    Simulates bursty traffic patterns.
    """
    def apply(self, env):
        # Patch the workload generator's generate method
        # The new method signature must match WorkloadGenerator.generate(self, current_time, queue, metrics)
        bound_func = functools.partial(self._generate_requests_impl)
        env.workload.generate = types.MethodType(bound_func, env.workload)
        return env

    def _generate_requests_impl(self, workload_instance, current_time, queue, metrics):
        if (workload_instance.current_step % 200) < 100:
            workload_instance.request_rate = 100.0
        else:
            workload_instance.request_rate = 0.0
            
        avg_requests = workload_instance.request_rate * (workload_instance.time_window_ms / 1000.0)
        num_new_requests = workload_instance.rng.poisson(avg_requests)
        
        for _ in range(num_new_requests):
            if len(queue) >= workload_instance.max_queue_length: break
            
            workload_instance.current_request = workload_instance.request_size_types[0]
            model_type = workload_instance.rng.choice(list(workload_instance.device_delay_params.keys()))
            input_size = workload_instance.current_request['memory_mb']
            max_latency = workload_instance.rng.uniform(50.0, 200.0)
            
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
