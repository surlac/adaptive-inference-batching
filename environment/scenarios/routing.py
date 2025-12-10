import types
import functools
from environment.simulator import Request
from .base import Scenario

class RoutingScenario(Scenario):
    """
    Simulates Multi-GPU routing with fixed batch sizes.
    """
    def apply(self, env):
        bound_gen = functools.partial(self._generate_requests_mixed_impl)
        env.workload.generate = types.MethodType(bound_gen, env.workload)
        
        env._original_step = env.step
        bound_step = functools.partial(self._step_routing_impl)
        env.step = types.MethodType(bound_step, env)
        return env

    def _generate_requests_mixed_impl(self, workload_instance, current_time, queue, metrics):
        workload_instance.request_rate = 80.0 
        avg_requests = workload_instance.request_rate * (workload_instance.time_window_ms / 1000.0)
        num_new_requests = workload_instance.rng.poisson(avg_requests)
        for _ in range(num_new_requests):
            if len(queue) >= workload_instance.max_queue_length: break
            
            workload_instance.current_request = workload_instance.request_size_types[0]
            if workload_instance.rng.random() < 0.5:
                model_type = "resnet50"
                max_latency = 50.0
            else:
                model_type = "gpt2"
                max_latency = 200.0
                
            input_size = workload_instance.current_request['memory_mb']
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
    
    def _step_routing_impl(self, env_instance, action):
        # Action mapping: 0=Wait, 1=GPU0(Batch8), 2=GPU1(Batch8)
        # Also supports standard action space (1-32 -> GPU0, 33-64 -> GPU1) for baselines
        real_action = 0
        
        if action == 1: real_action = 8
        elif action == 2: real_action = 40
        elif 1 <= action <= 32: real_action = 8   # Map any GPU0 batch size to fixed 8
        elif 33 <= action <= 64: real_action = 40 # Map any GPU1 batch size to fixed 8
            
        next_state, _, done, metrics = env_instance._original_step(real_action)
        
        completed_in_window = [r for r in env_instance.completed_requests if r.end_time >= env_instance.current_time - env_instance.time_window_ms/1000.0]
        throughput = len(completed_in_window)
        weighted_latency = sum(r.latency * 10.0 for r in completed_in_window)
        
        penalty = 0.0
        if action > 0 and throughput == 0:
             penalty = -0.1
             
        reward = (throughput * 1.0) - weighted_latency + penalty
        return next_state, reward, done, metrics
