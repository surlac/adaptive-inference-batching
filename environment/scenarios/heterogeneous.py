import types
import functools
from environment.simulator import Request
from .base import Scenario

class HeterogeneousScenario(Scenario):
    """
    Simulates mixed model workloads (ResNet vs GPT2) with different latency sensitivities.
    """
    def apply(self, env):
        bound_gen = functools.partial(self._generate_requests_mixed_impl)
        env.workload.generate = types.MethodType(bound_gen, env.workload)
        
        env._original_step = env.step
        bound_step = functools.partial(self._step_mixed_impl)
        env.step = types.MethodType(bound_step, env)
        return env

    def _generate_requests_mixed_impl(self, workload_instance, current_time, queue, metrics):
        workload_instance.request_rate = 40.0 
        avg_requests = workload_instance.request_rate * (workload_instance.time_window_ms / 1000.0)
        num_new_requests = workload_instance.rng.poisson(avg_requests)
        for _ in range(num_new_requests):
            if len(queue) >= workload_instance.max_queue_length: break
            
            workload_instance.current_request = workload_instance.request_size_types[0]
            if workload_instance.rng.random() < 0.5:
                model_type = "resnet50" # Fast
                max_latency = 50.0
            else:
                model_type = "gpt2" # Slow
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
    
    def _step_mixed_impl(self, env_instance, action):
        next_state, _, done, metrics = env_instance._original_step(action)
        completed_in_window = [r for r in env_instance.completed_requests if r.end_time >= env_instance.current_time - env_instance.time_window_ms/1000.0]
        weighted_latency = 0.0
        throughput = len(completed_in_window)
        for r in completed_in_window:
            if r.model_type == "resnet50": weighted_latency += r.latency * 200.0
            else: weighted_latency += r.latency * 20.0
        
        reward = (throughput * 1.0) - weighted_latency
        return next_state, reward, done, metrics
