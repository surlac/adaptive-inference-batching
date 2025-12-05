import types
import functools
from .base import Scenario

class SLAScenario(Scenario):
    """
    Simulates Strict SLA requirements with heavy penalties for misses.
    """
    def apply(self, env):
        # Update workload parameters
        env.workload.initial_request_rate = 10.0
        env.workload.request_rate = 10.0
        
        env._original_step = env.step
        bound_func = functools.partial(self._step_sla_impl)
        env.step = types.MethodType(bound_func, env)
        return env
        
    def _step_sla_impl(self, env_instance, action):
        next_state, _, done, metrics = env_instance._original_step(action)
        completed_in_window = [r for r in env_instance.completed_requests if r.end_time >= env_instance.current_time - env_instance.time_window_ms/1000.0]
        timeouts_in_window = [r for r in env_instance.timed_out_requests if r.end_time >= env_instance.current_time - env_instance.time_window_ms/1000.0]
        sla_success = 0
        sla_fail = 0
        for r in completed_in_window:
            if r.latency <= 0.050: sla_success += 1
            else: sla_fail += 1
        sla_fail += len(timeouts_in_window)
        reward = (sla_success * 1.0) - (sla_fail * 5.0)
        return next_state, reward, done, metrics
