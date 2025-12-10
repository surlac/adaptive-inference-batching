
import numpy as np
import json
import os
import types
from tqdm import tqdm

from environment.simulator import Simulator
from agents.reinforce import REINFORCEAgent
from agents.baselines import StaticBatchingAgent
from agents.baselines.routing import RoundRobinRoutingAgent, ShortestQueueRoutingAgent, RandomRoutingAgent
from environment.scenarios import HeterogeneousScenario

def evaluate_trace_workload():
    print("Evaluating on Heterogeneous Workload (Fast/Slow Mix) - Multi-GPU Routing")
    print("-" * 100)
    
    # Helper function to apply HeterogeneousScenario with configurable request rate
    def apply_heterogeneous_with_rate(env, rate):
        scenario = HeterogeneousScenario()
        env = scenario.apply(env)
        
        def patched_generate(self, current_time, queue, metrics):
            avg_requests = rate * (self.time_window_ms / 1000.0)
            num_new_requests = self.rng.poisson(avg_requests)
            for _ in range(num_new_requests):
                if len(queue) >= self.max_queue_length: break
                
                self.current_request = self.request_size_types[0]
                if self.rng.random() < 0.5:
                    model_type = "resnet50" # Fast
                    max_latency = 50.0
                else:
                    model_type = "gpt2" # Slow
                    max_latency = 200.0
                    
                input_size = self.current_request['memory_mb']
                from environment.simulator.structures import Request
                req = Request(
                    self.next_request_id, 
                    current_time, 
                    model_type, 
                    input_size, 
                    max_latency
                )
                queue.append(req)
                self.next_request_id += 1
                metrics['total_requests'] += 1
            self.current_step += 1
            
        env.workload.generate = types.MethodType(patched_generate, env.workload)
        return env

    # Custom delays to enforce heterogeneous workload
    # ResNet-50: 10ms (fast model), GPT-2: 200ms (slow model)
    custom_delays = {
        "resnet50": (10.0, 2.0),
        "gpt2": (200.0, 10.0)
    }
    # Train REINFORCE agent
    print("Training REINFORCE on Heterogeneous Workload...")
    reinforce_agent = REINFORCEAgent(state_dim=10, action_dim=65, lr=1e-3)
    
    train_env = Simulator(seed=42, num_gpus=2, device_delay_params=custom_delays)
    train_env = apply_heterogeneous_with_rate(train_env, 40.0)

    # Train for 1000 episodes
    for _ in tqdm(range(1000)):
        state = train_env.reset()
        for _ in range(1000):
            action, log_prob, value = reinforce_agent.get_action(state)
            next_state, reward, done, _ = train_env.step(action)
            reinforce_agent.store_transition(state, action, reward, log_prob, value, done)
            state = next_state
        reinforce_agent.update()
        
    eval_agents = {
        "Random": RandomRoutingAgent(num_gpus=2),
        "Static-8": StaticBatchingAgent(batch_size=8),
        "Round-Robin": RoundRobinRoutingAgent(num_gpus=2),
        "Shortest-Queue": ShortestQueueRoutingAgent(num_gpus=2),
        "RL (REINFORCE)": reinforce_agent
    }

    # Evaluate across different load levels (10-70 req/s)
    rates = [10, 20, 30, 40, 50, 60, 70]
    
    tradeoff_data = {name: {"latency": [], "p99_latency": [], "throughput": [], "is_multi_gpu": []} for name in eval_agents.keys()}
    
    print(f"\nStarting Load Sweep over rates: {rates}")
    
    for rate in rates:
        print(f"\nEvaluating Rate {rate} req/s...")
        
        for name, agent in eval_agents.items():
            scale_latencies = []
            scale_p99_latencies = []
            scale_throughputs = []
            scale_gpu_usage = []
            
            # Run 5 evaluation episodes per configuration
            for i in range(5): 
                eval_env = Simulator(seed=2000 + i + int(rate), num_gpus=2, device_delay_params=custom_delays)
                eval_env = apply_heterogeneous_with_rate(eval_env, float(rate))
                
                state = eval_env.reset()
                for _ in range(1000):
                    if name == "RL (REINFORCE)":
                        res = agent.get_action(state, deterministic=True)
                        if isinstance(res, tuple) and len(res) >= 1:
                            action = res[0]
                        else:
                            action = res
                    else:
                        action = agent.get_action(state)
                        
                    _, _, _, metrics = eval_env.step(action)
                
                # Calculate stable average throughput over the entire 10s episode
                avg_throughput = metrics['completed_requests'] / 10.0
                
                # Calculate P99 Latency
                all_latencies = [r.latency for r in eval_env.completed_requests]
                if all_latencies:
                    p99 = np.percentile(all_latencies, 99)
                    avg_lat = np.mean(all_latencies)
                else:
                    p99 = 0.0
                    avg_lat = 0.0
                
                # Check if GPU 1 was used (Multi-GPU Mode)
                used_gpu1 = any(b.gpu_id == 1 for b in eval_env.completed_batches)
                
                scale_latencies.append(avg_lat)
                scale_p99_latencies.append(p99)
                scale_throughputs.append(avg_throughput)
                scale_gpu_usage.append(1 if used_gpu1 else 0)
            
            avg_lat = np.mean(scale_latencies)
            avg_p99 = np.mean(scale_p99_latencies)
            avg_thr = np.mean(scale_throughputs)
            avg_gpu_usage = np.mean(scale_gpu_usage)
            
            tradeoff_data[name]["latency"].append(avg_lat)
            tradeoff_data[name]["p99_latency"].append(avg_p99)
            tradeoff_data[name]["throughput"].append(avg_thr)
            tradeoff_data[name]["is_multi_gpu"].append(True if avg_gpu_usage > 0.5 else False)
            
            print(f"  {name}: Lat={avg_lat:.4f}, P99={avg_p99:.4f}, Thr={avg_thr:.2f}, MultiGPU={avg_gpu_usage > 0.5}")

    # Save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/trace_metrics.json", "w") as f:
        json.dump(tradeoff_data, f, indent=4)
    print("Saved trace metrics to results/trace_metrics.json")

if __name__ == "__main__":
    evaluate_trace_workload()
