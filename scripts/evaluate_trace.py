
import numpy as np
from tqdm import tqdm

from environment.simulator import Simulator
from agents.ppo import PPOAgent
from agents.reinforce import REINFORCEAgent
from agents.baselines import StaticBatchingAgent
from environment.scenarios import GammaTraceScenario

# Real-World Trace Evaluation Loop (Azure/Gamma)

def evaluate_trace_workload():
    print("Evaluating on Realistic Gamma-Distributed Workload (Mimicking Azure/BurstGPT)")
    print("Configuration: Gamma(k=0.1, theta=100) -> Mean=10 req/s, CV=3.16")
    print("-" * 100)
    print(f"{'Agent':<15} | {'Return':<10} | {'Throughput':<10} | {'Latency':<10} | {'Timeouts':<10}")
    print("-" * 100)
    
    # Setup Env with Gamma Scenario
    env = Simulator(seed=42)
    scenario = GammaTraceScenario(shape=0.1, scale=150.0)
    env = scenario.apply(env)
    
    # Train PPO (Fresh)
    print("Training PPO on Gamma Workload...")
    ppo_agent = PPOAgent(state_dim=9, action_dim=33, lr=3e-4)
    for _ in tqdm(range(200)): # 200 episodes
        state = env.reset()
        for _ in range(1000):
            action, log_prob, value = ppo_agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            ppo_agent.store_transition(state, action, reward, log_prob, value, done)
            state = next_state
        ppo_agent.update()
        
    # Train REINFORCE (Fresh)
    print("Training REINFORCE on Gamma Workload...")
    reinforce_agent = REINFORCEAgent(state_dim=9, action_dim=33, lr=3e-4)
    for _ in tqdm(range(200)):
        state = env.reset()
        for _ in range(1000):
            action, log_prob, value = reinforce_agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            reinforce_agent.store_transition(state, action, reward, log_prob, value, done)
            state = next_state
        reinforce_agent.update()
        
    # Evaluate All
    agents = {
        "Static-8": StaticBatchingAgent(batch_size=8),
        "REINFORCE": reinforce_agent,
        "PPO": ppo_agent
    }
    
    for name, agent in agents.items():
        returns = []
        throughputs = []
        latencies = []
        timeouts = []
        
        for i in range(50): # 50 Eval episodes
            # Re-create env with eval seed
            eval_env = Simulator(seed=1000 + i)
            eval_env = scenario.apply(eval_env)
            
            state = eval_env.reset()
            ep_ret = 0
            for _ in range(1000):
                if name == "Static-8":
                    action = agent.get_action(state)
                else:
                    action, _, _ = agent.get_action(state, deterministic=True)
                    
                next_state, reward, done, metrics = eval_env.step(action)
                ep_ret += reward
                state = next_state
                
            returns.append(ep_ret)
            throughputs.append(metrics['throughput'])
            latencies.append(metrics['avg_latency'])
            timeouts.append(metrics['timed_out_requests'])
            
        print(f"{name:<15} | {np.mean(returns):<10.2f} | {np.mean(throughputs):<10.2f} | {np.mean(latencies):<10.4f} | {np.mean(timeouts):<10.1f}")

if __name__ == "__main__":
    evaluate_trace_workload()
