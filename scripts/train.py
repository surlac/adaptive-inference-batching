
import numpy as np
from tqdm import tqdm

from environment.simulator import Simulator
from environment.scenarios import RoutingScenario
from agents.reinforce import REINFORCEAgent
from agents.baselines.routing import RoundRobinRoutingAgent, RandomRoutingAgent, ShortestQueueRoutingAgent

import torch
import os
import json

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_routing():
    print("Training Multi-GPU Routing Agent...")
    print("-" * 80)
    
    # Environment Setup
    env = Simulator(seed=SEED, num_gpus=2)
    scenario = RoutingScenario()
    env = scenario.apply(env)
    
    # Initialize REINFORCE Agent
    # Tuned hyperparameters for routing efficiency
    agent = REINFORCEAgent(
        state_dim=10, 
        action_dim=3, 
        lr=3e-4, 
        gamma=0.95, 
        hidden_dim=128
    )
    
    checkpoint_path = "checkpoints/reinforce_routing.pt"
    history = []
    
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path}...")
        print("To retrain, delete this file.")
        agent.load(checkpoint_path)
    else:
        print("Starting training...")
        for _ in tqdm(range(2000), desc="Episodes"):
            state = env.reset()
            episode_reward = 0
            # Run episode
            for _ in range(1000):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, log_prob, value, done)
                state = next_state
                episode_reward += reward
            
            agent.update()
            history.append(episode_reward)
            
        print(f"Saving model to {checkpoint_path}...")
        agent.save(checkpoint_path)

    print("\nEvaluating Baselines vs RL...")
    
    baseline_agents = {
        "Random": RandomRoutingAgent(),
        "Round-Robin": RoundRobinRoutingAgent(),
        "Shortest-Queue": ShortestQueueRoutingAgent(),
    }
    
    baseline_results = {}
    
    # Evaluate Baselines
    for name, b_agent in baseline_agents.items():
        returns = []
        for i in range(20):
            env = Simulator(seed=100 + i, num_gpus=2)
            env = scenario.apply(env)
            
            state = env.reset()
            ep_ret = 0
            for _ in range(1000):
                action = b_agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_ret += reward
                state = next_state
            returns.append(ep_ret)
        avg_ret = np.mean(returns)
        baseline_results[name] = avg_ret
        print(f"{name:<15}: {avg_ret:.2f}")

    # Evaluate RL Agent
    rl_returns = []
    for i in range(20):
        env = Simulator(seed=100 + i, num_gpus=2)
        env = scenario.apply(env)
        
        state = env.reset()
        ep_ret = 0
        for _ in range(1000):
            action, _, _ = agent.get_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            ep_ret += reward
            state = next_state
        rl_returns.append(ep_ret)
        
    avg_rl = np.mean(rl_returns)
    print(f"{'RL Routing':<15}: {avg_rl:.2f}")
    
    rr_score = baseline_results["Round-Robin"]
    if rr_score != 0:
        imp = ((avg_rl - rr_score) / abs(rr_score)) * 100
        print(f"Improvement over RR: {imp:.1f}%")
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    
    benchmark_data = {
        "Random": baseline_results["Random"],
        "Round-Robin": baseline_results["Round-Robin"],
        "Shortest-Queue": baseline_results["Shortest-Queue"],
        "RL (REINFORCE)": avg_rl,
        "Improvement": (avg_rl - rr_score) / abs(rr_score) if rr_score != 0 else 0
    }
    
    with open("results/routing_benchmark.json", "w") as f:
        json.dump(benchmark_data, f, indent=4)
    print("Saved benchmark results.")

    if history:
        with open("results/training_history.json", "w") as f:
            json.dump(history, f, indent=4)
        print("Saved training history.")

if __name__ == "__main__":
    train_routing()
