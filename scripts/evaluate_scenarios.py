
import numpy as np
import json
import os

from environment.simulator import Simulator
from agents.reinforce import REINFORCEAgent
from agents.baselines import StaticBatchingAgent
from environment.scenarios import BurstScenario, SLAScenario, HeterogeneousScenario, AdaptiveScenario

# Synthetic Scenario Evaluation Loop (Parametric Tests)

def evaluate_ppo_scenarios():
    print("Comprehensive PPO Evaluation across 5 Scenarios")
    print("-" * 100)
    print(f"{'Scenario':<25} | {'Static-8':<10} | {'REINFORCE':<10} | {'Improvement':<10}")
    print("-" * 100)
    
    scenarios = [
        ("Standard", None),
        ("Extreme Burst", BurstScenario()),
        ("Strict SLA", SLAScenario()),
        ("Heterogeneous", HeterogeneousScenario()),
        ("Adaptive", AdaptiveScenario())
    ]
    
    results_history = {}
    
    for name, scenario_obj in scenarios:
        # Setup Environment
        env = Simulator(seed=42)
        if scenario_obj:
            env = scenario_obj.apply(env)
            
        # Train REINFORCE Agent
        agent = REINFORCEAgent(state_dim=9, action_dim=33, lr=0.0005)
        
        scenario_history = []
        
        # Train for 500 episodes to give it time to find the optimal policy
        for _ in range(500):
            state = env.reset()
            episode_reward = 0
            for _ in range(1000):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, log_prob, value, done)
                state = next_state
                episode_reward += reward
            agent.update()
            scenario_history.append(episode_reward)
            
        results_history[name] = scenario_history
            
        # Evaluate REINFORCE
        reinforce_returns = []
        for i in range(20):
            # Re-create env with correct seed
            eval_env = Simulator(seed=100 + i)
            if scenario_obj:
                eval_env = scenario_obj.apply(eval_env)
                
            state = eval_env.reset()
            ep_ret = 0
            for _ in range(1000):
                action, _, _ = agent.get_action(state, deterministic=True)
                next_state, reward, done, _ = eval_env.step(action)
                ep_ret += reward
                state = next_state
            reinforce_returns.append(ep_ret)
            
        # Evaluate Static-8 Baseline
        static_agent = StaticBatchingAgent(batch_size=8)
        static_returns = []
        for i in range(20):
            # Re-create env with correct seed
            eval_env = Simulator(seed=100 + i)
            if scenario_obj:
                eval_env = scenario_obj.apply(eval_env)
                
            state = eval_env.reset()
            ep_ret = 0
            for _ in range(1000):
                action = static_agent.get_action(state)
                next_state, reward, done, _ = eval_env.step(action)
                ep_ret += reward
                state = next_state
            static_returns.append(ep_ret)
            
        avg_reinforce = np.mean(reinforce_returns)
        avg_static = np.mean(static_returns)
        imp = ((avg_reinforce - avg_static) / abs(avg_static)) * 100 if avg_static != 0 else 0
        
        print(f"{name:<25} | {avg_static:<10.2f} | {avg_reinforce:<10.2f} | {imp:<10.1f}%")

    # Save history for plotting
    os.makedirs("results", exist_ok=True)
    with open("results/scenarios_history.json", "w") as f:
        json.dump(results_history, f, indent=4)
    print("Saved scenarios history to results/scenarios_history.json")

if __name__ == "__main__":
    evaluate_ppo_scenarios()
