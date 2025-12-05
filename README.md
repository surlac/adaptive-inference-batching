# Adaptive Inference Batching using Policy Gradients

This repository contains the implementation of a Reinforcement Learning (RL) agent for adaptive batching and routing in inference serving systems. We model the problem as a Markov Decision Process (MDP) and use Policy Gradient methods (REINFORCE, PPO) to optimize the trade-off between throughput and latency.

## Abstract

Inference serving systems face the challenge of balancing high throughput with low latency, especially under bursty and heterogeneous workloads. Static batching policies often fail to adapt to dynamic traffic patterns. In this work, we demonstrate that a Policy Gradient-based routing agent achieves a **3.5x performance improvement** over Round-Robin scheduling in multi-GPU environments by dynamically segregating heterogeneous workloads to minimize Head-of-Line blocking.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/surlac/adaptive-inference-batching.git
    cd adaptive-inference-batching
    ```

2.  **Install dependencies**:
    ```bash
    pip install -e .
    ```

## Quick Start

### 1. Train the Agent
To train the REINFORCE agent on the Multi-GPU Routing task:

```bash
python -m scripts.train
```
*This will train the agent for 2000 episodes and save the model to `checkpoints/`. If the model already exists, training is skipped.*

**Note:** To reproduce the full learning curve from scratch (and overwrite `results/training_history.json`), delete the existing checkpoint before running:
```bash
rm checkpoints/reinforce_routing.pt
python -m scripts.train
```

### 2. Evaluate Performance
To evaluate the trained agent against baselines (Random, Round-Robin, Shortest-Queue) on Standard, Burst, and Multi-GPU scenarios:

```bash
python -m scripts.evaluate_scenarios
```

To evaluate on the **Real-World Trace** (Gamma Workload) scenario:

```bash
python -m scripts.evaluate_trace
```

### 3. Generate Plots
Results saved to `results/figures/`:

```bash
python -m scripts.plot_results
```

## Repository Structure

*   `environment/`: Simulator and Scenarios.
    *   `simulator/`: Core simulation logic (`Simulator`, `WorkloadGenerator`, `ComputeCluster`).
    *   `scenarios/`: Workload scenarios (`Burst`, `Gamma`, `Routing`).
*   `agents/`: RL Agents.
    *   `reinforce.py`: REINFORCE implementation.
    *   `ppo.py`: PPO implementation.
*   `models/`: Network architectures (`policy.py`).
*   `scripts/`: Execution scripts.
    *   `train.py`: Main training loop.
    *   `evaluate_scenarios.py`: Evaluation against baselines.
    *   `evaluate_trace.py`: Trace-driven evaluation.
    *   `plot_results.py`: Generates figures for the report.


