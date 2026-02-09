# ITPS Paper Summary

**Title**: Inference-Time Policy Steering through Human Interactions
**Authors**: Yanwei Wang, Lirui Wang, Yilun Du, et al. (MIT CSAIL + NVIDIA)
**arXiv**: 2411.16627v2 [cs.RO] 26 Mar 2025

---

## Abstract

Generative policies trained with human demonstrations can autonomously accomplish multimodal, long-horizon tasks. However, during inference, humans are removed from the policy execution loop, limiting the ability to guide a pre-trained policy towards specific sub-goals or trajectory shapes.

**Problem**: Naive human intervention may exacerbate distribution shift → constraint violations or execution failures.

**Solution**: ITPS framework that leverages human interactions to bias the generative sampling process, rather than fine-tuning the policy on interaction data.

**Key Result**: Among six sampling strategies, **Stochastic Sampling with Diffusion Policy** achieves the best trade-off between alignment and distribution shift.

---

## 1. Introduction

### Motivation
- Behavior cloning has enabled generalist policies (RT-X, Octo, OpenVLA) capable of solving multiple tasks
- As models acquire more skills, the question becomes: **how can skills be tailored to follow specific user objectives?**
- Currently, few mechanisms exist to intervene and correct policy behavior at inference time

### Why Not Fine-Tuning?
- Requires additional data collection and training
- Language may not be the best modality for capturing low-level, continuous intent
- This work explores: **Can a frozen pre-trained policy be steered without fine-tuning?**

### The Distribution Shift Problem
- Inference-time interventions in task space can exacerbate distribution shift
- Prior works focus on single-task settings, limiting applicability to multi-task policies
- ITPS leverages multimodal generative models to produce trajectories that respect likelihood constraints

### Key Insight
Frame policy steering as **conditional sampling** from the likelihood distribution of a learned generative policy:
- Likelihood constraints → consistently synthesize valid trajectories
- Conditional sampling → trajectories align with user objectives

---

## 2. Policy Steering

### 2.1 Steering Towards User Intent

**Three Metrics:**
1. **Task Alignment (TA)**: Percentage of predicted skills that execute intended tasks (discrete)
2. **Motion Alignment (MA)**: Negative L2 distance between generated and target trajectories (continuous)
3. **Constraint Satisfaction (CS)**: Percentage of plans that satisfy physical constraints (implicit user intent)

**Definition**: Steering towards user intent = increasing TA or MA while maximizing CS

### Three Interaction Types and Objective Functions

#### Point Input
User specifies a pixel → mapped to 3D state z^point via depth info

```
ξ(τ, z^point) = Σ (1/T) ||s_t - z^point||_2
```
Average L2 distance between all trajectory states and target point.

#### Sketch Input
User draws partial trajectory sketch z^sketch ∈ R^(T×3)

```
ξ(τ, z^sketch) = Σ ||s_t - z^sketch_t||_2
```
Sum of L2 distances at each timestep (sketch resampled if length differs).

#### Physical Correction Input
User physically corrects first k steps of trajectory

```
ξ(τ, z^nudge) = { 0,     if s_t = z^nudge_t for t ≤ k
                { ∞,     otherwise
```
Overwrites beginning of trajectory with user-specified motion.

---

### 2.2 Inference-Time Interaction-Conditioned Sampling

Six methods for biasing trajectory generation:

#### Policy-Agnostic Methods (work with any generative model)

**1. Random Sampling (RS)** - Baseline
- Sample τ ∼ π_θ directly without modification
- No explicit optimization of objective ξ

**2. Output Perturbation (OP)**
- Sample trajectory, apply post-hoc perturbation to minimize ξ(τ, z^nudge)
- Resample from z^nudge_k to complete remainder
- Maximizes alignment up to step k, but no constraint satisfaction guarantee

**3. Post-Hoc Ranking (PR)**
- Generate batch of N trajectories {τ_j}
- Select τ* that minimizes ξ(τ, z^point) or ξ(τ, z^sketch)
- Works well when at least one sample aligns with input
- Cannot discover modes absent from initial batch

#### Diffusion-Specific Methods

**4. Biased Initialization (BI)**
- Instead of τ_N ∼ N(0,I), use Gaussian-corrupted version of user input z
- Brings diffusion process closer to desired mode from start
- Sampling may still deviate from input

**5. Guided Diffusion (GD)**
- Use objective ξ to guide trajectory synthesis during diffusion
- At each timestep i, compute alignment gradient ∇_{τ_i} ξ(τ_i, z)

```
τ_{i-1} = α_i(τ_i - γ_i(ε_θ(τ_i, i) + β_i ∇_{τ_i} ξ(τ_i, z))) + σ_i η
```

- **Problem**: Samples from weighted SUM of distributions, not product
- Can result in out-of-distribution samples

**6. Stochastic Sampling (SS)** ⭐ Best Method
- Uses annealed MCMC to optimize COMPOSITION of diffusion model and objective
- Samples from PRODUCT of distributions: p_0(τ)q(τ)

```
Algorithm: Stochastic Sampling
1: Initialize τ_N ∼ N(0, I)
2: for i = N, ..., 1:          // denoising steps
3:   for j = 1, ..., M:        // sampling steps (M=4)
4:     ε ← π_θ(τ_i)            // denoising gradient
5:     δ ← ∇ξ(τ_i, z)          // alignment gradient
6:     if j < M:
7:       τ_i ← reverse(τ_i, ε + β_i δ, i)
8:     else:
9:       τ_{i-1} ← reverse(τ_i, ε + β_i δ, i-1)
```

**Why SS Works Better**:
- GD: sum of distributions → can sample out-of-distribution when input doesn't align with any mode
- SS: product of distributions → identifies closest in-distribution mode

---

## 3. Experiments

### 3.1 Maze2D - Continuous Motion Alignment

**Setup:**
- Train ACT and Diffusion Policy on 4M navigation steps (collision-free random walks)
- No goal-oriented objectives during training
- Test: 100 random locations, each with sketch z^sketch (may violate collision constraints)
- Batch of 32 trajectories per trial

**Results (Table I):**
| Method | Min L2 ↓ | Avg L2 ↓ | Collision ↓ |
|--------|----------|----------|-------------|
| DP: RS | 0.27 | 0.28 | 0.01 |
| DP: PR | 0.16 | 0.28 | 0.01 |
| DP: BI | 0.11 | 0.14 | 0.06 |
| DP: GD | 0.11 | 0.18 | 0.06 |
| **DP: SS** | **0.10** | **0.12** | **0.01** |

**Key Findings:**
1. Steering frozen policies improves alignment at cost of constraint satisfaction
2. Multimodal policies (DP) + PR enhance alignment without significant distribution shift
3. Unimodal policies (ACT) are harder to steer effectively
4. **SS achieves best alignment-constraint satisfaction trade-off**

### 3.2 Block Stacking - Discrete Task Alignment

**Setup:**
- 4-block stacking in Isaac Sim
- Train DP on 5M steps from CuRobo motion planner
- VR-based system for 3D sketch input

**Results (Table II):**
| Method | TA (Alignment) | CS (Success) | Aligned Success |
|--------|----------------|--------------|-----------------|
| PR | 33% | 100% | 33% |
| GD (β<50=0) | 83% | 84% | **67%** |
| GD (β=100) | 86% | 15% | 15% |

**Key Finding**: Deactivating steering in later diffusion steps (β_{i≤I}=0) balances alignment vs. distribution adherence.

### 3.3 Real World Kitchen - Discrete Task Alignment

**Setup:**
- Toy kitchen with two tasks: place bowl in microwave / sink
- 60 demonstrations per task, combined dataset
- DP trained for 40K steps
- Real-time policy rollouts at 7 Hz

**Results (Table III):**
| Method | TA | CS | Aligned Success |
|--------|-----|-----|-----------------|
| RS | 38% | 90% | 34% |
| GD (point) | 37% | 82% | 32% |
| **SS (point)** | **71%** | **73%** | **55%** |
| OP (correction) | 89% | 37% | 30% |

**Key Finding**: SS improves Aligned Success by **21%** without any fine-tuning.

---

## 4. Key Contributions

1. **Novel Framework**: ITPS incorporates real-time user interactions to steer frozen imitation policies

2. **Alignment Objectives**: Set of objectives with sampling methods, illustrating alignment-constraint satisfaction trade-off

3. **Stochastic Sampling Algorithm**: New inference algorithm for diffusion policy that improves alignment while maintaining constraints within data manifold

---

## 5. Limitations and Future Work

**Limitation**: Reliance on expensive sampling procedure to produce aligned behaviors

**Future Work**:
- Distill steering process into interaction-conditioned policy for faster responses
- Conduct user study to validate steerability

---

## 6. Technical Details

### Diffusion Policy Configuration
- DDIM scheduler with N=100 training steps
- 10 inference steps
- Guide ratio β=20 for GD, β=60 for SS
- M=4 MCMC sampling steps for SS

### Guided Diffusion vs. Stochastic Sampling (Figure 3)

**GD**: Samples approximate SUM of two distributions
- When point input doesn't align with any mode → introduces distribution shift

**SS**: Samples approximate PRODUCT of distributions
- When point input doesn't align with any mode → identifies closest in-distribution mode

This is the key theoretical insight that explains SS's superior performance.

