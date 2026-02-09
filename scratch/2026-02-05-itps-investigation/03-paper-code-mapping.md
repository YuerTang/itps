# Paper to Code Mapping

This document maps concepts from the ITPS paper to their implementations in the codebase.

---

## 1. Alignment Objectives (Paper Section II.A)

### Point Input (Equation 1)
```
ξ(τ, z^point) = Σ (1/T) ||s_t - z^point||_2
```

**Code Location**: `itps/common/policies/diffusion/modeling_diffusion.py`

```python
def guide_gradient(self, sample, guide):
    """
    Compute gradient of L2 distance between sample and guide.
    Used for both point and sketch input alignment.
    """
    # sample: (B, horizon, action_dim)
    # guide: (horizon, action_dim) or (1, action_dim) for point
    distance = torch.sum((sample - guide) ** 2, dim=-1)
    grad = torch.autograd.grad(distance.sum(), sample)[0]
    return grad
```

### Sketch Input (Equation 2)
```
ξ(τ, z^sketch) = Σ ||s_t - z^sketch_t||_2
```

**Code Location**: Same as above, with guide having shape `(horizon, action_dim)`

The sketch is resampled to match trajectory length in `interact_maze2d.py`:
```python
# Resample sketch to match horizon
guide = interpolate_sketch(sketch_points, horizon=64)
```

### Physical Correction Input (Equation 3-4)
```
τ = [z^nudge_1, ..., z^nudge_k, s_{k+1}, ..., s_T]
```

**Code Location**: Implemented via Output Perturbation strategy

---

## 2. Sampling Strategies (Paper Section II.B)

### Random Sampling (RS)

**Paper**: Sample τ ∼ π_θ directly without modification

**Code**: Default inference without guide
```python
# In DiffusionModel.run_inference()
actions = self.run_inference(batch, guide=None)
```

### Output Perturbation (OP)

**Paper**: Overwrite first k steps with z^nudge, resample remainder

**Code Location**: `modeling_diffusion.py`
```python
if strategy == "output-perturb":
    # Overwrite beginning of trajectory with guide
    actions[:, :k, :] = guide[:k, :]
    # Resample from perturbed state
    ...
```

### Post-Hoc Ranking (PR)

**Paper**: Generate batch of N trajectories, select τ* minimizing ξ

**Code Location**: `interact_maze2d.py`
```python
if strategy == "post-hoc":
    # Generate batch of trajectories
    trajectories = policy.run_inference(batch, batch_size=32)

    # Score by L2 distance to sketch
    scores = []
    for traj in trajectories:
        dist = torch.sum((traj - guide) ** 2)
        scores.append(dist)

    # Select best
    best_idx = torch.argmin(torch.tensor(scores))
    selected_trajectory = trajectories[best_idx]
```

### Biased Initialization (BI)

**Paper**: Initialize τ_N with Gaussian-corrupted user input instead of N(0,I)

**Code Location**: `modeling_diffusion.py`
```python
if strategy == "biased-initialization":
    # Add noise to guide to create biased initialization
    noise_level = self.noise_scheduler.timesteps[0]
    noisy_guide = self.noise_scheduler.add_noise(
        guide.unsqueeze(0),
        torch.randn_like(guide.unsqueeze(0)),
        noise_level
    )
    sample = noisy_guide  # Instead of torch.randn(...)
```

### Guided Diffusion (GD)

**Paper Equation 5**:
```
τ_{i-1} = α_i(τ_i - γ_i(ε_θ(τ_i, i) + β_i ∇_{τ_i} ξ(τ_i, z))) + σ_i η
```

**Code Location**: `modeling_diffusion.py`
```python
if strategy == "guided-diffusion":
    for i, t in enumerate(self.noise_scheduler.timesteps):
        # Get model prediction (denoising gradient)
        model_output = self.unet(sample, t, global_cond)

        # Compute alignment gradient if within influence range
        if guide is not None and i < len(self.noise_scheduler.timesteps) - final_influence_step:
            sample.requires_grad_(True)
            grad = self.guide_gradient(sample, guide)
            model_output = model_output + guide_ratio * grad
            sample = sample.detach()

        # Reverse diffusion step
        sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
```

### Stochastic Sampling (SS) ⭐

**Paper Algorithm 1**:
```
for i = N, ..., 1:          // denoising steps
  for j = 1, ..., M:        // sampling steps
    ε ← π_θ(τ_i)            // denoising gradient
    δ ← ∇ξ(τ_i, z)          // alignment gradient
    if j < M:
      τ_i ← reverse(τ_i, ε + β_i δ, i)
    else:
      τ_{i-1} ← reverse(τ_i, ε + β_i δ, i-1)
```

**Code Location**: `modeling_diffusion.py`
```python
if strategy == "stochastic-sampling":
    M = 4  # MCMC sampling steps

    for i, t in enumerate(self.noise_scheduler.timesteps):
        for j in range(M):
            # Denoising gradient
            model_output = self.unet(sample, t, global_cond)

            # Alignment gradient
            if guide is not None:
                sample.requires_grad_(True)
                grad = self.guide_gradient(sample, guide)
                model_output = model_output + guide_ratio * grad
                sample = sample.detach()

            if j < M - 1:
                # MCMC step at same noise level
                # Get clean prediction
                clean_pred = self.noise_scheduler.step(model_output, t, sample).pred_original_sample
                # Re-noise to same level
                sample = self.noise_scheduler.add_noise(clean_pred, torch.randn_like(sample), t)
            else:
                # Final step: advance to next noise level
                sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
```

---

## 3. Metrics (Paper Section II.A)

### Task Alignment (TA)
Percentage of predicted skills executing intended tasks

**Code**: Computed in evaluation scripts, counting successful task switches

### Motion Alignment (MA)
Negative L2 distance between generated and target trajectories

**Code Location**: `interact_maze2d.py`
```python
def compute_alignment(trajectories, guide):
    """Compute L2 distance to guide."""
    distances = []
    for traj in trajectories:
        dist = torch.sqrt(torch.sum((traj - guide) ** 2, dim=-1)).mean()
        distances.append(dist.item())
    return {
        'min_l2': min(distances),
        'avg_l2': sum(distances) / len(distances)
    }
```

### Constraint Satisfaction (CS)
Percentage of plans satisfying physical constraints (e.g., no collisions)

**Code Location**: `interact_maze2d.py`
```python
def check_collision(trajectory, maze_grid):
    """Check if trajectory collides with maze walls."""
    for point in trajectory:
        grid_x, grid_y = world_to_grid(point)
        if maze_grid[grid_y, grid_x] == 1:  # Wall
            return True
    return False

def compute_collision_rate(trajectories, maze_grid):
    collisions = sum(check_collision(t, maze_grid) for t in trajectories)
    return collisions / len(trajectories)
```

---

## 4. Experimental Setup

### Maze2D (Paper Section III.A)

**Paper**:
- Train ACT and DP on 4M navigation steps
- DDIM with N=100 training steps, 10 inference steps
- Guide ratio β=20 for GD, β=60 for SS
- M=4 MCMC steps for SS
- Batch size: 32 trajectories per trial

**Code Configuration** (`weights_dp/pretrained_model/config.json`):
```json
{
    "n_obs_steps": 2,
    "horizon": 64,
    "n_action_steps": 8,
    "num_inference_steps": 10,
    "input_shapes": {
        "observation.state": [2],
        "observation.environment_state": [2]
    },
    "output_shapes": {
        "action": [2]
    }
}
```

**Interactive Application** (`interact_maze2d.py`):
```python
# Hyperparameters matching paper
BATCH_SIZE = 32
GUIDE_RATIO_GD = 20
GUIDE_RATIO_SS = 60
MCMC_STEPS = 4
```

---

## 5. Key Insight: GD vs SS (Paper Figure 3)

### Guided Diffusion Problem

GD samples from **sum** of distributions:
```
p(τ) + q(τ)  where q(τ) ∝ e^{-ξ(τ,z)}
```

When guide doesn't align with any mode → samples out-of-distribution

### Stochastic Sampling Solution

SS samples from **product** of distributions:
```
p(τ) × q(τ)
```

When guide doesn't align with any mode → finds closest in-distribution mode

**Code Implementation Difference**:

GD: Single reverse step per noise level
```python
sample = scheduler.step(model_output + β*grad, t, sample).prev_sample
```

SS: Multiple MCMC steps per noise level
```python
for j in range(M):
    # Multiple steps at same noise level
    if j < M - 1:
        clean = scheduler.step(...).pred_original_sample
        sample = scheduler.add_noise(clean, noise, t)  # Stay at level t
    else:
        sample = scheduler.step(...).prev_sample  # Advance to t-1
```

---

## 6. File Path Reference

| Paper Concept | Code File |
|--------------|-----------|
| Diffusion Policy | `itps/common/policies/diffusion/modeling_diffusion.py` |
| ACT Policy | `itps/common/policies/act/modeling_act.py` |
| Alignment gradient | `modeling_diffusion.py:guide_gradient()` |
| Stochastic Sampling | `modeling_diffusion.py:run_inference(strategy="stochastic-sampling")` |
| Maze2D experiments | `itps/interact_maze2d.py` |
| Collision detection | `interact_maze2d.py:check_collision()` |
| Dataset loading | `itps/common/datasets/lerobot_dataset.py` |
| Normalization | `itps/common/policies/normalize.py` |

