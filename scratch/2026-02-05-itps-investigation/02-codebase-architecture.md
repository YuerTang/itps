# ITPS Codebase Architecture

## Project Structure

```
/Users/yuertang/Desktop/MIT/itps/
├── README.md                                    # Installation & usage guide
├── pyproject.toml                              # Poetry project config
├── LICENSE                                      # MIT License
├── Inference-Time Policy Steering...pdf        # Research paper
├── itps/                                        # Main package
│   ├── interact_maze2d.py                      # Interactive GUI application (540 lines)
│   ├── common/                                 # Shared utilities
│   │   ├── policies/                           # Policy implementations
│   │   │   ├── policy_protocol.py             # Policy interface protocol
│   │   │   ├── factory.py                     # Policy factory functions
│   │   │   ├── normalize.py                   # Normalization classes
│   │   │   ├── rollout_wrapper.py             # Rollout management (306 lines)
│   │   │   ├── utils.py                       # Policy utilities
│   │   │   ├── diffusion/                     # Diffusion Policy
│   │   │   │   ├── modeling_diffusion.py      # Core implementation (400+ lines)
│   │   │   │   └── configuration_diffusion.py # Config dataclass
│   │   │   └── act/                           # Action Chunking Transformer
│   │   │       ├── modeling_act.py            # ACT implementation
│   │   │       └── configuration_act.py       # Config dataclass
│   │   ├── datasets/                          # Dataset handling
│   │   │   ├── lerobot_dataset.py            # LeRobot dataset loader
│   │   │   ├── factory.py                    # Dataset factory
│   │   │   ├── transforms.py                 # Image transforms
│   │   │   ├── compute_stats.py              # Stats aggregation
│   │   │   ├── sampler.py                    # Data samplers
│   │   │   ├── utils.py                      # Dataset utilities
│   │   │   └── video_utils.py                # Video decoding
│   │   ├── envs/                             # Environment factories
│   │   │   ├── factory.py                    # make_env()
│   │   │   └── utils.py                      # Env utilities
│   │   ├── utils/                            # General utilities
│   │   │   ├── utils.py                      # Device, random state, Hydra
│   │   │   ├── benchmark.py                  # Benchmarking
│   │   │   ├── import_utils.py               # Dynamic imports
│   │   │   └── io_utils.py                   # I/O operations
│   │   └── logger.py                         # Training/logging (247 lines)
│   ├── weights_act/pretrained_model/         # Pre-trained ACT weights
│   │   └── config.json                       # ACT configuration
│   └── weights_dp/pretrained_model/          # Pre-trained Diffusion weights
│       └── config.json                       # DP configuration
└── media/                                     # Visualization GIFs
```

---

## Core Components

### 1. Policy Protocol (`itps/common/policies/policy_protocol.py`)

Defines the interface that all policies must implement:

```python
class Policy(Protocol):
    """Runtime-checkable Protocol for policies."""

    @property
    def n_obs_steps(self) -> int: ...

    @property
    def input_keys(self) -> list[str]: ...

    def forward(self, batch: dict) -> dict: ...

    def run_inference(self, batch: dict) -> dict: ...
```

### 2. Diffusion Policy (`itps/common/policies/diffusion/modeling_diffusion.py`)

The main policy class implementing ITPS steering methods:

```python
class DiffusionPolicy(nn.Module, PyTorchModelHubMixin):
    """High-level policy wrapper with alignment strategies."""

    # Key alignment strategies implemented:
    # - post-hoc: Similarity ranking of sampled trajectories
    # - biased-initialization: Initialize with guide-biased noise
    # - guided-diffusion: Gradient-based guidance during denoising
    # - stochastic-sampling: Multi-step MCMC-like sampling
    # - output-perturb: Perturbation at output

class DiffusionModel(nn.Module):
    """Core diffusion model with inference logic."""

    def guide_gradient(self, sample, guide):
        """Compute L2 distance gradient to guide samples toward user sketches."""
        ...

    def run_inference(self, batch, guide=None, strategy="stochastic-sampling"):
        """Generate trajectories with optional steering."""
        ...

class DiffusionConditionalUnet1d(nn.Module):
    """U-Net architecture for action sequence diffusion."""
    ...

class DiffusionRgbEncoder(nn.Module):
    """Vision backbone encoder for image observations."""
    ...
```

**Key Configuration** (`configuration_diffusion.py`):
- `n_obs_steps=2`: Number of observation steps
- `horizon=64`: Planning horizon (trajectory length)
- `n_action_steps=8`: Actions executed per inference
- DDIM scheduler with 10 inference steps
- Down-dims: [128, 256, 512]

### 3. ACT Policy (`itps/common/policies/act/modeling_act.py`)

Action Chunking Transformer for comparison:

```python
class ACTPolicy(nn.Module):
    """VAE-based transformer policy."""

class ACT(nn.Module):
    """Transformer encoder-decoder with optional VAE."""
    # - ResNet18 vision backbone
    # - Multi-head attention
    # - Chunk-based action prediction
```

**Key Configuration** (`configuration_act.py`):
- `n_obs_steps=1`: Single observation step
- `chunk_size=64`: Action chunk size
- `n_action_steps=64`: Full chunk execution
- VAE with `latent_dim=32`

### 4. Interactive Application (`itps/interact_maze2d.py`)

Pygame-based GUI with three modes:

```python
class UnconditionalMaze:
    """Drag agent to explore policy's motion manifold."""
    # - Real-time inference triggered by mouse movement
    # - Displays batch of 32 sampled trajectories

class ConditionalMaze:
    """Interactive sketching with policy guidance."""
    # - Click-drag to draw sketch input
    # - Policy generates guided trajectory samples
    # - Supports different alignment strategies

class MazeExp:
    """Experiment replay and benchmarking."""
    # - Loads saved sketches from JSON files
    # - Applies different alignment strategies
    # - Logs collision detection and metrics
```

**Visualization Features:**
- Collision detection using maze grid
- Trajectory scoring by L2 distance to sketch
- Rainbow color gradient for temporal progression
- White tinting for colliding trajectories

### 5. Dataset Handling (`itps/common/datasets/`)

Based on LeRobot dataset format:

```python
class LeRobotDataset(torch.utils.data.Dataset):
    """Loads data from HuggingFace Hub or locally."""
    # - Episode-based indexing
    # - Video (.mp4) and image (.png) support
    # - delta_timestamps for frame offset loading
    # - Stats: mean, std, min, max per feature
```

### 6. Normalization (`itps/common/policies/normalize.py`)

```python
class Normalize(nn.Module):
    """Normalize inputs using dataset statistics."""
    # Modes: "mean_std", "min_max"

class Unnormalize(nn.Module):
    """Reverse normalization for outputs."""
```

### 7. Rollout Wrapper (`itps/common/policies/rollout_wrapper.py`)

```python
class PolicyRolloutWrapper:
    """Manages observations/actions during environment rollout."""
    # - Observation cache (timestamp → observations)
    # - Action cache (timestamp → actions)
    # - Supports sync and async inference via ThreadPoolExecutor
```

---

## Data Flow

### Input Observations
```python
{
    "observation.state": torch.Tensor(B, n_obs_steps, 2),      # 2D agent position
    "observation.environment_state": torch.Tensor(B, 2),       # Environment context
    "observation.image": torch.Tensor(B, n_obs_steps, C, H, W) # Optional image
}
```

### Output Actions
```python
{
    "action": torch.Tensor(B, horizon, 2)  # 2D velocity commands
}
```

### User Guide (Sketch) Input
```python
guide: torch.Tensor(guide_horizon, 2)  # User-drawn path in maze coordinates
```

---

## Inference Pipeline (Diffusion Policy)

```python
# 1. Normalize observations
observation_batch = normalize_inputs(observation_batch)

# 2. Encode visual features (if using environment_state)
global_cond = _prepare_global_conditioning(batch)

# 3. Sample actions via diffusion
for t in noise_scheduler.timesteps:
    model_output = unet(sample, t, global_cond)

    # Add guidance gradient if user sketch provided
    if guide is not None and t > final_influence_step:
        grad = guide_gradient(sample, guide)
        model_output += guide_ratio * grad

    # Denoise step
    sample = noise_scheduler.step(model_output, t, sample)

# 4. Extract n_action_steps and unnormalize
actions = unnormalize_outputs(actions)
```

---

## Pre-trained Models

### Diffusion Policy (`weights_dp/pretrained_model/config.json`)
- Input: observation.state (2D), observation.environment_state (2D)
- Output: action (2D)
- Horizon: 64 timesteps
- Inference steps: 10 (DDIM)

### ACT Policy (`weights_act/pretrained_model/config.json`)
- Input: observation.state (2D), observation.environment_state (2D)
- Output: action (2D)
- Chunk size: 64
- Vision backbone: ResNet18
- VAE with latent_dim=32

---

## Dependencies

### Core Libraries
- `torch`, `torchvision`: Deep learning framework
- `diffusers`: Diffusion model schedulers (DDPM/DDIM)
- `hydra`, `omegaconf`: Configuration management
- `huggingface-hub`, `datasets`: Model/data hosting
- `pygame`: Interactive GUI
- `einops`: Tensor operations
- `wandb`: Experiment tracking

### Integration
- **LeRobot**: Dataset format and loading (`CODEBASE_VERSION = "v1.5"`)
- **D4RL Maze2D**: 2D navigation benchmark
- **Gymnasium**: Environment interface

