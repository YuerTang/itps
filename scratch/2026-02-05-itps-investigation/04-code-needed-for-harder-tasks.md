# Code Needed for Block Stacking / Kitchen Tasks

## Current Codebase Analysis

### What's GENERIC (Reusable)
| Component | File | Notes |
|-----------|------|-------|
| Diffusion Policy | `common/policies/diffusion/modeling_diffusion.py` | Works with any state/action dims |
| ACT Policy | `common/policies/act/modeling_act.py` | Works with any state/action dims |
| Alignment strategies | `modeling_diffusion.py:167-270` | SS, GD, BI, PR, OP all generic |
| Normalization | `common/policies/normalize.py` | Generic |
| Dataset loading | `common/datasets/lerobot_dataset.py` | LeRobot format |

### What's MAZE2D-SPECIFIC (Needs Replacement)
| Component | File | Lines | Why It's Maze-Specific |
|-----------|------|-------|------------------------|
| MazeEnv class | `interact_maze2d.py` | 60-210 | Hardcoded 2D maze grid, collision checking |
| GUI visualization | `interact_maze2d.py` | 151-188 | Pygame 2D rendering |
| Coordinate transforms | `interact_maze2d.py` | 134-143 | `xy2gui`, `gui2xy` for 2D maze |
| Guide gradient | `modeling_diffusion.py` | 254-271 | `assert naction.shape[2] == 2` (2D only!) |

---

## Code Needed for KITCHEN Task

### Difficulty: Medium
### Can use existing datasets: Yes (UCSD Kitchen, Open-X)

### 1. New Interactive Interface (~300 lines)

```python
# interact_kitchen.py

class KitchenEnv:
    """Kitchen manipulation environment with camera view."""

    def __init__(self, camera_intrinsics, camera_extrinsics):
        self.camera_K = camera_intrinsics  # 3x3 matrix
        self.camera_T = camera_extrinsics  # 4x4 transform

    def pixel_to_3d(self, pixel_xy, depth):
        """Convert clicked pixel + depth to 3D world coordinate."""
        # z_point input from paper Equation 1
        ...

    def render_camera_view(self, rgb_image, trajectories=None):
        """Display camera view with optional trajectory overlay."""
        ...

    def project_trajectory_to_image(self, trajectory_3d):
        """Project 3D end-effector trajectory to 2D image for visualization."""
        ...

class KitchenInteractive(KitchenEnv):
    """Point-and-click interface for kitchen task steering."""

    def on_click(self, pixel_x, pixel_y):
        """Handle user click to set z_point."""
        depth = self.get_depth_at_pixel(pixel_x, pixel_y)
        self.guide_point = self.pixel_to_3d((pixel_x, pixel_y), depth)

    def run_inference_with_steering(self, observation):
        """Run policy with point input steering."""
        # Use SS or GD with point objective (Equation 1)
        ...
```

### 2. Modify Guide Gradient for 3D (~20 lines)

```python
# In modeling_diffusion.py, modify guide_gradient()

def guide_gradient(self, naction, guide):
    # REMOVE: assert naction.shape[2] == 2
    # CHANGE TO:
    action_dim = naction.shape[2]  # Could be 2, 7, or any dim

    # Handle point input (guide is single point repeated)
    if guide.dim() == 1:  # Point input: (action_dim,)
        guide = guide.unsqueeze(0).expand(naction.shape[1], -1)

    # Rest stays the same...
```

### 3. Config for Kitchen Dataset (~50 lines)

```python
# configs/kitchen_dp.yaml

input_shapes:
  observation.state: [21]  # 21-DOF robot state (from UCSD kitchen)
  observation.images.image: [3, 480, 640]  # RGB camera

output_shapes:
  action: [8]  # 8-DOF action (from UCSD kitchen)

horizon: 32  # Shorter horizon for real-time control
n_obs_steps: 2
n_action_steps: 8
```

### 4. Training Script Modifications (~100 lines)

```python
# train_kitchen.py

from common.datasets.factory import make_dataset
from common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Load UCSD Kitchen dataset
dataset = make_dataset(
    repo_id="lerobot/ucsd_kitchen_dataset",
    ...
)

# Train diffusion policy
policy = DiffusionPolicy(config=kitchen_config)
# Standard training loop...
```

### 5. Real Robot Integration (if using real robot) (~200 lines)

```python
# robot_interface.py

class RobotInterface:
    """Interface to real robot for executing actions."""

    def __init__(self, robot_ip):
        # Connect to robot
        ...

    def get_observation(self):
        """Get current robot state + camera image."""
        return {
            "observation.state": self.get_joint_positions(),
            "observation.images.image": self.get_camera_image(),
        }

    def execute_action(self, action):
        """Send action to robot."""
        ...
```

### Total New Code for Kitchen: ~700 lines

---

## Code Needed for BLOCK STACKING Task

### Difficulty: Hard
### Requires: Isaac Sim, CuRobo, VR headset

### 1. Isaac Sim Environment (~500 lines)

```python
# isaac_block_stacking_env.py

from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
import curobo

class BlockStackingEnv:
    """Isaac Sim environment for block stacking."""

    def __init__(self):
        self.world = World()
        self.robot = self.load_robot("franka")
        self.blocks = [self.spawn_block(i) for i in range(4)]
        self.motion_planner = curobo.MotionPlanner(...)

    def get_observation(self):
        """Get robot state + block positions."""
        return {
            "observation.state": self.robot.get_joint_positions(),
            "observation.environment_state": self.get_block_positions(),
        }

    def step(self, action):
        """Execute action in simulation."""
        self.robot.set_joint_positions(action)
        self.world.step()

    def generate_demonstration(self, block_sequence):
        """Generate demo trajectory using CuRobo planner."""
        trajectory = []
        for block_id in block_sequence:
            # Plan pick
            pick_traj = self.motion_planner.plan(
                target=self.blocks[block_id].position
            )
            # Plan place
            place_traj = self.motion_planner.plan(
                target=self.get_stack_target()
            )
            trajectory.extend(pick_traj + place_traj)
        return trajectory
```

### 2. VR Interface for 3D Sketching (~400 lines)

```python
# vr_sketch_interface.py

from omni.isaac.kit import VRController

class VRSketchInterface:
    """VR interface for drawing 3D trajectory sketches."""

    def __init__(self, env):
        self.env = env
        self.vr = VRController()
        self.sketch_points = []

    def update(self):
        """Capture VR controller position when trigger pressed."""
        if self.vr.trigger_pressed():
            pos = self.vr.get_controller_position()
            self.sketch_points.append(pos)

    def get_sketch(self):
        """Return sketch as (T, 3) array."""
        return np.array(self.sketch_points)

    def visualize_sketch(self):
        """Render sketch in Isaac Sim."""
        for i, point in enumerate(self.sketch_points):
            self.env.draw_sphere(point, color=self.time_color(i))
```

### 3. Data Generation Pipeline (~300 lines)

```python
# generate_block_stacking_data.py

def generate_dataset(env, num_episodes=10000):
    """Generate training data using CuRobo planner."""
    dataset = []

    for ep in range(num_episodes):
        # Random block positions
        env.reset()

        # Random stacking sequence
        sequence = np.random.permutation([0, 1, 2, 3])

        # Generate trajectory
        traj = env.generate_demonstration(sequence)

        # Record observations and actions
        for t, action in enumerate(traj):
            obs = env.get_observation()
            dataset.append({
                "observation.state": obs["observation.state"],
                "action": action,
                "episode_index": ep,
                "frame_index": t,
            })
            env.step(action)

    return dataset
```

### 4. Modified Alignment for Discrete Tasks (~100 lines)

```python
# In modeling_diffusion.py

def guide_gradient_discrete(self, naction, guide, t, final_step=50):
    """
    For discrete task alignment, only steer in early diffusion steps.
    This recovers training distribution in later steps (paper Section III.B).
    """
    if t < final_step:
        # No guidance in later steps - recover training distribution
        return torch.zeros_like(naction)
    else:
        # Standard guidance in early steps
        return self.guide_gradient(naction, guide)
```

### 5. Experiment Runner (~200 lines)

```python
# run_block_stacking_exp.py

class BlockStackingExperiment:
    """Run block stacking experiments with VR steering."""

    def __init__(self, policy, env, vr_interface):
        self.policy = policy
        self.env = env
        self.vr = vr_interface

    def run_trial(self):
        """Run one interaction trial."""
        obs = self.env.get_observation()

        # Show unconditional rollouts
        unguided_trajs = self.policy.run_inference(obs, guide=None)
        self.visualize(unguided_trajs)

        # Get user sketch via VR
        sketch = self.vr.get_sketch()

        # Run guided inference
        guided_trajs = self.policy.run_inference(obs, guide=sketch)

        # Execute best trajectory
        best_traj = self.select_best(guided_trajs, sketch)
        success = self.env.execute(best_traj)

        return {
            "aligned": self.check_alignment(best_traj, sketch),
            "success": success,
        }
```

### Total New Code for Block Stacking: ~1500 lines

---

## Summary: What to Ask Your Mentor

### Option A: Kitchen Task (Recommended to Start)

**Ask**: "Can I use the UCSD Kitchen dataset on HuggingFace to extend ITPS to kitchen manipulation?"

**What you need**:
- Dataset: Already available at `lerobot/ucsd_kitchen_dataset`
- Code: ~700 lines of new code
- Hardware: None (can run on any GPU)
- Time estimate: 1-2 weeks

### Option B: Block Stacking

**Ask**: "Do you have access to an Isaac Sim setup? Or should I generate my own block stacking data?"

**What you need**:
- Dataset: Need to generate using Isaac Sim + CuRobo
- Code: ~1500 lines of new code
- Hardware: NVIDIA GPU with Isaac Sim license, optionally VR headset
- Time estimate: 3-4 weeks

### Specific Questions for Your Mentor

1. "Which task should I prioritize - Kitchen (easier) or Block Stacking (harder)?"

2. "For Kitchen: Can I use the public UCSD Kitchen dataset, or do you have a specific dataset in mind?"

3. "For Block Stacking: Do you have access to Isaac Sim and CuRobo, or should I set these up?"

4. "What's the main goal - replicate the paper's results, or extend to new scenarios?"

5. "Should I implement the VR interface for 3D sketches, or is point-click sufficient?"

