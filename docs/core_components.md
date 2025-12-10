# Core Components Reference

This document provides a comprehensive guide to the core files, classes, and functions in the Alpamayo-R1 codebase.

---

## Table of Contents

1. [Core Files Overview](#core-files-overview)
2. [Model Components](#model-components)
3. [Action Space](#action-space)
4. [Diffusion](#diffusion)
5. [Utilities](#utilities)
6. [Dataset Loader](#dataset-loader)

---

## Core Files Overview

### Directory Structure

```
src/alpamayo_r1/
├── config.py                    # Configuration dataclass
├── helper.py                    # Input preprocessing utilities
├── test_inference.py            # End-to-end inference test
├── load_physical_aiavdataset.py # Dataset loading
│
├── models/
│   ├── alpamayo_r1.py           # Main model implementation
│   ├── base_model.py            # Base VLA architecture
│   ├── action_in_proj.py        # Action projection module
│   ├── token_utils.py           # Token extraction utilities
│   └── delta_tokenizer.py       # Trajectory tokenizer
│
├── action_space/
│   ├── action_space.py          # Abstract action space
│   ├── unicycle_accel_curvature.py  # Unicycle kinematics
│   ├── discrete_action_space.py # Discrete tokenizer
│   └── utils.py                 # Action utilities
│
├── diffusion/
│   ├── base.py                  # Abstract diffusion class
│   └── flow_matching.py         # Flow matching implementation
│
└── geometry/
    └── rotation.py              # Rotation utilities
```

### Lines of Code by File

| File | Lines | Purpose |
|------|-------|---------|
| `alpamayo_r1.py` | 450 | Main model class, sampling methods |
| `base_model.py` | 380 | Base VLA architecture, token fusion |
| `unicycle_accel_curvature.py` | 350 | Unicycle action space implementation |
| `action_in_proj.py` | 240 | Fourier encoding + MLP projection |
| `load_physical_aiavdataset.py` | 420 | Dataset loading and preprocessing |
| `delta_tokenizer.py` | 180 | Delta trajectory tokenization |
| `flow_matching.py` | 150 | Flow matching diffusion |
| `rotation.py` | 247 | Rotation matrix utilities |
| `helper.py` | 100 | Message creation, processor init |
| `token_utils.py` | 120 | Token extraction functions |
| `discrete_action_space.py` | 210 | Discrete action tokenizer |
| `config.py` | 85 | Configuration parameters |

---

## Model Components

### 1. AlpamayoR1 (Main Model)

**File:** `src/alpamayo_r1/models/alpamayo_r1.py`

#### Class: `AlpamayoR1(ReasoningVLA)`

Main model class that orchestrates VLM reasoning and trajectory generation.

**Key Attributes:**
- `vlm`: Qwen3-VL vision-language model (8B parameters)
- `expert_model`: Separate transformer for trajectory refinement
- `action_head`: Linear layer mapping hidden states to action velocities
- `action_in_proj`: Projects actions to expert model input embeddings
- `action_space`: UnicycleActionSpace for trajectory unrolling
- `diffusion`: FlowMatchingDiffusion sampler

**Core Methods:**

##### `sample_trajectories_from_data_with_vlm_rollout()`
- **Location:** Line 198
- **Purpose:** End-to-end inference from images to trajectories
- **Inputs:**
  - `images`: Multi-camera frames `(B, N_cams, C, H, W)`
  - `traj_history`: Historical trajectory `(B, T_hist, 3)`
  - `traj_history_rot_mat`: Historical rotations `(B, T_hist, 3, 3)`
- **Outputs:**
  - `trajectories`: Predicted future paths `(B, N_samples, T_pred, 3)`
  - `rotations`: Predicted orientations `(B, N_samples, T_pred, 3, 3)`
  - `cot_reasoning`: Generated reasoning text
- **Process:**
  1. Tokenize trajectory history
  2. Run VLM generation for reasoning
  3. Cache VLM key-values
  4. Sample trajectories via diffusion

##### `_sample_with_expert_model()`
- **Location:** Line 291
- **Purpose:** Diffusion-based trajectory sampling
- **Inputs:**
  - `past_key_values`: Cached VLM context
  - `traj_future_start_token_id`: Token ID to start generation
  - `initial_state`: Current velocity for trajectory unrolling
- **Outputs:**
  - `trajectories`: Sampled trajectories `(B, N_samples, T, 3)`
  - `rotations`: Rotation matrices `(B, N_samples, T, 3, 3)`
- **Process:**
  1. Initialize random noise
  2. Run diffusion loop (10 steps)
  3. Convert actions to trajectories

##### `forward()`
- **Location:** Line 121
- **Purpose:** Training forward pass (not used in inference)
- **Note:** Computes loss for action prediction

---

### 2. ReasoningVLA (Base Model)

**File:** `src/alpamayo_r1/models/base_model.py`

#### Class: `ReasoningVLA(nn.Module)`

Base architecture for vision-language-action models.

**Core Methods:**

##### `_fuse_traj_history_tokens()`
- **Location:** Line 199
- **Purpose:** Replace placeholder tokens with trajectory tokens
- **Inputs:**
  - `input_ids`: Token IDs from processor `(B, seq_len)`
  - `traj_hist_token_ids`: Tokenized trajectory `(B, T*3)`
- **Outputs:**
  - Modified `input_ids` with embedded trajectory
- **Process:**
  1. Find positions of `<|traj_history|>` tokens
  2. Replace with actual trajectory tokens
  3. Validate token count matches

##### `_extract_special_token_ranges()`
- **Location:** Line 156
- **Purpose:** Find token ranges for trajectory/reasoning sections
- **Used for:** Extracting loss computation ranges during training

---

### 3. ActionInputProjection

**File:** `src/alpamayo_r1/models/action_in_proj.py`

#### Class: `ActionInputProjection(nn.Module)`

Projects (action, timestep) to expert model input embeddings.

**Architecture:**
```
Input: (action, timestep)
  ↓
Fourier Encoding (timestep) → [sin(2πkt), cos(2πkt)] for k=0..K
  ↓
Concatenate: [action, fourier_features]
  ↓
MLP (3 layers with SiLU activation)
  ↓
Output: embeddings (hidden_dim)
```

**Core Methods:**

##### `__call__(actions, timestep)`
- **Location:** Line 124
- **Purpose:** Project actions and time to embeddings
- **Inputs:**
  - `actions`: Shape `(B, T, action_dim)`
  - `timestep`: Scalar or tensor, diffusion timestep
- **Outputs:**
  - `embeddings`: Shape `(B, T, hidden_dim)`
- **Note:** Uses Fourier features for temporal encoding

##### `_fourier_encode(timestep)`
- **Location:** Line 85
- **Purpose:** Encode timestep with sinusoidal features
- **Formula:** `[sin(2πkt), cos(2πkt)]` for k = 0, 1, ..., K-1

---

### 4. DeltaTrajectoryTokenizer

**File:** `src/alpamayo_r1/models/delta_tokenizer.py`

#### Class: `DeltaTrajectoryTokenizer`

Converts continuous trajectories to discrete token sequences.

**Tokenization Strategy:**
- Uses delta encoding: `Δx_t = x_t - x_{t-1}`
- Quantizes deltas into bins
- Maps bins to token IDs
- Separate vocabularies for x, y, z dimensions

**Core Methods:**

##### `tokenize(trajectory)`
- **Location:** Line 81
- **Purpose:** Convert trajectory to token IDs
- **Inputs:**
  - `trajectory`: Shape `(B, T, 3)` - xyz positions
- **Outputs:**
  - `token_ids`: Shape `(B, T*3)` - flattened token sequence
- **Process:**
  1. Compute deltas: `traj[t] - traj[t-1]`
  2. Quantize: `bin_idx = (delta - min_val) / bin_size`
  3. Map: `token_id = bin_idx + vocab_offset`

##### `detokenize(token_ids)`
- **Location:** Line 123
- **Purpose:** Convert token IDs back to trajectory
- **Inputs:**
  - `token_ids`: Shape `(B, T*3)`
- **Outputs:**
  - `trajectory`: Shape `(B, T, 3)`
- **Note:** Reconstructs by cumulative sum of deltas

---

## Action Space

### UnicycleActionSpace

**File:** `src/alpamayo_r1/action_space/unicycle_accel_curvature.py`

#### Class: `UnicycleActionSpace(ActionSpace)`

Implements unicycle kinematic model with acceleration and curvature controls.

**Action Representation:**
- **Dimension 0:** Acceleration (m/s²)
- **Dimension 1:** Curvature (1/m, inverse turning radius)

**Core Methods:**

##### `unroll(actions, initial_state)`
- **Location:** Line 223
- **Purpose:** Convert action sequence to xyz trajectory
- **Inputs:**
  - `actions`: Shape `(B, T, 2)` - (acceleration, curvature)
  - `initial_state`: Dictionary with `velocity`, `position`, `heading`
- **Outputs:**
  - `trajectory`: Shape `(B, T, 3)` - xyz positions
  - `rotations`: Shape `(B, T, 3, 3)` - rotation matrices
- **Kinematic Equations:**
  ```python
  v[t+1] = v[t] + a[t] * dt
  ω[t] = v[t] * κ[t]  # angular velocity
  θ[t+1] = θ[t] + ω[t] * dt
  x[t+1] = x[t] + v[t] * cos(θ[t]) * dt
  y[t+1] = y[t] + v[t] * sin(θ[t]) * dt
  ```

##### `_solve_actions_from_traj()`
- **Location:** Line 305
- **Purpose:** Inverse kinematics - trajectory to actions
- **Used for:** Training data preparation
- **Process:**
  1. Compute velocities from positions
  2. Compute heading from velocity direction
  3. Extract acceleration and curvature

##### `normalize_actions()` / `denormalize_actions()`
- **Location:** Line 180, Line 195
- **Purpose:** Scale actions to [-1, 1] range for neural network
- **Normalization:**
  ```python
  accel_norm = (accel - accel_mean) / accel_std
  curv_norm = (curv - curv_mean) / curv_std
  ```

---

### DiscreteActionSpace

**File:** `src/alpamayo_r1/action_space/discrete_action_space.py`

#### Class: `DiscreteActionSpace(ActionSpace)`

Alternative action representation using discrete trajectory tokens.

**Usage:** Primarily for research/ablation studies. Main model uses continuous actions.

---

## Diffusion

### FlowMatchingDiffusion

**File:** `src/alpamayo_r1/diffusion/flow_matching.py`

#### Class: `FlowMatchingDiffusion(BaseDiffusion)`

Implements flow matching for trajectory generation.

**Core Methods:**

##### `sample(model_fn, shape, num_inference_steps)`
- **Location:** Line 87
- **Purpose:** Sample from diffusion model
- **Inputs:**
  - `model_fn`: Function that predicts velocity field
  - `shape`: Output shape `(B, T, D)`
  - `num_inference_steps`: Number of diffusion steps (default 10)
- **Outputs:**
  - `samples`: Generated actions
- **Algorithm:**
  ```python
  x_t ~ N(0, I)  # Initialize with noise
  for t in [1.0, 0.9, ..., 0.1, 0.0]:
      v_t = model_fn(x_t, t)  # Predict velocity
      x_t = x_t + v_t * dt    # Euler integration
  return x_t
  ```

##### `get_timesteps(num_steps)`
- **Location:** Line 62
- **Purpose:** Create diffusion timestep schedule
- **Returns:** Linear schedule from 1000 to 0

---

## Utilities

### 1. Token Utilities

**File:** `src/alpamayo_r1/models/token_utils.py`

#### Key Functions:

##### `extract_token_ids_between_markers()`
- **Location:** Line 23
- **Purpose:** Extract tokens between start/end markers
- **Example:** Extract trajectory tokens between `<|traj_future_start|>` and `<|traj_future_end|>`

##### `extract_all_token_ranges()`
- **Location:** Line 78
- **Purpose:** Find all occurrences of token ranges in sequence
- **Used for:** Batch processing during training

---

### 2. Helper Functions

**File:** `src/alpamayo_r1/helper.py`

#### Key Functions:

##### `create_message(frames)`
- **Location:** Line 35
- **Purpose:** Create chat message format for VLM
- **Returns:** List of message dictionaries with images and placeholders

##### `get_processor(tokenizer)`
- **Location:** Line 94
- **Purpose:** Initialize Qwen3-VL processor with custom tokenizer
- **Configuration:**
  - `min_pixels = 163840`
  - `max_pixels = 196608`

##### `to_device(data, device, dtype)`
- **Location:** Line 119
- **Purpose:** Recursively transfer nested data structures to device
- **Handles:** Tensors, dicts, lists, tuples

---

### 3. Rotation Utilities

**File:** `src/alpamayo_r1/geometry/rotation.py`

#### Key Functions:

##### `so3_to_yaw_torch(rot_mat)`
- **Location:** Line 25
- **Purpose:** Extract yaw angle from 3D rotation matrix
- **Formula:** `yaw = atan2(R[1,0], R[0,0])`

##### `euler_2_so3(euler_angles, degrees, seq)`
- **Location:** Line 56
- **Purpose:** Convert Euler angles to SO(3) rotation matrix
- **Uses:** `scipy.spatial.transform.Rotation`

##### `rotation_matrix_torch(angle)`
- **Location:** Line 109
- **Purpose:** Create 2D rotation matrix from angle
- **Returns:** Matrix `[[cos θ, -sin θ], [sin θ, cos θ]]`

##### `stable_gramschmidt(M)`
- **Location:** Line 156
- **Purpose:** Orthonormalize 3D vectors robustly
- **Used for:** Constructing valid rotation matrices from predicted vectors

##### `rot_3d_to_2d(rot)` / `rot_2d_to_3d(rot)`
- **Location:** Line 187, Line 207
- **Purpose:** Convert between 2D and 3D rotation representations

##### `round_2pi_torch(x)`
- **Location:** Line 237
- **Purpose:** Normalize angles to [-π, π]
- **Formula:** `atan2(sin(x), cos(x))`

---

## Dataset Loader

### PhysicalAI Dataset Loader

**File:** `src/alpamayo_r1/load_physical_aiavdataset.py`

#### Key Functions:

##### `prepare_data_from_token(token, cam_list)`
- **Location:** Line 123
- **Purpose:** Load multi-camera images and trajectory from dataset
- **Inputs:**
  - `token`: Sample identifier
  - `cam_list`: List of camera names
- **Outputs:**
  - `images`: Tensor `(N_cams, C, H, W)`
  - `ego_hist_traj`: Historical trajectory `(T_hist, 3)`
  - `ego_gt_traj`: Ground truth future trajectory `(T_pred, 3)`
  - `ego_hist_rot_mat`: Historical rotations `(T_hist, 3, 3)`
- **Process:**
  1. Load images from multiple cameras
  2. Extract ego vehicle trajectory from annotations
  3. Transform to local (ego-centric) coordinate frame
  4. Normalize and preprocess

##### `global_to_local_frame(traj_global, current_pose)`
- **Location:** Line 85
- **Purpose:** Transform trajectory from global to local coordinates
- **Transformation:**
  ```python
  traj_local = R_ego^T @ (traj_global - pos_ego)
  ```

---

## Configuration

### AlpamayoR1Config

**File:** `src/alpamayo_r1/config.py`

#### Class: `AlpamayoR1Config(ReasoningVLAConfig)`

Configuration dataclass for model hyperparameters.

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vlm_model_name` | `"Qwen/Qwen3-VL-8B-Instruct"` | Vision-language model |
| `expert_model_name` | `"Qwen/Qwen2.5-0.5B-Instruct"` | Expert trajectory model |
| `action_dim` | `2` | Action space dimension |
| `pred_horizon` | `12` | Prediction horizon (timesteps) |
| `num_inference_steps` | `10` | Diffusion sampling steps |
| `num_samples` | `32` | Number of trajectory samples |
| `dt` | `0.5` | Timestep duration (seconds) |
| `hidden_dim` | `896` | Expert model hidden size |
| `action_head_hidden_sizes` | `[512, 256]` | MLP layer sizes |
| `fourier_encode_dim` | `128` | Fourier encoding dimension |

---

## Testing

### Inference Test Script

**File:** `src/alpamayo_r1/test_inference.py`

#### Main Function: `test_inference()`

End-to-end inference test that:
1. Loads a sample from PhysicalAI dataset
2. Runs model inference
3. Computes minADE (minimum Average Displacement Error)
4. Validates output shapes

**Usage:**
```bash
python -m alpamayo_r1.test_inference
```

**Expected Output:**
- Predicted trajectories: `(1, 32, 12, 3)`
- Rotations: `(1, 32, 12, 3, 3)`
- minADE: < 5.0 meters (typical range: 1.5-3.0)

---

## Module Dependencies

```
alpamayo_r1.py
  ├── base_model.py
  │   ├── delta_tokenizer.py
  │   └── token_utils.py
  ├── action_in_proj.py
  ├── diffusion/flow_matching.py
  ├── action_space/unicycle_accel_curvature.py
  │   ├── geometry/rotation.py
  │   └── action_space/utils.py
  └── helper.py

load_physical_aiavdataset.py
  └── geometry/rotation.py

test_inference.py
  ├── alpamayo_r1.py
  ├── load_physical_aiavdataset.py
  └── helper.py
```

---

## External Dependencies

- **transformers**: Qwen3-VL, Qwen2.5 models
- **torch**: Neural network framework
- **scipy**: Rotation conversions
- **numpy**: Numerical operations
- **Pillow**: Image loading

---

## Key Insights

1. **Two-Stage Architecture:** VLM generates reasoning, expert model generates trajectories
2. **Action Space Design:** Unicycle model provides physically plausible trajectories
3. **Delta Encoding:** Tokenizes trajectories efficiently for discrete representation
4. **Flow Matching:** Uses only 10 steps for fast inference (~0.5s per sample)
5. **Multi-Sample Generation:** Produces 32 diverse trajectories to capture uncertainty
6. **KV Cache Reuse:** VLM context computed once and shared across all samples

---

## Performance Characteristics

| Operation | Time (ms) | Device |
|-----------|-----------|--------|
| VLM Generation (reasoning) | ~800 | A100 GPU |
| Diffusion Sampling (32 samples) | ~450 | A100 GPU |
| Trajectory Unrolling | ~20 | CPU/GPU |
| **Total Inference** | **~1300** | **A100 GPU** |

---

## Common Usage Patterns

### 1. Load Model
```python
from alpamayo_r1 import AlpamayoR1, AlpamayoR1Config

config = AlpamayoR1Config()
model = AlpamayoR1.from_pretrained("NVIDIA/Alpamayo-R1-8B", config=config)
model = model.to("cuda")
```

### 2. Prepare Input
```python
from alpamayo_r1.helper import create_message, get_processor

messages = create_message(images)
processor = get_processor(model.tokenizer)
inputs = processor(text=messages, images=images, return_tensors="pt")
```

### 3. Run Inference
```python
trajectories, rotations = model.sample_trajectories_from_data_with_vlm_rollout(
    images=images,
    traj_history=traj_hist,
    traj_history_rot_mat=traj_hist_rot,
)
```

### 4. Compute Metrics
```python
from alpamayo_r1.action_space.utils import compute_ADE

min_ade = compute_ADE(trajectories, ground_truth).min(dim=1)  # Best of 32 samples
```

---

This reference guide covers all core components of Alpamayo-R1. For detailed inference flow, see [model_inference_flow.md](model_inference_flow.md).
