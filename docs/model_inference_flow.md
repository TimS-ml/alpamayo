# Model Inference Flow

This document describes the complete function call sequence during Alpamayo-R1 model inference, from data loading to trajectory prediction.

## Overview

Alpamayo-R1 inference combines a Vision-Language Model (VLM) for reasoning with a diffusion-based expert model for trajectory generation. The process involves:

1. **Data Loading & Preprocessing**
2. **VLM Generation** (Chain-of-Causation reasoning)
3. **Diffusion Sampling** (Trajectory generation)
4. **Action Space Conversion** (Control to trajectory)

---

## Detailed Inference Pipeline

### 1. Data Loading & Preprocessing

**Entry Point:** `load_physical_aiavdataset.py`

#### Key Functions:
- **`load_physical_aiavdataset.prepare_data_from_token()`**
  - **Location:** `src/alpamayo_r1/load_physical_aiavdataset.py:123`
  - **Purpose:** Load multi-camera images and ego trajectory history from dataset
  - **Inputs:**
    - `token`: Dataset sample identifier
    - `cam_list`: List of camera names (e.g., `["CAM_FRONT", "CAM_FRONT_LEFT", ...]`)
  - **Outputs:**
    - `images`: Tensor of shape `(N_cameras, C, H, W)`
    - `ego_hist_traj`: Historical trajectory in local frame, shape `(T_hist, 3)`
    - `ego_gt_traj`: Ground truth future trajectory (for evaluation)
    - `ego_hist_rot_mat`: Historical rotation matrices, shape `(T_hist, 3, 3)`

#### Coordinate Transformation:
```python
# Transform from global to local (ego-centric) frame
ego_hist_traj = global_to_local_frame(traj_global, current_pose)
```

---

### 2. Input Preparation

**Module:** `helper.py`

#### Key Functions:

##### **`create_message(frames)`**
- **Location:** `src/alpamayo_r1/helper.py:35`
- **Purpose:** Create chat message format with images and trajectory placeholders
- **Inputs:**
  - `frames`: Image tensor `(N, C, H, W)`
- **Outputs:**
  - Chat messages with:
    - System prompt
    - User message (images + trajectory placeholder tokens)
    - Assistant prompt starting with `<|cot_start|>`
- **Special Tokens:**
  - `<|traj_history_start|>`, `<|traj_history|>` (×48), `<|traj_history_end|>`

##### **`get_processor(tokenizer)`**
- **Location:** `src/alpamayo_r1/helper.py:94`
- **Purpose:** Initialize VLM processor with custom tokenizer
- **Inputs:**
  - `tokenizer`: Custom tokenizer with special tokens
- **Outputs:**
  - `AutoProcessor` configured for Qwen3-VL-2B

#### Tokenization:
```python
processor = get_processor(tokenizer)
messages = create_message(frames)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = processor(text=text, images=frames, return_tensors="pt")
```

---

### 3. Trajectory Tokenization

**Module:** `base_model.py`, `delta_tokenizer.py`

#### Key Functions:

##### **`DeltaTrajectoryTokenizer.tokenize()`**
- **Location:** `src/alpamayo_r1/models/delta_tokenizer.py:81`
- **Purpose:** Convert trajectory to token IDs using delta encoding
- **Process:**
  1. Compute deltas: `Δx_t = x_t - x_{t-1}`
  2. Quantize deltas to discrete bins
  3. Map bins to token IDs
- **Inputs:**
  - `trajectory`: Shape `(B, T, 3)` (xyz positions)
- **Outputs:**
  - `token_ids`: Shape `(B, T*3)` (flattened xyz tokens)

##### **`ReasoningVLA._fuse_traj_history_tokens()`**
- **Location:** `src/alpamayo_r1/models/base_model.py:199`
- **Purpose:** Replace placeholder tokens with trajectory tokens
- **Process:**
  1. Extract positions of `<|traj_history|>` tokens
  2. Replace with actual trajectory token IDs
- **Inputs:**
  - `input_ids`: Token IDs from processor
  - `traj_hist_token_ids`: Tokenized trajectory
- **Outputs:**
  - Modified `input_ids` with embedded trajectory

---

### 4. VLM Generation (Chain-of-Causation)

**Module:** `alpamayo_r1.py`

#### Main Function:

##### **`AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout()`**
- **Location:** `src/alpamayo_r1/models/alpamayo_r1.py:198`
- **Purpose:** Generate reasoning text and cache VLM context

#### Detailed Steps:

**Step 4.1: Prepare VLM Inputs**
```python
# Location: alpamayo_r1.py:231
input_ids = self._fuse_traj_history_tokens(inputs['input_ids'], traj_hist_token_ids)
pixel_values = inputs['pixel_values']
image_grid_thw = inputs['image_grid_thw']
```

**Step 4.2: Autoregressive VLM Generation**
```python
# Location: alpamayo_r1.py:240
outputs = self.vlm.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    max_new_tokens=512,
    eos_token_id=traj_future_start_token_id,  # Stop at trajectory generation
    return_dict_in_generate=True,
    output_hidden_states=True,
    use_cache=True,
)
```

**Step 4.3: Extract KV Cache**
```python
# Location: alpamayo_r1.py:252
past_key_values = outputs.past_key_values  # Cached VLM context
generated_text = tokenizer.decode(outputs.sequences[0])  # CoC reasoning
```

**Example Generated Text:**
```
<|cot_start|>I observe the vehicle is approaching an intersection with a red light.
The pedestrian is crossing from the right. I need to decelerate and stop before
the crosswalk to ensure safety.<|cot_end|><|traj_future_start|>
```

---

### 5. Diffusion Sampling (Trajectory Generation)

**Module:** `alpamayo_r1.py`, `flow_matching.py`

#### Main Function:

##### **`AlpamayoR1._sample_with_expert_model()`**
- **Location:** `src/alpamayo_r1/models/alpamayo_r1.py:291`
- **Purpose:** Sample trajectories using flow matching diffusion

#### Detailed Steps:

**Step 5.1: Initialize Diffusion**
```python
# Location: alpamayo_r1.py:308
num_samples = 32  # Number of trajectory samples
action_dim = 2  # (acceleration, curvature)
pred_horizon = 12  # Future timesteps

# Random noise initialization
actions = torch.randn(batch_size, num_samples, pred_horizon, action_dim)
```

**Step 5.2: Diffusion Loop**
```python
# Location: alpamayo_r1.py:320
for step_idx in range(num_inference_steps):  # Default: 10 steps
    # (a) Compute timestep
    t = timesteps[step_idx] / 1000.0  # Normalize to [0, 1]

    # (b) Project action to token embeddings
    action_embeds = self.action_in_proj(actions, t)  # Shape: (B*N, T, D)

    # (c) Run expert model with cached VLM context
    expert_outputs = self.expert_model(
        inputs_embeds=action_embeds,
        past_key_values=past_key_values,  # From VLM
        use_cache=True,
    )

    # (d) Predict velocity field
    expert_last_hidden_state = expert_outputs.last_hidden_state
    predicted_velocity = self.action_head(expert_last_hidden_state)  # (B*N, T, 2)

    # (e) Update actions via Euler integration
    dt = timesteps[step_idx] - timesteps[step_idx + 1]
    actions = actions + predicted_velocity * dt
```

**Step 5.3: Action Projection Details**

##### **`ActionInputProjection.__call__()`**
- **Location:** `src/alpamayo_r1/models/action_in_proj.py:124`
- **Process:**
  1. Apply Fourier feature encoding to timestep `t`
  2. Concatenate action and Fourier features: `[action, fourier(t)]`
  3. Pass through MLP to get embeddings of size `hidden_dim`

```python
# Fourier encoding
time_embeds = self.fourier_encode(t)  # Shape: (B*N, fourier_dim)

# Concatenate with action
x = torch.cat([actions, time_embeds.unsqueeze(1).expand(-1, T, -1)], dim=-1)

# MLP projection
action_embeds = self.mlp(x)  # Shape: (B*N, T, hidden_dim)
```

---

### 6. Action Space Conversion

**Module:** `unicycle_accel_curvature.py`

#### Key Function:

##### **`UnicycleActionSpace.unroll()`**
- **Location:** `src/alpamayo_r1/action_space/unicycle_accel_curvature.py:223`
- **Purpose:** Convert (acceleration, curvature) controls to xyz trajectory
- **Inputs:**
  - `actions`: Shape `(B, N_samples, T, 2)` - (acceleration, curvature)
  - `initial_state`: Current velocity and position
- **Outputs:**
  - `trajectory`: Shape `(B, N_samples, T, 3)` - xyz positions
  - `rotation_matrices`: Shape `(B, N_samples, T, 3, 3)` - orientations

#### Unicycle Kinematics:
```python
# At each timestep t:
for t in range(pred_horizon):
    # Extract controls
    acceleration = actions[..., t, 0]
    curvature = actions[..., t, 1]

    # Update velocity
    velocity = velocity + acceleration * dt

    # Update heading
    angular_velocity = velocity * curvature
    heading = heading + angular_velocity * dt

    # Update position
    dx = velocity * cos(heading) * dt
    dy = velocity * sin(heading) * dt
    position = position + [dx, dy, 0]

    # Store trajectory
    trajectory[..., t, :] = position
```

---

## Complete Call Graph

```
1. load_physical_aiavdataset.prepare_data_from_token()
   └─> Returns: images, ego_hist_traj, ego_gt_traj

2. helper.create_message(images)
   └─> Returns: chat_messages

3. helper.get_processor(tokenizer)
   └─> processor.apply_chat_template(messages)
   └─> processor(text, images)
   └─> Returns: input_ids, pixel_values, image_grid_thw

4. delta_tokenizer.DeltaTrajectoryTokenizer.tokenize(ego_hist_traj)
   └─> Returns: traj_hist_token_ids

5. base_model.ReasoningVLA._fuse_traj_history_tokens()
   └─> Returns: input_ids (with embedded trajectory)

6. alpamayo_r1.AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout()
   │
   ├─> 6.1: vlm.generate()  # VLM autoregressive generation
   │   └─> Returns: generated_sequences, past_key_values
   │
   └─> 6.2: _sample_with_expert_model()  # Diffusion sampling
       │
       ├─> 6.2.1: Initialize random noise
       │
       ├─> 6.2.2: Diffusion loop (10 steps):
       │   │
       │   ├─> action_in_proj.ActionInputProjection()
       │   │   ├─> fourier_encode(timestep)
       │   │   └─> mlp([actions, fourier_features])
       │   │
       │   ├─> expert_model.forward(action_embeds, past_key_values)
       │   │
       │   ├─> action_head(expert_last_hidden_state)
       │   │   └─> Returns: predicted_velocity
       │   │
       │   └─> Euler step: actions += velocity * dt
       │
       └─> 6.2.3: unicycle_accel_curvature.UnicycleActionSpace.unroll()
           └─> Returns: trajectories (xyz), rotations

7. Returns: trajectories, rotations, reasoning_text
```

---

## Function Reference Table

| Function | Location | Purpose | Input → Output |
|----------|----------|---------|----------------|
| `prepare_data_from_token()` | `load_physical_aiavdataset.py:123` | Load dataset sample | `token` → `images, traj_hist, traj_gt` |
| `create_message()` | `helper.py:35` | Create chat format | `frames` → `chat_messages` |
| `get_processor()` | `helper.py:94` | Init VLM processor | `tokenizer` → `processor` |
| `DeltaTrajectoryTokenizer.tokenize()` | `delta_tokenizer.py:81` | Trajectory → tokens | `traj (B,T,3)` → `token_ids (B,T*3)` |
| `_fuse_traj_history_tokens()` | `base_model.py:199` | Embed traj in input | `input_ids, traj_tokens` → `input_ids` |
| `vlm.generate()` | Transformers library | VLM reasoning | `input_ids, images` → `text, kv_cache` |
| `_sample_with_expert_model()` | `alpamayo_r1.py:291` | Diffusion sampling | `kv_cache` → `actions (B,N,T,2)` |
| `ActionInputProjection()` | `action_in_proj.py:124` | Action → embeddings | `actions, t` → `embeds` |
| `expert_model.forward()` | Transformers library | Predict velocity | `embeds, kv_cache` → `hidden_states` |
| `action_head()` | `alpamayo_r1.py:92` | Hidden → velocity | `hidden (B,N,T,D)` → `velocity (B,N,T,2)` |
| `UnicycleActionSpace.unroll()` | `unicycle_accel_curvature.py:223` | Actions → trajectory | `actions (B,N,T,2)` → `traj (B,N,T,3)` |

---

## Execution Timeline

```
Time Step │ Component        │ Operation
──────────┼──────────────────┼─────────────────────────────────────
t=0       │ DataLoader       │ Load images + trajectory history
t=1       │ Preprocessor     │ Create chat messages, tokenize
t=2       │ Tokenizer        │ Tokenize trajectory history
t=3       │ VLM             │ Generate reasoning (autoregressive)
          │                 │ "I see red light → must decelerate"
t=4       │ VLM             │ Cache key-value states
t=5       │ Diffusion       │ Initialize noise (32 samples)
t=6..15   │ Expert Model    │ 10 diffusion steps:
          │                 │   - Project action + time
          │                 │   - Predict velocity
          │                 │   - Update action
t=16      │ Action Space    │ Convert actions → xyz trajectory
t=17      │ Output          │ Return: traj, rotations, reasoning
```

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_inference_steps` | 10 | Number of diffusion sampling steps |
| `num_samples` | 32 | Number of trajectory samples to generate |
| `pred_horizon` | 12 | Future trajectory length (timesteps) |
| `action_dim` | 2 | Action space dimension (accel, curvature) |
| `dt` | 0.5 | Timestep duration (seconds) |
| `min_pixels` | 163840 | VLM min image resolution |
| `max_pixels` | 196608 | VLM max image resolution |

---

## Notes

- **KV Cache Reuse:** The VLM's key-value cache is computed once and reused across all diffusion steps, significantly reducing computation.
- **Batch Processing:** All 32 trajectory samples are generated in parallel within the diffusion loop.
- **Action Space:** The model predicts acceleration and curvature (unicycle model), which provides smooth, physically plausible trajectories.
- **Coordinate Frame:** All trajectories are in the ego-centric (local) coordinate frame, centered at the current vehicle position.
