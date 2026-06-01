# Core Components Reference

A guide to the core files, classes, and functions in the Alpamayo-R1 codebase.
All names, signatures, and line numbers below were verified against the source
(post-`upstream/main` merge). Line numbers are approximate and may drift as the
code evolves.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Models](#models)
3. [Action Input Projection](#action-input-projection)
4. [Trajectory Tokenizers](#trajectory-tokenizers)
5. [Action Space](#action-space)
6. [Diffusion](#diffusion)
7. [Geometry](#geometry)
8. [Data Loading](#data-loading)
9. [Helpers & Logging](#helpers--logging)
10. [Configuration](#configuration)
11. [Test / Entry Point](#test--entry-point)
12. [Module Dependencies](#module-dependencies)

---

## Directory Structure

```
src/alpamayo_r1/
├── config.py                          # AlpamayoR1Config (HF PretrainedConfig)
├── helper.py                          # create_message / get_processor / to_device
├── test_inference.py                  # end-to-end inference script
├── load_physical_aiavdataset.py       # PhysicalAI-AV dataset loader
│
├── models/
│   ├── alpamayo_r1.py                 # AlpamayoR1 main model + inference
│   ├── base_model.py                  # ReasoningVLA base, config, token fusion
│   ├── action_in_proj.py             # Fourier + MLP action→embedding projection
│   ├── delta_tokenizer.py             # DeltaTrajectoryTokenizer (delta encoding)
│   └── token_utils.py                # token extraction / stopping criteria
│
├── action_space/
│   ├── action_space.py                # ActionSpace abstract base
│   ├── unicycle_accel_curvature.py    # UnicycleAccelCurvatureActionSpace
│   ├── utils.py                       # least-squares kinematic solvers
│   └── discrete_action_space.py       # DiscreteTrajectoryTokenizer (quantizer)
│
├── diffusion/
│   ├── base.py                        # BaseDiffusion, StepFn protocol
│   └── flow_matching.py               # FlowMatching (rectified-flow sampler)
│
├── geometry/
│   ├── rotation.py                    # rotation / yaw / Gram-Schmidt utilities
│   └── coordinates.py                 # 3D bounding-box corner helper
│
└── common/
    └── logging.py                     # colored, rank-aware logging
```

### Lines of Code by File (verified)

| File | Lines | Purpose |
|------|-------|---------|
| `action_space/utils.py` | 514 | Tikhonov-smoothed least-squares kinematic solvers |
| `models/base_model.py` | 445 | `ReasoningVLA`, config, trajectory-token fusion |
| `action_space/unicycle_accel_curvature.py` | 382 | Unicycle (accel, curvature) action space |
| `models/alpamayo_r1.py` | 334 | Main model + end-to-end inference |
| `geometry/rotation.py` | 266 | Rotation / yaw / angle utilities |
| `models/token_utils.py` | 253 | Token extraction, `StopAfterEOS` |
| `load_physical_aiavdataset.py` | 222 | Dataset sample loader |
| `models/delta_tokenizer.py` | 216 | Delta trajectory tokenizer |
| `diffusion/flow_matching.py` | 173 | Flow matching (Euler) sampler |
| `models/action_in_proj.py` | 169 | Fourier + MLP action projection |
| `helper.py` | 158 | Message / processor / device helpers |
| `common/logging.py` | 123 | Rank-aware colored logging |
| `action_space/discrete_action_space.py` | 108 | Discrete action quantizer |
| `action_space/action_space.py` | 94 | Abstract `ActionSpace` |
| `diffusion/base.py` | 88 | Abstract `BaseDiffusion`, `StepFn` |
| `test_inference.py` | 77 | Inference test script |
| `geometry/coordinates.py` | 66 | 3D bbox corners |
| `config.py` | 50 | `AlpamayoR1Config` |

---

## Models

### AlpamayoR1 (main model)

**File:** `src/alpamayo_r1/models/alpamayo_r1.py`

#### `class AlpamayoR1(ReasoningVLA)` — line 75

The full Vision-Language-Action model. Built on top of `ReasoningVLA` (which owns
the VLM backbone and tokenizers); `AlpamayoR1` adds the expert transformer, the
action projections, the action space, and the diffusion sampler.

Key sub-modules created in `__init__` (line 81):

| Attribute | Built from | Role |
|-----------|-----------|------|
| `self.vlm` | `Qwen3VLForConditionalGeneration` (in base) | Vision-language reasoning (Qwen3-VL-8B by default) |
| `self.expert` | `AutoModel.from_config(deepcopy(vlm.text_config) + expert_cfg)` | Lightweight text transformer that denoises actions; `embed_tokens` is deleted (it consumes `inputs_embeds`) |
| `self.action_space` | `hydra.instantiate(config.action_space_cfg)` | e.g. `UnicycleAccelCurvatureActionSpace` |
| `self.diffusion` | `hydra.instantiate(config.diffusion_cfg, x_dims=...)` | e.g. `FlowMatching`; `x_dims = (n_waypoints, 2)` |
| `self.action_in_proj` | `config.action_in_proj_cfg` | Projects `(action, t)` → expert input embeddings |
| `self.action_out_proj` | `config.action_out_proj_cfg` | Linear head: expert hidden → velocity field (the real "action head") |

> The expert is **not** a hardcoded "Qwen2.5-0.5B"; it is a re-configured copy of
> the VLM's own text config, overridden by `config.expert_cfg`.

#### `sample_trajectories_from_data_with_vlm_rollout(...)` — line 124

End-to-end inference entry point.

```python
def sample_trajectories_from_data_with_vlm_rollout(
    self,
    data: dict[str, Any],
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    num_traj_samples: int = 6,
    num_traj_sets: int = 1,
    diffusion_kwargs: dict[str, Any] | None = None,
    *args, **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
```

Returns `(pred_xyz, pred_rot)`, or `(pred_xyz, pred_rot, extra)` when
`kwargs["return_extra"]` is set (where `extra` holds decoded `cot` / `meta_action`
/ `answer` text). Pipeline:

1. Fuse history-trajectory tokens into `input_ids` via `fuse_traj_tokens` (line 164).
2. Run `self.vlm.generate(...)` autoregressively (line 194), stopping at the
   `<|traj_future_start|>` token (`StopAfterEOS`), masking discrete-trajectory
   logits with `ExpertLogitsProcessor`.
3. Save the VLM KV cache: `prompt_cache = vlm_outputs.past_key_values` (line 209).
4. Build the expert's `position_ids` (3-axis mRoPE) and `attention_mask`.
5. Define the denoising closure `step_fn(x, t)` (line 257) and run
   `self.diffusion.sample(batch_size, step_fn, ...)` (line 293).
6. Convert sampled actions to trajectories with
   `self.action_space.action_to_traj(...)` (line 305).
7. Rearrange to `(B, num_traj_sets, num_traj_samples, T, 3)` and return.

> There is **no** `_sample_with_expert_model()`, `action_head()`, or `forward()`
> in this file; the diffusion loop lives in the `step_fn` closure and the
> velocity head is `action_out_proj`.

#### `step_fn(x, t)` — line 257 (inner closure)

The per-step denoiser used by the diffusion sampler:

```python
future_token_embeds = self.action_in_proj(x, t)         # (b*, n_waypoints, hidden)
expert_out = self.expert(inputs_embeds=future_token_embeds,
                         position_ids=position_ids,
                         past_key_values=prompt_cache,    # reuse VLM context
                         attention_mask=attention_mask, use_cache=True)
prompt_cache.crop(prefill_seq_len)                       # drop expert's appended KV
last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
pred = self.action_out_proj(last_hidden).view(-1, n_waypoints, 2)  # velocity field
```

#### `class ExpertLogitsProcessor(LogitsProcessor)` — line 41

`__call__` (line 55) sets the logits of the contiguous discrete-trajectory token
block `[traj_token_offset, traj_token_offset + traj_vocab_size)` to `-inf`, so the
VLM never emits discrete `<iN>` tokens during chain-of-causation generation.

Module bottom (lines 333-334) registers the model with HF Auto classes:
```python
AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
```

---

### ReasoningVLA (base model)

**File:** `src/alpamayo_r1/models/base_model.py`

#### `class ReasoningVLA(PreTrainedModel, TrajectoryFusionMixin)` — line 285

Owns the VLM backbone and tokenizers. Notable methods:

- `_initialize_qwenvl3_vlm(config)` — line 367: builds
  `Qwen3VLForConditionalGeneration` from `config.vlm_name_or_path`
  (default `"Qwen/Qwen3-VL-8B-Instruct"`) and resizes the vocab to include the
  768 discrete trajectory tokens plus special tokens.
- `from_pretrained_submodules(config)` — line 403: loads real pretrained VLM
  weights and resizes embeddings.
- `get_input_embeddings` / `get_output_embeddings` / `tie_weights` — delegate to
  the VLM.

> This class has **no** training `forward()`; the two model files contain only
> construction + inference plumbing.

#### `class TrajectoryFusionMixin` — line 125

- `fuse_traj_tokens(input_ids, traj_data)` — line 168: tokenizes the ego history
  with `tokenize_history_trajectory` (line 91) and replaces the
  `<|traj_history|>` placeholder ids in the prompt via `replace_pad_token`
  (line 85). This is the real fusion method (not `_fuse_traj_history_tokens`).

#### `class ReasoningVLAConfig(PretrainedConfig)` — line 200

`model_type = "alpamayo_reasoning_vla"`. Real fields and defaults:

| Field | Default |
|-------|---------|
| `vlm_name_or_path` | `"Qwen/Qwen3-VL-8B-Instruct"` |
| `vlm_backend` | `"qwenvl3"` |
| `traj_vocab_size` | `768` |
| `tokens_per_history_traj` | `16` |
| `tokens_per_future_traj` | `64` |
| `model_dtype` | `"bfloat16"` |
| `attn_implementation` | `"flash_attention_2"` |
| `min_pixels` / `max_pixels` | `None` |
| `add_special_tokens` | `False` |

`_build_processor` (line 251) adds 768 discrete tokens `<i0>..<i767>` and records
`traj_token_start_idx` / `traj_token_ids`.

---

## Action Input Projection

**File:** `src/alpamayo_r1/models/action_in_proj.py`

Projects a noisy action tensor and a diffusion timestep into expert input
embeddings. There is **no** `ActionInputProjection`/`__call__`/`_fourier_encode`;
the real classes are below.

| Class | Line | Role |
|-------|------|------|
| `RMSNorm` | 22 | RMS normalization used inside the MLP |
| `MLPEncoder` | 38 | `Linear→SiLU` then `num_enc_layers` of `RMSNorm→Linear(→SiLU)`; default `hidden_size=1024`, `num_enc_layers=4` |
| `FourierEncoderV2` | 73 | Log-spaced Fourier features |
| `PerWaypointActionInProjV2` | 104 | The action projection module |

#### `FourierEncoderV2.forward(x)` — line 91

Uses **log-spaced** frequencies `freqs = logspace(0, log10(max_freq), dim//2)`
(default `max_freq=100`), then:

```python
arg = x[..., None] * self.freqs * 2 * torch.pi      # line 100
return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)
```

#### `PerWaypointActionInProjV2.forward(x, timesteps)` — line 148

Inputs `x: (B, n_waypoints, action_dim)`, `timesteps: (B, ...)`; output
`(B, n_waypoints, out_dim)`. Each action component is Fourier-encoded by its own
`FourierEncoderV2` (`num_fourier_feats=20`), the timestep by a separate encoder;
features are concatenated and passed through `MLPEncoder` + `LayerNorm`.

---

## Trajectory Tokenizers

**File:** `src/alpamayo_r1/models/delta_tokenizer.py`

#### `class DeltaTrajectoryTokenizer` — line 21

Genuine delta encoding. Methods are `encode` / `decode` (not `tokenize` /
`detokenize`). Defaults: `ego_xyz_min=(-4,-4,-10)`, `ego_xyz_max=(4,4,10)`,
`num_bins=1000`, `predict_yaw=False`. `vocab_size == num_bins`.

- `encode(...)` — line 47: pads the time axis, computes deltas
  `xyz[:,1:] - xyz[:,:-1]`, min-max normalizes to `[0,1]` over the configured
  ranges, quantizes to `num_bins`, flattens `(B, T, 3) → (B, T*3)`. With
  `predict_yaw=True` it also encodes wrapped yaw deltas → `(B, T*4)`.
- `decode(...)` — line 99: de-quantizes to deltas and `cumsum`s over time to
  recover absolute xyz; yaw rotations rebuilt via `get_yaw_rotation_matrices`
  (line 157, numpy polyfit) or from decoded yaw deltas.

**File:** `src/alpamayo_r1/models/token_utils.py`

| Function / class | Line | Purpose |
|------------------|------|---------|
| `to_special_token(token)` | 24 | `name → "<|name|>"` |
| `extract_traj_tokens(...)` | 29 | Vectorized extraction of tokens between `<|traj_future_start/end|>`, minus `future_token_start_idx` |
| `extract_between_special_tokens(...)` | 123 | Substring between `<|x_start|>` / `<|x_end|>` |
| `extract_text_tokens(tokenizer, tokens)` | 151 | Decode `cot` / `meta_action` / `answer` text |
| `StopAfterEOS(StoppingCriteria)` | 172 | Stops one token after all sequences emit EOS |
| `replace_padding_after_eos(...)` | 212 | Pad everything after the first EOS |

---

## Action Space

**File:** `src/alpamayo_r1/action_space/action_space.py`

#### `class ActionSpace(ABC, nn.Module)` — line 23

Abstract interface. Abstract methods: `traj_to_action` (line 26, future traj →
action), `action_to_traj` (line 50, action → traj + rotations),
`get_action_space_dims` (line 73). Concrete `is_within_bounds` (line 81) returns
all-True by default.

**File:** `src/alpamayo_r1/action_space/unicycle_accel_curvature.py`

#### `class UnicycleAccelCurvatureActionSpace(ActionSpace)` — line 36

Unicycle kinematic model; per-waypoint action is `(acceleration, curvature)`.
`__init__` defaults (line 39): `dt=0.1`, `n_waypoints=64`,
`accel_bounds=(-9.8, 9.8)`, `curvature_bounds=(-0.2, 0.2)`, plus accel/curvature
normalization buffers and Tikhonov λ/ridge weights. `get_action_space_dims()`
(line 98) returns `(n_waypoints, 2)`.

- `action_to_traj(action, hist_xyz, hist_rot, ...)` — line 300 (forward unroll).
  De-normalizes, then integrates:
  - velocity: `v_{t+1} = v0 + cumsum(a · dt)`
  - heading: `Δθ_t = κ_t · (v_t·dt + ½·a_t·dt²)` (second-order term included)
  - position: **trapezoidal** `Δx_t = dt/2·(v_t cosθ_t + v_{t+1} cosθ_{t+1})`
    (and similarly for y); `z` copied from the last history point.
  - rotations: `rot_2d_to_3d(rotation_matrix_torch(theta))`.
- `traj_to_action(...)` — line 227 (inverse, for training data): estimates `v0`,
  recovers `v` (`dxy_theta_to_v`), `accel` (`_v_to_a`, line 127), `kappa`
  (`_theta_v_a_to_kappa`, line 164), then normalizes.
- `estimate_t0_states(hist_xyz, hist_rot)` — line 209: estimates the initial
  velocity from history.

> Normalization is **inlined** (no `normalize_actions`/`denormalize_actions`
> methods) using registered buffers `accel_mean/std`, `curvature_mean/std`.

**File:** `src/alpamayo_r1/action_space/utils.py` (514 lines)

A numerical toolkit of 10 functions (no metrics like ADE here). Highlights:

| Function | Line | Purpose |
|----------|------|---------|
| `unwrap_angle` | 26 | Unwrap angle sequence via `round_2pi_torch` + cumsum |
| `first/second/third_order_D` | 33 / 47 / 62 | Finite-difference (smoothing) matrices |
| `construct_DTD` | 81 | Combined `DᵀWD` smoothing matrix |
| `solve_single_constraint` | 165 | Smoothed sequence with fixed first value (Cholesky) |
| `solve_xs_eq_y` | 241 | Solve `x·s ≈ y` with smoothing + adaptive ridge |
| `dxy_theta_to_v_without_v0` / `dxy_theta_to_v` | 319 / 405 | Trapezoidal velocity estimation |
| `theta_smooth` | 491 | Heading smoother (`so3_to_yaw_torch`→unwrap→solve) |

**File:** `src/alpamayo_r1/action_space/discrete_action_space.py`

#### `class DiscreteTrajectoryTokenizer` — line 24

A thin quantizer (not an `ActionSpace` subclass). It wraps an inner
`ActionSpace` (instantiated via Hydra), and in `encode` (line 47) maps a
continuous action → `traj_to_action` → min-max bins → integer tokens; `decode`
(line 80) inverts and calls `action_to_traj`.

---

## Diffusion

**File:** `src/alpamayo_r1/diffusion/base.py`

- `StepFn` (Protocol, line 26): a keyword-only callable `(*, x, t) -> Tensor`
  representing the denoiser.
- `class BaseDiffusion(ABC, nn.Module)` — line 45: stores `x_dims` (the per-sample
  shape, e.g. `(n_waypoints, 2)`); abstract `sample(...)` (line 64).

**File:** `src/alpamayo_r1/diffusion/flow_matching.py`

#### `class FlowMatching(BaseDiffusion)` — line 22

Rectified-flow / flow-matching sampler (not `FlowMatchingDiffusion`). `__init__`
(line 32): `int_method="euler"`, `train_timestep_sampler="beta"`,
`num_inference_steps=10`.

- `sample(batch_size, step_fn, device, return_all_steps, inference_step, int_method)`
  — line 61 → dispatches to `_euler`.
- `_euler(...)` — line 100. **Integrates from noise (t=0) to data (t=1):**
  ```python
  x = torch.randn(batch_size, *self.x_dims, device=device)
  time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
  for i in range(inference_step):
      dt = time_steps[i + 1] - time_steps[i]
      v = step_fn(x=x, t=time_steps[i]...)
      x = x + dt * v
  ```
- `construct_training_data(x)` — line 140: `noisy_x = t*x + (1-t)*noise`
  (so `t=1`→data, `t=0`→noise); beta-distributed `t` by default.
- `compute_loss_from_pred(...)` — line 166: `MSE(target = x - noise, pred)`
  (the velocity target).

---

## Geometry

**File:** `src/alpamayo_r1/geometry/rotation.py` (13 functions)

| Function | Line | Purpose |
|----------|------|---------|
| `so3_to_yaw_torch` / `so3_to_yaw_np` | 35 / 51 | Yaw from SO(3): `atan2(R[1,0], R[0,0])` |
| `euler_2_so3` | 66 | Euler → SO(3) via scipy |
| `angle_wrap` | 81 | Wrap to `[-π, π)` |
| `rotation_matrix` / `rotation_matrix_torch` | 95 / 119 | 2D rotation matrix `[[c,-s],[s,c]]` |
| `transform_coords_2d_np` | 138 | Rotate + translate 2D coords |
| `stable_gramschmidt` | 166 | Orthonormalize `(...,3,2) → (...,3,3)` |
| `rot_3d_to_2d` / `rot_2d_to_3d` | 187 / 207 | Convert between 2D/3D rotations |
| `ratan2` | 226 | Robust `atan2` (avoids NaN at origin) |
| `round_2pi` / `round_2pi_torch` | 245 / 257 | Normalize to `[-π, π]` via `atan2(sin, cos)` |

**File:** `src/alpamayo_r1/geometry/coordinates.py`

- `xyzrot_to_corners(xyz, rot, dims)` — converts a box center `(...,3)`, rotation
  `(...,3,3)`, and dims `(...,3)` into 8 corners `(...,8,3)` via scale → rotate →
  translate. First 4 corners are the bottom face, last 4 the top.

---

## Data Loading

**File:** `src/alpamayo_r1/load_physical_aiavdataset.py`

#### `load_physical_aiavdataset(...)` — line 27

The single dataset loader (there is no `prepare_data_from_token`).

```python
def load_physical_aiavdataset(
    clip_id: str, t0_us: int = 5_100_000,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    maybe_stream: bool = True, num_history_steps: int = 16,
    num_future_steps: int = 64, time_step: float = 0.1,
    camera_features: list | None = None, num_frames: int = 4,
) -> dict[str, Any]:
```

Loads egomotion + 4 camera streams, builds history/future timestamps, queries
ego poses, and transforms everything into the **ego-centric frame at `t0`** (the
last history pose):

```python
# lines 134-147
t0_rot_inv = spt.Rotation.from_quat(t0_quat).inv()
ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
ego_future_xyz_local  = t0_rot_inv.apply(ego_future_xyz  - t0_xyz)
ego_history_rot_local = (t0_rot_inv * Rotation.from_quat(ego_history_quat)).as_matrix()
```

Returns a dict with `image_frames (N_cam, num_frames, 3, H, W)`,
`camera_indices`, `ego_history_xyz (1,1,16,3)`, `ego_history_rot (1,1,16,3,3)`,
`ego_future_xyz (1,1,64,3)`, `ego_future_rot`, timestamps, `t0_us`, `clip_id`.
Includes a `raise ValueError` history-range guard (lines 99-103).

---

## Helpers & Logging

**File:** `src/alpamayo_r1/helper.py`

| Function | Line | Purpose |
|----------|------|---------|
| `create_message(frames)` | 35 | Build the system/user/assistant chat list. Inserts `<|traj_history_start|>` + `<|traj_history|>`×48 + `<|traj_history_end|>` and primes the assistant with `<|cot_start|>`. Raises `ValueError` if `frames.ndim != 4`. |
| `get_processor(tokenizer)` | 95 | Load `AutoProcessor` for `Qwen/Qwen3-VL-2B-Instruct` (`min_pixels=163840`, `max_pixels=196608`), then override `.tokenizer`. |
| `to_device(data, device, dtype)` | 120 | Recursively move tensors in nested dict/list structures. |

**File:** `src/alpamayo_r1/common/logging.py`

`setup_logging()` (line 34, colored root logger), `rank_prefixed_message` (57),
`get_global_rank` (65), and `class RankedLogger(logging.LoggerAdapter)` (77) for
distributed-aware logging with optional rank-zero-only output.

---

## Configuration

**File:** `src/alpamayo_r1/config.py`

#### `class AlpamayoR1Config(ReasoningVLAConfig)` — line 23

A HuggingFace config (`model_type = "alpamayo_r1"`), **not** a flat dataclass.
Sub-models are passed as Hydra config dicts:

| Field | Type | Default |
|-------|------|---------|
| `diffusion_cfg` | `dict \| None` | `None` |
| `action_space_cfg` | `dict \| None` | `None` |
| `action_in_proj_cfg` | `dict \| None` | `None` |
| `action_out_proj_cfg` | `dict \| None` | `None` |
| `expert_cfg` | `dict \| None` | `None` |
| `keep_same_dtype` | `bool` | `True` |
| `expert_non_causal_attention` | `bool` | `True` |
| `include_camera_ids` | `bool` | `False` |
| `include_frame_nums` | `bool` | `False` |

Scalar hyperparameters (`n_waypoints`, `dt`, `num_inference_steps`, …) live on the
individual sub-modules, not on this config.

---

## Test / Entry Point

**File:** `src/alpamayo_r1/test_inference.py`

A flat script (no `test_inference()` function). It loads one clip via
`load_physical_aiavdataset`, builds messages, loads
`AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16)`,
runs `sample_trajectories_from_data_with_vlm_rollout(..., num_traj_samples=1,
return_extra=True)`, prints the CoC text, and computes **minADE** (meters) over
the xy components of the predicted vs. ground-truth future. Requires a CUDA GPU
and dataset/model access. Run with `python -m alpamayo_r1.test_inference`.

---

## Module Dependencies

```
alpamayo_r1.py
  ├── base_model.py (ReasoningVLA, TrajectoryFusionMixin, configs)
  │     └── delta_tokenizer.py (history/future tokenizers via Hydra)
  ├── action_in_proj.py
  ├── action_out_proj (nn.Linear via Hydra)
  ├── diffusion/flow_matching.py  ──> diffusion/base.py
  ├── action_space/unicycle_accel_curvature.py
  │     ├── action_space/utils.py
  │     └── geometry/rotation.py
  ├── token_utils.py
  └── config.py

load_physical_aiavdataset.py ──> physical_ai_av, scipy
test_inference.py ──> alpamayo_r1.py, load_physical_aiavdataset.py, helper.py
```

See [model_inference_flow.md](model_inference_flow.md) for the end-to-end call
sequence and [architecture_overview.md](architecture_overview.md) for the design
rationale.
