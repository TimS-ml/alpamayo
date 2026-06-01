# Model Inference Flow

The end-to-end function call sequence during Alpamayo-R1 inference, from data
loading to trajectory prediction. Names, signatures, and line numbers were
verified against the source (post-`upstream/main` merge).

## Overview

Alpamayo-R1 pairs a Vision-Language Model (VLM) that reasons in text with a
diffusion **expert** transformer that generates trajectories. One inference call
performs:

1. **Data loading & ego-frame transform** — `load_physical_aiavdataset.py`
2. **Input preparation** — `helper.create_message` + processor
3. **Trajectory-history token fusion** — `TrajectoryFusionMixin.fuse_traj_tokens`
4. **VLM rollout** (Chain-of-Causation reasoning + KV cache) — `self.vlm.generate`
5. **Flow-matching diffusion sampling** (expert reuses the VLM KV cache)
6. **Action → trajectory unroll** — `UnicycleAccelCurvatureActionSpace.action_to_traj`

The entry point that orchestrates steps 3-6 is
`AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout()`
(`models/alpamayo_r1.py:124`).

---

## 1. Data Loading & Ego-Frame Transform

**`load_physical_aiavdataset(clip_id, t0_us=5_100_000, ...)`** —
`load_physical_aiavdataset.py:27`

- Loads egomotion and 4 camera streams (default cross-left/front-wide/
  cross-right/front-tele, `num_frames=4`).
- Builds `num_history_steps=16` history timestamps ending at `t0` and
  `num_future_steps=64` future timestamps (`time_step=0.1s`).
- Transforms all poses into the **ego frame at `t0`** (the last history pose):
  `xyz_local = R_t0⁻¹ · (xyz_world − xyz_t0)` (lines 134-147).

Outputs (batch dims prepended → `(B=1, n_traj_group=1, …)`):

| Key | Shape |
|-----|-------|
| `image_frames` | `(N_cam, num_frames, 3, H, W)` |
| `ego_history_xyz` | `(1, 1, 16, 3)` |
| `ego_history_rot` | `(1, 1, 16, 3, 3)` |
| `ego_future_xyz` | `(1, 1, 64, 3)` |
| `ego_future_rot` | `(1, 1, 64, 3, 3)` |

---

## 2. Input Preparation

**`create_message(frames)`** — `helper.py:35`

Builds a 3-message chat list:

- **system:** `"You are a driving assistant that generates safe and accurate actions."`
- **user:** one image entry per frame, then the text
  `"<|traj_history_start|>" + "<|traj_history|>"*48 + "<|traj_history_end|>" +
  "output the chain-of-thought reasoning ... then output the future trajectory."`
- **assistant:** `"<|cot_start|>"` (primes Chain-of-Causation).

**`get_processor(tokenizer)`** — `helper.py:95`: loads the
`Qwen/Qwen3-VL-2B-Instruct` processor (`min_pixels=163840`, `max_pixels=196608`)
and swaps in the custom tokenizer carrying the special trajectory/CoC tokens.

`test_inference.py` then calls
`processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
continue_final_message=True, return_dict=True, return_tensors="pt")` and moves
everything to CUDA with `helper.to_device`.

---

## 3. Trajectory-History Token Fusion

**`TrajectoryFusionMixin.fuse_traj_tokens(input_ids, traj_data)`** —
`base_model.py:168` (called from `alpamayo_r1.py:164`)

1. `tokenize_history_trajectory(...)` (`base_model.py:91`) encodes
   `ego_history_xyz/rot` into discrete history tokens.
2. `replace_pad_token(...)` (`base_model.py:85`) replaces the `<|traj_history|>`
   placeholder ids in `input_ids` with those tokens.

History tokenization uses `DeltaTrajectoryTokenizer.encode` (`delta_tokenizer.py:47`):
delta encoding `Δx_t = x_t − x_{t-1}`, min-max normalization over
`ego_xyz_min/max`, quantization to `num_bins=1000`.

---

## 4. VLM Rollout (Chain-of-Causation)

Inside `sample_trajectories_from_data_with_vlm_rollout` (lines 167-210):

```python
eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
logits_processor = LogitsProcessorList([ExpertLogitsProcessor(
    traj_token_offset=config.traj_token_start_idx, traj_vocab_size=config.traj_vocab_size)])

vlm_outputs = self.vlm.generate(
    input_ids=input_ids, generation_config=generation_config,
    stopping_criteria=stopping_criteria, logits_processor=logits_processor,
    **tokenized_data,                      # pixel_values, image_grid_thw, ...
)
prompt_cache = vlm_outputs.past_key_values        # the KV cache reused below
prefill_seq_len = prompt_cache.get_seq_length()
```

- Generation samples `num_traj_samples` sequences (`num_return_sequences`),
  `top_p=0.98`, `temperature=0.6`, up to `tokens_per_future_traj` (=64) new tokens.
- It **stops at `<|traj_future_start|>`** (`StopAfterEOS` allows one extra token so
  the KV cache is complete).
- `ExpertLogitsProcessor` masks the discrete trajectory-token block to `-inf` so
  the VLM produces only reasoning text, not discrete trajectory tokens.
- The VLM KV cache (`prompt_cache`) is saved and reused by the expert for every
  diffusion step — the VLM runs **once**.

Per-sequence `<|traj_future_start|>` positions are located to build the expert's
3-axis mRoPE `position_ids` and an `attention_mask` that hides padding
(lines 212-250).

---

## 5. Flow-Matching Diffusion Sampling

The expert denoises actions of shape `(b*, n_waypoints, 2)` where
`n_waypoints = action_space.get_action_space_dims()[0]` (=64) and `2 =
(acceleration, curvature)`.

**Denoiser closure `step_fn(x, t)`** — `alpamayo_r1.py:257`:

```python
future_token_embeds = self.action_in_proj(x, t)          # (b*, n_waypoints, hidden)
expert_out = self.expert(inputs_embeds=future_token_embeds,
                         position_ids=position_ids,
                         past_key_values=prompt_cache,     # reuse VLM context
                         attention_mask=attention_mask, use_cache=True, **fk)
prompt_cache.crop(prefill_seq_len)                        # remove expert's KV
last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
pred = self.action_out_proj(last_hidden).view(-1, n_waypoints, 2)  # velocity field
```

**Sampler** — `FlowMatching.sample → _euler` (`flow_matching.py:61 / 100`),
default `num_inference_steps=10`, integrating **from noise (t=0) to data (t=1):**

```python
x = torch.randn(batch_size, *self.x_dims, device=device)   # x_dims = (n_waypoints, 2)
time_steps = torch.linspace(0.0, 1.0, inference_step + 1)
for i in range(inference_step):
    dt = time_steps[i+1] - time_steps[i]
    v  = step_fn(x=x, t=time_steps[i]...)
    x  = x + dt * v
```

This direction matches training, where
`noisy_x = t*x + (1-t)*noise` and the target velocity is `x - noise`
(`construct_training_data` / `compute_loss_from_pred`).

`action_in_proj` (`PerWaypointActionInProjV2`, `action_in_proj.py:148`)
Fourier-encodes each action component and the timestep (log-spaced frequencies),
concatenates, and passes them through an MLP. `action_out_proj` is the linear
velocity head.

---

## 6. Action → Trajectory Unroll

**`UnicycleAccelCurvatureActionSpace.action_to_traj(action, hist_xyz, hist_rot)`**
— `unicycle_accel_curvature.py:300`

After de-normalizing `(accel, curvature)` with the registered buffers, integrate
the unicycle model (`dt=0.1`, `n_waypoints=64`):

```
velocity:  v_{t+1} = v0 + cumsum(a · dt)
heading:   Δθ_t    = κ_t · (v_t·dt + ½·a_t·dt²)           # 2nd-order term
position:  Δx_t    = dt/2 · (v_t cosθ_t + v_{t+1} cosθ_{t+1})   # trapezoidal
           Δy_t    = dt/2 · (v_t sinθ_t + v_{t+1} sinθ_{t+1})
z:         copied from the last history point
rotations: rot_2d_to_3d(rotation_matrix_torch(theta))
```

Returns `pred_xyz (..., 64, 3)` and `pred_rot (..., 64, 3, 3)`, which the caller
rearranges to `(B, num_traj_sets, num_traj_samples, 64, 3)`.

---

## Complete Call Graph

```
test_inference.py
└─ load_physical_aiavdataset()                         # images + ego history/future
└─ helper.create_message(frames)                       # chat messages
└─ helper.get_processor(model.tokenizer)               # processor (Qwen3-VL-2B)
   └─ processor.apply_chat_template(...)                # input_ids, pixel_values, ...
└─ AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout(data, ...)
   ├─ TrajectoryFusionMixin.fuse_traj_tokens()          # embed history tokens
   │   ├─ tokenize_history_trajectory()
   │   └─ DeltaTrajectoryTokenizer.encode()
   ├─ self.vlm.generate(...)                            # CoC text + KV cache
   │   ├─ StopAfterEOS (stop at <|traj_future_start|>)
   │   └─ ExpertLogitsProcessor (mask discrete traj tokens)
   ├─ self.diffusion.sample(batch_size, step_fn, ...)   # FlowMatching._euler, 10 steps
   │   └─ step_fn(x, t):
   │       ├─ action_in_proj(x, t)                      # PerWaypointActionInProjV2
   │       ├─ self.expert(inputs_embeds=..., past_key_values=prompt_cache)
   │       ├─ prompt_cache.crop(prefill_seq_len)
   │       └─ action_out_proj(last_hidden)              # velocity field
   └─ action_space.action_to_traj(sampled_action, ...)  # unicycle unroll
└─ minADE over xy(pred, ego_future_xyz)
```

---

## Function Reference Table

| Function | Location | Purpose |
|----------|----------|---------|
| `load_physical_aiavdataset()` | `load_physical_aiavdataset.py:27` | Load clip → images + ego history/future |
| `create_message()` | `helper.py:35` | Build chat messages |
| `get_processor()` | `helper.py:95` | Init Qwen3-VL-2B processor |
| `fuse_traj_tokens()` | `base_model.py:168` | Embed history-trajectory tokens |
| `DeltaTrajectoryTokenizer.encode()` | `delta_tokenizer.py:47` | Trajectory → discrete tokens |
| `sample_trajectories_from_data_with_vlm_rollout()` | `alpamayo_r1.py:124` | End-to-end inference |
| `self.vlm.generate()` | transformers | VLM reasoning + KV cache |
| `StopAfterEOS` | `token_utils.py:172` | Stop at `<|traj_future_start|>` |
| `ExpertLogitsProcessor` | `alpamayo_r1.py:41` | Mask discrete trajectory logits |
| `step_fn()` | `alpamayo_r1.py:257` | Per-step denoiser (expert) |
| `FlowMatching._euler()` | `flow_matching.py:100` | Euler flow-matching loop |
| `PerWaypointActionInProjV2.forward()` | `action_in_proj.py:148` | (action, t) → embeddings |
| `action_to_traj()` | `unicycle_accel_curvature.py:300` | Actions → trajectory + rotations |

---

## Notes

- **VLM runs once.** Its KV cache (`prompt_cache`) is reused by the expert for
  every diffusion step; `prompt_cache.crop(prefill_seq_len)` removes the expert's
  appended keys/values after each step.
- **Diffusion direction:** noise (t=0) → data (t=1), Euler `x += dt·v`. This is
  rectified-flow / flow-matching, not score-based DDPM.
- **Action space:** the model predicts `(acceleration, curvature)` per waypoint;
  the unicycle unroll produces smooth, physically-plausible trajectories.
- **Coordinate frame:** everything is ego-centric, centered at the vehicle pose
  at `t0` (the last history step).
- **Shapes:** `num_traj_samples` (default 6 in the API; the test script uses 1)
  trajectories of `n_waypoints=64` steps are produced per call.
