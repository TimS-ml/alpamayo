# Alpamayo-R1 Architecture Overview

A high-level overview of the Alpamayo-R1 architecture, design principles, and key
mechanisms. Specifics (class names, shapes, defaults) were verified against the
source (post-`upstream/main` merge). For exact APIs see
[core_components.md](core_components.md); for the call sequence see
[model_inference_flow.md](model_inference_flow.md).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Model Architecture](#model-architecture)
3. [Design Principles](#design-principles)
4. [Key Mechanisms](#key-mechanisms)
5. [Data Flow & Shapes](#data-flow--shapes)
6. [Training vs Inference](#training-vs-inference)

---

## System Overview

**Alpamayo-R1** (released model: `nvidia/Alpamayo-R1-10B`) is a
Vision-Language-Action (VLA) model for autonomous driving that bridges reasoning
and action prediction. It combines:

- **Vision + reasoning:** a Qwen3-VL backbone processes multi-camera images and
  generates Chain-of-Causation (CoC) text explaining the driving decision.
- **Action prediction:** a smaller **expert** transformer denoises trajectories
  with flow matching, reusing the VLM's KV cache.
- **Kinematic decoding:** a unicycle action space turns predicted
  `(acceleration, curvature)` controls into smooth xyz trajectories.

| Property | Value |
|----------|-------|
| Model type | Vision-Language-Action (VLA) |
| Architecture | Two-stage (VLM reasoning + diffusion expert) |
| VLM backbone | `Qwen/Qwen3-VL-8B-Instruct` (config default) |
| Expert model | Re-configured copy of the VLM text transformer (`expert_cfg`) |
| Action space | Unicycle `(acceleration, curvature)` |
| Waypoints (horizon) | `n_waypoints = 64` at `dt = 0.1 s` |
| Diffusion | Flow matching, `num_inference_steps = 10` (Euler) |
| Samples per call | `num_traj_samples = 6` (API default) |
| Trajectory tokens | 768 discrete tokens `<i0>..<i767>` |

---

## Model Architecture

```
                Multi-camera images          Ego trajectory history
                (N_cam, num_frames,           (1, 1, 16, 3) + rotations
                 3, H, W)                            │
                       │                             ▼
                       │                   DeltaTrajectoryTokenizer.encode
                       │                   (delta + quantize → history tokens)
                       │                             │
                       │                   fuse_traj_tokens (replace <|traj_history|>)
                       ▼                             ▼
        ┌──────────────────────────────────────────────────────┐
        │  STAGE 1 — Vision-Language Model (Qwen3-VL)            │
        │  vlm.generate(...)                                     │
        │   • Chain-of-Causation reasoning text                 │
        │   • stop at <|traj_future_start|>                     │
        │   • ExpertLogitsProcessor masks discrete traj tokens  │
        │   → reasoning text + KV cache (prompt_cache)          │
        └───────────────────────────┬──────────────────────────┘
                                    │ past_key_values (reused)
                                    ▼
        ┌──────────────────────────────────────────────────────┐
        │  STAGE 2 — Flow-Matching Expert (10 Euler steps)      │
        │  x ~ N(0, I), shape (b*, 64, 2)                       │
        │  for t in linspace(0, 1, 11):                         │
        │    emb  = action_in_proj(x, t)   # Fourier + MLP      │
        │    h    = expert(emb, past_key_values=prompt_cache)   │
        │    v    = action_out_proj(h)     # velocity field     │
        │    x    = x + dt * v                                  │
        │  → actions (b*, 64, 2) = (accel, curvature)           │
        └───────────────────────────┬──────────────────────────┘
                                    ▼
        ┌──────────────────────────────────────────────────────┐
        │  Unicycle unroll — action_to_traj                     │
        │   v_{t+1} = v0 + Σ a·dt                               │
        │   Δθ_t    = κ_t·(v_t·dt + ½·a_t·dt²)                  │
        │   Δx_t    = dt/2·(v_t cosθ_t + v_{t+1} cosθ_{t+1})    │
        └───────────────────────────┬──────────────────────────┘
                                    ▼
            Trajectories (B, sets, samples, 64, 3) + rotations (…, 64, 3, 3)
            + CoC reasoning text
```

---

## Design Principles

### 1. Separation of reasoning and action

- **VLM (stage 1):** high-resolution visual understanding + natural-language
  reasoning; runs **once**.
- **Expert (stage 2):** a lighter transformer specialized for trajectory
  denoising; reuses the VLM context via the KV cache.

Benefits: modularity, interpretability (the CoC text), and efficiency (the heavy
VLM is not re-run per diffusion step or per sample).

### 2. Unicycle action space `(acceleration, curvature)`

Instead of predicting xyz directly, the model predicts per-waypoint acceleration
and curvature, then integrates a unicycle model. This yields smooth, physically
plausible motion and a compact 2-D action per step. Curvature `κ = 1/R` encodes
steering (positive = left, negative = right, zero = straight). The implementation
uses an explicit second-order heading term and trapezoidal position integration
(see [model_inference_flow.md](model_inference_flow.md)).

### 3. Flow matching for generation

Trajectory generation uses **flow matching** (rectified flow), integrating from
noise (`t=0`) to data (`t=1`) with a few Euler steps (default 10). Training
interpolates `noisy_x = t·x + (1−t)·noise` and regresses the velocity target
`x − noise`. Compared to score-based DDPM this needs far fewer steps and trains
stably.

### 4. Multi-sample generation

A single call samples multiple trajectories (`num_traj_samples`), driven by the
diffusion noise, to capture the inherent uncertainty of driving (e.g. multiple
plausible maneuvers). A downstream planner selects among them.

### 5. Discrete trajectory tokens

A `DeltaTrajectoryTokenizer` delta-encodes and quantizes trajectories into 768
discrete tokens (`<i0>..<i767>`). These are used to embed the **history**
trajectory into the VLM prompt (and are available for discrete-token research
paths). The diffusion expert itself operates on continuous actions.

---

## Key Mechanisms

### KV cache reuse

The VLM is expensive, so it runs once; its `past_key_values` (`prompt_cache`) is
passed to the expert at every diffusion step. After each step,
`prompt_cache.crop(prefill_seq_len)` removes the expert's appended keys/values so
the next step starts from the original prefill cache.

### Fourier (log-spaced) time & action encoding

`PerWaypointActionInProjV2` encodes each action component and the diffusion
timestep with `FourierEncoderV2`, which uses **log-spaced** frequencies
(`logspace(0, log10(max_freq), dim/2)`, default `max_freq=100`) and returns
`[sin(2π·f·x), cos(2π·f·x)]·√2`. The features pass through an `MLPEncoder`
(`SiLU` + `RMSNorm`, hidden 1024).

### Stopping & logits masking

Generation stops at the special `<|traj_future_start|>` token via `StopAfterEOS`,
and `ExpertLogitsProcessor` masks the 768 discrete trajectory logits to `-inf`
during CoC so the VLM emits reasoning text only.

---

## Data Flow & Shapes

| Stage | Data | Shape |
|-------|------|-------|
| Input | Camera frames | `(N_cam, num_frames, 3, H, W)` |
| Input | Ego history xyz / rot | `(1, 1, 16, 3)` / `(1, 1, 16, 3, 3)` |
| Tokenization | History tokens | `(B, 16·tokens_per_history_traj)` |
| VLM | Reasoning text | string(s) |
| VLM | KV cache | `prompt_cache` (Cache object) |
| Diffusion init | Noise `x` | `(b*, 64, 2)` |
| Diffusion | Action embeddings | `(b*, 64, hidden)` |
| Diffusion | Velocity field | `(b*, 64, 2)` |
| Output | Actions | `(b*, 64, 2)` = (accel, curvature) |
| Output | Trajectories | `(B, sets, samples, 64, 3)` |
| Output | Rotations | `(B, sets, samples, 64, 3, 3)` |

(`b* = B · num_traj_samples · num_traj_sets`.)

---

## Training vs Inference

- **Inference** is implemented in
  `sample_trajectories_from_data_with_vlm_rollout` (VLM rollout + diffusion
  sampling, no ground truth needed). These public files do **not** contain a
  training `forward()`.
- **Training** (scripts not included here) would supervise the flow-matching
  velocity target `x − noise` (`compute_loss_from_pred`) and may also supervise
  the CoC text. The inverse map `traj_to_action` converts ground-truth
  trajectories into `(accel, curvature)` targets.

---

## Role in the Autonomous-Driving Stack

```
Perception ──> Alpamayo-R1 (reasoning + trajectory candidates) ──> Planning ──> Control
```

Alpamayo-R1 consumes multi-camera images plus a short ego-trajectory history and
produces multiple candidate future trajectories together with a textual rationale,
which a downstream planner can score for safety, comfort, and progress.
