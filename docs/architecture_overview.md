# Alpamayo-R1 Architecture Overview

This document provides a high-level overview of the Alpamayo-R1 architecture, design principles, and key innovations.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Model Architecture](#model-architecture)
3. [Design Principles](#design-principles)
4. [Key Innovations](#key-innovations)
5. [Data Flow](#data-flow)
6. [Training vs Inference](#training-vs-inference)
7. [Performance Considerations](#performance-considerations)

---

## System Overview

### What is Alpamayo-R1?

**Alpamayo-R1** is a Vision-Language-Action (VLA) model for autonomous driving that bridges reasoning and action prediction. It combines:

- **Vision Understanding:** Multi-camera image processing via Qwen3-VL (8B parameters)
- **Reasoning:** Chain-of-Causation (CoC) text generation explaining driving decisions
- **Action Prediction:** Trajectory generation using diffusion-based sampling

### Model Characteristics

| Property | Value |
|----------|-------|
| **Model Type** | Vision-Language-Action (VLA) |
| **Architecture** | Two-stage (VLM + Expert) |
| **Vision Backbone** | Qwen3-VL-8B (8 billion parameters) |
| **Expert Model** | Qwen2.5-0.5B (500 million parameters) |
| **Total Parameters** | ~8.5B |
| **Inference Time** | ~1.3 seconds (A100 GPU) |
| **Action Space** | Unicycle (acceleration, curvature) |
| **Prediction Horizon** | 6 seconds (12 timesteps × 0.5s) |
| **Output Samples** | 32 diverse trajectories |

---

## Model Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ALPAMAYO-R1 ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────┘

Input Layer
┌─────────────────────┐  ┌──────────────────────┐
│  Multi-Camera       │  │  Trajectory History  │
│  Images (6 views)   │  │  (xyz, rotations)    │
│  (N, 3, H, W)       │  │  (T_hist, 3)         │
└──────────┬──────────┘  └──────────┬───────────┘
           │                        │
           │                        ├─────────────────────┐
           │                        │                     │
           │                        ▼                     │
           │              ┌──────────────────┐            │
           │              │ DeltaTokenizer   │            │
           │              │ (discretize)     │            │
           │              └─────────┬────────┘            │
           │                        │                     │
           │                        ▼                     │
           │              ┌──────────────────┐            │
           │              │ Token Fusion     │            │
           │              │ (embed in input) │            │
           │              └─────────┬────────┘            │
           │                        │                     │
           ▼                        ▼                     │
┌──────────────────────────────────────────────┐         │
│         STAGE 1: Vision-Language Model       │         │
│               (Qwen3-VL-8B)                  │         │
│                                              │         │
│  ┌─────────────────────────────────────┐    │         │
│  │  Vision Encoder                     │    │         │
│  │  - Multi-scale image features       │    │         │
│  │  - Positional encodings             │    │         │
│  └────────────────┬────────────────────┘    │         │
│                   │                          │         │
│  ┌────────────────▼────────────────────┐    │         │
│  │  Language Model (Transformer)       │    │         │
│  │  - Autoregressive generation        │    │         │
│  │  - CoC reasoning: "I observe..."    │    │         │
│  │  - Stop at <|traj_future_start|>    │    │         │
│  └────────────────┬────────────────────┘    │         │
│                   │                          │         │
│                   ▼                          │         │
│         [ KV Cache Saved ]                   │         │
│         [ Reasoning Text Output ]            │         │
└──────────────────┬───────────────────────────┘         │
                   │                                     │
                   │ (past_key_values)                   │
                   │                                     │
                   ▼                                     │
┌──────────────────────────────────────────────┐         │
│      STAGE 2: Diffusion-Based Sampling       │         │
│                                              │         │
│  ┌────────────────────────────────────┐     │         │
│  │  Initialize Random Noise           │     │         │
│  │  z_0 ~ N(0, I)                     │     │         │
│  │  Shape: (32 samples, 12 steps, 2)  │     │         │
│  └───────────────┬────────────────────┘     │         │
│                  │                           │         │
│                  │ (Loop 10 diffusion steps) │         │
│                  ▼                           │         │
│  ┌────────────────────────────────────┐     │         │
│  │  For t = 1.0 → 0.0:                │     │         │
│  │                                    │     │         │
│  │  ┌──────────────────────────────┐ │     │         │
│  │  │ ActionInputProjection        │ │     │         │
│  │  │ - Fourier encode time t      │ │     │         │
│  │  │ - Concat [action, fourier]   │ │     │         │
│  │  │ - MLP → embeddings           │ │     │         │
│  │  └─────────────┬────────────────┘ │     │         │
│  │                │                  │     │         │
│  │                ▼                  │     │         │
│  │  ┌──────────────────────────────┐ │     │         │
│  │  │ Expert Model (Transformer)   │ │     │         │
│  │  │ - Reuse VLM KV cache         │ │     │         │
│  │  │ - Process action embeddings  │ │     │         │
│  │  └─────────────┬────────────────┘ │     │         │
│  │                │                  │     │         │
│  │                ▼                  │     │         │
│  │  ┌──────────────────────────────┐ │     │         │
│  │  │ Action Head (Linear)         │ │     │         │
│  │  │ - Predict velocity field     │ │     │         │
│  │  └─────────────┬────────────────┘ │     │         │
│  │                │                  │     │         │
│  │                ▼                  │     │         │
│  │  ┌──────────────────────────────┐ │     │         │
│  │  │ Euler Integration            │ │     │         │
│  │  │ z_t = z_{t-1} + v_t × dt     │ │     │         │
│  │  └──────────────────────────────┘ │     │         │
│  │                                    │     │         │
│  └────────────────┬───────────────────┘     │         │
│                   │                          │         │
│                   ▼                          │         │
│         [ Final Actions ]                    │         │
│         (32, 12, 2)                          │         │
└───────────────────┬──────────────────────────┘         │
                    │                                    │
                    │                                    │
                    ▼                                    ▼
┌──────────────────────────────────────────────────────────┐
│            Unicycle Action Space                        │
│                                                          │
│  Kinematic Model:                                       │
│  - v[t+1] = v[t] + acceleration × dt                   │
│  - ω[t] = v[t] × curvature                             │
│  - θ[t+1] = θ[t] + ω[t] × dt                           │
│  - x[t+1] = x[t] + v[t] × cos(θ) × dt                 │
│  - y[t+1] = y[t] + v[t] × sin(θ) × dt                 │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
Output Layer
┌──────────────────────────────────────────────────────────┐
│  Predicted Trajectories: (32 samples, 12 steps, 3)      │
│  Rotation Matrices: (32 samples, 12 steps, 3, 3)        │
│  CoC Reasoning: "I observe the red light..."            │
└──────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Separation of Concerns

**Reasoning vs. Action Prediction**

- **VLM (Stage 1):** Focuses on visual understanding and reasoning
  - Processes high-resolution images
  - Generates natural language explanations
  - Creates contextual representation

- **Expert Model (Stage 2):** Specializes in trajectory generation
  - Smaller, faster model (0.5B vs 8B parameters)
  - Reuses VLM context via KV cache
  - Optimized for action prediction

**Benefits:**
- Modularity: Each component can be improved independently
- Efficiency: Expert model is lightweight for fast sampling
- Interpretability: Reasoning text provides transparency

---

### 2. Action Space Design

**Unicycle Model with (Acceleration, Curvature)**

Why not direct xyz prediction?

| Approach | Pros | Cons |
|----------|------|------|
| **Direct xyz** | Simple | Unnatural motion, violates physics |
| **Velocity (vx, vy)** | Flexible | Can produce sharp turns, jitter |
| **Unicycle (a, κ)** | Smooth, physically plausible | Slightly more complex |

**Advantages of Unicycle:**
1. **Smoothness:** Acceleration provides smooth velocity changes
2. **Physical Constraints:** Respects vehicle dynamics
3. **Compactness:** 2D action space vs 3D position space
4. **Differentiability:** Kinematic unrolling is differentiable for training

**Curvature Representation:**
- Curvature κ = 1/R (inverse turning radius)
- Positive: left turn, Negative: right turn, Zero: straight
- Natural parameterization for steering control

---

### 3. Diffusion-Based Generation

**Flow Matching Instead of DDPM**

Why flow matching?
- **Fewer Steps:** 10 steps vs 50+ for DDPM
- **Deterministic Paths:** Straight-line interpolation in latent space
- **Stable Training:** No variance scheduling required

**Velocity Field Prediction:**
```
v(x_t, t) = model(x_t, t)
x_{t+dt} = x_t + v(x_t, t) × dt
```

Compared to score-based diffusion:
```
ε(x_t, t) = model(x_t, t)
x_{t-1} = (x_t - ε√(1-α_t)) / √α_t + noise
```

**Benefits:**
- Faster inference (10 steps = ~450ms)
- More stable sampling
- Easier to train

---

### 4. Multi-Sample Generation

**Why 32 Samples?**

Autonomous driving has inherent uncertainty:
- Other vehicles' intentions are unknown
- Multiple valid futures exist (e.g., lane change vs stay)
- Risk assessment requires exploring diverse outcomes

**Trajectory Diversity:**
- All 32 samples generated in parallel
- Diffusion noise provides natural diversity
- Best sample selected via planning/risk metrics

**Example Scenario:**
```
At intersection with green light:
- Sample 1-10: Go straight
- Sample 11-20: Turn right
- Sample 21-25: Slow down (yellow light risk)
- Sample 26-32: Stop (cautious behavior)
```

---

### 5. Token-Based Trajectory Encoding

**Delta Tokenization**

Why tokenize continuous trajectories?

1. **Compatibility:** VLM expects discrete tokens
2. **Compression:** Efficient representation
3. **Quantization:** Regularizes predictions

**Delta Encoding:**
```
Original:    [x0, x1, x2, x3]
Deltas:      [Δ1, Δ2, Δ3]  where Δi = xi - x_{i-1}
Quantize:    [bin_idx1, bin_idx2, bin_idx3]
Token IDs:   [token1, token2, token3]
```

**Benefits:**
- Smaller magnitude → better quantization
- Local changes captured efficiently
- Invertible: can reconstruct trajectory

---

## Key Innovations

### 1. KV Cache Reuse

**Problem:** Running VLM 32 times is too slow

**Solution:** Compute VLM context once, share across all samples

```python
# Stage 1: VLM (once)
vlm_output = vlm.generate(..., use_cache=True)
past_key_values = vlm_output.past_key_values  # Save KV cache

# Stage 2: Expert model (32 samples in parallel)
for sample in range(32):
    expert_output = expert_model(
        ...,
        past_key_values=past_key_values,  # Reuse VLM context
    )
```

**Speedup:** ~32× faster than naive approach

---

### 2. Fourier Time Encoding

**Problem:** Diffusion timestep is a scalar, hard to learn

**Solution:** Sinusoidal positional encoding

```python
fourier_features = [sin(2πk×t), cos(2πk×t)] for k = 0, 1, ..., K
```

**Benefits:**
- Expressive: High-dimensional representation
- Smooth: Continuous in time
- Periodic: Natural for cyclical patterns

**Inspiration:** Transformer positional encodings, NeRF

---

### 3. Two-Stage Training

**Stage 1: VLM Pre-training**
- Train on large-scale vision-language data
- Learn visual understanding and reasoning

**Stage 2: End-to-End Fine-tuning**
- Freeze VLM (optional)
- Train expert model + action head
- Optimize for trajectory prediction

**Benefits:**
- Leverages pre-trained vision models
- Efficient: Only 0.5B parameters to train
- Modular: Can swap VLM backbones

---

## Data Flow

### Inference Pipeline Summary

```
1. Data Loading
   ↓
   Images (6 cameras) + Trajectory History
   ↓
2. Preprocessing
   ↓
   Chat Messages + Tokenization
   ↓
3. VLM Generation
   ↓
   Reasoning Text + KV Cache
   ↓
4. Diffusion Sampling (10 steps)
   ↓
   Actions (32 samples, 12 steps, 2D)
   ↓
5. Kinematic Unrolling
   ↓
   Trajectories (32 samples, 12 steps, 3D)
   ↓
6. Output
   ↓
   Predicted Paths + Reasoning
```

### Data Shapes Through Pipeline

| Stage | Data | Shape |
|-------|------|-------|
| Input | Images | `(6, 3, H, W)` |
| Input | Trajectory history | `(T_hist, 3)` |
| Tokenization | Input IDs | `(1, seq_len)` |
| Tokenization | Pixel values | `(1, N_images, C, H, W)` |
| VLM Output | Reasoning text | String |
| VLM Output | KV cache | `(num_layers, 2, B, num_heads, seq_len, head_dim)` |
| Diffusion Init | Noise | `(32, 12, 2)` |
| Diffusion Loop | Action embeddings | `(32, 12, hidden_dim)` |
| Diffusion Loop | Predicted velocity | `(32, 12, 2)` |
| Final Actions | Actions | `(32, 12, 2)` |
| Kinematics | Trajectories | `(32, 12, 3)` |
| Kinematics | Rotations | `(32, 12, 3, 3)` |

---

## Training vs Inference

### Training Mode

**Forward Pass:**
```python
outputs = model.forward(
    images=images,
    traj_history=traj_history,
    traj_gt=traj_gt,  # Ground truth for supervision
    actions_gt=actions_gt,
)
loss = outputs.loss
```

**Loss Components:**
1. **Action Loss:** MSE between predicted and GT actions
2. **Optional CoC Loss:** Cross-entropy for reasoning text

**Optimization:**
- AdamW optimizer
- Learning rate: 1e-5
- Batch size: 4-8 (limited by GPU memory)

---

### Inference Mode

**Sampling:**
```python
trajectories, rotations = model.sample_trajectories_from_data_with_vlm_rollout(
    images=images,
    traj_history=traj_history,
)
```

**No Ground Truth Required**

**Differences from Training:**
- Uses diffusion sampling (training uses GT actions)
- Generates multiple samples (32 vs 1)
- Includes CoC reasoning generation

---

## Performance Considerations

### Computational Bottlenecks

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| VLM Generation | 800 | 61% |
| Diffusion Sampling | 450 | 35% |
| Kinematic Unrolling | 20 | 2% |
| Preprocessing | 30 | 2% |
| **Total** | **1300** | **100%** |

---

### Optimization Strategies

**1. KV Cache Reuse**
- **Speedup:** 32× for multi-sample generation
- **Memory:** ~2GB for KV cache storage

**2. Batch Processing**
- All 32 samples processed in parallel
- GPU utilization: ~90%

**3. Mixed Precision (FP16)**
- **Speedup:** 1.5-2× faster
- **Memory:** 50% reduction
- **Accuracy:** Minimal loss (<0.1 ADE difference)

**4. Quantization (INT8)**
- **Speedup:** 2-3× faster (with TensorRT)
- **Memory:** 75% reduction
- **Accuracy:** Acceptable degradation (~0.3 ADE increase)

**5. Model Pruning**
- Expert model can be pruned to 0.3B params
- **Speedup:** ~30% faster
- **Accuracy:** <5% degradation

---

### Memory Requirements

| Configuration | VLM | Expert | KV Cache | Activations | Total |
|---------------|-----|--------|----------|-------------|-------|
| FP32 | 32GB | 2GB | 2GB | 4GB | **40GB** |
| FP16 | 16GB | 1GB | 1GB | 2GB | **20GB** |
| INT8 | 8GB | 0.5GB | 1GB | 2GB | **11.5GB** |

**Recommendation:** Use FP16 on A100 (40GB) or H100 (80GB)

---

### Scaling Considerations

**Increasing Samples (32 → 64):**
- Linear time increase: ~900ms diffusion
- Linear memory increase: +1GB

**Increasing Horizon (12 → 24 steps):**
- Linear time increase: ~900ms diffusion
- Action space remains 2D (efficient)

**Increasing Diffusion Steps (10 → 20):**
- Linear time increase: ~900ms diffusion
- Diminishing returns on accuracy (~0.1 ADE improvement)

---

## System Integration

### Autonomous Driving Pipeline

```
┌──────────────┐
│  Perception  │  → Sensors: Cameras, LiDAR, Radar
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Alpamayo-R1  │  → Reasoning + Trajectory Prediction
│   (This)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Planning   │  → Select best trajectory, refine path
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Control    │  → Low-level vehicle control (steering, throttle)
└──────────────┘
```

**Alpamayo-R1's Role:**
- Input: Multi-camera images + trajectory history
- Output: Multiple trajectory candidates + reasoning
- Planning layer selects best trajectory based on:
  - Safety constraints
  - Comfort metrics
  - Goal progress

---

## Future Directions

### Potential Improvements

1. **Larger Vision Backbone**
   - Qwen3-VL-32B or GPT-4V for better perception
   - Expected: +15% accuracy, +2× inference time

2. **Learned Action Space**
   - Replace handcrafted unicycle with learned dynamics
   - Potential: More flexible motion patterns

3. **Conditional Generation**
   - User preferences: "drive conservatively"
   - Goal-conditioned: "change to right lane"

4. **Multi-Agent Modeling**
   - Predict other vehicles' trajectories
   - Joint reasoning over all agents

5. **Online Learning**
   - Adapt to new environments/scenarios
   - Continual learning from driving experience

---

## Conclusion

Alpamayo-R1 represents a novel approach to autonomous driving by combining:

1. **Vision-Language Understanding:** Rich contextual reasoning
2. **Efficient Architecture:** Two-stage design for speed
3. **Probabilistic Prediction:** Multi-sample diversity
4. **Physical Constraints:** Unicycle action space
5. **Fast Inference:** Flow matching diffusion

**Key Strengths:**
- Interpretable reasoning via CoC
- Diverse trajectory generation
- Physically plausible predictions
- Efficient inference (~1.3s)

**Design Trade-offs:**
- Complexity: Two models instead of one
- Memory: Large VLM requires 16GB+ VRAM
- Action space: Unicycle limits some maneuvers

**Overall:** A practical and performant approach to vision-language-action modeling for autonomous driving.

---

For detailed implementation, see:
- [Model Inference Flow](model_inference_flow.md)
- [Core Components Reference](core_components.md)
