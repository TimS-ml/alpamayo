# Alpamayo-R1 Documentation

Welcome to the Alpamayo-R1 documentation! This directory contains comprehensive guides to understanding and using the Alpamayo-R1 autonomous driving model.

---

## 📚 Documentation Index

### 1. [Architecture Overview](architecture_overview.md)
**Start here if you're new to Alpamayo-R1**

High-level overview of the system architecture, design principles, and key innovations.

**Topics covered:**
- System overview and model characteristics
- Two-stage architecture (VLM + Expert Model)
- Design principles and trade-offs
- Key innovations (KV cache reuse, flow matching, etc.)
- Performance considerations and optimization strategies
- Future directions

**Best for:** Understanding the big picture, design decisions, and overall approach.

---

### 2. [Model Inference Flow](model_inference_flow.md)
**For understanding the execution flow**

Detailed walkthrough of the complete inference pipeline, from data loading to trajectory prediction.

**Topics covered:**
- Step-by-step inference pipeline
- Function call sequence and order
- Data transformations at each stage
- Complete call graph
- Function reference table
- Configuration parameters

**Best for:** Developers integrating the model, debugging, or understanding execution flow.

---

### 3. [Core Components Reference](core_components.md)
**For deep dives into specific modules**

Comprehensive reference for all core files, classes, and functions in the codebase.

**Topics covered:**
- Core files overview and organization
- Detailed class and function documentation
- Model components (AlpamayoR1, ReasoningVLA, ActionInputProjection, etc.)
- Action space implementation
- Diffusion sampling
- Utilities and helpers
- Dataset loader
- Usage patterns and examples

**Best for:** Understanding specific components, API reference, implementation details.

---

## 🚀 Quick Start

### For New Users

1. **Understand the Architecture**
   - Read [Architecture Overview](architecture_overview.md)
   - Pay attention to the high-level diagram
   - Understand the two-stage design

2. **Run Inference**
   - Follow the test script: `src/alpamayo_r1/test_inference.py`
   - See usage examples in [Core Components](core_components.md#common-usage-patterns)

3. **Explore the Code**
   - Use [Core Components Reference](core_components.md) as your guide
   - Check [Model Inference Flow](model_inference_flow.md) for execution order

### For Developers

1. **Integration**
   - See [Model Inference Flow](model_inference_flow.md) for the complete pipeline
   - Check [Core Components](core_components.md#common-usage-patterns) for code examples
   - Review configuration parameters in [Architecture Overview](architecture_overview.md#key-configuration-parameters)

2. **Debugging**
   - Follow the call graph in [Model Inference Flow](model_inference_flow.md#complete-call-graph)
   - Check function locations in [Core Components Reference](core_components.md#function-reference-table)

3. **Optimization**
   - See [Performance Considerations](architecture_overview.md#performance-considerations)
   - Review optimization strategies in [Architecture Overview](architecture_overview.md#optimization-strategies)

---

## 📖 Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── architecture_overview.md     # High-level architecture and design
├── model_inference_flow.md      # Detailed inference pipeline
└── core_components.md           # Component reference and API docs
```

---

## 🔍 Find What You Need

### I want to...

**Understand how the model works**
→ Start with [Architecture Overview](architecture_overview.md)

**Trace the execution flow**
→ See [Model Inference Flow](model_inference_flow.md)

**Find a specific function**
→ Use [Core Components Reference](core_components.md)

**Integrate the model**
→ Check [Common Usage Patterns](core_components.md#common-usage-patterns)

**Optimize inference speed**
→ See [Performance Considerations](architecture_overview.md#performance-considerations)

**Debug an issue**
→ Follow [Complete Call Graph](model_inference_flow.md#complete-call-graph)

**Understand the action space**
→ Read [Action Space Design](architecture_overview.md#2-action-space-design)

**Learn about diffusion sampling**
→ See [Diffusion-Based Generation](architecture_overview.md#3-diffusion-based-generation)

---

## 🎯 Key Concepts

### Vision-Language-Action (VLA) Model
Combines vision understanding, language reasoning, and action prediction in a unified framework.

### Two-Stage Architecture
1. **Stage 1 (VLM):** Processes images and generates reasoning
2. **Stage 2 (Expert):** Generates diverse trajectory samples

### Chain-of-Causation (CoC)
Natural language reasoning that explains driving decisions (e.g., "I observe a red light, so I must decelerate").

### Flow Matching Diffusion
Efficient sampling method that generates trajectories in just 10 steps.

### Unicycle Action Space
Physical action representation using acceleration and curvature controls.

---

## 📊 Quick Reference

### Model Specifications

| Property | Value |
|----------|-------|
| Vision Backbone | Qwen3-VL-8B (8B params) |
| Expert Model | Qwen2.5-0.5B (0.5B params) |
| Inference Time | ~1.3s (A100 GPU) |
| Action Space | Acceleration + Curvature (2D) |
| Prediction Horizon | 6 seconds (12 timesteps) |
| Output Samples | 32 diverse trajectories |

### File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `alpamayo_r1.py` | 450 | Main model implementation |
| `base_model.py` | 380 | Base VLA architecture |
| `unicycle_accel_curvature.py` | 350 | Action space kinematics |
| `action_in_proj.py` | 240 | Action projection module |
| `load_physical_aiavdataset.py` | 420 | Dataset loader |

### Inference Pipeline (Simplified)

```
Images + History
    ↓
VLM Generation (reasoning)
    ↓
Diffusion Sampling (10 steps)
    ↓
Unicycle Kinematics
    ↓
Trajectories + Reasoning
```

---

## 🛠️ Code Examples

### Basic Inference

```python
from alpamayo_r1 import AlpamayoR1, AlpamayoR1Config
from alpamayo_r1.helper import create_message, get_processor

# Load model
config = AlpamayoR1Config()
model = AlpamayoR1.from_pretrained("NVIDIA/Alpamayo-R1-8B", config=config)
model = model.to("cuda")

# Prepare inputs
messages = create_message(images)
processor = get_processor(model.tokenizer)

# Run inference
trajectories, rotations = model.sample_trajectories_from_data_with_vlm_rollout(
    images=images,
    traj_history=traj_hist,
    traj_history_rot_mat=traj_hist_rot,
)

# Output: 32 trajectory samples, shape (1, 32, 12, 3)
```

See [Common Usage Patterns](core_components.md#common-usage-patterns) for more examples.

---

## 📝 Additional Resources

### Code Annotations

The codebase includes comprehensive type hints and docstrings:

- **Type Hints:** All public functions have complete type annotations
- **Docstrings:** All classes and functions include detailed documentation
- **Examples:** Many docstrings include usage examples

### Test Scripts

- **`test_inference.py`:** End-to-end inference test with minADE computation
- **`load_physical_aiavdataset.py`:** Example data loading from PhysicalAI dataset

---

## 🤝 Contributing

When contributing to the codebase:

1. **Follow Type Annotations:** Use type hints for all function signatures
2. **Write Docstrings:** Include Args, Returns, and Raises sections
3. **Add Examples:** Include usage examples where helpful
4. **Update Docs:** Update relevant documentation when adding features

---

## 📧 Support

For questions or issues:

1. Check the relevant documentation section
2. Review code comments and docstrings
3. Run `test_inference.py` to verify setup
4. Consult the [Function Reference Table](model_inference_flow.md#function-reference-table)

---

## 📄 License

Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0.

---

**Happy coding with Alpamayo-R1!** 🚗💨
