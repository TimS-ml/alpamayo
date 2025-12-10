# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper utilities for message creation, processor initialization, and device transfer.

This module provides utility functions for preparing inputs to the Alpamayo-R1 model,
including creating chat messages with images and trajectory placeholders, initializing
the vision-language processor, and recursively transferring data to specified devices.
"""

from transformers import AutoProcessor, AutoTokenizer

from typing import Any

import torch
import collections.abc

MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def create_message(frames: torch.Tensor) -> list[dict[str, Any]]:
    """Construct a chat message with multi-camera images and trajectory placeholders.

    Creates a structured conversation format for the VLM with:
    - System prompt defining the model's role as a driving assistant
    - User message containing multiple camera frames and trajectory history placeholder tokens
    - Assistant message starting with Chain-of-Thought (CoT) reasoning marker

    Args:
        frames: Multi-camera image tensor of shape (N, C, H, W) where N is the number
            of camera views, C is channels (3 for RGB), H is height, W is width.

    Returns:
        A list of dictionaries representing the chat conversation, where each dictionary
        contains 'role' and 'content' keys formatted for the VLM processor.

    Raises:
        AssertionError: If frames does not have 4 dimensions.
    """
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"

    # NOTE: we expand the padding tokens to match training, so we can directly apply native processor from VLM.
    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def get_processor(tokenizer: AutoTokenizer) -> AutoProcessor:
    """Initialize the vision-language processor with custom tokenizer.

    Creates a Qwen3-VL-2B processor with specific image resolution constraints
    and replaces its tokenizer with the provided custom tokenizer (which includes
    additional special tokens for trajectory and reasoning).

    Args:
        tokenizer: The custom tokenizer containing special tokens for trajectory
            history, future trajectory, and chain-of-thought reasoning markers.

    Returns:
        An AutoProcessor instance configured with the custom tokenizer and
        image processing parameters (min_pixels=163840, max_pixels=196608).
    """
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }

    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively transfer data structures to specified device and dtype.

    Traverses nested data structures (tensors, dicts, lists) and moves all PyTorch
    tensors to the specified device and/or dtype. Non-tensor data is left unchanged.

    Args:
        data: Input data to transfer. Can be a torch.Tensor, dict, list, or any
            nested combination of these types. Primitive types (str, bytes, numbers)
            are returned as-is.
        device: Target device for tensors (e.g., 'cuda', 'cpu', torch.device('cuda:0')).
            If None, tensors remain on their current device.
        dtype: Target dtype for tensors (e.g., torch.float32, torch.float16).
            If None, tensors retain their current dtype.

    Returns:
        The data structure with all tensors moved to the specified device/dtype.
        The structure and non-tensor values are preserved.

    Examples:
        >>> batch = {"images": torch.randn(4, 3, 224, 224), "labels": [0, 1, 2, 3]}
        >>> batch_gpu = to_device(batch, device='cuda', dtype=torch.float16)
    """
    if isinstance(data, torch.Tensor):
        data = data.to(
            device=device,
            dtype=dtype,
        )
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device=device, dtype=dtype) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    else:
        return data
