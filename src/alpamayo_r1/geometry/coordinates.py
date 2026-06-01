# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Geometry helpers for 3D coordinate / bounding-box operations.

Currently provides :func:`xyzrot_to_corners`, which converts a center position,
SO(3) rotation matrix, and box dimensions into the 8 corner points of a 3D
bounding box. The implementation is fully batched: all leading dimensions are
treated as batch dimensions and only the trailing axes carry geometric meaning.
"""

import torch


def xyzrot_to_corners(xyz: torch.Tensor, rot: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """Get the corners of a 3d bounding box. The implementation interprets the tail dimensions as
    the meaningful ones, and everything else effectively as a batch dimension.

    Args:
        xyz: ...x3 location of the center of the bounding box
        rot: ...x3x3 orientation of the bounding box represented as a rotation matrix
        dims: ...x3 dimensions of the bounding box (length, width, height)

    Returns:
        corns: ...x8x3 corners of the bounding box. The first 4 points are the bottom
               corners and the next 4 are the top corners.
    """
    # Unit cube corners centered at the origin: first 4 rows are the bottom face
    # (z = -0.5), last 4 are the top face (z = +0.5). Scaled, rotated, translated below.
    corns = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        device=xyz.device,
        dtype=xyz.dtype,
    )

    # scale
    corns = dims.unsqueeze(-2) * corns

    # rotate
    corns = (rot.unsqueeze(-3) @ corns.unsqueeze(-1)).squeeze(-1)

    # translate
    corns = corns + xyz.unsqueeze(-2)

    return corns
