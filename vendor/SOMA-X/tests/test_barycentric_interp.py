# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from soma.geometry.barycentric_interp import BarycentricInterpolator, fabricate_tet


def _source_triangle() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return vertices, faces


def _target_from_tet(vertices: torch.Tensor, p3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bary = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float32)
    target = (
        vertices[0] * bary[0]
        + vertices[1] * bary[1]
        + vertices[2] * bary[2]
        + p3 * bary[3]
    )
    return target.unsqueeze(0), bary


def test_fabricate_tet_edge_scale_uses_local_edge_length() -> None:
    p0 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    p1 = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    p2 = np.array([[0.0, 2.0, 0.0]], dtype=np.float32)

    area_p3 = fabricate_tet(p0, p1, p2)
    edge_p3 = fabricate_tet(p0, p1, p2, normal_scale="edge")

    expected_edge_height = (2.0 + np.sqrt(8.0) + 2.0) / 3.0
    np.testing.assert_allclose(area_p3[0], [0.0, 0.0, 4.0])
    np.testing.assert_allclose(edge_p3[0], [0.0, 0.0, expected_edge_height])


def test_barycentric_interpolator_default_preserves_area_tet_behavior() -> None:
    vertices, faces = _source_triangle()
    p3 = torch.from_numpy(
        fabricate_tet(
            vertices[[0]].numpy(),
            vertices[[1]].numpy(),
            vertices[[2]].numpy(),
        )[0]
    )
    target, expected_bary = _target_from_tet(vertices, p3)

    interp = BarycentricInterpolator(vertices, faces, target)

    assert interp.tet_normal_scale == "area"
    torch.testing.assert_close(
        interp.bary_coords[0].to(dtype=torch.float32),
        expected_bary,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(interp(vertices)[0], target[0], atol=1e-5, rtol=1e-5)


def test_barycentric_interpolator_edge_tet_reconstructs_edge_scaled_target() -> None:
    vertices, faces = _source_triangle()
    p3 = torch.from_numpy(
        fabricate_tet(
            vertices[[0]].numpy(),
            vertices[[1]].numpy(),
            vertices[[2]].numpy(),
            normal_scale="edge",
        )[0]
    )
    target, expected_bary = _target_from_tet(vertices, p3)

    interp = BarycentricInterpolator(vertices, faces, target, tet_normal_scale="edge")

    assert interp.tet_normal_scale == "edge"
    torch.testing.assert_close(
        interp.bary_coords[0].to(dtype=torch.float32),
        expected_bary,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(interp(vertices)[0], target[0], atol=1e-5, rtol=1e-5)


def test_barycentric_interpolator_rejects_unknown_tet_scale() -> None:
    vertices, faces = _source_triangle()

    with pytest.raises(ValueError, match="Unsupported tet_normal_scale"):
        BarycentricInterpolator(vertices, faces, vertices[[0]], tet_normal_scale="unknown")

