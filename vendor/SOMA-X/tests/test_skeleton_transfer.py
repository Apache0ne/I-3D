# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from soma.geometry.skeleton_transfer import SkeletonTransfer


class TestSkeletonTransfer(unittest.TestCase):
    """Unit tests for SkeletonTransfer behavior without repository assets."""

    def _make_skeleton_transfer(self, device="cpu", *, use_sparse_rbf_matrix=False):
        torch.manual_seed(1234)
        num_joints, num_vertices = 5, 20
        # Pass joint_parent_ids as a tensor to exercise device movement paths.
        joint_parent_ids = torch.tensor([0, 0, 1, 2, 3])
        bind_world_transforms = torch.eye(4).unsqueeze(0).repeat(num_joints, 1, 1)
        bind_shape = torch.randn(num_vertices, 3)
        skinning_weights = torch.rand(num_vertices, num_joints)
        skinning_weights /= skinning_weights.sum(dim=1, keepdim=True)
        return SkeletonTransfer(
            joint_parent_ids.to(device),
            bind_world_transforms.to(device),
            bind_shape.to(device),
            skinning_weights.to(device),
            use_warp_for_rotations=False,
            use_sparse_rbf_matrix=use_sparse_rbf_matrix,
        )

    def test_init_with_tensor_joint_parent_ids(self):
        """joint_parent_ids passed as a CPU tensor must not cause device errors."""
        transfer = self._make_skeleton_transfer("cpu")
        self.assertIsNotNone(transfer.regressor_mask)

    def test_gpu_to_cpu_roundtrip(self):
        """Simulates DDP teardown: SkeletonTransfer on GPU moved back to CPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        transfer = self._make_skeleton_transfer("cpu")
        transfer.to("cuda")
        try:
            transfer.cpu()
        except RuntimeError as e:
            self.fail(f"Moving SkeletonTransfer from GPU to CPU failed: {e}")

    def test_cpu_to_gpu_roundtrip(self):
        """Moving from CPU to GPU and back must leave all buffers on CPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        transfer = self._make_skeleton_transfer("cpu")
        transfer.cuda()
        transfer.cpu()
        for name, buf in transfer.named_buffers():
            if buf is not None:
                self.assertEqual(
                    buf.device.type, "cpu", f"Buffer {name} not on CPU after round-trip"
                )

    def test_update_bind_refreshes_regressors(self):
        """Changing identity bind data should match constructing a fresh transfer."""
        torch.manual_seed(42)
        num_joints, num_vertices = 5, 20
        joint_parent_ids = torch.tensor([0, 0, 1, 2, 3])
        bind_world = torch.eye(4).unsqueeze(0).repeat(num_joints, 1, 1)
        bind_shape = torch.randn(num_vertices, 3)
        skinning_weights = torch.rand(num_vertices, num_joints)
        skinning_weights /= skinning_weights.sum(dim=1, keepdim=True)

        transfer = SkeletonTransfer(
            joint_parent_ids,
            bind_world,
            bind_shape,
            skinning_weights,
            use_warp_for_rotations=False,
            use_sparse_rbf_matrix=True,
        )

        new_bind_world = bind_world.clone()
        new_bind_world[:, :3, 3] = torch.randn(num_joints, 3)
        new_bind_shape = bind_shape * 1.4 + torch.tensor([0.2, -0.1, 0.3])
        target_shape = new_bind_shape + 0.01 * torch.randn_like(new_bind_shape)

        transfer.update_bind(new_bind_world, new_bind_shape)
        fresh = SkeletonTransfer(
            joint_parent_ids,
            new_bind_world,
            new_bind_shape,
            skinning_weights,
            use_warp_for_rotations=False,
            use_sparse_rbf_matrix=True,
        )

        self.assertTrue(
            torch.allclose(
                transfer.fit_joint_positions(target_shape),
                fresh.fit_joint_positions(target_shape),
                atol=1e-5,
                rtol=1e-5,
            )
        )
