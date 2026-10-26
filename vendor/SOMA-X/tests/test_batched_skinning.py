import pytest
import torch

from soma.geometry.batched_skinning import BatchedSkinning, FKTopology


def _make_dense_skinning(batch_size: int = 1) -> BatchedSkinning:
    joint_parent_ids = [0, 0]
    skinning_weights = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    bind_world = (
        torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4).expand(batch_size, 2, 4, 4).clone()
    )
    bind_world[:, 1, 0, 3] = torch.linspace(1.0, 1.5, batch_size)
    bind_shapes = (
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.75, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        .reshape(1, 3, 3)
        .expand(batch_size, 3, 3)
        .clone()
    )
    bind_shapes[:, 1, 0] = bind_world[:, 1, 0, 3]
    bind_shapes[:, 2, 0] = bind_world[:, 1, 0, 3] * 0.75

    return BatchedSkinning(
        joint_parent_ids=joint_parent_ids,
        skinning_weights=skinning_weights,
        bind_world_transforms=bind_world,
        bind_shapes=bind_shapes,
        mode="dense",
        global_translation_joint_idx=0,
    )


def test_forward_kinematics_and_linear_blend_skinning_match_pose_wrapper():
    skinning = _make_dense_skinning()
    rotations = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).expand(2, 2, 3, 3).clone()
    rotations[1, 0] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    translations = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )

    world_transforms = skinning.forward_kinematics(
        rotations,
        global_translation=translations,
    )
    vertices = skinning.linear_blend_skinning(world_transforms)
    pose_vertices, pose_world_transforms = skinning.pose(
        rotations,
        global_translation=translations,
        return_transforms=True,
    )

    torch.testing.assert_close(world_transforms, pose_world_transforms)
    torch.testing.assert_close(vertices, pose_vertices)


def test_forward_kinematics_expands_one_pose_to_batched_bind_shapes():
    skinning = _make_dense_skinning(batch_size=2)
    rotations = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).expand(1, 2, 3, 3).clone()
    translation = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    world_transforms = skinning.forward_kinematics(
        rotations,
        global_translation=translation,
    )
    vertices = skinning.linear_blend_skinning(world_transforms)
    pose_vertices, pose_world_transforms = skinning.pose(
        rotations,
        global_translation=translation,
        return_transforms=True,
    )

    assert world_transforms.shape == (2, 2, 4, 4)
    assert vertices.shape == (2, 3, 3)
    torch.testing.assert_close(world_transforms, pose_world_transforms)
    torch.testing.assert_close(vertices, pose_vertices)


def _make_source_target_skinning(batch_size: int = 1) -> BatchedSkinning:
    joint_parent_ids = [0, 0, 1]
    skinning_weights = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    bind_world = (
        torch.eye(4, dtype=torch.float32)
        .reshape(1, 1, 4, 4)
        .expand(
            batch_size,
            3,
            4,
            4,
        )
        .clone()
    )
    bind_world[:, 1, 0, 3] = 1.0
    bind_world[:, 2, 0, 3] = 3.0
    bind_shapes = torch.zeros(batch_size, 2, 3, dtype=torch.float32)

    return BatchedSkinning(
        joint_parent_ids=joint_parent_ids,
        skinning_weights=skinning_weights,
        bind_world_transforms=bind_world,
        bind_shapes=bind_shapes,
        mode="dense",
        global_translation_joint_idx=0,
        source_fk=FKTopology(
            parent_ids=[0, 0],
            target_joint_indices=torch.tensor([0, 2]),
            global_translation_joint_idx=0,
        ),
    )


def test_forward_source_kinematics_matches_standalone_public_fk():
    skinning = _make_source_target_skinning()
    source_bind_world = skinning.bind_world_transforms[:, [0, 2]]
    reference = BatchedSkinning(
        joint_parent_ids=[0, 0],
        skinning_weights=torch.zeros(2, 2, dtype=torch.float32),
        bind_world_transforms=source_bind_world,
        bind_shapes=skinning.bind_shapes,
        mode="dense",
        global_translation_joint_idx=0,
    )
    rotations = (
        torch.eye(3, dtype=torch.float32)
        .reshape(1, 1, 3, 3)
        .expand(
            2,
            2,
            3,
            3,
        )
        .clone()
    )
    rotations[1, 0] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    translations = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    actual = skinning.forward_source_kinematics(
        rotations,
        global_translation=translations,
        absolute_pose=True,
    )
    expected = reference.forward_kinematics(
        rotations,
        global_translation=translations,
        absolute_pose=True,
    )

    torch.testing.assert_close(actual, expected)


def test_rebind_refreshes_source_bind_state_from_target_subset():
    skinning = _make_source_target_skinning()
    new_bind_world = skinning.bind_world_transforms.clone()
    new_bind_world[:, 2, 0, 3] = 4.0

    skinning.rebind(new_bind_world, skinning.bind_shapes)

    torch.testing.assert_close(
        skinning.source_bind_world_transforms[:, 1],
        new_bind_world[:, 2],
    )
    torch.testing.assert_close(
        skinning.source_local_translations[:, 1],
        torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32),
    )


def test_expand_source_world_transforms_passes_target_lbs_state():
    skinning = _make_source_target_skinning()
    rotations = (
        torch.eye(3, dtype=torch.float32)
        .reshape(1, 1, 3, 3)
        .expand(
            1,
            2,
            3,
            3,
        )
    )
    source_world = skinning.forward_source_kinematics(rotations, absolute_pose=True)
    expected = (
        torch.eye(4, dtype=torch.float32)
        .reshape(1, 1, 4, 4)
        .expand(
            1,
            3,
            4,
            4,
        )
        .clone()
    )
    seen = {}

    def expander(**kwargs):
        seen.update(kwargs)
        return expected

    actual = skinning.expand_source_world_transforms(
        source_rotations=rotations,
        source_world_transforms=source_world,
        transform_expander=expander,
    )

    assert actual is expected
    assert seen["target_local_rotations"] is skinning.local_rotations
    assert seen["target_local_translations"] is skinning.local_translations
    assert seen["target_joint_count"] == skinning.num_joints
    torch.testing.assert_close(seen["source_rotations"], rotations)
    torch.testing.assert_close(seen["source_world_transforms"], source_world)


def test_source_fk_topology_requires_mapping_or_explicit_bind_transforms():
    skinning = _make_dense_skinning()

    with pytest.raises(ValueError, match="target_joint_indices"):
        BatchedSkinning(
            joint_parent_ids=skinning.joint_parent_ids,
            skinning_weights=skinning.skinning_weights,
            bind_world_transforms=skinning.bind_world_transforms,
            bind_shapes=skinning.bind_shapes,
            mode="dense",
            source_fk=FKTopology(parent_ids=[0, 0]),
        )
