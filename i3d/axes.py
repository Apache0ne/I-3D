"""Coordinate conventions used by MotionBricks and I-3D.

SOMA: right handed, Y up, XZ floor.  I-3D/Blender: right handed,
Z up, XY floor. NVIDIA's SpaceConverter(MUJOCO) is Rx(+90 degrees).
Do not use a mirrored or negative-scale matrix to 'fix' facing.
"""
from __future__ import annotations
import math
import numpy as np

SOMA_TO_BLENDER = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=np.float64)
BLENDER_TO_SOMA = SOMA_TO_BLENDER.T
UP = np.array([0., 0., 1.])
FORWARD = {'+X': np.array([1., 0., 0.]), '-X': np.array([-1., 0., 0.]),
           '+Y': np.array([0., 1., 0.]), '-Y': np.array([0., -1., 0.])}


def check_rotation(rotation: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError('Rotation must be a finite 3x3 matrix')
    if not np.allclose(r.T @ r, np.eye(3), atol=tolerance):
        raise ValueError('Non-orthonormal rotation or unintended scale')
    determinant = float(np.linalg.det(r))
    if abs(determinant - 1.) > tolerance:
        raise ValueError(f'Mirrored/left-handed matrix rejected: det={determinant:.6f}')
    return r


def facing_rotation(axis: str) -> np.ndarray:
    """Rotate canonical +X facing to an explicitly chosen horizontal direction."""
    if axis not in FORWARD:
        raise ValueError(f'Choose character forward from {tuple(FORWARD)}')
    f = FORWARD[axis]
    left = np.cross(UP, f)
    # columns: forward(+X), left(+Y), up(+Z). This is right-handed.
    return check_rotation(np.column_stack((f, left, UP)))


def transform_points(points, rotation, scale=1., translation=None):
    r = check_rotation(rotation)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Only positive uniform scale is allowed')
    p = np.asarray(points, dtype=np.float64)
    translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError('Invalid translation')
    return scale * (p @ r.T) + translation


def soma_to_blender_points(points):
    return transform_points(points, SOMA_TO_BLENDER)


def soma_to_blender_rotations(rotations):
    """Convert world-space orientations; local parent rotations do NOT use this conjugation."""
    r = np.asarray(rotations, dtype=np.float64)
    return SOMA_TO_BLENDER @ r @ SOMA_TO_BLENDER.T


def verify_laterality(joints: dict[str, np.ndarray], forward_axis: str, *, min_separation=0.01):
    """Enforce semantic left/right positions rather than silently mirroring a pose."""
    left_direction = np.cross(UP, FORWARD[forward_axis])
    pairs = [('LeftArm', 'RightArm'), ('LeftLeg', 'RightLeg'),
             ('LeftHand', 'RightHand'), ('LeftFoot', 'RightFoot')]
    checked = 0
    for left, right in pairs:
        if left not in joints or right not in joints:
            continue
        sep = float(np.dot(np.asarray(joints[left]) - np.asarray(joints[right]), left_direction))
        if sep <= min_separation:
            raise ValueError(f'Left/right inversion or collapsed skeleton: {left}/{right}, separation={sep:.4f} m')
        checked += 1
    if not checked:
        raise ValueError('No left/right joint pair found to verify handedness')
    return checked


def fit_similarity_without_reflection(source, destination, *, max_relative_rms=0.08):
    """Kabsch/Umeyama least squares: destination ~= scale * R @ source + t.

    Reject reflection, coplanar/degenerate anchors and poorly aligned human landmarks.
    An RMS tolerance relative to destination height prevents plausible-looking wrong rigs.
    """
    x, y = np.asarray(source, dtype=float), np.asarray(destination, dtype=float)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 3 or len(x) < 4:
        raise ValueError('Require four or more matching 3D points (N,3)')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite landmarks')
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    xs, ys = x - xc, y - yc
    if np.linalg.matrix_rank(xs, tol=1e-7) < 3 or np.linalg.matrix_rank(ys, tol=1e-7) < 3:
        raise ValueError('Landmark geometry is degenerate (requires 3D spread)')
    u, singular, vt = np.linalg.svd(xs.T @ ys)
    sign = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = check_rotation(vt.T @ sign @ u.T)
    scale = float(np.trace(np.diag(singular) @ sign) / np.sum(xs ** 2))
    translation = yc - scale * (rotation @ xc)
    aligned = transform_points(x, rotation, scale, translation)
    rms = float(np.sqrt(np.mean(np.sum((aligned - y)**2, axis=1))))
    body_height = float(np.ptp(y[:, 2]))
    if body_height < 0.1 or rms > max_relative_rms * body_height:
        raise ValueError(f'Fit is implausible: RMS={rms:.4f}m, height={body_height:.4f}m')
    return {'rotation': rotation, 'scale': scale, 'translation': translation, 'rms': rms}
