# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap test preflights for optional identity-model assets."""

from importlib.util import find_spec
from pathlib import Path


def _missing(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if not path.is_file()]


def _missing_required(label: str, paths: list[Path]) -> str | None:
    missing = _missing(paths)
    if missing:
        return f"Missing optional {label} assets: {missing}"
    return None


def _missing_smpl_family_model(data_root: Path, model_type: str, gender: str) -> str | None:
    model_name = model_type.upper()
    model_dir = data_root / model_name
    candidates = [
        model_dir / f"{model_name}_{gender.upper()}.npz",
        model_dir / f"{model_name}_{gender.upper()}.pkl",
    ]
    if not any(path.is_file() for path in candidates):
        candidate_strings = [str(path) for path in candidates]
        return f"Missing optional {model_name} model asset; expected one of {candidate_strings}"
    return _missing_required(
        model_name,
        [
            model_dir / "base_body.obj",
            model_dir / "SOMA_wrap.obj",
        ],
    )


def body_identity_skip_reason(
    data_root: str | Path,
    identity_model_type: str,
    *,
    lod: str = "mid",
    gender: str = "neutral",
) -> str | None:
    """Return a skip reason when optional body identity assets are already absent."""
    root = Path(data_root)
    identity_model_type = identity_model_type.lower()

    if identity_model_type == "soma":
        return None
    if identity_model_type == "mhr":
        mhr_dir = root / "MHR"
        mhr_lod = "lod6" if lod == "low" else "lod1"
        return _missing_required(
            "MHR",
            [
                mhr_dir / f"mhr_model_{mhr_lod}.pt",
                mhr_dir / f"base_body_{mhr_lod}.obj",
                mhr_dir / "SOMA_wrap_lod1.obj",
            ],
        )
    if identity_model_type == "anny":
        if find_spec("anny") is None:
            return "Missing optional dependency: anny"
        return _missing_required(
            "Anny",
            [
                root / "Anny" / "base_body.obj",
                root / "Anny" / "SOMA_wrap.obj",
            ],
        )
    if identity_model_type in {"smpl", "smplh", "smplx"}:
        return _missing_smpl_family_model(root, identity_model_type, gender)
    if identity_model_type == "garment":
        return _missing_required(
            "GarmentMeasurements",
            [
                root / "GarmentMeasurements" / "point.npz",
                root / "GarmentMeasurements" / "mean.obj",
                root / "GarmentMeasurements" / "SOMA_wrap.obj",
            ],
        )
    return None


def hand_identity_skip_reason(
    data_root: str | Path,
    identity_model_type: str,
    *,
    hand_type: str,
) -> str | None:
    """Return a skip reason when optional hand identity assets are already absent."""
    root = Path(data_root)
    identity_model_type = identity_model_type.lower()

    if identity_model_type == "soma":
        return None
    if identity_model_type == "mano":
        model_name = identity_model_type.upper()
        model_dir = root / model_name
        return _missing_required(
            model_name,
            [
                model_dir / f"{model_name}_{hand_type.upper()}.pkl",
                model_dir / f"base_hand_{hand_type}.obj",
                model_dir / f"SOMA_wrap_{hand_type}.obj",
            ],
        )
    if identity_model_type == "mhr":
        mhr_dir = root / "MHR"
        return _missing_required(
            "MHR hand",
            [
                mhr_dir / "mhr_model_lod1.pt",
                mhr_dir / "base_body_lod1.obj",
                mhr_dir / "SOMA_wrap_lod1.obj",
            ],
        )
    return None
