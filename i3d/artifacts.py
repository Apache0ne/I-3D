"""Pure CPU artifact contracts for I-3D pipeline stages.

These checks deliberately avoid GPU, Blender, Torch and model imports.
A successful subprocess exit or a nonempty file alone is NOT proof of a valid GLB.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def validate_glb(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing GLB: {path}")
    size = path.stat().st_size
    if size < 20:
        raise ValueError(f"GLB is empty/truncated: {path}")
    with path.open("rb") as f:
        header = f.read(20)
        magic, version, declared = struct.unpack("<4sII", header[:12])
        length, chunk_type = struct.unpack("<I4s", header[12:])
        if magic != b"glTF" or version != 2:
            raise ValueError(f"Expected binary glTF v2, not OBJ/text: {path}")
        if declared != size or length < 2 or 20 + length > size or chunk_type != b"JSON":
            raise ValueError(f"Corrupt GLB header or missing JSON chunk: {path}")
        metadata = f.read(length)
    try:
        document = json.loads(metadata)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid glTF JSON chunk: {path}") from exc
    if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
        raise ValueError(f"Invalid glTF v2 asset descriptor: {path}")
    if not document.get("meshes"):
        raise ValueError(f"GLB contains no meshes: {path}")
    return path


def obj_intermediate_for_glb(destination: str | Path) -> Path:
    """Hunyuan Paint's save_mesh ALWAYS writes OBJ; it then exports .obj→.glb."""
    target = Path(destination)
    if target.suffix.lower() != ".glb":
        raise ValueError(f"Final Paint output must have a .glb extension: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_suffix(".obj")


def _stamp_path(asset: Path) -> Path:
    return asset.with_name(asset.name + ".i3d.json")


def cache_valid(path: str | Path, params: dict, *, glb: bool = True) -> bool:
    """Only reuse files created under the same inputs/options and unchanged bytes."""
    path = Path(path)
    stamp = _stamp_path(path)
    if not path.is_file() or not stamp.is_file():
        return False
    try:
        data = json.loads(stamp.read_text(encoding="utf-8"))
        if data.get("version") != 1 or data.get("params") != params:
            return False
        if path.stat().st_size != data.get("bytes"):
            return False
        if glb:
            validate_glb(path)
        elif path.stat().st_size == 0:
            return False
        return sha256(path) == data.get("sha256")
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def write_stamp(path: str | Path, params: dict, *, glb: bool = True) -> None:
    path = Path(path)
    if glb:
        validate_glb(path)
    elif not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Empty/missing stage artifact: {path}")
    data = {
        "version": 1,
        "params": params,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    stamp = _stamp_path(path)
    temporary = stamp.with_suffix(stamp.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, stamp)


def required_binary_asset(path: str | Path, *, min_size: int = 1024) -> Path:
    path = Path(path)
    if not path.is_file() or path.stat().st_size < min_size:
        raise FileNotFoundError(f"Missing or undersized checkpoint/asset: {path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer instead of real asset: {path}")
    return path
