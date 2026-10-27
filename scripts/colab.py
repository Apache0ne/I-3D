#!/usr/bin/env python3
"""I-3D Colab CLI. Notebook = git clone + call this file.

All model logic remains in upstream sources and existing I-3D scripts.
Commands:
  python scripts/colab.py setup --stage all
  python scripts/colab.py prepare --image /content/person.png --forward +X
  python scripts/colab.py animate
  python scripts/colab.py pack
Separate CUDA-13 Python environments prevent torch version clobbering.
All subprocess output is streamed to Colab (never hidden).
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = Path(os.getenv("I3D_ENV_ROOT", "/content/i3d_envs"))
WORK_DEFAULT = ROOT / "outputs" / "character"
WHEEL_URL = (
    "https://github.com/alkemiik-coder/"
    "FlashAttention-2.8.3-Custom-Linux-Wheels/releases/download/"
    "FA.2.8.3-custom-linux-wheels-x86_64/"
)
FLASH_TOKEN_NAME = (
    "flash_attn-2.8.3+cu130torch2.11cxx11abiTRUE"
    "fullsm80sm90sm100sm120nvcc130-cp311-cp311-linux_x86_64.whl"
)
FLASH_TOKEN_SHA = "e30c67ecd3c0f5994d70456f87bbb862b295a3e2dd19675343083721f8d8243a"


def call(*argv, cwd=None, env=None):
    cmd = [str(x) for x in argv]
    print("\nRUN:", " ".join(cmd), flush=True)
    # Inherits stderr/stdout: errors appear in the actual cell, rather than
    # a bare CalledProcessError whose underlying message is hidden.
    code = subprocess.call(cmd, cwd=str(cwd or ROOT), env=env)
    if code:
        raise RuntimeError(f"Command exited {code}: {' '.join(cmd)}")


def require(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(f"Required I-3D artifact not produced: {path}")
    return path


def py(name):
    return ENV_ROOT / name / "bin" / "python"


def uv_tool():
    u = shutil.which("uv")
    if not u:
        call(sys.executable, "-m", "pip", "install", "uv")
        u = shutil.which("uv") or "/usr/local/bin/uv"
    if not Path(u).exists():
        raise FileNotFoundError("uv installer unavailable after pip install")
    return u


def system_dependencies():
    needed = {"blender": "blender", "git-lfs": "git-lfs", "curl": "curl",
              "ffmpeg": "ffmpeg"}
    missing = [pkg for cmd, pkg in needed.items() if not shutil.which(cmd)]
    if missing:
        call("apt-get", "update", "-qq")
        call("apt-get", "install", "-y", "-qq", *sorted(set(missing)),
             "libgl1", "libglib2.0-0")
    if shutil.which("git-lfs"):
        call("git", "lfs", "install")


def ensure_env(label, version, uv):
    p = py(label)
    if not p.exists():
        call(uv, "python", "install", version)
        call(uv, "venv", "--seed", "--python", version, p.parent.parent)
    require(p)
    return p


def setup_hi3d(uv):
    p = ensure_env("hi3d", "3.10", uv)
    call(sys.executable, ROOT / "scripts" / "install_hi3d_cuda13.py",
         "--python", p, "--uv", uv)


def setup_paint(uv):
    p = ensure_env("paint", "3.10", uv)
    call(sys.executable, ROOT / "scripts" / "install_paint_cuda13.py",
         "--python", p, "--uv", uv)
    # Download original RealESRGAN checkpoint (required by Paint2.1).
    ckpt = ROOT / "vendor" / "Hunyuan3D-2.1" / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if not ckpt.exists() or ckpt.stat().st_size < 1024:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        call("curl", "-fL", "--retry", "4",
             "https://github.com/xinntao/Real-ESRGAN/releases/download/"
             "v0.1.0/RealESRGAN_x4plus.pth", "-o", ckpt)
    require(ckpt)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def setup_tokenrig(uv):
    p = ensure_env("tokenrig", "3.11", uv)
    token = ROOT / "vendor" / "SkinTokens"
    call(uv, "pip", "install", "--python", p,
         "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0",
         "--index-url", "https://download.pytorch.org/whl/cu130")
    call(uv, "pip", "install", "--python", p, "-r", token / "requirements.txt")
    # Avoid costly/unreliable CUDA source build; use an ABI-matched, checked
    # third-party prebuilt wheel. It is NOT an official TokenRig artifact.
    dest = Path("/content/i3d_cuda13_wheels") / FLASH_TOKEN_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or _sha256(dest) != FLASH_TOKEN_SHA:
        dest.unlink(missing_ok=True)
        call("curl", "-fL", "--retry", "4",
             WHEEL_URL + FLASH_TOKEN_NAME.replace("+", "%2B"), "-o", dest)
    if _sha256(dest) != FLASH_TOKEN_SHA:
        dest.unlink(missing_ok=True)
        raise RuntimeError("Third-party TokenRig FlashAttention wheel checksum mismatch")
    call(uv, "pip", "install", "--python", p, "--no-deps", dest)
    call(p, "-c",
         "import torch, flash_attn; assert torch.version.cuda=='13.0';"
         "print('TOKENRIG_CUDA13_IMPORTS_OK',torch.__version__)")
    # Don't download dataset/train checkpoints; original download.py handles models.
    marker = token / "experiments" / "articulation_xl_quantization_256_token_4" / "grpo_1400.ckpt"
    if not marker.is_file():
        call(p, token / "download.py", "--model", cwd=token)


def setup_motion(uv, precision):
    p = ensure_env("motion", "3.12", uv)
    src = ROOT / "vendor" / "motionbricks"
    call(uv, "pip", "install", "--python", p, "-e", src)
    # NVIDIA model is an original TensorRT runtime (may retain CUDA12
    # deps even when the Hi3D/Paint model environments run Torch cu130).
    ref = src / "motionbricks" / "assets" / "human" / "soma_base_skel_minimal_no_root.usd"
    if not ref.is_file() or ref.stat().st_size < 1024:
        call(sys.executable, ROOT / "scripts" / "hydrate_motionbricks.py",
             "--precision", precision)


def setup(a):
    system_dependencies()
    u = uv_tool()
    stages = ["hi3d", "paint", "tokenrig", "motion"] if a.stage == "all" else [a.stage]
    for s in stages:
        print("\n" + "=" * 50, "\nI-3D SETUP STAGE:", s, "\n" + "=" * 50, flush=True)
        globals()["setup_" + s](u) if s != "motion" else setup_motion(u, a.precision)
    print("SETUP COMPLETE for:", ", ".join(stages), flush=True)


def command_prepare(a):
    image = require(Path(a.image).resolve())
    work = Path(a.output).resolve()
    work.mkdir(parents=True, exist_ok=True)
    geometry = work / "01_geometry.glb"
    texture = work / "02_textured.glb"
    rigged = work / "03_tokenrig.glb"
    soma = work / "04_soma_character.glb"
    reference = work / "soma_reference.json"
    blender = shutil.which("blender") or "/usr/bin/blender"
    if not geometry.is_file() or a.force:
        call(py("hi3d"), "demo/image_to_mesh.py",
             "--image", image, "--output", geometry,
             "--model", "hi3d-official/Hi3Dv1.0",
             "--resolution", a.resolution,
             "--num-inference-steps", a.steps,
             "--remove-background",
             cwd=ROOT / "vendor" / "Hi3Dv1.0")
    require(geometry)
    if not texture.is_file() or a.force:
        call(py("paint"), ROOT / "scripts" / "paint_hunyuan.py",
             "--mesh", geometry, "--image", image, "--output", texture,
             "--resolution", 512, "--views", 6,
             cwd=ROOT / "vendor" / "Hunyuan3D-2.1")
    require(texture)
    if not rigged.is_file() or a.force:
        token = ROOT / "vendor" / "SkinTokens"
        call(py("tokenrig"), token / "demo.py",
             "--input", texture, "--output", rigged, "--use_transfer", cwd=token)
    require(rigged)
    if not reference.is_file() or a.force:
        src = ROOT / "vendor" / "motionbricks" / "motionbricks" / "assets" / "human" / "soma_base_skel_minimal_no_root.usd"
        require(src)
        call(py("motion"), ROOT / "scripts" / "export_soma_reference.py",
             "--usd", src, "--output", reference)
    require(reference)
    if not soma.is_file() or a.force:
        call(blender, "-b", "--python", ROOT / "scripts" / "blender_bridge.py", "--",
             "--input", rigged, "--reference", reference,
             "--output", soma, "--forward", a.forward)
    require(soma)
    require(soma.with_suffix(".bridge.json"))
    print("\nCHARACTER_READY", soma, flush=True)
    print("Open notebook 3D Studio and save BOTH START and TARGET poses.", flush=True)


def command_animate(a):
    work = Path(a.output).resolve()
    soma = require(work / "04_soma_character.glb")
    reference = require(work / "soma_reference.json")
    bridge = require(soma.with_suffix(".bridge.json"))
    viewer = require(work / "viewer_poses.json")
    blender = shutil.which("blender") or "/usr/bin/blender"
    blend = work / "character_poses.blend"
    poses = work / "05_poses.json"
    motion = work / "06_motion.npz"
    animated = work / "07_animated.glb"
    call(blender, "-b", "--python", ROOT / "scripts" / "author_viewer_poses.py", "--",
         "--character", soma, "--poses", viewer, "--output", blend)
    require(blend)
    call(blender, blend, "-b", "--python", ROOT / "scripts" / "export_two_poses.py",
         "--", "--output", poses)
    require(poses)
    call(py("motion"), ROOT / "scripts" / "generate_motion.py",
         "--poses", poses, "--reference", reference, "--bridge", bridge,
         "--output", motion, "--precision", a.precision)
    require(motion)
    call(blender, "-b", "--python", ROOT / "scripts" / "apply_motion_blender.py", "--",
         "--character", soma, "--reference", reference, "--bridge", bridge,
         "--motion", motion, "--output", animated)
    require(animated)
    print("\nANIMATION_READY", animated, flush=True)


def pack(a):
    work = Path(a.output).resolve()
    work.mkdir(parents=True, exist_ok=True)
    bundle = work / "I3D_output_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(work.iterdir()):
            if p.is_file() and p != bundle and p.suffix.lower() in (
                ".glb", ".png", ".jpg", ".webp", ".json", ".npz", ".blend", ".mp4", ".webm"
            ):
                z.write(p, arcname=p.name)
    require(bundle)
    print("BUNDLE_READY", bundle, flush=True)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("setup", help="Install real model dependencies using original scripts")
    s.add_argument("--stage", choices=["all", "hi3d", "paint", "tokenrig", "motion"],
                   default="all")
    s.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    prep = sub.add_parser("prepare", help="Generate the same textured and SOMA-rigged human")
    prep.add_argument("--image", required=True)
    prep.add_argument("--output", default=str(WORK_DEFAULT))
    prep.add_argument("--forward", choices=["+X", "-X", "+Y", "-Y"], default="+X")
    prep.add_argument("--resolution", type=int, choices=[512, 1024, 1536], default=512)
    prep.add_argument("--steps", type=int, default=25)
    prep.add_argument("--force", action="store_true", help="Re-run all inference stages")
    anim = sub.add_parser("animate", help="Consume START/TARGET poses from Colab viewer")
    anim.add_argument("--output", default=str(WORK_DEFAULT))
    anim.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    zip_p = sub.add_parser("pack")
    zip_p.add_argument("--output", default=str(WORK_DEFAULT))
    args = p.parse_args()
    {"setup":setup,"prepare":command_prepare,"animate":command_animate,"pack":pack}[args.action](args)


if __name__ == "__main__":
    cli()
