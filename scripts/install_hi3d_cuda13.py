#!/usr/bin/env python3
"""I-3D Hi3D/Hunyuan2 CUDA-13 setup with pinned binary wheels.

The original Hi3D release officially tested CUDA 12.4. This is an experimental
CUDA 13 compatibility environment. The two third-party wheel repositories
provide source/provenance; downloads are pinned to GitHub release SHA256s.
All upstream project source files are unchanged.

This installer prints stdout AND stderr for each operation. It stops at the
specific failing command. It never silently uses CUDA-12 packages.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HI3D = ROOT / "vendor" / "Hi3Dv1.0"
HUNYUAN = ROOT / "vendor" / "Hunyuan3D-2"
SP_URL = "https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8"
FA_URL = "https://github.com/alkemiik-coder/FlashAttention-2.8.3-Custom-Linux-Wheels/releases/download/FA.2.8.3-custom-linux-wheels-x86_64"
ARTIFACTS = (
    ("cumm_cu130-0.8.2-cp310-cp310-linux_x86_64.whl",
     SP_URL, "cb381fc83d3a1055f9b1c26fb42e0597e69c2f889524de9b6bece29ae80e8820"),
    ("spconv_cu130-2.3.8-cp310-cp310-linux_x86_64.whl",
     SP_URL, "bdf056d38e6ce9459a014c1871b4e9c0afa44dfe3d45558fe08b8c85011a46be"),
    ("flash_attn-2.8.3+cu130torch2.11cxx11abiTRUEfullsm80sm90sm100sm120nvcc130-cp310-cp310-linux_x86_64.whl",
     FA_URL, "450580a11cc637c53bc619b0c3a926879e2172341df2990a558d44033c95c956"),
)

def step(name: str, argv, *, env=None):
    command = [str(x) for x in argv]
    print(f"\n{'='*70}\n{name}\nRUN: {' '.join(command)}\n{'='*70}", flush=True)
    history = []
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               bufsize=1, env=env)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        history.append(line)
        if len(history) > 180:
            history.pop(0)
    status = process.wait()
    if status:
        raise RuntimeError(f"{name} FAILED (exit {status}). "
                           "The actual error is printed above.\n" +
                           "".join(history[-65:]))
    print(name, "DONE", flush=True)

def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def download_and_verify(filename, base, sha, outdir):
    path = outdir / filename
    if path.exists() and checksum(path) != sha:
        path.unlink()
    if not path.exists():
        # '+' in wheel build tag needs encoding in GitHub's release URL.
        url = f"{base}/{filename.replace('+', '%2B')}"
        step(f"Download {filename}", [
            "curl", "--fail", "--location", "--retry", "4",
            "--retry-all-errors", "--connect-timeout", "30",
            "-o", path, url
        ])
    got = checksum(path)
    if got != sha:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {filename}: {got} != {sha}")
    print(f"SHA256 verified: {filename} ({path.stat().st_size:,} bytes)")
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/content/i3d_envs/hi3d/bin/python")
    parser.add_argument("--uv", default=shutil.which("uv") or "/usr/local/bin/uv")
    args = parser.parse_args()
    py = Path(args.python)
    if not py.is_file():
        raise FileNotFoundError(f"Missing Hi3D environment: {py}")
    if not (HI3D / "pyproject.toml").is_file() or not (HUNYUAN / "setup.py").is_file():
        raise FileNotFoundError(f"Missing I-3D vendored source under {ROOT/'vendor'}")

    version = subprocess.check_output(
        [str(py), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        text=True).strip()
    if version != "3.10":
        raise RuntimeError(f"Hi3D expects Python 3.10, got {version}")

    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("CUDA nvcc compiler not on PATH. Select CUDA 13 Colab runtime.")
    nvcc_info = subprocess.check_output([nvcc, "--version"], text=True)
    if not re.search(r"release\s+13\.0", nvcc_info):
        raise RuntimeError("Expected CUDA 13.0 compiler, got:\n" + nvcc_info)
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("No nvidia-smi: enable GPU in Colab.")

    uv = args.uv
    step("Install matching PyTorch 2.11 CUDA13", [
        uv, "pip", "install", "--python", py,
        "torch==2.11.0", "torchvision==0.26.0",
        "--index-url", "https://download.pytorch.org/whl/cu130",
    ])
    print(subprocess.check_output([
        str(py), "-c",
        "import torch; "
        "assert torch.version.cuda=='13.0',torch.version.cuda; "
        "assert torch._C._GLIBCXX_USE_CXX11_ABI, 'FlashAttention wheel needs CXX11 ABI TRUE'; "
        "print('Torch',torch.__version__,'CUDA',torch.version.cuda,'GPU',"
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"],
        text=True))
    
    work = Path("/content/i3d_cuda13_wheels")
    work.mkdir(parents=True, exist_ok=True)
    spconv, cumm = None, None
    wheel_files = [download_and_verify(name, base, sha, work)
                   for name, base, sha in ARTIFACTS]
    # First install binary dependencies together, preserving the matched pair.
    step("Install SHA256-verified CUDA13 SpConv and cumm", [
        uv, "pip", "install", "--python", py, *wheel_files[:2]
    ])
    step("Install SHA256-verified CUDA13 FlashAttention 2.8.3", [
        uv, "pip", "install", "--python", py, "--no-deps", wheel_files[2]
    ])

    constraints = Path("/content/i3d_hi3d_cuda13_constraints.txt")
    constraints.write_text(
        "torch==2.11.0\n"
        "torchvision==0.26.0\n"
        "numpy==1.26.4\n"
        # Hunyuan's setup.py requires >=4.48; Hi3D requires >=4.46,<5.
        "transformers==4.49.0\n"
        "huggingface-hub==0.30.2\n"
        "rembg==2.0.61\n",
        encoding="utf-8"
    )
    # Original Hi3D project has the correct package/source. Its [inference]
    # extra would force spconv-cu120==2.3.6; do NOT install that extra here.
    step("Install ORIGINAL Hi3D source + basic dependencies", [
        uv, "pip", "install", "--python", py, "-c", constraints,
        "-e", HI3D, "accelerate>=0.31.0,<2", "triton>=3.1.0",
        "onnxruntime", "scikit-image==0.24.0", "pyyaml"
    ])
    # Hunyuan shape library's setup.py requires transformers>=4.48, hence
    # transformers 4.49.0 in constraints; this is NOT upstream validated.
    step("Install ORIGINAL Hunyuan3D-2mini coarse-stage source", [
        uv, "pip", "install", "--python", py, "-c", constraints,
        "-e", HUNYUAN
    ])
    step("Install background removal model dependencies", [
        uv, "pip", "install", "--python", py, "-c", constraints,
        "rembg==2.0.61", "onnxruntime"
    ])
    step("Verify CUDA13 Hi3D + Hunyuan imports", [
        py, "-c",
        "import torch,spconv.pytorch,flash_attn,transformers,numpy; "
        "from sparc3d import Sparc3DPipeline; "
        "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; "
        "assert torch.version.cuda=='13.0'; "
        "print('HI3D_CUDA13_IMPORTS_READY',torch.__version__,"
        "transformers.__version__,numpy.__version__)"
    ])
    print("\nHI3D_CUDA13_INSTALL_READY — imports passed, "
          "full pretrained model inference still requires a GPU run.", flush=True)

if __name__ == "__main__":
    main()
