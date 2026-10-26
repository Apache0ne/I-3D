#!/usr/bin/env python3
"""Experimental, CUDA 13-only installation for the ORIGINAL Hi3D pipeline.

Hi3D upstream validated Python 3.10 / PyTorch 2.5+cu124 / spconv-cu120.
This setup explicitly avoids those old CUDA kernels, instead pairing
PyTorch 2.11.0+cu130 with CUDA-13-compatible SpConv/cumm v2.3.8/v0.8.2
wheels from the auditable third-party build repository
https://github.com/fafraob/spconv-wheels (not NVIDIA/Hi3D official).
FlashAttention and Hi3D compatibility with this stack are experimental.
No model weights included. GPU inference still must be tested by the user.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HI = ROOT / "vendor" / "Hi3Dv1.0"
HUN = ROOT / "vendor" / "Hunyuan3D-2"
TORCH = ("torch==2.11.0", "torchvision==0.26.0")
SPCONV_RELEASE = "https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8"


def run(*args):
    command = [str(a) for a in args]
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/content/i3d_envs/hi3d/bin/python")
    ap.add_argument("--uv", default=shutil.which("uv") or "/usr/local/bin/uv")
    a = ap.parse_args()
    py = Path(a.python).resolve()
    if not py.is_file():
        raise FileNotFoundError(py)
    if not (HI / "pyproject.toml").exists() or not (HUN / "requirements.txt").exists():
        raise RuntimeError("I-3D vendor Hi3D and Hunyuan3D-2 source trees required.")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("Requires CUDA 13 nvcc on PATH.")
    detected = subprocess.check_output([nvcc, "--version"], text=True)
    if not re.search(r"release\s+13\.0", detected):
        raise RuntimeError("Only CUDA 13.0 compiler is supported by this setup:\n" + detected)

    uv = a.uv
    run(uv, "pip", "install", "--python", py, *TORCH,
        "--index-url", "https://download.pytorch.org/whl/cu130")
    # Match extension wheels to Python's minor ABI. Community wheel pair is
    # published for Linux x86_64 cp310-cp313, exactly version locked.
    v = subprocess.check_output([str(py), "-c",
       "import sys; print('cp%d%d' % sys.version_info[:2])"], text=True).strip()
    if v not in ("cp310", "cp311", "cp312", "cp313"):
        raise RuntimeError(f"No documented Hi3D CUDA-13 spconv wheel for {v}")
    wheels = [
        f"{SPCONV_RELEASE}/cumm_cu130-0.8.2-{v}-{v}-linux_x86_64.whl",
        f"{SPCONV_RELEASE}/spconv_cu130-2.3.8-{v}-{v}-linux_x86_64.whl",
    ]
    print("CUDA-13 third-party SpConv build provenance:",
          "https://github.com/fafraob/spconv-wheels")
    run(uv, "pip", "install", "--python", py, *wheels)
    # Unlike the upstream Hi3D inference extra, do NOT force CUDA12 spconv
    # or legacy flash-attn pins. Install original Hi3D and demo dependency only.
    run(uv, "pip", "install", "--python", py, "-e", str(HI) + "[demo]")
    run(uv, "pip", "install", "--python", py, "-r", HUN / "requirements.txt")
    run(uv, "pip", "install", "--python", py, "-e", HUN)
    run(uv, "pip", "install", "--python", py,
        "packaging", "ninja", "psutil", "setuptools", "wheel")
    # Build current flash-attn source against torch2.11/cu130. If upstream
    # changes ABI and this fails, show REAL compiler error and do not substitute
    # a CUDA12 binary or silently bypass the original Hi3D model.
    run(uv, "pip", "install", "--python", py,
        "--no-build-isolation", "flash-attn")
    run(py, "-c",
        "import torch; assert torch.version.cuda == '13.0', torch.version.cuda; "
        "import spconv.pytorch as sp; import flash_attn; "
        "print('HI3D_CUDA13_IMPORTS_OK', torch.__version__, torch.version.cuda)")
    print("Hi3D CUDA-13 experimental environment installed. Inference unverified.")


if __name__ == "__main__":
    main()
