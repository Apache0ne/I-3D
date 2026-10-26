#!/usr/bin/env python3
"""I-3D: Hunyuan3D-Paint v2.1 installation for Colab CUDA 13.0.

Keep the original Tencent source intact. Builds its original custom_rasterizer
using the CUDA-13 nvcc that is already on the host, paired with PyTorch cu130.
Requires Python 3.10 in the dedicated paint virtual environment.

CUDA/PyTorch ABI and Tencent inference compatibility are checked as far as
possible here, but end-to-end model inference requires a real Colab GPU run.
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
PAINT = ROOT / "vendor" / "Hunyuan3D-2.1"
TORCH = ("torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0")
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu130"


def run(args, *, env=None, cwd=None):
    cmd = list(map(str, args))
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/content/i3d_envs/paint/bin/python")
    ap.add_argument("--uv", default=shutil.which("uv") or "/usr/local/bin/uv")
    ap.add_argument("--skip-requirements", action="store_true",
                    help="Reuse already-installed original Paint requirements.")
    ap.add_argument("--skip-torch", action="store_true",
                    help="Reuse cu130 Torch. Version is still verified before build.")
    a = ap.parse_args()

    py = Path(a.python).resolve()
    if not py.is_file():
        raise FileNotFoundError(f"Missing paint virtual environment: {py}")
    if not (PAINT / "requirements.txt").is_file():
        raise FileNotFoundError(f"Missing original Hunyuan3D-Paint source: {PAINT}")
    uv = a.uv

    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found. CUDA 13.0 compiler is required on PATH.")
    nvcc_info = subprocess.check_output([nvcc, "--version"], text=True)
    m = re.search(r"release\s+(\d+)\.(\d+)", nvcc_info)
    if not m or m.group(1) != "13" or m.group(2) != "0":
        raise RuntimeError(
            "This notebook requires CUDA 13.0 nvcc. Detected:\n" + nvcc_info +
            "\nSelect a matching CUDA-13 Colab runtime; no CUDA-12 fallback."
        )

    if not a.skip_torch:
        run([uv, "pip", "install", "--python", py, *TORCH,
             "--index-url", CUDA_WHEEL_INDEX])

    # Never overwrite the original vendor requirements or remove its licenses.
    if not a.skip_requirements:
        original = (PAINT / "requirements.txt").read_text(encoding="utf-8")
        filtered = []
        skipped = []
        for line in original.splitlines():
            stripped = line.strip().lower().replace("_", "-")
            if stripped.startswith(("--extra-index-url", "--index-url")) or \
               stripped.startswith("bpy==4.0") or stripped == "deepspeed" or \
               stripped.startswith("cupy-cuda12x"):
                skipped.append(line.strip())
                continue
            filtered.append(line)
        req = py.parent.parent / "paint_colab_cuda13_inference_requirements.txt"
        req.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        print("Original requirements exclusions:", skipped)
        run([uv, "pip", "install", "--python", py, "-r", req,
             "--index-url", "https://pypi.org/simple"])
        # CuPy must match CUDA 13; never mix cupy-cuda12x here.\n        run([uv, "pip", "install", "--python", py, "cupy-cuda13x==13.6.0",\n             "--index-url", "https://pypi.org/simple"])\n        # bpy 4.0 wheel is archived on Blender\'s original package index.
        run([uv, "pip", "install", "--python", py, "bpy==4.0.0",
             "--extra-index-url", "https://download.blender.org/pypi/"])

    run([uv, "pip", "install", "--python", py, "setuptools", "wheel", "ninja"])

    diagnostics = subprocess.check_output(
        [str(py), "-c",
         "import torch, sys; "
         "print(sys.version_info[:2]); "
         "print(torch.__version__); "
         "print(torch.version.cuda); "
         "print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NO_GPU')"],
        text=True,
    ).strip().splitlines()
    print("Paint environment: ", diagnostics)
    if len(diagnostics) < 4 or diagnostics[2] != "13.0":
        raise RuntimeError("Expected PyTorch compiled for CUDA 13.0; got " +
                           repr(diagnostics))
    if diagnostics[3] == "NO_GPU":
        raise RuntimeError("Cannot build CUDA rasterizer without attached GPU.")

    env = dict(os.environ)
    cuda_home = Path(nvcc).resolve().parents[1]
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["CUDACXX"] = str(Path(nvcc).resolve())
    env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    env["MAX_JOBS"] = env.get("I3D_MAX_JOBS", "4")
    arch = re.findall(r"\d+", diagnostics[3])
    if len(arch) >= 2:
        env["TORCH_CUDA_ARCH_LIST"] = f"{arch[0]}.{arch[1]}"
    print("CUDA_HOME:", cuda_home, "NVCC:", nvcc, "GPU ARCH:",
          env.get("TORCH_CUDA_ARCH_LIST", "default"))

    ext = PAINT / "hy3dpaint" / "custom_rasterizer"
    if not (ext / "setup.py").exists():
        raise FileNotFoundError(f"Original Tencent CUDA rasterizer missing: {ext}")
    # Critical: build the original extension with build isolation disabled so
    # setup.py sees torch 2.11.0+cu130, and compiler NVCC 13.0.
    run([uv, "pip", "install", "--python", py, "--no-build-isolation",
         "-e", ext], env=env)
    run([py, "-c",
         "import torch, bpy, custom_rasterizer_kernel; "
         "assert torch.version.cuda == '13.0'; "
         "print('PAINT_CUDA13_EXTENSION_IMPORTED', torch.__version__, "
         "bpy.app.version_string)"], env=env)
    print("PAINT_CUDA13_INSTALL_SUCCESS (model inference not yet exercised)")


if __name__ == "__main__":
    main()
