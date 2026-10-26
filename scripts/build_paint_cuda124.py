#!/usr/bin/env python3
"""Build the ORIGINAL Hunyuan3D-Paint 2.1 CUDA rasterizer with nvcc 12.4.

Colab (October 2026) may ship CUDA nvcc 13.0 while the upstream paint
environment uses torch 2.5.1+cu124. This installs an isolated CUDA 12.4.1
compiler toolchain from NVIDIA's archived Conda channel, then builds only the
original vendor/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer extension.

Does not modify upstream C++/CUDA code, the host NVIDIA driver, /usr/local/cuda,
or torch. Does not weaken PyTorch's CUDA major-version compatibility check.
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
DEFAULT_PAINT_PY = Path("/content/i3d_envs/paint/bin/python")
DEFAULT_CUDA_HOME = Path("/content/i3d_cuda124")
OFFICIAL_LABEL = "nvidia/label/cuda-12.4.1"


def call(*argv: object, env: dict[str, str] | None = None) -> None:
    cmd = [str(x) for x in argv]
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def version_of_nvcc(path: Path) -> str | None:
    if not path.is_file():
        return None
    r = subprocess.run([str(path), "--version"], capture_output=True, text=True)
    if r.returncode:
        return None
    match = re.search(r"release\s+(\d+)\.(\d+)", r.stdout + r.stderr)
    return ".".join(match.groups()) if match else None


def matching_cuda_home() -> Path:
    candidates = (
        Path(os.getenv("I3D_CUDA124_HOME", "/usr/local/cuda-12.4")),
        DEFAULT_CUDA_HOME,
    )
    for p in candidates:
        if version_of_nvcc(p / "bin" / "nvcc") == "12.4":
            print("CUDA 12.4 compiler already available:", p, flush=True)
            return p

    # CUDA 12.4 toolchain from NVIDIA's own archived Conda channel.
    # Unlike apt's ubuntu2404 repository, this channel still offers CUDA 12.4.
    micromamba = shutil.which("micromamba")
    if micromamba is None:
        tools = Path("/content/i3d_tools")
        tools.mkdir(parents=True, exist_ok=True)
        micromamba = str(tools / "micromamba")
        if not Path(micromamba).is_file():
            archive = tools / "micromamba.tar.bz2"
            call("curl", "-fSL", "--retry", "3",
                 "https://micro.mamba.pm/api/micromamba/linux-64/latest",
                 "-o", archive)
            call("tar", "-xjf", archive, "-C", tools, "--strip-components=1",
                 "bin/micromamba")
    home = DEFAULT_CUDA_HOME
    call(micromamba, "create", "-y", "-p", home,
         "-c", OFFICIAL_LABEL, "-c", "conda-forge",
         "--strict-channel-priority",
         "cuda-compiler=12.4.1", "cuda-cudart-dev", "cuda-cccl")
    version = version_of_nvcc(home / "bin" / "nvcc")
    if version != "12.4":
        raise RuntimeError(
            f"Expected nvcc 12.4 from NVIDIA CUDA 12.4.1, got {version!r} "
            f"at {home / 'bin' / 'nvcc'}. Do not compile with CUDA 13."
        )
    return home


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(DEFAULT_PAINT_PY),
                        help="Paint Python 3.10 executable already containing torch cu124")
    parser.add_argument("--uv", default=shutil.which("uv") or "/usr/local/bin/uv")
    args = parser.parse_args()
    paint_py = Path(args.python).resolve()
    if not paint_py.is_file():
        raise FileNotFoundError(f"Missing Hunyuan paint environment: {paint_py}")

    module_dir = ROOT / "vendor" / "Hunyuan3D-2.1" / "hy3dpaint" / "custom_rasterizer"
    if not (module_dir / "setup.py").is_file():
        raise FileNotFoundError(f"Original custom_rasterizer setup.py not found: {module_dir}")

    torch_info = subprocess.check_output(
        [str(paint_py), "-c",
         "import torch; print(torch.__version__); print(torch.version.cuda); "
         "print('%s.%s' % torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'no_gpu')"],
        text=True,
    ).strip().splitlines()
    print("Paint Torch / compiled CUDA / device capability:", torch_info)
    if len(torch_info) < 3 or torch_info[1] != "12.4":
        raise RuntimeError(
            f"Paint PyTorch must be CUDA 12.4, got {torch_info!r}. "
            "Do not compile with an inconsistent CUDA version."
        )
    if torch_info[2] == "no_gpu":
        raise RuntimeError("No CUDA GPU available in this runtime.")
    major, minor = (int(n) for n in torch_info[2].split("."))
    if major >= 10:
        raise RuntimeError(
            f"GPU sm_{major}{minor} is not supported by the CUDA 12.4 compiler "
            "or this upstream PyTorch 2.5.1 stack. Choose a suitable GPU "
            "(e.g., A100 sm80, L4 sm89, H100 sm90)."
        )

    cuda_home = matching_cuda_home()
    nvcc = cuda_home / "bin" / "nvcc"
    assert version_of_nvcc(nvcc) == "12.4"

    env = dict(os.environ)
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    env["CUDACXX"] = str(nvcc)
    env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    env["MAX_JOBS"] = os.getenv("I3D_MAX_JOBS", "4")

    # CUDA 12.4 supports GCC 12; CUDA 13 Colab images may default to GCC 13.
    if not shutil.which("g++-12"):
        call("apt-get", "update", "-qq")
        call("apt-get", "install", "-y", "-qq", "gcc-12", "g++-12")
    env["CC"] = shutil.which("gcc-12") or "/usr/bin/gcc-12"
    env["CXX"] = shutil.which("g++-12") or "/usr/bin/g++-12"

    # Avoid injecting CUDA 13 libraries into extension link/search paths.
    cuda_libs = [cuda_home / "lib64", cuda_home / "lib",
                 cuda_home / "targets" / "x86_64-linux" / "lib"]
    include_paths = [cuda_home / "include",
                     cuda_home / "targets" / "x86_64-linux" / "include"]
    env["LIBRARY_PATH"] = os.pathsep.join(
        str(p) for p in cuda_libs if p.is_dir()
    ) + (os.pathsep + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else "")
    env["CPATH"] = os.pathsep.join(
        str(p) for p in include_paths if p.is_dir()
    ) + (os.pathsep + env["CPATH"] if env.get("CPATH") else "")

    print("Building original Tencent rasterizer with:", nvcc)
    print("CUDA_HOME:", env["CUDA_HOME"], "arch:", env["TORCH_CUDA_ARCH_LIST"])
    call(args.uv, "pip", "install", "--python", paint_py,
         "setuptools", "wheel", "ninja")
    call(args.uv, "pip", "install", "--python", paint_py,
         "--no-build-isolation", "-e", module_dir, env=env)
    call(paint_py, "-c",
         "import torch, bpy, custom_rasterizer_kernel; "
         "print('HUNYUAN_PAINT_CUDA_EXTENSION_READY', torch.__version__, "
         "torch.version.cuda, bpy.app.version_string)",
         env=env)


if __name__ == "__main__":
    main()
