"""Download original MotionBricks Git LFS runtime artifacts without committing weights.

Requires Git LFS, network access and the upstream NVIDIA license.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
MODEL_BASE="motionbricks/assets/onnx/soma/"
SKEL="motionbricks/assets/human/soma_base_skel_minimal_no_root.usd"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--precision",choices=["fp16","fp32","both"],default="fp32")
    a=p.parse_args()
    assets=[SKEL,MODEL_BASE+"mmm_v6_soma_"+("fp16" if a.precision=="fp16" else "fp32")+".onnx"]
    if a.precision=="both":
        assets=[SKEL,MODEL_BASE+"mmm_v6_soma_fp32.onnx",MODEL_BASE+"mmm_v6_soma_fp16.onnx"]
    dest=ROOT/"vendor"/"motionbricks"
    if not dest.is_dir():
        raise FileNotFoundError("Run scripts/vendor_sources.py before hydrating assets")
    with tempfile.TemporaryDirectory(prefix="i3d_nvidia_lfs_") as d:
        env=dict(os.environ,GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(["git","lfs","version"],check=True)
        subprocess.run(["git","clone","--quiet","--depth","1","https://github.com/NVlabs/motionbricks.git",d],env=env,check=True)
        subprocess.run(["git","lfs","pull","--include="+",".join(assets),"--exclude="],cwd=d,check=True)
        for path in assets:
            original=Path(d)/path
            if not original.is_file() or original.read_bytes()[:100].startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(f"Missing actual Git LFS artifact: {path}")
            target=dest/path
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,target)
            print(f"hydrated {target} ({target.stat().st_size:,} bytes)",flush=True)

if __name__=="__main__":
    main()
