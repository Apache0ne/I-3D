"""Reproducibly vendor full upstream source trees, preserving LICENSE/NOTICE.

Source is copied 1:1 from upstream Git checkout (no .git metadata, no model
weights). This tool does NOT download restricted weights or accept licenses.
If --commit is supplied, all upstream commit SHAs are recorded in manifest.
"""
import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
# Model source code, *not* safetensors/ONNX/Git-LFS weights.
REPOS={
 'Hi3Dv1.0':'https://github.com/hi3d-official/Hi3Dv1.0.git',
 'Hunyuan3D-2':'https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git',
 'Hunyuan3D-2.1':'https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git',
 'SkinTokens':'https://github.com/VAST-AI-Research/SkinTokens.git',
 'SOMA-X':'https://github.com/NVlabs/SOMA-X.git',
 'motionbricks':'https://github.com/NVlabs/motionbricks.git',
}
# RMBG-2.0 implementation is remote-model-code via Hugging Face, not a standalone
# fully released source repository with a separate complete local runtime.
EXCLUDED={'.git','.gitattributes','.lfsconfig','.venv','__pycache__','.trt-cache','outputs','results','checkpoints','experiments'}
WEIGHT_SUFFIX={'.safetensors','.ckpt','.pth','.pt','.onnx','.bin','.pkl','.npz'}

def run(*cmd,cwd=None):
    return subprocess.check_output(cmd,cwd=cwd,text=True).strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--destination',default=str(ROOT/'vendor'))
    p.add_argument('--revision-file',help='Optional JSON mapping repo keys to exact Git commit SHAs')
    a=p.parse_args()
    pins=json.loads(Path(a.revision_file).read_text()) if a.revision_file else {}
    dst=Path(a.destination).resolve();dst.mkdir(parents=True,exist_ok=True)
    manifest={}
    for name,url in REPOS.items():
        out=dst/name
        if out.exists():raise FileExistsError(f'Refusing to overwrite: {out}')
        with tempfile.TemporaryDirectory(prefix='i3d_git_') as tmp:
            env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
            subprocess.run(['git','clone','--quiet','--filter=blob:none',url,tmp],env=env,check=True)
            if name in pins:subprocess.run(['git','checkout','--quiet',pins[name]],cwd=tmp,check=True)
            sha=run('git','rev-parse','HEAD',cwd=tmp)
            # Preserve release sources and licenses/notice, but not .git or heavyweight artifacts.
            shutil.copytree(tmp,out,ignore=lambda dir,files:[f for f in files if f in EXCLUDED or
                            (Path(f).suffix.lower() in WEIGHT_SUFFIX and f not in ('pyproject.toml',))])
            manifest[name]={'upstream':url,'commit':sha,'copied_source':'vendor/'+name}
            print(name,sha,flush=True)
    (dst/'SOURCE_REVISIONS.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
