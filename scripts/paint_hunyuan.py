"""Invoke actual Hunyuan3D-2.1 Paint on an existing Hi3D-refined GLB.

Run this with cwd pointing to vendor/Hunyuan3D-2.1 and its own dependencies
installed (Hunyuan's documented environment differs from Hi3D's).
"""
import argparse
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ('mesh','image','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--resolution',type=int,choices=[512,768],default=512)
    p.add_argument('--views',type=int,choices=[6,7,8,9],default=6)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'vendor'/'Hunyuan3D-2.1'
    if not root.is_dir():raise RuntimeError(f'Missing vendored source: {root}')
    import os
    os.chdir(root)
    sys.path.insert(0,str(root/'hy3dpaint'))
    sys.path.insert(0,str(root))
    from textureGenPipeline import Hunyuan3DPaintPipeline,Hunyuan3DPaintConfig
    conf=Hunyuan3DPaintConfig(a.views,a.resolution)
    conf.realesrgan_ckpt_path='hy3dpaint/ckpt/RealESRGAN_x4plus.pth'
    conf.multiview_cfg_path='hy3dpaint/cfgs/hunyuan-paint-pbr.yaml'
    conf.custom_pipeline='hy3dpaint/hunyuanpaintpbr'
    if not Path(conf.realesrgan_ckpt_path).is_file():
        raise FileNotFoundError('Missing Hunyuan Paint RealESRGAN weights; see upstream README setup')
    model=Hunyuan3DPaintPipeline(conf)
    output=model(mesh_path=str(Path(a.mesh).resolve()),image_path=str(Path(a.image).resolve()),
                 output_mesh_path=str(Path(a.output).resolve()))
    print('PAINTED',output)

if __name__=='__main__': main()
