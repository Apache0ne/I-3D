"""I-3D staged orchestration. Each heavy upstream model uses its own environment.

The `prepare` command runs original Hi3D code for Hunyuan2mini + Hi3D,
HunyuanPaint, TokenRig; SOMA bridge is original code. Motion is a second stage
because a human needs to pose and keyframe the SAME rig at frames 1 and 2.
Set per-stage executable via environment: I3D_HI3D_PYTHON, I3D_PAINT_PYTHON,
I3D_TOKENRIG_PYTHON. BLENDER defaults to `blender`.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'vendor'


def invoke(argv, cwd=ROOT):
    print('RUN', ' '.join(map(str,argv)),flush=True)
    subprocess.run(list(map(str,argv)),cwd=cwd,check=True)


def prepare(a):
    work=Path(a.output).resolve();work.mkdir(parents=True,exist_ok=True)
    image=Path(a.image).resolve()
    if not image.is_file():raise FileNotFoundError(image)
    refined=work/'01_geometry.glb';textured=work/'02_textured.glb'
    rigged=work/'03_tokenrig.glb';soma=work/'04_soma_character.glb'
    ref=work/'soma_reference.json'
    hi=V/'Hi3Dv1.0'
    if not (hi/'demo'/'image_to_mesh.py').exists():raise FileNotFoundError('Vendor original Hi3D source first')
    invoke([os.getenv('I3D_HI3D_PYTHON','python'),'demo/image_to_mesh.py',
            '--image',image,'--output',refined,'--resolution',a.geometry_resolution,
            '--num-inference-steps',a.steps,'--remove-background'],cwd=hi)
    # Paint depends on its own torch / CUDA / extension configuration.
    invoke([os.getenv('I3D_PAINT_PYTHON','python'),ROOT/'scripts'/'paint_hunyuan.py',
            '--mesh',refined,'--image',image,'--output',textured],cwd=V/'Hunyuan3D-2.1')
    token=V/'SkinTokens'
    invoke([os.getenv('I3D_TOKENRIG_PYTHON','python'),token/'demo.py',
            '--input',textured,'--output',rigged,'--use_transfer'],cwd=token)
    src=V/'motionbricks'/'motionbricks'/'assets'/'human'/'soma_base_skel_minimal_no_root.usd'
    invoke([a.usd_python,ROOT/'scripts'/'export_soma_reference.py',
            '--usd',src,'--output',ref])
    invoke([a.blender,'-b','--python',ROOT/'scripts'/'blender_bridge.py','--',
            '--input',rigged,'--reference',ref,'--output',soma,'--forward',a.forward])
    print('PREPARED:',soma)
    print('NEXT: Open GLB in Blender; keyframe two SOMA skeleton poses at 1 and 2; save .blend.')


def animate(a):
    work=Path(a.output).resolve();ref=work/'soma_reference.json'
    skeleton=work/'04_soma_character.glb';bridge=skeleton.with_suffix('.bridge.json')
    outpose=work/'05_poses.json';motion=work/'06_motion.npz';animated=work/'07_animated.glb'
    invoke([a.blender,Path(a.blend).resolve(),'-b','--python',ROOT/'scripts'/'export_two_poses.py',
            '--','--output',outpose])
    invoke([a.motion_python,ROOT/'scripts'/'generate_motion.py','--poses',outpose,
            '--reference',ref,'--bridge',bridge,'--output',motion,'--precision',a.precision])
    invoke([a.blender,'-b','--python',ROOT/'scripts'/'apply_motion_blender.py','--',
            '--character',skeleton,'--reference',ref,'--bridge',bridge,
            '--motion',motion,'--output',animated])
    print('ANIMATED:',animated)


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    a=sp.add_parser('prepare');a.add_argument('--image',required=True);a.add_argument('--output',default='outputs/character')
    a.add_argument('--forward',choices=['+X','-X','+Y','-Y'],required=True)
    a.add_argument('--geometry-resolution',type=int,choices=[512,1024,1536],default=512)
    a.add_argument('--steps',type=int,default=25);a.add_argument('--blender',default=os.getenv('BLENDER','blender'))
    a.add_argument('--usd-python',default='python')
    b=sp.add_parser('animate');b.add_argument('--blend',required=True);b.add_argument('--output',default='outputs/character')
    b.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
    b.add_argument('--motion-python',default='python');b.add_argument('--blender',default=os.getenv('BLENDER','blender'))
    parsed=p.parse_args()
    (prepare if parsed.command=='prepare' else animate)(parsed)

if __name__=='__main__':main()
