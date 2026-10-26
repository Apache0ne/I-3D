"""Blender: save pose on frames 1 (Start) and 2 (Target) to machine-readable JSON.

Open the SOMA GLB in Blender, pose the same armature at frames 1 and 2, keyframe
all bones for each frame, save .blend, then run this script from that blend file.
No second mesh or second image is created.

blender character_poses.blend -b --python scripts/export_two_poses.py -- --output poses.json
"""
import argparse
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--start-frame',type=int,default=1)
    parser.add_argument('--target-frame',type=int,default=2)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    rigs=[o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.name.startswith('I3D_SOMA')]
    if len(rigs)!=1:raise ValueError('Expected one I3D_SOMA armature')
    arm=rigs[0]
    names=[b.name for b in arm.data.bones]
    if len(names)!=30 or names[0]!='Hips':raise ValueError('Not a SOMA 30-joint armature')
    if not arm.animation_data or not arm.animation_data.action:
        raise ValueError('No keyframed animation found; keyframe Start at 1 and Target at 2')
    rest={b.name:np.asarray(b.matrix_local).tolist() for b in arm.data.bones}
    poses=[]
    for f in (a.start_frame,a.target_frame):
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        poses.append({'frame':f,'world_pose':{n:np.asarray(arm.pose.bones[n].matrix).tolist() for n in names}})
    if all(np.allclose(poses[0]['world_pose'][n],poses[1]['world_pose'][n],atol=1e-5) for n in names):
        raise ValueError('Start and target are identical; edit and keyframe the target pose')
    out={'joint_names':names,'armature_world':np.asarray(arm.matrix_world).tolist(),
         'armature_rest':rest,'poses':poses,'coordinate_system':'Blender_Z_UP_RH','units':'m'}
    Path(a.output).write_text(json.dumps(out,indent=2))
    print('POSE_EXPORT_OK',a.output)

if __name__=='__main__':main()
