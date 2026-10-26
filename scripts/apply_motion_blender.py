"""Apply MotionBricks-generated SOMA local transforms to the SAME textured GLB.

blender -b --python scripts/apply_motion_blender.py -- --character soma_character.glb \
  --reference soma_reference.json --bridge soma_character.bridge.json \
  --motion generated_motion.npz --output animated_character.glb
"""
import argparse
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from i3d.pose_math import (soma_to_character_matrix,soma_world_to_pose,
                          local_to_world,transform_to_matrix)


def main():
    p=argparse.ArgumentParser()
    for name in ('character','reference','bridge','motion','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(Path(a.character).resolve()))
    arms=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
    if len(arms)!=1:raise ValueError('Expected one SOMA armature')
    arm=arms[0]
    ref=json.loads(Path(a.reference).read_text());bridge=json.loads(Path(a.bridge).read_text())
    names=ref['joint_names'];parents=ref['parents']
    assert len(names)==30
    if set(names)!=set(arm.data.bones.keys()):raise ValueError('Joint names do not match SOMA reference')
    npz=np.load(a.motion,allow_pickle=False)
    motion=npz['poses'];rate=int(npz['rate'])
    if rate!=30 or motion.ndim!=3 or motion.shape[1:]!=(30,7):raise ValueError('Wrong MotionBricks output')
    if not np.allclose(np.array(arm.matrix_world),np.eye(4),atol=1e-4):
        raise ValueError('Armature has an unexpected world transform')
    conversion=soma_to_character_matrix(bridge['template_scale'],bridge['template_translation'],bridge['forward'])
    rest=np.asarray(ref['rest_world_soma'],float)
    blender_rest={n:np.asarray(arm.data.bones[n].matrix_local) for n in names}
    scene=bpy.context.scene;scene.render.fps=30;scene.frame_start=1;scene.frame_end=len(motion)
    for j,frame in enumerate(motion):
        local=np.stack([transform_to_matrix(m) for m in frame])
        soma_world=local_to_world(local,parents)
        scene.frame_set(j+1)
        for i,n in enumerate(names):
            desired=soma_world_to_pose(soma_world[i],rest[i],blender_rest[n],conversion)
            bone=arm.pose.bones[n]
            bone.rotation_mode='QUATERNION'
            bone.matrix=Matrix(desired.tolist())
            bpy.context.view_layer.update()
            bone.keyframe_insert(data_path='location',frame=j+1)
            bone.keyframe_insert(data_path='rotation_quaternion',frame=j+1)
            bone.keyframe_insert(data_path='scale',frame=j+1)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(path.resolve()),export_format='GLB',export_animations=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path.with_suffix('.blend').resolve()))
    print('EXPORTED_ANIMATION',str(path),'frames',len(motion))

if __name__=='__main__':main()
