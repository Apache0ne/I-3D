"""Convert 2 poses authored in the in-Colab I-3D Three.js viewer into Blender keyframes.

This script is launched by headless Blender only: the end user never leaves
the notebook and never needs to open Blender. It rejects guessed axes or
incompatible source meshes. glTF Y-up -> Blender Z-up is Rx(+90 degrees).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix

GLTF_TO_BLENDER=np.array([[1.,0.,0.,0.],
                          [0.,0.,-1.,0.],
                          [0.,1.,0.,0.],
                          [0.,0.,0.,1.]],dtype=np.float64)
BLENDER_TO_GLTF=np.linalg.inv(GLTF_TO_BLENDER)

def three_matrix(flat):
    m=np.asarray(flat,dtype=np.float64).reshape((4,4),order="F")
    if not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-5):
        raise ValueError("Bad Three.js affine matrix")
    rot=m[:3,:3]
    if abs(np.linalg.det(rot)-1)>5e-3 or not np.allclose(rot.T@rot,np.eye(3),atol=5e-3):
        raise ValueError("Reflected or non-rigid source joint transform")
    return m

def apply_frame(arm,entry,names,frame):
    if entry.get("coordinate_system")!="GLTF_Y_UP_RH":
        raise ValueError("Pose data must be GLTF_Y_UP_RH")
    records=entry.get("bones",{})
    if set(records)!=set(names):raise ValueError("SOMA joint sets differ")
    # Relative world rotations and translations are basis-converted, not
    # raw local Euler angles: glTF and Blender bone roll conventions differ.
    targets={}
    for name in names:
        rec=records[name]
        before=three_matrix(rec["rest_world"])
        after=three_matrix(rec["world"])
        delta_gltf=after@np.linalg.inv(before)
        delta_blender=GLTF_TO_BLENDER@delta_gltf@BLENDER_TO_GLTF
        targets[name]=delta_blender@np.asarray(arm.data.bones[name].matrix_local)
    bpy.context.scene.frame_set(frame)
    # Parent first. Blender's pose matrix setter accounts for parenting.
    for pb in sorted(arm.pose.bones,key=lambda b:len(b.parent_recursive)):
        pb.rotation_mode="QUATERNION"
        pb.matrix=Matrix(targets[pb.name].tolist())
        bpy.context.view_layer.update()
    for pb in arm.pose.bones:
        for prop in ("location","rotation_quaternion","scale"):
            pb.keyframe_insert(data_path=prop,frame=frame)

def main():
    p=argparse.ArgumentParser()
    for k in ("character","poses","output"):p.add_argument("--"+k,required=True,type=Path)
    a=p.parse_args(sys.argv[sys.argv.index("--")+1:])
    info=json.loads(a.poses.read_text(encoding="utf-8"))
    if info.get("version")!=1 or info.get("coordinate_system")!="GLTF_Y_UP_RH":
        raise ValueError("Viewer pose interchange format mismatch")
    states=info.get("poses",{})
    if "start" not in states or "target" not in states:
        raise ValueError("Save BOTH START and TARGET poses inside the I-3D Colab viewer")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(a.character.resolve()))
    arms=[x for x in bpy.context.scene.objects
          if x.type=="ARMATURE" and x.name.startswith("I3D_SOMA")]
    if len(arms)!=1:raise ValueError("Require one canonical I3D_SOMA armature")
    arm=arms[0]
    names=[x.name for x in arm.data.bones]
    if len(names)!=30 or names[0]!="Hips":raise ValueError("Not a Hips-rooted 30-joint SOMA rig")
    if not np.allclose(np.asarray(arm.matrix_world),np.eye(4),atol=1e-4):
        raise ValueError("Reflected/nonidentity armature transform; cannot guess axes")
    for slot in ("start","target"):
        entry=states[slot]
        if (entry.get("source")!=a.character.name or
            entry.get("source_bytes")!=a.character.stat().st_size or
            abs(entry.get("source_mtime",0)-a.character.stat().st_mtime)>.1):
            raise ValueError(f"{slot} pose refers to another GLB; resave both poses")
    scene=bpy.context.scene
    scene.render.fps=30
    scene.frame_start,scene.frame_end=1,2
    apply_frame(arm,states["start"],names,1)
    apply_frame(arm,states["target"],names,2)
    scene.frame_set(1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output.resolve()))
    print("I3D_COLAB_POSES_SAVED",str(a.output),flush=True)

if __name__=="__main__":main()
