"""Blender background-mode bridge: TokenRig skinned GLB -> SOMA 30-joint GLB.

Uses the *actual* MotionBricks SOMA reference bind matrices, not guessed bone names.
Copies geometry and materials, remaps weights using spatially nearest source rest
joints, and creates the canonically named SOMA hierarchy. Requires a human mesh
whose rest pose roughly matches the SOMA rest pose. Produces diagnostics; never
uses negative scale, an axis mirror, or source-bone name heuristics.

Usage (after TokenRig's --use_transfer):
 blender -b --python scripts/blender_bridge.py -- --input person_rigged.glb \
   --reference soma_reference.json --output soma_character.glb --forward +X

A joint-mapping review is essential before animation, especially for exotic poses.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from i3d.axes import (SOMA_TO_BLENDER, check_rotation, facing_rotation,
                      transform_points, verify_laterality)


def arg_parser():
    ap=argparse.ArgumentParser()
    for name in ('input','reference','output'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--forward',choices=['+X','-X','+Y','-Y'],required=True,
                    help='Visual forward direction of the imported character in Blender Z-up coordinates')
    ap.add_argument('--maximum-unmatched-distance',type=float,default=0.25)
    return ap.parse_args(sys.argv[sys.argv.index('--')+1:])


def world_bbox(meshes):
    points=[]
    for ob in meshes:
        for v in ob.bound_box:
            points.append(np.array(ob.matrix_world @ Vector(v)))
    p=np.array(points)
    return p.min(axis=0),p.max(axis=0)


def create_armature(names,parents,xyz):
    arm=bpy.data.armatures.new('I3D_SOMA_30')
    ob=bpy.data.objects.new('I3D_SOMA_30',arm)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active=ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for i,name in enumerate(names):
        b=arm.edit_bones.new(name)
        b.head=tuple(xyz[i])
        children=[k for k,p in enumerate(parents) if p==i and k!=i]
        if children:
            child=xyz[children[0]]
            direction=child-xyz[i]
        elif parents[i]>=0:
            direction=xyz[i]-xyz[parents[i]]
        else:
            direction=np.array([0.,0.,1.])
        norm=np.linalg.norm(direction)
        if norm<0.0001: direction=np.array([0.,0.,1.]);norm=1.
        b.tail=tuple(xyz[i]+direction/norm*max(0.035,min(norm,0.16)))
    for i,p in enumerate(parents):
        if p>=0 and p!=i: arm.edit_bones[names[i]].parent=arm.edit_bones[names[p]]
    bpy.ops.object.mode_set(mode='OBJECT')
    return ob


def main():
    a=arg_parser()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(Path(a.input).resolve()))
    armatures=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
    if len(armatures)!=1:
        raise ValueError(f'Expected ONE TokenRig armature; got {len(armatures)}')
    source_arm=armatures[0]
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if not meshes:raise ValueError('No imported mesh')
    src_names=list(source_arm.data.bones.keys())
    if len(src_names)<8:raise ValueError('Source rig has too few joints')
    for obj in meshes:
        if not obj.vertex_groups or not any(v.groups for v in obj.data.vertices):
            raise ValueError(f'{obj.name} has no skinning weights')
    lo,hi=world_bbox(meshes)
    char_height=float(hi[2]-lo[2])
    if not 0.2<char_height<8:raise ValueError('Bad mesh height or scene units')

    ref=json.loads(Path(a.reference).read_text())
    names=ref['joint_names'];parents=ref['parents']
    if len(names)!=30 or names[0]!='Hips':raise ValueError('Not the SOMA MotionBricks 30-joint reference')
    rest=np.array(ref['rest_world_soma'],dtype=float)
    if rest.shape!=(30,4,4):raise ValueError('Invalid SOMA reference matrices')
    for i in range(30):check_rotation(rest[i,:3,:3])
    # Canonical SOMA Y-up to Blender Z-up: +90 degrees about X.
    zup=rest[:,:3,3] @ SOMA_TO_BLENDER.T
    zup=zup@facing_rotation(a.forward).T
    # Only positive uniform scale; no per-axis flip or deformation.
    height=float(np.ptp(zup[:,2]))
    if height<=0:raise ValueError('Invalid reference height')
    scale=char_height/height
    center=(lo+hi)*.5
    rest_mid=(zup.min(axis=0)+zup.max(axis=0))*.5
    shift=center-scale*rest_mid
    aligned=scale*zup+shift
    verify_laterality(dict(zip(names,aligned)),a.forward)
    # No left-right swap: all reference bones use the same non-reflective similarity.
    src_world=[]
    for b in source_arm.data.bones:
        src_world.append(np.array(source_arm.matrix_world @ b.matrix_local)[:3,3])
    src_world=np.array(src_world)
    # Match each source influence to nearest SOMA joint, normalized by body height.
    distance=np.linalg.norm(src_world[:,None,:]-aligned[None,:,:],axis=-1)
    mapping=np.argmin(distance,axis=1)
    dmin=distance[np.arange(len(src_names)),mapping]
    extreme=dmin>min(a.maximum_unmatched_distance,char_height*.25)
    if np.mean(extreme)>.35:
        raise ValueError(f'Poor rig alignment: {np.mean(extreme)*100:.1f}% source joints far from SOMA; choose correct --forward or pose')
    soma_arm=create_armature(names,parents,aligned)
    soma_arm['i3d_forward']=a.forward
    soma_arm['i3d_reference_json']=str(Path(a.reference).resolve())
    diagnostics={'forward':a.forward,'soma_to_blender':SOMA_TO_BLENDER.tolist(),
       'handedness':'right-handed','template_scale':scale,'template_translation':shift.tolist(),
       'height_m':char_height,'source_joints':len(src_names),
       'source_to_soma':{src_names[i]:names[int(mapping[i])] for i in range(len(src_names))},
       'unmatched_fraction':float(np.mean(extreme)),
       'median_mapping_error_m':float(np.median(dmin)),
       'warning':'Spatial influence mapping is approximate; review limb joints and skinning before animation.'}
    # Keep original mesh data, geometry and materials, but replace deformer with SOMA.
    for ob in meshes:
        # Capture weights by actual vertex-group name (index order is object-specific).
        source_groups=[g.name for g in ob.vertex_groups]
        weights=[]
        for v in ob.data.vertices:
            per={}
            for g in v.groups:
                name=source_groups[g.group]
                if name not in src_names:continue
                i=src_names.index(name)
                k=int(mapping[i]);per[k]=per.get(k,0.)+float(g.weight)
            if not per:raise ValueError(f'Unskinned vertex {v.index} in {ob.name}')
            best=sorted(per.items(),key=lambda p:-p[1])[:4]
            total=sum(w for _,w in best)
            if total<=1e-8:raise ValueError('Zero weight normalization')
            weights.append([(k,w/total) for k,w in best])
        # Bake GLB node transform into the mesh; preserve materials and UVs.
        ob.data=ob.data.copy()
        ob.data.transform(ob.matrix_world)
        ob.matrix_world=Matrix.Identity(4)
        for modifier in list(ob.modifiers):
            if modifier.type=='ARMATURE':ob.modifiers.remove(modifier)
        for grp in list(ob.vertex_groups):ob.vertex_groups.remove(grp)
        target_groups=[ob.vertex_groups.new(name=n) for n in names]
        for v,groups in enumerate(weights):
            for k,w in groups:target_groups[k].add([v],w,'REPLACE')
        mod=ob.modifiers.new('SOMA_skin','ARMATURE');mod.object=soma_arm
        ob.parent=soma_arm
        ob.matrix_parent_inverse=Matrix.Identity(4)
    bpy.data.objects.remove(source_arm,do_unlink=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    soma_arm.select_set(True)
    for ob in meshes:ob.select_set(True)
    bpy.context.view_layer.objects.active=soma_arm
    bpy.ops.export_scene.gltf(filepath=str(Path(a.output).resolve()),export_format='GLB',use_selection=True,
                              export_skins=True,export_animations=True)
    Path(a.output).with_suffix('.bridge.json').write_text(json.dumps(diagnostics,indent=2))
    print('SOMA_BRIDGE_RESULT',json.dumps({'glb':a.output,'diagnostics':str(Path(a.output).with_suffix('.bridge.json'))}))

if __name__=='__main__':main()
