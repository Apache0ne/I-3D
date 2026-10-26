"""Invertible math for Blender Z-up posed skeleton <-> SOMA Y-up joint transforms.

A = [uniform-scale * facing_R * Rx(90), translation] maps SOMA joint world
matrices to Blender/world matrices.  Its determinant is positive.  Pose
corrections are conjugated by A, preserving target geometry without mirroring.
"""
from __future__ import annotations
import numpy as np
from .axes import SOMA_TO_BLENDER, facing_rotation, check_rotation


def soma_to_character_matrix(scale: float, translation, forward: str):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be positive')
    r = check_rotation(facing_rotation(forward) @ SOMA_TO_BLENDER)
    a=np.eye(4);a[:3,:3]=scale*r;a[:3,3]=np.asarray(translation,float)
    return a


def check_affine(m, *, allow_uniform_scale=False):
    m=np.asarray(m,float)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-6):
        raise ValueError('Invalid affine transform')
    r=m[:3,:3];s=float(np.cbrt(np.linalg.det(r)))
    if s<=1e-8 or (not allow_uniform_scale and abs(s-1)>1e-4):
        raise ValueError('Reflection or scale drift')
    check_rotation(r/s)
    return m


def pose_world_to_soma(blender_pose, blender_bind, soma_bind, conversion):
    """World-space bone pose from Blender rig -> canonical SOMA world pose."""
    p=check_affine(blender_pose);b=check_affine(blender_bind);s=check_affine(soma_bind)
    a=check_affine(conversion,allow_uniform_scale=True)
    delta=p @ np.linalg.inv(b)
    soma_delta=np.linalg.inv(a) @ delta @ a
    result=soma_delta @ s
    return check_affine(result)


def soma_world_to_pose(soma_pose,soma_bind,blender_bind,conversion):
    """Inverse of pose_world_to_soma. Allows Blender edits and generated poses to agree."""
    p=check_affine(soma_pose);s=check_affine(soma_bind);b=check_affine(blender_bind)
    a=check_affine(conversion,allow_uniform_scale=True)
    delta=a @ (p @ np.linalg.inv(s)) @ np.linalg.inv(a)
    return check_affine(delta @ b)


def world_to_local(world,parents):
    world=np.asarray(world,float)
    if world.ndim!=3 or world.shape[1:]!=(4,4) or len(parents)!=len(world):
        raise ValueError('Expect (J,4,4) and J parent indices')
    local=np.empty_like(world)
    for i,p in enumerate(parents):
        if p<0 or p==i:local[i]=world[i]
        else:
            if p>=i:raise ValueError('Joints must be parents-before-children')
            local[i]=np.linalg.inv(world[p]) @ world[i]
        check_affine(local[i])
    return local


def local_to_world(local,parents):
    local=np.asarray(local,float)
    if local.ndim!=3 or local.shape[1:]!=(4,4) or len(local)!=len(parents):
        raise ValueError('Invalid local pose matrices')
    world=np.empty_like(local)
    for i,p in enumerate(parents):
        if p<0 or p==i:world[i]=local[i]
        else:
            if p>=i:raise ValueError('Joints must be parents-before-children')
            world[i]=world[p]@local[i]
        check_affine(world[i])
    return world


def matrix_to_transform(m):
    """4x4 rigid matrix -> Warp transform (tx,ty,tz,qx,qy,qz,qw)."""
    m=check_affine(m); r=m[:3,:3]
    tr=float(np.trace(r))
    if tr>0:
        s=np.sqrt(tr+1.)*2
        q=[(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s,s/4]
    else:
        i=int(np.argmax(np.diag(r)))
        if i==0:
            s=2*np.sqrt(1+r[0,0]-r[1,1]-r[2,2]);q=[s/4,(r[0,1]+r[1,0])/s,(r[0,2]+r[2,0])/s,(r[2,1]-r[1,2])/s]
        elif i==1:
            s=2*np.sqrt(1+r[1,1]-r[0,0]-r[2,2]);q=[(r[0,1]+r[1,0])/s,s/4,(r[1,2]+r[2,1])/s,(r[0,2]-r[2,0])/s]
        else:
            s=2*np.sqrt(1+r[2,2]-r[0,0]-r[1,1]);q=[(r[0,2]+r[2,0])/s,(r[1,2]+r[2,1])/s,s/4,(r[1,0]-r[0,1])/s]
    q=np.asarray(q,dtype=float);q/=np.linalg.norm(q)
    if q[3]<0:q=-q
    return np.concatenate((m[:3,3],q))


def transform_to_matrix(pose):
    x=np.asarray(pose,float)
    if x.shape!=(7,) or not np.isfinite(x).all():raise ValueError('Expected finite 7-vector')
    a,b,c,d=x[3:];qnorm=np.linalg.norm([a,b,c,d]);
    if qnorm<1e-8:raise ValueError('Quaternion has no length')
    a,b,c,d=np.array([a,b,c,d])/qnorm
    r=np.array([[1-2*(b*b+c*c),2*(a*b-c*d),2*(a*c+b*d)],
                [2*(a*b+c*d),1-2*(a*a+c*c),2*(b*c-a*d)],
                [2*(a*c-b*d),2*(b*c+a*d),1-2*(a*a+b*b)]])
    o=np.eye(4);o[:3,:3]=r;o[:3,3]=x[:3];return check_affine(o)
