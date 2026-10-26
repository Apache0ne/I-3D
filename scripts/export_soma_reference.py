"""Extract the *actual* 30 MotionBricks SOMA world rest matrices from its USD.

Run in an environment with `usd-core` (pxr) and numpy installed. The complete
Git LFS asset is required, not its 134-byte pointer. Geometry is in meters.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdSkel


def extract(path):
    data = Path(path).read_bytes()[:100]
    if b'git-lfs.github.com' in data:
        raise RuntimeError('Reference USD is an LFS POINTER. Run git lfs pull in vendor/motionbricks')
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError('Cannot open USD skeleton')
    skeletons = [p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    if len(skeletons) != 1:
        raise ValueError(f'Expected one SOMA skeleton, found {len(skeletons)}')
    skeleton = UsdSkel.Skeleton(skeletons[0])
    paths = [str(x) for x in skeleton.GetJointsAttr().Get()]
    names = [x.split('/')[-1] for x in paths]
    if len(names) != 30 or len(set(names)) != 30 or names[0] != 'Hips':
        raise ValueError(f'Wrong MotionBricks SOMA joint layout ({len(names)} joints)')
    # Match NVIDIA MotionBricks io/usd.py: the engine loads LOCAL restTransforms,
    # NOT the UsdSkel world bindTransforms used for skinning. Composition to world
    # is mandatory; confusing these two transforms misaligns the first pose.
    local_raw = skeleton.GetRestTransformsAttr().Get()
    if local_raw is None or len(local_raw) != 30:
        raise ValueError('Expected 30 local restTransforms')
    local = np.asarray(local_raw, dtype=np.float64).transpose(0, 2, 1)
    metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    xform = np.asarray(UsdGeom.Xformable(skeletons[0]).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()), dtype=np.float64)
    axis_lengths = np.linalg.norm(xform[:3,:3], axis=0)
    if np.min(axis_lengths) <= 0 or not np.allclose(axis_lengths, axis_lengths[0], rtol=1e-4):
        raise ValueError('Nonuniform or zero skeleton prim scale not supported')
    if np.linalg.det(xform[:3,:3]) <= 0:
        raise ValueError('Mirrored USD skeleton prim rejected')
    scale = metres_per_unit * float(axis_lengths[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid USD scale')
    local[:,:3,3] *= scale
    parents = [paths.index(p.rsplit('/', 1)[0]) if '/' in p and
               p.rsplit('/', 1)[0] in paths else -1 for p in paths]
    rest_world = np.empty_like(local)
    for i, m in enumerate(local):
        if not np.allclose(m[:3,:3].T @ m[:3,:3], np.eye(3), atol=1e-4) or \
           abs(np.linalg.det(m[:3,:3]) - 1.) > 1e-4:
            raise ValueError(f'Invalid reflection/local rest matrix at {names[i]}')
        p=parents[i]
        if p>=i:
            raise ValueError(f'Invalid SOMA joint hierarchy: {names[i]} parent {p}')
        rest_world[i] = m if p<0 else rest_world[p] @ m
    zup = rest_world[:,:3,3] @ np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
    height = float(np.ptp(zup[:,2]))
    if not 0.35 < height < 3.0:
        raise ValueError(f'Implausible SOMA rig height {height:.2f} m; check USD units')
    return {'joint_names':names,'parents':parents,'rest_world_soma':rest_world.tolist(),
            'metres_per_unit':metres_per_unit,'prim_scale':float(axis_lengths[0]),
            'units':'m','source':str(path),'height_m':height,
            'convention':'SOMA_Y_UP_RH; MotionBricks skeleton reference restTransforms'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--usd',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(extract(a.usd),indent=2))
    print('Saved actual SOMA reference:',a.output)

if __name__=='__main__':main()
