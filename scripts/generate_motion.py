"""Generate one MotionBricks-SOMA transition from two authored poses.

Uses vendor/motionbricks original engine. All poses converted back to SOMA
Y-up model frame before inference (model_to_runtime = identity). Produces
30 fps joint transforms and checks first/last root alignment.

python scripts/generate_motion.py --poses poses.json --reference soma_reference.json \
  --bridge soma_character.bridge.json --output generated_motion.npz
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'/'motionbricks'))
from i3d.pose_math import (soma_to_character_matrix,pose_world_to_soma,
                          world_to_local,matrix_to_transform,check_affine)


def load_inputs(poses_path,reference_path,bridge_path):
    poses=json.loads(Path(poses_path).read_text())
    ref=json.loads(Path(reference_path).read_text())
    bridge=json.loads(Path(bridge_path).read_text())
    names=ref['joint_names'];parents=ref['parents']
    if names!=poses['joint_names'] or len(names)!=30 or names[0]!='Hips':
        raise ValueError('Pose and reference joint names differ')
    conv=soma_to_character_matrix(float(bridge['template_scale']),
                                  bridge['template_translation'],bridge['forward'])
    bind=np.asarray(ref['rest_world_soma'],float)
    if bind.shape!=(30,4,4):raise ValueError('Invalid SOMA bind transforms')
    rest=poses['armature_rest']
    A=np.asarray(poses['armature_world'],float)
    if not np.allclose(A,np.eye(4),atol=1e-4):
        raise ValueError('Armature has unapplied global transform. Apply transforms before posing.')
    outputs=[]
    for frame in poses['poses']:
        w=np.stack([pose_world_to_soma(np.array(frame['world_pose'][n]),
              np.array(rest[n]),bind[i],conv) for i,n in enumerate(names)])
        local=world_to_local(w,parents)
        outputs.append(np.stack([matrix_to_transform(m) for m in local]).astype('float32'))
    return np.stack(outputs),names


def main():
    p=argparse.ArgumentParser()
    for name in ('poses','reference','bridge','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
    a=p.parse_args()
    inputs,names=load_inputs(a.poses,a.reference,a.bridge)
    # Lazy imports: keep deterministic math and CPU tests independent of Warp/TensorRT.
    import warp as wp
    from motionbricks.io.usd import load_skeleton_from_usd
    from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH,get_asset_file
    from motionbricks.model.soma.contract import SOMA_N1_FP32_ONNX_PATH,SOMA_N1_FP16_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine
    wp.init();wp.set_device('cuda:0')
    skeleton=load_skeleton_from_usd(get_asset_file('human','soma_base_skel_minimal_no_root.usd'),
                                   SOMA_SKELETON_PRIM_PATH)
    onnx=SOMA_N1_FP16_ONNX_PATH if a.precision=='fp16' else SOMA_N1_FP32_ONNX_PATH
    with MotionBricksSOMAEngine(onnx,model_to_runtime=wp.transform_identity()) as engine:
        context=engine.create_input(skeleton);target=engine.create_input(skeleton)
        result=engine.create_output(skeleton)
        start=np.repeat(inputs[0][None],4,axis=0).astype('float32')
        dest=inputs[1][None].astype('float32')
        context.update_from_poses(wp.array(start,dtype=wp.transform,device=engine.device),stream=engine.stream)
        target.update_from_poses(wp.array(dest,dtype=wp.transform,device=engine.device),stream=engine.stream)
        engine.generate_animation(context,target,result)
        engine.synchronize()
        frames=int(result.num_frames)
        motion=np.array(result.poses[:frames].numpy()).reshape(frames,30,7)
        if not np.isfinite(motion).all():raise ValueError('Nonfinite generated motion')
    # Root boundary check, not a claim of exact endpoint for every joint.
    start_error=float(np.linalg.norm(motion[0,0,:3]-inputs[0,0,:3]))
    target_error=float(np.linalg.norm(motion[-1,0,:3]-inputs[1,0,:3]))
    if start_error>.10 or target_error>.20:
        raise ValueError(f'Generated root endpoint discontinuity: start={start_error:.3f}m end={target_error:.3f}m')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,poses=motion,rate=np.int32(30),joint_names=np.array(names),
                        bridge_info=str(Path(a.bridge)),start_error=start_error,target_error=target_error)
    print('MOTIONBRICKS_GENERATED',frames,'frames',frames/30,'sec',a.output)

if __name__=='__main__':main()
