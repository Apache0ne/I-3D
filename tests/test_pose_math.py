import unittest
import numpy as np
from i3d.axes import facing_rotation
from i3d.pose_math import (soma_to_character_matrix,check_affine,pose_world_to_soma,
    soma_world_to_pose,world_to_local,local_to_world,matrix_to_transform,transform_to_matrix)

class PoseMath(unittest.TestCase):
    def test_round_trip_transform(self):
        for deg in [0,10,90,170,180,270]:
            rad=np.deg2rad(deg)
            m=np.eye(4);m[:3,:3]=[[np.cos(rad),0,np.sin(rad)],[0,1,0],[-np.sin(rad),0,np.cos(rad)]]
            m[:3,3]=[3,2,-1]
            np.testing.assert_allclose(transform_to_matrix(matrix_to_transform(m)),m,atol=1e-8)
    def test_invert_pose(self):
        bind=np.eye(4);bind[:3,3]=[0,.4,0]
        blender=np.eye(4);blender[:3,3]=[1,2,3]
        dest=np.eye(4);dest[:3,3]=[2,2,3]
        a=soma_to_character_matrix(1.4,[5,-2,.5],'-Y')
        soma=pose_world_to_soma(dest,blender,bind,a)
        np.testing.assert_allclose(soma_world_to_pose(soma,bind,blender,a),dest,atol=1e-9)
    def test_fk_roundtrip(self):
        mats=np.repeat(np.eye(4)[None],3,axis=0)
        mats[:,0,3]=[0,1,2]
        mats[:,1,3]=[0,1,2]
        parents=[-1,0,1]
        np.testing.assert_allclose(local_to_world(world_to_local(mats,parents),parents),mats)
    def test_invalid(self):
        with self.assertRaises(ValueError):check_affine(np.diag([-1,1,1,1]))
        with self.assertRaises(ValueError):world_to_local(np.repeat(np.eye(4)[None],2,axis=0),[1,0])

if __name__=='__main__':unittest.main()
