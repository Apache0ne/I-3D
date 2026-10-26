import unittest
import numpy as np
from i3d.axes import (SOMA_TO_BLENDER, BLENDER_TO_SOMA, check_rotation,
                       facing_rotation, fit_similarity_without_reflection,
                       soma_to_blender_points, transform_points, verify_laterality)

class Axes(unittest.TestCase):
    def test_conversion(self):
        np.testing.assert_allclose(SOMA_TO_BLENDER @ np.array([0,1,0]), [0,0,1])
        np.testing.assert_allclose(SOMA_TO_BLENDER @ np.array([0,0,1]), [0,-1,0])
        np.testing.assert_allclose(BLENDER_TO_SOMA @ SOMA_TO_BLENDER, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(SOMA_TO_BLENDER), 1.)
    def test_facing(self):
        for direction, vec in {'+X':[1,0,0],'-X':[-1,0,0],'+Y':[0,1,0],'-Y':[0,-1,0]}.items():
            r=facing_rotation(direction)
            np.testing.assert_allclose(r@np.array([1,0,0]),vec)
            self.assertAlmostEqual(np.linalg.det(r),1.)
    def test_no_reflection(self):
        with self.assertRaises(ValueError): check_rotation(np.diag([-1,1,1]))
        with self.assertRaises(ValueError): transform_points([[0,0,0]],np.eye(3),-1)
    def test_fit(self):
        rng=np.random.default_rng(37)
        x=rng.random((16,3)); x[:,2]*=2
        y=transform_points(x,facing_rotation('+Y'),1.7,[.4,-.3,1.2])
        fit=fit_similarity_without_reflection(x,y)
        np.testing.assert_allclose(transform_points(x,fit['rotation'],fit['scale'],fit['translation']),y,atol=1e-10)
    def test_laterality(self):
        correct={'LeftHand':[0,.4,1], 'RightHand':[0,-.4,1],
                 'LeftLeg':[0,.1,.7], 'RightLeg':[0,-.1,.7]}
        self.assertEqual(verify_laterality(correct,'+X'),2)
        wrong={k:([-v[0],-v[1],v[2]]) for k,v in correct.items()}
        with self.assertRaises(ValueError): verify_laterality(wrong,'+X')

if __name__ == '__main__':unittest.main()
