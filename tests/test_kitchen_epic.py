import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from lime_pipeline.kitchen_epic import fit_scale,clip_actions


class EpicAlignmentTests(unittest.TestCase):
    def fixture(self):
        n=251;t=np.arange(n)/50;cam=np.repeat(np.eye(4)[None],n,axis=0)
        cam[:,:3,:3]=Rotation.from_euler('y',t*.1).as_matrix()
        cam[:,:3,3]=np.c_[t*.025,np.zeros(n),np.zeros(n)]
        # A stationary hand in a metric world: expected manipulation deltas zero.
        world=np.zeros((21,3));world[:,2]=.8
        world[0]=[0,.03,.8];world[4]=[.04,0,.8];world[8]=[-.04,0,.8];world[12]=[-.03,0,.8]
        j=np.einsum('nji,nkj->nki',cam[:,:3,:3],world[None]-cam[:,None,:3,3]*3)
        pairs=np.array([(a,a+25,3.,0,0,.05,.05/3) for a in range(0,200,5)])
        return {'camera_to_world':cam,'left_joints_camspace':j,'right_joints_camspace':j.copy(),
                'pose_status':np.ones(n,np.uint8),'reliable':np.ones(n,bool),'calibration_pairs':pairs,
                'intrinsics':np.eye(3)}, {'source_fps':50.}

    def test_metric_scale_and_fixed_reference(self):
        s,m=self.fixture();a,q=clip_actions(s,m,25,200)
        self.assertTrue(q['passed']);self.assertAlmostEqual(q['metric_scale_calibration']['metres_per_reconstruction_unit'],3.)
        np.testing.assert_array_equal(a['source_frame'][[0,-1]],[25,200])
        np.testing.assert_allclose(a['timestamp_s'],a['source_frame']/50)
        np.testing.assert_allclose(a['camera_pose_c0'][0],np.eye(4),atol=1e-6)
        np.testing.assert_allclose(a['left_action_7d'][:,:6],0,atol=1e-6)
        expected=np.linalg.inv(s['camera_to_world'][25,:3,:3]) @ ((s['camera_to_world'][200,:3,3]-s['camera_to_world'][25,:3,3])*3)
        np.testing.assert_allclose(a['camera_pose_c0'][-1,:3,3],expected,atol=1e-6)

    def test_absent_native_hand_is_not_bridged(self):
        s,m=self.fixture();s['left_joints_camspace'][27]=np.nan
        a,q=clip_actions(s,m,25,200)
        self.assertFalse(a['left_action_valid'][0]);self.assertTrue(np.isnan(a['left_action_7d'][0]).all())

    def test_insufficient_scale_and_camera_join_rejected(self):
        s,m=self.fixture();s['calibration_pairs']=s['calibration_pairs'][:2]
        self.assertIsNone(fit_scale(s,25,200,50))
        s,m=self.fixture();s['reliable'][40]=False
        a,q=clip_actions(s,m,25,200);self.assertIsNone(a);self.assertFalse(q['passed'])

    def test_scale_outliers(self):
        s,_=self.fixture();s['calibration_pairs'][-5:,2]=30
        scale=fit_scale(s,25,200,50)
        self.assertAlmostEqual(scale['metres_per_reconstruction_unit'],3)
        self.assertEqual(scale['inlier_pair_count'],35)


if __name__=='__main__':unittest.main()
