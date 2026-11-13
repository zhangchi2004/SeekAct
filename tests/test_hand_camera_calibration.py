import unittest
import cv2
import numpy as np
from lime_pipeline.hand_camera_calibration import fit_full_camera

class HandCalibrationTests(unittest.TestCase):
    def test_recovers_metric_rigid_pose_with_off_center_hand(self):
        rng=np.random.default_rng(4);j=rng.normal(0,.03,(21,3));r=np.array([.15,-.2,.04]);t=np.array([.25,.15,.5])
        k=np.array([[700,0,640],[0,700,360],[0,0,1.]])
        uv,_=cv2.projectPoints(j,r,t,k,None)
        result,report=fit_full_camera(j,uv[:,0],k,t*1.02,130)
        rr,_=cv2.Rodrigues(r);expected=j@rr.T+t
        self.assertTrue(report['passed']);np.testing.assert_allclose(result,expected,atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(np.diff(result,axis=0),axis=1),np.linalg.norm(np.diff(j,axis=0),axis=1),atol=1e-6)

    def test_missing_geometry_rejected(self):
        j=np.full((21,3),np.nan)
        self.assertEqual(fit_full_camera(j,np.zeros((21,2)),np.eye(3),[0,0,1],100),(None,None))

if __name__=='__main__':unittest.main()
