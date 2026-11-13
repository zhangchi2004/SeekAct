"""Uncrop monocular hand geometry into a measured full-camera model.

The upstream demo uses a nominal large focal length. With real intrinsics,
refit a rigid pose to the model's own full-image 2D projections. Residuals
measure consistency with that prediction, not ground-truth hand accuracy.
"""
import cv2
import numpy as np


def fit_full_camera(joints, pixels, intrinsic, initial_translation, box_size):
    xyz=np.asarray(joints,np.float64);uv=np.asarray(pixels,np.float64)
    k=np.asarray(intrinsic,np.float64);t=np.asarray(initial_translation,np.float64).reshape(3,1).copy()
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all() or t[2,0]<=0:return None,None
    ok,r,t=cv2.solvePnP(xyz,uv,k,None,np.zeros((3,1)),t,True,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:return None,None
    rr,_=cv2.Rodrigues(r);points=xyz@rr.T+t.reshape(1,3)
    projected=points@k.T;projected=projected[:,:2]/projected[:,2:]
    error=np.linalg.norm(projected-uv,axis=1);angle=float(np.linalg.norm(r)*180/np.pi)
    report={'median_projection_error_px':float(np.median(error)),'max_projection_error_px':float(error.max()),
            'global_orientation_correction_deg':angle,'depth_m':float(np.median(points[:,2]))}
    passed=(points[:,2]>.03).all() and (points[:,2]<3).all() and angle<=70
    passed &= np.median(error)<=max(3.,.03*box_size) and error.max()<=max(10.,.10*box_size)
    report['passed']=bool(passed)
    return (points.astype(np.float32) if passed else None),report
