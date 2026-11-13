"""Native-rate camera checks, before temporal subsampling can hide jumps."""
import numpy as np
from scipy.spatial.transform import Rotation


def validate_native_camera(camera, fps, max_speed_mps=4.0, max_spin_dps=400.0):
    camera = np.asarray(camera, dtype=float)
    if camera.ndim != 3 or camera.shape[1:] != (4, 4) or len(camera) < 2:
        raise ValueError("native camera requires at least two 4x4 poses")
    if not np.isfinite(camera).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("nonfinite native camera or invalid FPS")
    rotation = camera[:, :3, :3]
    orthogonality = float(np.max(np.abs(rotation.transpose(0, 2, 1) @ rotation - np.eye(3))))
    if (orthogonality > 2e-3 or np.max(np.abs(np.linalg.det(rotation)-1)) > 2e-3
            or not np.allclose(camera[:, 3], [0, 0, 0, 1], atol=2e-3)):
        raise ValueError("native camera is not a rigid pose sequence")
    relative = np.linalg.inv(camera[:-1]) @ camera[1:]
    speed = float((np.linalg.norm(relative[:, :3, 3], axis=1)*fps).max())
    spin = float((np.rad2deg(Rotation.from_matrix(relative[:, :3, :3]).magnitude())*fps).max())
    if speed > max_speed_mps or spin > max_spin_dps:
        raise ValueError(f"native camera jump: {speed:.3f} m/s, {spin:.3f} deg/s")
    return {"max_native_speed_mps": speed, "max_native_spin_dps": spin,
            "max_rotation_orthogonality_error": orthogonality}
