import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lime_pipeline.camera_quality import validate_native_camera


@pytest.mark.parametrize("kind", ["translation", "rotation"])
def test_native_spike_cannot_hide_between_sampled_states(kind):
    camera = np.tile(np.eye(4), (10, 1, 1))
    if kind == "translation":
        camera[4, 0, 3] = .2
    else:
        camera[4, :3, :3] = Rotation.from_euler('y', 20, degrees=True).as_matrix()
    # States 0,3,6,9 look stationary, while frame 4 exceeds the native-rate gate.
    validate_native_camera(camera[::3], 10)
    with pytest.raises(ValueError, match="native camera jump"):
        validate_native_camera(camera, 30)


def test_smooth_metric_motion_passes_native_check():
    camera = np.tile(np.eye(4), (91, 1, 1))
    camera[:, 0, 3] = np.linspace(0, .3, 91)
    metrics = validate_native_camera(camera, 30)
    assert metrics['max_native_speed_mps'] == pytest.approx(.1)
