import numpy as np
from scipy.spatial.transform import Rotation

from lime_pipeline.dense_actions import make_actions, rotation_6d


def fixture(moving_camera=False):
    n = 91
    camera = np.tile(np.eye(4), (n, 1, 1))
    if moving_camera:
        camera[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 35, n), degrees=True).as_matrix()
        camera[:, 0, 3] = np.linspace(0, .2, n)
    joint = np.zeros((n, 21, 3))
    joint[:, :, 2] = .5
    joint[:, 4, 0] = -.03
    joint[:, 8, 0] = .03; joint[:, 8, 1] = .06
    joint[:, 12, 0] = .03; joint[:, 12, 1] = .06
    return {"camera_to_world": camera, "hands": {
        "right": {"joints_world": joint, "confidence": np.ones((n, 21))}}, "intrinsics": np.eye(3)}


def test_stationary_world_hand_not_confounded_by_camera_motion():
    arrays, quality = make_actions(fixture(True), 0, 90, 30)
    assert quality["passed"]
    assert arrays["camera_action_9d"].shape == (30, 9)
    assert arrays["manipulation_action_7d"].shape == (30, 7)
    np.testing.assert_allclose(arrays["manipulation_action_7d"][:, :6], 0, atol=1e-6)
    np.testing.assert_allclose(arrays["camera_target_9d"][:3], [.2, 0, 0], atol=1e-6)
    assert len(arrays["source_frame"]) == len(arrays["camera_action_9d"]) + 1


def test_unobserved_frame_breaks_action_even_when_endpoints_valid():
    data = fixture()
    data["hands"]["right"]["confidence"][4] = 0
    arrays, quality = make_actions(data, 0, 90, 30)
    assert arrays["right_state_valid"][1:3].all()
    assert not arrays["right_action_valid"][1]
    assert np.isnan(arrays["right_action_7d"][1]).all()
    assert not arrays["left_action_valid"].any()


def test_column_order_and_rotation_direction():
    data = fixture()
    rotation = Rotation.from_euler("z", np.linspace(0, 30, 91), degrees=True).as_matrix()
    data["hands"]["right"]["joints_world"] = np.einsum("tij,tkj->tki", rotation, data["hands"]["right"]["joints_world"])
    arrays, quality = make_actions(data, 0, 90, 30)
    assert quality["passed"]
    assert (arrays["right_action_7d"][3:-3, 5] > 0).all()
    r = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
    np.testing.assert_allclose(rotation_6d(r), np.r_[r[:, 0], r[:, 1]])


def test_last_short_step_has_own_dt_no_fabricated_transition():
    arrays, _ = make_actions(fixture(), 0, 89, 30)
    assert arrays["source_frame"][-1] == 89
    np.testing.assert_allclose(arrays["action_dt_s"][-1], 2 / 30)
    assert len(arrays["action_dt_s"]) == 30
