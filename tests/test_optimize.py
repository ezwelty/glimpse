"""Tests of the optimize module."""
import copy
import os

import glimpse


def test_optimizes_camera_viewdir_with_ransac() -> None:
    """
    Optimizes relative camera view directions with RANSAC.

    Simulates a camera rotation by projecting the image into a rotated camera.
    Detects and matches keypoints between the original and synthetic image,
    filters the matches with RANSAC,
    and solves for the rotation.
    """
    path = os.path.join("tests", "AK10b_20141013_020336.JPG")
    imgA = glimpse.Image(path)
    imgA.cam.resize(0.5)
    imgB = copy.deepcopy(imgA)
    viewdir = (2, 2, 2)
    imgB.cam.viewdir = viewdir
    # Match features
    keypoints = [
        glimpse.optimize.detect_keypoints(a)
        for a in (imgA.read(), imgA.project(imgB.cam))
    ]
    uvs = glimpse.optimize.match_keypoints(*keypoints, max_ratio=0.8)
    matches = glimpse.optimize.Matches(cams=(imgA.cam, imgB.cam), uvs=uvs)
    model = glimpse.optimize.Cameras(
        cams=[imgB.cam], controls=[matches], cam_params=[{"viewdir": True}]
    )
    values = model.fit()
    assert any(abs(values - viewdir) > 0.1)
    rvalues, rindex = glimpse.optimize.ransac(
        model, n=12, max_error=5, min_inliers=10, iterations=10
    )
    assert all(abs(rvalues - viewdir) < 0.1)
