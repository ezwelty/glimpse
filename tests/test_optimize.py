import copy
import os

import glimpse
import glimpse.optimize

import numpy as np


def test_ransac_polynomial():
    data = np.column_stack(
        ((0, 1.1, 1.9, 3.1, 3.0, 0.1, 4.1), (0, 1.0, 2.0, 3.1, 0.1, 3.0, 4.0))
    )
    model = glimpse.optimize.Polynomial(data, deg=1)
    inliers = (0, 1, 2, 3, 6)
    rvalues, rindex = glimpse.optimize.ransac(
        model, sample_size=2, max_error=0.5, min_inliers=2, iterations=100
    )
    assert set(rindex) == set(inliers)


def test_ransac_camera_viewdir(tol=0.1):
    path = os.path.join("tests", "AK10b_20141013_020336.JPG")
    imgA = glimpse.Image(path)
    imgA.cam.resize(0.5)
    imgB = copy.deepcopy(imgA)
    viewdir = (2, 2, 2)
    imgB.cam.viewdir = viewdir
    imgB.I = imgA.project(imgB.cam)
    # Match features
    keypoints = [glimpse.optimize.detect_keypoints(img.read()) for img in (imgA, imgB)]
    uvA, uvB = glimpse.optimize.match_keypoints(
        keypoints[0], keypoints[1], max_ratio=0.8
    )
    matches = glimpse.optimize.Matches(cams=(imgA.cam, imgB.cam), uvs=(uvA, uvB))
    model = glimpse.optimize.Cameras(
        cams=[imgB.cam], controls=[matches], cam_params=[{"viewdir": True}]
    )
    values = model.fit()
    assert any(abs(values - viewdir) > tol)
    rvalues, rindex = glimpse.optimize.ransac(
        model, sample_size=12, max_error=5, min_inliers=10, iterations=10
    )
    assert all(abs(rvalues - viewdir) < tol)
