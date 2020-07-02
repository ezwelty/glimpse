import copy
import os

import glimpse
import glimpse.optimize
import numpy as np


def test_ransac_camera_viewdir():
    path = os.path.join("tests", "AK10b_20141013_020336.JPG")
    imgA = glimpse.Image(path)
    imgA.cam.resize(0.5)
    imgB = copy.deepcopy(imgA)
    viewdir = (2, 2, 2)
    imgB.cam.viewdir = viewdir
    imgB.I = imgA.project(imgB.cam)
    # Match features
    keypoints = [glimpse.optimize.detect_keypoints(img.read()) for img in (imgA, imgB)]
    uvs = glimpse.optimize.match_keypoints(*keypoints, max_ratio=0.8)
    matches = glimpse.optimize.Matches(cams=(imgA.cam, imgB.cam), uvs=uvs)
    model = glimpse.optimize.Cameras(
        cams=[imgB.cam], controls=[matches], cam_params=[{"viewdir": True}]
    )
    values = model.fit()
    assert any(abs(values - viewdir) > 0.1)
    rvalues, rindex = glimpse.optimize.ransac(
        model, sample_size=12, max_error=5, min_inliers=10, iterations=10
    )
    assert all(abs(rvalues - viewdir) < 0.1)
