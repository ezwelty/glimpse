# ! pip install pytest
import numpy as np
import Camera
reload(Camera)

def test_reprojection_ideal():
    cam = Camera.Camera(xyz=[1, 2, -3], viewdir=[10, 20, -30])
    uv = np.random.rand(1000, 2) * cam.imgsz
    dxyz = cam.invproject(uv)
    uv2 = cam.project(dxyz, directions=True)
    assert np.abs(uv - uv2).max() < 1e-13
    
def test_reprojection_distorted():
    cam = Camera.Camera(xyz=[1, 2, -3], viewdir=[10, 20, -30], k = [0.1, -0.1] * 3, p = [0.01, -0.01])
    uv = np.random.rand(1000, 2) * cam.imgsz
    dxyz = cam.invproject(uv)
    uv2 = cam.project(dxyz, directions=True)
    assert np.abs(uv - uv2).max() < 0.2
