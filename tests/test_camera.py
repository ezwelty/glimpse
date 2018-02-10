from .context import glimpse
from glimpse.imports import (np)

def test_init_fmm(fmm=(20, 10), sensorsz=(20, 10)):
    cam = glimpse.Camera(fmm=fmm, sensorsz=sensorsz)
    assert all(cam.f == fmm * cam.imgsz / sensorsz)

def test_resize(imgsz=(200, 100)):
    cam = glimpse.Camera(imgsz=imgsz)
    cam.resize(0.5)
    assert all(cam.imgsz * 2 == imgsz)
    cam.resize(2)
    assert all(cam.imgsz == imgsz)

def test_idealize(c=1, k=1, p=1):
    cam = glimpse.Camera(c=c, k=k, p=p)
    cam.idealize()
    assert all(cam.c == 0)
    assert all(cam.k == 0)
    assert all(cam.p == 0)

def reprojection_errors(cam):
    """Compute reprojection errors at all pixel centers."""
    uv = cam.centers()
    dxyz = cam.invproject(uv)
    puv = cam.project(dxyz, directions=True)
    return np.linalg.norm(puv - uv, axis=1)

def test_reprojection_ideal(tol=1e-14):
    cam = glimpse.Camera()
    err = reprojection_errors(cam)
    assert err.max() < tol

def test_reprojection_distorted(tol=1e-12):
    # Positive k1
    cam = glimpse.Camera(k=0.1)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Negative k1
    cam = glimpse.Camera(k=-0.1)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Radial distortion only
    cam = glimpse.Camera(k=[0.1] * 6)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Tangential distortion only
    cam = glimpse.Camera(p=[0.01] * 2)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # All distortion
    cam = glimpse.Camera(k=[0.1] * 6, p=[0.01] * 2)
    err = reprojection_errors(cam)
    assert err.max() < tol

def test_reprojection_distorted_extreme(tol=1e-12):
    # Positive k1
    cam = glimpse.Camera(k=2)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Negative k1
    cam = glimpse.Camera(k=-2)
    err = reprojection_errors(cam)
    assert err.max() < tol
