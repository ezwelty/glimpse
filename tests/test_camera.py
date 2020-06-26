import glimpse
import numpy as np


def test_init_fmm(imgsz=(100, 100), fmm=(20, 10), sensorsz=(20, 10)):
    cam = glimpse.Camera(imgsz=imgsz, fmm=fmm, sensorsz=sensorsz)
    assert all(cam.f == fmm * cam.imgsz / sensorsz)


def test_resize(imgsz=(200, 100), f=(100, 100)):
    cam = glimpse.Camera(imgsz=imgsz, f=f)
    cam.resize(0.5)
    assert all(cam.imgsz * 2 == imgsz)
    cam.resize(1)
    assert all(cam.imgsz == imgsz)


def test_idealize(imgsz=(100, 100), f=(100, 100), c=1, k=1, p=1):
    cam = glimpse.Camera(imgsz=imgsz, f=f, c=c, k=k, p=p)
    cam.idealize()
    assert all(cam.c == 0)
    assert all(cam.k == 0)
    assert all(cam.p == 0)


def reprojection_errors(cam):
    """Compute reprojection errors at all pixel centers."""
    uv = cam.grid(step=1, snap=(0.5, 0.5), mode="points")
    dxyz = cam.uv_to_xyz(uv)
    puv = cam.xyz_to_uv(dxyz, directions=True)
    return np.linalg.norm(puv - uv, axis=1)


def test_reprojection_ideal(imgsz=(100, 100), f=(100, 100), tol=1e-14):
    cam = glimpse.Camera(imgsz=imgsz, f=f)
    err = reprojection_errors(cam)
    assert err.max() < tol


def test_reprojection_distorted(imgsz=(100, 100), f=(100, 100), tol=1e-12):
    # Positive k1
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=0.1)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Negative k1
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=-0.1)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Radial distortion only
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=[0.1] * 6)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Tangential distortion only
    cam = glimpse.Camera(imgsz=imgsz, f=f, p=[0.01] * 2)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # All distortion
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=[0.1] * 6, p=[0.01] * 2)
    err = reprojection_errors(cam)
    assert err.max() < tol


def test_reprojection_distorted_extreme(imgsz=(100, 100), f=(100, 100), tol=1e-12):
    # Positive k1
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=2)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Negative k1
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=-2)
    err = reprojection_errors(cam)
    assert err.max() < tol
