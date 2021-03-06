"""Tests of the camera module."""
import numpy as np

import glimpse


def test_converts_millimeter_focal_length_to_pixels() -> None:
    """Converts millimeter focal length and sensor size to pixel focal length."""
    fmm = (20, 10)
    sensorsz = (20, 10)
    cam = glimpse.Camera(imgsz=(100, 100), fmm=fmm, sensorsz=sensorsz)
    assert all(cam.f == fmm * cam.imgsz / sensorsz)


def test_resizes_image() -> None:
    """Resizes the image."""
    imgsz = (200, 100)
    cam = glimpse.Camera(imgsz=imgsz, f=(100, 100))
    cam.resize(0.5)
    assert all(cam.imgsz * 2 == imgsz)
    cam.resize(1)
    assert all(cam.imgsz == imgsz)


def test_idealizes_distortion() -> None:
    """Sets distortion coefficients to zero."""
    cam = glimpse.Camera(imgsz=(100, 100), f=(100, 100), c=1, k=1, p=1)
    cam.idealize()
    assert all(cam.c == 0)
    assert all(cam.k == 0)
    assert all(cam.p == 0)


def reprojection_errors(cam: glimpse.Camera) -> np.ndarray:
    """Compute reprojection errors at all pixel centers."""
    uv = cam.grid(step=1, snap=(0.5, 0.5), mode="points")
    dxyz = cam.uv_to_xyz(uv)
    puv = cam.xyz_to_uv(dxyz, directions=True)
    return np.linalg.norm(puv - uv, axis=1)


def test_reprojects_without_distortion() -> None:
    """Reprojects image points in the absence of distortion."""
    cam = glimpse.Camera(imgsz=(100, 100), f=(100, 100))
    err = reprojection_errors(cam)
    assert err.max() < 1e-14


def test_reprojects_with_distortion() -> None:
    """Reprojects image points in the presence of distortion."""
    imgsz = (100, 100)
    f = (100, 100)
    tol = 1e-12
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


def test_reprojects_with_extreme_distortion() -> None:
    """Reprojects image points in the presence of extreme distortion."""
    imgsz = (100, 100)
    f = (100, 100)
    tol = 1e-12
    # Positive k1
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=2)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Negative k1
    cam = glimpse.Camera(imgsz=imgsz, f=f, k=-2)
    err = reprojection_errors(cam)
    assert err.max() < tol
