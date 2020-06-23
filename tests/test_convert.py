"""Tests of the convert module."""
import os
from typing import Any, Dict

from glimpse import Camera
from glimpse.convert import (
    AgisoftCamera,
    MatlabCamera,
    OpenCVCamera,
    PhotoModelerCamera,
)
from glimpse.convert import Converter
import numpy as np
import pytest

# ---- Matlab ----


def test_reads_matlab_means_from_report() -> None:
    """Reads Matlab camera means from report."""
    means: Dict[str, Any] = {
        "fc": (3750.8, 3747.9),
        "cc": (2148.1, 1417.0),
        "alpha_c": 0.0,
        "kc": (-0.1, 0.1, 0.0, 0.0, -0.0),
        "imgsz": (4288, 2848),
    }
    path = os.path.join("tests", "Calib_Results.m")
    xcam_auto = MatlabCamera.from_report(path, sigmas=False)
    xcam_manual = MatlabCamera(**means)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_reads_matlab_sigmas_from_report() -> None:
    """Reads Matlab camera sigmas from report."""
    sigmas: Dict[str, Any] = {
        "fc": (1.80 / 3, 1.82 / 3),
        "cc": (1.0 / 3, 1.4 / 3),
        "alpha_c": 0,
        "kc": (0.002 / 3, 0.004 / 3, 0.000, 0.000, 0.000),
        "imgsz": (0, 0),
    }
    path = os.path.join("tests", "Calib_Results.m")
    xcam_auto = MatlabCamera.from_report(path, sigmas=True)
    xcam_manual = MatlabCamera(**sigmas)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_converts_to_matlab_and_back_exactly() -> None:
    """Converts to Matlab camera and back exactly."""
    # k[3:] must be zero
    cam = Camera(
        imgsz=(4288, 2848),
        f=(3100, 3200),
        c=(5, -4),
        k=(0.1, -0.05, 0.02),
        p=(0.03, 0.04),
    )
    xcam = MatlabCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-11)
    cam2 = xcam.to_camera()
    np.testing.assert_equal(cam.vector, cam2.vector)


def test_converts_to_matlab_and_back_by_optimization() -> None:
    """Converts to Matlab camera and back with optimized parameters."""
    # k[3:] must be non-zero
    cam = Camera(
        imgsz=(4288, 2848),
        f=(3100, 3200),
        c=(5, -4),
        k=(0.1, -0.05, 0.02, 0.003),
        p=(0.03, 0.04),
    )
    xcam_initial = MatlabCamera.from_camera(cam, optimize=False)
    residuals_initial = Converter(xcam_initial, cam).residuals()
    xcam = MatlabCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-2)
    # alpha_c must be non-zero (but small)
    xcam.alpha_c = 1e-6
    cam_initial = xcam.to_camera(optimize=False)
    residuals_initial = Converter(xcam, cam_initial).residuals()
    cam = xcam.to_camera()
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-2)


# ---- Agisoft ----


def test_reads_agisoft_from_xml() -> None:
    """Reads Agisoft camera from XML."""
    xml: Dict[str, Any] = {
        "imgsz": (4288, 2848),
        "f": 3570.0,
        "cx": 3.0,
        "cy": 4.0,
        "b2": 15.0,
        "k1": 0.1,
        "k2": -0.1,
        "k3": 0.01,
        "p1": 0.01,
        "p2": -0.01,
    }
    path = os.path.join("tests", "agisoft.xml")
    xcam_auto = AgisoftCamera.from_xml(path)
    xcam_manual = AgisoftCamera(**xml)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_converts_to_agisoft_and_back_exactly() -> None:
    """Converts to Agisoft camera and back exactly."""
    # k[3:] must be zero
    cam = Camera(
        imgsz=(4288, 2848),
        f=(3100, 3200),
        c=(5, -4),
        k=(0.1, -0.05, 0.02),
        p=(0.03, 0.04),
    )
    xcam = AgisoftCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-11)
    cam2 = xcam.to_camera()
    np.testing.assert_equal(cam.vector, cam2.vector)


def test_converts_to_agisoft_and_back_by_optimization() -> None:
    """Converts to Agisoft camera and back with optimized parameters."""
    # k[3:] must be non-zero
    cam = Camera(
        imgsz=(4288, 2848),
        f=(3100, 3200),
        c=(5, -4),
        k=(0.1, -0.05, 0.02, 0.003),
        p=(0.03, 0.04),
    )
    xcam_initial = AgisoftCamera.from_camera(cam, optimize=False)
    residuals_initial = Converter(xcam_initial, cam).residuals()
    xcam = AgisoftCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-2)
    # k4 or b2 must be non-zero (but small)
    xcam.k4 = 1e-7
    xcam.b2 = 1e-12
    cam_initial = xcam.to_camera(optimize=False)
    residuals_initial = Converter(xcam, cam_initial).residuals()
    cam = xcam.to_camera()
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-9)


# ---- PhotoModeler ----


def test_reads_photomodeler_means_from_report() -> None:
    """Reads PhotoModeler camera means from report."""
    imgsz = (4288, 2848)
    means = {
        "focal": 29.414069,
        "xp": 12.009446,
        "yp": 8.105847,
        "fw": 24.001371,
        "fh": 15.940299,
        "k1": 1.423e-004,
        "k2": -1.576e-007,
        "k3": 0.0,
        "p1": 3.703e-006,
        "p2": 0.0,
    }
    path = os.path.join("tests", "CalibrationReport.txt")
    xcam_auto = PhotoModelerCamera.from_report(path, imgsz=imgsz)
    xcam_manual = PhotoModelerCamera(imgsz=imgsz, **means)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_reads_photomodeler_sigmas_from_report() -> None:
    """Reads PhotoModeler camera sigmas from report."""
    imgsz = (4288, 2848)
    sigmas = {
        "focal": 0.001,
        "xp": 0.001,
        "yp": 7.1e-004,
        "fw": 1.7e-004,
        "fh": 0.0,
        "k1": 2.0e-007,
        "k2": 1.2e-009,
        "k3": 0.0,
        "p1": 3.5e-007,
        "p2": 0.0,
    }
    path = os.path.join("tests", "CalibrationReport.txt")
    xcam_auto = PhotoModelerCamera.from_report(path, imgsz=imgsz, sigmas=True)
    xcam_manual = PhotoModelerCamera(imgsz=imgsz, **sigmas)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_converts_to_photomodeler_and_back_exactly() -> None:
    """Converts to PhotoModeler camera and back exactly."""
    # fmm must be equal, k* and p* must be zero
    cam = Camera(
        imgsz=(4288, 2848), fmm=(3200, 3200), cmm=(0.5, -0.4), sensorsz=(35.1, 24.2),
    )
    xcam = PhotoModelerCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-12)
    cam2 = xcam.to_camera()
    np.testing.assert_allclose(cam.vector, cam2.vector, rtol=0, atol=1e-13)


def test_converts_to_photomodeler_and_back_by_optimization() -> None:
    """Converts to PhotoModeler camera and back with optimized parameters."""
    # fmm must be non-equal
    cam = Camera(
        imgsz=(4288, 2848), fmm=(3100, 3200), cmm=(0.5, -0.4), sensorsz=(35.1, 24.2)
    )
    xcam_initial = PhotoModelerCamera.from_camera(cam, optimize=False)
    residuals_initial = Converter(xcam_initial, cam).residuals()
    xcam = PhotoModelerCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-12)
    # k* or p* must be non-zero (but small)
    cam = Camera(
        imgsz=(4288, 2848),
        fmm=(3200, 3200),
        cmm=(0.5, -0.4),
        sensorsz=(35.1, 24.2),
        k=(0.1, -0.05),
        p=(0.03, 0.04),
    )
    xcam_initial = PhotoModelerCamera.from_camera(cam, optimize=False)
    residuals_initial = Converter(xcam_initial, cam).residuals()
    xcam = PhotoModelerCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-2)
    cam_initial = xcam.to_camera(optimize=False)
    residuals_initial = Converter(xcam, cam_initial).residuals()
    cam = xcam.to_camera()
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-2)


# ---- OpenCV ----


def test_reads_opencv_from_xml() -> None:
    """Reads OpenCV camera from XML."""
    imgsz = (4288, 2848)
    f = {"fx": 3.57e03, "fy": 3.58e03}
    c = {"cx": 2.15e03, "cy": 1.43e03}
    coeffs = {
        "k1": 1.1e-01,
        "k2": -1.2e-01,
        "p1": -9.98e-03,
        "p2": 9.99e-03,
        "k3": 1.0e-02,
        "k4": 1.1e-03,
        "k5": 1.2e-03,
        "k6": 1.3e-03,
        "s1": 1.0e-05,
        "s2": 1.1e-05,
        "s3": 1.2e-05,
        "s4": 1.3e-05,
    }
    arrays: Dict[str, Any] = {
        "cameraMatrix": [(f["fx"], 0, c["cx"]), (0, f["fy"], c["cy"]), (0, 0, 1)],
        "distCoeffs": list(coeffs.values()),
    }
    path = os.path.join("tests", "opencv.xml")
    xcam_auto = OpenCVCamera.from_xml(path, imgsz=imgsz)
    xcam_params = OpenCVCamera(imgsz=imgsz, **{**f, **c, **coeffs})
    assert xcam_auto.__dict__ == xcam_params.__dict__
    xcam_arrays = OpenCVCamera.from_arrays(imgsz=imgsz, **arrays)
    assert xcam_auto.__dict__ == xcam_arrays.__dict__


def test_converts_to_opencv_and_back_exactly() -> None:
    """Converts to OpenCV camera and back exactly."""
    cam = Camera(
        imgsz=(4288, 2848),
        f=(3100, 3200),
        c=(5, -4),
        k=(0.1, -0.05, 0.02, 0.003, 0.004, 0.005),
        p=(0.03, 0.04),
    )
    xcam = OpenCVCamera.from_camera(cam)
    residuals = Converter(xcam, cam).residuals()
    np.testing.assert_equal(residuals, 0)
    cam2 = xcam.to_camera()
    np.testing.assert_equal(cam.vector, cam2.vector)


def test_converts_to_opencv_and_back_by_optimization() -> None:
    """Converts to OpenCV camera and back with optimized parameters."""
    # Initial conversion is exact
    cam = Camera(
        imgsz=(4288, 2848),
        f=(3100, 3200),
        c=(5, -4),
        k=(0.1, -0.05, 0.02, 0.003, 0.004, 0.005),
        p=(0.03, 0.04),
    )
    xcam = OpenCVCamera.from_camera(cam)
    # s* must be non-zero
    xcam.s1 = 1e-5
    cam_initial = xcam.to_camera(optimize=False)
    residuals_initial = Converter(xcam, cam_initial).residuals()
    cam = xcam.to_camera()
    residuals = Converter(xcam, cam).residuals()
    assert np.sum(residuals ** 2) < np.sum(residuals_initial ** 2)
    np.testing.assert_allclose(residuals, 0, rtol=0, atol=1e-2)


# ---- Converter ----


def test_plots_residuals_as_quivers() -> None:
    """Plots residuals as quivers."""
    cam = Camera(imgsz=(4288, 2848), f=(3100, 3200), c=(5, -4), k=(0.1, -0.05, 0.02))
    xcam = MatlabCamera(imgsz=(4288, 2848), fc=(3100, 3200))
    converter = Converter(xcam, cam, uv=100)
    quivers = converter.plot()
    np.testing.assert_equal(quivers.X, converter.uv[:, 0])
    np.testing.assert_equal(quivers.Y, converter.uv[:, 1])
    residuals = converter.residuals()
    np.testing.assert_equal(quivers.U, residuals[:, 0])
    np.testing.assert_equal(quivers.V, residuals[:, 1])


def test_errors_for_unequal_image_size() -> None:
    """Raises error when camera image size are not equal."""
    cam = Camera(imgsz=(100, 200), f=(10, 10))
    xcam = MatlabCamera(imgsz=(100, 100), fc=(10, 10))
    with pytest.raises(ValueError):
        Converter(xcam, cam)
