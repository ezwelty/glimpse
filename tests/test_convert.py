import os

import glimpse.convert
import numpy as np

# ---- Matlab ----

matlab_report_means = {
    "fc": [3750.8, 3747.9],
    "cc": [2148.1, 1417.0],
    "alpha_c": 0.0,
    "kc": [-0.1, 0.1, 0.0, 0.0, -0.0],
    "nx": 4288.0,
    "ny": 2848.0,
}

matlab_report_sigmas = {
    "fc": [1.80 / 3, 1.82 / 3],
    "cc": [1.0 / 3, 1.4 / 3],
    "alpha_c": 0,
    "kc": [0.002 / 3, 0.004 / 3, 0.000, 0.000, 0.000],
    "nx": None,
    "ny": None,
}


def test_matlab_report_means():
    path = os.path.join("tests", "Calib_Results.m")
    xcam_auto = glimpse.convert.MatlabCamera.from_report(path, sigmas=False)
    xcam_manual = glimpse.convert.MatlabCamera(**matlab_report_means)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_matlab_report_sigmas():
    path = os.path.join("tests", "Calib_Results.m")
    xcam_auto = glimpse.convert.MatlabCamera.from_report(path, sigmas=True)
    xcam_manual = glimpse.convert.MatlabCamera(**matlab_report_sigmas)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_matlab_camera_exact():
    # alpha_c must be zero
    xcam = glimpse.convert.MatlabCamera(
        nx=200, ny=200, fc=(200, 200), cc=(105, 95), kc=(0.1, -0.1, 0.01, 0.01, -0.01)
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-13, rtol=0)


def test_matlab_camera_estimate():
    # alpha_c should be small
    xcam = glimpse.convert.MatlabCamera(
        nx=200,
        ny=200,
        fc=(200, 200),
        cc=(105, 95),
        kc=(0.1, -0.1, 0.01, 0.01, -0.01),
        alpha_c=1e-10,
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-7, rtol=0)


# ---- Agisoft ----

agisoft_xml = {
    "width": 4288,
    "height": 2848,
    "f": 3570,
    "cx": 3,
    "cy": 4,
    "b2": 15,
    "k1": 0.1,
    "k2": -0.1,
    "k3": 0.01,
    "p1": 0.01,
    "p2": -0.01,
}


def test_agisoft_xml():
    path = os.path.join("tests", "agisoft.xml")
    xcam_auto = glimpse.convert.AgisoftCamera.from_xml(path)
    xcam_manual = glimpse.convert.AgisoftCamera(**agisoft_xml)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_agisoft_camera_exact():
    # k4 and b2 must be zero
    xcam = glimpse.convert.AgisoftCamera(
        width=200,
        height=200,
        f=200,
        cx=105,
        cy=95,
        k1=0.1,
        k2=-0.1,
        k3=0.01,
        p1=0.01,
        p2=-0.01,
        b1=5,
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-10, rtol=0)


def test_agisoft_camera_estimate():
    # b2 and k4 should be small
    xcam = glimpse.convert.AgisoftCamera(
        width=200,
        height=200,
        f=200,
        cx=105,
        cy=95,
        k1=0.1,
        k2=-0.1,
        k3=0.01,
        p1=0.01,
        p2=-0.01,
        b1=5,
        b2=1e-12,
        k4=1e-7,
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-6, rtol=0)


# ---- PhotoModeler ----

pm_imgsz = (4288, 2848)

pm_report_means = {
    "focal": 29.414069,
    "xp": 12.009446,
    "yp": 8.105847,
    "fw": 24.001371,
    "fh": 15.940299,
    "k1": 1.423e-004,
    "k2": -1.576e-007,
    "k3": 0,
    "p1": 3.703e-006,
    "p2": 0,
}

pm_report_sigmas = {
    "focal": 0.001,
    "xp": 0.001,
    "yp": 7.1e-004,
    "fw": 1.7e-004,
    "fh": None,
    "k1": 2.0e-007,
    "k2": 1.2e-009,
    "k3": None,
    "p1": 3.5e-007,
    "p2": None,
}


def test_photomodeler_report_means():
    path = os.path.join("tests", "CalibrationReport.txt")
    xcam_auto = glimpse.convert.PhotoModelerCamera.from_report(
        path, imgsz=pm_imgsz, sigmas=False
    )
    xcam_manual = glimpse.convert.PhotoModelerCamera(imgsz=pm_imgsz, **pm_report_means)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_photomodeler_report_sigmas():
    path = os.path.join("tests", "CalibrationReport.txt")
    xcam_auto = glimpse.convert.PhotoModelerCamera.from_report(
        path, imgsz=pm_imgsz, sigmas=True
    )
    xcam_manual = glimpse.convert.PhotoModelerCamera(imgsz=pm_imgsz, **pm_report_sigmas)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_photomodeler_camera_exact():
    # k* and p* must be zero
    xcam = glimpse.convert.PhotoModelerCamera(
        imgsz=(200, 200), focal=28, xp=12.2, yp=8.1, fw=24, fh=16
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-13, rtol=0)


def test_photomodeler_camera_estimate():
    # k* and p* should be small
    xcam = glimpse.convert.PhotoModelerCamera(
        imgsz=(200, 200),
        focal=28,
        xp=12.2,
        yp=8.1,
        fw=24,
        fh=16,
        k1=1e-6,
        k2=-1e-6,
        k3=1e-8,
        p1=1e-6,
        p2=1e-6,
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-2, rtol=0)


# ---- Matlab ----

opencv_imgsz = (4288, 2848)

opencv_xml = {
    "fx": 3.57e03,
    "fy": 3.58e03,
    "cx": 2.15e03,
    "cy": 1.43e03,
    "k1": 1.1e-01,
    "k2": -1.2e-01,
    "k3": 1.0e-02,
    "k4": 1.1e-03,
    "k5": 1.2e-03,
    "k6": 1.3e-03,
    "p1": -9.98e-03,
    "p2": 9.99e-03,
    "s1": 1.0e-05,
    "s2": 1.1e-05,
    "s3": 1.2e-05,
    "s4": 1.3e-05,
}


def test_opencv_xml():
    path = os.path.join("tests", "opencv.xml")
    xcam_auto = glimpse.convert.OpenCVCamera.from_xml(path, imgsz=opencv_imgsz)
    xcam_manual = glimpse.convert.OpenCVCamera(imgsz=opencv_imgsz, **opencv_xml)
    assert xcam_auto.__dict__ == xcam_manual.__dict__


def test_opencv_camera_exact():
    # s* must be zero
    xcam = glimpse.convert.OpenCVCamera(
        imgsz=(200, 200),
        fx=200,
        fy=200,
        cx=105,
        cy=95,
        k1=0.1,
        k2=-0.1,
        p1=0.01,
        p2=0.01,
        k3=-0.01,
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-13, rtol=0)


def test_opencv_camera_estimate():
    # s* should be small
    xcam = glimpse.convert.OpenCVCamera(
        imgsz=(200, 200),
        fx=200,
        fy=200,
        cx=105,
        cy=95,
        k1=0.1,
        k2=-0.1,
        p1=0.01,
        p2=0.01,
        k3=-0.01,
        s1=1e-5,
    )
    cam = xcam.as_camera()
    control = xcam._points(xcam, cam, step=10)
    residuals = control.predicted() - control.observed()
    np.testing.assert_allclose(residuals, 0, atol=1e-3, rtol=0)


# ---- General ----


def test_as_camera_sigma():
    xmean = glimpse.convert.MatlabCamera(**matlab_report_means)
    xsigma = glimpse.convert.MatlabCamera(**matlab_report_sigmas)
    mean, sigma = glimpse.convert.as_camera_sigma(xmean, xsigma)
    np.testing.assert_equal(mean.vector, xmean.as_camera().vector)
    np.testing.assert_allclose(sigma.f, xsigma.fc, atol=1e-1, rtol=0)
    np.testing.assert_allclose(sigma.c, xsigma.cc, atol=1e-1, rtol=0)
    np.testing.assert_allclose(sigma.k[0:2], xsigma.kc[0:2], atol=1e-1, rtol=0)
