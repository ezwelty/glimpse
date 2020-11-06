"""Tests of the image module."""
import datetime
import os

import glimpse
import numpy as np

path = os.path.join("tests", "AK10b_20141013_020336.JPG")


def test_initializes_with_attributes_from_file() -> None:
    """Initializes with default attributes loaded from file."""
    img = glimpse.Image(path)
    assert img.path == path
    assert img.datetime == img.exif.datetime
    np.testing.assert_equal(img.cam.imgsz, img.exif.imgsz)
    np.testing.assert_allclose(
        img.cam.f, img.exif.fmm * np.divide(img.exif.imgsz, img.exif.sensorsz)
    )


def test_initializes_with_custom_attributes() -> None:
    """Initializes with custom attributes."""
    args = {
        "cam": {"imgsz": (100, 100), "sensorsz": (10, 10)},
        "datetime": datetime.datetime(2010, 1, 1),
    }
    img = glimpse.Image(path, **args)
    assert img.datetime == args["datetime"]
    np.testing.assert_equal(img.cam.imgsz, args["cam"]["imgsz"])
    np.testing.assert_allclose(
        img.cam.f,
        img.exif.fmm * np.divide(args["cam"]["imgsz"], args["cam"]["sensorsz"]),
    )


def test_reads_data_from_file() -> None:
    """Reads raster data from file."""
    # Default size
    img = glimpse.Image(path)
    A = img.read()
    np.testing.assert_equal(A.shape[0:2][::-1], img.cam.imgsz)
    # Subset (cached)
    x, y, w, h = 0, 5, 100, 94
    box = x, y, x + w, y + h
    a = img.read(box, cache=True)
    assert a.shape[0:2][::-1] == (w, h)
    np.testing.assert_equal(a, A[y : (y + h), x : (x + w)])
    # Subset (not cached)
    img = glimpse.Image(path)
    a_nc = img.read(box, cache=False)
    np.testing.assert_equal(a, a_nc)
    # Resize camera
    img.cam.resize(0.5)
    A = img.read()
    np.testing.assert_equal(A.shape[0:2][::-1], img.cam.imgsz)


def test_projects_into_camera() -> None:
    """Projects itself into a camera."""
    img = glimpse.Image(path)
    img.cam.resize(0.1)
    A = img.project(img.cam, method="nearest")
    np.testing.assert_equal(A[1:], img.read()[1:])
