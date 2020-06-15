import datetime
import os

import glimpse

import numpy as np


def test_image_init_defaults():
    path = os.path.join("tests", "AK10b_20141013_020336.JPG")
    img = glimpse.Image(path)
    assert img.path == path
    assert img.datetime == img.exif.datetime
    np.testing.assert_equal(img.cam.imgsz, img.exif.imgsz)
    np.testing.assert_allclose(
        img.cam.f, img.exif.fmm * np.divide(img.exif.imgsz, img.exif.sensorsz)
    )


def test_image_init_custom():
    path = os.path.join("tests", "AK10b_20141013_020336.JPG")
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


def test_image_read():
    path = os.path.join("tests", "AK10b_20141013_020336.JPG")
    # Default size
    img = glimpse.Image(path)
    I = img.read()
    np.testing.assert_equal(I.shape[0:2][::-1], img.cam.imgsz)
    # Subset (cached)
    x, y, w, h = 0, 5, 100, 94
    box = x, y, x + w, y + h
    i = img.read(box, cache=True)
    assert i.shape[0:2][::-1] == (w, h)
    np.testing.assert_equal(i, I[y : (y + h), x : (x + w)])
    # Subset (not cached)
    img.I = None
    inc = img.read(box, cache=False)
    np.testing.assert_equal(i, inc)
    # Resize camera
    img.cam.resize(0.5)
    I = img.read()
    np.testing.assert_equal(I.shape[0:2][::-1], img.cam.imgsz)


def test_image_project():
    path = os.path.join("tests", "AK10b_20141013_020336.JPG")
    img = glimpse.Image(path)
    img.cam.resize(0.1)
    I = img.project(img.cam, method="nearest")
    np.testing.assert_equal(I[1:], img.read()[1:])
