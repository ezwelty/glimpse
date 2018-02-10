from .context import glimpse
from glimpse.imports import (PIL, datetime)

def test_exif_empty():
    glimpse.Exif()

def test_exif_image():
    path = "AK10b_20141013_020336.JPG"
    exif = glimpse.Exif(path)
    assert all(exif.size == PIL.Image.open(path).size)
    assert exif.fmm == 20
    assert exif.make == "NIKON CORPORATION"
    assert exif.model == "NIKON D200"
    assert exif.iso == 200
    assert exif.shutter == 0.0125
    assert exif.aperture == 8
    assert exif.datetime == datetime(2014, 10, 13, 2, 3, 36, 280000)

def test_exif_subseconds():
    path = "AK10b_20141013_020336.JPG"
    exif = glimpse.Exif(path)
    subseconds = exif.get_tag('SubSecTimeOriginal')
    exif.set_tag('SubSecTimeOriginal', None)
    assert exif.datetime == datetime(2014, 10, 13, 2, 3, 36)
    microseconds = int(float('0.' + subseconds) * 10e6 / 10)
    exif.set_tag('SubSecTimeOriginal', subseconds)
    assert exif.datetime == datetime(2014, 10, 13, 2, 3, 36, microseconds)
