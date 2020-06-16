"""Tests of the exif module."""
import datetime
import os

import glimpse

path = os.path.join("tests", "AK10b_20141013_020336.JPG")


def test_drops_thumbnail() -> None:
    """Drops thumbnail tag groups."""
    exif = glimpse.Exif(path)
    assert "thumbnail" not in exif.tags
    assert "1st" not in exif.tags


def test_retains_thumbnail() -> None:
    """Retains thumbnail tag groups."""
    exif = glimpse.Exif(path, thumbnail=True)
    assert "thumbnail" in exif.tags
    assert "1st" in exif.tags


def test_returns_empty_properties() -> None:
    """Properties are empty when corresponding tags are missing."""
    exif = glimpse.Exif(path)
    exif.tags = {}
    assert exif.datetime is None
    assert exif.exposure is None
    assert exif.fmm is None
    assert exif.imgsz is None
    assert exif.iso is None
    assert exif.make is None
    assert exif.model is None
    assert exif.sensorsz is None


def test_returns_datetime_without_subsec() -> None:
    """Datetime property works when SubSecTimeOriginal tag is missing."""
    exif = glimpse.Exif(path)
    exif.tags["Exif"]["SubSecTimeOriginal"] = None
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36)


def test_dumps_with_thumbnail() -> None:
    """Dumps tags in the presence of a thumbnail image."""
    exif = glimpse.Exif(path, thumbnail=True)
    exif.dump()
