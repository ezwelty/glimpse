import os
import tempfile

import glimpse.svg


def test_gets_image_coordinates():
    path = os.path.join("tests", "simple.svg")
    xy = glimpse.svg.read(path, imgsz=(10, 10))
    assert xy["path"] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy["polygon"] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy["rect"] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy["polyline"] == [(0, 0), (1, 1)]
    assert xy["line"] == [(0, 0), (1, 1)]
    assert xy["circle"] == [(0, 0)]
    assert xy["image"] == [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    xy2 = glimpse.svg.read(path, imgsz=(20, 20))
    assert xy2["path"] == [(xo * 2.0, yo * 2.0) for xo, yo in xy["path"]]


def test_sets_element_attribute_as_tag():
    path = os.path.join("tests", "simple.svg")
    xy = glimpse.svg.read(path, key=None)
    xyid = glimpse.svg.read(path, key="id")
    assert xy["path"] == xyid["land"]
    assert xy["polygon"] == xyid["glacier"]


def test_writes_and_reads_coordinates():
    xy = [(0, 0), (100, 100), (200, 200)]
    e = glimpse.svg.svg(
        glimpse.svg.image(
            href="tests/AK10b_20141013_020336.JPG", width=800, height=536
        ),
        glimpse.svg.g(glimpse.svg.path(id="horizon", d=xy, stroke="red"), id="control"),
        width=800,
        height=536,
    )
    _, new = tempfile.mkstemp()
    glimpse.svg.write(e, new)
    coords = glimpse.svg.read(path=new, key="id")
    assert xy == coords["control"]["horizon"]
    os.remove(new)
