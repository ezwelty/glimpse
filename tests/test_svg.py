"""Tests of the svg module."""
import io
from typing import Tuple, Union

import glimpse.svg
import pytest


def test_errors_for_missing_svg() -> None:
    """Raises error if xml has no <svg> elements."""
    fp = io.StringIO("<xml />")
    with pytest.raises(ValueError):
        glimpse.svg.read(fp)


def test_errors_for_multiple_svg() -> None:
    """Raises error if xml has multiple <svg> elements."""
    fp = io.StringIO("<svg><svg /></svg>")
    with pytest.raises(ValueError):
        glimpse.svg.read(fp)


def test_errors_for_imgsz_and_missing_image() -> None:
    """Raises error if xml has no <image> elements and image scaling is requested."""
    fp = io.StringIO("<svg />")
    glimpse.svg.read(fp)
    fp.seek(0)
    with pytest.raises(ValueError):
        glimpse.svg.read(fp, imgsz=(12, 8))


def test_warns_for_multiple_image() -> None:
    """Raises warning if xml has multiple <image> elements."""
    image = '<image width="6" height="4" />'
    fp = io.StringIO(f"<svg>{image * 2}</svg>")
    with pytest.warns(UserWarning):
        glimpse.svg.read(fp)


@pytest.mark.parametrize(
    "element",
    [
        "<path d='M 1,1.0' />",
        "<polygon points='1,1.0' />",
        "<polyline points='1,1.0' />",
        "<line x1='1' y1='1.0' x2='2' y2='2' />",
        "<circle cx='1' cy='1.0' />",
        "<rect x='1' y='1.0' width='1' height='1' />",
    ],
)
def test_preserves_integers(element: str) -> None:
    """Preserves element coordinates as integer when possible."""
    fp = io.StringIO(f"<svg>{element}</svg>")
    coords = glimpse.svg.read(fp)
    x, y = coords[list(coords.keys())[0]][0]
    assert isinstance(x, int) and x == 1
    assert isinstance(y, float) and y == 1


@pytest.mark.parametrize(
    "s, xy",
    [
        ["1,-0.1", (1, -0.1)],
        ["1 -0.1", (1, -0.1)],
        ["1-0.1", (1, -0.1)],
        ["0.1.2", (0.1, 0.2)],
        ["1-1.2e-01", (1, -0.12)],
        ["1 1.2e+01", (1, 12)],
        ["1 1.2e01", (1, 12)],
        ["1 1.2e1", (1, 12)],
    ],
)
def test_parses_coordinate_formats(
    s: str, xy: Tuple[Union[int, float], Union[int, float]]
) -> None:
    """Parses all possible coordinate sequence formats."""
    fp = io.StringIO(f"<svg><path d='M {s}' /></svg>")
    coords = glimpse.svg.read(fp)
    assert coords["path"][0] == xy
    fp = io.StringIO(f"<svg><polyline points='{s}' /></svg>")
    coords = glimpse.svg.read(fp)
    assert coords["polyline"][0] == xy
    fp = io.StringIO(f"<svg><polygon points='{s}' /></svg>")
    coords = glimpse.svg.read(fp)
    assert coords["polygon"][0] == xy


@pytest.mark.parametrize(
    "cmd, dxy",
    [
        ["M 1,2", (1, 2)],
        ["L 1,2", (1, 2)],
        ["H 1", (1, 0)],
        ["V 2", (0, 2)],
        ["C 0,0 0,0 1,2", (1, 2)],
        ["S 0,0 1,2", (1, 2)],
        ["Q 0,0 1,2", (1, 2)],
        ["A 0 0 0 0 0 1,2", (1, 2)],
        ["Z", (0, 0)],
    ],
)
def test_parses_path_commands(
    cmd: str, dxy: Tuple[Union[int, float], Union[int, float]]
) -> None:
    """Parses all possible path commands."""
    xo, yo = 1, 2
    # Uppercase
    fp = io.StringIO(f"<svg><path d='M {xo},{yo} {cmd}' /></svg>")
    coords = glimpse.svg.read(fp)
    assert coords["path"][1] == (dxy[0] or xo, dxy[1] or yo)
    # Lowercase
    fp = io.StringIO(f"<svg><path d='M {xo},{yo} {cmd.lower()}' /></svg>")
    coords = glimpse.svg.read(fp)
    assert coords["path"][1] == (xo + dxy[0], yo + dxy[1])


def test_errors_for_invalid_path_command() -> None:
    """Raises error if path has an invalid command."""
    fp = io.StringIO("<svg><path d='X 0,0' /></svg>")
    with pytest.raises(ValueError):
        glimpse.svg.read(fp)


def test_parses_image_coordinates() -> None:
    """Parses coordinates with respect to <image> element."""
    xml = """
    <svg xmlns="http://www.w3.org/2000/svg">
        <path d="M 0,1 L 1,1 1,2 0,2 Z" />
        <polygon points="0,1 1,1 1,2 0,2" />
        <rect x="0" y="1" width="1" height="1" />
        <polyline points="-1,2 0,3" transform="matrix(1 0 0 1 1 -1)" />
        <line x1="0" y1="0.5" x2="0.5" y2="1" transform="scale(4,0.5)scale(0.5 4)" />
        <circle cx="-1" cy="2" r="1" transform="translate(1,-1)" />
        <image x="0" y="1" width="11" height="10" />
    </svg>
    """
    xy = glimpse.svg.read(io.StringIO(xml), imgsz=(11, 10))
    assert xy["path"] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy["polygon"] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy["rect"] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy["polyline"] == [(0, 0), (1, 1)]
    assert xy["line"] == [(0, 0), (1, 1)]
    assert xy["circle"] == [(0, 0)]
    assert xy["image"] == [(0, 0), (11, 0), (11, 10), (0, 10), (0, 0)]
    xy2 = glimpse.svg.read(io.StringIO(xml), imgsz=(22, 20))
    assert xy2["path"] == [(x * 2, y * 2) for x, y in xy["path"]]


def test_sets_element_attribute_as_key() -> None:
    """Sets element attribute as key."""
    txt = "<svg><g id='gcp'><circle id='rock' cx='0' cy='1'/></g></svg>"
    xy = glimpse.svg.read(io.StringIO(txt))
    xyid = glimpse.svg.read(io.StringIO(txt), key="id")
    assert xy["g"]["circle"] == xyid["gcp"]["rock"]


def test_groups_values_by_key() -> None:
    """Groups values with the same key into lists."""
    txt = "<svg><path id='gcp' d='M 0, 0' /><path id='gcp' d='M 0, 0' /></svg>"
    coords = glimpse.svg.read(io.StringIO(txt))
    assert coords["path"] == [[(0, 0)], [(0, 0)]]
    coords = glimpse.svg.read(io.StringIO(txt), key="id")
    assert coords["gcp"] == [[(0, 0)], [(0, 0)]]


def test_sets_svg_size() -> None:
    """Correctly sets <svg> width and height attributes."""
    iw, ih = "6", "4"
    sw, sh = "12", "8"
    # None
    e = glimpse.svg.svg()
    assert "width" not in e.attrib
    assert "height" not in e.attrib
    # Image
    e = glimpse.svg.svg(glimpse.svg.image(width=iw, height=ih))
    assert e.attrib["width"], e.attrib["height"] == (iw, ih)
    # Custom
    e = glimpse.svg.svg(glimpse.svg.image(width=iw, height=ih), width=sw, height=sh)
    assert e.attrib["width"], e.attrib["height"] == (sw, sh)


def test_writes_and_reads_coordinates() -> None:
    """Reads back coordinates written to file."""
    xy = [(0, 0), (100, 100), (200, 200)]
    e = glimpse.svg.svg(
        glimpse.svg.image(href="photo.jpg", width=800, height=536),
        glimpse.svg.g(glimpse.svg.path(d=xy), id="control"),
    )
    txt = glimpse.svg.write(e)
    coords = glimpse.svg.read(io.StringIO(txt), key="id")
    assert xy == coords["control"]["path"]
