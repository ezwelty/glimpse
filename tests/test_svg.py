from .context import *
from glimpse.imports import (np)

def test_parse_polyline():
    points = "20,100 40,60 70,80"
    expected = ((20, 100), (40, 60), (70, 80))
    result = glimpse.svg.parse_polyline(points)
    assert np.array_equal(expected, result)

def test_parse_polygon():
    points = "20,100 40,60 70,80"
    expected = ((20, 100), (40, 60), (70, 80), (20, 100))
    result = glimpse.svg.parse_polygon(points, closed=True)
    assert np.array_equal(expected, result)
    expected = ((20, 100), (40, 60), (70, 80))
    result = glimpse.svg.parse_polygon(points, closed=False)
    assert np.array_equal(expected, result)

def test_parse_line():
    args = dict(x1=20, y1=100, x2=40, y2=60)
    expected = ((20, 100), (40, 60))
    result = glimpse.svg.parse_line(**args)
    assert np.array_equal(expected, result)

def test_parse_path():
    d = "M 100 100 L 300 100 L 200 300 z"
    expected = ((100, 100), (300, 100), (200, 300), (100, 100))
    result = glimpse.svg.parse_path(d)
    assert np.array_equal(expected, result)
    d = "M100,200 C100,100 250,100 250,200 S400,300 400,200"
    expected = ((100, 200), (250, 200), (400, 200))
    result = glimpse.svg.parse_path(d)
    assert np.array_equal(expected, result)
    d = "M600,350 l 50,-25 a25,25 -30 0,1 50,-25"
    expected = ((600, 350), (600 + 50, 350 - 25), (600 + 50 + 50, 350 - 25 - 25))
    result = glimpse.svg.parse_path(d)
    assert np.array_equal(expected, result)

def test_parse_circle():
    args = dict(cx=1, cy=2)
    expected = [(1, 2)]
    result = glimpse.svg.parse_circle(**args)
    assert np.array_equal(expected, result)

def test_parse_svg():
    path = os.path.join(test_dir, 'AK10b_20141013_020336.svg')
    svg = glimpse.svg.parse_svg(path)
    assert set(svg) == {'glacier', 'gcp', 'land', 'horizon', 'coast'}
    assert set(svg['gcp']) == {'pier', 'orb', 'slant', 'beetle'}
    svg_1 = glimpse.svg.parse_svg(path, imgsz=(1, 1))
    svg_2 = glimpse.svg.parse_svg(path, imgsz=(2, 2))
    assert np.array_equal(svg_2['gcp']['orb'], 2 * svg_1['gcp']['orb'])
