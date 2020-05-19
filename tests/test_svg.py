import os
import glimpse.svg
from .context import test_dir

def test_gets_image_coordinates():
    path = os.path.join(test_dir, 'simple.svg')
    xy = glimpse.svg.get_image_coordinates(path, imgsz=(10, 10))
    assert xy['path'] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy['polygon'] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy['rect'] == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    assert xy['polyline'] == [(0, 0), (1, 1)]
    assert xy['line'] == [(0, 0), (1, 1)]
    assert xy['circle'] == [(0, 0)]
    assert xy['image'] == [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    xy2 = glimpse.svg.get_image_coordinates(path, imgsz=(20, 20))
    assert xy2['path'] == [(xo * 2.0, yo * 2.0) for xo, yo in xy['path']]

def test_sets_element_attribute_as_tag():
    path = os.path.join(test_dir, 'simple.svg')
    xy = glimpse.svg.get_image_coordinates(path, key=None)
    xyid = glimpse.svg.get_image_coordinates(path, key='id')
    assert xy['path'] == xyid['land']
    assert xy['polygon'] == xyid['glacier']
