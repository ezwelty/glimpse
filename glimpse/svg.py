"""
Read image annotations from scalable vector graphics (svg) files.
"""
import re
import inspect
import xml.etree.ElementTree
from collections import defaultdict
import warnings

def _strip_etree_namespaces(tree):
    """
    Strip namespaces from tags and attributes.

    Arguments:
        tree (xml.etree.ElementTree)
    """
    regex = re.compile(r'\{.*\}')
    for e in tree.iter():
        e.tag = regex.sub('', e.tag)
        for key in e.attrib.keys():
            new_key = regex.sub('', key)
            new_value = regex.sub('', e.attrib.pop(key))
            e.attrib[new_key] = new_value

def get_image_coordinates(path, key=None, imgsz=None):
    """
    Get SVG element vertices as image coordinates.

    SVG element vertices are returned as image coordinates [(x, y), ...],
    where (0, 0) is the upper-left corner of the upper-left pixel of the image.
    If no `image` element is present, coordinates are returned relative to the
    SVG viewport.

    Limitations:

    - Does not support multiple `svg` elements.
    - Only extracts coordinates from elements `path` (ignoring curvature),
      `polyline`, `polygon`, `rect`, `circle` (as point), and `image` (as
      bounding box).
    - Only recognizes `svg` and `g` as grouping elements.
    - Does not support percent (e.g. "100%") or unit (e.g. "10px") dimensions.
    - Transform functions `rotate`, `skewX`, and `skewY` are not supported (see
      https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform).

    Arguments:
        path (str): Path to SVG file
        key (str): Name of the element attribute whose value should be used as
            the dictionary key. If `None` (default) or if the attribute is not
            defined, the SVG element tag is used (e.g. "path").
        imgsz (iterable): Target image size (nx, ny). If `None` (default), uses
            the width and height of the first `image` element (before any
            transformations).

    Returns:
        dict: Image coordinates [(x, y), ...].
            Dictionary keys are either the element tag (e.g. "path") or the value of
            the element `key` attribute (e.g. "id"). If multiple elements share the
            same tag, their values are given in a list (e.g. 'path': [[(x, y), ...],
            [(x, y), ...]]).
    """
    tree = xml.etree.ElementTree.parse(path)
    _strip_etree_namespaces(tree)
    # Find <svg> tags
    svgs = list(tree.iter('svg'))
    if not svgs:
        raise ValueError('No <svg> tag found')
    if len(svgs) > 1:
        raise ValueError('Multiple <svg> tags not supported')
    svg = svgs[0]
    # Check <image> tags
    images = list(tree.iter('image'))
    if imgsz and not images:
        raise ValueError('Cannot apply `imgsz` since no <image> found')
    if len(svgs) > 1:
        warnings.warn('Transforming coordinates to first of multiple <image>')
    # Iterate over tree
    img = {}
    def parse_elements(e, key=None, transform=''):
        nonlocal img
        # Choose element name for dictionary
        tag = e.get(key) or e.tag
        d = {tag: {}}
        # Grow transform
        transform += e.get('transform', '')
        # Parse coordinates
        if e.tag in ('image', 'path', 'polyline', 'polygon', 'line', 'circle', 'rect'):
            points = Points.from_element(e.tag, **e.attrib)
            bbox = points.bbox()
            points.transform(transform)
            d[tag] = points.xy
            if e.tag == 'image' and not img:
                img = {'o': bbox, 't': points.bbox()}
        elif e.tag in ('svg', 'g') and e:
            dd = defaultdict(list)
            for dc in [parse_elements(ee, key=key, transform=transform)
                for ee in e]:
                for k, v in dc.items():
                    dd[k].append(v)
            for k, v in dd.items():
                d[tag][k] = v[0] if len(v) == 1 else v
        return d
    points = parse_elements(svg, key=key)
    # Translate to image origin
    translate = 0, 0
    scale = 1, 1
    if img:
        x, y = img['t']['x'], img['t']['y']
        if (x, y) != (0, 0):
            translate = -x, -y
    # Scale to image size
    if imgsz is None and img:
        imgsz = img['o']['width'], img['o']['height']
    if imgsz and img:
        if imgsz != (img['t']['width'], img['t']['height']):
            scale = imgsz[0] / img['t']['width'], imgsz[1] / img['t']['height']
    def transform(e):
        keys = e.keys() if isinstance(e, dict) else range(len(e))
        for key in keys:
            if not e[key]:
                pass
            elif isinstance(e[key], list) and isinstance(e[key][0], tuple):
                e[key] = Points(e[key]).translate(*translate).scale(*scale).xy
            else:
                transform(e[key])
    transform(points)
    return points['svg']

def _chunks(x, n):
    """
    Generate a zip that returns sequential chunks.

    Incomplete trailing chunks (of length < n) are ignored.

    Arguments:
        x (iterable)
        n (int): Number of elements in each chunk

    Returns:
        zip: Zip object that returns sequential tuples of length `n`
            (x0, ..., xn-1), (xn, ..., x2n-1), ...
    """
    each = iter(x)
    return zip(*([each] * n))

def _num(string):
    """
    Cast string to integer or float.
    """
    try:
        return int(string)
    except ValueError:
        return float(string)

class Points:
    """
    Reader and writer of SVG element point coordinates.

    Attributes:
        xy (array-like): Point coordinates [(x, y), ...]
    """
    def __init__(self, xy):
        self.xy = xy

    def closed(self):
        """
        Whether the last point is equal to the first point.
        """
        if len(self.xy) > 1:
            return self.xy[0] == self.xy[-1]
        else:
            return True

    def bbox(self):
        """
        Return the box bounding the points.

        Returns:
            dict: Bounding box (x, y, width, height)
        """
        if not self.xy:
            return None
        xs, ys = [], []
        for x, y in self.xy:
            xs.append(x)
            ys.append(y)
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        return {'x': x, 'y': y, 'width': w, 'height': h}

    def scale(self, x, y=None):
        """
        Scale coordinates.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform

        Arguments:
            x (float): Scale in x
            y (float): Scale in y. Equal to `x` if not specified.

        Returns:
            Points: Points with scaled coordinates.
        """
        if y is None:
            y = x
        self.xy = [(xo * x, yo * y) for xo, yo in self.xy]
        return self

    def translate(self, x, y=0):
        """
        Translate coordinates.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform

        Arguments:
            x,y (float): Translation in x,y

        Returns:
            Points: Points with translated coordinates.
        """
        self.xy = [(xo + x, yo + y) for xo, yo in self.xy]
        return self

    def matrix(self, a, b, c, d, e, f):
        """
        Matrix transform coordinates.

        Transform original coordinates xo, yo such that x, y =
        a * xo + c * yo + e, b * xo + d * yo + f.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform

        Arguments:
            a,b,c,d,e,f (float): Matrix components

        Returns:
            Points: Points with transformed coordinates.
        """
        self.xy = [(a * xo + c * yo + e, b * xo + d * yo + f)
            for xo, yo in self.xy]
        return self

    def transform(self, transform):
        """
        Apply `transform` attribute.

        Supports only `matrix`, `scale`, and `translate` functions.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform

        Arguments:
            transform (str): `transform` attribute

        Returns:
            Points: Points with transformed coordinates.

        >>> Points([(3, 1)]).transform('translate(-1 1) scale(1,0.5)').xy
        [(2, 1.0)]
        >>> Points([(3, 1)]).transform('matrix(1 0 0 0.5 -1 0.5)').xy
        [(2, 1.0)]
        """
        for func, params in re.findall(r'([A-Za-z]+)\(([^\)]+)\)', transform):
            method = getattr(self, func, None)
            if not method:
                raise ValueError(
                    'Unsupported (or invalid) transform function:', func)
            args = [_num(s) for s in re.split(r'\s+|,', params)]
            method(*args)
        return self

    @staticmethod
    def _points_to_xy(points):
        numbers = re.findall(r'[0-9\.\-]+', points)
        return [(_num(x), _num(y)) for x, y in _chunks(numbers, 2)]

    @staticmethod
    def _xy_to_points(xy):
        return ' '.join(['{0},{1}'.format(x, y) for x, y in xy])

    @classmethod
    def from_polyline(cls, points=''):
        """
        From `polyline` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/points.

        Arguments:
            points (str): `polyline.points` attribute

        >>> Points.from_polyline('0,10 5,2.5 5,7.5 10,0').xy
        [(0, 10), (5, 2.5), (5, 7.5), (10, 0)]
        >>> Points.from_polyline('0,10 5').xy
        [(0, 10)]
        """
        xy = cls._points_to_xy(points)
        return cls(xy)

    def to_polyline(self):
        """
        To `polyline` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/points.

        >>> Points.from_polyline('0,10 5,2.5 5,7.5 10,0').to_polyline()
        {'points': '0,10 5,2.5 5,7.5 10,0'}
        """
        return {'points': self._xy_to_points(self.xy)}

    @classmethod
    def from_polygon(cls, points=''):
        """
        From `polygon` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/points.

        Arguments:
            points (str): `polygon.points` attribute

        >>> Points.from_polygon('0,10 5,2.5 5,7.5 10,0').xy
        [(0, 10), (5, 2.5), (5, 7.5), (10, 0), (0, 10)]
        """
        xy = cls._points_to_xy(points)
        pts = cls(xy)
        if not pts.closed():
            pts.xy.append(xy[0])
        return pts

    def to_polygon(self):
        """
        To `polygon` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/points.

        >>> Points.from_polygon('0,10 5,2.5 5,7.5 10,0').to_polygon()
        {'points': '0,10 5,2.5 5,7.5 10,0'}
        """
        xy = self.xy[:-1] if self.closed() else self.xy
        return {'points': self._xy_to_points(xy)}

    @classmethod
    def from_line(cls, x1=0, y1=0, x2=0, y2=0):
        """
        From `line` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/line

        Arguments:
            x1 (str): `line.x1` attribute
            y1 (str): `line.y1` attribute
            x2 (str): `line.x2` attribute
            y2 (str): `line.y2` attribute

        >>> Points.from_line(0, 0, 1, 1).xy
        [(0, 0), (1, 1)]
        """
        xy = [(_num(x1), _num(y1)), (_num(x2), _num(y2))]
        return cls(xy)

    def to_line(self):
        """
        To `line` attributes.

        Uses the first and last points as the line endpoints.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/line

        >>> Points.from_line(0, 0, 1, 1).to_line()
        {'x1': '0', 'y1': '0', 'x2': '1', 'y2': '1'}
        >>> Points([(0, 0), (2, 2), (1, 1)]).to_line()
        {'x1': '0', 'y1': '0', 'x2': '1', 'y2': '1'}
        """
        x1, y1 = [str(i) for i in (self.xy[0] if self.xy else (0, 0))]
        x2, y2 = [str(i) for i in (self.xy[-1] if self.xy else (0, 0))]
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    @classmethod
    def from_circle(cls, cx=0, cy=0):
        """
        From `circle` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/circle

        Arguments:
            cx (str): `circle.cx` attribute
            cy (str): `circle.cy` attribute

        >>> Points.from_circle(1, 1).xy
        [(1, 1)]
        """
        xy = [(_num(cx), _num(cy))]
        return cls(xy)

    def to_circle(self):
        """
        To `circle` attributes.

        Uses the first point as the circle center.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/circle

        >>> Points.from_circle(1, 1).to_circle()
        {'cx': '1', 'cy': '1'}
        >>> Points([(0, 0), (1, 1)]).to_circle()
        {'cx': '0', 'cy': '0'}
        """
        cx, cy = [str(i) for i in (self.xy[0] if self.xy else (0, 0))]
        return {'cx': cx, 'cy': cy}

    @classmethod
    def from_rect(cls, width, height, x=0, y=0):
        """
        From `rect` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/rect

        Arguments:
            width (str): `rect.width` attribute
            height (str): `rect.height` attribute
            x (str): `rect.x` attribute
            y (str): `rect.y` attribute

        >>> Points.from_rect(1, 1).xy
        [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        >>> Points.from_rect(1, 1, 1, 1).xy
        [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]
        """
        x, y, width, height = [_num(arg) for arg in (x, y, width, height)]
        xy = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        xy.append(xy[0])
        return cls(xy)

    def to_rect(self):
        """
        To `rect` attributes.

        Returns the bounding box of all points.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/rect

        >>> Points.from_rect(1, 1, 1, 1).to_rect()
        {'x': '1', 'y': '1', 'width': '1', 'height': '1'}
        """
        box = self.bbox()
        return {key: str(box[key]) for key in box}

    @classmethod
    def from_svg(cls, viewBox=None):
        """
        From `svg` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/svg

        Arguments:
            viewBox (str): `svg.viewBox` attribute
        """
        if viewBox:
            x, y, w, h = re.split(r'[\s,]+', viewBox)
            return cls.from_rect(w, h, x, y)
        else:
            return cls([])

    def to_svg(self):
        """
        To `svg` attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/svg
        """
        box = self.bbox()
        if box:
            viewbox = [str(box[key]) for key in ('x', 'y', 'width', 'height')]
            return {'viewBox': ' '.join(viewbox)}
        else:
            return {}

    @classmethod
    def from_image(cls, width, height, x=0, y=0):
        """
        From `image` attributes.

        Identical to `from_rect`.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/image

        Arguments:
            width (str): `image.width` attribute
            height (str): `image.height` attribute
            x (str): `image.x` attribute
            y (str): `image.y` attribute
        """
        return cls.from_rect(width, height, x, y)

    def to_image(self):
        """
        To `image` attributes.

        Identical to `to_rect`.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/image
        """
        return self.to_rect()

    @classmethod
    def from_path(cls, d=''):
        """
        From `path` attributes.

        Only vertices are preserved, so all curvature information is discarded.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/path

        Arguments:
            d (str): `path.d` attribute

        >>> Points.from_path('M 0,0 l 1,0 0,1 -1,-1').xy
        [(0, 0), (1, 0), (1, 1), (0, 0)]
        >>> Points.from_path('M 0,0 l 1,0 0,1 z').xy
        [(0, 0), (1, 0), (1, 1), (0, 0)]
        >>> Points.from_path('M 0,0 L 1,1 v 1 h 1 l -2,-2').xy
        [(0, 0), (1, 1), (1, 2), (2, 2), (0, 0)]
        >>> Points.from_path('M 0,0 m 1,1 c 0,0 0,0 1,1 s 0,0 1,1 t 1,1').xy
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        >>> Points.from_path('M 0,0 A 0 0 0 0 0 1,1 Z').xy
        [(0, 0), (1, 1), (0, 0)]
        """
        commands = re.findall(r'[a-zA-Z]+', d)
        parameters = [
            [_num(si) for si in re.split(r'\s+|,', re.sub(r'^\s+|\s$', '', s))]
            for s in re.findall(r'[0-9\-\.\,\s]+', d)]
        if commands and commands[-1] in ('Z', 'z'):
            parameters.append([])
        xy = []
        for cmd, params in zip(commands, parameters):
            # moveTo: M (x,y)+ | lineTo: L (x,y)+ | curveTo: T (x,y)+
            if cmd in ('M', 'L', 'T'):
                for x, y in _chunks(params, 2):
                    xy.append((x, y))
            # moveTo: m (dx,dy)+ | lineTo: l (dx,dy)+ | curveTo: T (dx,dy)+
            elif cmd in ('m', 'l', 't'):
                for dx, dy in _chunks(params, 2):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # lineTo: H (x)+
            elif cmd == 'H':
                for x, in _chunks(params, 1):
                    xy.append((x, xy[-1][1]))
            # lineTo: h (dx)+
            elif cmd == 'h':
                for dx, in _chunks(params, 1):
                    xy.append((xy[-1][0] + dx, xy[-1][1]))
            # lineTo: V (y)+
            elif cmd == 'V':
                for y, in _chunks(params, 1):
                    xy.append((xy[-1][0], y))
            # lineTo: v (dy)+
            elif cmd == 'v':
                for dy, in _chunks(params, 1):
                    xy.append((xy[-1][0], xy[-1][1] + dy))
            # curveTo: C (x1,y1 x2,y2 x,y)+
            elif cmd == 'C':
                for _, _, _, _, x, y in _chunks(params, 6):
                    xy.append((x, y))
            # curveTo: c (dx1,dy1 dx2,dy2 dx,dy)+
            elif cmd == 'c':
                for _, _, _, _, dx, dy in _chunks(params, 6):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # curveTo: S (x2,y2 x,y)+ | curveTo: Q (x1,y1 x,y)+
            elif cmd in ('S', 'Q'):
                for _, _, x, y in _chunks(params, 4):
                    xy.append((x, y))
            # curveTo: s (dx2,dy2 dx,dy)+ | curveTo: q (dx1,dy1 dx,dy)+
            elif cmd in ('s', 'q'):
                for _, _, dx, dy in _chunks(params, 4):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # arcTo: A (rx ry x-axis-rotation large-arc-flag sweep-flag x,y)+
            elif cmd == 'A':
                for _, _, _, _, _, x, y in _chunks(params, 7):
                    xy.append((x, y))
            # arcTo: a (rx ry x-axis-rotation large-arc-flag sweep-flag dx,dy)+
            elif cmd == 'a':
                for _, _, _, _, _, dx, dy in _chunks(params, 7):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # closePath: Z | z
            elif cmd in ('Z', 'z'):
                xy.append(xy[0])
            else:
                raise ValueError('Invalid command encountered:', cmd)
        return cls(xy)

    def to_path(self):
        """
        To `path` attributes.

        Uses only absolute moveTo (`M`) and lineTo (`L`) commands, as well as
        closePath (`Z`) if closed.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/path

        >>> Points.from_path('M 0,0 L 1,1').to_path()
        {'d': 'M 0,0 L 1,1'}
        >>> Points.from_path('M 0,0 L 1,1 0,0').to_path()
        {'d': 'M 0,0 L 1,1 Z'}
        >>> Points.from_path('M 0,0 L 1,1 0,0 2,2').to_path()
        {'d': 'M 0,0 L 1,1 0,0 2,2'}
        """
        commands = []
        for i, xy in enumerate(self.xy[:-1] if self.closed() else self.xy):
            if i == 0:
                cmd = 'M {0},{1}'.format(*xy)
            elif i == 1:
                cmd = 'L {0},{1}'.format(*xy)
            else:
                cmd = '{0},{1}'.format(*xy)
            commands.append(cmd)
        if self.closed():
            commands.append('Z')
        return {'d': ' '.join(commands)}

    @classmethod
    def from_element(cls, tag, **attrs):
        """
        From element tag and attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element

        Arguments:
            tag (str): Element tag (e.g. 'path')
            **attrs (dict): Element attributes

        >>> Points.from_element('path', d='M 0,0 L 1,1').xy
        [(0, 0), (1, 1)]
        >>> Points.from_element('polyline', points='0,0 1,1').xy
        [(0, 0), (1, 1)]
        >>> Points.from_element('polygon', points='0,0 1,1').xy
        [(0, 0), (1, 1), (0, 0)]
        >>> Points.from_element('line', x1=0, y1=0, x2=1, y2=1).xy
        [(0, 0), (1, 1)]
        >>> Points.from_element('circle', cx=1, cy=1, r=1).xy
        [(1, 1)]
        """
        method = getattr(cls, 'from_' + tag, None)
        if not method:
            raise ValueError('Unsupported (or invalid) element tag:', tag)
        args = inspect.getfullargspec(method).args[1:]
        attrs = {key: attrs[key] for key in attrs if key in args}
        return method(**attrs)

    def to_element(self, tag):
        """
        To element attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element

        Arguments:
            tag (str): Element tag (e.g. 'path')

        >>> Points.from_element('path', d='M 0,0 L 1,1').to_element('path')
        {'d': 'M 0,0 L 1,1'}
        >>> Points.from_element('polyline', points='0,0 1,1').to_element('polyline')
        {'points': '0,0 1,1'}
        >>> Points.from_element('polygon', points='0,0 1,1').to_element('polygon')
        {'points': '0,0 1,1'}
        >>> Points.from_element('line', x1=0, y1=0, x2=1, y2=1).to_element('line')
        {'x1': '0', 'y1': '0', 'x2': '1', 'y2': '1'}
        >>> Points.from_element('circle', cx=1, cy=1, r=1).to_element('circle')
        {'cx': '1', 'cy': '1'}
        """
        method = getattr(self, 'to_' + tag, None)
        if not method:
            raise ValueError('Unsupported (or invalid) element tag:', tag)
        return method()
