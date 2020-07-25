"""Read and write image annotations in scalable vector graphics (svg) files."""
from collections import defaultdict
import copy
import inspect
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union
import warnings
import xml.etree.ElementTree as ET

from typing_extensions import TypedDict

Number = Union[int, float]
Numeric = Union[str, Number]
Coordinates = List[Tuple[Number, Number]]
Box = TypedDict("Box", {"x": Number, "y": Number, "width": Number, "height": Number})
Polyline = TypedDict("Polyline", {"points": str})
Polygon = TypedDict("Polygon", {"points": str})
Line = TypedDict("Line", {"x1": str, "y1": str, "x2": str, "y2": str})
Circle = TypedDict("Circle", {"cx": str, "cy": str})
Rect = TypedDict("Rect", {"x": str, "y": str, "width": str, "height": str})
SVG = TypedDict("SVG", {"viewBox": str}, total=False)
Path = TypedDict("Path", {"d": str})

COORD_REGEX = re.compile(
    r"(?:\+|\-)?(?:\.[0-9]+|[0-9]+(?:\.[0-9]+)?)(?:[Ee][+-]?[0-9]+)?"
)


def _strip_etree_namespaces(tree: ET.ElementTree) -> None:
    """Strip namespaces from tags and attributes."""
    regex = re.compile(r"\{.*\}")
    for e in tree.iter():
        e.tag = regex.sub("", e.tag)
        attrib = {}
        for key in e.attrib:
            new_key = regex.sub("", key)
            new_value = regex.sub("", e.attrib[key])
            attrib[new_key] = new_value
        e.attrib = attrib


def read(
    path: Union[str, TextIO], key: str = None, imgsz: Tuple[int, int] = None
) -> dict:
    """
    Get SVG element vertices as image coordinates.

    SVG element vertices are returned as image coordinates [(x, y), ...],
    where (0, 0) is the upper-left corner of the upper-left pixel of the image.
    If no `image` element is present, coordinates are returned relative to the
    SVG viewport.

    Limitations:

    - Does not support multiple `svg` elements.
    - Only extracts coordinates from elements `path` (ignoring curvature),
      `polyline`, `polygon`, `rect`, `circle` (as center point), and `image` (as
      bounding box).
    - Only recognizes `svg` and `g` as grouping elements.
    - Does not support percent (e.g. "100%") or unit (e.g. "10px") dimensions.
    - Only supports transform functions `translate`, `scale`, and `matrix` (see
      https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform).

    Arguments:
        path: Path or file object pointing to the SVG file
        key: Name of the element attribute whose value should be used as
            the dictionary key. If `None` (default) or if the attribute is not
            defined, the SVG element tag is used (e.g. "path").
        imgsz: Target image size (nx, ny). If `None` (default), uses
            the width and height of the last (top) `image` element (before any
            transformations), if present.

    Returns:
        dict: Image coordinates [(x, y), ...].
            Dictionary keys are either the element tag (e.g. "path") or the value of
            the element `key` attribute (e.g. "id"). If multiple elements share the
            same tag, their values are given in a list (e.g. 'path': [[(x, y), ...],
            [(x, y), ...]]).

    Raises:
        ValueError: No or multiple <svg> tag(s) found.
        ValueError: No <image> tag found and imgsz not provided.

    Example:
        >>> import io
        >>> xml = '''
        >>> <svg xmlns="http://www.w3.org/2000/svg">
        >>>     <image width="6" height="4" />
        >>>     <g id="gcp">
        >>>         <circle id="flag" cx="1" cy="2" />
        >>>         <circle id="cairn" cx="1.1" cy="1.5" />
        >>>     </g>
        >>>     <g id="horizon">
        >>>         <path d="M 0,0 L 1,0" />
        >>>         <path d="M 2,1 L 3,1" />
        >>>     </g>
        >>> </svg>
        >>> '''
        >>> xy = read(io.StringIO(xml), key='id')
        >>> xy['gcp']
        {'flag': [(1, 2)], 'cairn': [(1.1, 1.5)]}
        >>> xy['horizon']['path']
        [[(0, 0), (1, 0)], [(2, 1), (3, 1)]]
    """
    tree = ET.parse(path)
    _strip_etree_namespaces(tree)
    # Find <svg> tags
    svgs = list(tree.iter("svg"))
    if not svgs:
        raise ValueError("No <svg> tag found")
    if len(svgs) > 1:
        raise ValueError("Multiple <svg> tags not supported")
    svg = svgs[0]
    # Check <image> tags
    images = list(tree.iter("image"))
    if imgsz is not None and not images:
        raise ValueError("Cannot apply `imgsz` since no <image> found")
    if len(images) > 1:
        warnings.warn("Transforming coordinates to last (top) of multiple <image>")
    # Iterate over tree
    img = {}

    def parse_elements(e: ET.Element, key: str = None, transform: str = "") -> dict:
        nonlocal img
        # Choose element name for dictionary
        tag = (e.get(key) if key else None) or e.tag
        d: Dict[str, Any] = {tag: {}}
        # Grow transform
        transform += e.get("transform", "")
        # Parse coordinates
        if e.tag in ("image", "path", "polyline", "polygon", "line", "circle", "rect"):
            points = Points.from_element(e.tag, **e.attrib)
            bbox = points.bbox()
            points = points.transform(transform)
            d[tag] = points.xy
            if e.tag == "image":
                img = {"o": bbox, "t": points.bbox()}
        elif e.tag in ("svg", "g") and e:
            dd = defaultdict(list)
            for dc in [parse_elements(ee, key=key, transform=transform) for ee in e]:
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
        x, y = img["t"]["x"], img["t"]["y"]
        if (x, y) != (0, 0):
            translate = -x, -y
    # Scale to image size
    if imgsz is None and img:
        imgsz = img["o"]["width"], img["o"]["height"]
    if imgsz is not None and img:
        if imgsz[0] != img["t"]["width"] or imgsz[1] != img["t"]["height"]:
            scale = imgsz[0] / img["t"]["width"], imgsz[1] / img["t"]["height"]

    def transform(e: Union[dict, list]) -> None:
        keys = e.keys() if isinstance(e, dict) else range(len(e))
        for key in keys:
            if not e[key]:
                pass
            elif isinstance(e[key], list) and isinstance(e[key][0], tuple):
                e[key] = Points(e[key]).translate(*translate).scale(*scale).xy
            else:
                transform(e[key])

    transform(points)
    return next(iter(points.values()))


def _chunks(x: Sequence, n: int) -> Iterable:
    """
    Generate a zip that returns sequential chunks.

    Incomplete trailing chunks (of length < n) are ignored.

    Arguments:
        x: Sequence from which to build chunks
        n: Number of items in each chunk

    Returns:
        Zip object that returns sequential tuples of length `n`
            (x0, ..., xn-1), (xn, ..., x2n-1), ...
    """
    each = iter(x)
    return zip(*([each] * n))


def _num(x: Numeric) -> Number:
    """
    Cast to integer or float.

    Arguments:
        x: Value to cast as number.

    Example:
        >>> _num('1')
        1
        >>> _num(1)
        1
        >>> _num('1.0')
        1.0
        >>> _num(1.0)
        1.0
    """
    if isinstance(x, str):
        try:
            return int(x)
        except ValueError:
            return float(x)
    return x


def svg(*children: ET.Element, **attrib: str,) -> ET.Element:
    """
    Create `svg` element.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/element/svg.

    If not provided, attributes `width` and `height` are populated to fit the last
    <image> child element, if present.

    Arguments:
        *children: Child elements
        **attrib: Additional element attributes

    Returns:
        <svg> element with attributes.

    Example:
        >>> e = svg(path(), image(width=12, height=8))
        >>> e.tag
        'svg'
        >>> len(e)
        2
        >>> e.attrib
        {'xmlns': '.../svg', 'xmlns:xlink': '.../xlink', 'width': '12', 'height': '8'}
    """
    e = ET.Element("svg")
    e.extend(children)
    if "width" not in attrib and "height" not in attrib:
        images = list(e.iter("image"))
        if images:
            width, height = images[-1].get("width"), images[-1].get("height")
            if width and height:
                attrib = {"width": width, "height": height, **attrib}
    e.attrib = {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
        **attrib,
    }
    return e


def g(*children: ET.Element, **attrib: str) -> ET.Element:
    """
    Create `g` element.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/element/g.

    Arguments:
        *children: Child elements
        **attrib: Element attributes

    Returns:
        <g> element with attributes.

    Example:
        >>> e = g(path(), id='horizon')
        >>> e.tag
        'g'
        >>> len(e)
        1
        >>> e.attrib
        {'id': 'horizon'}
    """
    e = ET.Element("g", attrib=attrib)
    e.extend(children)
    return e


def image(
    width: Numeric, height: Numeric, href: str = None, **attrib: str
) -> ET.Element:
    """
    Create `image` element.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/element/image.

    Arguments:
        width: Display width of the image
        height: Display height of the image
        href: Path to image file (either absolute or relative to target SVG path)
        **attrib: Additional element attributes

    Returns:
        <image> element with attributes.

    Example:
        >>> e = image(width=12, height=8, href='photo.jpg')
        >>> e.tag
        'image'
        >>> e.attrib
        {'width': '12', 'height': '8', 'xlink:href': 'photo.jpg'}
    """
    optional = {"xlink:href": href} if href else {}
    attrib = {"width": str(width), "height": str(height), **optional, **attrib}
    return ET.Element("image", attrib=attrib)


def path(d: Union[str, Coordinates] = "", **attrib: str) -> ET.Element:
    """
    Create `path` element.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/path.

    Arguments:
        d: Shape of the path. Either pre-formatted as a str (e.g. 'M 0,0 L 1,1')
            or an iterable of point coordinates (e.g. [(0, 0), (1, 1)]). See
            https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d.
        **attrib: Additional element attributes

    Returns:
        <path> element with attributes.

    Example:
        >>> e = path(d=[(0, 0), (0, 1), (1, 1)], id='horizon')
        >>> e.tag
        'path'
        >>> e.attrib
        {'d': 'M 0,0 L 0,1 1,1', 'id': 'horizon'}
    """
    if not isinstance(d, str):
        d = Points(d).to_element("path")["d"]
    attrib = {"d": d, **attrib}
    return ET.Element("path", attrib=attrib)


def _indent_etree(
    e: ET.Element, level: int = 0, indent: Union[int, str] = None, last: bool = False,
) -> None:
    if indent is None:
        sep, tab = "", ""
    else:
        sep, tab = "\n", (indent if isinstance(indent, str) else indent * " ")
    if len(e):
        if not e.text or not e.text.strip():
            e.text = sep + tab * (level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = sep + tab * level
        for i, child in enumerate(e, start=1):
            _indent_etree(child, level=level + 1, indent=indent, last=i == len(e))
        if not e.tail or not e.tail.strip():
            e.tail = sep + tab * (level - 1)
    else:
        if level and (not e.tail or not e.tail.strip()):
            e.tail = sep + tab * (level - last)
    if level == 0:
        e.tail = None


def _etree_to_string(e: ET.Element, indent: Union[int, str] = None) -> str:
    e = copy.deepcopy(e)
    _indent_etree(e, indent=indent)
    return ET.tostring(e, encoding="unicode")


def write(
    e: ET.Element, path: str = None, indent: Union[int, str] = None
) -> Optional[str]:
    r"""
    Returns XML as a string or writes it to file.

    Arguments:
        e: Element to write
        path: Path to file
        indent: If an integer or string, XML elements are pretty-printed with that
            indent level.
            A negative integer, 0, or "" only inserts line breaks.
            A positive integer indents that many spaces per level.
            A string (e.g. "\t") indents that string per level.
            `None` (the default) prints on a single line.

    Returns:
        String representation of XML (if `path` is not provided).

    Example:
        >>> children = (
        >>>     path([(0, 0), (1, 1)], id='horizon'),
        >>>     image(href='photo.jpeg', width=12, height=8)
        >>> )
        >>> e = svg(*children)
        >>> print(write(e, indent=4))
        <svg height="8" width="12" xmlns=".../svg" xmlns:xlink=".../xlink">
            <path d="M 0,0 L 1,1" id="horizon" />
            <image height="8" width="12" xlink:href="photo.jpeg" />
        </svg>
    """
    txt = _etree_to_string(e, indent=indent)
    if not path:
        return txt
    with open(path, "w") as fp:
        fp.write(txt)
    return None


class Points:
    """
    Reader and writer of SVG element point coordinates.

    Attributes:
        xy: Point coordinates [(x, y), ...]
    """

    def __init__(self, xy: Coordinates) -> None:
        self.xy = xy

    def closed(self) -> bool:
        """
        Test whether the last point is equal to the first point.

        Example:
            >>> Points([(0, 0), (1, 1), (2, 2)]).closed()
            False
            >>> Points([(0, 0), (1, 1), (0, 0)]).closed()
            True
            >>> Points([(0, 0)]).closed()
            True
            >>> Points([]).closed()
            True
        """
        if len(self.xy) > 1:
            return self.xy[0] == self.xy[-1]
        return True

    def bbox(self) -> Optional[Box]:
        """
        Return the box bounding the points.

        Example:
            >>> Points([(0, 0), (1, 1)]).bbox()
            {'x': 0, 'y': 0, 'width': 1, 'height': 1}
            >>> Points([]).bbox() is None
            True
        """
        if not self.xy:
            return None
        xs, ys = [], []
        for x, y in self.xy:
            xs.append(x)
            ys.append(y)
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        return {"x": x, "y": y, "width": w, "height": h}

    def scale(self, x: Number, y: Number = None) -> "Points":
        """
        Scale coordinates.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform.

        Arguments:
            x: Scale in x
            y: Scale in y. Equal to `x` if not specified.

        Example:
            >>> points = Points([(1, 2)])
            >>> points.scale(2).xy
            [(2, 4)]
            >>> points.scale(2, 1).xy
            [(2, 2)]
        """
        if y is None:
            y = x
        xy = [(xo * x, yo * y) for xo, yo in self.xy]
        return type(self)(xy)

    def translate(self, x: Number, y: Number = 0) -> "Points":
        """
        Translate coordinates.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform.

        Arguments:
            x: Translation in x
            y: Translation in y

        Example:
            >>> Points([(1, 2)]).translate(1, 2).xy
            [(2, 4)]
        """
        xy = [(xo + x, yo + y) for xo, yo in self.xy]
        return type(self)(xy)

    def matrix(
        self, a: Number, b: Number, c: Number, d: Number, e: Number, f: Number
    ) -> "Points":
        """
        Matrix transform coordinates.

        Transform original coordinates xo, yo such that x, y =
        a * xo + c * yo + e, b * xo + d * yo + f.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform.

        Arguments:
            a: First component (0, 0)
            b: Second component (1, 0)
            c: Third component (0, 1)
            d: Fourth component (1, 1)
            e: Fifth component (0, 2)
            f: Sixth component (1, 2)

        Example:
            >>> Points([(1, 2)]).matrix(1, 2, 3, 4, 5, 6).xy
            [(12, 16)]
        """
        xy = [(a * xo + c * yo + e, b * xo + d * yo + f) for xo, yo in self.xy]
        return type(self)(xy)

    def transform(self, transform: str) -> "Points":
        """
        Apply `transform` attribute to coordinates.

        Supports only `matrix`, `scale`, and `translate` functions.
        See https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform.

        Arguments:
            transform (str): `transform` attribute

        Raises:
            ValueError: Unsupported or invalid transform function.

        Example:
            >>> points = Points([(1, 2)])
            >>> points.transform('translate(1)').xy
            [(2, 2)]
            >>> points.transform('translate(1 2)').xy
            [(2, 4)]
            >>> points.transform('translate(1,2)scale(2)').xy
            [(4, 8)]
            >>> points.transform('translate(1, 2)scale(1-1)').xy
            [(2, -4)]
            >>> points.transform('magic()')
            Traceback (most recent call last):
                ...
            ValueError: Unsupported (or invalid) transform function: magic
        """
        points = self
        for func, params in re.findall(r"([A-Za-z]+)\(([^\)]*)\)", transform):
            method = getattr(points, func, None)
            if not method:
                raise ValueError(f"Unsupported (or invalid) transform function: {func}")
            args = [_num(s) for s in COORD_REGEX.findall(params)]
            points = method(*args)
        return points

    @staticmethod
    def _points_to_xy(points: str) -> Coordinates:
        numbers = COORD_REGEX.findall(points)
        return [(_num(x), _num(y)) for x, y in _chunks(numbers, 2)]

    @staticmethod
    def _xy_to_points(xy: Coordinates) -> str:
        return " ".join([f"{x},{y}" for x, y in xy])

    @classmethod
    def _from_polyline(cls, points: str = "") -> "Points":
        xy = cls._points_to_xy(points)
        return cls(xy)

    def _to_polyline(self) -> Polyline:
        return {"points": self._xy_to_points(self.xy)}

    @classmethod
    def _from_polygon(cls, points: str = "") -> "Points":
        xy = cls._points_to_xy(points)
        pts = cls(xy)
        if not pts.closed():
            pts.xy.append(xy[0])
        return pts

    def _to_polygon(self) -> Polygon:
        xy = self.xy[:-1] if self.closed() else self.xy
        return {"points": self._xy_to_points(xy)}

    @classmethod
    def _from_line(
        cls, x1: Numeric = 0, y1: Numeric = 0, x2: Numeric = 0, y2: Numeric = 0
    ) -> "Points":
        xy = [(_num(x1), _num(y1)), (_num(x2), _num(y2))]
        return cls(xy)

    def _to_line(self) -> Line:
        # NOTE: Uses first and last points as line endpoints, or (0, 0) if empty.
        x1, y1 = [str(i) for i in (self.xy[0] if self.xy else (0, 0))]
        x2, y2 = [str(i) for i in (self.xy[-1] if self.xy else (0, 0))]
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @classmethod
    def _from_circle(cls, cx: Numeric = 0, cy: Numeric = 0) -> "Points":
        xy = [(_num(cx), _num(cy))]
        return cls(xy)

    def _to_circle(self) -> Circle:
        # NOTE: Uses the first point as the circle center, or (0, 0) if empty.
        cx, cy = [str(i) for i in (self.xy[0] if self.xy else (0, 0))]
        return {"cx": cx, "cy": cy}

    @classmethod
    def _from_rect(
        cls, width: Numeric, height: Numeric, x: Numeric = 0, y: Numeric = 0
    ) -> "Points":
        x, y, w, h = [_num(arg) for arg in (x, y, width, height)]
        xy = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        return cls(xy)

    def _to_rect(self) -> Rect:
        # NOTE: Returns the bounding box of all points.
        box = self.bbox()
        if not box:
            box = {"x": 0, "y": 0, "width": 0, "height": 0}
        return {
            "x": str(box["x"]),
            "y": str(box["y"]),
            "width": str(box["width"]),
            "height": str(box["height"]),
        }

    @classmethod
    def _from_svg(cls, viewBox: str = None) -> "Points":
        if viewBox:
            x, y, w, h = COORD_REGEX.findall(viewBox)
            return cls._from_rect(w, h, x, y)
        else:
            return cls([])

    def _to_svg(self) -> SVG:
        # NOTE: Returns the bounding box of all points.
        box = self.bbox()
        if box:
            return {"viewBox": f"{box['x']} {box['y']} {box['width']} {box['height']}"}
        return {}

    @classmethod
    def _from_image(
        cls, width: Numeric, height: Numeric, x: Numeric = 0, y: Numeric = 0
    ) -> "Points":
        return cls._from_rect(width, height, x, y)

    def _to_image(self) -> Rect:
        return self._to_rect()

    @classmethod
    def _from_path(cls, d: str = "") -> "Points":
        # NOTE: Only vertices are preserved. All curvature information is discarded.
        regex = {
            "cmd": re.compile(r"[a-df-zA-DF-Z]+"),
            "seq": re.compile(r"[^a-df-zA-DF-Z]+"),
            "coord": COORD_REGEX,
        }
        commands = regex["cmd"].findall(d)
        parameters = [
            [_num(coord) for coord in regex["coord"].findall(seq)]
            for seq in regex["seq"].findall(d)
        ]
        if commands and commands[-1] in ("Z", "z"):
            parameters.append([])
        xy = []
        for cmd, params in zip(commands, parameters):
            # moveTo: M (x,y)+ | lineTo: L (x,y)+ | curveTo: T (x,y)+
            if cmd in ("M", "L", "T"):
                for x, y in _chunks(params, 2):
                    xy.append((x, y))
            # moveTo: m (dx,dy)+ | lineTo: l (dx,dy)+ | curveTo: T (dx,dy)+
            elif cmd in ("m", "l", "t"):
                for dx, dy in _chunks(params, 2):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # lineTo: H (x)+
            elif cmd == "H":
                for (x,) in _chunks(params, 1):
                    xy.append((x, xy[-1][1]))
            # lineTo: h (dx)+
            elif cmd == "h":
                for (dx,) in _chunks(params, 1):
                    xy.append((xy[-1][0] + dx, xy[-1][1]))
            # lineTo: V (y)+
            elif cmd == "V":
                for (y,) in _chunks(params, 1):
                    xy.append((xy[-1][0], y))
            # lineTo: v (dy)+
            elif cmd == "v":
                for (dy,) in _chunks(params, 1):
                    xy.append((xy[-1][0], xy[-1][1] + dy))
            # curveTo: C (x1,y1 x2,y2 x,y)+
            elif cmd == "C":
                for _, _, _, _, x, y in _chunks(params, 6):
                    xy.append((x, y))
            # curveTo: c (dx1,dy1 dx2,dy2 dx,dy)+
            elif cmd == "c":
                for _, _, _, _, dx, dy in _chunks(params, 6):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # curveTo: S (x2,y2 x,y)+ | curveTo: Q (x1,y1 x,y)+
            elif cmd in ("S", "Q"):
                for _, _, x, y in _chunks(params, 4):
                    xy.append((x, y))
            # curveTo: s (dx2,dy2 dx,dy)+ | curveTo: q (dx1,dy1 dx,dy)+
            elif cmd in ("s", "q"):
                for _, _, dx, dy in _chunks(params, 4):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # arcTo: A (rx ry x-axis-rotation large-arc-flag sweep-flag x,y)+
            elif cmd == "A":
                for _, _, _, _, _, x, y in _chunks(params, 7):
                    xy.append((x, y))
            # arcTo: a (rx ry x-axis-rotation large-arc-flag sweep-flag dx,dy)+
            elif cmd == "a":
                for _, _, _, _, _, dx, dy in _chunks(params, 7):
                    xy.append((xy[-1][0] + dx, xy[-1][1] + dy))
            # closePath: Z | z
            elif cmd in ("Z", "z"):
                xy.append(xy[0])
            else:
                raise ValueError(f"Invalid command encountered: {cmd}")
        return cls(xy)

    def _to_path(self) -> Path:
        # NOTE: Uses only moveTo (`M`), lineTo (`L`), and closePath (`Z`) if closed.
        commands = []
        for i, xy in enumerate(self.xy[:-1] if self.closed() else self.xy):
            x, y = xy
            if i == 0:
                cmd = f"M {x},{y}"
            elif i == 1:
                cmd = f"L {x},{y}"
            else:
                cmd = f"{x},{y}"
            commands.append(cmd)
        if self.closed():
            commands.append("Z")
        return {"d": " ".join(commands)}

    @classmethod
    def from_element(cls, tag: str, **attrs: Numeric) -> "Points":
        """
        Extract coordinates from an element's tag and attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element.

        Arguments:
            tag: Element tag (e.g. 'path')
            **attrs: Element attributes

        Raises:
            ValueError: Unsupported or invalid element tag or attribute value.

        Example:
            >>> Points.from_element('path', d='M 0,0 l 1,0 0,1 z').xy
            [(0, 0), (1, 0), (1, 1), (0, 0)]
            >>> Points.from_element('polygon', points='0,0 1,0 1,1').xy
            [(0, 0), (1, 0), (1, 1), (0, 0)]
            >>> Points.from_element('polyline', points='0,0 1,0 1,1').xy
            [(0, 0), (1, 0), (1, 1)]
            >>> Points.from_element('line', x1='0', y1='1', x2='1', y2='2').xy
            [(0, 1), (1, 2)]
            >>> Points.from_element('circle', cx='0', cy='1').xy
            [(0, 1)]
            >>> Points.from_element('rect', x='0', y='1', width='1', height='2').xy
            [(0, 1), (1, 1), (1, 3), (0, 3), (0, 1)]
            >>> Points.from_element('image', x='0', y='1', width='1', height='2').xy
            [(0, 1), (1, 1), (1, 3), (0, 3), (0, 1)]
            >>> Points.from_element('svg', viewBox='0 1 1 2').xy
            [(0, 1), (1, 1), (1, 3), (0, 3), (0, 1)]
            >>> Points.from_element('magic')
            Traceback (most recent call last):
                ...
            ValueError: Unsupported (or invalid) element tag: magic
        """
        method = getattr(cls, "_from_" + tag, None)
        if not method:
            raise ValueError(f"Unsupported (or invalid) element tag: {tag}")
        args = inspect.getfullargspec(method).args[1:]
        attrs = {key: attrs[key] for key in attrs if key in args}
        return method(**attrs)

    def to_element(self, tag: str) -> Dict[str, str]:
        """
        Convert coordinates to element attributes.

        See https://developer.mozilla.org/en-US/docs/Web/SVG/Element.

        Arguments:
            tag: Element tag (e.g. 'path')

        Raises:
            ValueError: Unsupported or invalid element tag.

        Example:
            >>> points = Points([(0, 0), (0, 1), (1, 1), (0, 0)])
            >>> points.to_element('path')
            {'d': 'M 0,0 L 0,1 1,1 Z'}
            >>> points.to_element('polygon')
            {'points': '0,0 0,1 1,1'}
            >>> points.to_element('polyline')
            {'points': '0,0 0,1 1,1 0,0'}
            >>> # Returns a line from the first to the last point
            >>> points.to_element('line')
            {'x1': '0', 'y1': '0', 'x2': '0', 'y2': '0'}
            >>> # Returns a circle centered at the first point
            >>> points.to_element('circle')
            {'cx': '0', 'cy': '0'}
            >>> # Returns a rectangle bounding the points
            >>> points.to_element('rect')
            {'x': '0', 'y': '0', 'width': '1', 'height': '1'}
            >>> points.to_element('image')
            {'x': '0', 'y': '0', 'width': '1', 'height': '1'}
            >>> points.to_element('svg')
            {'viewBox': '0 0 1 1'}
            >>> points.to_element('magic')
            Traceback (most recent call last):
                ...
            ValueError: Unsupported (or invalid) element tag: magic
        """
        method = getattr(self, "_to_" + tag, None)
        if not method:
            raise ValueError(f"Unsupported (or invalid) element tag: {tag}")
        return method()
