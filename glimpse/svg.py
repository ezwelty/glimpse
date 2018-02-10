from .imports import (np, lxml, re)

# ---- Parse SVG file ----

def parse_svg(path, imgsz=None):
    """
    Return vertices of elements in SVG file.

    All path, line, polyline, and polygon elements are returned as Nx2 arrays of vertices.
    All <g> XML elements with 'id' attributes and their descendants (if these include any of the above)
    are retained to form the structure of the result.
    Dictionary keys are either the element 'id' or tag with an optional numeric counter (e.g., 'line-1').

    Arguments:
        path (str): Path to SVG file
        imgsz (array_like): Image width and height [nx, ny] to which to scale SVG coordinates.
            If `None` (default), image size is determined from <image> 'width' and 'height' attributes.

    Returns:
        dict: Element vertices (Nx2 arrays)
    """
    tree = lxml.etree.parse(path)
    # Remove namespaces from tags and attributes
    regex = re.compile("\{.*\}")
    for node in tree.iter():
        node.tag = regex.sub("", node.tag)
        for key in node.attrib.keys():
            new_key = regex.sub("", key)
            new_value = regex.sub("", node.attrib.pop(key))
            node.attrib[new_key] = new_value
    # SVG size
    svgs = tree.xpath("//svg")
    if len(svgs) > 1:
        warnings.warn("Using first (of multiple) <svg> for scale calculation.")
    svg_width = svgs[0].get('width')
    svg_height = svgs[0].get('height')
    regex = re.compile("[0-9\.]+")
    if svg_width and svg_height:
        svg_width = regex.findall(svg_width)
        svg_height = regex.findall(svg_height)
        svgsz = np.array(svg_width + svg_height, dtype=float)
    else:
        svgsz = None
    # Image size
    images = tree.xpath("//image")
    if imgsz is None and svgsz is not None:
        if images:
            # Get image size from SVG
            if len(images) > 1:
                warnings.warn("Using first (of multiple) <image> for scale calculation.")
            # TODO: Support transform attribute
            img_width = images[0].get('width')
            img_height = images[0].get('height')
            if img_width and img_height:
                regex = re.compile("([0-9\.]+)")
                img_width = regex.findall(img_width)
                img_height = regex.findall(img_height)
                imgsz = np.array(img_width + img_height, dtype=float)
    # Scale factor
    if svgsz is not None and imgsz is not None:
        # Compute scale
        scale = imgsz / svgsz
    else:
        # Use default scale
        scale = np.array([1, 1], dtype=float)
    # Parse all data or g[@id] node not ancestors of g[@id] and who are (or are ancestors of) data
    tags = ["path", "line", "polyline", "polygon", "circle"]
    selfs = " | ".join(["self::" + tag for tag in tags])
    descendants = " | ".join(["descendant-or-self::" + tag for tag in tags])
    nodes = tree.xpath("//*[not(ancestor::g[@id]) and (self::g[@id] or (" + selfs + ")) and (" + descendants + ")]")
    return _parse_nodes(nodes, scale=scale)

def _parse_nodes(nodes, scale=None):
    """
    Return vertices of SVG elements in ElementTree nodes.

    All path, line, polyline, and polygon elements are returned as Nx2 arrays of vertices.
    Dictionary keys are either the element 'id' or tag with an optional numeric counter (e.g., 'line-1').

    Arguments:
        nodes (list): ElementTree elements
        scale (array_like): Coordinate scale factor [x, y]

    Returns:
        dict: SVG element vertices (Nx2 arrays)
    """
    branch = {}
    tags = [node.tag for node in nodes]
    for i in range(len(nodes)):
        tag = tags[i]
        # Use @id, tag-<counter>, or tag
        if nodes[i].attrib.has_key('id'):
            id = nodes[i].attrib['id']
        elif tags.count(tag) > 1:
            id = tag + "-" + str(i)
        else:
            id = tag
        if len(nodes[i]):
            # Iterage on parent node
            branch[id] = _parse_nodes(nodes[i], scale=scale)
        else:
            # Parse element (if supported)
            if tag == 'path':
                branch[id] = parse_path(nodes[i].attrib['d'])
            elif tag == 'polygon':
                branch[id] = parse_polygon(nodes[i].attrib['points'], closed=True)
            elif tag == 'polyline':
                branch[id] = parse_polyline(nodes[i].attrib['points'])
            elif tag == 'line':
                branch[id] = parse_line(**dict((k, nodes[i].attrib[k]) for k in ('x1', 'y1', 'x2', 'y2')))
            elif tag == 'circle':
                branch[id] = parse_circle(**dict((k, nodes[i].attrib[k]) for k in ('cx', 'cy')))
            if scale is not None and isinstance(branch[id], np.ndarray):
                branch[id] *= scale
    return branch

# ---- Helpers ----

def parse_polyline(points):
    """
    Return polyline vertices.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/polyline

    Arguments:
        points (str): Polyline 'points' attribute

    Returns:
        array: Coordinates x,y (Nx2)
    """
    num_str = re.findall("[0-9\.\-]+", points)
    return np.array(num_str, dtype=float).reshape((-1, 2))

def parse_polygon(points, closed=True):
    """
    Return polygon vertices.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/polygon

    Arguments:
        points (str): Polygon 'points' attribute
        closed (bool): Whether to return a closed polygon.
            If `True` (default), the first vertices is also appended as last.

    Returns:
        array: Coordinates x,y (Nx2)
    """
    num_str = re.findall("[0-9\.\-]+", points)
    if closed:
        num_str += num_str[0:2]
    return np.array(num_str, dtype=float).reshape((-1, 2))

def parse_line(x1, y1, x2, y2):
    """
    Return line vertices.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/line

    Arguments:
        x1 (str): Line 'x1' attribute
        y1 (str): Line 'y1' attribute
        x2 (str): Line 'x2' attribute
        y2 (str): Line 'y2' attribute

    Returns:
        array: Coordinates x,y (2x2)
    """
    return np.array([x1, y1, x2, y2], dtype=float).reshape((2, 2))

def parse_path(d):
    """
    Return path vertices.

    See https://www.w3.org/TR/SVG/paths.html#DAttribute

    Arguments:
        d (str): Path 'd' attribute

    Returns:
        array: Coordinates x,y (Nx2)
    """
    # Parse letter and number lists
    letters = re.findall("[a-zA-Z]+", d)
    numbers = re.findall("[\.,0-9\- ]+", d)
    n = len(letters)
    X = []
    # Compute coordinates of vertices
    for i in range(n):
        tag = letters[i]
        # closepath: Z | z
        if tag in ('Z', 'z'):
            X.insert(i, X[0][0])
            continue
        else:
            data = np.array(re.findall("\-*[\.0-9]+", numbers[i]), dtype=float)
        if len(data) % 2 == 0:
            data = data.reshape((-1, 2))
        # moveto: M (x,y)+ | m (dx,dy)+
        if tag == 'M':
            # Path always begins with 'M'
            X.insert(i, data)
        elif tag == 'm':
            X.insert(i, data + X[-1][-1])
        # lineto: L (x,y)+ | l (dx,dy)+
        elif tag == 'L':
            X.insert(i, data)
        elif tag == 'l':
            X.insert(i, data + X[-1][-1])
        # lineto (horizontal): H (x)+ | h (dx)+
        elif tag == 'H':
            temp = np.column_stack((
                data,
                np.repeat(X[-1][-1][1], len(data))
            ))
            X.insert(i, temp)
        elif tag == 'h':
            temp = np.column_stack((
                data + X[-1][-1][0],
                np.repeat(X[-1][-1][1], len(data))
            ))
            X.insert(i, temp)
        # lineto (vertical): V (y)+ | v (dy)+
        elif tag == 'V':
            temp = np.column_stack((
                np.repeat(X[-1][-1][0], len(data)),
                data
            ))
            X.insert(i, temp)
        elif tag == 'v':
            temp = np.column_stack((
                np.repeat(X[-1][-1][0], len(data)),
                data + X[-1][-1][1]
            ))
            X.insert(i, temp)
        # curveto (Bezier): C (c1x,c1y c2x,c2y x,y)+ | c (relative)
        elif tag == 'C':
            X.insert(i, data[2::3])
        elif tag == 'c':
            X.insert(i, data[2::3] + X[-1][-1])
        # curveto (Smooth Bezier): S (x2 y2 x y)+ | s (relative)
        elif tag == 'S':
            X.insert(i, data[1::2])
        elif tag == 's':
            X.insert(i, data[1::2] + X[-1][-1])
        # curveto (Quadratic Bezier): Q (x1 y1 x y)+ | q (relative)
        elif tag == 'Q':
            X.insert(i, data[1::2])
        elif tag == 'q':
            X.insert(i, data[1::2] + X[-1][-1])
        # curveto (Smooth Quadratic Bezier): T (x y)+ | t (relative)
        elif tag == 'T':
            X.insert(i, data)
        elif tag == 't':
            X.insert(i, data + X[-1][-1])
        # arcto: A (rx ry x-axis-rotation large-arc-flag sweep-flag x y)+ | a (relative)
        elif tag == 'A':
            X.insert(i, data.reshape((-1, 7))[:, 5:7])
        elif tag == 'a':
            X.insert(i, data.reshape((-1, 7))[:, 5:7] + X[-1][-1])
        else:
            print "Unsupported tag encountered: " + tag
    return np.vstack(X)

def parse_circle(cx, cy):
    """
    Return circle center coordinate.

    See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/circle

    Arguments:
        cx (str): Circle 'cx' attribute
        cy (str): Circle 'cy' attribute

    Returns:
        array: Coordinates x,y (1x2)
    """
    return np.array((cx, cy), dtype=float).reshape((1, 2))
