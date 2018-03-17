from .imports import (
    np, cPickle, pyproj, json, collections, copy, pandas, scipy, gzip, PIL,
        sklearn, cv2, copy_reg, os, re)

# ---- General ---- #

def merge_dicts(*args):
    """
    Merge dictionaries.

    Precedence goes to the latter value for each key.

    Arguments:
        *args: Dictionaries
    """
    merge = dict()
    for d in args:
        merge.update(d)
    return merge

def format_list(obj, length=1, default=None, dtype=float, ltype=np.array):
    """
    Format a list-like object.

    Arguments:
        obj (object): Object
        length (int): Output object length
        default (scalar): Default element value.
            If `None`, `obj` is repeated to achieve length `length`.
        dtype (type): Data type to coerce list elements to.
            If `None`, data is left as-is.
        ltype (type): List type to coerce object to.
    """
    if obj is None:
        return obj
    try:
        obj = list(obj)
    except TypeError:
        obj = [obj]
    if len(obj) < length:
        if default is not None:
            # Fill remaining slots with 0
            obj.extend([default] * (length - len(obj)))
        else:
            # Repeat list
            if len(obj) > 0:
                assert length % len(obj) == 0
                obj *= length / len(obj)
    if dtype:
        obj = [dtype(i) for i in obj[0:length]]
    if ltype:
        obj = ltype(obj)
    return obj

def make_path_directories(path, is_file=True):
    # https://stackoverflow.com/a/14364249
    if is_file:
        path = os.path.dirname(path)
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

# ---- Pickles ---- #

def write_pickle(obj, path, gz=False, binary=True, protocol=cPickle.HIGHEST_PROTOCOL):
    """
    Write object to pickle file.
    """
    make_path_directories(path, is_file=True)
    mode = 'wb' if binary else 'w'
    if gz:
        fp = gzip.open(path, mode=mode)
    else:
        fp = open(path, mode=mode)
    cPickle.dump(obj, file=fp, protocol=protocol)
    fp.close()

def read_pickle(path, gz=False, binary=True):
    """
    Read object from pickle file.
    """
    mode = 'rb' if binary else 'r'
    if gz:
        fp = gzip.open(path, mode=mode)
    else:
        fp = open(path, mode=mode)
    obj = cPickle.load(fp)
    fp.close()
    return obj

# Register Pickle method for cv2.KeyPoint
# https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
def _pickle_cv2_keypoints(k):
    return cv2.KeyPoint, (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
copy_reg.pickle(cv2.KeyPoint().__class__, _pickle_cv2_keypoints)

# ---- Arrays: General ---- #

def normalize(array):
    """
    Normalize a numeric array to mean 0, variance 1.

    Arguments:
        array (array): Input array
    """
    return (array - array.mean()) * (1 / array.std())

def normalize_range(array, interval=None):
    """
    Translate and scale a numeric array to the interval (0, 1).

    Arguments:
        array (array): Input array
        interval: Measurement interval (min, max) as either an iterable or np.dtype.
            If `None`, the min and max of `array` are used.

    Returns:
        array (optional): Normalized copy of array, cast to float
    """
    if isinstance(interval, np.dtype):
        dtype = interval.type
        if issubclass(dtype, np.integer):
            info = np.iinfo(dtype)
        elif issubclass(dtype, np.floating):
            info = np.finfo(dtype)
        interval = (info.min, info.max)
    else:
        if interval is None:
            interval = array
        interval = (min(interval), max(interval))
    return (array + (-interval[0])) * (1.0 / (interval[1] - interval[0]))

# ---- Arrays: Images ---- #

def linear_to_gamma(array, gamma=2.2):
    """
    Converts linear values to gamma-corrected values.

    f(x) = x ^ (1 / gamma)
    Assumes the array values are normalized to interval (0, 1).

    Arguments:
        array (array): Input array
        gamma (float): Gamma coefficient
    """
    return array**gamma

def gamma_to_linear(array, gamma=2.2):
    """
    Converts gamma-corrected values to linear values.

    f(x) = x ^ (1 / gamma)
    Assumes the array values are normalized to interval (0, 1).

    Arguments:
        array (array): Input array
        gamma (float): Gamma coefficient
    """
    return array**(1 / gamma)

GRAY_PCA = sklearn.decomposition.PCA(n_components=1, svd_solver='arpack', whiten=False)

def rgb_to_gray(rgb, method='average', weights=None, pca=None):
    """
    Convert a color image to a grayscale image.

    Arguments:
        rgb (array): 3-d color image
        method (str): Either 'average' for a weighted average of each channel,
            or 'pca' for a principal components transform.
        weights (array-like): Weights for each channel of `rgb`.
            If `None`, the channels are assigned equal weight.
        pca (sklearn.decomposition.pca.PCA): PCA object.
            If `None`, `glimpse.helpers.GRAY_PCA` is used
            (n_components=1, svd_solver='arpack', whiten=True).

    Returns:
        array: 2-d grayscale image
    """
    if method is 'average':
        return np.average(rgb, axis=2, weights=weights)
    else:
        if pca is None:
            pca = GRAY_PCA
        Q = rgb.reshape(-1, rgb.shape[2])
        pca.fit(Q)
        pca.components_ = np.sign(pca.components_[0]) * pca.components_
        return pca.transform(Q).reshape(rgb.shape[0:2])

def compute_cdf(array, return_inverse=False):
    """
    Compute the cumulative distribution function of an array.

    Arguments:
        array (array): Input array
        return_inverse (bool): Whether to return the indices of the returned
            `values` that reconstruct `array`

    Returns:
        array: Sorted unique values
        array: Quantile of each value in `values`
        array (optional): Indices of `values` which reconstruct `array`.
            Only returned if `return_inverse=True`.
    """
    results = np.unique(array, return_inverse=return_inverse, return_counts=True)
    # Normalize cumulative sum of counts by the number of pixels
    quantiles = np.cumsum(results[-1]) * (1.0 / array.size)
    if return_inverse:
        return results[0], quantiles, results[1]
    else:
        return results[0], quantiles

def match_histogram(source, template): # hist_match
    """
    Adjust the values of an array such that its histogram matches that of a target array.

    Arguments:
        source (array): Array to transform.
        template: Histogram template as either an array (of any shape)
            or an iterable (unique values, unique value quantiles).

    Returns:
        array: Transformed `source` array
    """
    s_values, s_quantiles, inverse_index = compute_cdf(source, return_inverse=True)
    if isinstance(template, np.ndarray):
        template = compute_cdf(template, return_inverse=False)
    # Interpolate new values based on source and template quantiles
    new_values = np.interp(s_quantiles, template[1], template[0])
    return new_values[inverse_index].reshape(source.shape)

# ---- GIS ---- #

# FIXME: Unused?
def dms_to_degrees(degrees, minutes, seconds):
    """
    Convert degree-minute-second to decimal degrees.
    """
    return degrees + minutes / 60.0 + seconds / 3600.0

def sp_transform(points, current, target):
    """
    Transform points between spatial coordinate systems.

    Coordinate systems can be specified either as an
    `int` (EPSG code), `dict` (arguments to `pyproj.Proj()`), or `pyproj.Proj`.

    Arguments:
        points (array): Point coordinates [[x, y(, z)]]
        current: Current coordinate system
        target: Target coordinate system
    """
    def build_proj(obj):
        if isinstance(obj, pyproj.Proj):
            return obj
        elif isinstance(obj, int):
            return pyproj.Proj(init="epsg:" + str(obj))
        elif isinstance(obj, dict):
            return pyproj.Proj(**obj)
        else:
            raise ValueError("Cannot coerce input to pyproj.Proj")
    current = build_proj(current)
    target = build_proj(target)
    if points.shape[1] < 3:
        z = None
    else:
        z = points[:, 2]
    result = pyproj.transform(current, target, x=points[:, 0], y=points[:, 1], z=z)
    return np.column_stack(result)

def read_json(path, **kwargs):
    """
    Read JSON from file.

    Arguments:
        path (str): Path to file
        **kwargs: Additional arguments passed to `json.load()`
    """
    with open(path, "r") as fp:
        return json.load(fp, **kwargs)

def write_json(obj, path=None, flat_arrays=False, **kwargs):
    """
    Write object to JSON.

    Arguments:
        obj: Object to write as JSON
        path (str): Path to file. If `None`, result is returned as a string.
        flat_arrays (bool): Whether to flatten json arrays to a single line.
            By default, `json.dumps` puts each array element on a new line if
            `indent` is `0` or greater.
        **kwargs: Additional arguments passed to `json.dumps()`
    """
    txt = json.dumps(obj, **kwargs)
    if flat_arrays and kwargs.get('indent') >= 0:
        separators = kwargs.get('separators')
        sep = separators[0] if separators else ', '
        squished_sep = re.sub(r'\s', '', sep)
        def flatten(match):
            return re.sub(squished_sep, sep, re.sub(r'\s', '', match.group(0)))
        txt = re.sub(r'(\[\s*)+[^\]\{]*(\s*\])+', flatten, txt)
    if path:
        make_path_directories(path, is_file=True)
        with open(path, "w") as fp:
            fp.write(txt)
        return None
    else:
        return txt

def geojson_iterfeatures(obj):
    features = obj['features']
    if isinstance(features, list):
        index = range(len(features))
    else:
        index = features.keys()
    for i in index:
        yield features[i]

def _get_geojson_coords(feature):
    if feature.has_key('geometry'):
        return feature['geometry']['coordinates']
    else:
        return feature['coordinates']

def _set_geojson_coords(feature, coords):
    if feature.has_key('geometry'):
        feature['geometry']['coordinates'] = coords
    else:
        feature['coordinates'] = coords

def geojson_itercoords(obj):
    for feature in geojson_iterfeatures(obj):
        yield _get_geojson_coords(feature)

def apply_geojson_coords(obj, fun, **kwargs):
    for feature in geojson_iterfeatures(obj):
        coords = _get_geojson_coords(feature)
        _set_geojson_coords(feature, fun(coords, **kwargs))

def read_geojson(path, key=None, crs=None, **kwargs):
    """
    Read GeoJSON from file.

    Arguments:
        path (str): Path to file
        key (str): Feature property to set as feature key
        crs: Target coordinate system for transformation.
            Assumes GeoJSON coordinates are WGS 84
            [Longitude (degrees), Latitude (degrees), Height above ellipsoid (meters)].
            If `None`, coordinates are unchanged.
        kwargs (dict): Additional arguments passed to `read_json()`
    """
    obj = read_json(path, **kwargs)
    apply_geojson_coords(obj, np.atleast_2d)
    if crs:
        apply_geojson_coords(obj, sp_transform, current=4326, target=crs)
    if key:
        obj['features'] = dict((feature['properties'][key], feature) for feature in obj['features'])
    return obj

def write_geojson(obj, path=None, crs=None, decimals=None, **kwargs):
    """
    Write object to GeoJSON.

    Arguments:
        obj (dict): Object to write as GeoJSON
        path (str): Path to file. If `None`, result is returned as a string.
        crs: Current coordinate system for transformation to WGS 84
            [Longitude (degrees), Latitude (degrees), Height above ellipsoid (meters)].
            If `None`, coordinates are unchanged.
        kwargs (dict): Additional arguments passed to `write_json()`
    """
    def round_coords(coords, decimals):
        for i, decimal in enumerate(decimals):
            if i < coords.shape[1]:
                coords[:, i] = np.round(coords[:, i], decimal)
        return coords
    obj = copy.deepcopy(obj)
    if isinstance(obj['features'], dict):
        # Revert named features back to list
        obj['features'] = obj['features'].values()
    apply_geojson_coords(obj, np.atleast_2d)
    if crs:
        apply_geojson_coords(obj, sp_transform, current=crs, target=4326)
    if decimals:
        apply_geojson_coords(obj, round_coords, decimals=decimals)
    apply_geojson_coords(obj, np.squeeze)
    apply_geojson_coords(obj, np.ndarray.tolist)
    return write_json(obj, path=path, **kwargs)

def elevate_geojson(obj, elevation):
    """
    Add or update GeoJSON elevations.

    Arguments:
        obj (dict): Object to modify
        elevation: Elevation, as either
            the elevation of all features (int or float),
            the name of the feature property containing elevation (str), or
            digital elevation model from which to sample elevation (`dem.DEM`).
    """
    def set_z(coords, z):
        if len(z) == 1:
            z = np.repeat(z, len(coords))
        return np.column_stack((coords[:, 0:2], z))
    def set_from_elevation(coords, elevation):
        if isinstance(elevation, (int, float)):
            return set_z(coords, elevation)
        elif isinstance(elevation, dem.DEM):
            return set_z(coords, elevation.sample(coords[:, 0:2]))
    if isinstance(elevation, str):
        for feature in geojson_iterfeatures(obj):
            coords = _get_geojson_coords(feature)
            _set_geojson_coords(feature, set_z(coords, feature['properties'][elevation]))
    else:
        apply_geojson_coords(obj, set_from_elevation, elevation=elevation)

def ordered_geojson(obj, properties=None,
    keys=['type', 'properties', 'features', 'geometry', 'coordinates']):
    """
    Return ordered GeoJSON.

    Arguments:
        obj (dict): Object to order
        properties (list): Order of properties (any unnamed are returned last)
        keys (list): Order of keys (any unnamed are returned last)
    """
    def order_item(d, name=None):
        if isinstance(d, list):
            index = range(len(d))
        elif isinstance(d, dict):
            index = d.keys()
        else:
            return d
        for i in index:
            d[i] = order_item(d[i], name=i)
        if isinstance(d, dict):
            if properties and name == 'properties':
                ordered_keys = ([key for key in properties if key in index]
                    + [key for key in index if key not in properties])
                return collections.OrderedDict((k, d[k]) for k in ordered_keys)
            elif keys and name != 'properties':
                ordered_keys = ([key for key in keys if key in index]
                    + [key for key in index if key not in keys])
                return collections.OrderedDict((k, d[k]) for k in ordered_keys)
            else:
                return d
        else:
            return d
    obj = copy.deepcopy(obj)
    return order_item(obj)

# ---- Geometry ---- #

def boolean_split(x, mask, axis=0, circular=False, include='all'):
    """
    Split array by boolean mask.

    Select True groups with [0::2] if mask[0] is True,
    else [1::2].
    """
    # See https://stackoverflow.com/a/36518315/8161503
    cuts = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    splits = np.split(x, cuts, axis=axis)
    if circular and len(splits) > 1 and mask[0] is mask[-1]:
        splits[0] = np.concatenate((splits[-1], splits[0]), axis=axis)
        splits.pop(-1)
    if include == 'all':
        return splits
    elif include == 'true':
        index = slice(0, None, 2) if mask[0] else slice(1, None, 2)
        return splits[index]
    elif include == 'false':
        index = slice(1, None, 2) if mask[0] else slice(0, None, 2)
        return splits[index]
    else:
        return list()

def intersect_rays_plane(origin, directions, plane):
    """
    Return intersections of rays with a plane.

    Optimized for rays with a common origin.

    Arguments:
        origin (array-like): Common origin of rays [x, y , z]
        directions (array-like): Directions of rays [[dx, dy, dz], ...]
        plane (array-like): Plane [a, b, c, d], where ax + by + cz + d = 0

    Returns:
        array: Intersection coordinates [[xi, yi, zi], ...] (`nan` if none)
    """
    # plane: ax + by + cz + d = 0 | normal n = [a, b, c]
    # ray: xyz + t * dxyz
    # intersect at t = - (xyz . n + d) / (dxyz . n), if t >= 0
    n = plane[0:3]
    d = plane[3]
    t = - (np.dot(n, origin) + d) / np.dot(n, directions.T)
    t[t < 0] = np.nan
    return origin + t[:, None] * directions

def in_box(points, box):
    """
    Tests whether each point is in (or on) the box.

    Works in any dimension.

    Arguments:
        points (array): Point coordinates (npts, ndim)
        box (array-like): Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, )
    """
    box = np.reshape(box, (2, -1))
    return np.all((points >= box[0, :]) & (points <= box[1, :]), axis=1)

def clip_polyline_box(line, box, t=False):
    """
    Returns segments of line within the box.

    Vertices are inserted as needed on the box boundary.
    For speed, does not check for segments within the box
    entirely between two adjacent line vertices.

    Arguments:
        line (array): 2 or 3D point coordinates (npts, ndim)
        box (array): Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, )
        t (bool): Last column of `line` are distances along line, linearly interpolated at splits
    """
    if t:
        cols = slice(None, -1)
    else:
        cols = slice(None)
    mask = in_box(line[:, cols], box)
    segments = boolean_split(line, mask)
    trues = slice(int(~mask[0]), None, 2)
    nsegments = len(segments)
    for i in range(*trues.indices(nsegments)):
        if i > 0:
            origin = segments[i - 1][-1, :]
            distance = segments[i][0, :] - origin
            ti = intersect_edge_box(origin[cols], distance[cols], box)
            if ti is not None:
                segments[i] = np.insert(segments[i], 0, origin + ti * distance, axis=0)
        if i < nsegments - 1:
            origin = segments[i][-1, :]
            distance = segments[i + 1][0, :] - origin
            ti = intersect_edge_box(origin[cols], distance[cols], box)
            if ti is not None:
                segments[i] = np.insert(segments[i], len(segments[i]), origin + ti * distance, axis=0)
    return segments[trues]

def intersect_edge_box(origin, distance, box):
    """
    Returns intersection of edge with box.

    Arguments:
        origin (array): Coordinates of 2 or 3D point (ndim, )
        distance (array): Distance to end point (ndim, )
        box (array): Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, )
    """
    t = np.nanmin(intersect_rays_box(origin, distance[None, :], box, t=True))
    if t > 0 and t < 1:
        return t
    else:
        return None

def intersect_rays_box(origin, directions, box, t=False):
    """
    Return intersections of rays with a(n axis-aligned) box.

    Works in both 2 and 3 dimensions.
    Vectorized version of algorithm by Williams et al. (2011) optimized for rays with a common origin.
    Also inspired by https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

    Arguments:
        origin (array-like): Common origin of rays [x, y(, z)]
        directions (array-like): Directions of rays [[dx, dy(, dz)], ...]
        box (array-like): Box min and max vertices [xmin, ymin(, zmin), xmax, ymax(, zmax)]

    Returns:
        array: Entrance coordinates (`nan` if a miss or `origin` inside `box`)
        array: Exit coordinates (`nan` if a miss)
    """
    # Precompute constants
    bounds = np.repeat(np.atleast_2d(box), len(directions), axis=0)
    fbounds = bounds.flatten()
    invdir = np.divide(1.0, directions)
    sign = (invdir < 0).astype(int)
    nrays = directions.shape[0]
    ndims = directions.shape[1]
    all_rays = np.arange(nrays)
    # Initialize intersections on x-axis
    idx = np.ravel_multi_index((all_rays, sign[:, 0] * ndims), bounds.shape)
    tmin = (fbounds[idx] - origin[0]) * invdir[:, 0]
    idx = np.ravel_multi_index((all_rays, (1 - sign[:, 0]) * ndims), bounds.shape)
    tmax = (fbounds[idx] - origin[0]) * invdir[:, 0]
    # Apply y-axis intersections
    idx = np.ravel_multi_index((all_rays, 1 + sign[:, 1] * ndims), bounds.shape)
    tymin = (fbounds[idx] - origin[1]) * invdir[:, 1]
    idx = np.ravel_multi_index((all_rays, 1 + (1 - sign[:, 1]) * ndims), bounds.shape)
    tymax = (fbounds[idx] - origin[1]) * invdir[:, 1]
    misses = (tmin > tymax) | (tymin > tmax)
    tmin[misses] = np.nan
    tmax[misses] = np.nan
    ymin_intersects = tymin > tmin
    tmin[ymin_intersects] = tymin[ymin_intersects]
    ymax_intersects = tymax < tmax
    tmax[ymax_intersects] = tymax[ymax_intersects]
    if ndims > 2:
        # Apply z-axis intersections
        idx = np.ravel_multi_index((all_rays, 2 + sign[:, 2] * ndims), bounds.shape)
        tzmin = (fbounds[idx] - origin[2]) * invdir[:, 2]
        idx = np.ravel_multi_index((all_rays, 2 + (1 - sign[:, 2]) * ndims), bounds.shape)
        tzmax = (fbounds[idx] - origin[2]) * invdir[:, 2]
        misses = (tmin > tzmax) | (tzmin > tmax)
        tmin[misses] = np.nan
        tmax[misses] = np.nan
        zmin_intersects = tzmin > tmin
        tmin[zmin_intersects] = tzmin[zmin_intersects]
        zmax_intersects = tzmax < tmax
        tmax[zmax_intersects] = tzmax[zmax_intersects]
    # Discard intersections behind ray (t < 0)
    tmin[tmin < 0] = np.nan
    tmax[tmax < 0] = np.nan
    if t:
        return tmin[:, None], tmax[:, None]
    else:
        return origin + tmin[:, None] * directions, origin + tmax[:, None] * directions

def bresenham_line(start, end):
    """
    Return grid indices along a line between two grid indices.

    Uses Bresenham's run-length algorithm.
    Not all intersected grid cells are returned, only those with centers closest to the line.
    Code modified for speed from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm.

    TODO: Replace with faster run-slice (http://www.phatcode.net/res/224/files/html/ch36/36-03.html)

    Arguments:
        start (array-like): Start position [xi, yi]
        end (array-like): End position [xi, yi]

    Returns:
        array: Grid indices [[xi, yi], ...]
    """
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]
    # Determine how steep the line is
    is_steep = abs(y2 - y1) > abs(x2 - x1)
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # Swap start and end points if necessary and store swap state
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    else:
        swapped = False
    # Calculate new differentials
    dx = x2 - x1
    abs_dy = abs(y2 - y1)
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    if is_steep:
        for x in xrange(x1, x2 + 1):
            points.append((y, x))
            error -= abs_dy
            if error < 0:
                y += ystep
                error += dx
    else:
        for x in xrange(x1, x2 + 1):
            points.append((x, y))
            error -= abs_dy
            if error < 0:
                y += ystep
                error += dx
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return np.array(points)

# FIXME: Unused?
def bresenham_circle(center, radius):
    x0 = center[0]
    y0 = center[1]
    # Compute number of points
    octant_size = int(np.floor((np.sqrt(2) * (radius - 1) + 4) / 2))
    n_points = 8 * octant_size
    x = 0
    y = radius
    f = 1 - radius
    dx = 1
    dy = - 2 * radius
    xy = np.full((n_points, 2), np.nan)
    # 1st octant
    xy[0, :] = [x0 + x, y0 + y]
    # 2nd octant
    xy[8 * octant_size - 1, :] = [x0 - x, y0 + y]
    # 3rd octant
    xy[4 * octant_size - 1, :] = [x0 + x, y0 - y]
    # 4th octant
    xy[4 * octant_size, :] = [x0 - x, y0 - y]
    # 5th octant
    xy[2 * octant_size - 1, :] = [x0 + y, y0 + x]
    # 6th octant
    xy[6 * octant_size, :] = [x0 - y, y0 + x]
    # 7th octant
    xy[2 * octant_size, :] = [x0 + y, y0 - x]
    # 8th octant
    xy[6 * octant_size - 1, :] = [x0 - y, y0 - x]
    for i in range(2, octant_size + 1):
        if f > 0:
            y -= 1
            dy += 2
            f += dy
        x += 1
        dx += 2
        f += dx
        # 1st octant
        xy[i - 1, :] = [x0 + x, y0 + y]
        # 2nd octant
        xy[8 * octant_size - i, :] = [x0 - x, y0 + y]
        # 3rd octant
        xy[4 * octant_size - i, :] = [x0 + x, y0 - y]
        # 4th octant
        xy[4 * octant_size + i - 1, :] = [x0 - x, y0 - y]
        # 5th octant
        xy[2 * octant_size - i, :] = [x0 + y, y0 + x]
        # 6th octant
        xy[6 * octant_size + i - 1, :] = [x0 - y, y0 + x]
        # 7th octant
        xy[2 * octant_size + i - 1, :] = [x0 + y, y0 - x]
        # 8th octant
        xy[6 * octant_size - i, :] = [x0 - y, y0 - x]
    return xy

# FIXME: Unused?
def inverse_kernel_distance(data, bandwidth=None, function='gaussian'):
    # http://pysal.readthedocs.io/en/latest/library/weights/Distance.html#pysal.weights.Distance.Kernel
    nd = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))
    if bandwidth is None:
        bandwidth = np.max(nd)
    nd /= bandwidth
    included = nd <= 1
    if function == 'triangular':
        temp = np.where(included, 1 - nd, 0)
    elif function == 'uniform':
        temp = np.where(included, 0.5, 0)
    elif function == 'quadratic':
        temp = np.where(included, (3./4) * (1 - nd**2), 0)
    elif function == 'quartic':
        temp = np.where(included, (15./16) * (1 - nd**2)**2, 0)
    elif function == 'gaussian':
        temp = np.where(included, (2 * np.pi)**(-0.5) * np.exp((-nd**2) / 2), 0)
    # Compute weights as inverse sum of kernel distances
    return 1 / np.sum(temp, axis=1)

# TODO: Deprecate?
def nearest_neighbours(x, lst):
    if x <= lst[0]:
        return 0
    elif x >= lst[-1]:
        return len(lst)-1
    else:
        for i, y in enumerate(lst[:-1]):
            if y <= x <= lst[i+1]:
                return i, i+1

def intersect_ranges(ranges):
    # ranges: ((min, max), ...) or 2-d array
    if not isinstance(ranges, np.ndarray):
        ranges = np.array(ranges)
    ranges.sort(axis=1)
    rmin = np.nanmax(ranges[:, 0])
    rmax = np.nanmin(ranges[:, 1])
    if rmax - rmin <= 0:
        raise ValueError("Ranges do not intersect")
    else:
        return np.hstack((rmin, rmax))

def intersect_boxes(boxes):
    # boxes: ((minx, ..., maxx, ...), ...) or 2-d array
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    ndim = boxes.shape[1] / 2
    boxmin = np.nanmax(boxes[:, 0:ndim], axis=0)
    boxmax = np.nanmin(boxes[:, ndim:], axis=0)
    if any(boxmax - boxmin <= 0):
        raise ValueError("Boxes do not intersect")
    else:
        return np.hstack((boxmin, boxmax))

# ---- Image formation ---- #

def rasterize_points(rows, cols, values, shape, fun=np.mean):
    """
    Rasterize points by array indices.

    Points are aggregated by equal row and column indices and the specified function,
    then inserted into an empty array.

    Arguments:
        rows (array): Point row indices
        cols (array): Point column indices
        values (array): Point value
        shape (tuple): Output array row and column size
        fun (function): Aggregate function to apply to values of overlapping points

    Returns:
        array: Float array of shape `shape` with aggregated point values
            where available and `NaN` elsewhere
    """
    df = pandas.DataFrame(dict(row=rows, col=cols, value=values))
    groups = df.groupby(('row', 'col')).aggregate(fun).reset_index()
    idx = np.ravel_multi_index((groups.row.as_matrix(), groups.col.as_matrix()), shape)
    grid = np.full(shape, np.nan)
    grid.flat[idx] = groups.value.as_matrix()
    return grid

def compute_mask_array_from_svg(path_to_svg, array_shape, skip=[]):
    from svgpathtools import svg2paths
    from matplotlib import path
    paths, attributes = svg2paths(path_to_svg)
    q = [np.array([(l.point(0).real, l.point(0).imag) for l in p] + [(p[0].point(0).real, p[0].point(0).imag)]) for p in paths]
    paths = [path.Path(qq) for qq in q]
    mask = np.zeros(array_shape).astype(bool)
    cols, rows = np.meshgrid(range(mask.shape[1]), range(mask.shape[0]))
    cols_f = cols.ravel()
    rows_f = rows.ravel()
    for i, p in enumerate(paths):
        if i not in skip:
            inside = p.contains_points(zip(cols.ravel(), rows.ravel()))
            mask+=inside.reshape(mask.shape)
    return mask.astype('uint8')

def mask(imgsz, polygons, inverse=False):
    im_mask = PIL.Image.new(mode='1', size=tuple(np.array(imgsz).astype(int)))
    draw = PIL.ImageDraw.ImageDraw(im_mask)
    if isinstance(polygons, dict):
        polygons = polygons.values()
    for polygon in polygons:
        if isinstance(polygon, np.ndarray):
            polygon = [tuple(row) for row in polygon]
        draw.polygon(polygon, fill=1)
    mask = np.array(im_mask)
    if inverse:
        mask = ~mask
    return mask

def elevation_corrections(origin=None, xyz=None, squared_distances=None, earth_radius=6.3781e6, refraction=0.13):
    """
    Return elevation corrections for earth curvature and refraction.

    Arguments:
        origin (iterable): World coordinates of origin (x, y, (z))
        xyz (array): World coordinates of target points (n, 2+)
        squared_distances (iterable): Squared Euclidean distances
            between `origin` and `xyz`. Takes precedent if not `None`.
        earth_radius (float): Radius of the earth in the same units as `xyz`.
            Default is the equatorial radius in meters.
        refraction (float): Coefficient of refraction of light.
            Default is an average for standard atmospheric conditions.
    """
    # http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?topicname=how_viewshed_works
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/3d-analyst-toolbox/how-line-of-sight-works.htm
    # https://en.wikipedia.org/wiki/Atmospheric_refraction#Terrestrial_refraction
    if squared_distances is None:
        squared_distances = np.sum((xyz[:, 0:2] - origin[0:2])**2, axis=1)
    return (refraction - 1) * squared_distances / (2 * earth_radius)
