import numpy as np
import cPickle
import pyproj
import json
import collections
import copy
import dem
import pandas
import scipy

# Save and load commands for efficient pickle objects
def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

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
        kwargs (dict): Additional arguments passed to `json.load()`
    """
    with open(path, "r") as fp:
        return json.load(fp, **kwargs)

def write_json(obj, path=None, **kwargs):
    """
    Write object to JSON.

    Arguments:
        obj: Object to write as JSON
        path (str): Path to file. If `None`, result is returned as a string.
        kwargs (dict): Additional arguments passed to `json.dump()` or `json.dumps()`
    """
    if path:
        with open(path, "w") as fp:
            json.dump(obj, fp, **kwargs)
        return None
    else:
        return json.dumps(obj, **kwargs)

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
            return set_z(coords, elevation.sample(coords))
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

def boolean_split(x, mask, axis=0, circular=False):
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
    return splits

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

def intersect_rays_box(origin, directions, box):
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
    # Apply y-axis intersections
    idx = np.ravel_multi_index((all_rays, (1 - sign[:, 0]) * ndims), bounds.shape)
    tmax = (fbounds[idx] - origin[0]) * invdir[:, 0]
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
        idx = np.ravel_multi_index((all_rays, sign[:, 2] * ndims), bounds.shape)
        tzmin = (fbounds[idx] - origin[2]) * invdir[:, 2]
        idx = np.ravel_multi_index((all_rays, (1 - sign[:, 2]) * ndims), bounds.shape)
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

def dms_to_degrees(degrees, minutes, seconds):
    """
    Convert degree-minute-second to decimal degrees.
    """
    return degrees + minutes / 60.0 + seconds / 3600.0
