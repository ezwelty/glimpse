import datetime
import gzip
import pathlib
import json
import os
import pickle
import re
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import osgeo.gdal
import osgeo.gdal_array
import osgeo.ogr
import osgeo.osr
import progress.bar
import scipy.ndimage
import scipy.spatial


# ---- General ---- #


def format_list(
    x: Iterable, length: int = None, default: Any = None, dtype: Callable = None
) -> Optional[list]:
    """
    Return object as a formatted list.

    Arguments:
        x: Object to format
        length: Output object length.
            If `None`, ouput length is equal to input length.
            If shorter than the input, a subset of the input is returned.
            If longer than the input and `default` is `None`,
            must be a multiple of the input length (`x` is repeated).
            If `default` is not `None`, the output is padded with `default` elements.
        default: Default element value.
            If `None`, `x` is repeated to achieve length `length`.
        dtype: Data type to coerce list elements to.
            If `None`, data is left as-is.

    Raise:
        ValueError: Output length is not multiple of input length.

    Examples:
        >>> format_list([0, 1], length=1)
        [0]
        >>> format_list([0, 1], length=3)
        Traceback (most recent call last):
          ...
        ValueError: Output length is not multiple of input length
        >>> format_list([0, 1], length=3, default=2)
        [0, 1, 2]
        >>> format_list([0, 1], length=4)
        [0, 1, 0, 1]
        >>> format_list([0, 1], dtype=float)
        [0.0, 1.0]
        >>> format_list(None) is None
        True
    """
    if x is None:
        return x
    if not np.iterable(x):
        x = [x]
    elif not isinstance(x, list):
        x = list(x)
    if length:
        nx = len(x)
        if nx > length:
            x = x[0:length]
        elif nx < length:
            if default is not None:
                x += [default] * (length - nx)
            elif nx > 0:
                # Repeat list
                if length % nx != 0:
                    raise ValueError("Output length is not multiple of input length")
                x *= length // nx
    if dtype:
        x = [dtype(i) for i in x]
    return x


def numpy_dtype_minmax(dtype):
    """
    Return min, max allowable values for a numpy datatype.

    Arguments:
        dtype: Numpy datatype.

    Returns:
        tuple: Minimum and maximum values
    """
    if issubclass(dtype.type, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    if issubclass(dtype.type, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    if dtype.type in (np.bool_, np.bool):
        return False, True
    raise ValueError("Cannot determine min, max for " + str(dtype))


def numpy_to_native(x: Any) -> Any:
    """
    Convert numpy or native type to native type.

    Converts numpy types to native type,
    while leaving other objects (without :meth:`numpy.ndarray.tolist` method) unchanged.
    """
    # https://stackoverflow.com/a/42923092
    return getattr(x, "tolist", lambda: x)()


def strip_path(path, extensions=True):
    """
    Return the final component of a path with file extensions removed.

    Arguments:
        path (str): Path to file
        extensions: Maximum number of extensions to remove or `True` for all
    """
    basename = os.path.basename(path)
    if extensions:
        if extensions is True:
            extensions = -1
        return basename[::-1].split(".", maxsplit=extensions)[-1][::-1]
    return basename


def sorted_neighbors(x, y):
    """
    Return indices of neighbors.

    Arguments:
        x (iterable): Values sorted in ascending order
        y (iterable): Values to find neighbors for

    Returns:
        array: Index (in `x`) of left and right neighbors for each value in `y` (n, 2)
    """
    x, y = np.asarray(x), np.asarray(y)
    index = np.searchsorted(x, y)
    # index = 0 snap to 0
    # 0 < index < len(x) snap to index -1
    index[(index > 0) & (index < len(x))] -= 1
    # index = len(x) snap to index - 2
    index[index == len(x)] -= 2
    return np.column_stack((index, index + 1))


def sorted_nearest(x, y):
    """
    Return indices of nearest neighbors.

    Arguments:
        x (iterable): Values sorted in ascending order
        y (iterable): Values to find neighbors for

    Returns:
        array: Index (in `x`) of nearest neighbor for each value in `y` (n, )
    """
    x, y = np.asarray(x), np.asarray(y)
    neighbors = sorted_neighbors(x, y)
    nearest = np.argmin(np.abs(y.reshape(-1, 1) - x[neighbors]), axis=1)
    return neighbors[range(len(y)), nearest]


# ---- Pickles ---- #


def write_pickle(obj, path, gz=False, binary=True, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Write object to pickle file.

    Arguments:
        obj: Object to write
        path (str): Path to file
        gz (bool): Whether to use gzip compression
        binary (bool): Whether to write a binary pickle
        protocol (int): Protocol to use
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    if gz:
        fp = gzip.open(path, mode=mode)
    else:
        fp = open(path, mode=mode)
    pickle.dump(obj, file=fp, protocol=protocol)
    fp.close()


def read_pickle(path, gz=False, binary=True, **kwargs):
    """
    Read object from pickle file.

    Arguments:
        path (str): Path to file
        gz (bool): Whether pickle is gzip compressed
        binary (bool): Whether pickle is binary
        **kwargs: Arguments to :func:`pickle.load`.
    """
    mode = "rb" if binary else "r"
    if gz:
        fp = gzip.open(path, mode=mode)
    else:
        fp = open(path, mode=mode)
    obj = pickle.load(fp, **kwargs)
    fp.close()
    return obj


# ---- JSON ---- #


def read_json(path, **kwargs):
    """
    Read JSON from file.

    Arguments:
        path (str): Path to file
        **kwargs: Additional arguments passed to :func:`json.load`.
    """
    with open(path, mode="r") as fp:
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
        **kwargs: Additional arguments passed to :func:`json.dumps`.
    """
    txt = json.dumps(obj, **kwargs)
    if flat_arrays and kwargs.get("indent") >= 0:
        separators = kwargs.get("separators")
        sep = separators[0] if separators else ", "
        squished_sep = re.sub(r"\s", "", sep)

        def flatten(match):
            return re.sub(squished_sep, sep, re.sub(r"\s", "", match.group(0)))

        txt = re.sub(r"(\[\s*)+[^\]\{]*(\s*\])+", flatten, txt)
    if path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(txt)
        return None
    return txt


# ---- Arrays: General ---- #


def normalize(array):
    """
    Normalize a numeric array to mean 0, variance 1.

    Arguments:
        array (array): Input array
    """
    return (array - array.mean()) * (1 / array.std())


def gaussian_filter(array, mask=None, fill=False, **kwargs):
    """
    Return a gaussian-filtered array.

    Excludes cells by the method described in https://stackoverflow.com/a/36307291.

    Arguments:
        array (array): Array to filter
        mask (array): Boolean mask of cells to include (True) or exclude (False).
            If `None`, all cells are included.
        fill (bool): Whether to fill cells excluded by `mask` with interpolated values
        **kwargs (dict): Additional arguments to
            :func:`scipy.ndimage.filters.gaussian_filter`.
    """
    if mask is None:
        return scipy.ndimage.filters.gaussian_filter(array, **kwargs)
    x = array.copy()
    x[~mask] = 0
    xf = scipy.ndimage.filters.gaussian_filter(x, **kwargs)
    x[mask] = 1
    xf_sum = scipy.ndimage.filters.gaussian_filter(x, **kwargs)
    x = xf / xf_sum
    if not fill:
        x[~mask] = array[~mask]
    return x


def maximum_filter(array, mask=None, fill=False, **kwargs):
    """
    Return a maximum-filtered array.

    Excludes cells by setting them to the minimum value allowable by the datatype.

    Arguments:
        array (array): Array to filter
        mask (array): Boolean mask of cells to include (True) or exclude (False).
            If `None`, all cells are included.
        fill (bool): Whether to fill cells excluded by `mask` with interpolated values
        **kwargs (dict): Additional arguments to
            :func:`scipy.ndimage.filters.maximum_filter`.
    """
    if mask is None:
        return scipy.ndimage.filters.maximum_filter(array, **kwargs)
    dtype_min = numpy_dtype_minmax(array)[0]
    x = array.copy()
    mask = ~mask
    x[mask] = dtype_min
    x = scipy.ndimage.filters.maximum_filter(x, **kwargs)
    if fill:
        mask = x == dtype_min
    x[mask] = array[mask]
    return x


# ---- Arrays: Images ---- #


def compute_cdf(array, return_inverse=False):
    """
    Compute the cumulative distribution function of an array.

    Arguments:
        array (array): Input array
        return_inverse (bool): Whether to return the indices of the returned
            `values` that reconstruct `array`

    Returns:
        tuple:

            - array: Sorted unique values
            - array: Quantile of each value in `values`
            - array (optional): Indices of `values` which reconstruct `array`.
              Only returned if `return_inverse=True`.
    """
    results = np.unique(array, return_inverse=return_inverse, return_counts=True)
    # Normalize cumulative sum of counts by the number of pixels
    quantiles = np.cumsum(results[-1]) * (1.0 / array.size)
    if return_inverse:
        return results[0], quantiles, results[1]
    return results[0], quantiles


def match_histogram(source, template):
    """
    Adjust the values of an array such that its histogram matches that of a target
    array.

    Arguments:
        source (array): Array to transform.
        template: Histogram template as either an array (of any shape)
            or an iterable (unique values, unique value quantiles).

    Returns:
        array: Transformed `source` array
    """
    _, s_quantiles, inverse_index = compute_cdf(source, return_inverse=True)
    if isinstance(template, np.ndarray):
        template = compute_cdf(template, return_inverse=False)
    # Interpolate new values based on source and template quantiles
    new_values = np.interp(s_quantiles, template[1], template[0])
    return new_values[inverse_index].reshape(source.shape)


# ---- GIS ---- #


def crs_to_wkt(crs):
    """
    Convert coordinate reference system (CRS) to well-known text (WKT).

    Arguments:
        crs: Coordinate reference system as int (EPSG) or str (Proj4 or WKT)
    """
    obj = osgeo.osr.SpatialReference()
    if isinstance(crs, int):
        obj.ImportFromEPSG(crs)
    elif isinstance(crs, str):
        if re.findall(r"\[", crs):
            return crs
        elif re.findall(r":", crs):
            obj.ImportFromProj4(crs.lower())
        else:
            raise ValueError("crs string format not Proj4 or WKT")
    else:
        raise ValueError("crs must be int (EPSG) or str (Proj4 or WKT)")
    return obj.ExportToWkt()


def gdal_driver_from_path(path, raster=True, vector=True):
    ext = os.path.splitext(path)[1][1:].lower()
    for i in range(osgeo.gdal.GetDriverCount()):
        driver = osgeo.gdal.GetDriver(i)
        meta = driver.GetMetadata()
        is_raster = raster and meta.get(osgeo.gdal.DCAP_RASTER)
        is_vector = vector and meta.get(osgeo.gdal.DCAP_VECTOR)
        if is_raster or is_vector:
            extensions = meta.get(osgeo.gdal.DMD_EXTENSIONS)
            if extensions and ext in extensions.split(" "):
                return driver
    return None


def write_raster(a, path, driver=None, nan=None, crs=None, transform=None):
    a = np.atleast_3d(a)
    dtype = osgeo.gdal_array.NumericTypeCodeToGDALTypeCode(a.dtype)
    if not dtype:
        dtypes = ", ".join([x.__name__ for x in osgeo.gdal_array.codes.values()])
        raise ValueError(
            f"Unsupported array data type: {a.dtype}.\nSupported: {dtypes}"
        )
    if driver:
        msg = f"Unrecognized GDAL driver: {driver}"
        driver = osgeo.gdal.GetDriverByName(driver)
        if not driver:
            raise ValueError(msg)
    else:
        driver = gdal_driver_from_path(path, vector=False)
        if not driver:
            raise ValueError(f"Could not guess GDAL driver from path: {path}")
    meta = driver.GetMetadata()
    can_create = meta.get(osgeo.gdal.DCAP_CREATE)
    can_copy = meta.get(osgeo.gdal.DCAP_CREATECOPY)
    if not can_create and not can_copy:
        raise ValueError(f"Driver {driver.ShortName} cannot create files")
    create_driver = driver if can_create else osgeo.gdal.GetDriverByName("mem")
    output = create_driver.Create(
        utf8_path=path if can_create else "",
        xsize=a.shape[1],
        ysize=a.shape[0],
        bands=a.shape[2],
        eType=dtype,
    )
    if transform:
        output.SetGeoTransform(transform)
    if crs:
        wkt = crs_to_wkt(crs)
        output.SetProjection(wkt)
    for i in range(output.RasterCount):
        if nan is not None:
            output.GetRasterBand(i + 1).SetNoDataValue(nan)
        output.GetRasterBand(i + 1).WriteArray(a[:, :, i])
    if not can_create:
        output = driver.CreateCopy(path, output, 0)
    output.FlushCache()


# ---- Geometry ---- #


def boolean_split(x, mask, axis=0, circular=False, include="all"):
    """
    Split array by boolean mask.

    Select True groups with [0::2] if mask[0] is True, else [1::2].

    Arguments:
        x (array): Array to split
        mask (array): Boolean array with same length as `x` along `axis`
        axis (int): Axis along which to split
        circular (bool): Whether to treat `x` as closed (x[-1] -> x[0])
        include (str): Whether to return 'all', 'true', or 'false' groups
    """
    # See https://stackoverflow.com/a/36518315/8161503
    cuts = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    splits = np.split(x, cuts, axis=axis)
    if circular and len(splits) > 1 and mask[0] is mask[-1]:
        splits[0] = np.concatenate((splits[-1], splits[0]), axis=axis)
        splits.pop(-1)
    if include == "all":
        return splits
    if include == "true":
        index = slice(0, None, 2) if mask[0] else slice(1, None, 2)
        return splits[index]
    if include == "false":
        index = slice(1, None, 2) if mask[0] else slice(0, None, 2)
        return splits[index]
    return []


def in_box(points, box):
    """
    Tests whether each point is in (or on) the box.

    Works in any dimension.

    Arguments:
        points (array): Point coordinates (npts, ndim)
        box (iterable): Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, )
    """
    box = unravel_box(box)
    return np.all((points >= box[0, :]) & (points <= box[1, :]), axis=1)


def clip_polyline_box(line, box, t=False):
    """
    Returns segments of line within the box.

    Vertices are inserted as needed on the box boundary.
    For speed, does not check for segments within the box
    entirely between two adjacent line vertices.

    Arguments:
        line (array): 2 or 3D point coordinates (npts, ndim)
        box (iterable): Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, )
        t (bool): Last column of `line` are distances along line, linearly interpolated
            at splits
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
                segments[i] = np.insert(
                    segments[i], len(segments[i]), origin + ti * distance, axis=0
                )
    return segments[trues]


def intersect_edge_box(origin, distance, box):
    """
    Returns intersection of edge with box.

    Arguments:
        origin (iterable): Coordinates of 2 or 3D point (ndim, )
        distance (iterable): Distance to end point (ndim, )
        box (iterable): Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, )
    """
    distance_2d = np.asarray(distance).reshape(1, -1)
    t = np.nanmin(intersect_rays_box(origin, distance_2d, box, t=True))
    if t > 0 and t < 1:
        return t
    return None


def intersect_rays_box(origin, directions, box, t=False):
    """
    Return intersections of rays with a(n axis-aligned) box.

    Works in both 2 and 3 dimensions. Vectorized version of algorithm by Williams et al.
    (2011) optimized for rays with a common origin. Also inspired by
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

    Arguments:
        origin (iterable): Common origin of rays [x, y(, z)]
        directions (array): Directions of rays [[dx, dy(, dz)], ...]
        box (iterable): Box min and max vertices
            [xmin, ymin(, zmin), xmax, ymax(, zmax)]

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
        idx = np.ravel_multi_index(
            (all_rays, 2 + (1 - sign[:, 2]) * ndims), bounds.shape
        )
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
    return origin + tmin[:, None] * directions, origin + tmax[:, None] * directions


# TODO: Implement faster run-slice
# (http://www.phatcode.net/res/224/files/html/ch36/36-03.html)
def bresenham_line(start, end):
    """
    Return grid indices along a line between two grid indices.

    Uses Bresenham's run-length algorithm. Not all intersected grid cells are returned,
    only those with centers closest to the line. Code modified for speed from
    http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm.

    Arguments:
        start (iterable): Start position (xi, yi)
        end (iterable): End position (xi, yi)

    Returns:
        array: Grid indices [[xi, yi], ...]
    """
    x1, y1 = start
    x2, y2 = end
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
        for x in range(x1, x2 + 1):
            points.append((y, x))
            error -= abs_dy
            if error < 0:
                y += ystep
                error += dx
    else:
        for x in range(x1, x2 + 1):
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
    """
    Return grid indices along a circular path.

    Uses Bresenham's circle algorithm.
    Code modified from https://en.wikipedia.org/wiki/Midpoint_circle_algorithm.

    Arguments:
        center (iterable): Circle center (x, y)
        radius (float): Circle radius in pixels

    Returns:
        array: Grid indices [[xi, yi], ...]
    """
    x0, y0 = center
    # Compute number of points
    octant_size = int(np.floor((np.sqrt(2) * (radius - 1) + 4) / 2))
    n_points = 8 * octant_size
    x = 0
    y = radius
    f = 1 - radius
    dx = 1
    dy = -2 * radius
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


def intersect_boxes(boxes):
    """
    Return intersection of boxes.

    Arguments:
        boxes (iterable): Boxes, each in the format (minx, ..., maxx, ...)
    """
    boxes = np.asarray(boxes)
    assert boxes.shape[1] % 2 == 0
    ndim = boxes.shape[1] // 2
    boxmin = np.nanmax(boxes[:, 0:ndim], axis=0)
    boxmax = np.nanmin(boxes[:, ndim:], axis=0)
    if any(boxmax - boxmin <= 0):
        raise ValueError("Boxes do not intersect")
    return np.hstack((boxmin, boxmax))


def pairwise_distance(x, y, metric="sqeuclidean", **params):
    """
    Return the pairwise distance between two sets of points.

    Arguments:
        x (iterable): First set of n-d points
        y (iterable): Second set of n-d points
        metric (str): Distance metric. See :func:`scipy.spatial.distance.cdist`.
        **params (dict): Additional arguments to :func:`scipy.spatial.distance.cdist`.

    Returns:
        array: Pairwise distances, where [i, j] = distance(x[i], y[j])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return scipy.spatial.distance.cdist(
        x if x.ndim > 1 else x.reshape(-1, 1),
        y if y.ndim > 1 else y.reshape(-1, 1),
        metric=metric,
        **params,
    )


def interpolate_line(
    vertices, x=None, xi=None, n=None, dx=None, error=True, fill="endpoints"
):
    """
    Return points at the specified distances along a line.

    Arguments:
        vertices (array): Coordinates of vertices (n, d)
        x (iterable): Distance measure at each vertex (n, ). If `None`,
            the cumulative Euclidean distance is used.
            Undefined behavior results if not strictly monotonic.
        xi (iterable): Distance of interpolated points along line
        n (int): Number of evenly-spaced points to return
            (ignored if `xi` is not `None`)
        dx (float): Nominal distance between evenly-spaced points
            (ignored if `xi` or `n` is not `None`)
        error (bool): Whether to raise ValueError if any `xi` are outside range of `x`
        fill: Value(s) to use for `xi` beyond `x[0]` and `xi` beyond `x[-1]`.
            If 'endpoints', uses (`vertices[0]`, `vertices[-1]`).
    """
    assert not all((xi is None, n is None, dx is None))
    if x is None:
        # Compute total distance at each vertex
        x = np.cumsum(np.sqrt(np.sum(np.diff(vertices, axis=0) ** 2, axis=1)))
        # Set first vertex at 0
        x = np.insert(x, 0, 0)
    if xi is None:
        if n is None:
            n = abs((x[-1] - x[0]) / dx)
            if n == int(n):
                n += 1
            n = int(round(n))
        xi = np.linspace(start=x[0], stop=x[-1], num=n, endpoint=True)
        # Ensure defaults for speed
        error = False
        fill = "endpoints"
    # x must be increasing
    if len(x) > 1 and x[1] < x[0]:
        sort_index = np.argsort(x)
        x = x[sort_index]
        vertices = vertices[sort_index, :]
    # Interpolate each dimension and combine
    result = np.column_stack(
        [np.interp(xi, x, vertices[:, i]) for i in range(vertices.shape[1])]
    )
    if fill == "endpoints":
        if error is False:
            return result
        fill = (vertices[0], vertices[-1])
    if not np.iterable(fill):
        fill = (fill, fill)
    left = np.less(xi, x[0])
    right = np.greater(xi, x[-1])
    if x[0] > x[-1]:
        right, left = left, right
    if error and (left.any() or right.any()):
        raise ValueError("Requested distance outside range")
    result[left, :] = fill[0]
    result[right, :] = fill[1]
    return result


def unravel_box(box):
    """
    Return a box in unravelled format.

    Arguments:
        box (iterable): Box (minx, ..., maxx, ...)

    Returns:
        array: [[minx, ...], [maxx, ...]]
    """
    box = np.asarray(box)
    assert box.size % 2 == 0
    ndim = box.size // 2
    return box.reshape(-1, ndim)


def bounding_box(points):
    """
    Return bounding box of points.

    Arguments:
        points (iterable): Points, each in the format (x, ...)
    """
    points = np.asarray(points)
    return np.hstack((np.min(points, axis=0), np.max(points, axis=0)))


def box_to_polygon(box):
    """
    Return box as polygon.

    Arguments:
        box (iterable): Box (minx, ..., maxx, ...)
    """
    box = unravel_box(box)
    return np.column_stack((box[(0, 0, 1, 1, 0), 0], box[(0, 1, 1, 0, 0), 1]))


def box_to_grid(box, step, snap=None, mode="grids"):
    """
    Return grid of points inside box.

    Arguments:
        box (iterable): Box (minx, ..., maxx, ...)
        step: Grid spacing for all (float) or each (iterable) dimension
        snap (iterable): Point to align grid to (need not be inside box).
            If `None`, the `box` minimum is used
        mode (str): Return format ('vectors' or 'grids')

    Returns:
        tuple: Either vectors or grids for each dimension
    """
    box = unravel_box(box)
    ndim = box.shape[1]
    step = step if np.iterable(step) else (step,) * ndim
    if snap is None:
        snap = box[0, :]
    shift = (snap - box[0, :]) % step
    n = (np.diff(box, axis=0).ravel() - shift) // step
    arrays = (
        np.linspace(
            box[0, i] + shift[i], box[0, i] + shift[i] + n[i] * step[i], int(n[i]) + 1
        )
        for i in range(ndim)
    )
    if mode == "vectors":
        return tuple(arrays)
    return np.meshgrid(*arrays)


def grid_to_points(grid):
    """
    Return grid as points.

    Arguments:
        grid (iterable): Array of grid coordinates for each dimension (X, ...)

    Returns:
        array: Point coordinates [[x, ...], ...]
    """
    return np.reshape(grid, (len(grid), -1)).T


def get_scale_from_size(old: Iterable[int], new: Iterable[int]) -> Optional[float]:
    """
    Return the scale factor that achieves a target integer grid size.

    Arguments:
        old: Initial size (nx, ny)
        new: Target size (nx, ny)

    Returns:
        Scale factor, or `None` if the **new** size cannot be achieved exactly.

    Example:
        >>> get_scale_from_size(1, 2)
        2.0
        >>> get_scale_from_size((1, 1, 1), (2, 2, 2))
        2.0
        >>> old, new = (133, 311), (40, 94)
        >>> scale = get_scale_from_size(old, new)
        >>> (round(old[0] * scale), round(old[1] * scale)) == new
        True
        >>> get_scale_from_size((1, 1), (1, 2)) is None
        True
    """
    old = np.atleast_1d(old)
    new = np.atleast_1d(new)
    if all(new == old):
        return 1.0
    initial = new / old
    if all(initial[0] == initial):
        return initial[0]

    def err(scale: float) -> float:
        return np.sum(np.abs(np.round(scale * old) - new))

    bounds = [(np.floor(initial.min()), np.ceil(initial.max()))]
    fit = scipy.optimize.differential_evolution(func=err, bounds=bounds)
    if fit["fun"] == 0:
        return float(fit["x"])
    return None


# ---- Image formation ---- #


def rasterize_points(
    rows: Iterable[int],
    cols: Iterable[int],
    values: Iterable[Union[Union[int, float, bool], Iterable[Union[int, float, bool]]]],
    shape: Iterable[int] = None,
    a: np.ndarray = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Rasterize points by array indices.

    Points are aggregated by equal row and column indices by their mean.

    Arguments:
        rows: Point row indices (n, ).
        cols: Point column indices (n, ).
        values: Point values (n, ) or (n, d).
        shape: Output array row and column size. Ignored if `a` is provided.
        a: Array to modify in-place with point mean values.
            Must be a 2-d array or 3-d array of depth 1 if `values` is (n, ) or (n, 1)
            and a 3-d array of depth d if `values` is (n, d).

    Returns:
        If `a` is `None`,
        unique (and sorted) flat point indices and mean point values (m, ) or (m, d).

    Examples:
        Standard usage is to pass point indices, values, and a target 2-d array shape,
        and in return receive unique flat indices and mean values.

        >>> rows = (0, 0, 1)
        >>> cols = (0, 0, 1)
        >>> values = (1, 2, 3)
        >>> shape = (4, 3)
        >>> rasterize_points(rows, cols, values, shape=shape)
        (array([0, 4]), array([1.5, 3. ]))

        Alternatively, an existing array can be passed instead of an array shape,
        to be modified in-place.

        >>> a = np.full(shape, np.nan)
        >>> rasterize_points(rows, cols, values, a=a)
        >>> a
        array([[1.5, nan, nan],
               [nan, 3. , nan],
               [nan, nan, nan],
               [nan, nan, nan]])

        Multi-dimensional point values are also supported.

        >>> values = [[1, 10], [2, 20], [3, 30]]
        >>> a = np.full((shape[0], shape[1], 2), np.nan)
        >>> rasterize_points(rows, cols, values, a=a)
        >>> a[..., 1]
        array([[15., nan, nan],
               [nan, 30., nan],
               [nan, nan, nan],
               [nan, nan, nan]])
    """
    values = np.asarray(values)
    if shape is None:
        shape = a.shape
    idx = np.ravel_multi_index((rows, cols), shape[0:2])
    uidx, labels = np.unique(idx, return_inverse=True)
    counts = np.bincount(labels)
    if values.ndim == 1 or (a is not None and values.shape[1] == 1):
        sums = np.bincount(labels, weights=values.flat)
    else:
        sums = np.column_stack(
            [np.bincount(labels, weights=values[:, i]) for i in range(values.shape[1])]
        )
        counts = counts.reshape(-1, 1)
    means = sums * (1 / counts)
    if a is None:
        return uidx, sums * (1 / counts)
    if means.ndim == 1:
        a.flat[uidx] = means
    else:
        ij = np.unravel_index(uidx, shape[0:2])
        a[ij] = means
    return None


def polygons_to_mask(
    polygons: Iterable[Iterable[Iterable[Union[int, float]]]],
    size: Iterable[int],
    holes: Iterable[Iterable[Iterable[Union[int, float]]]] = None,
) -> np.ndarray:
    """
    Returns a boolean array of cells inside polygons.

    The upper-left corner of the upper-left cell of the array is (0, 0).

    Arguments:
        polygons: Polygons [ [ (x, y), ...], ... ].
        size: Array size (nx, ny).
        holes: Polygons representing holes in `polygons`.

    Examples:
        >>> polygons = [
        ...     [(1, 1), (4, 1), (4, 4), (1, 4)],
        ...     [(0, 0), (0.6, 0), (0.6, 0.6), (0, 0.6)]
        ... ]
        >>> holes = [[(2, 2), (3, 2), (3, 3), (2, 3)]]
        >>> polygons_to_mask(polygons, (5, 5), holes)
        array([[ True, False, False, False, False],
               [False,  True,  True,  True, False],
               [False,  True, False,  True, False],
               [False,  True,  True,  True, False],
               [False, False, False, False, False]])
    """

    def _gdal_polygon(
        polygon: Iterable[Iterable[Union[int, float]]]
    ) -> osgeo.ogr.Geometry:
        ring = osgeo.ogr.Geometry(osgeo.ogr.wkbLinearRing)
        for x, y in polygon:
            ring.AddPoint(x, y)
        polygon = osgeo.ogr.Geometry(osgeo.ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        return polygon

    def _gdal_polygon_datasource(
        polygons: Iterable[Iterable[Iterable[Union[int, float]]]]
    ) -> osgeo.ogr.DataSource:
        driver = osgeo.ogr.GetDriverByName("Memory")
        ds = driver.CreateDataSource("out")
        layer = ds.CreateLayer(
            "polygons", srs=osgeo.osr.SpatialReference(), geom_type=osgeo.ogr.wkbPolygon
        )
        defn = layer.GetLayerDefn()
        for polygon in polygons:
            feature = osgeo.ogr.Feature(defn)
            feature.SetGeometry(_gdal_polygon(polygon))
            layer.CreateFeature(feature)
            feature = None
        return ds

    driver = osgeo.gdal.GetDriverByName("MEM")
    raster = driver.Create("", size[0], size[1], 1, osgeo.gdal.GDT_Byte)
    raster.SetGeoTransform((0, 1, 0, 0, 0, 1))
    ds = _gdal_polygon_datasource(polygons)
    layer = ds.GetLayer(0)
    osgeo.gdal.RasterizeLayer(raster, [1], layer, burn_values=[1])
    if holes:
        ds = _gdal_polygon_datasource(holes)
        layer = ds.GetLayer(0)
        osgeo.gdal.RasterizeLayer(raster, [1], layer, burn_values=[0])
    return raster.ReadAsArray().astype(bool)


def elevation_corrections(
    origin=None, xyz=None, squared_distances=None, radius=6.3781e6, refraction=0.13,
):
    """
    Return elevation corrections for surface curvature and atmospheric refraction.

    Arguments:
        origin (iterable): World coordinates of origin (x, y, (z))
        xyz (array): World coordinates of target points (n, 2+)
        squared_distances (iterable): Squared Euclidean distances
            between `origin` and `xyz`. Takes precedence if not `None`.
        radius (float): Radius of curvature in the same units as `xyz`.
            Default is the Earth's equatorial radius in meters.
        refraction (float): Coefficient of refraction of light.
            Default is an average for standard Earth atmospheric conditions.
    """
    # http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?topicname=how_viewshed_works
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/3d-analyst-toolbox/how-line-of-sight-works.htm
    # https://en.wikipedia.org/wiki/Atmospheric_refraction#Terrestrial_refraction
    if squared_distances is None:
        squared_distances = np.sum((xyz[:, 0:2] - origin[0:2]) ** 2, axis=1)
    return (refraction - 1) * squared_distances / (2 * radius)


# ---- Time ----


def datetimes_to_float(datetimes):
    """
    Return datetimes as float.

    Converts datetimes to POSIX timestamps - the number of seconds since
    1970-01-01 00:00:00 UTC.

    Arguments:
        datetimes (iterable): Datetime objects
    """
    try:
        return [xi.timestamp() for xi in datetimes]
    except AttributeError:
        # Python 2
        epoch = datetime.datetime.fromtimestamp(0)
        return [(xi - epoch).total_seconds() for xi in datetimes]


def pairwise_distance_datetimes(x, y):
    """
    Return the pairwise distances between two sets of datetimes.

    Datetime wrapper for :func:`pairwise_distance`.

    Arguments:
        x (iterable): Datetime objects
        y (iterable): Datetime objects

    Returns:
        array: Pairwise distances in seconds, where [i, j] = distance(x[i], y[j])
    """
    return pairwise_distance(
        datetimes_to_float(x), datetimes_to_float(y), metric="minkowski", p=1
    )


def datetime_range(start, stop, step):
    """
    Return a sequence of datetime.

    Arguments:
        start (datetime): Start datetime
        stop (datetime): End datetime (inclusive)
        step (timedelta): Time step
    """
    max_steps = (stop - start) // step
    return [start + n * step for n in range(max_steps + 1)]


def select_datetimes(datetimes, start=None, end=None, snap=None, maxdt=None):
    """
    Return indices of datetimes matching the specified criteria.

    Arguments:
        datetimes (iterable): Datetime objects in ascending order
        start (datetime): Start datetime, or `min(datetimes)` if `None`
        end (datetime): End datetime, or `max(datetimes)` if `None`
        snap (timedelta): Interval (relative to 1970-01-01 00:00:00)
            on which to select nearest `datetimes`, or all if `None`
        maxdt (timedelta): Maximum distance from nearest `snap` to
            select `datetimes`. If `None`, defaults to half of `snap`.
    """
    datetimes = np.asarray(datetimes)
    selected = np.ones(datetimes.shape, dtype=bool)
    if start:
        selected &= datetimes >= start
    else:
        start = datetimes[0]
        if snap:
            start -= snap
    if end:
        selected &= datetimes <= end
    else:
        end = datetimes[-1]
        if snap:
            end += snap
    assert end >= start
    if snap:
        origin = datetime.datetime(1970, 1, 1, 0, 0, 0)
        shift = (origin - start) % snap
        start = start + shift
        targets = datetime_range(start=start, stop=end, step=snap)
        nearest = sorted_nearest(datetimes, targets)
        if maxdt is None:
            maxdt = snap * 0.5
        distances = np.abs(targets - datetimes[nearest])
        nearest = np.unique(nearest[distances <= maxdt])
        temp = np.zeros(datetimes.shape, dtype=bool)
        temp[nearest] = True
        selected &= temp
    return np.nonzero(selected)[0]


# ---- Internal ----


def _progress_bar(max):
    return progress.bar.Bar(
        "",
        fill="#",
        max=max,
        hide_cursor=False,
        suffix="%(percent)3d%% (%(index)d of %(max)d) %(elapsed_td)s",
    )


def _parse_parallel(parallel):
    if parallel is True:
        n = os.cpu_count()
        if n is None:
            raise NotImplementedError("Cannot determine number of CPUs")
    elif parallel is False:
        n = 0
    else:
        n = parallel
    return n
