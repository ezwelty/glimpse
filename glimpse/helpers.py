import datetime
import gzip
import pathlib
import json
import os
import pickle
import re
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

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

    Raises:
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


def numpy_dtype_minmax(
    dtype: np.dtype,
) -> Union[Tuple[int, int], Tuple[float, float], Tuple[bool, bool]]:
    """
    Return min, max allowable values for a numpy datatype.

    Arguments:
        dtype: Numpy datatype.

    Returns:
        Minimum and maximum values.

    Raises:
        ValueError: Cannot determine min, max for datatype.

    Examples:
        >>> numpy_dtype_minmax(np.dtype(int))
        (-9223372036854775808, 9223372036854775807)
        >>> numpy_dtype_minmax(np.dtype(float))
        (-1.7976931348623157e+308, 1.7976931348623157e+308)
        >>> numpy_dtype_minmax(np.dtype(bool))
        (False, True)
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

    Leaves objects without a `tolist()` method unchanged.

    Examples:
        >>> numpy_to_native(np.array([1, 2]))
        [1, 2]
        >>> numpy_to_native([1, 2])
        [1, 2]
    """
    # https://stackoverflow.com/a/42923092
    return getattr(x, "tolist", lambda: x)()


def strip_path(path: str, extensions: bool = True) -> str:
    """
    Return the final component of a path with file extensions removed.

    Arguments:
        path: Path.
        extensions: Maximum number of extensions to remove, or `True` for all.

    Examples:
        >>> strip_path('foo/bar')
        'bar'
        >>> strip_path('foo/bar.ext')
        'bar'
        >>> strip_path('foo/bar.ext.ext2')
        'bar'
        >>> strip_path('foo/bar.ext.ext2', extensions=1)
        'bar.ext'
    """
    basename = os.path.basename(path)
    if extensions:
        if extensions is True:
            extensions = -1
        return basename[::-1].split(".", maxsplit=extensions)[-1][::-1]
    return basename


def _sorted_neighbors(x: Iterable, y: Iterable) -> np.ndarray:
    """
    Return indices of neighbors.

    Arguments:
        x: Values sorted in ascending order.
        y: Values to find neighbors for in `x` (n, ).

    Returns:
        Index (in `x`) of left and right neighbors for each value in `y` (n, 2).
    """
    index = np.searchsorted(x, y)
    # index = 0 snap to 0
    # 0 < index < len(x) snap to index - 1
    index[(index > 0) & (index < len(x))] -= 1
    # index = len(x) snap to index - 2
    index[index == len(x)] -= 2
    return np.column_stack((index, index + 1))


def sorted_nearest(x: Iterable, y: Iterable) -> np.ndarray:
    """
    Return indices of nearest neighbors.

    Arguments:
        x: Values sorted in ascending order.
        y: Values to find neighbors for in `x` (n, ).

    Returns:
        Index (in `x`) of nearest neighbor for each value in `y` (n, ).

    Examples:
        >>> sorted_nearest([0, 1, 2], [-1, 0, 3, 1.1])
        array([0, 0, 2, 1])
    """
    x, y = np.asarray(x), np.asarray(y)
    neighbors = _sorted_neighbors(x, y)
    nearest = np.argmin(np.abs(y.reshape(-1, 1) - x[neighbors]), axis=1)
    return neighbors[range(len(y)), nearest]


# ---- Pickles ---- #


def write_pickle(
    obj: Any, path: str, gz: bool = False, binary: bool = True, **kwargs: Any
) -> None:
    """
    Write object to pickle file.

    Arguments:
        obj: Object to write.
        path: Path to file.
        gz: Whether to use gzip compression.
        binary: Whether to write a binary pickle.
        **kwargs: Optional arguments to :func:`pickle.dump`.
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    if gz:
        fp = gzip.open(path, mode=mode)
    else:
        fp = open(path, mode=mode)
    pickle.dump(obj, file=fp, protocol=protocol)
    fp.close()


def read_pickle(path: str, gz: bool = False, binary: bool = True, **kwargs: Any) -> Any:
    """
    Read object from pickle file.

    Arguments:
        path: Path to file.
        gz: Whether pickle is gzip compressed.
        binary: Whether pickle is binary.
        **kwargs: Optional arguments to :func:`pickle.load`.
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


def read_json(path: str, **kwargs: Any) -> Union[dict, list]:
    """
    Read JSON from file.

    Arguments:
        path: Path to file.
        **kwargs: Optional arguments to :func:`json.load`.
    """
    with open(path, mode="r") as fp:
        return json.load(fp, **kwargs)


def write_json(
    obj: Union[dict, list], path: str = None, flat_arrays: bool = False, **kwargs: Any
) -> Optional[str]:
    """
    Write object to JSON.

    Arguments:
        obj: Object to write as JSON.
        path: Path to file.
        flat_arrays: Whether to flatten JSON arrays to a single line.
            By default, :func:`json.dumps` puts each array element on a new line if
            `indent` is `0` or greater.
        **kwargs: Optional arguments to :func:`json.dumps`.

    Returns:
        JSON string (if `path` is `None`). 

    Examples:
        >>> write_json({'x': [0, 1]})
        '{"x": [0, 1]}'
        >>> write_json({'x': [0, 1]}, indent=2)
        '{\\n  "x": [\\n    0,\\n    1\\n  ]\\n}'
        >>> write_json({'x': [0, 1]}, indent=2, flat_arrays=True)
        '{\\n  "x": [0, 1]\\n}'
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


# ---- Arrays ---- #


def normalize(a: np.ndarray) -> np.ndarray:
    """
    Normalize an array to mean 0, variance 1.

    Arguments:
        a: Array to normalize.

    Returns:
        Normalized floating-point array of the same shape as `a`.

    Examples:
        >>> a = np.array([0, 1, 2, 3])
        >>> x = normalize(a)
        >>> x.shape == a.shape
        True
        >>> x.mean()
        0.0
        >>> x.std()
        1.0
    """
    return (a - a.mean()) * (1 / a.std())


def gaussian_filter(
    a: np.ndarray, mask: np.ndarray = None, fill: bool = False, **kwargs: Any
) -> np.ndarray:
    """
    Apply gaussian filter to array.

    Excludes cells by the method described in https://stackoverflow.com/a/36307291.

    Arguments:
        a: Array to filter.
        mask: Boolean mask of cells to include (True) or exclude (False).
            If `None`, all cells are included.
        fill: Whether to fill cells excluded by `mask` with interpolated values.
        **kwargs: Optional arguments to :func:`scipy.ndimage.filters.gaussian_filter`.

    Returns:
        Gaussian-filtered array of the same shape as `a`.

    Examples:
        >>> a = np.array([[np.nan, 1], [2, np.nan]])
        >>> gaussian_filter(a, sigma=1, truncate=1)
        array([[nan, nan],
               [nan, nan]])
        >>> gaussian_filter(a, sigma=1, mask=~np.isnan(a))
        array([[       nan, 1.23154033],
               [1.76845967,        nan]])
        >>> gaussian_filter(a, sigma=1, mask=~np.isnan(a), fill=True)
        array([[1.5       , 1.23154033],
               [1.76845967, 1.5       ]])
    """
    if mask is None:
        return scipy.ndimage.filters.gaussian_filter(a, **kwargs)
    x = a.copy()
    x[~mask] = 0
    xf = scipy.ndimage.filters.gaussian_filter(x, **kwargs)
    x[mask] = 1
    xf_sum = scipy.ndimage.filters.gaussian_filter(x, **kwargs)
    x = xf / xf_sum
    if not fill:
        x[~mask] = a[~mask]
    return x


def maximum_filter(
    a: np.ndarray, mask: np.ndarray = None, fill: bool = False, **kwargs: Any
) -> np.ndarray:
    """
    Apply maximum filter to array.

    Excludes cells by setting them to the minimum value allowable by the datatype.

    Arguments:
        a: Array to filter.
        mask: Boolean mask of cells to include (True) or exclude (False).
            If `None`, all cells are included.
        fill: Whether to fill cells excluded by `mask` with interpolated values.
        **kwargs: Optional arguments to :func:`scipy.ndimage.filters.maximum_filter`.

    Returns:
        Maximum-filtered array of the same shape as `a`.

    Examples:
        >>> a = np.array([[np.nan, 1], [2, np.nan]])
        >>> maximum_filter(a, size=3)
        array([[nan, nan],
               [nan, nan]])
        >>> maximum_filter(a, size=3, mask=~np.isnan(a))
        array([[nan,  2.],
               [ 2., nan]])
        >>> maximum_filter(a, size=3, mask=~np.isnan(a), fill=True)
        array([[2., 2.],
               [2., 2.]])
    """
    if mask is None:
        return scipy.ndimage.filters.maximum_filter(a, **kwargs)
    dtype_min = numpy_dtype_minmax(a.dtype)[0]
    x = a.copy()
    mask = ~mask
    x[mask] = dtype_min
    x = scipy.ndimage.filters.maximum_filter(x, **kwargs)
    if fill:
        mask = x == dtype_min
    x[mask] = a[mask]
    return x


def compute_cdf(
    a: np.ndarray, return_inverse: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return the cumulative distribution function (CDF) of an array.

    Arguments:
        a: Array.
        return_inverse: Whether to return the indices of the unique values
            that reconstruct `a`.

    Returns:
        Tuple of sorted unique values (n, ),
            the probability of being less than or equal to each value (n, ),
            and, if `return_inverse` is `True`,
            the indices which reconstruct `a` from the unique values.

    Examples:
        >>> a = np.array([3, 2, 1, 2])
        >>> compute_cdf(a)
        (array([1, 2, 3]), array([0.25, 0.75, 1.  ]))
        >>> compute_cdf(a, return_inverse=True)
        (array([1, 2, 3]), array([0.25, 0.75, 1.  ]), array([2, 1, 0, 1]))
    """
    results = np.unique(a, return_inverse=return_inverse, return_counts=True)
    # Normalize cumulative sum of counts by the number of cells
    quantiles = np.cumsum(results[-1]) / a.size
    if return_inverse:
        return results[0], quantiles, results[1]
    return results[0], quantiles


def match_cdf(
    a: np.ndarray, cdf: Union[Tuple[Iterable, Iterable], np.ndarray]
) -> np.ndarray:
    """
    Transform array to match a cumulative distribution function (CDF).

    Arguments:
        a: Array to transform.
        cdf: Cumulative distribution function (sorted unique values, probabilities)
            or an array from which to compute a CDF.

    Returns:
        Values of `a` transformed to match `cdf`.

    Examples:
        >>> a = np.array([3, 2, 1, 2])
        >>> b = np.array([4, 2, 1, 2, 4, 2, 1, 2])
        >>> match_cdf(a, b)
        array([4., 2., 1., 2.])
        >>> match_cdf(a, compute_cdf(b))
        array([4., 2., 1., 2.])
    """
    _, quantiles, inverse = compute_cdf(a, return_inverse=True)
    if isinstance(cdf, np.ndarray):
        cdf = compute_cdf(cdf, return_inverse=False)
    values = np.interp(quantiles, cdf[1], cdf[0])
    return values[inverse].reshape(a.shape)


# ---- GIS ---- #


def crs_to_wkt(crs: Union[int, str]) -> str:
    """
    Convert coordinate reference system (CRS) to well-known text (WKT).

    Arguments:
        crs: Coordinate reference system as int (EPSG) or str (Proj4 or WKT).

    Returns:
        Coordinate reference system as well-known text (WKT).

    Raises:
        ValueError: String CRS format not Proj4 or WKT.
        ValueError: Unsupported CRS format.

    Examples:
        >>> crs_to_wkt(4326)
        'GEOGCS["WGS 84",DATUM["WGS_1984",...]'
        >>> crs_to_wkt('+init=epsg:4326')
        'GEOGCS["WGS 84",DATUM["WGS_1984",...]'
    """
    obj = osgeo.osr.SpatialReference()
    if isinstance(crs, int):
        obj.ImportFromEPSG(crs)
    elif isinstance(crs, str):
        if re.findall(r"\[", crs):
            obj.ImportFromWkt(crs)
        elif re.findall(r"\+", crs):
            obj.ImportFromProj4(crs)
        else:
            raise ValueError(f"String CRS format not Proj4 or WKT: {crs}")
    else:
        raise ValueError(f"Unsupported CRS format: {crs}")
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


def write_raster(
    a: np.ndarray,
    path: str,
    driver: str = None,
    nan: Union[float, int] = None,
    crs: Union[int, str] = None,
    transform: Iterable[Union[int, float]] = None,
) -> None:
    """
    Write array to raster dataset.

    Arguments:
        a: Array to write as raster.
        path: Path to file.
        driver: GDAL driver name (see https://gdal.org/drivers/raster).
            If `None`, guessed from `path`.
        nan: Value to use in raster to represent NaN in array.
        crs: Coordinate reference system as either EPSG code or Proj4 or WKT string.
        transform: Affine transform mapping pixel positions to map positions
            (see https://gdal.org/user/raster_data_model.html?#affine-geotransform).

    Raises:
        ValueError: Unsupported array data type.
        ValueError: Unrecognized GDAL driver.
        ValueError: Could not guess GDAL driver from path.
        ValueError: GDAL driver cannot write files.
    """
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
        raise ValueError(f"GDAL driver {driver.ShortName} cannot write files")
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


def boolean_split(
    a: np.ndarray,
    mask: np.ndarray,
    axis: int = 0,
    circular: bool = False,
    include: str = "all",
) -> List[np.ndarray]:
    """
    Split array by a boolean mask.

    Arguments:
        a: Array to split.
        mask: Boolean array with same length as `a` along `axis`.
        axis: Axis along which to split.
        circular: Whether to treat `a` as a closed loop (a[-1] -> a[0]).
        include: Whether to return 'all' groups or only 'true' or 'false' groups.

    Returns:
        List of subsets of `a` with contiguous `True` or `False` values in `mask`.

    Examples:
        >>> a = np.array([0, 1, 2, 3, 4])
        >>> mask = np.array([True, True, False, False, True])
        >>> boolean_split(a, mask)
        [array([0, 1]), array([2, 3]), array([4])]
        >>> boolean_split(a, mask, circular=True)
        [array([4, 0, 1]), array([2, 3])]
        >>> boolean_split(a, mask, circular=True, include="true")
        [array([4, 0, 1])]
        >>> boolean_split(a.reshape(-1, 1), mask)
        [array([[0],
                [1]]),
         array([[2],
                [3]]),
         array([[4]])]
    """
    # See https://stackoverflow.com/a/36518315/8161503
    cuts = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    splits = np.split(a, cuts, axis=axis)
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


def in_box(points: np.ndarray, box: Iterable) -> np.ndarray:
    """
    Tests whether points are in (or on) a box.

    Arguments:
        points: Point coordinates (npts, ndim).
        box: Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, ).

    Returns:
        Boolean mask of points in or on the box (npts, ).

    Examples:
        >>> points = np.array([(0, 0), (1, 1), (2, 2), (3, 3)])
        >>> in_box(points, box=[1, 1, 2.5, 2.5])
        array([False,  True,  True, False])
    """
    box = unravel_box(box)
    return np.all((points >= box[0, :]) & (points <= box[1, :]), axis=1)


def clip_polyline_box(
    line: np.ndarray, box: Iterable, t: bool = False
) -> List[np.ndarray]:
    """
    Return segments of a line within a box.

    Vertices are inserted as needed on the box boundary.
    For speed, intersections between two consecutive line vertices are not checked.

    Arguments:
        line: Coordinates of 2 or 3-d line vertices with optional distance measures,
            linearly interpolated at splits [[x, y(, z(, m))], ...].
        box: Minimun and maximum bounds [xmin, ymin(, zmin), xmax, ymax(, zmax)].
        t: Whether last column of `line` are optional distance measures.

    Returns:

    Examples:
        >>> line = np.array([(0, 0), (1, 1), (3, 3)])
        >>> box = 0.5, 0.5, 1.5, 1.5
        >>> clip_polyline_box(line, box)
        [array([[0.5, 0.5],
                [1. , 1. ],
                [1.5, 1.5]])]

        For speed, intersections between two consecutive line vertices are not checked.

        >>> line = np.array([(0, 0), (10, 10)])
        >>> box = 4, 4, 6, 6
        >>> clip_polyline_box(line, box)
        []
    """
    cols = slice(None, -1) if t else slice(None)
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
                segments[i] = np.row_stack((origin + ti * distance, segments[i]))
        if i < nsegments - 1:
            origin = segments[i][-1, :]
            distance = segments[i + 1][0, :] - origin
            ti = intersect_edge_box(origin[cols], distance[cols], box)
            if ti is not None:
                segments[i] = np.row_stack((segments[i], origin + ti * distance))
    return segments[trues]


def intersect_edge_box(
    origin: Iterable, distance: Iterable, box: Iterable
) -> Optional[float]:
    """
    Return intersection of edge with box.

    Arguments:
        origin: Coordinates of 2 or 3-d point [x, y(, z)].
        distance: Distance to end point [dx, dy(, dz)].
        box: Box min and max vertices [xmin, ymin(, zmin), xmax, ymax(, zmax)].

    Returns:
        Multiple of `distance` where edge intersects box, or `None` for no intersection.

    Examples:
        >>> origin = 0, 0
        >>> box = 1, -1, 2, 2
        >>> intersect_edge_box(origin, (1, 1), box) is None
        True
        >>> intersect_edge_box(origin, (2, 2), box)
        0.5
    """
    distance = np.asarray(distance).reshape(1, -1)
    t = np.nanmin(intersect_rays_box(origin, distance, box, t=True))
    if t > 0 and t < 1:
        return t
    return None


def intersect_rays_box(
    origin: Iterable, directions: np.ndarray, box: Iterable, t: bool = False
) -> Tuple[np.ndarray]:
    """
    Return intersections of rays with a(n axis-aligned) box.

    Works in both 2 and 3 dimensions. Vectorized version of algorithm by Williams et al.
    (2011) optimized for rays with a common origin. Also inspired by
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection

    Arguments:
        origin: Common origin of rays [x, y(, z)].
        directions: Directions of rays [[dx, dy(, dz)], ...].
        box: Box min and max vertices [xmin, ymin(, zmin), xmax, ymax(, zmax)].
        t: Whether to return relative (instead of absolute) coordinates of intersection.

    Returns:
        Ray box entrances (`nan` if a miss or `origin` inside `box`)
        and exits (`nan` if a miss) as either absolute coordinates, or if `t` is `True`,
        multipliers of ray `directions`.

    Examples:
        >>> origin = 0, 0
        >>> directions = np.array([(1, 0), (1, 1)])
        >>> box = 1, -1, 2, 2
        >>> intersect_rays_box(origin, directions, box, t=True)
        (array([[1.],
                [1.]]),
         array([[2.],
                [2.]]))
        >>> intersect_rays_box(origin, directions, box)
        (array([[1., 0.],
                [1., 1.]]),
         array([[2., 0.],
                [2., 2.]]))
    """
    # Precompute constants
    bounds = np.repeat(np.atleast_2d(box), len(directions), axis=0)
    fbounds = bounds.flatten()
    with np.errstate(divide="ignore"):
        invdir = 1 / directions
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
def bresenham_line(start: Iterable[int], end: Iterable[int]) -> np.ndarray:
    """
    Return grid indices along a line between two grid indices.

    Uses Bresenham's run-length algorithm. Not all intersected grid cells are returned,
    only those with centers closest to the line. Code modified for speed from
    http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm.

    Arguments:
        start: Start position (xi, yi).
        end: End position (xi, yi).

    Returns:
        Grid indices [(xi, yi), ...].

    Examples:
        >>> bresenham_line((0, 0), (2, 0))
        array([[0, 0],
               [1, 0],
               [2, 0]])
        >>> bresenham_line((0, 0), (0, 2))
        array([[0, 0],
               [0, 1],
               [0, 2]])
        >>> bresenham_line((0, 0), (2, 2))
        array([[0, 0],
               [1, 1],
               [2, 2]])
        >>> bresenham_line((0, 0), (2, 1))
        array([[0, 0],
               [1, 0],
               [2, 1]])
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
    error = int(dx / 2)
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


def bresenham_circle(center: Iterable, radius: float) -> np.ndarray:
    """
    Return grid indices along a circular path.

    Uses Bresenham's circle algorithm.
    Code modified from https://en.wikipedia.org/wiki/Midpoint_circle_algorithm.

    Arguments:
        center: Circle center (x, y).
        radius: Circle radius in pixels.

    Returns:
        Grid indices [(xi, yi), ...].

    Examples:
        >>> bresenham_circle((0, 0), 1)
        array([[ 0.,  1.],
               [ 1.,  1.],
               [ 1.,  0.],
               [ 1., -1.],
               [ 0., -1.],
               [-1., -1.],
               [-1.,  0.],
               [-1.,  1.],
               [ 0.,  1.]])
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
    # Remove duplicate points
    unique = [True] + (np.diff(xy, axis=0).sum(axis=1) != 0).tolist()
    return xy[unique]


def intersect_boxes(boxes: Iterable[Iterable]) -> np.ndarray:
    """
    Return intersection of boxes.

    Arguments:
        boxes: Boxes, each in the format (xmin, ..., xmax, ...).

    Returns:
        Box of the intersection of all boxes (xmin, ..., xmax, ...).

    Raises:
        ValueError: Box lengths are not divisible by 2.
        ValueError: Boxes do not intersect.

    Examples:
        >>> boxes = (0, 0, 10, 10), (5, 5, 15, 15)
        >>> intersect_boxes(boxes)
        array([ 5,  5, 10, 10])
    """
    boxes = np.asarray(boxes)
    if boxes.shape[1] % 2 != 0:
        raise ValueError("Box lengths are not divisible by 2")
    ndim = boxes.shape[1] // 2
    boxmin = np.nanmax(boxes[:, 0:ndim], axis=0)
    boxmax = np.nanmin(boxes[:, ndim:], axis=0)
    if any(boxmax - boxmin <= 0):
        raise ValueError("Boxes do not intersect")
    return np.hstack((boxmin, boxmax))


def pairwise_distance(x: Iterable, y: Iterable, **kwargs: Any) -> np.ndarray:
    """
    Return the pairwise distance between two sets of points.

    Arguments:
        x: First set of n-d points.
        y: Second set of n-d points.
        **kwargs (dict): Optional arguments to :func:`scipy.spatial.distance.cdist`.

    Returns:
        array: Pairwise distances, where [i, j] = distance(x[i], y[j]).

    Examples:
        >>> x = [(0, 0), (1, 1), (2, 2)]
        >>> y = [(0, 1), (1, 2)]
        >>> pairwise_distance(x, y, metric='sqeuclidean')
        array([[1., 5.],
               [1., 1.],
               [5., 1.]])
    """
    x, y = np.asarray(x), np.asarray(y)
    return scipy.spatial.distance.cdist(
        x if x.ndim > 1 else x.reshape(-1, 1),
        y if y.ndim > 1 else y.reshape(-1, 1),
        **kwargs,
    )


def interpolate_line(
    vertices: np.ndarray,
    x: Iterable = None,
    xi: Iterable = None,
    n: int = None,
    dx: float = None,
    error: bool = True,
    fill: Any = "endpoints",
) -> np.ndarray:
    """
    Return points at the specified distances along a line.

    Arguments:
        vertices: Coordinates of line vertices [(x, ...), ...].
        x: Distance measure at each vertex (n, ).
            If `None`, the cumulative Euclidean distance is used.
            Undefined behavior results if not strictly monotonic.
        xi: Distance of interpolated points along line.
            Takes precedence over `n` and `dx`.
        n: Number of evenly-spaced points to return.
            Takes precedence over `dx`.
        dx: Nominal distance between evenly-spaced points.
        error: Whether to raise an error if any `xi` are outside the range of `x`.
        fill: Value(s) to use for `xi` beyond `x[0]` and `xi` beyond `x[-1]`.
            If 'endpoints', uses (`vertices[0]`, `vertices[-1]`).

    Returns:
        Coordinates of interpolated points [(x, ...), ...].

    Raises:
        ValueError: One of xi, n, or dx is required.
        ValueError: Requested distance outside range (if `error` is `True`).

    Examples:
        >>> line = np.array([(0, 0), (1, 0), (1, 1)])
        >>> interpolate_line(line, xi=(1.5, 2))
        array([[1. , 0.5],
               [1. , 1. ]])
        >>> interpolate_line(line, n=2)
        array([[0., 0.],
               [1., 1.]])
        >>> interpolate_line(line, dx=1)
        array([[0., 0.],
               [1., 0.],
               [1., 1.]])
        >>> interpolate_line(line, xi=(-1, 3), error=False)
        array([[0., 0.],
               [1., 1.]])
    """
    if all((xi is None, n is None, dx is None)):
        raise ValueError("One of xi, n, or dx is required")
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


def unravel_box(box: Iterable) -> np.ndarray:
    """
    Return a box in unravelled format.

    Arguments:
        box: Box (xmin, ..., xmax, ...).

    Returns:
        Box as 2-row array [(xmin, ...), (xmax, ...)].

    Raises:
        ValueError: Box length is not divisible by 2.

    Examples:
        >>> box = 1, 2, 10, 20
        >>> unravel_box(box)
        array([[ 1,  2],
               [10, 20]])
    """
    box = np.asarray(box)
    if box.size % 2 != 0:
        raise ValueError("Box length is not divisible by 2")
    return box.reshape(-1, box.size // 2)


def bounding_box(points: Iterable[Iterable]) -> np.ndarray:
    """
    Return bounding box of points.

    Arguments:
        points: Point coordinates [(x, ...), ...].

    Returns:
        Bounding box [xmin, ..., xmax, ...].

    Examples:
        >>> points = [(0, 0), (0, 1), (1, 10)]
        >>> bounding_box(points)
        array([ 0,  0,  1, 10])
    """
    points = np.asarray(points)
    return np.hstack((np.min(points, axis=0), np.max(points, axis=0)))


def box_to_polygon(box: Iterable) -> np.ndarray:
    """
    Return box as polygon.

    Arguments:
        box: 2-dimensional box (xmin, ymin, xmax, ymax).

    Returns:
        Polygon vertices (5, 2).

    Examples:
        >>> box = 0, 0, 1, 1
        >>> box_to_polygon(box)
        array([[0, 0],
               [0, 1],
               [1, 1],
               [1, 0],
               [0, 0]])
    """
    box = unravel_box(box)
    return np.column_stack((box[(0, 0, 1, 1, 0), 0], box[(0, 1, 1, 0, 0), 1]))


def box_to_grid(
    box: Iterable,
    step: Union[float, Iterable[float]],
    snap: Iterable = None,
    mode: str = "grids",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Return grid of points inside box.

    Arguments:
        box: Box (xmin, ..., xmax, ...).
        step: Grid spacing for all (float) or each (iterable) dimension.
        snap: Point to align grid to (need not be inside box).
            If `None`, the `box` minimum is used
        mode: Return format.

            - 'vectors': x (nx, ) and y (ny, ) coordinates.
            - 'grids': x (ny, nx) and y (ny, nx) coordinates.
            - 'points': x, y coordinates (ny * nx, [x, y]).

    Returns:
        Either vectors or grids for each dimension, or point coordinates.

    Raises:
        ValueError: Unsupported mode.

    Examples:
        >>> box = 0, 0, 10, 10
        >>> box_to_grid(box, step=4)
        (array([[0., 4., 8.],
                [0., 4., 8.],
                [0., 4., 8.]]),
         array([[0., 0., 0.],
                [4., 4., 4.],
                [8., 8., 8.]]))
        >>> box_to_grid(box, step=4, mode='points')
        array([[0., 0.],
               [4., 0.],
               [8., 0.],
               [0., 4.],
               [4., 4.],
               [8., 4.],
               [0., 8.],
               [4., 8.],
               [8., 8.]])
        >>> box_to_grid(box, step=4, mode='vectors')
        (array([0., 4., 8.]), array([0., 4., 8.]))
        >>> box_to_grid(box, step=4, snap=(1, 2), mode='vectors')
        (array([1., 5., 9.]), array([ 2.,  6., 10.]))
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
    grid = tuple(np.meshgrid(*arrays))
    if mode == "grids":
        return grid
    if mode == "points":
        return grid_to_points(grid)
    raise ValueError(f"Unsupported mode: {mode}")


def grid_to_points(grid: Iterable[np.ndarray]) -> np.ndarray:
    """
    Return grid as points.

    Arguments:
        grid: Array of grid coordinates for each dimension (X, ...).

    Returns:
        Point coordinates [(Xi, ...), ...].

    Examples:
        >>> grid = np.array([(1, 2)]), np.array([(10, 20)])
        >>> grid_to_points(grid)
        array([[ 1, 10],
               [ 2, 20]])
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
    squared_distances: Iterable, radius: float = 6.3781e6, refraction: float = 0.13
) -> np.ndarray:
    """
    Return elevation corrections for surface curvature and atmospheric refraction.

    Arguments:
        squared_distances: Squared horizontal distances (n, ).
        radius: Radius of curvature in the same units as `squared_distances`.
            Default is the Earth's equatorial radius in meters.
        refraction: Coefficient of refraction of light.
            Default is an average for standard Earth atmospheric conditions.

    Returns:
        Elevation correction for each distance (n, ).
    """
    # http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?topicname=how_viewshed_works
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/3d-analyst-toolbox/how-line-of-sight-works.htm
    # https://en.wikipedia.org/wiki/Atmospheric_refraction#Terrestrial_refraction
    return (refraction - 1) * squared_distances / (2 * radius)


# ---- Time ----


def pairwise_distance_datetimes(
    x: Iterable[datetime.datetime], y: Iterable[datetime.datetime]
) -> np.ndarray:
    """
    Return the pairwise distances between two sets of datetimes.

    Arguments:
        x: Datetimes (n, ).
        y: Datetimes (m, ).

    Returns:
        Pairwise distances in seconds (n, m), where [i, j] = distance(x[i], y[j]).

    Examples:
        >>> t = [datetime.datetime(2020, 1, 1, 0, 0, sec) for sec in range(5)]
        >>> pairwise_distance_datetimes(t[0:3], t[3:5])
        array([[3., 4.],
               [2., 3.],
               [1., 2.]])
    """
    # Convert datetimes to POSIX timestamps (seconds since 1970-01-01 00:00:00 UTC)
    x = [xi.timestamp() for xi in x]
    y = [yi.timestamp() for yi in y]
    return pairwise_distance(x, y, metric="minkowski", p=1)


def datetime_range(
    start: datetime.datetime, stop: datetime.datetime, step: datetime.timedelta
) -> List[datetime.datetime]:
    """
    Return evenly spaced datetimes within a given interval.

    Arguments:
        start: Start datetime.
        stop: End datetime (inclusive).
        step: Time step.

    Returns:
        Array of evenly spaced datetimes.

    Examples:
        >>> base = 2020, 1, 1, 0, 0
        >>> dt = datetime.timedelta(seconds=1)
        >>> datetime_range(datetime.datetime(*base, 0), datetime.datetime(*base, 2), dt)
        [datetime.datetime(2020, 1, 1, 0, 0),
         datetime.datetime(2020, 1, 1, 0, 0, 1),
         datetime.datetime(2020, 1, 1, 0, 0, 2)]
    """
    max_steps = (stop - start) // step
    return [start + n * step for n in range(max_steps + 1)]


def select_datetimes(
    datetimes: Iterable[datetime.datetime],
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    snap: datetime.timedelta = None,
    maxdt: datetime.timedelta = None,
) -> np.ndarray:
    """
    Select datetimes matching the specified criteria.

    Arguments:
        datetimes: Datetimes in ascending order.
        start: Minimum datetime (inclusive).
        end: Maximum datetime (inclusive).
        snap: Interval (relative to 1970-01-01 00:00:00)
            on which to select nearest `datetimes`, or all if `None`.
        maxdt: Maximum distance from nearest `snap` to select `datetimes`.
            If `None`, defaults to half of `snap`.

    Returns:
        Boolean mask of selected `datetimes`.

    Raises:
        ValueError: Start datetime is after end datetime.

    Examples:
        >>> t = [datetime.datetime(2020, 1, 1, 0, 0, x) for x in (0, 1, 2, 4, 5)]
        >>> select_datetimes(t)
        array([ True,  True,  True,  True,  True])
        >>> select_datetimes(t, start=t[1])
        array([False,  True,  True,  True,  True])
        >>> select_datetimes(t, start=t[1], end=t[1])
        array([False,  True, False, False, False])
        >>> snap = datetime.timedelta(seconds=2)
        >>> select_datetimes(t, snap=snap)
        array([ True, False,  True,  True,  True])
        >>> select_datetimes(t, snap=snap, maxdt=0 * snap)
        array([ True, False,  True,  True,  False])
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
    if start > end:
        raise ValueError("Start datetime is after end datetime")
    if snap:
        origin = datetime.datetime(1970, 1, 1, 0, 0, 0)
        shift = (origin - start) % snap
        start = start + shift
        targets = datetime_range(start, end, step=snap)
        nearest = sorted_nearest(datetimes, targets)
        if maxdt is None:
            maxdt = snap * 0.5
        distances = np.abs(targets - datetimes[nearest])
        nearest = np.unique(nearest[distances <= maxdt])
        temp = np.zeros(datetimes.shape, dtype=bool)
        temp[nearest] = True
        selected &= temp
    return selected


# ---- Internal ----


def _progress_bar(max: int) -> progress.bar.Bar:
    return progress.bar.Bar(
        "",
        fill="#",
        max=max,
        hide_cursor=False,
        suffix="%(percent)3d%% (%(index)d of %(max)d) %(elapsed_td)s",
    )


def _parse_parallel(parallel: Union[int, bool]) -> int:
    if parallel is True:
        n = os.cpu_count()
        if n is None:
            raise NotImplementedError("Cannot determine number of CPUs")
    elif parallel is False:
        n = 0
    else:
        n = parallel
    return n
