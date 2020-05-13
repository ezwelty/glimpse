from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (require,
    np, pickle, pyproj, json, collections, copy, pandas, scipy, gzip, PIL,
    sklearn, cv2, copyreg, os, re, datetime, matplotlib, shapely, sharedmem,
    sys, progress, osgeo)

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

def format_list(x, length=None, default=None, dtype=None):
    """
    Return object as a formatted list.

    Arguments:
        x: Object to format
        length (int): Output object length.
            If `None`, length of `x` is unchanged
        default (scalar): Default element value.
            If `None`, `x` is repeated to achieve length `length`.
        dtype (callable): Data type to coerce list elements to.
            If `None`, data is left as-is.
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
                assert length % nx == 0
                x *= length // nx
    if dtype:
        x = [dtype(i) for i in x]
    return x

def make_path_directories(path, is_file=True):
    """
    Make directories as needed to build a path.

    Arguments:
        path (str): Directory or file path
        is_file (bool): Whether `path` is a file, in which case
            `path` is reduced to `os.path.dirname(path)`
    """
    # https://stackoverflow.com/a/14364249
    if is_file:
        path = os.path.dirname(path)
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

def numpy_dtype(obj):
    """
    Return numpy data type.

    Arguments:
        obj: Either `numpy.ndarray`, `numpy.dtype`, `type`, or `str`
    """
    if isinstance(obj, np.ndarray):
        return obj.dtype
    else:
        return np.dtype(obj)

def numpy_dtype_minmax(dtype):
    """
    Return min, max allowable values for a numpy datatype.

    Arguments:
        dtype: Either `numpy.ndarray`, `numpy.dtype`, `type`, or `str`

    Returns:
        tuple: Minimum and maximum values
    """
    dtype = numpy_dtype(dtype)
    if issubclass(dtype.type, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif issubclass(dtype.type, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif dtype.type in (np.bool_, np.bool):
        return False, True
    else:
        raise ValueError('Cannot determine min, max for ' + str(dtype))

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
        return basename[::-1].split('.', maxsplit=extensions)[-1][::-1]
    else:
        return basename

def first_not(*args, value=None, default=None):
    """
    Return first object which is not a particular object.

    Arguments:
        *args: Objects to evaluate
        value: Object which `*args` should not be
        default: Object to return if all `*args` are `value`
    """
    return next((xi for xi in args if xi is not value), default)

def as_array(a, dtype=None):
    """
    Return object as array.

    Equivalent to `np.asarray()` but faster if already an array or already `dtype`.

    Arguments:
        a (array-like): Input data
        dtype (data-type): If `None`, inferred from `a`
    """
    if isinstance(a, np.ndarray):
        if dtype is None or numpy_dtype(dtype) is a.dtype.type:
            return a
        else:
            return a.astype(dtype)
    else:
        return np.asarray(a, dtype=dtype)

def diag_indices(a, k=0):
    """
    Return the indices of diagonals in an array.

    Arguments:
        a (array): Array
        k: kth diagonal (int) or diagonals (iterable)
    """
    # https://stackoverflow.com/a/18081653
    assert all(np.array(a.shape) == a.shape[0])
    rows, cols = np.diag_indices_from(a)
    if not np.iterable(k):
        k = k,
    def diag(ki):
        if ki < 0:
            return rows[-ki:], cols[:ki]
        elif ki > 0:
            return rows[:-ki], cols[ki:]
        else:
            return rows, cols
    rowcols = [diag(ki) for ki in k]
    return np.hstack([rc[0] for rc in rowcols]), np.hstack([rc[1] for rc in rowcols])

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

def tile_axis(a, shape, axis=None):
    """
    Construct an array by repeating an input array.

    Arguments:
        a (array-like): Input array
        shape (iterable): Output array shape
        axis: Axis (int) or axes (iterable) of output array along which to
            repeat `a`
    """
    a = np.atleast_1d(a)
    shape = np.asarray(shape)
    if axis is None:
        assert a.size == 1
    elif isinstance(axis, int):
        axis = np.atleast_1d(axis)
    else:
        axis = np.sort(np.asarray(axis))
    if axis is None:
        reps = shape
    else:
        reps = np.ones(len(shape), dtype=int)
        reps[axis] = shape[axis]
        for ax in axis:
            a = np.expand_dims(a, axis=ax)
    result = np.tile(a, reps=reps)
    assert np.all(result.shape == shape)
    return result

def weighted_nanmean(a, weights, axis=None):
    """
    Return the weighted mean of non-missing values in an array.

    Arguments:
        a (array-like): Input array
        weights (array-like): Weights with same shape as `a`
        axis: Axis (int) or axes (iterable) along which to compute mean
    """
    a = np.asarray(a)
    weights = np.asarray(weights)
    return (np.nansum(a * weights, axis=axis) /
        np.nansum(weights * ~np.isnan(a), axis=axis))

def weighted_nanstd(a, weights, axis=None, means=None):
    """
    Return the weighted standard deviation of non-missing values in an array.

    Arguments:
        a (array-like): Input array
        weights (array-like): Weights with same shape as `a`
        axis: Axis (int) or axes (iterable) along which to compute standard
            deviation
        means (array-like): Means computed over the same `axis`.
            If `None`, computed with `weighted_nanmean()`.
    """
    a = np.asarray(a)
    weights = np.asarray(weights)
    if means is None:
        means = weighted_nanmean(a, weights=weights, axis=axis)
    means = tile_axis(means, shape=a.shape, axis=axis)
    std = np.sqrt(weighted_nanmean(
        (a - means)**2, weights=weights, axis=axis))
    return std

def hypot_sigma(x, y):
    """
    Returns hypotenuse means and standard deviations.

    Arguments:
        x (iterable): Means and standard deviations in x
        y (iterable): Means and standard deviations in y
    """
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas
    z = np.hypot(x[0], y[0])
    z_sigma = np.sqrt(
        (x[0] / z)**2 * x[1]**2 +
        (y[0] / z)**2 * y[1]**2)
    return z, z_sigma

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
    make_path_directories(path, is_file=True)
    mode = 'wb' if binary else 'w'
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
        **kwargs: Arguments to `pickle.load()`
    """
    mode = 'rb' if binary else 'r'
    if gz:
        fp = gzip.open(path, mode=mode)
    else:
        fp = open(path, mode=mode)
    obj = pickle.load(fp, **kwargs)
    fp.close()
    return obj

# Register Pickle method for cv2.KeyPoint
# https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
def _pickle_cv2_keypoints(k):
    return cv2.KeyPoint, (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_cv2_keypoints)

# ---- JSON ---- #

def read_json(path, **kwargs):
    """
    Read JSON from file.

    Arguments:
        path (str): Path to file
        **kwargs: Additional arguments passed to `json.load()`
    """
    with open(path, mode='r') as fp:
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
        with open(path, mode='w') as fp:
            fp.write(txt)
        return None
    else:
        return txt

# ---- Arrays: General ---- #

def normalize(array):
    """
    Normalize a numeric array to mean 0, variance 1.

    Arguments:
        array (array): Input array
    """
    return (array - array.mean()) * (1 / array.std())

def normalize_range(array, interval=(0, 1)):
    """
    Translate and scale a numeric array to a specified interval.

    Arguments:
        array (array): Input array
        interval: Interval as either (min, max) or a numpy data type.

    Returns:
        array: Normalized array
    """
    if np.iterable(interval):
        interval = min(interval), max(interval)
    else:
        interval = numpy_dtype_minmax(interval)
    scale = (interval[1] - interval[0]) / (np.nanmax(array) - np.nanmin(array))
    scaled = array * scale
    return scaled - np.nanmin(scaled) + interval[0]

def gaussian_filter(array, mask=None, fill=False, **kwargs):
    """
    Return a gaussian-filtered array.

    Excludes cells by the method described in https://stackoverflow.com/a/36307291.

    Arguments:
        array (array): Array to filter
        mask (array): Boolean mask of cells to include (True) or exclude (False).
            If `None`, all cells are included.
        fill (bool): Whether to fill cells excluded by `mask` with interpolated values
        **kwargs (dict): Additional arguments to `scipy.ndimage.filters.gaussian_filter()`
    """
    if mask is None:
        return scipy.ndimage.filters.gaussian_filter(array, **kwargs)
    else:
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
        **kwargs (dict): Additional arguments to `scipy.ndimage.filters.maximum_filter()`
    """
    if mask is None:
        return scipy.ndimage.filters.maximum_filter(array, **kwargs)
    else:
        dtype_min = numpy_dtype_minmax(array)[0]
        x = array.copy()
        mask = ~mask
        x[mask] = dtype_min
        x = scipy.ndimage.filters.maximum_filter(x, **kwargs)
        if fill:
            mask = x == dtype_min
        x[mask] = array[mask]
        return x

def median_filter(array, mask=None, fill=False, **kwargs):
    """
    Return a median-filtered array.

    Excludes cells by setting them to `np.nan` and using
    `scipy.ndimage.filters.generic_filter` with `np.nanmedian`. This is much
    slower than `scipy.ndimage.filters.median_filter`.

    Arguments:
        array (array): Array to filter
        mask (array): Boolean mask of cells to include (True) or exclude (False).
            If `None`, all cells are included.
        fill (bool): Whether to fill cells excluded by `mask` with interpolated values
        **kwargs (dict): Additional arguments to either
            `scipy.ndimage.filters.median_filter` (`mask` is False) or
            `scipy.ndimage.filters.generic_filter()`
    """
    if mask is None:
        return scipy.ndimage.filters.median_filter(array, **kwargs)
    else:
        x = array.copy()
        mask = ~mask
        x[mask] = np.nan
        x = scipy.ndimage.filters.generic_filter(x, function=np.nanmedian, **kwargs)
        if not fill:
            x[mask] = array[mask]
        return x

# ---- Arrays: Images ---- #

# NOTE: Unused
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

# NOTE: Unused
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
    if method == 'average':
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

def match_histogram(source, template):
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

@require('osgeo')
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
        if re.findall(r'\[', crs):
            return crs
        elif re.findall(r':', crs):
            obj.ImportFromProj4(crs.lower())
        else:
            raise ValueError('crs string format not Proj4 or WKT')
    else:
        raise ValueError('crs must be int (EPSG) or str (Proj4 or WKT)')
    return obj.ExportToWkt()

def sp_transform(points, current, target):
    """
    Transform points between spatial coordinate systems.

    Coordinate systems can be specified either as an EPSG code (int),
    arguments to `pyproj.Proj()` (dict), or `pyproj.Proj`.

    Arguments:
        points (array): Point coordinates [[x, y(, z)]]
        current: Current coordinate system
        target: Target coordinate system
    """
    def build_proj(obj):
        if isinstance(obj, pyproj.Proj):
            return obj
        elif isinstance(obj, int):
            return pyproj.Proj(init='EPSG:' + str(obj))
        elif isinstance(obj, dict):
            return pyproj.Proj(**obj)
        else:
            raise ValueError('Cannot coerce input to pyproj.Proj')
    current = build_proj(current)
    target = build_proj(target)
    if points.shape[1] < 3:
        z = None
    else:
        z = points[:, 2]
    result = pyproj.transform(current, target, x=points[:, 0], y=points[:, 1], z=z)
    return np.column_stack(result)

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

def geojson_iterfeatures(obj):
    """
    Return an iterator over GeoJSON features.
    """
    features = obj['features']
    if isinstance(features, list):
        index = range(len(features))
    else:
        index = features.keys()
    for i in index:
        yield features[i]

def _get_geojson_coords(feature):
    if 'geometry' in feature:
        return feature['geometry']['coordinates']
    else:
        return feature['coordinates']

def _set_geojson_coords(feature, coords):
    if 'geometry' in feature:
        feature['geometry']['coordinates'] = coords
    else:
        feature['coordinates'] = coords

def geojson_itercoords(obj):
    """
    Return an iterator over GeoJSON feature coordinates.
    """
    for feature in geojson_iterfeatures(obj):
        yield _get_geojson_coords(feature)

def apply_geojson_coords(obj, fun, **kwargs):
    """
    Apply a function to all GeoJSON feature coordinates.

    Arguments:
        obj (dict): GeoJSON
        fun (callable): Called as `fun(coordinates, **kwargs)`
        **kwargs (dict): Additional arguments passed to `fun`
    """
    for feature in geojson_iterfeatures(obj):
        coords = _get_geojson_coords(feature)
        ndim = np.ndim(coords)
        # Case 1 (ndim=1): [x, y]
        # Case 2 (ndim=2): [[x, y], [x, y]]
        # Case 3 (ndim=3): [[[x, y], [x, y]], [[x, y], [x, y]]]
        # Case 4 (ndim=3): [[[x, y], [x, y]]]
        if ndim < 3:
            _set_geojson_coords(feature, fun(coords, **kwargs))
        elif ndim == 3:
            _set_geojson_coords(feature, [fun(X, **kwargs) for X in coords])
        else:
            raise ValueError('Unknown coordinates format')

def elevate_geojson(obj, elevation):
    """
    Add or update GeoJSON elevations.

    Arguments:
        obj (dict): Object to modify
        elevation: Elevation, as either
            the elevation of all features (int or float),
            the name of the feature property containing elevation (str), or
            digital elevation model from which to sample elevations (`dem.DEM`)
    """
    def set_z(coords, z):
        if len(z) == 1:
            z = np.repeat(z, len(coords))
        return np.column_stack((coords[:, 0:2], z))
    def set_from_elevation(coords, elevation):
        if isinstance(elevation, (int, float)):
            return set_z(coords, elevation)
        else:
            return set_z(coords, elevation.sample(coords[:, 0:2]))
    if isinstance(elevation, (bytes, str)):
        for feature in geojson_iterfeatures(obj):
            coords = _get_geojson_coords(feature)
            _set_geojson_coords(feature, set_z(coords, feature['properties'][elevation]))
    else:
        apply_geojson_coords(obj, set_from_elevation, elevation=elevation)

def ordered_geojson(obj, properties=None,
    keys=('type', 'properties', 'features', 'geometry', 'coordinates')):
    """
    Return GeoJSON as a nested ordered dictionary.

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

def project_points_plane(points, plane):
    """
    Return projection of points on plane.

    Arguments:
        points (iterable): Point coordinates ((x, y, z), ...)
        plane (iterable): Plane (a, b, c, d), where ax + by + cz + d = 0
    """
    # http://www.9math.com/book/projection-point-plane
    n = np.asarray(plane[0:3])
    d = plane[3]
    return points - n * ((np.dot(points, n) + d) * (1 / sum(n**2)))[:, None]

def intersect_rays_plane(origin, directions, plane):
    """
    Return intersections of rays with a plane.

    Optimized for rays with a common origin.

    Arguments:
        origin (iterable): Common origin of rays (x, y , z)
        directions (array): Directions of rays [[dx, dy, dz], ...]
        plane (iterable): Plane (a, b, c, d), where ax + by + cz + d = 0

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
        origin (iterable): Coordinates of 2 or 3D point (ndim, )
        distance (iterable): Distance to end point (ndim, )
        box (iterable): Minimun and maximum bounds [xmin, ..., xmax, ...] (2 * ndim, )
    """
    distance_2d = as_array(distance).reshape(1, -1)
    t = np.nanmin(intersect_rays_box(origin, distance_2d, box, t=True))
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
        origin (iterable): Common origin of rays [x, y(, z)]
        directions (array): Directions of rays [[dx, dy(, dz)], ...]
        box (iterable): Box min and max vertices [xmin, ymin(, zmin), xmax, ymax(, zmax)]

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

# TODO: Implement faster run-slice (http://www.phatcode.net/res/224/files/html/ch36/36-03.html)
def bresenham_line(start, end):
    """
    Return grid indices along a line between two grid indices.

    Uses Bresenham's run-length algorithm.
    Not all intersected grid cells are returned, only those with centers closest to the line.
    Code modified for speed from http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm.

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

# NOTE: Unused
def inverse_kernel_distance(data, bandwidth=None, function='gaussian'):
    """
    Return spatial weights based on inverse kernel distances.

    Arguments:
        data (array): Observations (n, d)
        bandwidth (float): Kernel bandwidth.
            If `None`, the maximum pairwwise distance between `data`
            observations is used.
        function (str): Kernel function, either 'triangular', 'uniform', 'quadratic', 'quartic', 'gaussian'.
            See http://pysal.readthedocs.io/en/latest/library/weights/Distance.html#pysal.weights.Distance.Kernel.
    """
    nd = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))
    if bandwidth is None:
        bandwidth = np.max(nd)
    nd *= 1 / bandwidth
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

def intersect_ranges(ranges):
    """
    Return intersection of ranges.

    Arguments:
        ranges (iterable): Ranges, each in the format (min, max)
    """
    ranges = np.sort(ranges, axis=1)
    rmin = np.nanmax(ranges[:, 0])
    rmax = np.nanmin(ranges[:, 1])
    if rmax - rmin <= 0:
        raise ValueError('Ranges do not intersect')
    else:
        return np.hstack((rmin, rmax))

def cut_ranges(ranges, cuts):
    ranges = np.sort(ranges, axis=1)
    for x in cuts:
        is_cut = (ranges[:, 0] < x) & (x < ranges[:, 1])
        if np.any(is_cut):
            cut = [[(r[0], x), (x, r[1])] for r in ranges[is_cut]]
            not_cut = ranges[~is_cut]
            ranges = np.vstack((not_cut, np.vstack(cut)))
    order = np.lexsort((ranges[:, 1], ranges[:, 0]))
    return ranges[order]

def cut_out_ranges(ranges, cutouts):
    cutouts = np.reshape(cutouts, (-1, 2))
    ranges = cut_ranges(ranges, cuts=cutouts.ravel())
    for x in cutouts:
        is_cutout = (ranges[:, 0] >= x[0]) & (ranges[:, 1] <= x[1])
        ranges = ranges[~is_cutout]
    return ranges

def intersect_boxes(boxes):
    """
    Return intersection of boxes.

    Arguments:
        boxes (iterable): Boxes, each in the format (minx, ..., maxx, ...)
    """
    boxes = as_array(boxes)
    assert boxes.shape[1] % 2 == 0
    ndim = boxes.shape[1] // 2
    boxmin = np.nanmax(boxes[:, 0:ndim], axis=0)
    boxmax = np.nanmin(boxes[:, ndim:], axis=0)
    if any(boxmax - boxmin <= 0):
        raise ValueError('Boxes do not intersect')
    else:
        return np.hstack((boxmin, boxmax))

def union_boxes(boxes):
    points = np.row_stack([unravel_box(box) for box in boxes])
    return bounding_box(points)

def pairwise_distance(x, y, metric='sqeuclidean', **params):
    """
    Return the pairwise distance between two sets of points.

    Arguments:
        x (iterable): First set of n-d points
        y (iterable): Second set of n-d points
        metric (str): Distance metric.
            See `scipy.spatial.distance.cdist()`.
        **params (dict): Additional arguments to `scipy.spatial.distance.cdist()`

    Returns:
        array: Pairwise distances, where [i, j] = distance(x[i], y[j])
    """
    x = as_array(x)
    y = as_array(y)
    return scipy.spatial.distance.cdist(
        x if x.ndim > 1 else x.reshape(-1, 1),
        y if y.ndim > 1 else y.reshape(-1, 1),
        metric=metric, **params)

def interpolate_line(vertices, x=None, xi=None, n=None, dx=None, error=True, fill='endpoints'):
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
        x = np.cumsum(np.sqrt(np.sum(np.diff(vertices, axis=0)**2, axis=1)))
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
        fill = 'endpoints'
    # x must be increasing
    if len(x) > 1 and x[1] < x[0]:
        sort_index = np.argsort(x)
        x = x[sort_index]
        vertices = vertices[sort_index, :]
    # Interpolate each dimension and combine
    result = np.column_stack((
        np.interp(xi, x, vertices[:, i]) for i in range(vertices.shape[1])))
    if fill == 'endpoints':
        if error is False:
            return result
        else:
            fill = (vertices[0], vertices[-1])
    if not np.iterable(fill):
        fill = (fill, fill)
    left = np.less(xi, x[0])
    right = np.greater(xi, x[-1])
    if x[0] > x[-1]:
        right, left = left, right
    if error and (left.any() or right.any()):
        raise ValueError('Requested distance outside range')
    else:
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
    box = as_array(box)
    assert box.size % 2 == 0
    ndim = box.size // 2
    return box.reshape(-1, ndim)

def bounding_box(points):
    """
    Return bounding box of points.

    Arguments:
        points (iterable): Points, each in the format (x, ...)
    """
    points = as_array(points)
    return np.hstack((
        np.min(points, axis=0),
        np.max(points, axis=0)))

def box_to_polygon(box):
    """
    Return box as polygon.

    Arguments:
        box (iterable): Box (minx, ..., maxx, ...)
    """
    box = unravel_box(box)
    return np.column_stack((
        box[(0, 0, 1, 1, 0), 0],
        box[(0, 1, 1, 0, 0), 1]))

def box_to_grid(box, step, snap=None, mode='grids'):
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
    step = step if np.iterable(step) else (step, ) * ndim
    if snap is None:
        snap = box[0, :]
    shift = (snap - box[0, :]) % step
    n = (np.diff(box, axis=0).ravel() - shift) // step
    arrays = (np.linspace(box[0, i] + shift[i],
        box[0, i] + shift[i] + n[i] * step[i],
        int(n[i]) + 1)
        for i in range(ndim))
    if mode == 'vectors':
        return tuple(arrays)
    else:
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

def points_in_polygon(points, polygon):
    """
    Return whether each point is contained by the polygon.

    Arguments:
        points (iterable): Point coordinates ((x, y), ...)
        polygon (iterable): Polygon vertices ((x, y), ...)
    """
    path = matplotlib.path.Path(polygon)
    return path.contains_points(points)

def polygon_to_grid_points(polygon, holes=None, **params):
    """
    Return grid of points inside polygon.

    Generates a grid of points inside the polygon bounding box, then returns only
    those grid points inside the polygon.

    Arguments:
        polygon (iterable): Polygon vertices
        holes (iterable): Polygons representing holes in `polygon`
        **params (dict): Arguments passed to `box_to_grid()`
    """
    box = bounding_box(polygon)
    grid = box_to_grid(box, mode='grids', **params)
    points = grid_to_points(grid)
    is_in = points_in_polygon(points, polygon)
    if holes:
        for hole in holes:
            is_in &= ~points_in_polygon(points, hole)
    return points[is_in, :]

def side(points, edge):
    """
    Return which side points are relative to an edge.

    Arguments:
        points (array): Point coordinates (n, 2)
        edge (array): Start and end points (2, 2)

    Returns:
        array: Side for each point (-1: left, 0: colinear, 1: right)
    """
    cross = np.cross(edge[0] - edge[1], points - edge[0])
    return np.sign(cross).astype(int)

def cartesian_to_polyline(points, line):
    """
    Convert cartesian coordinates to polyline coordinates.

    Arguments:
        points (array): Cartesian coordinates (n, 2)
        line (array): Cartesian coordinates (m, 2)

    Returns:
        array: Polyline coordinates - distance along line, distance from line (n, 2)
    """
    Line = shapely.geometry.LineString(line)
    # Compute distance along line
    # NOTE: Slow
    M = [Line.project(shapely.geometry.Point(p)) for p in points]
    # Compute signed distance to line (- left, + right)
    x = np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1)))
    x = np.insert(x, 0, 0)
    mi = np.interp(x=M, xp=x, fp=np.arange(len(x)))
    starts = np.floor(mi).astype(int)
    is_last = starts == len(line) - 1
    starts[is_last] = len(line) - 2
    signs = np.zeros(len(points), dtype=int)
    for start in np.unique(starts):
        index = starts == start
        signs[index] = side(points[index, :], line[start:(start + 2), :])
    projected_points = interpolate_line(line, x=x, xi=M)
    # Distances to line
    D = np.linalg.norm(projected_points - points, axis=1)
    return np.column_stack((M, D * signs))

def polyline_to_cartesian(points, line):
    """
    Convert polyline coordinates to cartesian coordinates.

    Arguments:
        points (array): Polyline coordinates (n, 2)
        line (array): Cartesian coordinates (m, 2)

    Returns:
        array: Cartesian coordinates (n, 2)
    """
    projected_points = interpolate_line(line, xi=points[:, 0])
    # Locate start of line segment
    x = np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1)))
    x = np.insert(x, 0, 0)
    mi = np.interp(x=points[:, 0], xp=x, fp=np.arange(len(x)))
    starts = np.floor(mi).astype(int)
    is_last = starts == len(line) - 1
    is_inner_vertex = (mi == starts) & (starts != 0) & ~is_last
    # Last vertex: Segment starts at previous vertex
    starts[is_last] = len(line) - 2
    # Compute directions to points
    dxy = np.diff(line, axis=0)
    directions = np.arctan2(dxy[:, 1], dxy[:, 0])
    new_directions = directions[starts]
    # Inner vertex: Take average of adjacent segment slopes
    new_directions[is_inner_vertex] = (0.5 * (directions[:-1] + directions[1:]))[starts[is_inner_vertex] - 1]
    new_directions -= np.sign(points[:, 1]) * (np.pi * 0.5)
    return projected_points + np.abs(points[:, 1:2]) * np.column_stack((
        np.cos(new_directions), np.sin(new_directions)))

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
    idx = np.ravel_multi_index((groups.row.values, groups.col.values), shape)
    grid = np.full(shape, np.nan)
    grid.flat[idx] = groups.value.values
    return grid

def polygons_to_mask(polygons, size, holes=None):
    """
    Returns a boolean array of cells inside polygons.

    The upper-left corner of the upper-left cell of the array is (0, 0).

    Arguments:
        polygons (iterable): Polygons
        size (iterable): Array size (nx, ny)
        holes (iterable): Polygons representing holes in `polygons`
    """
    im_mask = PIL.Image.new(mode='1', size=(int(size[0]), int(size[1])))
    draw = PIL.ImageDraw.ImageDraw(im_mask)
    for polygon in polygons:
        if isinstance(polygon, np.ndarray):
            polygon = [tuple(row) for row in polygon]
        draw.polygon(polygon, fill=1)
    if holes is None:
        holes = []
    for hole in holes:
        if isinstance(hole, np.ndarray):
            hole = [tuple(row) for row in hole]
        draw.polygon(hole, fill=0)
    return np.array(im_mask)

def elevation_corrections(origin=None, xyz=None, squared_distances=None, earth_radius=6.3781e6, refraction=0.13):
    """
    Return elevation corrections for earth curvature and refraction.

    Arguments:
        origin (iterable): World coordinates of origin (x, y, (z))
        xyz (array): World coordinates of target points (n, 2+)
        squared_distances (iterable): Squared Euclidean distances
            between `origin` and `xyz`. Takes precedence if not `None`.
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

    Datetime wrapper for `pairwise_distance()`.

    Arguments:
        x (iterable): Datetime objects
        y (iterable): Datetime objects

    Returns:
        array: Pairwise distances in seconds, where [i, j] = distance(x[i], y[j])
    """
    return pairwise_distance(
        datetimes_to_float(x), datetimes_to_float(y),
        metric='minkowski', p=1)

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

def interpolate_line_datetimes(vertices, x, xi=None, n=None, dx=None, **kwargs):
    """
    Return points at the specified datetimes along a line.

    Arguments:
        vertices (array): Coordinates of vertices (n, d)
        x (iterable): Datetimes of vertices (n, ).
            Undefined behavior results if not strictly monotonic.
        xi (iterable): Datetimes of interpolated points
        n (int): Number of evenly-spaced points to return
            (ignored if `xi` is not `None`)
        dx (timedelta): Nominal timedelta between evenly-spaced points
            (ignored if `xi` or `n` is not `None`)
        **kwargs (dict): Additional arguments passed to `interpolate_line()`
    """
    t0 = x[0]
    x = np.asarray([(t - t0).total_seconds() for t in x])
    if xi is not None:
        xi = np.asarray([(t - t0).total_seconds() for t in xi])
    if dx is not None:
        dx = dx.total_seconds()
    return interpolate_line(vertices, x=x, xi=xi, n=n, dx=dx, **kwargs)

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
    datetimes = as_array(datetimes)
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

# ---- Velocity analysis ----

def triangle_area(xy):
    """
    Return the area of a triangle.

    Arguments:
        xy (array): Triangle vertices (3, 2)

    Returns:
        float: Triangle area
    """
    return 0.5 * np.linalg.det(np.hstack((np.ones((3, 1)), xy)))

def triangle_centroid(xy):
    """
    Return the centroid of a triangle.

    Arguments:
        xy (array): Triangle vertices (3, 2)

    Returns:
        array: Triangle centroid (x, y)
    """
    return np.mean(xy, axis=0)

def triangle_velocity_strain(xy, vxy):
    """
    Return the centroid velocity and strain of a triangle.

    From Hubner et al. (2001). The Finite Element Method for Engineers.
    4th edition. John Wiley & Sons. ISBN 0-471-37078-9

    Arguments:
        xy (array): Triangle vertices in counter-clockwise order (3, 2)
        vxy (array): Velocity components at `xy` (3, 2)

    Returns:
        array: Centroid velocity (vx, vy)
        array: Strain vector with components

            - epsilon_xx: normal strain in x
            - epsilon_yy: normal strain in y
            - gamma_xy: shear strain
    """
    x, y = xy[:, 0], xy[:, 1]
    idx_1 = [1, 2, 0]
    idx_2 = [2, 0, 1]
    # a1 = x2y3 - x3y2, a2 = x3y1 - x1y3, a3 = x1y2 - x2y1
    a = x[idx_1] * y[idx_2] - x[idx_2] * y[idx_1]
    # b1 = y2 - y3, b2 = y3 - y1, b3 = y1 - y2
    b = y[idx_1] - y[idx_2]
    # c1 = x3 - x2, c2 = x1 - x3, c3 = x2 - x1
    c = x[idx_2] - x[idx_1]
    # Centroid velocity
    u, v = vxy[:, 0], vxy[:, 1]
    centroid = triangle_centroid(xy)
    area = triangle_area(xy)
    velocity = (1 / (2 * area)) * np.array((
        np.sum(a * u) + centroid[0] * np.sum(b * u) + centroid[1] * np.sum(c * u),
        np.sum(a * v) + centroid[0] * np.sum(b * v) + centroid[1] * np.sum(c * v)))
    # Strain
    B = [[b[0], 0, b[1], 0, b[2], 0],
        [0, c[0], 0, c[1], 0, c[2]],
        [c[0], b[0], c[1], b[1], c[2], b[2]]]
    delta = vxy.reshape(-1, 1)
    strain = (1 / (2 * area)) * np.matmul(B, delta)
    return velocity, strain.ravel()

def strain_to_principal_strain(strain):
    """
    Convert strain vector to principal strain.

    Arguments:
        strain (iterable): Strain vector with components

            - epsilon_xx: normal strain in x
            - epsilon_yy: normal strain in y
            - gamma_xy: shear strain

    Returns:
        float: Principal strain along x' (epsilon_max)
        float: Principal strain along y' (epsilon_min)
        float: Rotation of x' axis from x axis (theta_max)
        float: Rotation of y' axis from x axis (theta_min)
    """
    # http://www.continuummechanics.org/principalstressesandstrains.html
    theta = np.arctan2(strain[2], (strain[0] - strain[1])) * 0.5
    cos, sin = np.cos(theta), np.sin(theta)
    Q = np.array([(cos, sin), (-sin, cos)])
    E = np.array([(strain[0], strain[2] * 0.5), (strain[2] * 0.5, strain[1])])
    emax, emin = np.diag(np.matmul(Q, np.matmul(E, Q.T)))
    return emax, emin, theta, theta + np.pi * 0.5

def compute_strain(xy, vxy):
    """
    Return centroids, velocities, and strains of triangular elements.

    Arguments:
        xy (array): Point coordinates (n, 2)
        vxy (array): Velocity components at `xy` (n, 2)

    Returns:
        array: Triangle centroids (m, 2)
        array: Centroid velocities (m, 2)
        array: Strain vectors (m, 3)
    """
    tri = scipy.spatial.Delaunay(xy)
    centroids = [triangle_centroid(xy[indices]) for indices in tri.simplices]
    temp = [triangle_velocity_strain(xy[indices], vxy[indices]) for indices in tri.simplices]
    velocities = [x[0] for x in temp]
    strains = [x[1] for x in temp]
    return np.row_stack(centroids), np.row_stack(velocities), np.row_stack(strains)

def angle_between_vectors(x, y):
    """
    Return the angle between pairs of vectors.

    Arguments:
        x (array-like): Vectors (n vectors, m dimensions)
        y (array-like): Vectors (n vectors, m dimensions)

    Returns:
        array: Angle in radians between each vector pair x[i], y[i] (n, )
    """
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    radians = np.arccos(np.sum(x * y, axis=1) /
        (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)))
    is_nan_angle = np.isnan(radians)
    is_nan_vector = np.any(np.isnan(x), axis=1) | np.any(np.isnan(y), axis=1)
    radians[is_nan_angle & ~is_nan_vector] = 0
    return radians

def normalize_angles(x):
    """
    Return angles normalized between -π and π.
    """
    # https://stackoverflow.com/a/7869457
    return np.mod(x + np.pi, 2 * np.pi) - np.pi

# ---- Uncertainties ----

def take_along_axis(a, indices, axis):
    """
    Return array values matching 1-d indices along axis.

    Arguments:
        a (numpy.ndarray): Source array
        indices: Indices to take along each 1-d slice of **a**
        axis (int): Axis to take slices along
    """
    # https://github.com/numpy/numpy/issues/8708
    if axis < 0:
       if axis >= -a.ndim:
           axis += a.ndim
       else:
           raise IndexError('axis out of range')
    ind_shape = (1,) * indices.ndim
    ins_ndim = indices.ndim - (a.ndim - 1)   # inserted dimensions
    dest_dims = list(range(axis)) + [None] + list(range(axis + ins_ndim, indices.ndim))
    inds = []
    for dim, n in zip(dest_dims, a.shape):
        if dim is None:
            inds.append(indices)
        else:
            ind_shape_dim = ind_shape[:dim] + (-1,) + ind_shape[(dim + 1):]
            inds.append(np.arange(n).reshape(ind_shape_dim))
    return a[tuple(inds)]

def normal_weighted_mean(means, sigmas, axis=None, correlated=False):
    """
    Return inverse-variance weighted mean and standard deviation of
    normal distributions.

    Arguments:
        means (array-like): Means
        sigmas (array-like): Standard deviations
        axis (int): Axis along wich to take the mean
        correlated (bool): Whether variables are correlated
    """
    isnan_mean = np.isnan(means)
    isnan_sigmas = np.isnan(sigmas)
    if (isnan_mean != isnan_sigmas).any():
        raise ValueError('mean and sigma NaNs do not match')
    if correlated:
        # Reorder nan to end
        order = np.argsort(isnan_mean, axis=axis)
        means = take_along_axis(means, order, axis=axis)
        sigmas = take_along_axis(sigmas, order, axis=axis)
    if (sigmas == 0).any():
        raise ValueError('sigmas cannot be 0')
    weights = sigmas**-2
    weights *= np.expand_dims(1 / np.nansum(weights, axis=axis), axis)
    wmeans = np.nansum(weights * means, axis=axis)
    isnan = isnan_mean.all(axis=axis)
    wmeans[isnan] = np.nan
    variances = np.nansum(weights**2 * sigmas**2, axis=axis)
    variances[isnan] = np.nan
    if correlated:
        ab = np.product(weights.take(range(2), axis), axis=axis)
        single = np.isnan(weights.take(range(2), axis)).sum(axis=axis) == 1
        ab[single] = 0
        variances += 2 * np.nansum(weights.take(
            range(2, weights.shape[axis]), axis), axis=axis) + 2 * ab
    return wmeans, np.sqrt(variances)

def circular_normal(x, weights=None, axis=None):
    """
    Return mean and standard deviation of circular quantities.

    Missing values are ignored.

    Arguments:
        x (numpy.ndarray): Angles in radians
        weights (numpy.ndarray): Weights, same shape as **x**
        axis (int): Axis along which the statistics are computed

    Returns:
        numpy.ndarray: Circular means
        numpy.ndarray: Circular standard deviations
    """
    if weights is None:
        unit_yx = (
            np.nanmean(np.sin(x), axis=axis),
            np.nanmean(np.cos(x), axis=axis))
    else:
        unit_yx = (
            weighted_nanmean(np.sin(x), weights=weights, axis=axis),
            weighted_nanmean(np.cos(x), weights=weights, axis=axis))
    return np.arctan2(*unit_yx), np.sqrt(-2 * np.log(np.hypot(*unit_yx)))

# ---- Flotation ----

def ice_flotation_thickness(bed, water, density_ice=916.7, density_water=1025):
    """
    Return the thickness of ice at flotation.

    Arguments:
        bed (float or numpy.ndarray): Bed elevations (m)
        water (float or numpy.ndarray): Water elevations (m).
            Must be scalar or same dimensions as **bed**.
        density_ice (float): Density of ice (kg / m^3)
        density_water (float): Density of water (kg / m^3)
    """
    return (water - bed) * (density_water / density_ice)

def ice_thickness(ice, bed, water, density_ice=916.7, density_water=1025):
    """
    Return the thickness of ice, considering flotation.

    Arguments:
        ice (float or numpy.ndarray): Ice surface elevations (m)
        bed (float or numpy.ndarray): Bed elevations (m).
            Must be scalar or same dimensions as **ice**.
        water (float or numpy.ndarray): Water elevations (m).
            Must be scalar or same dimensions as **ice**.
        density_ice (float): Density of ice (kg / m^3)
        density_water (float): Density of water (kg / m^3)
    """
    hmax = ice - bed
    hf = ice_flotation_thickness(bed, water, density_ice, density_water)
    h = hmax
    floating = hmax < hf
    if np.iterable(water):
        water = water[floating]
    if np.iterable(ice):
        ice = ice[floating]
    floating_h = (ice - water) * (1 + density_ice / (density_water - density_ice))
    if np.iterable(ice):
        h[floating] = floating_h
    elif floating:
        h = floating_h
    return h

def ice_flotation_probability(ice, bed, water, density_ice=916.7, density_water=1025):
    """
    Return the probability of ice flotation.

    Arguments:
        ice (unumpy.uarray): Ice surface elevations (m)
        bed (unumpy.uarray): Bed elevations (m).
            Must be scalar or same dimensions as **ice**.
        water (unumpy.uarray): Water elevations (m).
            Must be scalar or same dimensions as **ice**.
        density_ice (float): Density of ice (kg / m^3)
        density_water (float): Density of water (kg / m^3)
    """
    hmax = ice - bed
    hf = ice_flotation_thickness(bed, water, density_ice, density_water)
    # Probability that hmax < hf
    # https://math.stackexchange.com/a/40236
    dh = hmax - hf
    return scipy.stats.norm().cdf(-dh.mean / dh.sigma)

# ---- Internal ----

def _progress_bar(max):
    return progress.bar.Bar('', fill='#', max=max, hide_cursor=False,
        suffix='%(percent)3d%% (%(index)d of %(max)d) %(elapsed_td)s')

def _parse_parallel(parallel):
    if parallel is True:
        n = os.cpu_count()
        if n is None:
            raise NotImplementedError('Cannot determine number of CPUs')
    elif parallel is False:
        n = 0
    else:
        n = parallel
    return n

def save_observercams(observer,directory,print_path=False):
    """
    Saves each camera model for the respective Image objects in an Observer object as a .JSON file
    
    Arguments:
        observer (glimpse.Observer(): Observer object with Images

        directory (path) : Path to the directory used to save camera models

        print_path (bool) : Print the respective path of each camera model saved
    """
    if not os.path.isdir(directory):
        raise Exception("Directory {} Not Found".format(directory))
    for images in observer.images:
        filename = images.path.split("/")[-1]
        filename = filename.split(".")[0]
        path = os.path.join(directory,filename)
        path += ".JSON"
        if print_path: print("Path: {}\n".format(path))
        try:
            images.cam.write(path)
        except:
            print("Image {} Has Undefined Camera".format(images.path))
def change_extenstion(infile,extension):
    """
    Changes a given file path to an identical file path but with a different extension
    ex: image_01.JPG to image_01.JSON

    Arguments:

        infile (str): Input file path to be changed
        extension (str): output file extension
    
    Return:
        outfile (str) file with new extension
    """
    if not os.isfile(infile):
        rase Exception("File {} Not Found".format(infile))
    token = infile.split(".")[-1] #get old extension
    outfile = infile.split(token)[0] # check for redundent extensions
    outfile += token
    return outfile
