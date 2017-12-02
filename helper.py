import numpy as np
import cPickle
import pyproj
import geojson
import json
import collections
import copy
import dem

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
    for feature in obj['features']:
        coords = np.atleast_2d(feature['geometry']['coordinates'])
        if crs:
            coords = sp_transform(coords, 4326, crs)
        feature['geometry']['coordinates'] = coords
    if key:
        obj['features'] = dict((feature['properties'][key], feature) for feature in obj['features'])
    return obj

def write_geojson(obj, path=None, crs=None, **kwargs):
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
    obj = copy.deepcopy(obj)
    if isinstance(obj['features'], dict):
        obj['features'] = obj['features'].values()
    for feature in obj['features']:
        coords = np.atleast_2d(feature['geometry']['coordinates'])
        if crs:
            coords = sp_transform(coords, crs, 4326)
        if len(coords) < 2:
            coords = coords.reshape(-1)
        feature['geometry']['coordinates'] = coords.tolist()
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
    for feature in obj['features']:
        xy = np.atleast_2d(feature['geometry']['coordinates'])[:, 0:2]
        if isinstance(elevation, str):
            z = np.repeat(feature['properties'][elevation], len(xy))
        elif isinstance(elevation, (int, float)):
            z = np.repeat(elevation, len(xy))
        elif isinstance(elevation, dem.DEM):
            z = elevation.sample(xy)
        xyz = np.column_stack((xy, z))
        feature['geometry']['coordinates'] = xyz

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
