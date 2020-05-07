from .imports import (np, lxml, re, warnings)
from . import (helpers)

def read(path, **kwargs):
    obj = helpers.read_json(path, **kwargs)
    apply_geojson_coords(obj, np.atleast_2d)
    if crs:
        apply_geojson_coords(obj, sp_transform, current=4326, target=crs)
    if key:
        obj['features'] = dict((feature['properties'][key], feature) for feature in obj['features'])
    return obj

def items(obj):
    """
    Return all Geometries or Features.
    """
    is_dict = isinstance(obj, dict)
    if is_dict:
        if _is_item(obj):
            # NOTE: Return single item as list of items (?)
            return [obj]
        elif 'features' in obj:
            return obj['features']
        elif 'geometries' in obj:
            return obj['geometries']
    if np.iterable(obj) and _is_item(next(_itervalues(obj))):
        return obj
    else:
        raise ValueError('Cannot parse items from object')

def geometries(obj):
    """
    Return all Geometries.
    """
    return [_geometry(item) for item in _itervalues(items(obj))]

def coordinates(obj):
    """
    Return all Geometry coordinates.
    """
    return [_geometry(item)['coordinates'] for item in _itervalues(items(obj))]

def properties(obj):
    """
    Return all Feature properties
    """
    return [item['properties'] for item in items(obj)]

# ---- Helpers ----

def _is_item(obj):
    return isinstance(obj, dict) and ('geometry' in obj or 'coordinates' in obj)

def _itervalues(obj):
    """
    Return iterator for the values of an iterable.
    """
    if isinstance(obj, dict):
        obj = obj.values()
    return iter(obj)

def _geometry(item):
    """
    Return item Geometry.
    """
    if 'geometry' in item:
        return item['geometry']
    else:
        return item

def _depth(geometry):
    """
    Return Geometry coordinate depth.
    """
    # depths = dict(
    #     Point=0,
    #     MultiPoint=1,
    #     LineString=1,
    #     MultiLineString=2,
    #     Polygon=2,
    #     MultiPolygon=3)
    coords = geometry['coordinates']
    n = -1
    while np.iterable(coords):
        coords = coords[0]
        n += 1
    return n

def _ddepth(geometry, depth=None):
    """
    Return depth distance from Geometry coordinate depth.
    """
    geometry_depth = _depth(geometry)
    if depth is None:
        depth = geometry_depth
    elif depth < 0:
        raise ValueError('depth cannot be negative')
    return geometry_depth - depth

# def _iter_coordinates(geometry, depth=None, pad=False):
#     n = _ddepth(geometry, depth=depth)
#     # Define recursion for n > 0
#     def iter_level(x, levels):
#         level = level - 1
#         for xi in x:
#             if level:
#                 iter_level(xi, level)
#             else:
#                 yield xi
#     # Return coordinates to requested depth
#     coordinates = geometry['coordinates']
#     if n < 0:
#         # Pad to higher depth
#         if pad:
#             for _ in range(abs(n)):
#                 coordinates = [coordinates]
#     if n <= 0:
#         # Return coordinates
#         yield coordinates
#     else:
#         iter_level(coordinates, n)

def coordinates_apply(obj, fun, depth=None, pad=False, **kwargs):
    # Define recursion for level >= 0
    def apply_level(x, level):
        if level:
            return [apply_level(xi, level - 1) for xi in x]
        else:
            return fun(x, **kwargs)
    # Define apply for each geometry
    def apply(geometry):
        n = _ddepth(geometry, depth=depth)
        # Return coordinates to requested depth
        coords = geometry['coordinates']
        if n < 0:
            if pad:
                # Pad to higher depth
                for _ in range(abs(n)):
                    coords = [coords]
        if n <= 0:
            # Apply function
            return fun(x, **kwargs)
        else:
            return apply_level(coords, level=n)
    # Apply to all coordinates
    for geometry in geometries(obj):
        geometry['coordinates'] = apply(geometry)

# def _is_item(obj, error=False):
#     valid = isinstance(obj, dict) and ('geometry' in obj or 'coordinates' in obj)
#     if not valid and error:
#         raise ValueError('Object is not a Feature or Geometry')
#     return valid
#
# def _is_collection(obj, error=False):
#     valid = isinstance(obj, dict) and ('features' in obj or 'geometries' or obj)
#     if not valid and error:
#         raise ValueError('Object is not a GeometryCollection or FeatureCollection')
#     return valid

# def items(obj):
#     if _is_item(obj):
#         return [obj]
#     elif _is_collection(obj):
#         return _get_collection_items(obj)
#     elif np.iterable(obj):
#         return obj
#     else:
#         raise ValueError('Cannot parse items from object')

# def _get_collection_items(collection):
#     if 'features' in collection:
#         return collection['features']
#     else:
#         return collection['geometries']

# def _get_items(obj):
#     if _is_item(obj):
#         return [obj]
#     elif _is_collection(obj):
#         return _get_collection_items(obj)
#     elif np.iterable(obj):
#         return obj
#     else:
#         raise ValueError('Cannot parse items from object')
