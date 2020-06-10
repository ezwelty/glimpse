# ---- Standard ----

import collections
import copy
import copyreg
import pickle
import datetime
import gzip
import json
import math
import numbers
import os
import re
import shutil
import sys
import time
import traceback
import warnings

warnings.formatwarning = lambda msg, *args, **kwargs: f"[warning] {msg}\n"
import xml.etree.ElementTree
import inspect

# ---- Required ----

import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import scipy.spatial

# ---- Optional ----

import cv2
import lmfit
import matplotlib
import matplotlib.animation
import pandas
import piexif
import PIL.Image
import PIL.ImageDraw
import progress.bar
import pyproj
import shapely
import shapely.geometry
import shapely.ops
import sharedmem
import sklearn.decomposition

# NOTE: Import shapely before gdal/osgeo/ogr
# https://github.com/Toblerity/Shapely/issues/260#issue-65012660
try:
    import osgeo.gdal
    import osgeo.gdal_array
    import osgeo.osr
except ImportError:
    warnings.warn("Module osgeo not found: Reading and writing rasters disabled")
    osgeo = None

# ---- Decorators ----

_imports = locals()


def require(modules):
    if isinstance(modules, str):
        modules = (modules,)

    def decorator(f):
        def wrapped(*args, **kwargs):
            for module in modules:
                if module in _imports and _imports[module] is None:
                    raise ImportError(
                        "Missing module " + module + " required for " + f.__name__
                    )
            return f(*args, **kwargs)

        return wrapped

    return decorator
