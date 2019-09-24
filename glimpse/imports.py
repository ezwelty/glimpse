# ---- Standard ----

import collections
import copy
try:
    import copyreg
except ImportError:
    # Python 2
    import copy_reg as copyreg
try:
    # Python 2
    import cPickle as pickle
except ImportError:
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

# ---- Required ----

import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import scipy.spatial

# ---- Optional ----

import cv2
import lmfit
import lxml.etree
import lxml.builder
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
import osgeo.gdal
import osgeo.gdal_array
import osgeo.osr
