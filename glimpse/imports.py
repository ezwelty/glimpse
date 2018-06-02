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
import cv2
import datetime
# NOTE: Import shapely before gdal/osgeo/ogr
# https://github.com/Toblerity/Shapely/issues/260#issue-65012660
import shapely
import shapely.geometry
import shapely.ops
import gdal
import gzip
import json
import lmfit
import lxml.etree
import lxml.builder
import math
import matplotlib
import matplotlib.animation
import numpy as np
import os
import pandas
import piexif
import PIL.Image
import PIL.ImageDraw
import pyproj
import re
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import scipy.spatial
import sharedmem
import shutil
import sklearn.decomposition
import sys
import time
import warnings
