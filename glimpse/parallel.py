from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, sharedmem, cv2)
from . import (helpers, optimize, image)

def track(tracker, xy, n, xy_sigma, vxy=(0, 0), vxy_sigma=(0, 0),
    datetimes=None, maxdt=0, tile_size=(15, 15), axy=(0, 0), axy_sigma=(0, 0)):
    """
    Run Tracker.track() in parallel for multiple initial particle states.
    """
    for obs in tracker.observers:
        obs.cache_images()
    def process(xyi):
        tracker.initialize_particles(
            n=n, xy=xyi, xy_sigma=xy_sigma, vxy=vxy, vxy_sigma=vxy_sigma)
        tracker.track(
            datetimes=datetimes, maxdt=maxdt, tile_size=tile_size,
            axy=axy, axy_sigma=axy_sigma)
        return np.vstack(tracker.means), np.dstack(tracker.covariances)
    with sharedmem.MapReduce() as pool:
        return pool.map(process, xy)

def detect_keypoints(arrays, masks=None, paths=None, method='sift', root=True,
    **params):
    """
    Detect keypoints and descriptors in parallel for multiple image arrays.
    """
    if masks is np.ndarray or masks is None:
        masks = (masks, ) * len(arrays)
    def process(array, mask, path=None):
        keypoints = optimize.detect_keypoints(
            array, mask=mask, method=method, root=root, **params)
        if path:
            helpers.write_pickle(keypoints, path)
        else:
            return keypoints
    with sharedmem.MapReduce() as pool:
        if paths:
            return pool.map(process, zip(arrays, masks, paths), star=True)
        else:
            return pool.map(process, zip(arrays, masks), star=True)
