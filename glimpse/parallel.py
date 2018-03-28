from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, sharedmem, cv2)
from . import (helpers, optimize, image)

def track(tracker, xy, n, xy_sigma, vxy=(0, 0), vxy_sigma=(0, 0),
    datetimes=None, maxdt=0, tile_size=(15, 15), axy=(0, 0), axy_sigma=(0, 0),
    processes=None):
    """
    Run Tracker.track() in parallel for multiple initial particle states.
    """
    def process(xyi):
        tracker.initialize_particles(
            n=n, xy=xyi, xy_sigma=xy_sigma, vxy=vxy, vxy_sigma=vxy_sigma)
        tracker.track(
            datetimes=datetimes, maxdt=maxdt, tile_size=tile_size,
            axy=axy, axy_sigma=axy_sigma)
        return np.vstack(tracker.means), np.dstack(tracker.covariances)
    with sharedmem.MapReduce(np=processes) as pool:
        return pool.map(process, xy)

def build_keypoints(matcher, mask=None, overwrite=False,
    clear_images=True, clear_keypoints=False, processes=None, **params):
    """
    Run KeypointsMatcher.build_keypoints() in parallel.
    """
    def process(img):
        matcher._build_image_keypoints(img=img, mask=mask, overwrite=overwrite,
            clear_images=clear_images, clear_keypoints=clear_keypoints, **params)
    with sharedmem.MapReduce(np=processes) as pool:
        pool.map(process, matcher.images)
