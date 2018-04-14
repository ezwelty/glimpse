from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, sharedmem, cv2, datetime)
from . import (helpers, optimize, image)

def build_keypoints(matcher, masks=None, overwrite=False,
    clear_images=True, clear_keypoints=False, processes=None, **params):
    """
    Run KeypointsMatcher.build_keypoints() in parallel.
    """
    if masks is None or isinstance(masks, np.ndarray):
        masks = (masks, ) * len(matcher.images)
    def process(img, mask):
        matcher._build_image_keypoints(img=img, mask=mask, overwrite=overwrite,
            clear_images=clear_images, clear_keypoints=clear_keypoints, **params)
    with sharedmem.MapReduce(np=processes) as pool:
        pool.map(process, tuple(zip(matcher.images, masks)), star=True)
