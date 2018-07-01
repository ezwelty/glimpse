# import matplotlib
# matplotlib.use('agg')
import cg
from cg import glimpse
import glimpse.unumpy as unp
import scipy.stats
# glimpse.config.use_numpy_matmul(False)
from glimpse.imports import (datetime, np, os, shapely, re, matplotlib, collections)
import glob
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')

# Horizontal (xy)
# Typical max velocity: 15
# Max velocity: 25 (2009|2010 aerial, 2009 survey, 2010 aerial), 30 (1994|1997|1999 aerial)
# Max acceleration: 4 (precipitation event, 1984 survey), 8 (tidal and precipitation forcing, 2014 radar)
# Vertical (z)
# Max velocity: 5 (tidal flotation, 2009 survey)
# Max acceleration: ~40 (tidal flotation, 2009 survey)
# = vz_sigma 3.5 (std of 5 * sin), az_sigma 44? (std of derivative of 5 * sin)

# n = 10000
# xy = (computed)
# xy_sigma = (2, 2)
# vxyz_sigma = (computed) + (fast vx, fast vy, min(a * flotation, b))
# axyz_sigma = (0, 0, 0)
# axyz_sigma = (2, 2, min(c * flotation, d))

# flotation_az_sigma = 12 # m / d^2, az_sigma at full flotation
# default_xy_sigma = (2, 2)
# default_axy_sigma = (2, 2)
# fast_vxy_sigma = (2, 2)

# # ---- Track points ----
# # tracks
#
# multi = np.all(mask, axis=1)
# xy = xy[multi][:100]
# mask = mask[multi][:100]
#
# time_unit = datetime.timedelta(days=1)
# tracker = glimpse.Tracker(
#     observers=observers, dem=dem, dem_sigma=dem_sigma, time_unit=time_unit)
# tracks = tracker.track(
#     xy=xy, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2),
#     tile_size=(15, 15), parallel=6, observer_mask=mask, datetimes=tracker.datetimes)
# matplotlib.pyplot.figure(figsize=(20, 20))
# tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
# matplotlib.pyplot.savefig('temp-forward.png', dpi=100)
#
# ends = tracks.xyz[:, -1, 0:2].copy()
#
# for obs in observers:
#     obs.datetimes = obs.datetimes[::-1]
# tracks = tracker.track(
#     xy=xy, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2),
#     tile_size=(15, 15), parallel=6, observer_mask=mask, datetimes=tracker.datetimes[::-1])
# matplotlib.pyplot.figure(figsize=(20, 20))
# tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
# matplotlib.pyplot.savefig('temp-forward-times-flipped.png', dpi=100)
# for obs in observers:
#     obs.datetimes = obs.datetimes[::-1]
#
# tracks = tracker.track(
#     xy=xy, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2),
#     tile_size=(15, 15), parallel=6, observer_mask=mask, datetimes=tracker.datetimes[::-1])
# matplotlib.pyplot.figure(figsize=(20, 20))
# tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
# matplotlib.pyplot.savefig('temp-reverse-fromstart.png', dpi=100)
#
# tracks = tracker.track(
#     xy=ends, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2),
#     tile_size=(15, 15), parallel=6, observer_mask=mask, datetimes=tracker.datetimes[::-1])
# matplotlib.pyplot.figure(figsize=(20, 20))
# tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
# matplotlib.pyplot.savefig('temp-reverse-fromend.png', dpi=100)
#
# for obs in observers:
#     obs.images = obs.images[::-1]
# tracks = tracker.track(
#     xy=xy, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2),
#     tile_size=(15, 15), parallel=6, observer_mask=mask, datetimes=tracker.datetimes)
# matplotlib.pyplot.figure(figsize=(20, 20))
# tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
# matplotlib.pyplot.savefig('temp-forward-fromstart-images-flipped.png', dpi=100)
# for obs in observers:
#     obs.images = obs.images[::-1]
#
# for obs in observers:
#     obs.images = obs.images[::-1]
# tracks = tracker.track(
#     xy=ends, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2),
#     tile_size=(15, 15), parallel=6, observer_mask=mask, datetimes=tracker.datetimes)
# matplotlib.pyplot.figure(figsize=(20, 20))
# tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
# matplotlib.pyplot.savefig('temp-forward-fromend-images-flipped.png', dpi=100)
# for obs in observers:
#     obs.images = obs.images[::-1]
#
# # ---- Plot tracks ----
# # datetimes (m, ) | means (n, m, 6) | covariances (n, m, 6, 6)
