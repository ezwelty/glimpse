import matplotlib
matplotlib.use('agg')
import cg
from cg import glimpse
glimpse.config.use_numpy_matmul(False)
from glimpse.imports import (datetime, np, os, shapely, matplotlib)
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')

# ---- Select DEMs ----

dem_sigmas = {
    'aerometric': 1.5,
    'ifsar': 1.5 + 0.5, # additional time uncertainty
    'arcticdem': 3,
    'tandem': 3 # after bulk corrections
}
dem_keys = [
    ('20040618', 'aerometric'),
    ('20040707', 'aerometric'),
    ('20050811', 'aerometric'),
    ('20050827', 'aerometric'),
    ('20060712', 'aerometric'),
    ('20060727', 'aerometric'),
    ('20070922', 'aerometric'),
    ('20080811', 'aerometric'),
    ('20090803', 'aerometric'),
    ('20090827', 'aerometric'),
    ('20100525', 'aerometric'),
    ('20100602', 'aerometric'),
    ('20100720', 'ifsar'), # +- 10 days mosaic
    ('20100906', 'arcticdem'),
    ('20110618', 'tandem'),
    ('20110721', 'tandem'),
    ('20110812', 'tandem'),
    ('20110903', 'tandem'),
    ('20111211', 'tandem'),
    ('20120102', 'tandem'),
    ('20120204', 'tandem'),
    ('20120308', 'tandem'),
    ('20120329', 'arcticdem'),
    ('20120507', 'arcticdem'),
    ('20120617', 'arcticdem'),
    ('20120717', 'arcticdem'),
    ('20120813', 'arcticdem'),
    ('20121012', 'arcticdem'),
    ('20121123', 'arcticdem'),
    ('20130326', 'arcticdem'),
    ('20130610', 'arcticdem'),
    ('20130712', 'arcticdem'),
    ('20131119', 'arcticdem'),
    ('20140417', 'tandem'),
    # ('20140419', 'arcticdem'), # Coverage too small
    ('20140531', 'tandem'),
    ('20140622', 'tandem'),
    ('20140703', 'tandem'),
    ('20150118', 'arcticdem'),
    ('20150227', 'arcticdem'),
    ('20150423', 'arcticdem'),
    ('20150527', 'arcticdem'),
    ('20150801', 'arcticdem'),
    ('20150824', 'arcticdem'),
    ('20150930', 'arcticdem'),
    ('20160614', 'arcticdem'),
    ('20160820', 'arcticdem')
]
dem_keys.sort(key=lambda x: x[0])
surface_sigma = 3 # surface roughness
cg.DEM_PATHS = [os.path.join(root, 'dem-' + demtype, 'data', datestr + '.tif')
    for datestr, demtype in dem_keys]
cg.DEM_SIGMAS = [dem_sigmas[demtype] + surface_sigma for _, demtype in dem_keys]
dem_interpolant = cg.DEMInterpolant()
dem_args = dict(
    d=20,
    zlim=(1, np.inf),
    fun=glimpse.Raster.fill_crevasses,
    mask=lambda x: ~np.isnan(x), fill=False,
    return_sigma=True)
grid_step = (100, 100) # m
max_depth = 10e3 # m

# ---- Load Observers ----
# observers

observers = []
for station, service in (('AK01b', '20131106'), ('AK10b', '20130924')):
    images = cg.load_images(
        station=station, services=service, snap=datetime.timedelta(hours=2),
        service_exif=True, anchors=False, viewdir=True, viewdir_as_anchor=True)
    images = [img for img in images if img.anchor]
    # start = images[0].datetime
    # observer = glimpse.Observer(images).subset(
    #     start=start, end=start + datetime.timedelta(days=3))
    start = datetime.datetime(2013, 6, 12, 20)
    end = datetime.datetime(2013, 6, 15, 20)
    observer = glimpse.Observer(images).subset(start=start, end=end)
    observers.append(observer)

# ---- Load DEM ----
# dem | dem_sigma

boxes = [obs.images[0].cam.viewbox(max_depth) for obs in observers]
box = glimpse.helpers.union_boxes(boxes)
t = min([obs.datetimes[0] for obs in observers])
dem, dem_sigma = dem_interpolant(t, xlim=box[0::3], ylim=box[1::3], **dem_args)
for obs in observers:
    dem.fill_circle(obs.xyz, radius=400)

# ---- Load track points ----
# xy (n, ), mask (n, o)

xy, mask = cg.load_track_points(
    images=[obs.images[0] for obs in observers],
    dem=dem, max_depth=max_depth, step=grid_step)

# # Plot (map)
# matplotlib.pyplot.figure()
# dem.plot(dem.hillshade(), cmap='gray')
# for i, obs in enumerate(observers):
#     viewpoly = obs.images[0].cam.viewpoly(max_depth, step=100)[:, 0:2]
#     matplotlib.pyplot.plot(viewpoly[:, 0], viewpoly[:, 1], color='black')
#     matplotlib.pyplot.plot(xy[mask[:, i], 0], xy[mask[:, i], 1],
#         marker='.', linestyle='none')
# matplotlib.pyplot.savefig('temp.png', dpi=100)
# # Plot (image)
# for i, obs in enumerate(observers):
#     matplotlib.pyplot.figure()
#     xyz = np.column_stack((xy[mask[:, i]], dem.sample(xy[mask[:, i]])))
#     uv = obs.images[0].cam.project(xyz, correction=True)
#     obs.images[0].plot()
#     matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], color='red', marker='.',
#         linestyle='none')
#     matplotlib.pyplot.savefig('temp-' + str(i) + '.png', dpi=100)

# ---- Track points ----
# tracks

time_unit = datetime.timedelta(days=1)
tracker = glimpse.Tracker(
    observers=observers, dem=dem, dem_sigma=dem_sigma, time_unit=time_unit)
tracks = tracker.track(
    xy=xy, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2), axyz_sigma=(2, 2, 0.2),
    tile_size=(15, 15), parallel=6, observer_mask=mask)

# Plot
matplotlib.pyplot.figure(figsize=(20, 20))
tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
matplotlib.pyplot.savefig('temp.png', dpi=100)

# ---- Plot tracks ----
# datetimes (m, ) | means (n, m, 6) | covariances (n, m, 6, 6)
