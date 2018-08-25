
def circle(center=(0, 0), r=1, n=100):
    x = 2 * (np.pi / n) * np.arange(n + 1)
    return center + r * np.column_stack((np.cos(x), np.sin(x)))

xyz0 = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y', 'z')].as_matrix(),
    x=benchmark.t, xi=observer.datetimes[0:1]).reshape(-1)
xy = circle(center=xyz0[0:2], r=20)
xyz = np.column_stack((xy, np.zeros(len(xy))))
xyz[:, 2] = xyz0[2]
uv = observer.images[0].cam.project(xyz)
box = glimpse.helpers.bounding_box(uv)
np.diff(glimpse.helpers.unravel_box(box), axis=0)


import cg
from cg import (glimpse, glob)
from glimpse.imports import (datetime, os, pandas, np, matplotlib)

# ---- Set cg constants ----

root = '/volumes/science-b/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
cg.KEYPOINT_PATH = os.path.join(root, 'timelapse-keypoints')
cg.MATCHES_PATH = os.path.join(root, 'timelapse-matches')
cg.DEM_PATHS = (
    os.path.join(root, 'dem'),
    os.path.join(root, 'dem-tandem/data'))

# ---- Set script constants ----

STATION = 'CG05'
SERVICE = '20050916'
START = datetime.datetime(2005, 6, 2)
END = datetime.datetime(2005, 6, 25)
STEP = datetime.timedelta(hours=2)
BENCHMARK_PATH = os.path.join(root, 'optical-surveys-2005/data/positions.csv')
BENCHMARK_TRACK = 1

# ---- Load images ----

images = cg.load_images(
    station=STATION, services=SERVICE,
    start=START, end=END, snap=STEP)
observer = glimpse.Observer(images, cache=False)
dem_interpolant = cg.load_dem_interpolant(
    d=20, zlim=(1, np.inf),
    fun=glimpse.DEM.fill_crevasses, mask=lambda x: ~np.isnan(x))

# ---- Load benchmark (t, x, y, z) ----

positions = pandas.read_csv(BENCHMARK_PATH, parse_dates=['t'])
benchmark = positions[positions.track == BENCHMARK_TRACK]

# ---- Load DEM ----

dem = dem_interpolant(observer.datetimes[0])

# HACK: Linearly correct along flow (~y-axis)
xyz = benchmark.loc[:, ('x', 'y', 'z')].as_matrix()
dz = dem.sample(xyz[:, 0:2]) - xyz[:, 2]
fit = np.polyfit(xyz[:, 1], dz, deg=1)
dzi = np.polyval(fit, dem.y)
dem.Z -= dzi.reshape(-1, 1)

# ---- Track with timelapse ----

# Crop observer
observer = observer.subset(start=benchmark.t.min(), end=benchmark.t.max())

# Track from benchmark position at first image
xy0 = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y')].as_matrix(),
    x=benchmark.t, xi=observer.datetimes[0:1]).reshape(2)
tracker = glimpse.Tracker(
    observers=[observer], dem=dem,
    time_unit=datetime.timedelta(days=1).total_seconds())
tracker.initialize_particles(n=5000, xy=xy0, xy_sigma=(2, 2),
    vxy=(0, 0), vxy_sigma=(10, 10))
tracker.track(axy=(0, 0), axy_sigma=(2, 2), tile_size=(15, 15))

# ---- Plot results ----

# Collect results
datetimes = tracker.datetimes
means, covariances = np.vstack(tracker.means), np.dstack(tracker.covariances)

# Animate result
ani = observer.track(
    means[:, 0:3], size=(200, 200),
    subplots=dict(figsize=(12, 8), tight_layout=True))

# Plot x, y sigma
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(datetimes, np.sqrt(covariances[0, 0, :]), marker='.')
matplotlib.pyplot.plot(datetimes, np.sqrt(covariances[1, 1, :]), marker='.')
matplotlib.pyplot.legend(('x sigma', 'y sigma'))

# Plot tracks
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(benchmark.x, benchmark.y, 'k-')
matplotlib.pyplot.gca().set_aspect(1)
days = glimpse.helpers.datetime_range(
    track.t.iloc[0], track.t.iloc[-1], datetime.timedelta(days=1))
xyi = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y')].as_matrix(), x=track.t, xi=days)
matplotlib.pyplot.plot(xyi[:, 0], xyi[:, 1], 'k.')
for i in range(len(xyi)):
    matplotlib.pyplot.text(xyi[i, 0], xyi[i, 1], str(i))
matplotlib.pyplot.plot(means[:, 0], means[:, 1], 'r-')
xyi = glimpse.helpers.interpolate_line_datetimes(means[:, 0:2],
    x=datetimes, xi=days, error=False, fill=np.nan)
matplotlib.pyplot.plot(xyi[:, 0], xyi[:, 1], 'r.')
for i in range(len(xyi)):
    if not np.isnan(xyi[i, 0]):
        matplotlib.pyplot.text(xyi[i, 0], xyi[i, 1], str(i))

# # ---- Load masks ----
#
# masks = cg.load_masks(images)
# # Plot
# i = 0
# images[i].plot()
# matplotlib.pyplot.imshow(masks[i], alpha=0.5)
#
# # ---- Build keypoints and matches ----
#
# anchors = [i for i, img in enumerate(images) if img.anchor]
# template = images[anchors[0]].read()
# matcher = glimpse.optimize.KeypointMatcher(images, template=template)
# matcher.build_keypoints(masks=masks, contrastThreshold=0.02, overwrite=False,
#     clear_images=True, clear_keypoints=True)
# matcher.build_matches(max_dt=datetime.timedelta(days=1), path=cg.MATCHES_PATH,
#     overwrite=False, max_ratio=0.6, max_distance=None)
#
# # ---- Align images ----
#
# matches = matcher.matches_as_type(glimpse.optimize.RotationMatchesXYZ)
# observer = glimpse.Observer(images)
# model = glimpse.optimize.ObserverCameras(observer, matches)
# start = timeit.default_timer()
# fit = model.fit(tol=0.1)
# print(timeit.default_timer() - start)
# cg.write_image_viewdirs(images, viewdirs=fit.x.reshape(-1, 3))
#
# model.set_cameras(fit.x.reshape(-1, 3))
# ani = observer.animate(uv=(2048, 993), size=(200, 200), tight_layout=True)
# ani.save('/users/admin/desktop/optical-surveys-2005.mp4')

# ---- V1: Continuous track ----

# Load observed track
path = '/volumes/science-b/data/columbia/optical-surveys-2005/data/positions.csv'
positions = pandas.read_csv(path, parse_dates=['t'])
track = positions[positions.track == 1]

# Crop Observer
track_xyz = glimpse.helpers.interpolate_line_datetimes(
    track.loc[:, ('x', 'y', 'z')].as_matrix(),
    x=track.t, xi=observer.datetimes, error=False, fill=np.nan)
frames = np.where(~np.isnan(track_xyz[:, 0]))[0]
observer = glimpse.Observer(np.array(images)[frames])
# TODO: Observer.crop(start=None, end=None)
# Project track
# ani = observer.track(track_xyz[frames], size=(200, 200), subplots=dict(figsize=(12, 8), tight_layout=True))
# ani.save('/users/admin/desktop/optical-surveys-2005-track-1.mp4')

# box = glimpse.helpers.bounding_box(track.loc[:, ('x', 'y')].as_matrix())
# box[0:2] -= 500
# box[2:4] += 500
# dem_mid = dem_interpolant(observer.datetimes[0], xlim=box[0::2], ylim=box[1::2])
# dem_left = dem_interpolant(datetime.datetime(2004, 7, 7, 22, 0), xlim=box[0::2], ylim=box[1::2])
# dem_right = dem_interpolant(datetime.datetime(2005, 8, 11, 22, 0), xlim=box[0::2], ylim=box[1::2])
# xyz = track.loc[:, ('x', 'y', 'z')].as_matrix()
# matplotlib.pyplot.plot(xyz[:, 1], dem_left.sample(xyz[:, 0:2]))
# matplotlib.pyplot.plot(xyz[:, 1], xyz[:, 2])
# matplotlib.pyplot.plot(xyz[:, 1], dem_mid.sample(xyz[:, 0:2]))
# matplotlib.pyplot.plot(xyz[:, 1], dem_right.sample(xyz[:, 0:2]))
# matplotlib.pyplot.legend((
#     '2004-07-07: DEM',
#     '2005-06-09: Optical survey',
#     '2005-06-09: DEM interpolation',
#     '2005-08-11: DEM'))

# Prepare DEM
dem = dem_interpolant(observer.datetimes[0])
dz = dem.sample(xyz[:, 0:2]) - xyz[:, 2]
# Linearly correct over y-axis
fit = np.polyfit(xyz[:, 1], dz, deg=1)
dzi = np.polyval(fit, dem.y)
dem.Z -= dzi.reshape(-1, 1)

# Split Observer into 3-day Observers with 1-image overlap
n = (observer.datetimes[-1] - observer.datetimes[0]) // datetime.timedelta(days=3)
split_observers = observer.split(n, overlap=1)
split_observers = [observer]
split_observers = [glimpse.Observer(observer.images[0:30]), glimpse.Observer(observer.images[29:])]

# Track point from true position at start of each Observer
tracks = []
for i, obs in enumerate(split_observers):
    obs.cache = False
    if i:
        previous_tracker = tracker
    tracker = glimpse.Tracker(
        observers=(obs, ), dem=dem,
        time_unit=datetime.timedelta(days=1).total_seconds())
    if i:
        tracker.particles = previous_tracker.particles
        tracker.weights = previous_tracker.weights
        # tracker.tiles = previous_tracker.tiles
        # tracker.histograms = previous_tracker.histograms
        # xy0 = previous_tracker.means[-1][0, 0:2]
        # tracker.initialize_particles(n=5000, xy=xy0, xy_sigma=(2, 2),
        #     vxy=(0, 0), vxy_sigma=(10, 10))
    else:
        xy0 = glimpse.helpers.interpolate_line_datetimes(
            track.loc[:, ('x', 'y')].as_matrix(),
            x=track.t, xi=obs.datetimes[0:1])[0]
        tracker.initialize_particles(n=5000, xy=xy0, xy_sigma=(2, 2),
            vxy=(0, 0), vxy_sigma=(10, 10))
    tracker.track(axy=(0, 0), axy_sigma=(2, 2), tile_size=(15, 15))
    tracks.append((
        obs.datetimes,
        np.vstack(tracker.means),
        np.dstack(tracker.covariances)))

# # Save to file
# glimpse.helpers.write_pickle(tracks, 'optical-surveys-2005-track-1g.pkl')
# tracks = glimpse.helpers.read_pickle('optical-surveys-2005-track-1b.pkl')

# Aggregate results
datetimes = observer.datetimes
means = np.vstack((
    tracks[0][1],
    np.vstack([track[1][1:] for track in tracks[1:]])))
covariances = np.dstack((
    tracks[0][2],
    np.dstack([track[2][:, :, 1:] for track in tracks[1:]])))

datetimes, means, covariances = tracks[0]

# Project result
ani = observer.track(means[:, 0:3], size=(200, 200), subplots=dict(figsize=(12, 8), tight_layout=True))
ani.save('/users/admin/desktop/optical-surveys-2005-1/continuous-animation.mp4')

# Plot results

# x, y sigma
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(datetimes, np.sqrt(covariances[0, 0, :]), marker='.')
matplotlib.pyplot.plot(datetimes, np.sqrt(covariances[1, 1, :]), marker='.')
matplotlib.pyplot.legend(('x sigma', 'y sigma'))

# tracks
matplotlib.pyplot.figure()
track_xy = track.loc[:, ('x', 'y')].as_matrix()
matplotlib.pyplot.plot(track_xy[:, 0], track_xy[:, 1], 'k-')
matplotlib.pyplot.gca().set_aspect(1)
days = glimpse.helpers.datetime_range(
    track.t.iloc[0], track.t.iloc[-1], datetime.timedelta(days=1))
xyi = glimpse.helpers.interpolate_line_datetimes(track_xy,
    x=track.t, xi=days)
matplotlib.pyplot.plot(xyi[:, 0], xyi[:, 1], 'k.')
for i in range(len(xyi)):
    matplotlib.pyplot.text(xyi[i, 0], xyi[i, 1], str(i))
matplotlib.pyplot.plot(means[:, 0], means[:, 1], 'r-')
xyi = glimpse.helpers.interpolate_line_datetimes(means[:, 0:2],
    x=datetimes, xi=days, error=False, fill=np.nan)
matplotlib.pyplot.plot(xyi[:, 0], xyi[:, 1], 'r.')
for i in range(len(xyi)):
    if not np.isnan(xyi[i, 0]):
        matplotlib.pyplot.text(xyi[i, 0], xyi[i, 1], str(i))

# for i in range(len(tracks)):
#     matplotlib.pyplot.plot(tracks[i][1][0, 0], tracks[i][1][0, 1], 'g.')
#     matplotlib.pyplot.plot(tracks[i][1][1:, 0], tracks[i][1][1:, 1], 'r-')
#     matplotlib.pyplot.text(tracks[i][1][0, 0], tracks[i][1][0, 1], str(i), color='green')

# ---- Build tracking points ----

# TODO: Load full extent by cutting outline polygon with terminus

# xy = positions.loc[:, ('x', 'y')].as_matrix()
# box = glimpse.helpers.bounding_box(xy)
# grid = glimpse.helpers.box_to_grid(box, step=50, snap=(0, 0))
# points = glimpse.helpers.grid_to_points(grid)

# Filter points by their visibility in all images
# FIXME: Add a buffer?
# FIXME: Return constant likelihoods for points not in view

# dem = dem_interpolant(observer.datetimes[0])
# xyz = np.column_stack((points, dem.sample(points)))
# inviews = (img.cam.inview(xyz) for img in observer.images)
# inview = np.column_stack(inviews).all(axis=1)
# points = points[inview, :]

# ---- Track points (continuous) ----

# Split Observer into 3-day Observers
n = (observer.datetimes[-1] - observer.datetimes[0]) // datetime.timedelta(days=3)
split_observers = observer.split(n, overlap=1)

# For each Observer, track points in parallel
xy0 = points
tracks = []
for obs in split_observers:
    obs.cache_images()
    midtime = obs.datetimes[0] + (obs.datetimes[-1] - obs.datetimes[0]) * 0.5
    dem = dem_interpolant(midtime)
    tracker = glimpse.Tracker(
        observers=(obs, ), dem=dem,
        time_unit=datetime.timedelta(days=1).total_seconds())
    track = glimpse.parallel.track(tracker, xy=xy0, n=5000, xy_sigma=(2, 2),
        vxy=(0, 0), vxy_sigma=(10, 10), axy=(0, 0), axy_sigma=(0, 0),
        tile_size=(15, 15))
    tracks.append(track)
    obs.clear_images()
    xy0 = track[0]

# Aggregate tracks
# tracks[i_obs] = [(means, covariances), ...] for each point
# datetime | mean
datetimes = observer.datetimes
means = np.vstack((
    np.dstack([means for means, covariances in tracks[0]]),
    np.vstack([np.dstack([means[1:, :] for means, covariances in track]) for track in tracks[1:]])
))

glimpse.helpers.write_pickle((datetimes, means))

# ---- Compare results ----

for i in range(len(points)):
    matplotlib.pyplot.plot(means[0, 0, i], means[0, 1, i], 'g.')
    matplotlib.pyplot.plot(means[1:, 0, i], means[1:, 1, i], 'r.')

for track in (1, 2, 5):
    xy = positions.loc[positions.track == track, ('x', 'y')].as_matrix()
    matplotlib.pyplot.plot(xy[:, 0], xy[:, 1], 'k-')
