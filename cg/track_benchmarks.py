import cg
from cg import (glimpse, glob)
from glimpse.imports import (datetime, os, pandas, np, matplotlib, shapely)

# ---- Set cg constants ----

root = '/volumes/science-b/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
cg.KEYPOINT_PATH = os.path.join(root, 'timelapse-keypoints')
cg.MATCHES_PATH = os.path.join(root, 'timelapse-matches')
cg.DEM_PATHS = (
    os.path.join(root, 'dem'),
    os.path.join(root, 'dem-tandem/data'))

# ---- Set script constants ----
# optical-surveys-2005 (1, 3, 4)

STATION = 'CG05'
SERVICE = '20050916'
START = datetime.datetime(2005, 6, 2)
END = datetime.datetime(2005, 6, 25)
STEP = datetime.timedelta(hours=2)
BENCHMARK = 'optical-surveys-2005'
BENCHMARK_TRACK = 4
BENCHMARK_PATH = os.path.join(root, BENCHMARK, 'data', 'positions.csv')
RESULTS_PATH = os.path.join('tracks', BENCHMARK + '-' + str(BENCHMARK_TRACK))
os.makedirs(RESULTS_PATH, exist_ok=True)

# ---- Load images ----

images = cg.load_images(
    station=STATION, services=SERVICE,
    start=START, end=END, snap=STEP)
observer = glimpse.Observer(images)
dem_interpolant = cg.load_dem_interpolant(
    d=20, zlim=(1, np.inf),
    fun=glimpse.Raster.fill_crevasses, mask=lambda x: ~np.isnan(x))

# ---- Load benchmark (t, x, y, z) ----

positions = pandas.read_csv(BENCHMARK_PATH, parse_dates=['t'])
benchmark = positions[positions.track == BENCHMARK_TRACK]
benchmark.t.loc[:] = np.array([t.to_pydatetime() for t in benchmark.t])

# ---- Load DEM ----

dem = dem_interpolant(observer.datetimes[0])

# HACK: Linearly correct along flow (~y-axis)
xyz = benchmark.loc[:, ('x', 'y', 'z')].as_matrix()
dz = dem.sample(xyz[:, 0:2]) - xyz[:, 2]
fit = np.polyfit(xyz[:, 1], dz, deg=1)
dzi = np.polyval(fit, dem.y)
dem.Z -= dzi.reshape(-1, 1)

# ---- Track with timelapse (continuous) ----

# Crop observer
observer = observer.subset(start=benchmark.t.min(), end=benchmark.t.max())

# Animation (reference)
xyz = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y', 'z')].as_matrix(),
    x=benchmark.t.as_matrix(), xi=observer.datetimes)
ani = observer.track(
    xyz, size=(200, 200),
    subplots=dict(figsize=(12, 8), tight_layout=True))
ani.save(os.path.join(RESULTS_PATH, 'reference.mp4'))

# Track from benchmark position at first image
xy0 = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y')].as_matrix(),
    x=benchmark.t.as_matrix(),
    xi=observer.datetimes[0:1]).reshape(-1)
tracker = glimpse.Tracker(
    observers=[observer], dem=dem,
    time_unit=datetime.timedelta(days=1))
tracks = tracker.track(
    xy=(xy0, ) * 10, n=5000, xy_sigma=(2, 2), vxy=(0, -10), vxy_sigma=(5, 5),
    axy=(0, 0), axy_sigma=(2, 2), tile_size=(15, 15), return_particles=True)

# Write to file
observer.clear_images()
glimpse.helpers.write_pickle(tracks, os.path.join(RESULTS_PATH, 'tracks.pkl'))

# ---- Track with timelapse (piecewise) ----

# # Split observer
# breaks = [datetime.datetime(2005, 6, 12, 17, 0)]
# observers = observer.split(breaks, overlap=1)
#
# # Track from benchmark position at first image
# trackers = []
# xy0 = glimpse.helpers.interpolate_line_datetimes(
#     benchmark.loc[:, ('x', 'y')].as_matrix(),
#     x=benchmark.t, xi=observers[0].datetimes[0:1]).reshape(-1)
# for i, obs in enumerate(observers):
#     if i:
#         previous_tracker = tracker
#     tracker = glimpse.Tracker(
#         observers=(obs, ), dem=dem,
#         time_unit=datetime.timedelta(days=1).total_seconds())
#     if i:
#         new_xy0 = previous_tracker.means[-1][0, 0:2]
#         tracker.initialize_particles(n=5000, xy=new_xy0, xy_sigma=(2, 2),
#             vxy=(0, -10), vxy_sigma=(5, 15))
#         # tracker.particles = previous_tracker.particles
#         # tracker.weights = previous_tracker.weights
#         # tracker.tiles = previous_tracker.tiles
#         # tracker.histograms = previous_tracker.histograms
#     else:
#         tracker.initialize_particles(n=5000, xy=xy0, xy_sigma=(2, 2),
#             vxy=(0, -5), vxy_sigma=(5, 15))
#     tracker.track(axy=(0, 0), axy_sigma=(2, 2), tile_size=(15, 15))
#     trackers.append(tracker)
#
# # Collect results
# datetimes = observer.datetimes
# means = np.vstack([
#     np.vstack(trackers[i].means)[(1 if i else 0):]
#     for i, tracker in enumerate(trackers)])
# covariances = np.dstack([
#     np.dstack(trackers[i].covariances)[:, :, (1 if i else 0):]
#     for i, tracker in enumerate(trackers)])

# ---- Plot results ----

# Read from file
tracks = glimpse.helpers.read_pickle(os.path.join(RESULTS_PATH, 'tracks.pkl'))

# Plot tracks
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(benchmark.x, benchmark.y, 'k-')
tracks.plot_xy(mean=dict(color='red', alpha=0.25), sigma=True)
matplotlib.pyplot.gca().set_aspect(1)
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.show()
matplotlib.pyplot.savefig(os.path.join(RESULTS_PATH, 'xy.png'))

# Plot velocities (x)
matplotlib.pyplot.figure()
tracks.plot_vx(mean=dict(color='red', alpha=0.25), sigma=True)
xyz = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y', 'z')].as_matrix(),
    x=benchmark.t.as_matrix(), xi=tracks.datetimes)
matplotlib.pyplot.plot(
    tracks.datetimes[1:],
    np.diff(xyz[:, 0]) / (np.diff(tracks.datetimes) / tracks.tracker.time_unit),
    color='black')
# matplotlib.pyplot.show()
matplotlib.pyplot.savefig(os.path.join(RESULTS_PATH, 'vx.png'))

# Plot velocities (y)
matplotlib.pyplot.figure()
tracks.plot_vy(mean=dict(color='red', alpha=0.25), sigma=True)
xyz = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y', 'z')].as_matrix(),
    x=benchmark.t.as_matrix(), xi=tracks.datetimes)
matplotlib.pyplot.plot(
    tracks.datetimes[1:],
    np.diff(xyz[:, 1]) / (np.diff(tracks.datetimes) / tracks.tracker.time_unit),
    color='black')
# matplotlib.pyplot.show()
matplotlib.pyplot.savefig(os.path.join(RESULTS_PATH, 'vy.png'))

# Plot tracks with days
track = 0
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(benchmark.x, benchmark.y, 'k-')
matplotlib.pyplot.gca().set_aspect(1)
days = glimpse.helpers.datetime_range(
    benchmark.t.iloc[0], benchmark.t.iloc[-1], datetime.timedelta(days=1))
xyi = glimpse.helpers.interpolate_line_datetimes(
    benchmark.loc[:, ('x', 'y')].as_matrix(),
    x=benchmark.t.as_matrix(), xi=days)
matplotlib.pyplot.plot(xyi[:, 0], xyi[:, 1], 'k.')
for i in range(len(xyi)):
    matplotlib.pyplot.text(xyi[i, 0], xyi[i, 1], str(i))
tracks.plot_xy(index=track, mean=dict(color='red'), sigma=dict(color='red'))
xyi = glimpse.helpers.interpolate_line_datetimes(
    tracks.xyz[track, :, 0:2],
    x=tracks.datetimes, xi=days, error=False, fill=np.nan)
matplotlib.pyplot.plot(xyi[:, 0], xyi[:, 1], 'r.')
for i in range(len(xyi)):
    if not np.isnan(xyi[i, 0]):
        matplotlib.pyplot.text(xyi[i, 0], xyi[i, 1], str(i))
# matplotlib.pyplot.show()
matplotlib.pyplot.savefig(os.path.join(RESULTS_PATH, 'xy-' + str(track) + '.png'))

# Plot x, y sigma
# track = 0
# matplotlib.pyplot.figure()
# matplotlib.pyplot.plot(tracks.datetimes, tracks.xyz_sigma[track, :, 0].T, marker='.')
# matplotlib.pyplot.plot(tracks.datetimes, tracks.xyz_sigma[track, :, 1].T, marker='.')
# matplotlib.pyplot.legend(('x sigma', 'y sigma'))

# Animation: track
track = 0
ani = observer.track(
    tracks.xyz[track, :, 0:3], size=(200, 200),
    frames=tracks.images[:, track],
    subplots=dict(figsize=(12, 8), tight_layout=True))
ani.save(os.path.join(RESULTS_PATH, 'track-' + str(track) + '.mp4'))

# Animation: particles
track = 2
ani = tracks.animate(track=track, map_size=(20, 20), subplots=dict(figsize=(14, 8), tight_layout=True))
ani.save(os.path.join(RESULTS_PATH, 'particles-' + str(track) + '.mp4'))
