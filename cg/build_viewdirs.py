import matplotlib
matplotlib.use('agg')

import cg
from cg import glimpse
from glimpse.imports import (sys, datetime, matplotlib, np, os)
import glob

root = '/volumes/science-b/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
cg.KEYPOINT_PATH = os.path.join(root, 'timelapse-keypoints')
cg.MATCHES_PATH = os.path.join(root, 'timelapse-matches')

# ---- Set script constants ----

STATIONS = (
    'CG04', 'CG05', 'CG06', 'AK01', 'AK01b', 'AK03', 'AK03b', 'AK09', 'AK09b',
    'AK10', 'AK10b', 'AK12', 'AKJNC', 'AKST03A', 'AKST03B')
SKIP_SEQUENCES = (
    'AK01_20070817', 'AK01_20080616',
    'AK01b_20080619', 'AK01b_20090824', 'AK01b_20090825', 'AK01b_20160908',
    'AK03_20070817', 'AK12_20100820')
SNAP = datetime.timedelta(hours=2)
MAXDT = datetime.timedelta(days=1)
MIN_NEAREST = 3

# ---- Find calibrated sequences ----

sequences = cg.Sequences()
cameras = np.intersect1d(
    sequences.camera,
    [glimpse.helpers.strip_path(path) for path in glob.glob(os.path.join(cg.CG_PATH, 'cameras', '*.json'))])
stations = np.intersect1d(
    STATIONS,
    [glimpse.helpers.strip_path(path) for path in glob.glob(os.path.join(cg.CG_PATH, 'stations', '*.json'))])
calibrated = np.isin(sequences.camera, cameras) & np.isin(sequences.station, stations)
skipped = np.isin(
    [station + '_' + service for station, service in zip(sequences.station, sequences.service)],
    SKIP_SEQUENCES)
sequence_dicts = sequences.loc[calibrated & ~skipped].to_dict(orient='records')
station_services = {station: sequences.loc[calibrated & ~skipped & (sequences.station == station)].service.tolist()
    for station in sequences.loc[calibrated & ~skipped].station.unique()}

# ---- Build keypoints (DONE) ----

for station in station_services:
    services = station_services[station]
    print(station, services)
    images = cg.load_images(
        station=station, services=services, snap=STEP,
        use_exif=False, service_exif=True, anchors=True)
    masks = cg.load_masks(images)
    matcher = glimpse.optimize.KeypointMatcher(images)
    matcher.build_keypoints(masks=masks, contrastThreshold=0.02, overwrite=False,
        clear_images=True, clear_keypoints=True, parallel=4)

# ---- Build keypoint matches (DONE) ----

for station in station_services:
    # Load all station sequences to match across service breaks
    services = station_services[station]
    print(station, services)
    images = cg.load_images(
        station=station, services=services, snap=STEP,
        use_exif=False, service_exif=True, anchors=True)
    # Build matches
    matcher = glimpse.optimize.KeypointMatcher(images)
    matcher.build_matches(
        maxdt=datetime.timedelta(days=1), min_nearest=3,
        path=cg.MATCHES_PATH, overwrite=False, max_ratio=0.75,
        max_distance=None, parallel=4, weights=True,
        clear_keypoints=True, clear_matches=True)

# ---- Algin images ----
# TODO: Select image intervals based on anchors and large gaps

# # Locate anchors
# anchors = np.where([img.anchor for img in images])[0]
# print('Anchors:', anchors)
# # Max distance to nearest anchor
# datetimes = np.array([img.datetime for img in images])
# dt = glimpse.helpers.pairwise_distance_datetimes(datetimes, datetimes[anchors])
# if dt.size:
#     nearest = np.argmin(dt, axis=1)
#     maxdt = datetime.timedelta(seconds=np.max(dt[range(len(nearest)), nearest]))
#     print('Max distance to nearest anchor:', maxdt)
# else:
#     raise ValueError('No anchors found')
# # Gaps larger than 1 day
# ddays = np.array([dt.total_seconds() / (3600 * 24) for dt in np.diff(datetimes)])
# gaps = np.where(ddays > 1)[0]
# for i in gaps:
#     print(glimpse.helpers.strip_path(images[i].path), '->',
#         glimpse.helpers.strip_path(images[i + 1].path),
#         '(' + str(round(ddays[i], 1)), 'days)')

# matches = matcher.matches_as_type(glimpse.optimize.RotationMatchesXYZ)
# observer = glimpse.Observer(images)
# model = glimpse.optimize.ObserverCameras(observer, matches)
# start = timeit.default_timer()
# fit = model.fit(tol=0.1)
# print(timeit.default_timer() - start)
# cg.write_image_viewdirs(images, viewdirs=fit.x.reshape(-1, 3))

# model.set_cameras(fit.x.reshape(-1, 3))
# ani = observer.animate(uv=(2048, 993), size=(200, 200), tight_layout=True)
# ani.save('/users/admin/desktop/optical-surveys-2005.mp4')
