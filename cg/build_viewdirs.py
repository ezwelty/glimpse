import matplotlib
matplotlib.use('agg')

import cg
from cg import glimpse
from glimpse.imports import (datetime, matplotlib, np, os)
import glob

root = '/volumes/science-b/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
cg.KEYPOINT_PATH = os.path.join(root, 'timelapse-keypoints')
cg.MATCHES_PATH = os.path.join(root, 'timelapse-matches')

# ---- Set script constants ----

STEP = datetime.timedelta(hours=2)

# Find all calibrated sequences
sequences = cg.Sequences()
cameras = np.intersect1d(
    sequences.camera,
    [glimpse.helpers.strip_path(path) for path in glob.glob(os.path.join(cg.CG_PATH, 'cameras', '*.json'))])
stations = np.intersect1d(
    sequences.station,
    [glimpse.helpers.strip_path(path) for path in glob.glob(os.path.join(cg.CG_PATH, 'stations', '*.json'))])
calibrated = np.isin(sequences.camera, cameras) & np.isin(sequences.station, stations)
sequence_dicts = sequences.loc[calibrated].to_dict(orient='records')

# Build keypoints
for sequence in sequence_dicts:
    print(sequence['station'], sequence['service'])
    images = cg.load_images(station=sequence['station'], service=sequence['service'], step=STEP)
    matcher = glimpse.optimize.KeypointMatcher(images)
    masks = cg.load_masks(images)
    matcher.build_keypoints(masks=masks, contrastThreshold=0.02, overwrite=False,
        clear_images=True, clear_keypoints=True, parallel=4)
    # # Distance to nearest anchor
    # anchors = np.where([img.anchor for img in images])[0]
    # dt = glimpse.helpers.pairwise_distance_datetimes(
    #     [img.datetime for img in images],
    #     [images[i].datetime for i in anchors])
    # nearest = np.argmin(dt, axis=1)
    # print('Max distance to anchor:', datetime.timedelta(seconds=np.max(dt[:, nearest])))
    # # Build keypoints and matches
    # template = images[anchors[0]].read()
    # matcher = glimpse.optimize.KeypointMatcher(images, template=template)
    # matcher.build_matches(max_dt=datetime.timedelta(days=1), path=cg.MATCHES_PATH,
    #     overwrite=False, max_ratio=0.6, max_distance=None, parallel=4)

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
