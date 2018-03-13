import glimpse
from glimpse.imports import (datetime, np, os, cv2, sys, datetime, matplotlib, sharedmem)
sys.path.append('cg-calibrations')
import cgcalib
import glob

# ---- Constants ----

STATION_DIR = os.path.join('data', 'AK01b')
IMG_DIR = os.path.join(STATION_DIR, 'images')
KEYPOINT_DIR = os.path.join(STATION_DIR, 'keypoints')
MATCH_DIR = os.path.join(STATION_DIR, 'matches')
MASK_PATH = os.path.join(STATION_DIR, 'mask', 'mask.npy')
START_TIME = datetime.datetime(2013, 6, 13, 0)
END_TIME = datetime.datetime(2013, 6, 16, 0)
ANCHOR_BASENAME = 'AK01b_20130615_220325'

# ---- Prepare Observer ----

all_img_paths = glob.glob(os.path.join(IMG_DIR, "*.JPG"))
datetimes = np.array([glimpse.Exif(path).datetime
    for path in all_img_paths])
inrange = np.logical_and(datetimes > START_TIME, datetimes < END_TIME)
img_paths = [all_img_paths[i] for i in np.where(inrange)[0]]
basenames = [os.path.splitext(os.path.basename(path))[0]
    for path in img_paths]
cam_args = cgcalib.load_calibration(image=ANCHOR_BASENAME)
images = [glimpse.Image(
    path, cam=cam_args, anchor=(basename == ANCHOR_BASENAME),
    keypoints_path=os.path.join(KEYPOINT_DIR, basename + '.pkl'))
    for path, basename in zip(img_paths, basenames)]
observer = glimpse.Observer(images)

# ---- Align Observer ----

mask = np.load(MASK_PATH)
model = glimpse.optimize.ObserverCameras(observer)
model.build_keypoints(masks=mask, contrastThreshold=0.02, overwrite=False, clear_images=True)
model.build_matches(max_dt=datetime.timedelta(days=1), path=MATCH_DIR, overwrite=False, max_ratio=0.6, max_distance=10)
fit = model.fit(tol=1e-3)
viewdirs = fit.x.reshape(-1, 3)
model.set_cameras(viewdirs=viewdirs)

# ---- Plot results ----

xyz = np.atleast_2d([499211.00725947437, 6783756.0918104537, 479.33179419876853])
fig, axs = matplotlib.pyplot.subplots(nrows=1, ncols=2, figsize=(16, 8))
for i, img in enumerate(observer.images):
    uv = img.cam.project(xyz)[0]
    if i == 0:
        uv0 = uv.copy()
    axs[0].imshow(img.read())
    axs[0].plot(uv[0], uv[1], 'r.')
    axs[0].set_xlim(uv[0] - 50, uv[0] + 50)
    axs[0].set_ylim(uv[1] + 50, uv[1] - 50)
    axs[1].imshow(img.read())
    axs[1].plot(uv[0], uv[1], 'r.')
    axs[1].set_xlim(uv0[0] - 50, uv0[0] + 50)
    axs[1].set_ylim(uv0[1] + 50, uv0[1] - 50)
    fig.savefig(os.path.join(STATION_DIR, 'anims', 'anim_{0:03}.jpg'.format(i)), dpi=300)
    axs[0].cla()
    axs[1].cla()
