import glimpse
from glimpse.imports import (datetime, np, os, cv2, sys, datetime, matplotlib, sharedmem)
import time
sys.path.append('cg-calibrations')
import cgcalib
import glob

# ---- Constants ----

STATION_DIR = os.path.join('data', 'AK01b')
IMG_DIR = os.path.join(STATION_DIR, 'images')
SIFT_DIR = os.path.join(STATION_DIR, 'keypoints')
MATCH_DIR = os.path.join(STATION_DIR, 'matches/')
MASK_PATH = os.path.join(STATION_DIR, 'mask', 'mask.npy')
START_TIME = datetime.datetime(2013, 6, 13, 0)
END_TIME = datetime.datetime(2013, 6, 16, 0)
ANCHOR_IMAGE_NAME = 'AK01b_20130615_220325'

# ---- Prepare Observer ----

cam_args = cgcalib.load_calibration(image=ANCHOR_IMAGE_NAME)
img_paths = glob.glob(os.path.join(IMG_DIR, "*.JPG"))
images = []
for img_path in img_paths:
    basename = cgcalib.parse_image_path(img_path)['basename']
    is_anchor = basename == ANCHOR_IMAGE_NAME
    images.append(glimpse.Image(
        path=img_path, cam=cam_args,
        siftpath=os.path.join(SIFT_DIR, basename + '.p'), anchor_image=is_anchor))
datetimes = np.array([img.datetime for img in images])
inrange = np.logical_and(datetimes > START_TIME, datetimes < END_TIME)
observer = glimpse.Observer(
    [images[i] for i in np.where(inrange)[0]])

# ---- Align Observer ----

mask = np.load(MASK_PATH)
ms = glimpse.optimize.CameraMotionSolver(observer)
ms.generate_image_kp_and_des(masks=mask, contrastThreshold=0.02, overwrite_cached_kp_and_des=False)
matches = ms.generate_matches(match_bandwidth=10, match_path=MATCH_DIR, overwrite_matches=False)
out = ms.align()
viewdirs = np.split(out.x, len(out.x) / 3)
for i in range(len(viewdirs)):
    observer.images[i].cam.viewdir = viewdirs[i]

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
