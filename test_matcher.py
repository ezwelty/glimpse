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

# ---- Align Observer (ObserverCameras + RotationMatchesXYZ) ----

mask = np.load(MASK_PATH)
model = glimpse.optimize.ObserverCameras(observer)
model.build_keypoints(masks=mask, contrastThreshold=0.02, overwrite=False, clear_images=True)
model.build_matches(max_dt=datetime.timedelta(days=1), path=MATCH_DIR, overwrite=False, max_ratio=0.6, max_distance=10)
fit = model.fit(tol=1)

# ---- Align Observer (Cameras) ----

cams = [img.cam for img in model.observer.images]
cam_params = [dict() if img.anchor else dict(viewdir=True) for img in model.observer.images]
matches_xyz = [m for m in np.triu(model.matches).ravel() if m]

# RotationMatches
matches = [glimpse.optimize.RotationMatches(
    cams=m.cams, uvs=(m.cams[0]._camera2image(m.xys[0]), m.cams[1]._camera2image(m.xys[1])))
    for m in matches_xyz]
model_Cameras = glimpse.optimize.Cameras(cams, matches, cam_params=cam_params)
fit_Cameras = model_Cameras.fit(ftol=1, full=True)

# RotationMatchesXY
matchesXY = [glimpse.optimize.RotationMatchesXY(
    cams=m.cams, uvs=(m.cams[0]._camera2image(m.xys[0]), m.cams[1]._camera2image(m.xys[1])))
    for m in matches_xyz]
model_CamerasXY = glimpse.optimize.Cameras(cams, matchesXY, cam_params=cam_params)
fit_CamerasXY = model_CamerasXY.fit(ftol=1, full=True)

# --- Compare results (UV) ----

# ObserverCameras
residuals = model_Cameras.residuals(params=np.delete(fit.x.reshape(-1, 3), model.anchors, axis=0).ravel())
print np.mean(np.linalg.norm(residuals, ord=1, axis=1))
print np.mean(np.linalg.norm(residuals, ord=2, axis=1))

# Cameras + RotationMatches
residuals = model_Cameras.residuals(params=fit_Cameras.params)
print np.mean(np.linalg.norm(residuals, ord=1, axis=1))
print np.mean(np.linalg.norm(residuals, ord=2, axis=1))

# Cameras + RotationMatchesXY
residuals = model_Cameras.residuals(params=fit_CamerasXY.params)
print np.mean(np.linalg.norm(residuals, ord=1, axis=1))
print np.mean(np.linalg.norm(residuals, ord=2, axis=1))

# --- Compare results (XYZ) ----

# ObserverCameras
viewdirs = fit.x.reshape(-1, 3)
model.set_cameras(viewdirs=viewdirs)
residuals = np.vstack([m.predicted(cam=0) - m.predicted(cam=1) for m in model.matches.ravel() if m])
print np.mean(np.linalg.norm(residuals, ord=1, axis=1))
model.reset_cameras()

# Cameras + RotationMatches
viewdirs = np.vstack((np.reshape(fit_Cameras.params.valuesdict().values(), (-1, 3)), fit.x.reshape(-1, 3)[model.anchors]))
model.set_cameras(viewdirs=viewdirs)
residuals = np.vstack([m.predicted(cam=0) - m.predicted(cam=1) for m in model.matches.ravel() if m])
print np.mean(np.linalg.norm(residuals, ord=1, axis=1))
model.reset_cameras()

# ---- Plot results ----

viewdirs = fit.x.reshape(-1, 3)
model.set_cameras(viewdirs=viewdirs)
observer.animate(uv=(303, 695), size=(100, 100), interval=200, subplots=dict(figsize=(16, 8), tight_layout=True))
