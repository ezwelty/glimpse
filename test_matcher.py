import glimpse
from glimpse.imports import (datetime, np, os, sys, matplotlib)
sys.path.insert(0, 'cg')
import cg
import glob

# ---- Constants ----

STATION_DIR = os.path.join('data', 'AK01b')
IMG_DIR = os.path.join(STATION_DIR, 'images')
KEYPOINT_DIR = os.path.join(STATION_DIR, 'keypoints')
MATCH_DIR = os.path.join(STATION_DIR, 'matches')
MASK_PATH = os.path.join(STATION_DIR, 'mask', 'mask.npy')
START_TIME = datetime.datetime(2013, 6, 1, 0)
END_TIME = datetime.datetime(2013, 6, 30, 0)
ANCHOR_BASENAME = 'AK01b_20130615_220325'

# ---- Prepare Observer ----

all_img_paths = glob.glob(os.path.join(IMG_DIR, "*.JPG"))
datetimes = np.array([glimpse.Exif(path).datetime
    for path in all_img_paths])
inrange = np.logical_and(datetimes > START_TIME, datetimes < END_TIME)
img_paths = [all_img_paths[i] for i in np.where(inrange)[0]]
basenames = [glimpse.helpers.strip_path(path)
    for path in img_paths]
cam_args = cg.load_calibrations(image=ANCHOR_BASENAME, merge=True)
images = [glimpse.Image(
    path, cam=cam_args.copy(), anchor=(basename == ANCHOR_BASENAME),
    keypoints_path=os.path.join(KEYPOINT_DIR, basename + '.pkl'))
    for path, basename in zip(img_paths, basenames)]
observer = glimpse.Observer(images)

# ---- Build keypoint matches ----

mask = np.load(MASK_PATH)
matcher = glimpse.optimize.KeypointMatcher(images)
matcher.build_keypoints(
    masks=mask, contrastThreshold=0.02, overwrite=False, parallel=True)
matcher.build_matches(
    maxdt=datetime.timedelta(days=1), path=MATCH_DIR,
    overwrite=False, max_ratio=0.6, max_distance=10, parallel=True)
# Precompute xys
_ = matcher.matches_as_type(glimpse.optimize.RotationMatches)

# ---- Align Observer (ObserverCameras + RotationMatchesXYZ) ----

model = glimpse.optimize.ObserverCameras(observer,
    matches=matcher.matches_as_type(glimpse.optimize.RotationMatchesXYZ))
fit = model.fit(tol=1)

# ---- Align Observer (Cameras) ----

cams = [img.cam for img in images]
cam_params = [dict() if img.anchor else dict(viewdir=True) for img in images]

# RotationMatches
matches = [m.as_type(glimpse.optimize.RotationMatches)
    for m in matcher.matches.ravel() if m]
model_Cameras = glimpse.optimize.Cameras(cams, matches, cam_params=cam_params)
fit_Cameras = model_Cameras.fit(ftol=1, full=True)

# RotationMatchesXY
matchesXY = [m.as_type(glimpse.optimize.RotationMatchesXY)
    for m in matcher.matches.ravel() if m]
model_CamerasXY = glimpse.optimize.Cameras(cams, matchesXY, cam_params=cam_params)
fit_CamerasXY = model_CamerasXY.fit(ftol=1, full=True)

# --- Compare results (UV) ----

# ObserverCameras
residuals = model_Cameras.residuals(params=np.delete(fit.x.reshape(-1, 3), model.anchors, axis=0).ravel())
print(np.mean(np.linalg.norm(residuals, ord=1, axis=1)))
print(np.mean(np.linalg.norm(residuals, ord=2, axis=1)))

# Cameras + RotationMatches
residuals = model_Cameras.residuals(params=fit_Cameras.params)
print(np.mean(np.linalg.norm(residuals, ord=1, axis=1)))
print(np.mean(np.linalg.norm(residuals, ord=2, axis=1)))

# Cameras + RotationMatchesXY
residuals = model_Cameras.residuals(params=fit_CamerasXY.params)
print(np.mean(np.linalg.norm(residuals, ord=1, axis=1)))
print(np.mean(np.linalg.norm(residuals, ord=2, axis=1)))

# --- Compare results (XYZ) ----

# ObserverCameras
viewdirs = fit.x.reshape(-1, 3)
model.set_cameras(viewdirs=viewdirs)
residuals = np.vstack([m.predicted(cam=0) - m.predicted(cam=1) for m in model.matches.ravel() if m])
print(np.mean(np.linalg.norm(residuals, ord=1, axis=1)))
model.reset_cameras()

# Cameras + RotationMatches
temp = np.reshape(list(fit_Cameras.params.valuesdict().values()), (-1, 3))
viewdirs = np.insert(temp, model.anchors - range(len(model.anchors)), fit.x.reshape(-1, 3)[model.anchors], axis=0)
model.set_cameras(viewdirs=viewdirs)
residuals = np.vstack([m.predicted(cam=0) - m.predicted(cam=1) for m in model.matches.ravel() if m])
print(np.mean(np.linalg.norm(residuals, ord=1, axis=1)))
model.reset_cameras()

# ---- RANSAC (inspect match pairs) ----

# Filter matches
i = 1537
ransac_model = glimpse.optimize.Cameras(
    cams=matches[i].cams, controls=matches[i:(i + 1)], cam_params=(dict(), dict(viewdir=True)),
    scales=False, sparsity=False)
rvalues, rindex = glimpse.optimize.ransac(ransac_model, sample_size=30, max_error=0.5, min_inliers=30, iterations=100)

# Plot matches
image_index = [cams.index(cam) for cam in matches[i].cams]
fig, ax = matplotlib.pyplot.subplots(ncols=2, sharex=True, sharey=True, figsize=(16, 8), tight_layout=True)
for j in (0, 1):
    fig.sca(ax[j])
    img = images[image_index[j]]
    img.plot()
    ransac_model.plot(cam=j, params=rvalues, index=rindex, scale=5, width=2, selected='green', unselected='red')
    matplotlib.pyplot.title(str(image_index[j]) + ' : ' + glimpse.helpers.strip_path(img.path))

# ---- RANSAC (filter matches) ----

failed = []
small = []
for i, m in enumerate(matches):
    sample_size = min(30, m.size / 4)
    min_inliers = sample_size
    if sample_size < 5:
        small.append(i)
        continue
    # http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
    iterations = int(max(10, np.log(1 - 0.99) / np.log(1 - (1 - 0.1)**sample_size)))
    ransac_model = glimpse.optimize.Cameras(
        cams=m.cams, controls=(m), cam_params=(dict(), dict(viewdir=True)),
        scales=False, sparsity=False)
    try:
        rvalues, rindex = glimpse.optimize.ransac(
            ransac_model, sample_size=sample_size, min_inliers=min_inliers,
            max_error=0.5, iterations=iterations)
    except ValueError:
        failed.append(i)
        continue
    m.uvs = (m.uvs[0][rindex], m.uvs[1][rindex])
    m.xys = (m.xys[0][rindex], m.xys[1][rindex])
    m.size = len(rindex)

selected = np.ones(len(matches), dtype=bool)
selected[small + failed] = False
ransac_matches = [matches[i] for i in np.where(selected)[0]]

# Cameras + RotationMatches (RANSAC)
model_Cameras_ransac = glimpse.optimize.Cameras(cams, ransac_matches, cam_params=cam_params)
fit_Cameras_ransac = model_Cameras_ransac.fit(ftol=1, full=True)

# ObserverCameras (RANSAC)
model_ransac = glimpse.optimize.ObserverCameras(observer)
model_ransac.matches = np.full((len(images), len(images)), None)
for m in ransac_matches:
    image_index = [cams.index(cam) for cam in m.cams]
    model_ransac.matches[image_index[0], image_index[1]] = glimpse.optimize.RotationMatchesXYZ(
        cams=m.cams, uvs=m.uvs)
fit_ransac = model_ransac.fit(tol=1)

# ---- Plot results ----

# ObserverCameras
model.set_cameras(viewdirs=fit.x.reshape(-1, 3))
# Cameras + RotationMatches
model_Cameras.set_cameras(params=fit_Cameras.params)
# Cameras + RotationMatchesXY
model_CamerasXY.set_cameras(params=fit_CamerasXY.params)

# ObserverCameras (RANSAC)
model_ransac.set_cameras(viewdirs=fit_ransac.x.reshape(-1, 3))
# Cameras + RotationMatches (RANSAC)
model_Cameras_ransac.set_cameras(params=fit_Cameras_ransac.params)

# Plot
observer.animate(uv=(303, 695), size=(100, 100), interval=200, subplots=dict(figsize=(16, 8), tight_layout=True))
