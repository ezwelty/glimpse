import os
import sys
import image
import optimize
import matplotlib
import glob
import re
import dem as DEM
cd ./cg-calibrations
import cgcalib

IMG_DIR = "/volumes/science-b/data/columbia/timelapse/"
GROUP_PARAMS = [
    dict(),
    dict(f=True),
    dict(f=True, k=[0]),
    dict(f=True, k=[0, 1]),
    dict(f=True, k=[0, 1], p=True),
    dict(f=True, k=[0, 1], p=True, c=True)]
SVG_KEYS = ['moraines', 'gcp', 'horizon', 'coast', 'terminus']

# ---- Calibrate camera ---- #

# CAMERA = 'nikon-d200-18-20' # AK10b
# CAMERA = 'nikon-d200-08-24' # AK01b
# CAMERA = 'nikon-e8700' # CG04
# IMG_SIZE = 0.5
# GROUP_PARAMS[2]['k'] = (0, -0.25, 0.25)
# GROUP_PARAMS[3]['k'] = ([0, 1], -0.25, 0.25)
# GROUP_PARAMS[4]['k'] = ([0, 1], -0.25, 0.25)
# GROUP_PARAMS[5]['k'] = ([0, 1], -0.25, 0.25)
# CAMERA = 'nikon-d2x' # CG05
# IMG_SIZE = 0.5
CAMERA = 'nikon-d200-03-20' # AK03(b)
IMG_SIZE = 0.5
GROUP_PARAMS.append(helper.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))

# Gather motion control
motion_images, motion_controls, motion_cam_params = cgcalib.camera_motion_matches(
    CAMERA, root=IMG_DIR, size=IMG_SIZE, force_size=True, ratio=0.3,
    station_calib=False, camera_calib=False)

# Gather svg control
svg_images, svg_controls, svg_cam_params = cgcalib.camera_svg_controls(
    CAMERA, root=IMG_DIR, size=IMG_SIZE, force_size=True, fixed=True, keys=SVG_KEYS,
    station_calib=False, camera_calib=False)

# Calibrate camera
# NOTE: Doesn't account for stations with non-fixed xyz
camera_model = optimize.Cameras(
    cams=[img.cam for img in motion_images + svg_images],
    controls=motion_controls + svg_controls,
    cam_params=motion_cam_params + svg_cam_params,
    group_params=GROUP_PARAMS[-1])
# Fit parameters to model
camera_fit = camera_model.fit(group_params=GROUP_PARAMS[:-1], full=True)
print np.array(camera_fit.params.valuesdict().values()[0:3]) - camera_model.cams[0].xyz
# camera_model.set_cameras(camera_fit.params)

# Mean error for SVG images
svg_size = np.sum([control.size() for control in svg_controls])
camera_model.errors(camera_fit.params)[-svg_size:].mean()

# ---- Verify with image plot (motion) ---- #

i = 0
motion_images[i].plot()
camera_model.plot(camera_fit.params, cam=i)
camera_model.set_cameras(camera_fit.params)
I = motion_images[i + 1].project(motion_images[i].cam)
matplotlib.pyplot.imshow(I, origin='upper', extent=(0, I.shape[1], I.shape[0], 0), alpha=0.5)
camera_model.reset_cameras()
motion_images[i].set_plot_limits()

# ---- Verify with image plot (svg) ---- #

i = 0
svg_images[i].plot()
camera_model.plot(camera_fit.params, cam=len(motion_images) + i)
svg_images[i].set_plot_limits()

# ---- Verify with ortho projection (svg) ---- #

DEM_DIR = "/volumes/science-b/data/columbia/dem/"
ORTHO_DIR = "/volumes/science-b/data/columbia/ortho/"
DEM_GRID_SIZE = 5
IMG_SIZE2 = 0.5 # relative to above

i = 1
date = re.findall("_([0-9]{8})_", svg_images[i].path)[0]
dem_path = glob.glob(DEM_DIR + date + "*.tif")[-1]
ortho_path = glob.glob(ORTHO_DIR + date + "*.tif")[-1]
# Apply fitted params
camera_model.set_cameras(camera_fit.params)
svg_images[i].cam.resize(IMG_SIZE2)
# Prepare dem
dem = DEM.DEM.read(dem_path)
smdem = dem.copy()
smdem.resize(smdem.d[0] / DEM_GRID_SIZE)
smdem.crop(zlim=[1, np.inf])
mask = smdem.visible(svg_images[i].cam.xyz)
# Prepare ortho
ortho = DEM.DEM.read(ortho_path)
smortho = ortho.copy()
smortho.resize(smortho.d[0] / DEM_GRID_SIZE)
smortho.resample(smdem, method="linear")
# Save original
basename = os.path.splitext(os.path.basename(svg_images[i].path))[0]
svg_images[i].write(basename + "-original.jpg", I=svg_images[i].read())
# Save ortho projection
I = cgcalib.dem_to_image(svg_images[i].cam, smdem, smortho.Z, mask=mask)
I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
I = (255 * (I / I.max() - I.min() / I.max()))
svg_images[i].write(basename + "-distorted.jpg", I.astype(np.uint8))
# Reset cameras
camera_model.reset_cameras()

# ---- Save calibration to file ---- #
# fmm, cmm, k, p, (sensorsz)

SUFFIX = ''

cam = (motion_images + svg_images)[0].cam.copy()
keys = camera_fit.params.keys()[:camera_model.group_mask.sum()]
# (mean values)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].value for key in keys]
cam.write(path="cameras/" + CAMERA + SUFFIX + ".json",
    attributes=['fmm', 'cmm', 'k', 'p', 'sensorsz'])
# (standard errors)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].stderr for key in keys]
cam.write(path="cameras/" + CAMERA + SUFFIX + "_stderr.json",
    attributes=['fmm', 'cmm', 'k', 'p'])

# ---- Check single image (svg) ---- #

svg_path = "svg/CG05_20050827_190000.svg"
img_path = cgcalib.find_image(svg_path, IMG_DIR)
ids = cgcalib.parse_image_path(img_path)
eop = cgcalib.station_eop(ids['station'])
img = image.Image(img_path, cam=dict(xyz=eop['xyz'], viewdir=eop['viewdir']))
controls = cgcalib.svg_controls(img, svg_path, keys=SVG_KEYS)
svg_model = optimize.Cameras(img.cam, controls,
    cam_params=dict(viewdir=True, xyz=True), group_params=GROUP_PARAMS[5])
svg_fit = svg_model.fit(full=True, group_params=GROUP_PARAMS[:5])
img.plot()
svg_model.plot(svg_fit.params)
img.set_plot_limits()
