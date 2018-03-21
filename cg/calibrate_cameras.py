import cg
from cg import (glimpse, glob)
from glimpse.imports import (os, re, np, matplotlib)
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

GROUP_PARAMS = (
    dict(),
    dict(f=True),
    dict(f=True, k=[0]),
    dict(f=True, k=[0, 1]),
    dict(f=True, k=[0, 1], p=True),
    dict(f=True, k=[0, 1], p=True, c=True))
SVG_KEYS = ('moraines', 'gcp', 'horizon', 'coast', 'terminus')

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
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'canon-20d' # CG06
# IMG_SIZE = 0.5
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-03-20' # AK03(b)
# IMG_SIZE = 0.5
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-13-20' # AK09
# IMG_SIZE = 0.5
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-14-20' # AK09b
# IMG_SIZE = np.array([968, 648]) * 1.5
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d300s' # AK12
# IMG_SIZE = 0.25
# svg_images[2].cam.viewdir = [5.5, -9, 0]
# svg_images[3].cam.viewdir = [5.5, -9, 0]
# CAMERA = 'canon-40d-01' # AKJNC
# IMG_SIZE = 0.5
# SVG_KEYS = ['moraines', 'gcp', 'horizon']
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-10-24' # AK10
# IMG_SIZE = [1936, 1296]
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-04-24' # AK01-2
# IMG_SIZE = 1
# CAMERA = 'nikon-d200-11-28' # AKST0XA
# IMG_SIZE = 0.5
# CAMERA = 'nikon-d200-12-28' # AKST0XB
# IMG_SIZE = 0.5
CAMERA = 'nikon-d200-04-35' # AK04
IMG_SIZE = 1

# Gather motion control
motion_images, motion_controls, motion_cam_params = cg.camera_motion_matches(
    CAMERA, size=IMG_SIZE, force_size=True, station_calib=False, camera_calib=False,
    match=dict(max_ratio=0.2))

# Gather svg control
svg_images, svg_controls, svg_cam_params = cg.camera_svg_controls(
    CAMERA, size=IMG_SIZE, force_size=True, keys=SVG_KEYS,
    correction=True, station_calib=False, camera_calib=False)

# Calibrate camera
# NOTE: Doesn't account for stations with non-fixed xyz
camera_model = glimpse.optimize.Cameras(
    cams=[img.cam for img in motion_images + svg_images],
    controls=motion_controls + svg_controls,
    cam_params=motion_cam_params + svg_cam_params,
    group_params=GROUP_PARAMS[-1])
# Fit parameters to model
camera_fit = camera_model.fit(group_params=GROUP_PARAMS[:-1], full=True)

# Mean error for SVG images
svg_size = np.sum([control.size for control in svg_controls])
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

DEM_DIR = '/volumes/science-b/data/columbia/dem'
ORTHO_DIR = '/volumes/science-b/data/columbia/ortho'
DEM_GRID_SIZE = 5
IMG_SIZE2 = 0.5 # relative to above

i = 0
ids = cg.parse_image_path(svg_images[i].path)
dem_path = glob.glob(os.path.join(DEM_DIR, ids['date_str'] + '*.tif'))[-1]
ortho_path = glob.glob(os.path.join(ORTHO_DIR, ids['date_str'] + '*.tif'))[-1]
# Apply fitted params
camera_model.set_cameras(camera_fit.params)
svg_images[i].cam.resize(IMG_SIZE2)
# Prepare dem
dem = glimpse.DEM.read(dem_path, d=DEM_GRID_SIZE)
dem.crop(zlim=(1, np.inf))
# Prepare ortho
ortho = glimpse.DEM.read(ortho_path, d=DEM_GRID_SIZE)
ortho.resample(dem, method='linear')
mask = dem.viewshed(svg_images[i].cam.xyz) & ~np.isnan(ortho.Z)
# Save original (resized)
svg_images[i].write(ids['basename'] + '-original.jpg', I=svg_images[i].read())
# Save ortho projection
I = cg.project_dem(svg_images[i].cam, dem, ortho.Z, mask=mask)
I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
I = (255 * (I / I.max() - I.min() / I.max()))
svg_images[i].write(ids['basename'] + '-distorted.jpg', I.astype(np.uint8))
# Reset cameras
camera_model.reset_cameras()

# ---- Save calibration to file ---- #
# fmm, cmm, k, p, (sensorsz)

SUFFIX = ''

cam = (motion_images + svg_images)[0].cam.copy()
keys = camera_fit.params.keys()[:camera_model.group_mask.sum()]
# (mean values)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].value for key in keys]
cam.write(path='cameras/' + CAMERA + SUFFIX + '.json',
    attributes=('fmm', 'cmm', 'k', 'p', 'sensorsz'))
# (standard errors)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].stderr for key in keys]
cam.write(path='cameras/' + CAMERA + SUFFIX + '_stderr.json',
    attributes=('fmm', 'cmm', 'k', 'p'))

# ---- Check single image (svg) ---- #

svg_path = 'svg/AKST03B_20100602_224800.svg'
img_path = cg.find_image(svg_path)
ids = cg.parse_image_path(img_path)
eop = cg.load_station_estimate(ids['station'])
img = glimpse.Image(img_path, cam=eop)
controls = cg.svg_controls(img, svg_path, keys=SVG_KEYS)
svg_model = glimpse.optimize.Cameras(img.cam, controls,
    cam_params=dict(viewdir=True), group_params=GROUP_PARAMS[-1])
svg_fit = svg_model.fit(full=True, group_params=GROUP_PARAMS[:-1])
img.plot()
svg_model.plot(svg_fit.params)
img.set_plot_limits()

# ---- Check undistorted image ---- #

img = glimpse.Image(
    cg.find_image('AKJNC_20120508_191103C'),
    cam='cameras/canon-40d-01.json')
ideal_cam = img.cam.copy()
ideal_cam.idealize()
I = img.project(ideal_cam)
img.write(path='test.jpg', I=I)
