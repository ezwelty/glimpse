import cg
from cg import (glimpse, glob)
from glimpse.imports import (os, re, np, matplotlib)
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

GROUP_PARAMS = [
    dict(),
    dict(f=True),
    dict(f=True, k=[0]),
    dict(f=True, k=[0, 1]),
    dict(f=True, k=[0, 1], p=True),
    dict(f=True, k=[0, 1], p=True, c=True)]
GROUP_PARAMS_XYZ = GROUP_PARAMS + [
    glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True))]
SVG_KEYS = ('moraines', 'gcp', 'horizon', 'coast', 'terminus')
IMG_SIZE = 1

# ---- Calibrate camera ---- #

CAMERA = 'nikon-e8700' # CG04
CAMERA = 'nikon-d2x' # CG05
# CAMERA = 'canon-20d' # CG06
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-03-20' # AK03, AK03b
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-04-35' # AK04
# CAMERA = 'nikon-d200-04-24' # AK01-2
# CAMERA = 'nikon-d200-08-24' # AK01b
# CAMERA = 'nikon-d200-11-28' # AKST0XA
# CAMERA = 'nikon-d200-12-28' # AKST0XB
# CAMERA = 'nikon-d200-10-24' # AK10
# IMG_SIZE = [1936, 1296]
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-13-20' # AK09
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-14-20' # AK09b
# IMG_SIZE = np.array([968, 648]) * 1.5
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d300s' # AK12
# svg_images[2].cam.viewdir = [5.5, -9, 0]
# svg_images[3].cam.viewdir = [5.5, -9, 0]
# CAMERA = 'canon-40d-01' # AKJNC
# SVG_KEYS = ['moraines', 'gcp', 'horizon']
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))
# CAMERA = 'nikon-d200-18-20' # AK10b
# GROUP_PARAMS.append(glimpse.helpers.merge_dicts(GROUP_PARAMS[-1], dict(xyz=True)))

# Gather motion control
motion_images, motion_controls, motion_cam_params = cg.camera_motion_matches(
    CAMERA, station_calib=False, camera_calib=True)

# Gather svg control
svg_images, svg_controls, svg_cam_params = cg.camera_svg_controls(
    CAMERA, keys=SVG_KEYS,
    correction=True, station_calib=False, camera_calib=True)

# Standardize image size
imgszs = np.unique([img.cam.imgsz for img in (motion_images + svg_images)], axis=0)
i = np.argmax(imgszs[:, 0])
if len(imgszs > 1):
    print('Resizing images and controls to', imgszs[i])
    for control in motion_controls + svg_controls:
        control.resize(size=imgszs[i], force=True)
    for img in motion_images + svg_images:
        img.cam.original_vector[6:8] = imgszs[i]

# Allow xyz to change if not fixed
sequences = cg.Sequences()
is_row = sequences.camera == CAMERA
stations = np.intersect1d(
    sequences.station.loc[is_row],
    [cg.parse_image_path(img.path)['station'] for img in svg_images])
stations_path = os.path.join(cg.CG_PATH, 'geojson', 'stations.geojson')
geo = glimpse.helpers.read_geojson(stations_path, crs=32606, key='id')
fixed = all([geo['features'][station]['properties']['fixed'] for station in stations])
if fixed:
    group_params = GROUP_PARAMS
else:
    group_params = GROUP_PARAMS_XYZ

# Calibrate camera
camera_model = glimpse.optimize.Cameras(
    cams=[img.cam for img in motion_images + svg_images],
    controls=motion_controls + svg_controls,
    cam_params=motion_cam_params + svg_cam_params,
    group_params=group_params[-1])

# Fit parameters to model
# NOTE: Have to use slower 'leastsq' for error bars
camera_fit = camera_model.fit(group_params=group_params[:-1], full=True, method='leastsq')
if not fixed:
    print('xyz deviation:', np.array(list(camera_fit.params.valuesdict().values())[0:3]) - camera_model.cams[0].xyz)

# ---- Save calibration to file ---- #
# fmm, cmm, k, p, (sensorsz)

SUFFIX = '-b'

cam = (motion_images + svg_images)[0].cam.copy()
keys = list(camera_fit.params)[:camera_model.group_mask.sum()]
# (mean values)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].value for key in keys]
cam.write(
    path=os.path.join('cameras', CAMERA + SUFFIX + '.json'),
    attributes=('fmm', 'cmm', 'k', 'p', 'sensorsz'),
    indent=4, flat_arrays=True)
# (standard errors)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].stderr for key in keys]
cam.write(
    path=os.path.join('cameras', CAMERA + SUFFIX + '_stderr.json'),
    attributes=('fmm', 'cmm', 'k', 'p'),
    indent=4, flat_arrays=True)

# ---- Resize for plotting ---- #

def set_figure_cameras(reset=False, resize=1):
    for control in camera_model.controls:
        control.resize(1)
    if reset:
        camera_model.reset_cameras()
    else:
        camera_model.set_cameras(camera_fit.params)
    for control in camera_model.controls:
        control.resize(resize)

# ---- Verify with image plot (motion) ---- #

i = 0
set_figure_cameras(resize=0.25)
motion_images[i].plot()
I = motion_images[i + 1].project(motion_images[i].cam)
matplotlib.pyplot.imshow(I, alpha=0.5)
motion_images[i].set_plot_limits()
set_figure_cameras(reset=True)

# ---- Verify with image plot (svg) ---- #

i = 0
set_figure_cameras()
svg_images[i].plot()
camera_model.plot(cam=len(motion_images) + i)
svg_images[i].set_plot_limits()
set_figure_cameras(reset=True)

# ---- Verify with ortho projection (svg) ---- #

DEM_DIR = '/volumes/science-b/data/columbia/dem'
ORTHO_DIR = '/volumes/science-b/data/columbia/ortho'
DEM_GRID_SIZE = 5
IMG_SIZE2 = 0.5

i = 0
ids = cg.parse_image_path(svg_images[i].path)
dem_path = glob.glob(os.path.join(DEM_DIR, ids['date_str'] + '*.tif'))[-1]
ortho_path = glob.glob(os.path.join(ORTHO_DIR, ids['date_str'] + '*.tif'))[-1]

# Set cameras
set_figure_cameras(resize=IMG_SIZE2)
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
set_figure_cameras(reset=True)

# ---- Check single image (svg) ---- #

basename = 'AKST03B_20100602_224800'
img = glimpse.Image(
    path=cg.find_image(basename),
    cam=cg.load_calibrations(basename, station_estimate=True, merge=True))
controls = cg.svg_controls(img, keys=SVG_KEYS)
svg_model = glimpse.optimize.Cameras(img.cam, controls,
    cam_params=dict(viewdir=True), group_params=GROUP_PARAMS[-1])
svg_fit = svg_model.fit(full=True, group_params=GROUP_PARAMS[:-1])
img.plot()
svg_model.plot(svg_fit.params)
img.set_plot_limits()

# ---- Check undistorted image ---- #

basename = 'AKJNC_20120508_191103C'
img = glimpse.Image(
    path=cg.find_image(basename),
    cam=cg.load_calibrations(basename, camera=True, merge=True))
ideal_cam = img.cam.copy()
ideal_cam.idealize()
I = img.project(ideal_cam)
img.write(path='test.jpg', I=I)
