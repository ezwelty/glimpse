import sys
sys.path.append("../")
import image
import optimize
import cgcalib

IMG_DIR = "/volumes/science-b/data/columbia/timelapse/"
IMG_SIZE = [968, 648]
GROUP_PARAMS = [
    dict(),
    dict(f=True),
    dict(f=True, k=[0]),
    dict(f=True, c=True, k=[0]),
    dict(f=True, c=True, k=[0, 1]),
    dict(f=True, c=True, k=[0, 1], p=[0,1])]
SVG_KEYS = ['gcp', 'horizon', 'coast', 'terminus', 'moraines']

# ---- Calibrate camera ---- #

# CAMERA = 'nikon-d200-18-20' # AK10b
CAMERA = 'nikon-d200-08-24' # AK01b

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
    group_params=GROUP_PARAMS[4])
camera_fit = camera_model.fit(group_params=GROUP_PARAMS[:4], full=True)
# camera_model.set_cameras(camera_fit.params)

# Mean error for SVG images
svg_size = np.sum([control.size() for control in svg_controls])
camera_model.errors(camera_fit.params)[-svg_size:].mean()

# ---- Verify with image plot ---- #

i = 0
svg_images[i].plot()
camera_model.plot(camera_fit.params, cam=len(motion_images) + i)
svg_images[i].set_plot_limits()

# ---- Save calibration to file ---- #
# fmm, cmm, k, p, (sensorsz)

cam = motion_images[0].cam.copy()
keys = camera_fit.params.keys()[:camera_model.group_mask.sum()]
# (mean values)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].value for key in keys]
cam.write(path="cameras/" + CAMERA + ".json",
    attributes=['fmm', 'cmm', 'k', 'p', 'sensorsz'])
# (standard errors)
cam.vector[camera_model.group_mask] = [camera_fit.params[key].stderr for key in keys]
cam.write(path="cameras/" + CAMERA + "_stderr.json",
    attributes=['fmm', 'cmm', 'k', 'p'])

# ---- Check svg image ---- #

svg_path = "svg/AK10b_20131106_200636.svg"
img_path = cgcalib.find_image(svg_path, IMG_DIR)
ids = cgcalib.parse_image_path(img_path)
eop = cgcalib.station_eop(ids['station'])
img = image.Image(img_path, cam=dict(xyz=eop['xyz'], viewdir=eop['viewdir']))
controls = cgcalib.svg_controls(img, svg_path, keys=SVG_KEYS)
svg_model = optimize.Cameras(img.cam, controls,
    cam_params=dict(viewdir=True, xyz=True), group_params=GROUP_PARAMS[4])
svg_fit = svg_model.fit(full=True, group_params=GROUP_PARAMS[:4], verbose=True)
img.plot()
svg_model.plot(svg_fit.params)
img.set_plot_limits()
