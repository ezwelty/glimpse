import glob
import os
import sys
sys.path.append("../")
import optimize
import cgcalib
import dem as DEM

IMG_DIR = "/volumes/science-b/data/columbia/timelapse"
IMG_SIZE = 0.5
SVG_KEYS = ['gcp', 'horizon', 'coast', 'terminus', 'moraines']

# ---- Calibrate station ---- #

# STATION = 'AK10b'
# STATION = 'AK01b'
# STATION = 'CG04'
# STATION = 'CG05'
# STATION = 'CG06'
# STATION = 'AK03'; STATION2 = 'AK03b'
# STATION = 'AK09'
# station_fit = camera_fit # use better calibrate_camera solution
# viewdir_keys = station_fit.params.keys()[-len(svg_images) * 3:]
# STATION = 'AK09b'
# station_fit = camera_fit # use better calibrate_camera solution
# viewdir_keys = station_fit.params.keys()[-len(svg_images) * 3:]
STATION = 'AK12'
images[2].cam.viewdir = [5.5, -9, 0]
images[3].cam.viewdir = [5.5, -9, 0]

# Gather svg control
images, controls, cam_params = cgcalib.station_svg_controls(
    STATION, root=IMG_DIR, keys=SVG_KEYS, size=IMG_SIZE,
    station_calib=False, camera_calib=True)

# Optional second station
images2, controls2, cam_params2 = [], [], []
# images2, controls2, cam_params2 = cgcalib.station_svg_controls(
#     STATION2, root=IMG_DIR, keys=SVG_KEYS, size=IMG_SIZE,
#     station_calib=False, camera_calib=True)

# Calibrate station
station_model = optimize.Cameras(
    cams=[img.cam for img in images + images2],
    controls=controls + controls2,
    cam_params=cam_params + cam_params2,
    group_params=dict(xyz=True))
station_fit = station_model.fit(group_params=[dict()], full=True)
print np.array(station_fit.params.valuesdict().values()[0:3]) - station_model.cams[0].xyz
# station_model.set_cameras(station_fit.params)

# ---- Verify with image plot ---- #

i = 0
images[i].plot()
station_model.plot(station_fit.params, cam=i)
images[i].set_plot_limits()

# ---- Save calibration to file ---- #
# xyz, mean(viewdir)

SUFFIX = ''

cam = images[0].cam.copy()
xyz_keys = station_fit.params.keys()[0:3]
viewdir_keys = station_fit.params.keys()[3:(len(images) + 1) * 3]
# (mean values)
cam.xyz = [station_fit.params[key].value for key in xyz_keys]
viewdir = [station_fit.params[key].value for key in viewdir_keys]
cam.viewdir = [np.mean(viewdir[0::3]), np.mean(viewdir[1::3]), np.mean(viewdir[2::3])]
cam.write(path="stations/" + STATION + SUFFIX + ".json",
    attributes=['xyz', 'viewdir'])
# (standard errors)
cam.xyz = [station_fit.params[key].stderr for key in xyz_keys]
viewdir = [station_fit.params[key].stderr for key in viewdir_keys]
cam.viewdir = [np.mean(viewdir[0::3]), np.mean(viewdir[1::3]), np.mean(viewdir[2::3])]
cam.write(path="stations/" + STATION + SUFFIX + "_stderr.json",
    attributes=['xyz', 'viewdir'])

# Optional second station

viewdir_keys = station_fit.params.keys()[(len(images) + 1) * 3:]
# (mean values)
cam.xyz = [station_fit.params[key].value for key in xyz_keys]
viewdir = [station_fit.params[key].value for key in viewdir_keys]
cam.viewdir = [np.mean(viewdir[0::3]), np.mean(viewdir[1::3]), np.mean(viewdir[2::3])]
cam.write(path="stations/" + STATION2 + SUFFIX + ".json",
    attributes=['xyz', 'viewdir'])
# (standard errors)
cam.xyz = [station_fit.params[key].stderr for key in xyz_keys]
viewdir = [station_fit.params[key].stderr for key in viewdir_keys]
cam.viewdir = [np.mean(viewdir[0::3]), np.mean(viewdir[1::3]), np.mean(viewdir[2::3])]
cam.write(path="stations/" + STATION2 + SUFFIX + "_stderr.json",
    attributes=['xyz', 'viewdir'])
