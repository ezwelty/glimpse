import glob
import os
import sys
sys.path.append("../")
import optimize
import cgcalib
import dem as DEM

IMG_DIR = "/volumes/science-b/data/columbia/timelapse"
IMG_SIZE = 0.25
SVG_KEYS = ['gcp', 'horizon', 'coast', 'terminus', 'moraines']

# ---- Calibrate station ---- #

STATION = 'AK10b'
STATION = 'AK01b'

# Gather svg control
images, controls, cam_params = cgcalib.station_svg_controls(
    STATION, root=IMG_DIR, keys=SVG_KEYS, size=IMG_SIZE,
    station_calib=False, camera_calib=True)

# Calibrate station
station_model = optimize.Cameras(
    cams=[img.cam for img in images],
    controls=controls,
    cam_params=cam_params,
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

cam = images[0].cam.copy()
xyz_keys = station_fit.params.keys()[0:3]
viewdir_keys = station_fit.params.keys()[3:]
# (mean values)
cam.xyz = [station_fit.params[key].value for key in xyz_keys]
viewdir = [station_fit.params[key].value for key in viewdir_keys]
cam.viewdir = [np.mean(viewdir[0::3]), np.mean(viewdir[1::3]), np.mean(viewdir[2::3])]
cam.write(path="stations/" + STATION + ".json",
    attributes=['xyz', 'viewdir'])
# (standard errors)
cam.xyz = [station_fit.params[key].stderr for key in xyz_keys]
viewdir = [station_fit.params[key].stderr for key in viewdir_keys]
cam.viewdir = [np.mean(viewdir[0::3]), np.mean(viewdir[1::3]), np.mean(viewdir[2::3])]
cam.write(path="stations/" + STATION + "_stderr.json",
    attributes=['xyz', 'viewdir'])
