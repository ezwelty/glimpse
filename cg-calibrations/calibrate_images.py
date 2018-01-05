import glob
import os
import sys
import re
import matplotlib
import cgcalib
sys.path.append('../')
import image
import optimize
import dem as DEM

IMG_DIR = "/volumes/science-b/data/columbia/timelapse"
IMG_SIZE = 0.5
SVG_KEYS = ['gcp', 'horizon', 'coast', 'terminus', 'moraines']

# ---- Batch calibrate and orient cameras ---- #

STATION = 'AK12'

svg_paths = glob.glob("svg/" + STATION + "_*.svg")
for path in svg_paths:
    for suffix in ['']:
        ids = cgcalib.parse_image_path(path)
        try:
            calibration = cgcalib.load_calibration(path,
                station=ids['station'] + suffix,
                camera=ids['camera'] + suffix)
        except IOError:
            break
        img_path = cgcalib.find_image(path, root=IMG_DIR)
        img = image.Image(img_path, cam=calibration)
        img.cam.resize(IMG_SIZE)
        controls = cgcalib.svg_controls(img, path, keys=SVG_KEYS)
        image_model = optimize.Cameras(
            cams=img.cam,
            controls=controls,
            cam_params=dict(viewdir=True))
        image_fit = image_model.fit(full=True)
        image_model.set_cameras(image_fit.params)
        basename = os.path.splitext(os.path.basename(img.path))[0]
        img.cam.write("images/" + basename + suffix + ".json",
            attributes=['xyz', 'viewdir', 'fmm', 'cmm', 'k', 'p', 'sensorsz'])
        fig = matplotlib.pyplot.figure(figsize=tuple(img.cam.imgsz / 72), frameon=False)
        ax = matplotlib.pyplot.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        img.plot()
        image_model.plot()
        img.set_plot_limits()
        matplotlib.pyplot.savefig("images/" + basename + suffix + "-markup.jpg", dpi=72)
        matplotlib.pyplot.close()

# ---- Ortho projections ---- #

DEM_DIR = "/volumes/science-b/data/columbia/dem/"
ORTHO_DIR = "/volumes/science-b/data/columbia/ortho/"
DEM_GRID_SIZE = 5
IMG_SIZE = 0.25

img_paths = glob.glob("images/" + STATION + "*.json")
img_dates = [re.findall("_([0-9]{8})_", path)[0] for path in img_paths]
previous_date = None
for i, date in enumerate(img_dates):
    dem_paths = glob.glob(DEM_DIR + date + "*.tif")
    ortho_paths = glob.glob(ORTHO_DIR + date + "*.tif")
    if dem_paths and ortho_paths:
        img = image.Image(cgcalib.find_image(img_paths[i], root=IMG_DIR), cam=img_paths[i])
        img.cam.resize(IMG_SIZE)
        if date != previous_date:
            # Prepare dem
            dem = DEM.DEM.read(dem_paths[-1])
            smdem = dem.copy()
            smdem.resize(smdem.d[0] / DEM_GRID_SIZE)
            smdem.crop(zlim=[1, np.inf])
            # FIXME: DEM.visible() not working from inside NAN
            smdem.fill_circle(center=img.cam.xyz, radius=100, value=0)
            # Prepare ortho
            ortho = DEM.DEM.read(ortho_paths[-1])
            smortho = ortho.copy()
            smortho.resize(smortho.d[0] / DEM_GRID_SIZE)
            smortho.resample(smdem, method="linear")
        # Save results as images
        basename = os.path.splitext(img_paths[i])[0]
        # (original)
        img.write(basename + "-original.jpg", I=img.read())
        # (projected into distorted camera)
        mask = smdem.visible(img.cam.xyz)
        I = cgcalib.dem_to_image(img.cam, smdem, smortho.Z, mask=mask)
        I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
        I = (255 * (I / I.max() - I.min() / I.max()))
        img.write(basename + "-distorted.jpg", I.astype(np.uint8))
        # (projected into ideal camera)
        img.cam.idealize()
        img.cam.f = img.exif.fmm * img.cam.imgsz / img.cam.sensorsz
        controls = cgcalib.svg_controls(img, "svg/" + cgcalib.parse_image_path(basename)['basename'] + ".svg", keys=SVG_KEYS)
        ideal_model = optimize.Cameras(img.cam, controls)
        img.cam.viewdir = ideal_model.fit()
        I = cgcalib.dem_to_image(img.cam, smdem, smortho.Z, mask=mask)
        I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
        I = (255 * (I / I.max() - I.min() / I.max()))
        img.write(basename + "-oriented.jpg", I.astype(np.uint8))
        img.cam.reset()
        # Cache DEM and Ortho
        previous_date = date
