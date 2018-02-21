import sys
sys.path.insert(0, '..')
import glimpse
from glimpse.imports import (os, re, np, matplotlib)
import glob
import cgcalib

IMG_DIR = "/volumes/science-b/data/columbia/timelapse"
IMG_SIZE = 1
SVG_KEYS = ['gcp', 'horizon', 'coast', 'terminus', 'moraines']

# ---- Batch calibrate and orient cameras ---- #

# STATION = 'AKJNC'
# SVG_KEYS = ['gcp', 'horizon', 'coast', 'moraines']
STATION = 'AK01b'
SVG_KEYS = ['gcp', 'horizon', 'terminus', 'moraines']

svg_paths = glob.glob(os.path.join('svg', STATION + '_*.svg'))
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
        img = glimpse.Image(img_path, cam=calibration)
        img.cam.resize(IMG_SIZE)
        controls = cgcalib.svg_controls(img, path, keys=SVG_KEYS)
        image_model = glimpse.optimize.Cameras(
            cams=img.cam,
            controls=controls,
            cam_params=dict(viewdir=True))
        image_fit = image_model.fit(full=True)
        image_model.set_cameras(image_fit.params)
        img.cam.write(os.path.join('images', ids['basename'] + suffix + '.json'),
            attributes=['xyz', 'viewdir', 'fmm', 'cmm', 'k', 'p', 'sensorsz'])
        # Plot image with markup
        dpi = 100
        fig = matplotlib.pyplot.figure(figsize=tuple(img.cam.imgsz / dpi), dpi=dpi * 0.25, frameon=False)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis('off')
        img.plot()
        image_model.plot(
            lines_observed=dict(color='yellow', linewidth=3),
            lines_predicted=dict(color='red', linewidth=2))
        img.set_plot_limits()
        matplotlib.pyplot.savefig(os.path.join('images', ids['basename'] + suffix + '-markup.jpg'), dpi=dpi)
        matplotlib.pyplot.close()

# ---- Ortho projections ---- #

DEM_DIR = "/volumes/science-b/data/columbia/dem/"
ORTHO_DIR = "/volumes/science-b/data/columbia/ortho/"
DEM_GRID_SIZE = 5
IMG_SIZE = 0.25

img_paths = glob.glob("images/" + STATION + "*[0-9].json")
img_dates = [re.findall("_([0-9]{8})_", path)[0] for path in img_paths]
previous_date = None
for i, date in enumerate(img_dates):
    dem_paths = glob.glob(DEM_DIR + date + "*.tif")
    ortho_paths = glob.glob(ORTHO_DIR + date + "*.tif")
    if dem_paths and ortho_paths:
        img = glimpse.Image(cgcalib.find_image(img_paths[i], root=IMG_DIR), cam=img_paths[i])
        img.cam.resize(IMG_SIZE)
        viewbox = img.cam.viewbox(radius=30e3)
        if date != previous_date:
            # Prepare dem
            dem = glimpse.DEM.read(dem_paths[-1], d=DEM_GRID_SIZE, xlim=viewbox[0::3], ylim=viewbox[1::3])
            dem.crop(zlim=[1, np.inf])
            # FIXME: glimpse.DEM.visible() not working from inside NAN
            dem.fill_circle(center=img.cam.xyz, radius=500, value=0)
            # Prepare ortho
            ortho = glimpse.DEM.read(ortho_paths[-1], d=DEM_GRID_SIZE, xlim=viewbox[0::3], ylim=viewbox[1::3])
            ortho.resample(dem, method="linear")
        # Save results as images
        basename = os.path.splitext(img_paths[i])[0]
        # (original)
        img.write(basename + "-original.jpg", I=img.read())
        # (projected into distorted camera)
        mask = dem.visible2(img.cam.xyz)
        I = cgcalib.dem_to_image(img.cam, dem, ortho.Z, mask=mask)
        I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
        I = (255 * (I / I.max() - I.min() / I.max()))
        img.write(basename + "-distorted.jpg", I.astype(np.uint8))
        # (projected into ideal camera)
        img.cam.idealize()
        img.cam.f = img.exif.fmm * img.cam.imgsz / img.cam.sensorsz
        controls = cgcalib.svg_controls(img, "svg/" + cgcalib.parse_image_path(basename)['basename'] + ".svg", keys=SVG_KEYS)
        ideal_model = glimpse.optimize.Cameras(img.cam, controls)
        img.cam.viewdir = ideal_model.fit()
        I = cgcalib.dem_to_image(img.cam, dem, ortho.Z, mask=mask)
        I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
        I = (255 * (I / I.max() - I.min() / I.max()))
        img.write(basename + "-oriented.jpg", I.astype(np.uint8))
        img.cam.reset()
        # Cache DEM and Ortho
        previous_date = date
