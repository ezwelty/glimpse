IMG_DIR = "/volumes/science-b/data/columbia/timelapse"
IMG_SIZE = 0.25
SVG_KEYS = ['gcp', 'horizon', 'coast', 'terminus', 'moraines']

# ---- Batch calibrate and orient cameras ---- #

svg_paths = glob.glob("svg/AK01b*.svg")
for path in svg_paths:
    calibration = cgcalib.load_calibration(path, station=True, camera=True)
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
    img.cam.write("images/" + basename + ".json",
        attributes=['xyz', 'viewdir', 'fmm', 'cmm', 'k', 'p', 'sensorsz'])
    fig = matplotlib.pyplot.figure(figsize=tuple(img.cam.imgsz / 72), frameon=False)
    ax = matplotlib.pyplot.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    img.plot()
    image_model.plot()
    img.set_plot_limits()
    matplotlib.pyplot.savefig("images/" + basename + "-markup.jpg", dpi=72)
    matplotlib.pyplot.close()

# ---- Ortho projections ---- #

DEM_DIR = "/volumes/science-b/data/columbia/dem/"
ORTHO_DIR = "/volumes/science-b/data/columbia/ortho/"
GRID_SIZE = 5
IMG_SIZE = 0.25

img_paths = glob.glob("images/AK01b*.json")
img_dates = [re.findall("_([0-9]{8})_", path)[0] for path in img_paths]
for i, date in enumerate(img_dates):
    dem_paths = glob.glob(DEM_DIR + date + "*.tif")
    ortho_paths = glob.glob(ORTHO_DIR + date + "*.tif")
    if dem_paths and ortho_paths:
        img = image.Image(cgcalib.find_image(img_paths[i], root=IMG_DIR), cam=img_paths[i])
        img.cam.resize(IMG_SIZE)
        # Prepare dem
        dem = DEM.DEM.read(dem_paths[-1])
        smdem = dem.copy()
        smdem.resize(smdem.d[0] / GRID_SIZE)
        smdem.crop(zlim=[1, np.inf])
        # Prepare ortho
        ortho = DEM.DEM.read(ortho_paths[-1])
        smortho = ortho.copy()
        smortho.resize(smortho.d[0] / DEM_GRID_SIZE)
        smortho.resample(smdem, method="linear")
        # Save results as images
        basename = os.path.splitext(img_paths[i])[0]
        # (original)
        img.write(basename + "-original.jpg")
        # (projected into distorted camera)
        I = cgcalib.dem_to_image(img.cam, smdem, smortho.Z) # mask=smdem.visible(img.cam.xyz)
        I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
        I = (255 * (I / I.max() - I.min() / I.max()))
        img.write(basename + "-distorted.jpg", I.astype(np.uint8))
        # (projected into ideal but oriented camera)
        img.cam.idealize()
        img.cam.f = img.exif.fmm * img.cam.imgsz / img.cam.sensorsz
        controls = cgcalib.svg_controls(img, "svg/" + os.path.basename(basename) + ".svg", keys=SVG_KEYS)
        viewdir_model = optimize.Cameras(img.cam, controls)
        viewdir = viewdir_model.fit()
        img.cam.viewdir = viewdir
        I = cgcalib.dem_to_image(img.cam, smdem, smortho.Z) # mask=smdem.visible(img.cam.xyz)
        I[np.isnan(I)] = np.nanmax(I) / 2 # fill holes with grey
        I = (255 * (I / I.max() - I.min() / I.max()))
        img.write(basename + "-oriented.jpg", I.astype(np.uint8))
        img.cam.reset()
