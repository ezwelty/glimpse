import cg
from cg import (glimpse, glob)
from glimpse.imports import (pandas, scipy, sharedmem, gdal, sys, os, re, np,
    matplotlib, datetime, cv2)
import timeit
root = '/volumes/science-b/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
glimpse.config.set_sharedmem_backend('thread')

# ---- Constants ----

color = False
img_size = 1
grid_size = 2
circle_radius_default = 50
circle_radius = {
    'AKJNC_20120813_205325': 400,
    'AK09b_20090827_200153': 200
}
viewshed_scale = 0.25
parallel = 4
scale = 2
scale_limits = (0.05, 20)
images = (
    'AK01_20070922_230210',
    'AK01b_20080811_193452',
    'AK01b_20090827_202939',
    # NOTE: Hillshade only
    # 'AK01b_20100525_200710',
    # 'AK01b_20100602_200707',
    'AK03b_20080811_191645',
    'AK09_20090803_200320',
    'AK09_20090827_200153',
    'AK09b_20090827_200153',
    # NOTE: Hillshade only
    # 'AK09b_20100525_200819',
    # 'AK09b_20100602_200818',
    # NOTE: 20100720 IfSAR is a +- 10 day mosaic. Radar hard to interpret.
    # 'AK09b_20100720_192819',
    'AK10_20090803_195410',
    'AK10_20090827_192655',
    # NOTE: 20120605 ortho shifted from Aerometric reference
    'AK10b_20120605_203759',
    'AKJNC_20120605_223513',
    'AKJNC_20120813_205325',
    'AKJNC_20121001_155719',
    # NOTE: Hillshade only
    # 'AKST03A_20100525_224800',
    # 'AKST03A_20100602_224800',
    # 'AKST03B_20100525_224800',
    # 'AKST03B_20100602_224800'
    # NOTE: 20100720 IfSAR is a +- 10 day mosaic. Radar hard to interpret.
    # 'AKST03A_20100720_184800',
    'CG04_20040707_200052',
    # NOTE: 20050811 ortho messed up
    # 'CG05_20050811_130000',
    # 'CG05_20050811_190000',
    # NOTE: More clarity than CG05_20050827_190000
    'CG05_20050826_190000',
    'CG06_20060712_195953',
    'CG06_20060727_195951'
)
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))

# ---- Select DEMs and Orthos ----

# Prepare DEMs
dem_paths = glob.glob(os.path.join(root, 'dem-aerometric', 'data', '*.tif'))
dem_paths += glob.glob(os.path.join(root, 'dem-arcticdem', 'data', '*.tif'))
dem_paths += glob.glob(os.path.join(root, 'dem-ifsar', 'data', '*.tif'))
dem_dates = [datetime.datetime.strptime(re.findall(r'([0-9]{8})', path)[0], '%Y%m%d')
    for path in dem_paths]

# Prepare orthophotos
ortho_paths = glob.glob(os.path.join(root, 'ortho', '*.tif'))
ortho_paths += glob.glob(os.path.join(root, 'ortho-ifsar', 'data', '*.tif'))
ortho_dates = [datetime.datetime.strptime(re.findall(r'([0-9]{8})', path)[0], '%Y%m%d')
    for path in ortho_paths]

# ---- Control synths (ideal camera) ----

for image in images:
    print(image)
    start = timeit.default_timer()
    basename = os.path.join('svg-synth', image)
    if os.path.isfile(basename + '-synth.JPG'):
        continue
    # Load image
    img_path = cg.find_image(image)
    cam_args = cg.load_calibrations(image,
        station_estimate=True, station=True, merge=True, file_errors=False)
    img = glimpse.Image(img_path, cam=cam_args)
    img.cam.resize(img_size)
    # Select nearest dem and ortho
    img_date = datetime.datetime.strptime(cg.parse_image_path(image)['date_str'], '%Y%m%d')
    i_dem = np.argmin(np.abs(np.asarray(dem_dates) - img_date))
    i_ortho = np.argmin(np.abs(np.asarray(ortho_dates) - img_date))
    dem_path = dem_paths[i_dem]
    ortho_path = ortho_paths[i_ortho]
    # Load raster metadata
    dem_grid = glimpse.Grid.read(dem_path, d=grid_size)
    ortho_grid = glimpse.Grid.read(ortho_path, d=grid_size)
    # Intersect bounding boxes
    cam_box = img.cam.viewbox(50e3)[[0, 1, 3, 4]]
    box = glimpse.helpers.intersect_boxes(np.row_stack((
        cam_box, dem_grid.box2d, ortho_grid.box2d)))
    # Read dem and ortho
    dem = glimpse.DEM.read(dem_path, xlim=box[0::2], ylim=box[1::2], d=grid_size)
    dem.crop(zlim=(0.1, np.inf))
    radius = circle_radius.get(image, circle_radius_default)
    dem.fill_circle(center=img.cam.xyz, radius=radius)
    nbands = gdal.Open(ortho_path).RasterCount
    bands = []
    for i in range(nbands):
        bands.append(glimpse.DEM.read(ortho_path, band=i + 1, d=grid_size,
            xlim=box[0::2], ylim=box[1::2]).Z)
    orthoZ = np.dstack(bands).astype(float)
    if not color:
        orthoZ = np.atleast_3d(glimpse.helpers.rgb_to_gray(orthoZ))
    orthoZ[orthoZ == 0] = np.nan
    # HACK: Clip dem and ortho to same size relative to x, y min
    ij = np.minimum(dem.shape, orthoZ.shape[0:2])
    dem = dem[(dem.shape[0] - ij[0]):, :ij[1]]
    orthoZ = orthoZ[(orthoZ.shape[0] - ij[0]):, :ij[1], :]
    # Compute mask
    if viewshed_scale != 1:
        smdem = dem.copy()
        smdem.resize(viewshed_scale)
    else:
        smdem = dem
    mask = glimpse.DEM(Z=smdem.viewshed(img.cam.xyz), x=dem.xlim, y=dem.ylim)
    if viewshed_scale != 1:
        mask.resample(dem)
    mask.Z = mask.Z.astype(bool)
    mask.Z &= ~np.isnan(orthoZ[:, :, 0])
    # Copy to shared memory
    if parallel:
        dem.Z = sharedmem.copy(dem.Z)
        orthoZ = sharedmem.copy(orthoZ)
        mask.Z = sharedmem.copy(mask.Z)
    # Project onto image
    aggregate = glimpse.helpers.merge_dicts(
        {i: np.mean for i in range(orthoZ.shape[2])},
        {orthoZ.shape[2]: [np.mean, np.std]})
    I = img.cam.project_dem(
        dem=dem, values=orthoZ, mask=mask.Z,
        tile_size=(256, 256), tile_overlap=(1, 1),
        scale=scale, scale_limits=scale_limits,
        parallel=parallel, correction=True,
        return_depth=True, aggregate=aggregate)
    # Split out depth
    I, depth, depth_sigma = I[:, :, :-2], I[:, :, -2], I[:, :, -1]
    # Process grayscale image
    if I.shape[2] == 1:
        I = I[:, :, 0]
    else:
        I = glimpse.helpers.rgb_to_gray(I)
    I = glimpse.helpers.normalize_range(I, interval=np.uint8)
    nanI = np.isnan(I)
    I[I == 127] = 126
    I[nanI] = 127
    # Equalize images
    I = clahe.apply(I.astype(np.uint8))
    I_ori = clahe.apply(glimpse.helpers.rgb_to_gray(img.read()).astype(np.uint8))
    # Write synthetic image
    I[I == 127] = 126
    I[nanI] = 127
    img.write(basename + '-synth.JPG', I, quality=95)
    # Write synthetic camera
    img.cam.write(
        path=basename + '-synth.json',
        attributes=('xyz', 'viewdir', 'fmm', 'cmm', 'k', 'p', 'sensorsz'),
        flat_arrays=True, indent=4)
    # Write copy of original image
    img.cam.resize()
    img.write(basename + '.JPG', I_ori, quality=95)
    # Write svg
    svg_path = basename + '.svg'
    if not os.path.isfile(svg_path):
        size = img.cam.imgsz
        svg_size = size * 0.25
        synth_size = np.array(I.shape[0:2][::-1])
        xml = glimpse.svg._svg(
            glimpse.svg._image(size=size, scale=svg_size / size, path=image + '.JPG'),
            glimpse.svg._image(size=synth_size, scale=svg_size / synth_size, path=image + '-synth.JPG'),
            id='svg', size=svg_size
        )
        glimpse.svg._write_svg(xml, path=svg_path, pretty_print=True)
    # Write depth map
    img.write(basename + '-depth.tif', depth)
    img.write(basename + '-depth_stderr.tif', depth_sigma)
    print(timeit.default_timer() - start)
