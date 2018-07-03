import cg
from cg import glimpse
import glimpse.unumpy as unp
import scipy.stats
from glimpse.imports import (datetime, np, os, shapely, re, matplotlib, collections)
import glob
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')

surface_sigma = 3 # surface roughness
dem_args = dict(
    d=20,
    zlim=(1, np.inf),
    fun=glimpse.Raster.fill_crevasses,
    mask=lambda x: ~np.isnan(x), fill=True,
    return_sigma=True)
grid_step = (100, 100) # m
max_distance = 10e3 # m
bed_sigma = 20 # m
density_water = 1025 # kg / m^3
density_ice = 916.7 # kg / m^3

# ---- Load first image from each observer station ----
# images

json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)
start_images = []
progress = glimpse.helpers._progress_bar(max=len(json))
for observers in json:
    starts = []
    for station, basenames in observers.items():
        ids = cg.parse_image_path(basenames[0], sequence=True)
        cam_args = cg.load_calibrations(station=station, camera=ids['camera'],
            image=basenames[0], viewdir=basenames[0], merge=True,
            file_errors=False)
        path = cg.find_image(basenames[0])
        starts.append(glimpse.Image(path, cam=cam_args))
    start_images.append(tuple(starts))
    progress.next()

# ---- Build DEM template ----

stations = set([station for x in json for station in x])
station_xy = np.vstack([f['geometry']['coordinates'][:, 0:2]
    for station, f in cg.Stations().items()
    if station in stations])
box = glimpse.helpers.bounding_box(cg.Glacier())
XY = glimpse.helpers.box_to_grid(box, step=(dem_args['d'], dem_args['d']),
    snap=(0, 0), mode='grid')
xy = glimpse.helpers.grid_to_points(XY)
distances = glimpse.helpers.pairwise_distance(xy, station_xy, metric='euclidean')
selected = distances.min(axis=1) < max_distance
box = glimpse.helpers.bounding_box(xy[selected]) + 0.5 * np.array([-1, -1, 1, 1]) * dem_args['d']
shape = np.diff([box[1::2], box[0::2]], axis=1) / dem_args['d']
dem_template = glimpse.Raster(np.ones(shape.astype(int).ravel(), dtype=bool),
    x=box[0::2], y=box[1::2][::-1])
dem_points = glimpse.helpers.grid_to_points((dem_template.X, dem_template.Y))

# ---- Select DEMs ----

dem_sigmas = {
    'aerometric': 1.5,
    'ifsar': 1.5 + 0.5, # additional time uncertainty
    'arcticdem': 3,
    'tandem': 3 # after bulk corrections
}
dem_keys = [
    ('20040618', 'aerometric'),
    ('20040707', 'aerometric'),
    ('20050811', 'aerometric'),
    ('20050827', 'aerometric'),
    ('20060712', 'aerometric'),
    ('20060727', 'aerometric'),
    ('20070922', 'aerometric'),
    ('20080811', 'aerometric'),
    ('20090803', 'aerometric'),
    ('20090827', 'aerometric'),
    ('20100525', 'aerometric'),
    ('20100602', 'aerometric'),
    ('20100720', 'ifsar'), # +- 10 days mosaic
    ('20100906', 'arcticdem'),
    ('20110618', 'tandem'),
    ('20110721', 'tandem'),
    ('20110812', 'tandem'),
    ('20110903', 'tandem'),
    ('20111211', 'tandem'),
    ('20120102', 'tandem'),
    ('20120204', 'tandem'),
    ('20120308', 'tandem'),
    ('20120329', 'arcticdem'),
    ('20120507', 'arcticdem'),
    ('20120617', 'arcticdem'),
    ('20120717', 'arcticdem'),
    ('20120813', 'arcticdem'),
    ('20121012', 'arcticdem'),
    ('20121123', 'arcticdem'),
    ('20130326', 'arcticdem'),
    ('20130610', 'arcticdem'),
    ('20130712', 'arcticdem'),
    ('20131119', 'arcticdem'),
    ('20140417', 'tandem'),
    # ('20140419', 'arcticdem'), # Coverage too small
    ('20140531', 'tandem'),
    ('20140622', 'tandem'),
    ('20140703', 'tandem'),
    ('20150118', 'arcticdem'),
    ('20150227', 'arcticdem'),
    ('20150423', 'arcticdem'),
    ('20150527', 'arcticdem'),
    ('20150801', 'arcticdem'),
    ('20150824', 'arcticdem'),
    ('20150930', 'arcticdem'),
    ('20160614', 'arcticdem'),
    ('20160820', 'arcticdem')
]
dem_keys.sort(key=lambda x: x[0])

# ---- Build DEM interpolant ----

# # Compute means and sigmas
# means, sigmas = [], []
# for datestr, demtype in dem_keys:
#     print(datestr, demtype)
#     path = os.path.join(root, 'dem-' + demtype, 'data', datestr + '.tif')
#     # HACK: Aerial and satellite imagery taken around local noon (~ 22:00 UTC)
#     t = datetime.datetime.strptime(datestr + str(22), '%Y%m%d%H')
#     dem = glimpse.Raster.read(path,
#         xlim=dem_template.xlim + np.array((-1, 1)) * dem_args['d'],
#         ylim=dem_template.ylim + np.array((1, -1)) * dem_args['d'],
#         d=dem_args['d'])
#     dem.crop(zlim=dem_args['zlim'])
#     z = dem.sample(dem_points, order=1, bounds_error=False).reshape(dem_template.shape)
#     dem = glimpse.Raster(z, x=dem_template.xlim, y=dem_template.ylim, datetime=t)
#     dem_args['fun'](self=dem, mask=dem_args['mask'], fill=dem_args['fill'])
#     # HACK: Embed dem type for later retrieving correct terminus
#     dem.type = demtype
#     means.append(dem)
#     sigma = dem_sigmas[demtype] + surface_sigma
#     sigmas.append(sigma)

# Initialize interpolant
# dem_interpolant = glimpse.RasterInterpolant(means=means, sigmas=sigmas,
#     x=[dem.datetime for dem in means])

# Cache glacier polygon for each DEM
# for dem in dem_interpolant.means:
#     dem.polygon = cg.load_glacier_polygon(t=dem.datetime, demtype=dem.type)

# Write to / Read from file
# glimpse.helpers.write_pickle(dem_interpolant, 'dem_interpolant.pkl')
dem_interpolant = glimpse.helpers.read_pickle('dem_interpolant.pkl')

# ---- Load canonical velocities ----
# vx, vx_sigma, vy, vy_sigma

names = 'vx', 'vx_stderr', 'vy', 'vy_stderr'
vx, vx_sigma, vy, vy_sigma = [glimpse.Raster.read(
    os.path.join('velocity', name + '.tif'))
    for name in names]

# ---- Load canonical bed ----
# bed

bed = glimpse.Raster.read('bed.tif')

# ---- Build track template ----

grid = glimpse.helpers.box_to_grid(dem_template.box2d, step=grid_step,
    snap=(0, 0))
track_template = glimpse.Raster(np.ones(grid[0].shape, dtype=bool),
    x=grid[0], y=grid[1][::-1])
xy = glimpse.helpers.grid_to_points((track_template.X, track_template.Y))
selected = glimpse.helpers.points_in_polygon(xy, cg.Glacier())
# Filter by velocity availability
# NOTE: Use nearest to avoid NaN propagation (and on same grid anyway)
selected &= ~np.isnan(vx.sample(xy, order=0))
mask = selected.reshape(track_template.shape)
track_points = xy[mask.ravel()]
track_ids = np.ravel_multi_index(np.nonzero(mask), track_template.shape)

# Write to file
track_template.Z &= mask
track_template.Z = track_template.Z.astype(np.uint8)
track_template.write(os.path.join('points', 'template.tif'), crs=32606)

# ---- For each observer ----

for obs in range(len(start_images)):
    print(obs)
    images = start_images[obs]
    # Check within DEM interpolant bounds
    t = np.min([img.datetime for img in images])
    if t > np.max(dem_interpolant.x):
        raise ValueError('Images begin after last DEM')
    # -- Load DEM --
    dem, dem_sigma = dem_interpolant(t, return_sigma=True)
    # Compute union of camera view boxes
    boxes = []
    for img in images:
        dxyz = img.cam.invproject(img.cam.edges(step=10))
        scale = max_distance / np.linalg.norm(dxyz[:, 0:2], axis=1)
        xyz = np.vstack((img.cam.xyz, img.cam.xyz + dxyz * scale.reshape(-1, 1)))
        boxes.append(glimpse.helpers.bounding_box(xyz))
    box = glimpse.helpers.union_boxes(boxes)
    # Crop DEM
    dem.crop(xlim=box[0::3], ylim=box[1::3])
    dem_sigma.crop(xlim=box[0::3], ylim=box[1::3])
    dem.crop_to_data()
    dem_sigma.crop_to_data()
    # Save DEM to file
    basename = os.path.join('dems', t.strftime('%Y%m%d') + '-' + str(obs))
    dem.write(basename + '.tif', nan=-9999, crs=32606)
    dem_sigma.write(basename + '_stderr.tif', nan=-9999, crs=32606)
    # Mask camera foreground
    for img in images:
        dem.fill_circle(center=img.cam.xyz, radius=400, value=np.nan)
    # --- Load track points and observer mask ---
    # Load glacier polygon
    ij = dem_interpolant.nearest(t)
    polygons = [glimpse.helpers.box_to_polygon(dem.box2d)]
    polygons += [dem_interpolant.means[i].polygon for i in ij]
    polygons += [cg.load_glacier_polygon(t)]
    polygon = cg.intersect_polygons(polygons)
    observer_mask = cg.select_track_points(track_points, images=images,
        polygon=polygon, dem=dem, max_distance=max_distance)
    selected = np.count_nonzero(observer_mask, axis=1) > 0
    # xy (n, ), observer_mask (n, o)
    xy, obsmask, ids = track_points[selected], observer_mask[selected], track_ids[selected]
    # ---- Compute motion parameters ----
    n = len(xy)
    # vxyz | vxyz_sigma
    vxyz = np.ones((n, 3), dtype=float)
    vxyz_sigma = np.ones((n, 3), dtype=float)
    # (x, y): Sample from velocity grids
    vxyz[:, 0] = vx.sample(xy, order=0)
    vxyz[:, 1] = vy.sample(xy, order=0)
    vxyz_sigma[:, 0] = vx_sigma.sample(xy, order=0)
    vxyz_sigma[:, 1] = vy_sigma.sample(xy, order=0)
    # (z): Compute by integrating dz/dx and dz/dy over vx and vy
    rowcol = dem.xy_to_rowcol(xy, snap=True)
    dz = np.dstack(dem.gradient())[rowcol[:, 0], rowcol[:, 1], :]
    # sigma for dz/dx * vx + dz/dy * vy, assume zi, zj are fully correlated
    udz = unp.uarray(dz, sigma=None)
    uvxy = unp.uarray(vxyz[:, 0:2], vxyz_sigma[:, 0:2])
    vxyz[:, 2], vxyz_sigma[:, 2] = (
        udz[:, 0] * uvxy[:, 0] + udz[:, 1] * uvxy[:, 1]).tuple()
    # Determine probability of flotation
    # vz_sigma, az_sigma: Typically very small, but large if glacier floating
    zw = unp.uarray(16, 2) # m, mean HAE
    Zs = unp.uarray(dem.sample(xy, order=1), dem_sigma.sample(xy, order=1)) # m
    Zb = unp.uarray(bed.sample(xy, order=1), bed_sigma) # m
    hmax = Zs - Zb
    hf = (zw - Zb) * (density_water / density_ice)
    # probability hf > hmax
    # https://math.stackexchange.com/a/40236
    dh = hf - hmax
    flotation = 1 - scipy.stats.norm().cdf(-dh.mean / dh.sigma)
    # ---- Save parameters to file ----
    # ids, xy, observer_mask
    # vxyz, vxyz_sigma (based on long term statistics only)
    # flotation: Probability of flotation
    glimpse.helpers.write_pickle(
        dict(ids=ids, xy=xy, observer_mask=obsmask, vxyz=vxyz,
        vxyz_sigma=vxyz_sigma, flotation=flotation),
        path=os.path.join('points', t.strftime('%Y%m%d') + '-' + str(obs) + '.pkl'))

# ---- Plotting ----

# # Plot (motion)
# matplotlib.pyplot.figure()
# dem.plot(dem.hillshade(), cmap='gray')
# dem.set_plot_limits()
# matplotlib.pyplot.plot(polygon[:, 0], polygon[:, 1], color='gray')
# matplotlib.pyplot.quiver(xy[:, 0], xy[:, 1], vxyz[:, 0], vxyz[:, 1],
#     angles='xy', color='red')
# # matplotlib.pyplot.quiver(xy[:, 0], xy[:, 1], vxyz_sigma[:, 0], vxyz_sigma[:, 1],
# #     angles='xy', color='red')
# # raster template
# temp = track_template.copy()
# temp.Z = np.full(temp.shape, np.nan)
# # plot data as raster
# r = temp.copy()
# r.Z.flat[ids] = flotation
# # r.Z.flat[ids] = np.hypot(vxyz[:, 0], vxyz[:, 1])
# # r.Z.flat[ids] = np.hypot(vxyz_sigma[:, 0], vxyz_sigma[:, 1])
# r.plot()
# # r.plot(vmin=0, vmax=2, cmap='bwr')
# matplotlib.pyplot.colorbar()

# # Plot (map)
# matplotlib.pyplot.figure()
# dem.plot(dem.hillshade(), cmap='gray')
# for i, obs in enumerate(observers):
#     viewpoly = obs.images[0].cam.viewpoly(max_depth, step=100)[:, 0:2]
#     matplotlib.pyplot.plot(viewpoly[:, 0], viewpoly[:, 1], color='black')
#     matplotlib.pyplot.plot(xy[mask[:, i], 0], xy[mask[:, i], 1],
#         marker='.', linestyle='none')
# matplotlib.pyplot.savefig('temp.png', dpi=100)

# # Plot (image)
# for i, obs in enumerate(observers):
#     matplotlib.pyplot.figure()
#     xyz = np.column_stack((xy[mask[:, i]], dem.sample(xy[mask[:, i]])))
#     uv = obs.images[0].cam.project(xyz, correction=True)
#     obs.images[0].plot()
#     matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], color='red', marker='.',
#         linestyle='none')
#     matplotlib.pyplot.savefig('temp-' + str(i) + '.png', dpi=100)
