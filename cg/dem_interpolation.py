import cg
from cg import glimpse
from glimpse.imports import (datetime, matplotlib, np, os, shapely, scipy, pandas)

root = '/volumes/science-b/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
cg.KEYPOINT_PATH = os.path.join(root, 'timelapse-keypoints')
cg.MATCHES_PATH = os.path.join(root, 'timelapse-matches')
cg.DEM_PATHS = (
    os.path.join(root, 'dem'),
    os.path.join(root, 'dem-tandem/data'))
BENCHMARK_PATH = os.path.join(root, 'optical-surveys-2005/data/positions.csv')
BENCHMARK_TRACK = 1

target_datestr = '2005-06-06' # Matches a terminus position and near benchmark
target_datetime = datetime.datetime.strptime(target_datestr, '%Y-%m-%d')

# ---- Load dems ----

# Prepare interpolant
dem_interpolant = cg.load_dem_interpolant(
    d=20, zlim=(1, np.inf),
    fun=glimpse.Raster.fill_crevasses, mask=lambda x: ~np.isnan(x))

# Read nearest DEMs (left, right)
ij = dem_interpolant.nearest(target_datetime)
dems = [dem_interpolant.read(k) for k in ij]
dems[1].resample(dems[0])

# --- Load termini ----

termini_json = glimpse.helpers.read_geojson(
    path=os.path.join(cg.CG_PATH, 'geojson', 'termini.geojson'),
    crs=32606, key='date')
termini_datestr = [dem.datetime.strftime('%Y-%m-%d') for dem in dems] + [target_datestr]
termini = [termini_json['features'][date]['geometry']['coordinates'] for date in termini_datestr]

# ---- Load centerline ----

centerline_json = glimpse.helpers.read_geojson(
    path=os.path.join(cg.CG_PATH, 'geojson', 'centerline.geojson'),
    crs=32606)
# Reverse direction so that upglacier is + distance
centerline = centerline_json['features'][0]['geometry']['coordinates'][::-1, :]

# ---- Build streamline grid ----
# m: measure along line, d: signed distance from line

size = 200
# Here, m is relative to terminus m
m = np.arange(0, 5000 + size, size)
d = np.arange(-1000, 1000 + size, size)
M, D = np.meshgrid(m, d)

# ---- Transform termini to streamline ----

d_bins = np.append(d[0] - size / 2, d + size / 2, )
termini_md = [glimpse.helpers.cartesian_to_polyline(term, centerline) for term in termini]
termini_md_origin = [term[:, 0].max() for term in termini_md]
termini_md_binned = [scipy.stats.binned_statistic(
    x=term[:, 1], values=term[:, 0], statistic='mean', bins=d_bins)[0]
    for term in termini_md]

# Plot termini
matplotlib.pyplot.figure()
for term in termini_md:
    matplotlib.pyplot.plot(term[:, 1], term[:, 0])
matplotlib.pyplot.gca().set_aspect(1)
matplotlib.pyplot.legend(termini_datestr)

# --- Sample nearest DEMs ----

dems_md = []
for i, dem in enumerate(dems):
    # Shift columns of grid up/down centerline relative to terminus
    pts_md = glimpse.helpers.grid_to_points((M + termini_md_binned[i][:, None], D))
    # NOTE: Can also shift grid by a scaler
    # pts_md = glimpse.helpers.grid_to_points((M + termini_md_origin[i], D))
    pts = glimpse.helpers.polyline_to_cartesian(pts_md, centerline)
    z = dem.sample(pts).reshape((len(d), len(m)))
    dems_md.append(glimpse.Raster(z, x=m, y=d, datetime=dem.datetime))

# Plot sampling points
i = 1
matplotlib.pyplot.figure()
dems[i].plot(dems[i].hillshade())
matplotlib.pyplot.plot(centerline[:, 0], centerline[:, 1], color='red')
matplotlib.pyplot.scatter(pts[:, 0], pts[:, 1], color='red', marker='.')

# Plot DEM difference (cartesian)
matplotlib.pyplot.figure()
dz = dems[1].Z - dems[0].Z
matplotlib.pyplot.imshow(dz, vmin=-30, vmax=30)
matplotlib.pyplot.colorbar()

# Plot DEM difference (streamline)
matplotlib.pyplot.figure()
dz = dems_md[1].Z - dems_md[0].Z
matplotlib.pyplot.imshow(dz)
matplotlib.pyplot.colorbar()

# --- Interpolate middle DEM ----

dt = dems[1].datetime - dems[0].datetime
dt_fraction = (target_datetime - dems[0].datetime) / dt

# Streamline
dz = dems_md[1].Z - dems_md[0].Z
target_dem_md = glimpse.Raster(
    dems_md[0].Z + dz * dt_fraction,
    x=m, y=d, datetime=target_datetime)

# Cartesian
dz = dems[1].Z - dems[0].Z
target_dem = glimpse.Raster(
    dems[0].Z + dz * dt_fraction,
    x=dems[0].x, y=dems[0].y, datetime=target_datetime)

# ---- Compare to benchmark ----

# Read benchmark (t, x, y, z)
positions = pandas.read_csv(BENCHMARK_PATH, parse_dates=['t'])
benchmark = positions[positions.track == BENCHMARK_TRACK]

# Resample to same streamline grid
xyz = benchmark.loc[:, ('x', 'y', 'z')].as_matrix()
xy_md = glimpse.helpers.cartesian_to_polyline(xyz[:, 0:2], centerline)
xy_md[:, 0] -= np.interp(xy_md[:, 1], d, termini_md_binned[2])
# xy_md[:, 0] -= termini_md_origin[2]

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(xy_md[:, 0], dems[0].sample(xyz[:, 0:2]))
matplotlib.pyplot.plot(xy_md[:, 0], xyz[:, 2])
matplotlib.pyplot.plot(xy_md[:, 0], target_dem.sample(xyz[:, 0:2]))
matplotlib.pyplot.plot(xy_md[:, 0], target_dem_md.sample(xy_md))
matplotlib.pyplot.plot(xy_md[:, 0], dems[1].sample(xyz[:, 0:2]))
matplotlib.pyplot.legend((
    '2004-07-07: DEM',
    '2005-06-09: Optical survey',
    '2005-06-09: DEM interpolation (cartesian)',
    '2005-06-06: DEM interpolation (streamline)',
    '2005-08-11: DEM'))
matplotlib.pyplot.xlabel('Distance upstream from terminus (m)')
matplotlib.pyplot.ylabel('Elevation (m)')
matplotlib.pyplot.tight_layout()
