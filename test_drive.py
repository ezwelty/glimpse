import glimpse
from glimpse.imports import (datetime, np, os)
import glob
import itertools
import os

# Required if numpy is built using OpenBLAS or MKL!
os.environ['OMP_NUM_THREADS']='1'

#glimpse.config.use_numpy_matmul(False)
# ---- Constants ----

DATA_DIR = 'data'
DEM_DIR = os.path.join(DATA_DIR, 'dem')
IMG_DIR = 'images'
CAM_DIR = 'images_json'
MAX_DEPTH = 30e3 # Camera world units (meters)
STATIONS = ('AK01b', 'AK10b')

# ---- Prepare Observers ----

start = datetime.datetime(2013, 6, 12, 20)
end = datetime.datetime(2013, 6, 18, 20)
observers = []
for station in STATIONS:
    station_dir = os.path.join(DATA_DIR, station)
    cam_paths = glob.glob(os.path.join(station_dir, CAM_DIR, '*.json'))
    cam_paths.sort()
    basenames = [glimpse.helpers.strip_path(path) for path in cam_paths]
    images = [glimpse.Image(
        path=os.path.join(station_dir, IMG_DIR, basename + '.JPG'),
        cam=os.path.join(station_dir, CAM_DIR, basename + '.json'))
        for basename in basenames]
    datetimes = np.array([img.datetime for img in images])
    inrange = np.logical_and(datetimes > start, datetimes < end)
    observers.append(glimpse.Observer(list(np.array(images)[inrange])))

# ---- Prepare DEM ----

boxes = [obs.images[0].cam.viewbox(MAX_DEPTH)
    for obs in observers]
box = glimpse.helpers.intersect_boxes(boxes)
paths = glob.glob(os.path.join(DEM_DIR, '*.tif'))
paths.sort()
path = paths[0]
dem = glimpse.Raster.read(path, xlim=box[0::3], ylim=box[1::3])
dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=True)

# ---- Prepare viewshed ----

for obs in observers:
    dem.fill_circle(obs.xyz, radius=100)
viewshed = dem.copy()
viewshed.Z = np.ones(dem.shape, dtype=bool)
for obs in observers:
    viewshed.Z &= dem.viewshed(obs.xyz)

# ---- Run Tracker ----

xy0 = np.array((4.988e5, 6.78186e6))
xy = xy0 + np.vstack([xy for xy in
    itertools.product(range(-400, 300, 50), range(-400, 300, 50))])
time_unit = datetime.timedelta(days=1)
motion_model = glimpse.CylindricalMotionModel(aUTz_sigma=(2.0/0.7,0.05,0.2))
#motion_model = glimpse.CartesianMotionModel(axyz_sigma=(2.0,2.0,0.2))
tracker = glimpse.Tracker(
    observers=observers, motion_model=motion_model, dem=dem, dem_sigma=3, viewshed=viewshed, time_unit=time_unit)
tracks = tracker.track(
    xy=xy, n=5000, xy_sigma=(2, 2), vxyz_sigma=(5, 5, 0.2),
    tile_size=(15, 15), n_processes=4, return_particles=False)

#n=3555 upper limit before major slowdown

# ---- Plot tracks ----

tracks.plot_xy(start=dict(color='green'), mean=dict(color='red'), sigma=dict(alpha=0.25))
