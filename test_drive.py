import glimpse
from glimpse.imports import (datetime, np, os, re, matplotlib, sharedmem)
import glob
import itertools

# ---- Constants ----

DATA_DIR = 'data'
DEM_DIR = os.path.join(DATA_DIR, 'dem')
IMG_DIR = 'images'
CAM_DIR = 'images_json'
MAX_DISTANCE = 30e3 # Camera world units (meters)
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

boxes = [obs.images[0].cam.viewbox(radius=MAX_DISTANCE)
    for obs in observers]
box = glimpse.helpers.intersect_boxes(boxes)
path = glob.glob(os.path.join(DEM_DIR, '*.tif'))[0]
dem = glimpse.DEM.read(path, xlim=box[0::3], ylim=box[1::3])
dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.Z), fill=False)
for obs in observers:
    dem.fill_circle(obs.xyz, radius=100)

# ---- Prepare viewshed ----

viewsheds = [dem.viewshed(obs.xyz) for obs in observers]
viewshed = np.ones(dem.Z.shape, dtype=bool)
for v in viewsheds:
    viewshed &= v

# ---- Run Tracker ----

xy0 = np.array((4.988e5, 6.78186e6))
xy = xy0 + np.vstack([xy for xy in
    itertools.product(range(-300, 400, 100), range(-300, 400, 100))])
time_unit = datetime.timedelta(days=1)
tracker = glimpse.Tracker(
    observers=observers, dem=dem, viewshed=viewshed, time_unit=time_unit)
import timeit
start = timeit.default_timer()
tracks = tracker.track(
    xy=xy, n=5000, xy_sigma=(2, 2), vxy_sigma=(10, 10), axy_sigma=(2, 2),
    tile_size=(15, 15), parallel=True)
timeit.default_timer() - start

# ---- Plot tracks ----

tracks.plot_xy(color='red')
