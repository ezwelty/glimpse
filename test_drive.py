import glimpse
from glimpse.imports import (datetime, np, os, re)
import glob
import matplotlib

# ---- Constants ----

DATA_DIR = 'data'
DEM_DIR = os.path.join(DATA_DIR, 'dem')
IMG_DIR = 'images'
CAM_DIR = 'images_json'
MAX_DISTANCE = 15e3 # Camera world units (meters)
STATIONS = ('AK01b', 'AK10b')

# ---- Prepare Observers ----

start = datetime.datetime(2013, 6, 12, 20)
end = datetime.datetime(2013, 6, 18, 20)
observers = []
for station in STATIONS:
    station_dir = os.path.join(DATA_DIR, station)
    cam_paths = glob.glob(os.path.join(station_dir, CAM_DIR, '*.json'))
    basenames = [os.path.splitext(os.path.basename(path))[0] for path in cam_paths]
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
dem.Z[dem.Z < 0] = np.nan
dem.fill_crevasses_simple()

# ---- Run Tracker ----

xy = (4.988e5, 6.78186e6)
time_unit = datetime.timedelta(days=1).total_seconds()
tracker = glimpse.Tracker(
    observers=observers, dem=dem,
    time_unit=time_unit, resample_method='systematic')
tracker.initialize_particles(
    n=5000, xy=xy, xy_sigma=(2, 2),
    vxy=(0, 0), vxy_sigma=(10, 10))
tracker.track(datetimes=None, maxdt=0, tile_size=(51, 51),
    axy=(0, 0), axy_sigma=(2, 2))

# ---- Plot track ----

means = np.vstack(tracker.means)
matplotlib.pyplot.plot(means[:, 0], means[:, 1], marker='.', color='red')
matplotlib.pyplot.plot(means[0, 0], means[0, 1], marker='.', color='green')
