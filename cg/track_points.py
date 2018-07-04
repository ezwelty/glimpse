# [Mac OS] Parallel requires disabling default matplotlib backend
import matplotlib
matplotlib.use('agg')

import cg
from cg import glimpse
from glimpse.imports import (datetime, np, os, collections, copy)

# ---- Environment ----

cg.IMAGE_PATH = '/volumes/science/data/columbia/timelapse' # Path to time-lapse images
flat_image_path = False # Whether path contains flat list of images
parallel = True # Number of parallel processes, True = all, False = disable parallel

# ---- Tracker heuristics ----
# units: meters, days

n = 10000
# xy = (pre-computed)
xy_sigma = (2, 2)
# vxyz = (pre-computed from mean long-term x, y velocities and DEM slope)
# vxyz_sigma = (pre-computed from long-term x, y velocities and DEM slope)
short_vxy_sigma = 0.25 # Additional short vxy_sigma, as fraction of long vxy_sigma
flotation_vz_sigma = 3.5 # Multiplied by flotation probability
min_vz_sigma = 0.2 # Minimum vz_sigma, applied after flotation_vz_sigma
axy_sigma_scale = 0.75 # Fraction of final vxy_sigma
flotation_az_sigma = 12 # Multiplied by flotation probability
min_az_sigma = 0.2 # Minimum vz_sigma, applied after flotation_vz_sigma
tile_size = (15, 15)

# ---- Load observer list ----

observer_json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)

# ---- Load DEM interpolant ----
# Checks for `dems/` (<datestr>-<obs>[_stderr].tif), then `dem_interpolant.pkl`

if os.path.isdir('dems'):
    dem_interpolant = None
else:
    dem_interpolant = glimpse.helpers.read_pickle('dem_interpolant.pkl')
    dem_padding = 200 # m

# ---- Track points ----

for i_obs in range(len(observer_json)):
    # ---- Load observers ----
    # observers
    observers = []
    for station, basenames in observer_json[i_obs].items():
        meta = cg.parse_image_path(basenames[0], sequence=True)
        service_calibration = cg.load_calibrations(
            station_estimate=meta['station'], station=meta['station'],
            camera=meta['camera'], merge=True, file_errors=False)
        datetimes = cg.paths_to_datetimes(basenames)
        # Use dummy Exif for speed
        service_exif = glimpse.Exif(cg.find_image(basenames[0]))
        images = []
        for basename, t in zip(basenames, datetimes):
            calibration = glimpse.helpers.merge_dicts(service_calibration,
                cg.load_calibrations(image=basename, viewdir=basename, merge=True,
                file_errors=False))
            if flat_image_path:
                path = os.path.join(cg.IMAGE_PATH, basename + '.JPG')
            else:
                path = cg.find_image(basename)
            image = glimpse.Image(path, cam=calibration, datetime=t, exif=service_exif)
            images.append(image)
        # NOTE: Determine sigma programmatically?
        observer = glimpse.Observer(images, cache=True, correction=True, sigma=0.3)
        observers.append(observer)
    # ---- Load track points ----
    # ids, xy, observer_mask, vxyz, vxyz_sigma, flotation
    t = min([observer.datetimes[0] for observer in observers])
    datestr = t.strftime('%Y%m%d')
    basename = datestr + '-' + str(i_obs)
    params = glimpse.helpers.read_pickle(os.path.join('points', basename + '.pkl'))
    print(basename)
    # ---- Load DEM ----
    # dem, dem_sigma
    if dem_interpolant is None:
        dem = glimpse.Raster.read(os.path.join('dems', basename + '.tif'))
        dem_sigma = glimpse.Raster.read(os.path.join('dems', basename + '_stderr.tif'))
    else:
        dem, dem_sigma = dem_interpolant(t, return_sigma=True)
        # Crop DEM
        box = (glimpse.helpers.bounding_box(params['xy']) +
            np.array([-1, -1, 1, 1]) * dem_padding)
        dem.crop(xlim=box[0::2], ylim=box[1::2])
        dem_sigma.crop(xlim=box[0::2], ylim=box[1::2])
        dem.crop_to_data()
        dem_sigma.crop_to_data()
    # ---- Compute final motion parameters ----
    # Tracker.track(**kwargs)
    kwargs = dict(
        n=n,
        xy=params['xy'],
        xy_sigma=xy_sigma,
        vxyz=params['vxyz'],
        vxyz_sigma=np.column_stack((
            params['vxyz_sigma'][:, 0] * (1 + short_vxy_sigma),
            params['vxyz_sigma'][:, 1] * (1 + short_vxy_sigma),
            params['vxyz_sigma'][:, 2] +
                np.maximum(params['flotation'] * flotation_vz_sigma, min_vz_sigma)
        )),
        axyz=(0, 0, 0),
        observer_mask=params['observer_mask']
    )
    kwargs['axyz_sigma'] = np.column_stack((
        kwargs['vxyz_sigma'][:, 0:2] * axy_sigma_scale,
        np.maximum(params['flotation'] * flotation_az_sigma, min_az_sigma)
    ))
    # ---- Track points ----
    # tracks, tracks_r
    time_unit = datetime.timedelta(days=1)
    tracker = glimpse.Tracker(
        observers=observers, dem=dem, dem_sigma=dem_sigma, time_unit=time_unit)
    # Initialize placeholders
    # forward | forward + last vxy | reverse | reverse + last vxy
    suffixes = ('f', 'fv', 'r', 'rv')
    directions = (1, 1, -1, -1)
    tracks = [None] * len(suffixes)
    paths = [os.path.join('tracks', basename + '-' + suffix + '.pkl')
        for suffix in suffixes]
    is_file = [os.path.isfile(path) for path in paths]
    # Run forward and backward
    for i in (0, 2):
        if not is_file[i]:
            print(basename + suffixes[i])
            tracks[i] = tracker.track(tile_size=tile_size, parallel=parallel,
                datetimes=tracker.datetimes[::directions[i]], **kwargs)
        if not is_file[i + 1]:
            print(basename + suffixes[i + 1])
            if not tracks[i]:
                tracks[i] = glimpse.helpers.read_pickle(paths[i])
            # Start with last vx, vy distribution of first run
            mask, first, last = tracks[i].endpoints()
            vxy_kwargs = copy.deepcopy(kwargs)
            vxy_kwargs['vxyz'][mask, 0:2] = tracks[i].vxyz[mask, last, 0:2]
            vxy_kwargs['vxyz_sigma'][mask, 0:2] = tracks[i].vxyz_sigma[mask, last, 0:2]
            tracks[i + 1] = tracker.track(tile_size=tile_size, parallel=parallel,
                datetimes=tracker.datetimes[::directions[i + 1]], **vxy_kwargs)
    # ---- Save tracks to file ----
    tracker.dem, tracker.dem_sigma = None, None
    for observer in observers:
        observer.clear_images()
    for i in range(len(suffixes)):
        if not os.path.isfile(paths[i]) and tracks[i] is not None:
            glimpse.helpers.write_pickle(tracks[i], paths[i])
