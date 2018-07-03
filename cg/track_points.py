import matplotlib
matplotlib.use('agg')
import cg
from cg import glimpse
# glimpse.config.use_numpy_matmul(False)
from glimpse.imports import (datetime, np, os, collections)
import glob
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
flat_image_path = False

# ---- Observer list ----

observer_json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)

# ---- DEM interpolant ----
# NOTE: If not defined, requires geotiffs dems/<datestr>-<obs>[_stderr].tif

dem_interpolant = None
# dem_interpolant = glimpse.helpers.read_pickle('dem_interpolant.pkl')
# dem_padding = 200 # m

# ---- Tracker heuristics ----
# NOTE: All units in meters, days

n = 10000
# xy = (pre-computed)
xy_sigma = (2, 2)
# vxyz = (pre-computed from mean long-term x, y velocities and DEM slope)
# vxyz_sigma = (pre-computed from long-term x, y velocities and DEM slope)
short_vxy_sigma = 0.25 # additional short vxy_sigma, as fraction of long vxy_sigma
flotation_vz_sigma = 3.5 # multiplied by flotation probability
min_vz_sigma = 0.2 # minimum vz_sigma, applied after flotation_vz_sigma
axyz = (0, 0, 0)
axy_sigma_scale = 0.75 # fraction of final vxy_sigma
flotation_az_sigma = 12 # multiplied by flotation probability
min_az_sigma = 0.2 # minimum vz_sigma, applied after flotation_vz_sigma
parallel = 7
tile_size = (15, 15)
reverse_run = False

# ---- Tracker points ----

for i_obs in (184, ): #range(184, len(observer_json)):
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
    # ---- Load DEM ----
    # dem, dem_sigma
    if dem_interpolant is None:
        dem = glimpse.Raster.read(os.path.join('dems', basename + '.tif'))
        dem_sigma = glimpse.Raster.read(os.path.join('dems', basename + '_stderr.tif'))
    else:
        dem, dem_sigma = dem_interpolant(t, return_sigma=True)
        # Crop DEM
        box = glimpse.helpers.bounding_box(params['xy']) + np.array([-1, -1, 1, 1]) * dem_padding
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
        axyz=axyz,
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
    tracks, tracks_r = None, None
    # Forward
    path = os.path.join('tracks', basename + '.pkl')
    if not os.path.isfile(path):
        tracks = tracker.track(tile_size=tile_size, parallel=parallel, **kwargs)
    # Reverse
    path_r = os.path.join('tracks', basename + '_r.pkl')
    if not os.path.isfile(path_r) and reverse_run:
        if tracks is None:
            tracks = glimpse.helpers.read_pickle(path)
        mask, first, last = tracks.endpoints()
        # Use last positions from forward track?
        # kwargs['xy'][mask] = tracks.xyz[mask, last, 0:2]
        # Use last distribution from forward track
        kwargs['vxyz'][mask, 0:2] = tracks.vxyz[mask, last, 0:2]
        kwargs['vxyz_sigma'][mask, 0:2] = tracks.vxyz_sigma[mask, last, 0:2]
        tracks_r = tracker.track(tile_size=tile_size, parallel=parallel,
            datetimes=tracker.datetimes[::-1], **kwargs)
    # ---- Save tracks to file ----
    tracker.dem, tracker.dem_sigma = None, None
    for observer in observers:
        observer.clear_images()
    if not os.path.isfile(path) and tracks is not None:
        glimpse.helpers.write_pickle(tracks, path)
    if not os.path.isfile(path_r) and tracks_r is not None:
        glimpse.helpers.write_pickle(tracks_r, path_r)
