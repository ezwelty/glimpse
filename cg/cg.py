from __future__ import (print_function, division, unicode_literals)
import os
CG_PATH = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(CG_PATH, '..'))
import glimpse
from glimpse.backports import *
from glimpse.imports import (np, pandas, re, datetime, sharedmem)
import glob
import requests
try:
    from functools import lru_cache
except ImportError:
    # Python 2
    from backports.functools_lru_cache import lru_cache
try:
    FileNotFoundError
except NameError:
    # Python 2
    FileNotFoundError = IOError

# ---- Environment variables ---

print('cg: Remember to set IMAGE_PATH, KEYPOINT_PATH, MATCH_PATH, and DEM_PATHS')
IMAGE_PATH = None
KEYPOINT_PATH = None
MATCH_PATH = None
DEM_PATHS = []

# ---- Images ----

@lru_cache(maxsize=1)
def Sequences():
    """
    Return sequences metadata.
    """
    df = pandas.read_csv(
        os.path.join(CG_PATH, 'sequences.csv'),
        parse_dates=['first_time_utc', 'last_time_utc'])
    # Floor start time subseconds for comparisons to filename times
    df.first_time_utc = df.first_time_utc.apply(
        datetime.datetime.replace, microsecond=0)
    return df

def parse_image_path(path, sequence=False):
    """
    Return metadata parsed from image path.

    Arguments:
        path (str): Image path or basename
        sequence (bool): Whether to include sequence metadata (camera, service, ...)
    """
    basename = glimpse.helpers.strip_path(path)
    station, date_str, time_str = re.findall('^([^_]+)_([0-9]{8})_([0-9]{6})', basename)[0]
    capture_time = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
    results = dict(basename=basename, station=station,
        date_str=date_str, time_str=time_str, datetime=capture_time)
    if sequence:
        sequences = Sequences()
        is_row = ((sequences.station == station) &
            (sequences.first_time_utc <= capture_time) &
            (sequences.last_time_utc >= capture_time))
        rows = np.where(is_row)[0]
        if len(rows) != 1:
            raise ValueError(
                'Image path has zero or multiple sequence matches: ' + path)
        results = glimpse.helpers.merge_dicts(
            sequences.loc[rows[0]].to_dict(), results)
    return results

def find_image(path):
    """
    Return path to image file.

    Arguments:
        path (str): Image path or basename
    """
    ids = parse_image_path(path, sequence=True)
    service_dir = os.path.join(IMAGE_PATH, ids['station'],
        ids['station'] + '_' + ids['service'])
    filename = ids['basename'] + '.JPG'
    if os.path.isdir(service_dir):
        subdirs = [''] + next(os.walk(service_dir))[1]
        for subdir in subdirs:
            image_path = os.path.join(service_dir, subdir, filename)
            if os.path.isfile(image_path):
                return image_path

def load_images(station, service, start=None, end=None, step=None, use_exif=True):
    """
    Return list of calibrated Image objects.

    Any available station, camera, image, and viewdir calibrations are loaded
    and images with image calibrations are marked as anchors.

    Arguments:
        station (str): Station identifier
        service (str): Service identifier
        start (datetime): Start datetime, or `None` to start at first image
        end (datetimes): End datetime, or `None` to end with last image
        step (timedelta): Target sampling frequency, or `None` for all images
        use_exif (bool): Whether to use image datetimes from EXIF rather than
            parsed from paths
    """
    img_paths = glob.glob(os.path.join(IMAGE_PATH, station, station + '_' + service, '*.JPG'))
    if use_exif:
        exifs = [glimpse.Exif(path) for path in img_paths]
        datetimes = np.array([exif.datetime for exif in exifs])
    else:
        datetimes = np.array([parse_image_path(path)['datetime']
            for path in img_paths])
    selected = np.ones(datetimes.shape, dtype=bool)
    if start:
        selected &= datetimes >= start
    if end:
        selected &= datetimes <= end
    if step:
        targets = glimpse.helpers.datetime_range(
            start=datetimes[selected][0], stop=datetimes[selected][-1], step=step)
        indices = glimpse.helpers.find_nearest_datetimes(targets, datetimes)
        temp = np.zeros(selected.shape, dtype=bool)
        temp[indices] = True
        selected &= temp
    base_calibration = load_calibrations(img_paths[0], station=True, camera=True, merge=True)
    anchor_paths = glob.glob(os.path.join(CG_PATH, 'images', station + '*.json'))
    anchor_basenames = [glimpse.helpers.strip_path(path) for path in anchor_paths]
    images = []
    for i in np.where(selected)[0]:
        path = img_paths[i]
        basename = glimpse.helpers.strip_path(path)
        calibrations = load_calibrations(image=basename, viewdir=basename,
            merge=False, file_errors=False)
        if calibrations['image']:
            calibration = glimpse.helpers.merge_dicts(base_calibration, calibrations['image'])
            anchor = True
        else:
            calibration = base_calibration
            anchor = False
        if calibrations['viewdir']:
            calibration = glimpse.helpers.merge_dicts(calibration, calibrations['viewdir'])
        exif = exifs[i] if use_exif else None
        image = glimpse.Image(path, cam=calibration, anchor=anchor, exif=exif,
            keypoints_path=os.path.join(KEYPOINT_PATH, basename + '.pkl'))
        images.append(image)
    return images

def load_masks(images):
    """
    Return a list of boolean land masks.

    Arguments:
        images (iterable): Image objects
    """
    glimpse.Observer.test_images(images)
    # NOTE: Assumes that all images are from same station
    station = parse_image_path(images[0].path)['station']
    imgsz = images[0].cam.imgsz
    # Find all svg files for station with 'land' markup
    svg_paths = glob.glob(os.path.join(CG_PATH, 'svg', station + '_*.svg'))
    markups = [glimpse.svg.parse_svg(path, imgsz=imgsz)
        for path in svg_paths]
    land_index = np.where(['land' in markup for markup in markups])[0]
    # Select svg files nearest to images
    svg_datetimes = [parse_image_path(path)['datetime']
        for path in np.array(svg_paths)[land_index]]
    img_datetimes = [img.datetime for img in images]
    nearest_index = glimpse.helpers.find_nearest_datetimes(
        img_datetimes, svg_datetimes)
    nearest = np.unique(nearest_index)
    # Make masks and expand per image without copying
    land_markups = np.array(markups)[land_index]
    masks = [None] * len(images)
    for i in nearest:
        polygons = land_markups[i]['land'].values()
        mask = glimpse.helpers.polygons_to_mask(polygons, imgsz=imgsz).astype(np.uint8)
        mask = sharedmem.copy(mask)
        for j in np.where(nearest_index == i)[0]:
            masks[j] = mask
    return masks

# ---- Calibration controls ----

def svg_controls(img, svg, keys=None, correction=True):
    """
    Return control objects for an Image.

    Arguments:
        img (Image): Image object
        svg (str or dict): Path to SVG file or parsed result
        keys (iterable): SVG layers to include, or all if `None`
        correction: Whether control objects should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
    """
    if isinstance(svg, (bytes, str)):
        svg = glimpse.svg.parse_svg(svg, imgsz=img.cam.imgsz)
    if keys is None:
        keys = svg.keys()
    controls = []
    for key in keys:
        if key in svg:
            if key == 'gcp':
                controls.append(gcp_points(img, svg[key], correction=correction))
            elif key == 'coast':
                controls.append(coast_lines(img, svg[key], correction=correction))
            elif key == 'terminus':
                controls.append(terminus_lines(img, svg[key], correction=correction))
            elif key == 'moraines':
                controls.extend(moraines_mlines(img, svg[key], correction=correction))
            elif key == 'horizon':
                controls.append(horizon_lines(img, svg[key], correction=correction))
    return controls

def gcp_points(img, markup, correction=True):
    """
    Return ground control Points object for an Image.

    Arguments:
        img (Image): Image object
        markup (dict): Parsed SVG layer
        correction: Whether Points should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
    """
    uv = np.vstack(markup.values())
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'gcp.geojson'), key='id', crs=32606)
    xyz = np.vstack((geo['features'][key]['geometry']['coordinates']
        for key in markup))
    return glimpse.optimize.Points(img.cam, uv, xyz, correction=correction)

def coast_lines(img, markup, correction=True):
    """
    Return coast Lines object for an Image.

    Arguments:
        img (Image): Image object
        markup (dict): Parsed SVG layer
        correction (bool): Whether to set Lines to use elevation correction
    """
    luv = tuple(markup.values())
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'coast.geojson'), crs=32606)
    lxy = [feature['geometry']['coordinates'] for feature in geo['features']]
    lxyz = [np.hstack((xy, sea_height(xy, t=img.datetime))) for xy in lxy]
    return glimpse.optimize.Lines(img.cam, luv, lxyz, correction=correction)

def terminus_lines(img, markup, correction=True):
    """
    Return terminus Lines object for an Image.

    Arguments:
        img (Image): Image object
        markup (dict): Parsed SVG layer
        correction: Whether Lines should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
    """
    luv = tuple(markup.values())
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'termini.geojson'), key='date', crs=32606)
    date_str = img.datetime.strftime('%Y-%m-%d')
    xy = geo['features'][date_str]['geometry']['coordinates']
    xyz = np.hstack((xy, sea_height(xy, t=img.datetime)))
    return glimpse.optimize.Lines(img.cam, luv, [xyz], correction=correction)

def horizon_lines(img, markup, correction=True):
    """
    Return horizon Lines object for an Image.

    Arguments:
        img (Image): Image object
        markup (dict): Parsed SVG layer
        correction: Whether Lines should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
    """
    luv = tuple(markup.values())
    station = parse_image_path(img.path)['station']
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'horizons', station + '.geojson'), crs=32606)
    lxyz = [coords for coords in glimpse.helpers.geojson_itercoords(geo)]
    return glimpse.optimize.Lines(img.cam, luv, lxyz, correction=correction)

def moraines_mlines(img, markup, correction=True):
    """
    Return list of moraine Lines objects for an Image.

    Arguments:
        img (Image): Image object
        markup (dict): Parsed SVG layer
        correction: Whether Lines should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
    """
    date_str = img.datetime.strftime('%Y%m%d')
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'moraines', date_str + '.geojson'), key='id', crs=32606)
    mlines = []
    for key, moraine in markup.items():
        luv = tuple(moraine.values())
        xyz = geo['features'][key]['geometry']['coordinates']
        mlines.append(glimpse.optimize.Lines(img.cam, luv, [xyz], correction=correction))
    return mlines

def sea_height(xy, t=None):
    """
    Return the height of sea level relative to the WGS 84 ellipsoid.

    Uses the EGM 2008 geoid height and the NOAA tide gauge in Valdez, Alaska.

    Arguments:
        xy (array): World coordinates (n, 2)
        t (datetime): Datetime at which to estimate tidal height.
            If `None`, tide is ignored in result.
    """
    egm2008 = glimpse.DEM.read(os.path.join(CG_PATH, 'egm2008.tif'))
    geoid_height = egm2008.sample(xy).reshape(-1, 1)
    if t:
        t_begin = t.replace(minute=0, second=0, microsecond=0)
        t_end = t_begin + datetime.timedelta(hours=1)
        # https://tidesandcurrents.noaa.gov/api/
        params = dict(
            format='json',
            units='metric',
            time_zone='gmt',
            datum='MLLW',
            product='hourly_height',
            station=9454240, # Valdez
            begin_date=t_begin.strftime('%Y%m%d %H:%M'),
            end_date=t_end.strftime('%Y%m%d %H:%M'))
        r = requests.get('https://tidesandcurrents.noaa.gov/api/datagetter', params=params)
        v = [float(item['v']) for item in r.json()['data']]
        tide_height = np.interp(
            (t - t_begin).total_seconds(),
            [0, (t_end - t_begin).total_seconds()], v)
    else:
        tide_height = 0
    return geoid_height + tide_height

# ---- Control bundles ----

def station_svg_controls(station, size=1, force_size=False, keys=None,
    correction=True, station_calib=False, camera_calib=True):
    """
    Return all SVG control objects for a station.

    Arguments:
        station (str): Station identifier
        size: Image scale factor (number) or image size in pixels (nx, ny)
        force_size (bool): Whether to force `size` even if different aspect ratio
            than original size.
        keys (iterable): SVG layers to include
        correction: Whether control objects should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
        station_calib (bool): Whether to load station calibration. If `False`,
            falls back to the station estimate.
        camera_calib (bool): Whether to load camera calibrations. If `False`,
            falls back to the EXIF estimate.

    Returns:
        list: Image objects
        list: Control objects (Points, Lines)
        list: Per-camera calibration parameters [{'viewdir': True}, ...]
    """
    svg_paths = glob.glob(os.path.join(CG_PATH, 'svg', station + '*.svg'))
    images, controls, cam_params = [], [], []
    for svg_path in svg_paths:
        calibration = load_calibrations(svg_path, camera=camera_calib,
            station=station_calib, station_estimate=not station_calib, merge=True)
        img_path = find_image(svg_path)
        images.append(glimpse.Image(img_path, cam=calibration))
        images[-1].cam.resize(size, force=force_size)
        controls.extend(svg_controls(images[-1], svg_path, keys=keys, correction=correction))
        cam_params.append(dict(viewdir=True))
    return images, controls, cam_params

def camera_svg_controls(camera, size=1, force_size=False, keys=None,
    correction=True, station_calib=False, camera_calib=False):
    """
    Return all SVG control objects available for a camera.

    Arguments:
        camera (str): Camera identifer
        size: Image scale factor (number) or image size in pixels (nx, ny)
        force_size (bool): Whether to force `size` even if different aspect ratio
            than original size.
        keys (iterable): SVG layers to include
        correction: Whether control objects should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
        station_calib (bool): Whether to load station calibration. If `False`,
            falls back to the station estimate.
        camera_calib (bool): Whether to load camera calibrations. If `False`,
            falls back to the EXIF estimate.

    Returns:
        list: Image objects
        list: Control objects (Points, Lines)
        list: Per-camera calibration parameters [{'viewdir': True}, ...]
    """
    svg_paths = glob.glob(os.path.join(CG_PATH, 'svg', '*.svg'))
    images, controls, cam_params = [], [], []
    for svg_path in svg_paths:
        ids = parse_image_path(svg_path, sequence=True)
        if ids['camera'] == camera:
            calibration = load_calibrations(svg_path, camera=camera_calib,
                station=station_calib, station_estimate=not station_calib, merge=True)
            img_path = find_image(svg_path)
            images.append(glimpse.Image(img_path, cam=calibration))
            images[-1].cam.resize(size, force=force_size)
            controls.extend(svg_controls(images[-1], svg_path, keys=keys, correction=correction))
            cam_params.append(dict(viewdir=True))
    return images, controls, cam_params

def camera_motion_matches(camera, size=1, force_size=False,
    station_calib=False, camera_calib=False, detect=dict(), match=dict()):
    """
    Returns all motion Matches objects available for a camera.

    Arguments:
        camera (str): Camera identifier
        size: Image scale factor (number) or image size in pixels (nx, ny)
        force_size (bool): Whether to force `size` even if different aspect ratio
            than original size.
        station_calib (bool): Whether to load station calibration. If `False`,
            falls back to the station estimate.
        camera_calib (bool): Whether to load camera calibrations. If `False`,
            falls back to the EXIF estimate.
        detect (dict): Arguments passed to `glimpse.optimize.detect_keypoints()`
        match (dict): Arguments passed to `glimpse.optimize.match_keypoints()`

    Returns:
        list: Image objects
        list: Matches objects
        list: Per-camera calibration parameters [{}, {'viewdir': True}, ...]
    """
    motion = glimpse.helpers.read_json(os.path.join(CG_PATH, 'motion.json'))
    sequences = [item['paths'] for item in motion
        if parse_image_path(item['paths'][0], sequence=True)['camera'] == camera]
    images, matches, cam_params = [], [], []
    for sequence in sequences:
        sys.stdout.write('.')
        sys.stdout.flush()
        paths = [find_image(path) for path in sequence]
        images.extend([glimpse.Image(path,
            cam=load_calibrations(path,  camera=camera_calib,
            station=station_calib, station_estimate=not station_calib, merge=True))
            for path in paths])
        idx = slice(-len(sequence), None)
        for img in images[idx]:
            img.cam.resize(size, force=force_size)
        matches.extend(build_sequential_matches(images[idx], detect=detect, match=match))
        cam_params.extend([dict()] + [dict(viewdir=True)] * (len(sequence) - 1))
    return images, matches, cam_params

def build_sequential_matches(images, detect=dict(), match=dict()):
    """
    Returns Matches objects for sequential Image pairs.

    Arguments:
        images (iterable): Image objects
        detect (dict): Arguments passed to `glimpse.optimize.detect_keypoints()`
        match (dict): Arguments passed to `glimpse.optimize.match_keypoints()`
    """
    keypoints = [glimpse.optimize.detect_keypoints(img.read(), **detect) for img in images]
    matches = []
    for i in range(len(images) - 1):
        uvA, uvB = glimpse.optimize.match_keypoints(keypoints[i], keypoints[i + 1], **match)
        matches.append(glimpse.optimize.Matches(
            cams=(images[i].cam, images[i + 1].cam), uvs=(uvA, uvB)))
    return matches

# ---- Calibrations ----

def load_calibrations(path=None, station_estimate=False, station=False,
    camera=False, image=False, viewdir=False, merge=False, file_errors=True):
    """
    Return camera calibrations.

    Arguments:
        path (str): Image basename or path
        station_estimate: Whether to load station estimate (bool) or
            station identifier to load (str).
            If `True`, the station identifier is parsed from `path`.
        station: Whether to load station (bool) or
            station identifier to load (str).
            If `True`, the station identifier is parsed from `path`.
        camera: Whether to load camera (bool) or
            camera identifier to load (str).
            If `True`, the camera identifier is parsed from `path`.
        image: Whether to load image (bool) or
            image to load (str).
            If `True`, the image basename is parsed from `path`.
        viewdir: Whether to load view direction (bool) or
            view direction to load (str).
            If `True`, the image basename is parsed from `path`.
        merge (bool): Whether to merge calibrations, in the order
            station_estimate, station, camera, image, viewdir
        file_errors (bool): Whether to raise an error if a requested calibration
            file is not found
    """
    def _try_except(fun, arg):
        try:
            return fun(arg)
        except FileNotFoundError as e:
            if file_errors:
                raise e
            else:
                return None
    if path:
        ids = parse_image_path(path, sequence=(camera is True))
        if station_estimate is True:
            station_estimate = ids['station']
        if station is True:
            station = ids['station']
        if camera is True:
            camera = ids['camera']
        if image is True:
            image = ids['basename']
        if viewdir is True:
            viewdir = ids['basename']
    calibrations = dict()
    if station_estimate:
        calibrations['station_estimate'] = _try_except(_load_station_estimate, station_estimate)
    if station:
        calibrations['station'] = _try_except(_load_station, station)
    if camera:
        calibrations['camera'] = _try_except(_load_camera, camera)
    if image:
        calibrations['image'] = _try_except(_load_image, image)
    if viewdir:
        calibrations['viewdir'] = _try_except(_load_viewdir, viewdir)
    if merge:
        return merge_calibrations(calibrations)
    else:
        return calibrations

def merge_calibrations(calibrations, keys=('station_estimate', 'station', 'camera', 'image', 'viewdir')):
    """
    Merge camera calibrations.

    Arguments:
        calibrations (iterable): Dictionaries of calibration parameters
        keys (iterable): Calibration types, in order from lowest to highest
            overwrite priority
    """
    calibration = dict()
    for key in keys:
        if key in calibrations and calibrations[key]:
            calibration = glimpse.helpers.merge_dicts(calibration, calibrations[key])
    return calibration

def _load_station_estimate(station):
    stations_path = os.path.join(CG_PATH, 'geojson', 'stations.geojson')
    geo = glimpse.helpers.read_geojson(stations_path, crs=32606, key='id')
    return dict(
        xyz=np.reshape(geo['features'][station]['geometry']['coordinates'], -1),
        viewdir=geo['features'][station]['properties']['viewdir'])

def _load_station(station):
    station_path = os.path.join(CG_PATH, 'stations', station + '.json')
    return glimpse.helpers.read_json(station_path)

def _load_camera(camera):
    camera_path = os.path.join(CG_PATH, 'cameras', camera + '.json')
    return glimpse.helpers.read_json(camera_path)

def _load_image(path):
    basename = glimpse.helpers.strip_path(path)
    image_path = os.path.join(CG_PATH, 'images', basename + '.json')
    return glimpse.helpers.read_json(image_path)

def _load_viewdir(path):
    basename = glimpse.helpers.strip_path(path)
    viewdir_path = os.path.join(CG_PATH, 'viewdirs', basename + '.json')
    return glimpse.helpers.read_json(viewdir_path)

def write_image_viewdirs(images, viewdirs=None):
    """
    Write Image view directions to file.

    Arguments:
        images (iterable): Image objects
        viewdirs (iterable): Camera view directions to write.
            If `None`, these are read from `images[i].cam.viewdirs`.
    """
    for i, img in enumerate(images):
        basename = glimpse.helpers.strip_path(img.path)
        path = os.path.join(CG_PATH, 'viewdirs', basename + '.json')
        if viewdirs is None:
            d = dict(viewdir=tuple(img.cam.viewdir))
        else:
            d = dict(viewdir=tuple(viewdirs[i]))
        glimpse.helpers.write_json(d, path=path)

# ---- Helpers ----

def load_dem_interpolant(**kwargs):
    """
    Return a canonical DEMInterpolant object.

    Loads all '*.tif' files found in `DEM_PATHS`, parsing dates from the first
    sequence of 8 numeric digits in each basename.

    Arguments:
        **kwargs (dict): Additional arguments to `glimpse.DEMInterpolant()`
    """
    paths = [path for directory in DEM_PATHS
        for path in glob.glob(os.path.join(directory, '*.tif'))]
    dates = [re.findall(r'([0-9]{8})', glimpse.helpers.strip_path(path))[0] for path in paths]
    # DEMs are produced from imagery taken near local noon, about 22:00 UTC
    datetimes = [datetime.datetime.strptime(date + '22', '%Y%m%d%H') for date in dates]
    return glimpse.DEMInterpolant(paths, datetimes, **kwargs)

def project_dem(cam, dem, array=None, mask=None, correction=True):
    """
    Return a grayscale image formed by projecting a DEM into a Camera.

    Arguments:
        cam (Camera): Camera object
        dem (DEM): DEM object
        array (array): Array (with shape `dem.Z`) of pixel values to project.
            If `None`, `dem.Z` is used.
        mask (array): Boolean array (with shape `dem.Z`) indicating which values
            to project
        correction: Whether to apply elevation corrections (bool) or arguments
            to `glimpse.helpers.elevation_corrections()`
    """
    if mask is None:
        mask = ~np.isnan(dem.Z)
    xyz = np.column_stack((dem.X[mask], dem.Y[mask], dem.Z[mask]))
    uv = cam.project(xyz, correction=correction)
    if array is None:
        array = dem.Z
    return cam.rasterize(uv, array[mask])
