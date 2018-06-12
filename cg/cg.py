from __future__ import (print_function, division, unicode_literals)
import os
CG_PATH = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(CG_PATH, '..'))
import glimpse
from glimpse.backports import *
from glimpse.imports import (np, pandas, re, datetime, sharedmem, cv2)
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
    return df.sort_values('first_time_utc').reset_index(drop=True)

@lru_cache(maxsize=1)
def Stations():
    """
    Return stations metadata.
    """
    stations_path = os.path.join(CG_PATH, 'geojson', 'stations.geojson')
    return glimpse.helpers.read_geojson(stations_path, crs=32606, key='id')['features']

def _station_break_index(path):
    """
    Return index of image in motion break sequence.

    Arguments:
        path (str): Image path

    Returns:
        int: Either 0 (original viewdir) or i (viewdir of break i + 1)
    """
    stations = Stations()
    ids = parse_image_path(path)
    station = stations[ids['station']]
    if 'breaks' not in station['properties']:
        return 0
    breaks = station['properties']['breaks']
    if not breaks:
        return 0
    break_images = np.array([x['start'] for x in breaks])
    idx = np.argsort(break_images)
    i = np.where(break_images[idx] <= ids['basename'])[0]
    if i.size > 0:
        return idx[i[-1]] + 1
    else:
        return 0

def paths_to_datetimes(paths):
    """
    Return datetime objects parsed from image paths.

    Arguments:
        paths (iterable): Image paths
    """
    pattern = re.compile(r'_([0-9]{8}_[0-9]{6})[^\/]*$')
    datetimes_str = [pattern.findall(path)[0] for path in paths]
    return pandas.to_datetime(datetimes_str, format='%Y%m%d_%H%M%S').to_pydatetime()

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

def load_images(station, services, use_exif=False, service_exif=False, anchors=False,
    viewdir=True, viewdir_as_anchor=False, **kwargs):
    """
    Return list of calibrated Image objects.

    Any available station, camera, image, and viewdir calibrations are loaded
    and images with image calibrations are marked as anchors.

    Arguments:
        station (str): Station identifier
        services (iterable): Service identifiers
        use_exif (bool): Whether to parse image datetimes from EXIF (slower)
            rather than parsed from paths (faster)
        service_exif (bool): Whether to extract EXIF from first image (faster)
            or all images (slower) in service.
            If `True`, `Image.datetime` is parsed from path.
            Always `False` if `use_exif=True`.
        anchors (bool): Whether to include anchor images even if
            filtered out by `kwargs['snap']`
        **kwargs: Arguments to `glimpse.helpers.select_datetimes()`
    """
    if use_exif:
        service_exif = False
    # Sort services in time
    if isinstance(services, str):
        services = services,
    services = np.sort(services)
    # Parse datetimes of all candidate images
    paths_service = [glob.glob(os.path.join(IMAGE_PATH, station, station + '_' + service, '*.JPG'))
        for service in services]
    paths = np.hstack(paths_service)
    basenames = [glimpse.helpers.strip_path(path) for path in paths]
    if use_exif:
        exifs = [glimpse.Exif(path) for path in paths]
        datetimes = np.array([exif.datetime for exif in exifs])
    else:
        datetimes = paths_to_datetimes(basenames)
    # Select images based on datetimes
    indices = glimpse.helpers.select_datetimes(datetimes, **kwargs)
    if anchors:
        # Add anchors
        # HACK: Ignore any <image>-<suffix>.json files
        anchor_paths = glob.glob(os.path.join(CG_PATH, 'images', station + '_*[0-9].json'))
        anchor_basenames = [glimpse.helpers.strip_path(path) for path in anchor_paths]
        if 'start' in kwargs or 'end' in kwargs:
            # Filter by start, end
            anchor_datetimes = np.asarray(paths_to_datetimes(anchor_basenames))
            inrange = glimpse.helpers.select_datetimes(
                anchor_datetimes, **glimpse.helpers.merge_dicts(kwargs, dict(snap=None)))
            anchor_basenames = np.asarray(anchor_basenames)[inrange]
        anchor_indices = np.where(np.isin(basenames, anchor_basenames))[0]
        indices = np.unique(np.hstack((indices, anchor_indices)))
    service_breaks = np.hstack((0, np.cumsum([len(x) for x in paths_service])))
    station_calibration = load_calibrations(station=station, merge=True)
    images = []
    for i, service in enumerate(services):
        index = indices[(indices >= service_breaks[i]) & (indices < service_breaks[i + 1])]
        if not index.size:
            continue
        service_calibration = glimpse.helpers.merge_dicts(
            station_calibration,
            load_calibrations(path=paths[index[0]], camera=True, merge=True))
        if service_exif:
            exif = glimpse.Exif(paths[index[0]])
        for j in index:
            basename = basenames[j]
            calibrations = load_calibrations(image=basename,
                viewdir=basename if viewdir else False,
                station_estimate=station, merge=False, file_errors=False)
            if calibrations['image']:
                calibration = glimpse.helpers.merge_dicts(
                    service_calibration, calibrations['image'])
                anchor = True
            else:
                calibration = glimpse.helpers.merge_dicts(
                    service_calibration,
                    dict(viewdir=calibrations['station_estimate']['viewdir']))
                anchor = False
            if viewdir and calibrations['viewdir']:
                calibration = glimpse.helpers.merge_dicts(
                    calibration, calibrations['viewdir'])
                if viewdir_as_anchor:
                    anchor = True
            if use_exif:
                exif = exifs[j]
            elif not service_exif:
                exif = None
            image = glimpse.Image(
                path=paths[j], cam=calibration, anchor=anchor, exif=exif,
                datetime=None if use_exif else datetimes[j],
                keypoints_path=os.path.join(KEYPOINT_PATH, basename + '.pkl'))
            images.append(image)
    return images

def load_masks(images):
    """
    Return a list of boolean land masks.

    Images must all be from the same station.

    Arguments:
        images (iterable): Image objects
    """
    # All images must be from the same station (for now)
    station = parse_image_path(images[0].path)['station']
    pattern = re.compile(station + r'_[0-9]{8}_[0-9]{6}[^\/]*$')
    is_station = [pattern.search(img.path) is not None for img in images[1:]]
    assert all(is_station)
    # Find all station svg with 'land' markup
    imgsz = images[0].cam.imgsz
    svg_paths = glob.glob(os.path.join(CG_PATH, 'svg', station + '_*.svg'))
    markups = [glimpse.svg.parse_svg(path, imgsz=imgsz) for path in svg_paths]
    land_index = np.where(['land' in markup for markup in markups])[0]
    if len(land_index) == 0:
        raise ValueError('No land masks found for station')
    svg_paths = np.array(svg_paths)[land_index]
    land_markups = np.array(markups)[land_index]
    # Select svg files nearest to images, with preference within breaks
    svg_datetimes = paths_to_datetimes(svg_paths)
    svg_break_indices = np.array([_station_break_index(path)
        for path in svg_paths])
    img_datetimes = [img.datetime for img in images]
    distances = glimpse.helpers.pairwise_distance_datetimes(
        img_datetimes, svg_datetimes)
    nearest_index = []
    for i, img in enumerate(images):
        break_index = _station_break_index(img.path)
        same_break = np.where(break_index == svg_break_indices)[0]
        if same_break.size > 0:
            i = same_break[np.argmin(distances[i][same_break])]
        else:
            raise ValueError('No mask found within motion breaks for image', i)
            i = np.argmin(distances[i])
        nearest_index.append(i)
    nearest = np.unique(nearest_index)
    # Make masks and expand per image without copying
    masks = [None] * len(images)
    image_sizes = np.array([img.cam.imgsz for img in images])
    sizes = np.unique(image_sizes, axis=0)
    for i in nearest:
        polygons = land_markups[i]['land'].values()
        is_nearest = nearest_index == i
        for size in sizes:
            scale = size / imgsz
            rpolygons = [polygon * scale for polygon in polygons]
            mask = glimpse.helpers.polygons_to_mask(rpolygons, size=size).astype(np.uint8)
            mask = sharedmem.copy(mask)
            for j in np.where(is_nearest & np.all(image_sizes == size, axis=1))[0]:
                masks[j] = mask
    return masks

# ---- Calibration controls ----

def svg_controls(img, svg=None, keys=None, correction=True, step=None):
    """
    Return control objects for an Image.

    Arguments:
        img (Image): Image object
        svg: Path to SVG file (str) or parsed result (dict).
            If `None`, looks for SVG file 'svg/<image>.svg'.
        keys (iterable): SVG layers to include, or all if `None`
        correction: Whether control objects should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
    """
    if svg is None:
        basename = parse_image_path(img.path)['basename']
        svg = os.path.join(CG_PATH, 'svg', basename + '.svg')
    controls = []
    if isinstance(svg, (bytes, str)):
        if not os.path.isfile(svg):
            return controls
        svg = glimpse.svg.parse_svg(svg, imgsz=img.cam.imgsz)
    if keys is None:
        keys = svg.keys()
    for key in keys:
        if key in svg:
            if key == 'gcp':
                controls.append(gcp_points(img, svg[key], correction=correction))
            elif key == 'coast':
                controls.append(coast_lines(img, svg[key], correction=correction, step=step))
            elif key == 'terminus':
                controls.append(terminus_lines(img, svg[key], correction=correction, step=step))
            elif key == 'moraines':
                controls.extend(moraines_mlines(img, svg[key], correction=correction, step=step))
            elif key == 'horizon':
                controls.append(horizon_lines(img, svg[key], correction=correction, step=step))
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

def coast_lines(img, markup, correction=True, step=None):
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
    return glimpse.optimize.Lines(img.cam, luv, lxyz, correction=correction, step=step)

def terminus_lines(img, markup, correction=True, step=None):
    """
    Return terminus Lines object for an Image.

    Arguments:
        img (Image): Image object
        markup (dict): Parsed SVG layer
        correction: Whether Lines should use elevation correction (bool)
            or arguments to `glimpse.helpers.elevation_corrections()`
    """
    luv = tuple(markup.values())
    # HACK: Select terminus with matching date and preferred type
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'termini.geojson'), crs=32606)
    date_str = img.datetime.strftime('%Y-%m-%d')
    features = [(feature, feature['properties']['type'])
        for feature in glimpse.helpers.geojson_iterfeatures(geo)
        if feature['properties']['date'] == date_str]
    type_order = ('aerometric', 'worldview', 'landsat-8', 'landsat-7', 'terrasar', 'tandem', 'arcticdem', 'landsat-5')
    order = [type_order.index(f[1]) for f in features]
    xy = features[np.argmin(order)][0]['geometry']['coordinates']
    xyz = np.hstack((xy, sea_height(xy, t=img.datetime)))
    return glimpse.optimize.Lines(img.cam, luv, [xyz], correction=correction, step=step)

def horizon_lines(img, markup, correction=True, step=None):
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
    return glimpse.optimize.Lines(img.cam, luv, lxyz, correction=correction, step=step)

def moraines_mlines(img, markup, correction=True, step=None):
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
        mlines.append(glimpse.optimize.Lines(img.cam, luv, [xyz], correction=correction, step=step))
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

def synth_controls(img, step=None, directions=False):
    image = glimpse.helpers.strip_path(img.path)
    basename = os.path.join(CG_PATH, 'svg-synth', image)
    controls = []
    # Load svg
    path = basename + '.svg'
    if os.path.isfile(path):
        svg = glimpse.svg.parse_svg(path, imgsz=img.cam.imgsz)
        if 'points' in svg or 'lines' in svg or 'points-auto' in svg:
            scam = glimpse.helpers.read_json(basename + '-synth.json')
            simg = glimpse.Image(basename + '-synth.JPG', cam=scam)
            if not directions:
                depth = glimpse.Raster.read(basename + '-depth.tif')
                scale = depth.n / img.cam.imgsz
            # NOTE: Ignoring parallax potential
        if 'points-auto' in svg:
            # Length-2 paths traced from image to synthetic image
            uv = np.vstack([x[0] for x in svg['points-auto'].values()])
            suv = np.vstack([x[1] for x in svg['points-auto'].values()])
            xyz = simg.cam.invproject(suv)
            if not directions:
                xyz = simg.cam.xyz + xyz * depth.sample(suv * scale).reshape(-1, 1)
            points = glimpse.optimize.Points(cam=img.cam, uv=uv, xyz=xyz,
                directions=directions, correction=False)
            controls.append(points)
        if 'points' in svg:
            # Length-2 paths traced from image to synthetic image
            uv = np.vstack([x[0] for x in svg['points'].values()])
            suv = np.vstack([x[1] for x in svg['points'].values()])
            xyz = simg.cam.invproject(suv)
            if not directions:
                xyz = simg.cam.xyz + xyz * depth.sample(suv * scale).reshape(-1, 1)
            points = glimpse.optimize.Points(cam=img.cam, uv=uv, xyz=xyz,
                directions=directions, correction=False)
            controls.append(points)
        if 'lines' in svg:
            for layer in svg['lines'].values():
                # Group with paths named 'image*' and 'synth*'
                uvs = [layer[key] for key in layer if key.find('image') == 0]
                suvs = [layer[key] for key in layer if key.find('synth') == 0]
                xyzs = [simg.cam.invproject(suv) for suv in suvs]
                if not directions:
                    depths = [depth.sample(suv * scale).reshape(-1, 1) for suv in suvs]
                    xyzs = [simg.cam.xyz + xyz * depth for xyz, depth in zip(xyzs, depths)]
                lines = glimpse.optimize.Lines(cam=img.cam, uvs=uvs, xyzs=xyzs,
                    step=step, directions=directions, correction=False)
                controls.append(lines)
    return controls

# ---- Control bundles ----

def station_svg_controls(station, size=1, force_size=False, keys=None,
    svgs=None, correction=True, step=None, station_calib=False, camera_calib=True,
    synth=True):
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
    paths = glob.glob(os.path.join(CG_PATH, 'svg', station + '*.svg'))
    if synth:
        paths += glob.glob(os.path.join(CG_PATH, 'svg-synth', station + '*.pkl'))
        paths += glob.glob(os.path.join(CG_PATH, 'svg-synth', station + '*.svg'))
    basenames = np.unique([glimpse.helpers.strip_path(path) for path in paths])
    images, controls, cam_params = [], [], []
    for basename in basenames:
        calibration = load_calibrations(basename, camera=camera_calib,
            station=station_calib, station_estimate=not station_calib, merge=True)
        img_path = find_image(basename)
        img = glimpse.Image(img_path, cam=calibration)
        control = []
        if svgs is None or basename in svgs:
            control += svg_controls(img, keys=keys, correction=correction, step=step)
        if synth:
            control += synth_controls(img, step=None, directions=False)
        if control:
            for x in control:
                x.resize(size, force=force_size)
            images.append(img)
            controls.extend(control)
            cam_params.append(dict(viewdir=True))
    return images, controls, cam_params

def camera_svg_controls(camera, size=1, force_size=False, keys=None,
    svgs=None, correction=True, step=None, station_calib=False,
    camera_calib=False, synth=True):
    """
    Return all SVG control objects available for a camera.

    Arguments:
        camera (str): Camera identifer
        size: Image scale factor (number) or image size in pixels (nx, ny)
        force_size (bool): Whether to force `size` even if different aspect ratio
            than original size.
        keys (iterable): SVG layers to include
        svgs (iterable): SVG basenames to include
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
    paths = glob.glob(os.path.join(CG_PATH, 'svg', '*.svg'))
    if synth:
        paths += glob.glob(os.path.join(CG_PATH, 'svg-synth', '*.pkl'))
        paths += glob.glob(os.path.join(CG_PATH, 'svg-synth', '*.svg'))
    basenames = np.unique([glimpse.helpers.strip_path(path) for path in paths])
    images, controls, cam_params = [], [], []
    for basename in basenames:
        ids = parse_image_path(basename, sequence=True)
        if ids['camera'] == camera:
            calibration = load_calibrations(basename, camera=camera_calib,
                station=station_calib, station_estimate=not station_calib, merge=True)
            img_path = find_image(basename)
            img = glimpse.Image(img_path, cam=calibration)
            control = []
            if svgs is None or basename in svgs:
                control += svg_controls(img, keys=keys, correction=correction, step=step)
            if synth:
                control += synth_controls(img, step=None, directions=False)
            if control:
                for x in control:
                    x.resize(size, force=force_size)
                images.append(img)
                controls.extend(control)
                cam_params.append(dict(viewdir=True))
    return images, controls, cam_params

def camera_motion_matches(camera, size=None, force_size=False,
    station_calib=False, camera_calib=False):
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

    Returns:
        list: Image objects
        list: Matches objects
        list: Per-camera calibration parameters [{}, {'viewdir': True}, ...]
    """
    motion = glimpse.helpers.read_json(os.path.join(CG_PATH, 'motion.json'))
    sequences = [item['paths'] for item in motion
        if parse_image_path(item['paths'][0], sequence=True)['camera'] == camera]
    all_images, all_matches, cam_params = [], [], []
    for sequence in sequences:
        paths = [find_image(path) for path in sequence]
        cams = [load_calibrations(path,  camera=camera_calib,
            station=station_calib, station_estimate=not station_calib, merge=True)
            for path in paths]
        images = [glimpse.Image(path, cam=cam)
            for path, cam in zip(paths, cams)]
        matches = [load_motion_match(images[i], images[i + 1])
            for i in range(len(sequence) - 1)]
        if size is not None:
            for match in matches:
                match.resize(size, force=force_size)
        all_images.extend(images)
        all_matches.extend(matches)
        cam_params.extend([dict()] + [dict(viewdir=True)] * (len(sequence) - 1))
    return all_images, all_matches, cam_params

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

def load_motion_match(imgA, imgB):
    """
    Returns motion Matches object for an Image pair.

    Arguments:
        imgA (Image): Image object
        imgB (Image): Image object
    """
    basename = glimpse.helpers.strip_path(imgA.path) + '-' + glimpse.helpers.strip_path(imgB.path)
    path = os.path.join(CG_PATH, 'motion', basename + '.pkl')
    match = glimpse.helpers.read_pickle(path)
    match.cams = (imgA.cam, imgB.cam)
    return match

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
            If `path` or `image` specified, `viewdir` is based on the position
            of the image in the motion break sequence.
        station: Whether to load station (bool) or
            station identifier to load (str).
            If `True`, the station identifier is parsed from `path`.
            viewdir is loaded from station_estimate.
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
    def _try_except(fun, arg, **kwargs):
        try:
            return fun(arg, **kwargs)
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
        img_path = image if isinstance(image, str) else path if path else None
        calibrations['station_estimate'] = _try_except(_load_station_estimate, station_estimate, path=img_path)
    if station:
        calibrations['station'] = _try_except(_load_station_estimate, station, path=path)
        station_calib = _try_except(_load_station, station)
        if station_calib and 'xyz' in station_calib:
            calibrations['station']['xyz'] = station_calib['xyz']
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

def _load_station_estimate(station, path=None):
    stations = Stations()
    feature = stations[station]
    viewdir = feature['properties']['viewdir']
    i = _station_break_index(path) if path else 0
    if i and 'viewdir' in feature['properties']['breaks'][i - 1]:
        viewdir = feature['properties']['breaks'][i - 1]['viewdir']
    return dict(
        xyz=np.reshape(feature['geometry']['coordinates'], -1),
        viewdir=viewdir)

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
            If `None`, these are read from `images[i].cam.viewdir`.
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
