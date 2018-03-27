import os
CG_PATH = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(CG_PATH, '..'))
import glimpse
from glimpse.imports import (np, pandas, re, datetime)
import glob
import requests
import backports.functools_lru_cache

# ---- Environment variables ---

print 'cg: Remember to set IMAGE_PATH'
# print 'cg: Remember to set IMAGE_PATH, KEYPOINT_PATH, and/or MATCH_PATH'
IMAGE_PATH = None
# KEYPOINT_PATH = None
# MATCH_PATH = None

# ---- Images ----

@backports.functools_lru_cache.lru_cache(maxsize=1)
def Sequences():
    df = pandas.read_csv(
        os.path.join(CG_PATH, 'sequences.csv'),
        parse_dates=['first_time_utc', 'last_time_utc'])
    # Floor start time subseconds for comparisons to filename times
    df.first_time_utc = df.first_time_utc.apply(
        datetime.datetime.replace, microsecond=0)
    return df

def parse_image_path(path):
    basename = os.path.splitext(os.path.basename(path))[0]
    station, date_str, time_str = re.findall('^([^_]+)_([0-9]{8})_([0-9]{6})', basename)[0]
    capture_time = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
    sequences = Sequences()
    is_row = ((sequences.station == station) &
        (sequences.first_time_utc <= capture_time) &
        (sequences.last_time_utc >= capture_time))
    rows = np.where(is_row)[0]
    if len(rows) != 1:
        raise ValueError('Image path has zero or multiple sequence matches: ' + path)
    # NOTE: All strings expected to be type str (i.e. default '')
    return dict(
        station=station.encode(), service=sequences.service.iloc[rows[0]],
        camera=sequences.camera.iloc[rows[0]], basename=basename.encode(),
        datetime=capture_time, date_str=date_str.encode(), time_str=time_str.encode())

def find_image(path):
    ids = parse_image_path(path)
    service_dir = os.path.join(IMAGE_PATH, ids['station'],
        ids['station'] + '_' + ids['service'])
    filename = ids['basename'] + '.JPG'
    if os.path.isdir(service_dir):
        subdirs = [''] + next(os.walk(service_dir))[1]
        for subdir in subdirs:
            image_path = os.path.join(service_dir, subdir, filename)
            if os.path.isfile(image_path):
                return image_path

# ---- Calibration controls ----

def svg_controls(img, svg, keys=None, correction=True):
    if isinstance(svg, str):
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
    uv = np.vstack(markup.values())
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'gcp.geojson'), key='id', crs=32606)
    xyz = np.vstack((geo['features'][key]['geometry']['coordinates']
        for key in markup.iterkeys()))
    return glimpse.optimize.Points(img.cam, uv, xyz, correction=correction)

def coast_lines(img, markup, correction=True):
    luv = markup.values()
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'coast.geojson'), crs=32606)
    lxy = [feature['geometry']['coordinates'] for feature in geo['features']]
    lxyz = [np.hstack((xy, sea_height(xy, t=img.datetime))) for xy in lxy]
    return glimpse.optimize.Lines(img.cam, luv, lxyz, correction=correction)

def terminus_lines(img, markup, correction=True):
    luv = markup.values()
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'termini.geojson'), key='date', crs=32606)
    date_str = img.datetime.strftime('%Y-%m-%d')
    xy = geo['features'][date_str]['geometry']['coordinates']
    xyz = np.hstack((xy, sea_height(xy, t=img.datetime)))
    return glimpse.optimize.Lines(img.cam, luv, [xyz], correction=correction)

def horizon_lines(img, markup, correction=True):
    luv = markup.values()
    station = parse_image_path(img.path)['station']
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'horizons', station + '.geojson'), crs=32606)
    lxyz = [coords for coords in glimpse.helpers.geojson_itercoords(geo)]
    return glimpse.optimize.Lines(img.cam, luv, lxyz, correction=correction)

def moraines_mlines(img, markup, correction=True):
    date_str = img.datetime.strftime('%Y%m%d')
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'moraines', date_str + '.geojson'), key='id', crs=32606)
    mlines = []
    for key, moraine in markup.iteritems():
        luv = moraine.values()
        xyz = geo['features'][key]['geometry']['coordinates']
        mlines.append(glimpse.optimize.Lines(img.cam, luv, [xyz], correction=correction))
    return mlines

def sea_height(xy, t=None):
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
    svg_paths = glob.glob(os.path.join(CG_PATH, 'svg', station + '*.svg'))
    images, controls, cam_params = [], [], []
    for svg_path in svg_paths:
        calibration = load_calibration(svg_path, camera=camera_calib,
            station=station_calib, station_estimate=not station_calib)
        img_path = find_image(svg_path)
        images.append(glimpse.Image(img_path, cam=calibration))
        images[-1].cam.resize(size, force=force_size)
        controls.extend(svg_controls(images[-1], svg_path, keys=keys, correction=correction))
        cam_params.append(dict(viewdir=True))
    return images, controls, cam_params

def camera_svg_controls(camera, size=1, force_size=False, keys=None,
    correction=True, station_calib=False, camera_calib=False, fixed=True):
    svg_paths = glob.glob(os.path.join(CG_PATH, 'svg', '*.svg'))
    images, controls, cam_params = [], [], []
    for svg_path in svg_paths:
        ids = parse_image_path(svg_path)
        if ids['camera'] == camera:
            calibration = load_calibration(svg_path, camera=camera_calib,
                station=station_calib, station_estimate=not station_calib)
            img_path = find_image(svg_path)
            images.append(glimpse.Image(img_path, cam=calibration))
            images[-1].cam.resize(size, force=force_size)
            controls.extend(svg_controls(images[-1], svg_path, keys=keys, correction=correction))
            params = dict(viewdir=True)
            if fixed is False:
                params['xyz'] = True
            cam_params.append(params)
    return images, controls, cam_params

def camera_motion_matches(camera, size=1, force_size=False,
    station_calib=False, camera_calib=False, detect=dict(), match=dict()):
    motion = glimpse.helpers.read_json(os.path.join(CG_PATH, 'motion.json'))
    sequences = [item['paths'] for item in motion
        if parse_image_path(item['paths'][0])['camera'] == camera]
    images, matches, cam_params = [], [], []
    for sequence in sequences:
        sys.stdout.write('.')
        sys.stdout.flush()
        paths = [find_image(path) for path in sequence]
        images.extend([glimpse.Image(path,
            cam=load_calibration(path,  camera=camera_calib,
            station=station_calib, station_estimate=not station_calib))
            for path in paths])
        idx = slice(-len(sequence), None)
        for img in images[idx]:
            img.cam.resize(size, force=force_size)
        matches.extend(build_sequential_matches(images[idx], detect=detect, match=match))
        cam_params.extend([dict()] + [dict(viewdir=True)] * (len(sequence) - 1))
    return images, matches, cam_params

def build_sequential_matches(images, detect=dict(), match=dict()):
    keypoints = [glimpse.optimize.detect_keypoints(img.read(), **detect) for img in images]
    matches = []
    for i in range(len(images) - 1):
        uvA, uvB = glimpse.optimize.match_keypoints(keypoints[i], keypoints[i + 1], **match)
        matches.append(glimpse.optimize.Matches(
            cams=(images[i].cam, images[i + 1].cam), uvs=(uvA, uvB)))
    return matches

# ---- Calibrations ----

def load_calibration(path=None, station=False, camera=False, image=False,
    station_estimate=False, **kwargs):
    calibration = dict()
    if path:
        ids = parse_image_path(path)
        if station is True or (station is False and station_estimate is True):
            station = ids['station']
        if camera is True:
            camera = ids['camera']
        if image is True:
            image = ids['basename']
    if isinstance(station, str):
        if station_estimate:
            calibration = glimpse.helpers.merge_dicts(calibration, load_station_estimate(station))
        else:
            calibration = glimpse.helpers.merge_dicts(calibration, load_station(station))
    if isinstance(camera, str):
        calibration = glimpse.helpers.merge_dicts(calibration, load_camera(camera))
    if isinstance(image, str):
        calibration = glimpse.helpers.merge_dicts(calibration, load_image(image))
    return glimpse.helpers.merge_dicts(calibration, kwargs)

def load_station_estimate(station):
    geo = glimpse.helpers.read_geojson(
        os.path.join(CG_PATH, 'geojson', 'stations.geojson'), crs=32606, key='id')
    return dict(
        xyz=np.reshape(geo['features'][station]['geometry']['coordinates'], -1),
        viewdir=geo['features'][station]['properties']['viewdir'])

def load_station(station):
    station_path = os.path.join(CG_PATH, 'stations', station + '.json')
    return glimpse.helpers.read_json(station_path)

def load_camera(camera):
    camera_path = os.path.join(CG_PATH, 'cameras', camera + '.json')
    return glimpse.helpers.read_json(camera_path)

def load_image(image):
    basename = os.path.splitext(os.path.basename(image))[0]
    image_path = os.path.join(CG_PATH, 'images', basename + '.json')
    return glimpse.helpers.read_json(image_path)

# ---- Helpers ----

def project_dem(cam, dem, array=None, mask=None, correction=True):
    if mask is None:
        mask = ~np.isnan(dem.Z)
    xyz = np.column_stack((dem.X[mask], dem.Y[mask], dem.Z[mask]))
    uv = cam.project(xyz, correction=correction)
    if array is None:
        array = dem.Z
    return cam.rasterize(uv, array[mask])
