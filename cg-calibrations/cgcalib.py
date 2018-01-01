import os
import numpy as np
import sys
sys.path.append("../")
import helper
import optimize
import pandas
import re
import datetime
import svg
import image
import dem as DEM
import glob
import requests

DIR = os.path.dirname(__file__)

def parse_image_path(path):
    basename = os.path.basename(path)
    station, date_str, time_str = re.findall("^([^_]+)_([0-9]{8})_([0-9]{6})", basename)[0]
    capture_time = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    csv = pandas.read_csv(os.path.join(DIR, "sequences.csv"), parse_dates=['first_time_utc', 'last_time_utc'])
    # Floor seconds on left side of interval since subseconds not in filename
    csv.first_time_utc = csv.first_time_utc.apply(datetime.datetime.replace, microsecond=0)
    rows = csv[(csv.station == station) & (csv.first_time_utc <= capture_time) & (csv.last_time_utc >= capture_time)]
    if len(rows) != 1:
        raise ValueError("Image path has zero or multiple sequence matches")
    return dict(station=station, service=rows.service.iloc[0], camera=rows.camera.iloc[0],
        basename=station + "_" + date_str + "_" + time_str)

def find_image(path, root="."):
    parts = parse_image_path(path)
    service_path = os.path.join(root, parts['station'],
        parts['station'] + "_" + parts['service'])
    filename = parts['basename'] + ".JPG"
    if os.path.isdir(service_path):
        subdirs = [''] + next(os.walk(service_path))[1]
        for subdir in subdirs:
            image_path = os.path.join(service_path, subdir, filename)
            if os.path.isfile(image_path):
                return image_path

def station_eop(station):
    geo = helper.read_geojson(os.path.join(DIR, "geojson", "stations.geojson"), crs=32606, key="id")
    return dict(
        xyz=np.reshape(geo['features'][station]['geometry']['coordinates'], -1),
        viewdir=geo['features'][station]['properties']['viewdir'],
        fixed=geo['features'][station]['properties']['fixed'])

def svg_controls(img, svg_markup, keys=None):
    controls = []
    if isinstance(svg_markup, str):
        svg_markup = svg.parse_svg(svg_markup, imgsz=img.cam.imgsz)
    if keys is None:
        keys = svg_markup.keys()
    for key, markup in svg_markup.iteritems():
        if key in keys:
            if key == 'gcp':
                controls.append(gcp_points(img, markup))
            elif key == 'coast':
                controls.append(coast_lines(img, markup))
            elif key == 'terminus':
                controls.append(terminus_lines(img, markup))
            elif key == 'moraines':
                controls.extend(moraines_mlines(img, markup))
            elif key == 'horizon':
                controls.append(horizon_lines(img, markup))
    return controls

def gcp_points(img, markup):
    uv = np.vstack(markup.values())
    geo = helper.read_geojson(os.path.join(DIR, "geojson", "gcp.geojson"), key="id", crs=32606)
    xyz = np.vstack((geo['features'][key]['geometry']['coordinates']
        for key in markup.iterkeys()))
    return optimize.Points(img.cam, uv, xyz)

def coast_lines(img, markup):
    luv = markup.values()
    geo = helper.read_geojson(os.path.join(DIR, "geojson", "coast.geojson"), crs=32606)
    lxy = [feature['geometry']['coordinates'] for feature in geo['features']]
    lxyz = [np.hstack((xy, sea_height(xy, t=img.datetime))) for xy in lxy]
    return optimize.Lines(img.cam, luv, lxyz)

def terminus_lines(img, markup):
    luv = markup.values()
    geo = helper.read_geojson(os.path.join(DIR, "geojson", "termini.geojson"), key="date", crs=32606)
    date_str = img.datetime.strftime("%Y-%m-%d")
    xy = geo['features'][date_str]['geometry']['coordinates']
    xyz = np.hstack((xy, sea_height(xy, t=img.datetime)))
    return optimize.Lines(img.cam, luv, [xyz])

def horizon_lines(img, markup):
    luv = markup.values()
    station = parse_image_path(img.path)['station']
    geo = helper.read_geojson(os.path.join(DIR, "geojson", "horizons", station + ".geojson"), crs=32606)
    lxyz = [coords for coords in helper.geojson_itercoords(geo)]
    return optimize.Lines(img.cam, luv, lxyz)

def moraines_mlines(img, markup):
    date_str = img.datetime.strftime("%Y%m%d")
    geo = helper.read_geojson(os.path.join(DIR, "geojson", "moraines", date_str + ".geojson"), key="id", crs=32606)
    mlines = []
    for key, moraine in markup.iteritems():
        luv = moraine.values()
        xyz = geo['features'][key]['geometry']['coordinates']
        mlines.append(optimize.Lines(img.cam, luv, [xyz]))
    return mlines

def sea_height(xy, t=None):
    egm2008 = DEM.DEM.read(os.path.join(DIR, "egm2008.tif"))
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
        tide_height = np.interp((t - t_begin).total_seconds(), [0, (t_end - t_begin).total_seconds()], v)
    else:
        tide_height = 0
    return geoid_height + tide_height

def load_calibration(path=None, station=False, camera=False, image=False, **kwargs):
    calibration = dict()
    if path:
        ids = parse_image_path(path)
        if station is True:
            station = ids['station']
        if camera is True:
            camera = ids['camera']
    if isinstance(station, str):
        calibration = helper.merge_dicts(calibration, load_station(station))
    elif path and not station:
        eop = station_eop(ids['station'])
        station_dict = dict(xyz=eop['xyz'], viewdir=eop['viewdir'])
        calibration = helper.merge_dicts(calibration, station_dict)
    if isinstance(camera, str):
        calibration = helper.merge_dicts(calibration, load_camera(camera))
    if image:
        if path is None and isinstance(image, str):
            path = image
        calibration = helper.merge_dicts(calibration, load_image(path))
    return helper.merge_dicts(calibration, kwargs)

def load_station(station):
    station_path = os.path.join(DIR, "stations", station + ".json")
    return helper.read_json(station_path)

def load_camera(camera):
    camera_path = os.path.join(DIR, "cameras", camera + ".json")
    return helper.read_json(camera_path)

def load_image(image):
    basename = os.path.splitext(os.path.basename(image))[0]
    image_path = os.path.join(DIR, "images", basename + ".json")
    return helper.read_json(image_path)

def camera_motion_matches(camera, root=".", size=1, force_size=False, method="sift",
    station_calib=False, camera_calib=False, **kwargs):
    motion = helper.read_json(os.path.join(DIR, "motion.json"))
    sequences = [item['paths'] for item in motion
        if parse_image_path(item['paths'][0])['camera'] == camera]
    images, matches, cam_params = [], [], []
    for sequence in sequences:
        sys.stdout.write(".")
        sys.stdout.flush()
        idx = slice(-len(sequence), None)
        paths = [find_image(path, root=root) for path in sequence]
        images.extend([image.Image(path,
            cam=load_calibration(path, station=station_calib, camera=camera_calib))
            for path in paths])
        for img in images[idx]:
            img.cam.resize(size, force=force_size)
        if method == "sift":
            matches.extend(optimize.sift_matches(images[idx], **kwargs))
        elif method == "surf":
            matches.extend(optimize.surf_matches(images[idx], **kwargs))
        cam_params.extend([dict()] + [dict(viewdir=True)] * (len(sequence) - 1))
    return images, matches, cam_params

def camera_svg_controls(camera, root=".", size=1, force_size=False, fixed=True, keys=None,
    station_calib=False, camera_calib=False):
    svg_paths = glob.glob(os.path.join(DIR, "svg", "*.svg"))
    images, controls, cam_params = [], [], []
    for svg_path in svg_paths:
        ids = parse_image_path(svg_path)
        if ids['camera'] == camera:
            calibration = load_calibration(svg_path,
                station=station_calib, camera=camera_calib)
            img_path = find_image(svg_path, root=root)
            images.append(image.Image(img_path, cam=calibration))
            images[-1].cam.resize(size, force=force_size)
            controls.extend(svg_controls(images[-1], svg_path, keys=keys))
            params = dict(viewdir=True)
            if fixed is False:
                params['xyz'] = True
            cam_params.append(params)
    return images, controls, cam_params

def station_svg_controls(station, root=".", size=1, force_size=False, keys=None,
    station_calib=False, camera_calib=True):
    svg_paths = glob.glob(os.path.join(DIR, "svg", station + "*.svg"))
    images, controls, cam_params = [], [], []
    for svg_path in svg_paths:
        calibration = load_calibration(svg_path,
            station=station_calib, camera=camera_calib)
        img_path = find_image(svg_path, root=root)
        images.append(image.Image(img_path, cam=calibration))
        images[-1].cam.resize(size, force=force_size)
        controls.extend(svg_controls(images[-1], svg_path, keys=keys))
        cam_params.append(dict(viewdir=True))
    return images, controls, cam_params

def dem_to_image(cam, dem, array=None, mask=None):
    if mask is None:
        mask = ~np.isnan(dem.Z)
    xyz = np.column_stack((dem.X[mask], dem.Y[mask], dem.Z[mask]))
    uv = cam.project(xyz)
    if array is None:
        array = dem.Z
    return cam.rasterize(uv, array[mask])
