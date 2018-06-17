import matplotlib
matplotlib.use('agg')
import cg
from cg import glimpse
from glimpse.imports import (sys, datetime, matplotlib, np, os)
import glob

root = '/volumes/science-b/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
cg.KEYPOINT_PATH = os.path.join(root, 'timelapse-keypoints')
cg.MATCHES_PATH = os.path.join(root, 'timelapse-matches')

# ---- Set script constants ----

STATIONS = (
    'CG04', 'CG05', 'CG06', 'AK01', 'AK01b', 'AK03', 'AK03b', 'AK09', 'AK09b',
    'AK10', 'AK10b', 'AK12', 'AKJNC', 'AKST03A', 'AKST03B')
SKIP_SEQUENCES = (
    'AK01_20070817', 'AK01_20080616',
    'AK01b_20080619', 'AK01b_20090824', 'AK01b_20090825', 'AK01b_20160908',
    'AK03_20070817', 'AK12_20100820')
SNAP = datetime.timedelta(hours=2)
MAXDT = datetime.timedelta(days=1)
MATCH_SEQ = np.concatenate((np.arange(12) + 1, (100, 300, 1000, 3000)))
MAX_RATIO = 0.6
MAX_ERROR = 0.03 # fraction of image width
N_MATCHES = 50
MIN_MATCHES = 50

# ---- Functions ----

def write_matches(matcher, **kwargs):
    matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=cg.MATCHES_PATH, overwrite=False, max_ratio=0.75,
        max_distance=None, parallel=4, weights=True,
        clear_keypoints=True, clear_matches=True, **kwargs)

def read_matches(matcher, **kwargs):
    matcher.build_matches(
        maxdt=MAXDT, seq=MATCH_SEQ,
        path=cg.MATCHES_PATH, overwrite=False, max_ratio=0.75,
        max_distance=None, parallel=True, weights=True,
        clear_keypoints=True, clear_matches=False,
        as_type=glimpse.optimize.RotationMatches,
        filter=dict(
            min_weight=1 / MAX_RATIO, max_error=MAX_ERROR,
            n_best=N_MATCHES, scaled=True),
        **kwargs)

# ---- Find calibrated sequences ----

sequences = cg.Sequences()
cameras = np.intersect1d(
    sequences.camera,
    [glimpse.helpers.strip_path(path) for path in glob.glob(os.path.join(cg.CG_PATH, 'cameras', '*.json'))])
stations = STATIONS
calibrated = np.isin(sequences.camera, cameras) & np.isin(sequences.station, stations)
skipped = np.isin(
    [station + '_' + service for station, service in zip(sequences.station, sequences.service)],
    SKIP_SEQUENCES)
sequence_dicts = sequences.loc[calibrated & ~skipped].to_dict(orient='records')
station_services = {station: sequences.loc[calibrated & ~skipped & (sequences.station == station)].service.tolist()
    for station in sequences.loc[calibrated & ~skipped].station.unique()}

# ---- Build keypoints ----

for station in station_services:
    services = station_services[station]
    print(station, services)
    images = cg.load_images(
        station=station, services=services, snap=SNAP,
        use_exif=False, service_exif=True, anchors=True)
    masks = cg.load_masks(images)
    matcher = glimpse.optimize.KeypointMatcher(images)
    matcher.build_keypoints(masks=masks, contrastThreshold=0.02, overwrite=False,
        clear_images=True, clear_keypoints=True, parallel=4)

# ---- Build keypoint matches ----

for station in station_services:
    # Load all station sequences to match across service breaks
    services = station_services[station]
    print(station, services)
    images = cg.load_images(
        station=station, services=services, snap=SNAP,
        use_exif=False, service_exif=True, anchors=True)
    # Build matches
    matcher = glimpse.optimize.KeypointMatcher(images)
    write_matches(matcher)

# ---- Compute view directions ----

tile_size = 1500
tile_overlap = 12

for station in station_services:
    services = station_services[station]
    print(station, services)
    images = cg.load_images(
        station=station, services=services, snap=SNAP,
        use_exif=False, service_exif=True, anchors=True,
        viewdir=False, viewdir_as_anchor=False)
    matcher = glimpse.optimize.KeypointMatcher(images)
    # Split sequence into overlapping tiles
    starts = np.arange(0, len(images), tile_size)
    ends = np.concatenate((starts[1:], [len(images)]))
    starts[1:] -= tile_overlap
    indices = np.arange(len(images))
    ntiles = len(starts)
    for tile in range(ntiles):
        if tile:
            # Load images again, this time with view directions
            images = cg.load_images(
                station=station, services=services, snap=SNAP,
                use_exif=False, service_exif=True, anchors=True,
                viewdir=True, viewdir_as_anchor=True)
            matcher = glimpse.optimize.KeypointMatcher(images[:ends[tile]])
            # De-anchor images in tile overlap
            for img in matcher.images[starts[tile]:ends[tile - 1]]:
                img.anchor = False
        # Load matches for tile
        read_matches(matcher, imgs=indices[starts[tile]:ends[tile]])
        # Remove images with too few matches
        # NOTE: Repeat until no additional images are below threshold
        imgs = [None]
        while len(imgs):
            n = matcher.matches_per_image()
            imgs = np.where(n < MIN_MATCHES)[0]
            matcher.drop_images(imgs)
        # Check for breaks in remaining matches
        breaks = matcher.match_breaks()
        if len(breaks):
            raise ValueError('Match breaks at:', breaks)
        # Check for an anchor image
        is_anchor = np.array([img.anchor for img in matcher.images])
        anchors = np.where(is_anchor)[0]
        if not len(anchors):
            raise ValueError('No anchor image present')
        # Free up memory and convert matches to XY
        matcher.filter_matches(clear_weights=True)
        matcher.convert_matches(glimpse.optimize.RotationMatchesXY, clear_uvs=True)
        # Orient cameras
        cams = [img.cam for img in matcher.images]
        controls = tuple(matcher.matches.data)
        cam_params = [dict() if img.anchor else dict(viewdir=True)
            for img in matcher.images]
        model = glimpse.optimize.Cameras(cams, controls, cam_params=cam_params)
        fit = model.fit(ftol=1, full=True, loss='soft_l1')
        model.set_cameras(fit.params)
        # Orient cameras (alternate)
        # matcher.convert_matches(glimpse.optimize.RotationMatchesXYZ)
        # observer = glimpse.Observer(matcher.images, cache=False)
        # model = glimpse.optimize.ObserverCameras(observer, matcher.matches)
        # fit = model.fit(tol=1)
        # model.set_cameras(fit.x.reshape(-1, 3))
        # Write results for non-anchors
        # NOTE: Assumes anchor viewdirs are fixed
        cg.write_image_viewdirs(matcher.images[np.where(~is_anchor)[0]])

# ---- Verify view directions ----

station_gcps = dict(
    AK01='shorenode',
    AK01b='nostril',
    AK03='xcrack',
    AK03b='butt',
    AK09='butt',
    AK09b='kissers',
    AK10='slant',
    AK10b='slant',
    AK12='kissers',
    AKJNC='T20120813-05',
    AKST03A='portaledge',
    AKST03B='portaledge',
    CG04='cluster',
    CG05='cluster',
    CG06='cluster'
)
gcps = glimpse.helpers.read_geojson(
    path=os.path.join('geojson', 'gcp.geojson'), crs=32606, key='id')['features']
snap = datetime.timedelta(days=1)
tile_size = (300, 300)

for station in station_services:
    services = station_services[station]
    print(station, services)
    images = cg.load_images(
        station=station, services=services, snap=SNAP,
        use_exif=False, service_exif=True, anchors=True,
        viewdir=True, viewdir_as_anchor=True)
    # Keep only oriented images
    images = np.array([img for img in images if img.anchor])
    # Write animation(s)
    # HACK: Split at camera changes to use observer.animate()
    # HACK: Use k1 to quickly differentiate between cameras
    ks = np.array([img.cam.k[0] for img in images])
    for k in np.unique(ks):
        idx = np.where(ks == k)[0]
        sizes = np.row_stack([img.cam.imgsz for img in images[idx]])
        unique_sizes = np.unique(sizes, axis=0)
        if len(unique_sizes) > 1:
            # Standardize image sizes
            size = unique_sizes.min(axis=0)
            not_size = np.any(sizes != size, axis=1)
            f = images[~not_size][0].cam.f
            for img in images[not_size]:
                img.cam.resize(size, force=True)
                # HACK: Fix focal length rounding errors
                if any(img.cam.f - f > 0.1):
                    raise ValueError('Focal lengths cannot be reconciled')
                img.cam.f = f
        observer = glimpse.Observer(images[idx], cache=False).subset(snap=snap)
        path = os.path.join(
            'viewdirs-animations',
            glimpse.helpers.strip_path(observer.images[0].path) + '-' +
            glimpse.helpers.strip_path(observer.images[-1].path) + '.mp4')
        if not os.path.isfile(path):
            print(path)
            xyz = gcps[station_gcps[station]]['geometry']['coordinates']
            uv = observer.images[0].cam.project(xyz, correction=True)[0]
            ani = observer.animate(uv=uv, size=tile_size, interval=200,
                subplots=dict(figsize=(8, 4),
                gridspec_kw=dict(left=0.075, right=0.975, bottom=0.05, top=0.975,
                wspace=0.175, hspace=0.1)))
            ani.save(path, dpi=96)

# ---- Count dropped images ----

for station in station_services:
    services = station_services[station]
    images = cg.load_images(
        station=station, services=services, snap=SNAP,
        use_exif=False, service_exif=True, anchors=True,
        viewdir=True, viewdir_as_anchor=True)
    is_anchor = np.array([img.anchor for img in images])
    ndropped = len(images) - is_anchor.sum()
    pdropped = 100 * ndropped / len(images)
    print(station, ndropped, '(' + str(round(pdropped, 1)) + '%)')
