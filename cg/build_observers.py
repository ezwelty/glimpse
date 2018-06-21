import cg
from cg import glimpse
from glimpse.imports import (os, np, datetime, matplotlib)
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')

# Stations to include
stations = (
    'CG04', 'CG05', 'CG06', 'AK01', 'AK03', 'AK03b', 'AK01b', 'AK10', 'AK09',
    'AK09b', 'AKST03A', 'AKST03B', 'AK12', 'AKJNC', 'AK10b')
# HACK: Use same snap as in build_viewdirs.py, etc.
snap = datetime.timedelta(hours=2)
# Max rotations (instantaneous or cumulative) to allow within Observer
max_rotation = 1 # degrees about optical axis
max_pan_tilt = 2 # degrees pan or tilt
# Max temporal gap to allow within Observer
max_gap = datetime.timedelta(days=2)
# Minimum number of images to require within Observer
min_images = 2
# Minimum time span of Observer stack
min_dt = datetime.timedelta(days=1)
# Minimum number of Observers in stack
min_observers = 1
# Nominal Observer time span
nominal_bin_dt = datetime.timedelta(days=3)

# ---- Compute station ranges ----
# station_ranges, station_images, station_datetimes

# Range cuts:
# - Image size change
# - Camera change
# - Large instantaneous camera motion
# - Large temporal gaps
# Range cutouts:
# - Periods of excessive motion
# - Ranges with few images
# - (Skipped: Ranges with a small temporal span)

sequences = cg.Sequences()
station_ranges = dict()
station_iranges = dict()
station_images = dict()
station_datetimes = dict()
for station in stations:
    print(station)
    services = sequences[sequences.station == station].service.values
    images = cg.load_images(
        station=station, services=services, snap=snap,
        service_exif=True, anchors=False, viewdir=True, viewdir_as_anchor=True,
        file_errors=False)
    images = [img for img in images if img.anchor]
    images.sort(key=lambda x: x.datetime)
    datetimes = np.array([img.datetime for img in images])
    # Endpoints
    iranges = np.atleast_2d((0, len(images)))
    # Changes in image size
    width = [img.cam.imgsz[0] for img in images]
    cuts = [i + 1 for i in np.nonzero(np.diff(width))[0]]
    iranges = glimpse.helpers.cut_ranges(iranges, cuts)
    # Camera changes
    # HACK: Use k1 as a proxy
    k = [img.cam.k[0] for img in images]
    cuts = [i + 1 for i in np.nonzero(np.diff(k))[0]]
    iranges = glimpse.helpers.cut_ranges(iranges, cuts)
    # Gaps in coverage
    dt = np.diff(datetimes)
    cuts = [i + 1 for i in np.nonzero(dt > max_gap)[0]]
    iranges = glimpse.helpers.cut_ranges(iranges, cuts)
    # Sudden very large motions
    viewdirs = np.array([img.cam.viewdir for img in images])
    dtheta = np.diff(viewdirs, axis=0)
    pan_tilt = np.linalg.norm(dtheta[:, 0:2], axis=1)
    rotation = np.abs(dtheta[:, 2])
    breaks = (pan_tilt > max_pan_tilt) | (rotation > max_rotation)
    cuts = [i + 1 for i in np.nonzero(breaks)[0]]
    iranges = glimpse.helpers.cut_ranges(iranges, cuts)
    # NOTE: Switch to index-based ranges (from range-based)
    ranges = iranges - (0, 1)
    # Periods of excessive motion
    # matplotlib.pyplot.figure()
    for i, j in iranges:
        viewdirs = np.array([img.cam.viewdir for img in images[i:j]])
        bins = np.column_stack((datetimes[i:j], datetimes[i:j] + datetime.timedelta(days=3)))
        ibins = np.searchsorted(datetimes[i:j], bins)
        pan_tilt = [np.linalg.norm(viewdirs[m, 0:2] - viewdirs[m:n, 0:2], axis=1).max()
            for m, n in ibins]
        rotation = [np.abs(viewdirs[m, 2] - viewdirs[m:n, 2]).max() for m, n in ibins]
        peaks = (np.array(pan_tilt) > max_pan_tilt) | (np.array(rotation) > max_rotation)
        # # (plot)
        # matplotlib.pyplot.plot(datetimes[i:j], pan_tilt, color='grey')
        # matplotlib.pyplot.plot(datetimes[i:j], rotation, color='black')
        # matplotlib.pyplot.plot(datetimes[i:j], peaks, color='red')
        # # (print)
        # print(i, j)
        # print(''.join(np.array(['.', 'o'])[peaks.astype(int)]))
        # (cut)
        cutouts = [(i, i + 1) for i in np.nonzero(peaks)[0]]
        ranges = glimpse.helpers.cut_out_ranges(ranges, cutouts)
    # Remove ranges with < 2 images
    not_small = np.diff(ranges, axis=1) + 1 > 1
    ranges = ranges[not_small.ravel()]
    # Convert to datetime
    tranges = datetimes[ranges]
    # # Remove small ranges (time)
    # not_small = np.diff(tranges, axis=1) >= min_dt
    # tranges = tranges[not_small.ravel()]
    # ranges = ranges[not_small.ravel()]
    # Save results
    station_iranges[station] = ranges
    station_ranges[station] = tranges
    station_images[station] = images
    station_datetimes[station] = datetimes

# ---- Count dropped images ----

print('--- Image loss (after station filtering) ----')
for station in stations:
    n = len(station_images[station])
    x = np.arange(n)
    nf = np.unique(np.concatenate([x[i:(j + 1)]
        for i, j in station_iranges[station]])).size
    dropped = n - nf
    print(station, dropped, '(' + str(round(100 * dropped / n, 1)) + '%)')

# ---- Combine station ranges ----
# coverage, station_coverage

# Cut all ranges at all range endpoints
ranges = np.vstack([ranges for ranges in station_ranges.values()])
cuts = np.unique(ranges.ravel())
cut_ranges = glimpse.helpers.cut_ranges(ranges, cuts)
# Flatten to coverage
unique_cut_ranges = np.array(list({tuple(r) for r in cut_ranges}))
order = np.lexsort((unique_cut_ranges[:, 1], unique_cut_ranges[:, 0]))
coverage = unique_cut_ranges[order]
# Append stubs to coverage
original_coverage = coverage.copy()
station_coverage = {station: coverage.copy() for station in stations}
nmerged = 1
masked = np.zeros(len(coverage), dtype=bool)
indices = np.arange(len(coverage))
while nmerged:
    is_small = np.diff(coverage[~masked], axis=1).ravel() < min_dt
    idx = indices[~masked][is_small]
    idx_masked = np.nonzero(is_small)[0]
    nmerged = 0
    for k in np.arange(len(idx)):
        # i: full array, im: masked array (ignores merged ranges)
        i, im = idx[k], idx_masked[k]
        r = coverage[i]
        # Test whether (masked) neighbors are adjacent and not small
        if im > 0:
            il = indices[~masked][im - 1]
            rl = coverage[il]
            left = r[0] == rl[1] and rl[1] - rl[0] >= min_dt
        else:
            left = False
        if im < (~masked).sum() - 1:
            ir = indices[~masked][im + 1]
            rr = coverage[ir]
            right = r[1] == rr[0] and rr[1] - rr[0] >= min_dt
        else:
            right = False
        # Skip if no such neighbors
        if not (left or right):
            continue
        # Check which stations do not have a break that would prohibit merging
        left_merges, right_merges = [], []
        if left:
            left_merges = [station for station, ranges in station_ranges.items()
                if np.any((ranges[:, 0] < r[0]) & (ranges[:, 1] > r[0]))]
        if right:
            right_merges = [station for station, ranges in station_ranges.items()
                if np.any((ranges[:, 0] < r[1]) & (ranges[:, 1] > r[1]))]
        # If both sides, merge into most merges or smaller neighbor if equal
        if left and right:
            if len(left_merges) == len(right_merges):
                left = (rl[1] - rl[0]) < (rr[1] - rr[0])
            else:
                left = len(left_merges) > len(right_merges)
            right = not left
            merges = left_merges if left else right_merges
        else:
            merges = left_merges + right_merges
        if left:
            # Extend previous coverage to the right
            coverage[il, 1] = r[1]
            for station in merges:
                station_coverage[station][il, 1] = r[1]
        else:
            # Extend next coverage to the left
            coverage[ir, 0] = r[0]
            for station in merges:
                station_coverage[station][ir, 0] = r[0]
        if merges:
            # Mask out stub, reload indices
            masked[i] = True
            idx_masked[idx_masked > im] -= 1
            nmerged += 1
# Remove dangling stubs from coverage
not_small = np.diff(coverage, axis=1) >= min_dt
coverage = coverage[not_small.ravel()]
for station, ranges in station_coverage.items():
    not_small = np.diff(ranges, axis=1) >= min_dt
    station_coverage[station] = ranges[not_small.ravel()]
# Check that each station has same number of coverage ranges
assert np.all(np.equal(
    len(coverage),
    [len(ranges) for ranges in station_coverage.values()]))

# ---- Print dropped coverage ----

dt_original = np.diff(original_coverage, axis=1).sum()
dt = np.diff(coverage, axis=1).sum()
print('Dropped coverage (total):', dt_original - dt)

# ---- Build Observers ----

# Choose bin sizes that evenly divide each range and least deviate from ideal
# NOTE: Convert datetimes and timedeltas to float seconds to avoid ms rounding
coverageS = np.array([xi.timestamp() for xi in coverage.flat]).reshape(coverage.shape)
nominal_bin_dtS = nominal_bin_dt.total_seconds()
dts = np.diff(coverageS, axis=1).ravel()
nbins = dts / nominal_bin_dtS
nbins[nbins < 1] = 1
smaller = dts / np.ceil(nbins)
larger = dts / np.floor(nbins)
dsmaller = np.abs(smaller - nominal_bin_dtS)
dlarger = np.abs(larger - nominal_bin_dtS)
bin_dts = np.where(dsmaller < dlarger, smaller, larger)
nbins = np.where(dsmaller < dlarger, np.ceil(nbins), np.floor(nbins)).astype(int)
# Compute Observer ranges
observer_ranges = []
for r, dt, n in zip(coverageS, bin_dts, nbins):
    endpoints = [datetime.datetime.fromtimestamp(r[0] + ni * dt)
        for ni in range(n + 1)]
    ranges = np.column_stack((endpoints[:-1], endpoints[1:])).astype(datetime.datetime)
    observer_ranges.append(ranges)
# Build Observer image lists
observers = []
for i, ranges in enumerate(observer_ranges):
    temp_basenames = [dict() for _ in range(len(ranges))]
    for station in stations:
        crop = station_coverage[station][i]
        selected = ((crop[0] <= station_datetimes[station]) &
            (crop[1] >= station_datetimes[station]))
        if not np.any(selected):
            continue
        images = np.asarray(station_images[station])[selected]
        datetimes = np.asarray(station_datetimes[station])[selected]
        previous_last = None
        for j, r in enumerate(ranges):
            selected = (r[0] <= datetimes) & (r[1] >= datetimes)
            n = np.count_nonzero(selected)
            if (n + bool(previous_last)) < min_images:
                if n:
                    previous_last = glimpse.helpers.strip_path(
                        images[selected][-1].path)
                else:
                    previous_last = None
                continue
            basenames = [glimpse.helpers.strip_path(img.path)
                for img in images[selected]]
            if previous_last:
                # HACK: Start with previous last image
                basenames.insert(0, previous_last)
            temp_basenames[j][station] = basenames
            previous_last = basenames[-1]
    observers += temp_basenames
# Remove small Observers
for i, obs in enumerate(observers):
    n = len(obs)
    if n < min_observers:
        print('Dropped', i, '-', n, 'observers')
        _ = observers.pop(i)
    else:
        paths = np.concatenate(tuple(obs.values()))
        datetimes = [cg.parse_image_path(path)['datetime'] for path in paths]
        dt = np.max(datetimes) - np.min(datetimes)
        if dt < min_dt:
            print('Dropped', i, '-', dt, 'span')
            _ = observers.pop(i)
# Write to file
glimpse.helpers.write_json(observers, path='observers.json', indent=4,
    flat_arrays=True)

# ---- Count dropped images ----

print('--- Image loss (final) ----')
station_paths = {station: [] for station in stations}
for obs in observers:
    for station in obs:
        station_paths[station] += obs[station]
for station in stations:
    n = len(station_images[station])
    dropped = n - np.unique(station_paths[station]).size
    print(station, dropped, '(' + str(round(100 * dropped / n, 1)) + '%)')

# ---- Plot station ranges ----

# matplotlib.pyplot.figure(tight_layout=True, figsize=(18, 6))
# for y, station in enumerate(stations[::-1]):
#     matplotlib.pyplot.gca().axhline(y, linestyle='dotted', color='black',
#         alpha=0.25)
#     ranges = station_ranges[station]
#     for i, r in enumerate(ranges):
#         start, end = [matplotlib.dates.date2num(x) for x in r]
#         matplotlib.pyplot.barh(y, width=end - start, left=start, height=1,
#             color='red' if i % 2 else 'black', edgecolor='none')
# matplotlib.pyplot.gca().xaxis_date()
# matplotlib.pyplot.yticks(range(len(stations)), stations[::-1])
# matplotlib.pyplot.savefig('station_ranges.pdf')
