import cg
from cg import glimpse
from glimpse.imports import (np, os, matplotlib, collections, datetime)
import glob

observer_json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)
template = glimpse.Raster.read(os.path.join('points', 'template.tif'))

def tracks_basename(i):
    date_str = min([cg.parse_image_path(basenames[0])['date_str']
        for basenames in observer_json[i].values()])
    return date_str + '-' + str(i)

def tracks_path(i, suffixes=None):
    basename = tracks_basename(i)
    basepath = os.path.join('tracks', basename)
    if suffixes is not None:
        return [basepath + '-' + suffix + '.pkl' for suffix in suffixes]
    else:
        return basepath + '.pkl'

# https://github.com/numpy/numpy/issues/8708
def take_along_axis(arr, ind, axis):
    if axis < 0:
       if axis >= -arr.ndim:
           axis += arr.ndim
       else:
           raise IndexError('axis out of range')
    ind_shape = (1,) * ind.ndim
    ins_ndim = ind.ndim - (arr.ndim - 1)   # inserted dimensions
    dest_dims = list(range(axis)) + [None] + list(range(axis+ins_ndim, ind.ndim))
    inds = []
    for dim, n in zip(dest_dims, arr.shape):
        if dim is None:
            inds.append(ind)
        else:
            ind_shape_dim = ind_shape[:dim] + (-1,) + ind_shape[dim+1:]
            inds.append(np.arange(n).reshape(ind_shape_dim))
    return arr[tuple(inds)]

def normal_weighted_mean(means, sigmas, axis=None, correlated=False):
    isnan_mean = np.isnan(means)
    isnan_sigmas = np.isnan(sigmas)
    if (isnan_mean != isnan_sigmas).any():
        raise ValueError('mean and sigma NaNs do not match')
    if correlated:
        # Reorder nan to end
        order = np.argsort(isnan_mean, axis=axis)
        means = take_along_axis(means, order, axis=axis)
        sigmas = take_along_axis(sigmas, order, axis=axis)
    if (sigmas == 0).any():
        raise ValueError('sigmas cannot be 0')
    weights = sigmas**-2
    weights *= np.expand_dims(1 / np.nansum(weights, axis=axis), axis)
    wmeans = np.nansum(weights * means, axis=axis)
    isnan = isnan_mean.all(axis=axis)
    wmeans[isnan] = np.nan
    variances = np.nansum(weights**2 * sigmas**2, axis=axis)
    variances[isnan] = np.nan
    if correlated:
        ab = np.product(weights.take(range(2), axis), axis=axis)
        single = np.isnan(weights.take(range(2), axis)).sum(axis=axis) == 1
        ab[single] = 0
        variances += 2 * np.nansum(weights.take(
            range(2, weights.shape[axis]), axis), axis=axis) + 2 * ab
    return wmeans, np.sqrt(variances)

# ---- Check coverage ----

for i in range(len(observer_json)):
    # paths = tracks_path(i, ('f', 'fv', 'r', 'rv'))
    paths = [tracks_path(i)]
    for path in paths:
        if not os.path.isfile(path):
            print(path, 'missing')

# ---- Merge tracks -----

for i_obs in range(len(observer_json)):
    basename = tracks_basename(i_obs)
    basepath = os.path.join('tracks', basename)
    if os.path.isfile(basename + '.pkl'):
        continue
    suffixes = ('f', 'fv', 'r', 'rv')
    try:
        tracks = [glimpse.helpers.read_pickle(basepath + '-' + suffix + '.pkl')
            for suffix in suffixes]
        print(basename)
    except FileNotFoundError:
        print(basename, 'not found')
        continue
    # Combine repeat runs (same direction, different initial state)
    for i in (0, 2):
        # # Choose run with smallest sigma at each timestep (and each variable)
        # mask = tracks[i].sigmas > tracks[i + 1].sigmas
        # tracks[i].means[mask] = tracks[i + 1].means[mask]
        # tracks[i].sigmas[mask] = tracks[i + 1].sigmas[mask]
        # Choose run with the smallest mean standard deviation for vx + vy
        sd0 = np.nanmean(np.sqrt(
            np.nansum(tracks[i].sigmas[:, :, 0:2]**2, axis=2)), axis=1)
        sd1 = np.nanmean(np.sqrt(
            np.nansum(tracks[i + 1].sigmas[:, :, 0:2]**2, axis=2)), axis=1)
        mask = sd0 > sd1
        tracks[i].means[mask, :, :] = tracks[i + 1].means[mask, :, :]
        tracks[i].sigmas[mask, :, :] = tracks[i + 1].sigmas[mask, :, :]
    # Combine (merged) reverse runs (opposite directions)
    # Shift position of reverse run to the same start as forward run
    i = 0
    mask, first, last = tracks[i].endpoints()
    m = tracks[i].xyz.shape[1]
    dxyz = tracks[i].xyz[mask, first, :] - tracks[i + 2].xyz[mask, last, :]
    tracks[i + 2].xyz[mask, :, :] += np.tile(
        np.expand_dims(dxyz, axis=1), reps=(1, m, 1))
    # Take weighted mean (1 / variance) at each timestep assuming covariance = 1
    # (for each variable)
    tracks[i].means, tracks[i].sigmas = normal_weighted_mean(
        means=np.stack((tracks[i].means, tracks[i + 2].means[:, ::-1, :]), axis=3),
        sigmas=np.stack((tracks[i].sigmas, tracks[i + 2].sigmas[:, ::-1, :]), axis=3),
        axis=3, correlated=True)
    # Write to file
    glimpse.helpers.write_pickle(tracks[0], basepath + '.pkl')

# ---- Merge tracks into continuous arrays -----

# Load tracks to process
paths = glob.glob(os.path.join('tracks', '*[0-9].pkl'))

# Load unique point ids
point_paths = glob.glob(os.path.join('points', '*.pkl'))
ids = None
for path in point_paths:
    points = glimpse.helpers.read_pickle(path)
    if ids is None:
        ids = points['ids']
    else:
        ids = np.unique(np.concatenate((ids, points['ids'])))

# Load mean time for each run of non-repeating observers (< max dt)
track_breaks = []
track_datetimes = []
origin = datetime.datetime(1970, 1, 1, 0, 0, 0)
for path in paths:
    print(path)
    tracks = glimpse.helpers.read_pickle(path)
    nobs = tracks.images.shape[1]
    if nobs == 1:
        track_datetimes.append(tracks.datetimes)
        track_breaks.append([])
        continue
    sequence = [[j for j in range(nobs) if tracks.images[i, j] is not None]
        for i in range(len(tracks.images))]
    current = []
    starts = [0]
    previous_time = tracks.datetimes[0]
    for i, obs in enumerate(sequence):
        dt = tracks.datetimes[i] - previous_time
        if np.isin(obs, current).any() or dt > datetime.timedelta(hours=1):
            starts.append(i)
            current = obs
        else:
            current.extend(obs)
        previous_time = tracks.datetimes[i]
    starts.append(len(tracks.datetimes))
    track_datetimes.append([origin + (tracks.datetimes[starts[i]:starts[i + 1]] - origin).mean()
        for i in range(len(starts[:-1]))])
    track_breaks.append(starts)
datetimes = np.unique([t for ts in track_datetimes for t in ts])

# Initialize giant arrays
n = len(ids)
m = len(datetimes)
means = np.zeros((n, m, 3), dtype=np.float16)
means[:] = np.nan
sigmas = means.copy()
nobservers = np.zeros((n, m), dtype=np.uint8)
flotation = np.zeros((n, m), dtype=np.float16)

# Process tracks
for i_obs in range(len(paths)):
    basename = glimpse.helpers.strip_path(paths[i_obs])
    print(basename)
    tracks = glimpse.helpers.read_pickle(paths[i_obs])
    points = glimpse.helpers.read_pickle(
        os.path.join('points', basename + '.pkl'))
    # Take weighted mean of grouped times
    tbreaks = np.asarray(track_breaks[i_obs])
    dbreaks = np.diff(tbreaks)
    if np.any(dbreaks > 1):
        shape = (len(tracks.vxyz), len(track_datetimes[i_obs]), 3)
        tmeans = np.zeros(shape, dtype=float)
        tsigmas = np.zeros(shape, dtype=float)
        singles = dbreaks == 1
        tmeans[:, singles] = tracks.vxyz[:, tbreaks[:-1][singles]]
        tsigmas[:, singles] = tracks.vxyz_sigma[:, tbreaks[:-1][singles]]
        for i in np.where(dbreaks > 1)[0]:
            idx = slice(tbreaks[i], tbreaks[i + 1])
            tmeans[:, i], tsigmas[:, i] = normal_weighted_mean(
                means=tracks.vxyz[:, idx], sigmas=tracks.vxyz_sigma[:, idx],
                axis=1, correlated=True)
    else:
        tmeans = tracks.vxyz
        tsigmas = tracks.vxyz_sigma
    # Corresponding indices in giant array
    point_mask = np.isin(ids, points['ids'])
    time_mask = np.isin(datetimes, track_datetimes[i_obs])
    indices = np.ix_(point_mask, time_mask)
    # Take weighted mean at overlaps
    overlap = ~np.isnan(means[indices]) & ~np.isnan(tmeans)
    if np.any(overlap):
        m, s = normal_weighted_mean(
            np.dstack((means[indices][overlap], tmeans[overlap])),
            np.dstack((sigmas[indices][overlap], tsigmas[overlap])),
            axis=2, correlated=True)
        tmeans[overlap], tsigmas[overlap] = m.ravel(), s.ravel()
    # Merge into giant arrays
    means[indices] = tmeans
    sigmas[indices] = tsigmas
    # Flotation
    # take flotation for tracks, expand to time grid, overwrite overlaps
    flotation_array = np.tile(points['flotation'].reshape(-1, 1),
        reps=(1, indices[1].size))
    flotation[indices] = flotation_array
    # Number of Observers
    # count from observer mask, expand to time grid, max of overlap
    nobs_array = np.tile(points['observer_mask'].sum(axis=1).reshape(-1, 1),
        reps=(1, indices[1].size))
    nobs_array[overlap[..., 0]] = np.nanmax(np.column_stack((
        nobs_array[overlap[..., 0]],
        nobservers[indices][overlap[..., 0]])), axis=1)
    nobservers[indices] = nobs_array

# Apply spatial filter
# NOTE: SLOW
cols = template.shape[1]
neighbors = np.column_stack((
    ids - 1, ids + 1, ids - cols, ids + cols,
    # ids - 1 - cols, ids + 1 - cols, ids - 1 + cols, ids + 1 + cols
    ))
fmeans, fsigmas = means.copy(), sigmas.copy()
for id in ids:
    i = np.searchsorted(ids, id)
    print(i)
    idx = np.where(np.isin(ids, np.concatenate(([id], neighbors[i]))))[0]
    # Weighted mean
    # fmeans[i], fsigmas[i] = normal_weighted_mean(
    #     means[idx], sigmas[idx], axis=0, correlated=True)
    # Median
    # NOTE: How to propagate uncertainty ? For now taking sigma of median value.
    m = means[idx].copy()
    few_cams = (nobservers[idx] < 1) & (nobservers[idx].max(axis=0) > 1)
    m[few_cams] = np.nan
    medians = np.nanmedian(m, axis=0)
    is_median = m == medians
    idxi = np.argmax(is_median, axis=0)
    I, J = np.ogrid[:is_median.shape[1], :is_median.shape[2]]
    isnan = np.isnan(means[i])
    fmeans[i, ~isnan] = m[idxi, I, J][~isnan]
    fsigmas[i, ~isnan] = sigmas[idx][idxi, I, J][~isnan]

# Save giant arrays to file
outdir = 'tracks-arrays'
xy = glimpse.helpers.grid_to_points((template.X, template.Y))[ids]
xyi = np.column_stack((xy, ids))
glimpse.helpers.write_pickle(xyi, os.path.join(outdir, 'xyi.pkl'))
np.savetxt(
    fname=os.path.join(outdir, 'xyi.csv'), X=xyi,
    delimiter=',', fmt='%d', header='x,y,id', comments='')
glimpse.helpers.write_pickle(datetimes, os.path.join(outdir, 'datetimes.pkl'))
glimpse.helpers.write_pickle(fmeans, os.path.join(outdir, 'means.pkl'))
glimpse.helpers.write_pickle(fsigmas, os.path.join(outdir, 'sigmas.pkl'))
nobservers[np.isnan(means[..., 0])] = 0
glimpse.helpers.write_pickle(nobservers, os.path.join(outdir, 'nobservers.pkl'))
glimpse.helpers.write_pickle(flotation, os.path.join(outdir, 'flotation.pkl'))

# ---- Collapse tracks ----

# Initialize arrays
idx = range(1067)
midtimes = []
shape = (len(ids), len(idx), 3)
cmeans = np.zeros(shape, dtype=float)
cmeans[:] = np.nan
csigmas = cmeans.copy()
cflotation = cmeans[..., 0].copy()
cnobservers = np.zeros(shape[0:2], dtype=np.uint8)

# Load tracks into arrays
origin = datetime.datetime(1970, 1, 1, 0, 0, 0)
for i_obs in idx:
    basename = tracks_basename(i_obs)
    print(basename)
    tracks = glimpse.helpers.read_pickle(tracks_path(i_obs))
    points = glimpse.helpers.read_pickle(
        os.path.join('points', basename + '.pkl'))
    # TODO: Weight each time step based on temporal width?
    # dt = np.diff(tracks.datetimes)
    # time = np.column_stack((
    #     np.concatenate(([datetime.timedelta(0)], dt)),
    #     np.concatenate((dt, [datetime.timedelta(0)])))).sum(axis=1)
    # time /= width.sum()
    m, s = normal_weighted_mean(tracks.vxyz, tracks.vxyz_sigma,
        axis=1, correlated=True)
    point_idx = np.searchsorted(ids, points['ids'])
    cmeans[point_idx, i_obs] = m
    csigmas[point_idx, i_obs] = s
    midtimes.append(origin + (tracks.datetimes - origin).mean())
    cflotation[point_idx, i_obs] = points['flotation']
    cnobservers[point_idx, i_obs] = points['observer_mask'].sum(axis=1)

# Apply spatial filter
cols = template.shape[1]
neighbors = np.column_stack((
    ids - 1, ids + 1, ids - cols, ids + cols,
    # ids - 1 - cols, ids + 1 - cols, ids - 1 + cols, ids + 1 + cols
    ))
fcmeans, fcsigmas = cmeans.copy(), csigmas.copy()
for id in ids:
    i = np.searchsorted(ids, id)
    print(i)
    idx = np.where(np.isin(ids, np.concatenate(([id], neighbors[i]))))[0]
    # Median
    # NOTE: How to propagate uncertainty ? For now taking sigma of median value.
    m = cmeans[idx].copy()
    few_cams = (cnobservers[idx] < 1) & (cnobservers[idx].max(axis=0) > 1)
    m[few_cams] = np.nan
    medians = np.nanmedian(m, axis=0)
    is_median = m == medians
    idxi = np.argmax(is_median, axis=0)
    I, J = np.ogrid[:is_median.shape[1], :is_median.shape[2]]
    isnan = np.isnan(cmeans[i])
    fcmeans[i, ~isnan] = m[idxi, I, J][~isnan]
    fcsigmas[i, ~isnan] = csigmas[idx][idxi, I, J][~isnan]

# Save to file
outdir = 'tracks-arrays-sm'
glimpse.helpers.write_pickle(xyi, os.path.join(outdir, 'xyi.pkl'))
np.savetxt(
    fname=os.path.join(outdir, 'xyi.csv'), X=xyi,
    delimiter=',', fmt='%d', header='x,y,id', comments='')
glimpse.helpers.write_pickle(np.array(midtimes), os.path.join(outdir, 'datetimes.pkl'))
glimpse.helpers.write_pickle(fcmeans, os.path.join(outdir, 'means.pkl'))
glimpse.helpers.write_pickle(fcsigmas, os.path.join(outdir, 'sigmas.pkl'))
cnobservers[np.isnan(cmeans[..., 0])] = 0
glimpse.helpers.write_pickle(cnobservers, os.path.join(outdir, 'nobservers.pkl'))
glimpse.helpers.write_pickle(cflotation, os.path.join(outdir, 'flotation.pkl'))

# ---- Rasterize -----

path = 'tracks-arrays-sm'
datetimes = glimpse.helpers.read_pickle(
    os.path.join(path, 'datetimes.pkl'))
xyi = glimpse.helpers.read_pickle(
    os.path.join(path, 'xyi.pkl'))
means = glimpse.helpers.read_pickle(
    os.path.join(path, 'means.pkl'))
sigmas = glimpse.helpers.read_pickle(
    os.path.join(path, 'sigmas.pkl'))
nobservers = glimpse.helpers.read_pickle(
    os.path.join(path, 'nobservers.pkl'))
flotation = glimpse.helpers.read_pickle(
    os.path.join(path, 'flotation.pkl'))
template = glimpse.Raster.read(os.path.join('points', 'template.tif'))

# Rasterize and write to file
path = 'tracks-rasters-sm'
raster = template.copy()
raster.Z = raster.Z.astype(float)
raster.Z[:] = np.nan
raster.Z.flat[xyi[:, 2].astype(int)] = True
raster.crop_to_data()
rowcols = raster.xy_to_rowcol(xyi[:, 0:2], snap=True)
# Data
base = np.full(raster.shape + (len(datetimes), ), np.nan, dtype=np.float32)
for i, basename in enumerate(('vx', 'vy', 'vz')):
    base[rowcols[:, 0], rowcols[:, 1]] = means[..., i]
    glimpse.helpers.write_pickle(base, os.path.join(path, basename + '.pkl'))
for i, basename in enumerate(('vx_sigma', 'vy_sigma', 'vz_sigma')):
    base[rowcols[:, 0], rowcols[:, 1]] = sigmas[..., i]
    glimpse.helpers.write_pickle(base, os.path.join(path, basename + '.pkl'))
base[rowcols[:, 0], rowcols[:, 1]] = flotation
glimpse.helpers.write_pickle(base, os.path.join(path, 'flotation.pkl'))
base[rowcols[:, 0], rowcols[:, 1]] = nobservers
base[np.isnan(base)] = 0
glimpse.helpers.write_pickle(base.astype(np.uint8), os.path.join(path, 'nobservers.pkl'))
# Template
raster.Z[np.isnan(raster.Z)] = 0
raster.Z = raster.Z.astype(np.uint8)
raster.write(os.path.join(path, 'template.tif'), crs=32606)
# Metadata
glimpse.helpers.write_pickle(datetimes, os.path.join(path, 'datetimes.pkl'))
glimpse.helpers.write_pickle(xyi, os.path.join(path, 'xyi.pkl'))

# ---- Compute strain rates -----

# Compute strain rates
path = 'tracks-rasters-sm'
rvx = glimpse.helpers.read_pickle(os.path.join(path, 'vx.pkl'))
rvy = glimpse.helpers.read_pickle(os.path.join(path, 'vy.pkl'))
dudy, dudx = np.gradient(rvx, axis=(0, 1))
dudx, dudy = dudx * (1 / raster.d[0]), dudy * (1 / raster.d[1])
dvdy, dvdx = np.gradient(rvy, axis=(0, 1))
dvdx, dvdy = dvdx * (1 / raster.d[0]), dvdy * (1 / raster.d[1])
strain = (dudx, dvdy, dudy + dvdx)
theta = np.arctan2(strain[2], (strain[0] - strain[1])) * 0.5
cos, sin = np.cos(theta), np.sin(theta)
Q = np.stack([cos.ravel(), sin.ravel(), -sin.ravel(), cos.ravel()]).reshape(2, 2, -1)
E = np.stack([strain[0].ravel(), strain[2].ravel() * 0.5, strain[2].ravel() * 0.5, strain[1].ravel()]).reshape(2, 2, -1)
E_prime = np.diagonal(np.matmul(Q.transpose(2, 0, 1),
    np.matmul(E.transpose(2, 0, 1), Q.transpose(1, 0, 2).transpose(2, 0, 1))
    ).transpose(1, 2, 0))
emax = E_prime[:, 0].reshape(theta.shape)
emin = E_prime[:, 1].reshape(theta.shape)
thetamax, thetamin = theta, theta + np.pi * 0.5
extension_u = emax * np.cos(thetamax)
extension_v = emax * np.sin(thetamax)
compression_u = emin * np.cos(thetamin)
compression_v = emin * np.sin(thetamin)

# Save as spatial arrays
glimpse.helpers.write_pickle(extension_u, os.path.join(path, 'extension_u.pkl'))
glimpse.helpers.write_pickle(extension_v, os.path.join(path, 'extension_v.pkl'))
glimpse.helpers.write_pickle(compression_u, os.path.join(path, 'compression_u.pkl'))
glimpse.helpers.write_pickle(compression_v, os.path.join(path, 'compression_v.pkl'))

# Save as flat arrays
template = glimpse.Raster.read(os.path.join(path, 'template.tif'))
rows, cols = np.nonzero(template.Z)
path = 'tracks-arrays-sm'
glimpse.helpers.write_pickle(
    np.dstack((extension_u[rows, cols, :], extension_v[rows, cols, :])),
    os.path.join(path, 'extension.pkl'))
glimpse.helpers.write_pickle(
    np.dstack((compression_u[rows, cols, :], compression_v[rows, cols, :])),
    os.path.join(path, 'compression.pkl'))

# Estimate slope vz
# vx, vy (rasters), z (raster),
dem_interpolant = glimpse.helpers.read_pickle('dem_interpolant.pkl')
datetimes = glimpse.helpers.read_pickle(os.path.join(path, 'datetimes.pkl'))
template = glimpse.Raster.read(os.path.join(path, 'template.tif'))
xy = glimpse.helpers.grid_to_points((template.X, template.Y))
rowcol = dem_interpolant.means[0].xy_to_rowcol(xy, snap=True)
rvz = glimpse.helpers.read_pickle(os.path.join(path, 'vz.pkl'))
rvz_slope = []
for i in range(rvz.shape[2]):
    print(i)
    dem = dem_interpolant(datetimes[i])
    dz = np.dstack(dem.gradient())[rowcol[:, 0], rowcol[:, 1], :].reshape(template.shape + (2, ))
    # dzx, dzy = dem.gradient()
    # dz = np.dstack((
    #     glimpse.Raster(dzx, dem.x, dem.y).sample(xy),
    #     glimpse.Raster(dzy, dem.x, dem.y).sample(xy))).reshape(template.shape + (2, ))
    rvz_slope.append(dz[..., 0] * rvx[..., i] + dz[..., 1] * rvy[..., i])

glimpse.helpers.write_pickle(np.dstack(rvz_slope), os.path.join(path, 'vz_slope.pkl'))

# ---- Animate -----

path = 'tracks-rasters-sm'
datetimes = glimpse.helpers.read_pickle(
    os.path.join(path, 'datetimes.pkl'))
vx = glimpse.helpers.read_pickle(
    os.path.join(path, 'vx.pkl'))
vy = glimpse.helpers.read_pickle(
    os.path.join(path, 'vy.pkl'))
vz = glimpse.helpers.read_pickle(
    os.path.join(path, 'vz.pkl'))
vz_slope = glimpse.helpers.read_pickle(
    os.path.join(path, 'vz_slope.pkl'))
nobservers = glimpse.helpers.read_pickle(
    os.path.join(path, 'nobservers.pkl'))
flotation = glimpse.helpers.read_pickle(
    os.path.join(path, 'flotation.pkl'))
template = glimpse.Raster.read(os.path.join(path, 'template.tif'))
extension_u = glimpse.helpers.read_pickle(
    os.path.join(path, 'extension_u.pkl'))
extension_v = glimpse.helpers.read_pickle(
    os.path.join(path, 'extension_v.pkl'))
compression_u = glimpse.helpers.read_pickle(
    os.path.join(path, 'compression_u.pkl'))
compression_v = glimpse.helpers.read_pickle(
    os.path.join(path, 'compression_v.pkl'))

# Crop to coverage
minobs = 2
raster = template.copy()
raster.Z = np.where((nobservers >= minobs).any(axis=2), True, np.nan)
point_mask = raster.data_extent()
time_mask = (nobservers >= minobs).any(axis=(0, 1))
indices = point_mask + (time_mask, )
raster.crop_to_data()

# Mask data
# dy, dx = np.gradient(nobservers, axis=(0, 1))
# few_obs = (nobservers < minobs) | (dx != 0) | (dy != 0)
few_obs = (nobservers < minobs)
# nobservers = nobservers.astype(float)
# nobservers[nobservers == 0] = np.nan
# few_obs |= (scipy.ndimage.minimum_filter(nobservers, size=(3, 3, 1)) < minobs)
vx[few_obs] = np.nan
vy[few_obs] = np.nan
vz[few_obs] = np.nan
vz_slope[few_obs] = np.nan
extension_u[few_obs] = np.nan
extension_v[few_obs] = np.nan
compression_u[few_obs] = np.nan
compression_v[few_obs] = np.nan
flotation[few_obs] = np.nan

# Compute speeds
speeds = np.sqrt(vx**2 + vy**2)

# Plot speeds
import mpl_toolkits.axes_grid1
i = 0
fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(12, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
ax.set_aspect(1)
im = ax.imshow(speeds[indices][..., i], vmin=0, vmax=20,
    extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", "5%", pad="3%")
matplotlib.pyplot.colorbar(im, cax=cax)
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)

def update_plot(i):
    print(i)
    im.set_array(speeds[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return im, txt
ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('speed_multi2.mp4')

# Plot vx, vy
fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(12, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
ax.set_aspect(1)
i = 0
scale = 15
mask = ~np.isnan(vx[indices][..., i])
quiver = matplotlib.pyplot.quiver(
    raster.X, raster.Y, vx[indices][..., i] * scale, vy[indices][..., i] * scale,
    # color='black',
    speeds[indices][..., i], clim=[0, 20],
    alpha=1, width=5, headaxislength=0, headwidth=1, minlength=0,
    pivot='tail', angles='xy', scale_units='xy', scale=1)
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", "5%", pad="3%")
matplotlib.pyplot.colorbar(quiver, cax=cax)
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)

def update_plot(i):
    print(i)
    quiver.set_UVC(vx[indices][..., i] * scale, vy[indices][..., i] * scale, speeds[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return quiver, txt
ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('vxy.mp4', dpi=150)

# Plot principal strains
fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(8, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
ax.set_aspect(1)
i = 0
scale = 15
strain_scale = 1e4
# quiver_velocity = ax.quiver(
#     raster.X, raster.Y, vx[indices][..., i] * scale, vy[indices][..., i] * scale,
#     color='black', alpha=0.25, width=5, headaxislength=0, headwidth=1,
#     minlength=0, pivot='tail', angles='xy', scale_units='xy', scale=1,
#     zorder=0)
# im_vz = ax.imshow(vz[indices][..., i] - vz_slope[indices][..., i],
#     cmap=cmocean.cm.balance, vmin=-1, vmax=1, zorder=0,
#     extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
im_vz = ax.imshow(vz[indices][..., i],
    cmap=cmocean.cm.balance, vmin=-2, vmax=2, zorder=0,
    extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
# matplotlib.pyplot.colorbar(im_vz)
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
kwargs = dict(
    angles='xy', scale=1, scale_units='xy', units='xy', width=20,
    headwidth=0, headlength=0, headaxislength=0)
quiver_extension = ax.quiver(
    raster.X, raster.Y,
    strain_scale * extension_u[indices][..., i], strain_scale * extension_v[indices][..., i],
    color='black', pivot='mid', zorder=2, **kwargs)
quiver_compression = ax.quiver(
    raster.X, raster.Y,
    strain_scale * compression_u[indices][..., i], strain_scale * compression_v[indices][..., i],
    color='red', pivot='mid', zorder=1, **kwargs)
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)
# matplotlib.pyplot.legend(('velocity', 'extension', 'compression'))
matplotlib.pyplot.legend(('extension', 'compression'))
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])

def update_plot(i):
    print(i)
    # quiver_velocity.set_UVC(vx[indices][..., i] * scale, vy[indices][..., i] * scale)
    # im_vz.set_array(vz[indices][..., i] - vz_slope[indices][..., i])
    im_vz.set_array(vz[indices][..., i])
    quiver_compression.set_UVC(strain_scale * compression_u[indices][..., i],
        strain_scale * compression_v[indices][..., i])
    quiver_extension.set_UVC(strain_scale * extension_u[indices][..., i],
        strain_scale * extension_v[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return quiver_velocity, quiver_compression, quiver_extension, txt
ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('strain_multi_filtered_vz.mp4', dpi=150)
matplotlib.pyplot.close('all')

# Plot flotation
import mpl_toolkits.axes_grid1
i = 0
fig = matplotlib.pyplot.figure(tight_layout=True, figsize=(12, 8))
ax = matplotlib.pyplot.gca()
ax.axis('off')
im = ax.imshow(flotation[indices][..., i], vmin=0, vmax=1,
    extent=(raster.xlim[0], raster.xlim[1], raster.ylim[1], raster.ylim[0]))
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes("right", "5%", pad="3%")
matplotlib.pyplot.colorbar(im, cax=cax)
ax.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
txt = ax.text(1, 1, datetimes[time_mask][i].strftime('%Y-%m-%d') + ' ' + str(i),
    color='black', horizontalalignment='right', transform=ax.transAxes,
    fontsize=14)

def update_plot(i):
    print(i)
    im.set_array(flotations[indices][..., i])
    ix = np.arange(len(datetimes))[time_mask][i]
    txt.set_text(datetimes[ix].strftime('%Y-%m-%d') + ' ' + str(ix))
    return im, txt
ani = matplotlib.animation.FuncAnimation(fig, update_plot, frames=range(time_mask.sum()),
    blit=True, interval=200)
ani.save('flotation_multi.mp4')
