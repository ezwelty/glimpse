import cg
from cg import glimpse
from glimpse.imports import (datetime, np, os, shapely, re, matplotlib)
import glob

grid_step = (100, 100) # m

# ---- Build template ----

box = glimpse.helpers.bounding_box(cg.Glacier())
x, y = glimpse.helpers.box_to_grid(box, step=grid_step, snap=(0, 0), mode='vectors')
template = glimpse.Raster(np.ones((len(y), len(x)), dtype=float), x=x, y=y[::-1])
points = glimpse.helpers.grid_to_points((template.X, template.Y))

# ---- Select velocity grids ----

velocity_keys = [
    ('20040618_20040707', 'aerometric'),
    ('20050811_20050827', 'aerometric'),
    ('20060712_20060727', 'aerometric'),
    ('20090803_20090827', 'aerometric'),
    ('20100525_20100602', 'aerometric'),
	('20110211_20110222', 'terrasar'),
	('20110222_20110305', 'terrasar'),
	('20110305_20110316', 'terrasar'),
	('20110316_20110327', 'terrasar'),
	('20110327_20110407', 'terrasar'),
	('20110510_20110521', 'terrasar'),
	('20110618_20110629', 'terrasar'),
	('20110629_20110710', 'terrasar'),
	('20110710_20110721', 'terrasar'),
	('20110721_20110801', 'terrasar'),
	('20110801_20110812', 'terrasar'),
	('20110812_20110823', 'terrasar'),
	('20110823_20110903', 'terrasar'),
	('20110903_20110914', 'terrasar'),
	('20111017_20111028', 'terrasar'),
	('20111028_20111108', 'terrasar'),
	('20111130_20111222', 'terrasar'),
	('20120215_20120226', 'terrasar'),
	('20120226_20120308', 'terrasar'),
	('20120308_20120319', 'terrasar'),
	('20120513_20120524', 'terrasar'),
	('20120701_20120712', 'terrasar'),
	('20120809_20120820', 'terrasar'),
	('20120820_20120831', 'terrasar'),
	('20120831_20120911', 'terrasar'),
	('20121003_20121014', 'terrasar'),
	('20121105_20121116', 'terrasar'),
	('20121116_20121127', 'terrasar'),
    # < holes>
	('20130306_20130317', 'terrasar'),
	('20130328_20130408', 'terrasar'),
	('20130419_20130430', 'terrasar'),
	('20130430_20130511', 'terrasar'),
	('20130602_20130613', 'terrasar'),
	('20130613_20130624', 'terrasar'),
    # </holes>
	('20130721_20130801', 'terrasar'),
	('20130801_20130812', 'terrasar'),
	('20130812_20130823', 'terrasar'),
	('20130823_20130903', 'terrasar'),
	('20130829_20130909', 'terrasar'),
	('20130903_20130914', 'terrasar'),
	('20130909_20130920', 'terrasar'),
	('20130914_20130925', 'terrasar'),
	('20130920_20131001', 'terrasar'),
	('20130925_20131006', 'terrasar'),
	('20131006_20131017', 'terrasar'),
	('20131017_20131028', 'terrasar'),
    ('20131118_20131204', 'golive'),
    ('20140208_20140224', 'golive'),
    ('20140224_20140328', 'golive'), # some holes
    ('20140326_20140411', 'golive'),
    ('20140411_20140427', 'golive'),
    ('20140427_20140513', 'golive'), # some holes
    ('20140718_20140819', 'golive'),
    ('20140826_20141013', 'golive'),
    ('20141013_20141029', 'golive'),
    ('20141029_20141114', 'golive'),
    ('20141114_20141130', 'golive'), # some holes
    ('20150322_20150423', 'golive'), # many holes
    ('20150430_20150516', 'golive'), # some holes
    ('20150502_20150518', 'golive'), # some holes
    ('20150617_20150703', 'golive'),
    ('20150703_20150719', 'golive'),
    ('20150719_20150804', 'golive'),
    ('20150804_20150820', 'golive'),
    ('20150907_20150923', 'golive'),
    ('20150923_20151025', 'golive'),
    ('20160621_20160707', 'golive'),
    ('20161009_20161025', 'golive')
]
# Add all Krimmel velocities
velocity_keys += [(re.findall(r'([0-9]{8}_[0-9]{8})', path)[0], 'krimmel')
    for path in glob.glob(os.path.join(root, 'velocities-krimmel', 'data', '*_vx.tif'))]
# Add all early McNabb landsat velocities
velocity_keys += [(re.findall(r'([0-9]{8}_[0-9]{8})', path)[0], 'landsat')
    for path in glob.glob(os.path.join(root, 'velocities-landsat', 'data', '*_vx.tif'))
    if re.findall(r'([0-9]{8}_[0-9]{8})', path)[0] < '20110101']
# Sort by time
velocity_keys.sort(key=lambda x: x[0])

# ---- Read and sample at template ----
# t: start, end, mid
# vx, vy: [x, y, t]

t0, vx0, vy0 = [], [], []
dropped = []
for i, key in enumerate(velocity_keys):
    print(key)
    # HACK: Deal with inconsistent NoDataValues
    nan = -9999 if key[1] == 'krimmel' else 0
    basepath = os.path.join(root, 'velocities-' + key[1], 'data', key[0])
    try:
        x = glimpse.Raster.read(basepath + '_vx.tif', xlim=template.xlim,
            ylim=template.ylim, nan=nan)
    except ValueError:
        dropped.append(i)
        continue
    y = glimpse.Raster.read(basepath + '_vy.tif', xlim=template.xlim,
        ylim=template.ylim, nan=nan)
    # NOTE: Avoiding faster grid sampling because of NaN
    vx0.append(x.sample(points, order=1, bounds_error=False).reshape(template.shape))
    vy0.append(y.sample(points, order=1, bounds_error=False).reshape(template.shape))
    datestr = re.findall(r'^([0-9]{8})', key[0])[0]
    t0.append(datetime.datetime.strptime(datestr, '%Y%m%d'))
# Remove unread keys
for i in dropped:
    velocity_keys.pop(i)
# Stack results
datetimes = np.array(t0)
vx = np.dstack(vx0)
vy = np.dstack(vy0)

# ---- Filter Landsat velocities ----

lmask = np.array([key[1] == 'landsat' for key in velocity_keys])
lvx = vx[..., lmask].copy()
lvy = vy[..., lmask].copy()
# Normalize vx, vy to the unit circle
theta = np.arctan2(lvy, lvx)
theta[theta < 0] += 2 * np.pi
uy = np.sin(theta)
ux = np.cos(theta)
# Compute moving-window median orientations
mask = ~np.isnan(ux)
mux = np.zeros(ux.shape, dtype=float)
muy = np.zeros(uy.shape, dtype=float)
for i in range(ux.shape[0]):
    for j in range(ux.shape[1]):
        mux[i, j, mask[i, j, :]] = scipy.ndimage.median_filter(
            ux[i, j, mask[i, j, :]], size=5)
        muy[i, j, mask[i, j, :]] = scipy.ndimage.median_filter(
            uy[i, j, mask[i, j, :]], size=5)
mtheta = np.arctan2(muy, mux)
# Remove outliers
mtheta[mtheta < 0] += 2 * np.pi
dtheta = np.pi - np.abs(np.abs(theta - mtheta) - np.pi)
bad = dtheta > np.pi / 10
lvx[bad] = np.nan
lvy[bad] = np.nan
# Mask by coverage of other platforms
out = np.count_nonzero(~np.isnan(vx[..., ~lmask]), axis=2) == 0
lvx[out, :] = np.nan
lvy[out, :] = np.nan
# Save back to stack
vx[..., lmask] = lvx
vy[..., lmask] = lvy

# Plot
matplotlib.pyplot.figure()
# glimpse.Raster(np.sum(bad, axis=2), template.x, template.y).plot()
glimpse.Raster(np.sum(~np.isnan(vx), axis=2) > 1, template.x, template.y).plot()
# matplotlib.pyplot.colorbar()
matplotlib.pyplot.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])

# ---- Compute weights ----
# Use sum of distances to neighbors
# Saturate at 0.5 years because of seasonal variability

dts = np.column_stack((
    np.concatenate(([datetime.timedelta(0)], np.diff(datetimes))),
    np.concatenate((np.diff(datetimes), [datetime.timedelta(0)])))).sum(axis=1)
weights = np.array([dt.total_seconds() / (3600 * 24 * 365) for dt in dts])
weights[weights > 0.5] = 0.5
# Lower weights for observations before 2004
dyears = np.array([dt.total_seconds() / (3600 * 24 * 365)
    for dt in datetime.datetime(2004, 6, 18) - datetimes])
weights[dyears > 0] *= (1 - dyears[dyears > 0] / max(dyears))

# ---- Compute summary statistics (cartesian) ----

w = glimpse.helpers.tile_axis(weights, vx.shape, axis=(0, 1))
vx_mean = glimpse.helpers.weighted_nanmean(vx, weights=w, axis=2)
vy_mean = glimpse.helpers.weighted_nanmean(vy, weights=w, axis=2)
vx_sigma = glimpse.helpers.weighted_nanstd(vx, weights=w, axis=2,
    means=vx_mean)
vy_sigma = glimpse.helpers.weighted_nanstd(vy, weights=w, axis=2,
    means=vy_mean)
# Compute orientations without Landsat and use for slow areas
mask = [key[1] != 'landsat' for key in velocity_keys]
w = glimpse.helpers.tile_axis(weights[mask], vx[..., mask].shape, axis=(0, 1))
vx_mean2 = glimpse.helpers.weighted_nanmean(vx[..., mask], weights=w, axis=2)
vy_mean2 = glimpse.helpers.weighted_nanmean(vy[..., mask], weights=w, axis=2)
vx_sigma2 = glimpse.helpers.weighted_nanstd(vx[..., mask], weights=w, axis=2,
    means=vx_mean2)
vy_sigma2 = glimpse.helpers.weighted_nanstd(vy[..., mask], weights=w, axis=2,
    means=vy_mean2)
vxy = np.hypot(vx_mean, vy_mean)
vxy2 = np.hypot(vx_mean2, vy_mean2)
vxy_ratio = vxy / vxy2
reorient = (vxy < 1) | (vxy2 < 1)
vx_mean[reorient] = vx_mean2[reorient] * vxy_ratio[reorient]
vy_mean[reorient] = vy_mean2[reorient] * vxy_ratio[reorient]
# Apply median filter
mask = ~np.isnan(vx_mean)
vx_mean = glimpse.helpers.median_filter(vx_mean, mask=mask, size=(3, 3))
vy_mean = glimpse.helpers.median_filter(vy_mean, mask=mask, size=(3, 3))
vx_sigma = glimpse.helpers.median_filter(vx_sigma, mask=mask, size=(3, 3))
vy_sigma = glimpse.helpers.median_filter(vy_sigma, mask=mask, size=(3, 3))

# ---- Plot results ----

speed_mean = np.hypot(vx_mean, vy_mean)
# speed_mean = np.hypot(vx_sigma, vy_sigma)
matplotlib.pyplot.figure()
glimpse.Raster(speed_mean, template.x, template.y).plot()
matplotlib.pyplot.colorbar()
mask = ~np.isnan(vx_mean)
matplotlib.pyplot.quiver(template.X[mask], template.Y[mask],
    100 * vx_mean[mask] / speed_mean[mask], 100 * vy_mean[mask] / speed_mean[mask],
    angles='xy', scale_units='xy', scale=1)
matplotlib.pyplot.gca().set_aspect(1)

# Plot
matplotlib.pyplot.figure()
glimpse.Raster(np.hypot(vx_mean, vy_mean), template.x, template.y).plot()
matplotlib.pyplot.colorbar()
matplotlib.pyplot.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])

# ---- Save to file ----

names = ('vx', 'vy', 'vx_stderr', 'vy_stderr')
Zs = (vx_mean, vy_mean, vx_sigma, vy_sigma)
for name, Z in zip(names, Zs):
    glimpse.Raster(Z, template.x, template.y).write(
        os.path.join('velocity', name + '.tif'), nan=-9999, crs=32606)

# ---- Compute summary statistics (polar) ----

speed = np.hypot(vx, vy)
theta = np.arctan2(vy, vx)
theta[theta < 0] += 2 * np.pi
unit_xy = (
    glimpse.helpers.weighted_nanmean(np.sin(theta), weights=w, axis=2),
    glimpse.helpers.weighted_nanmean(np.cos(theta), weights=w, axis=2))
theta_mean = np.arctan2(*unit_xy)
theta_mean[theta_mean < 0] += 2 * np.pi
theta_sigma = np.sqrt(-2 * np.log(np.hypot(*unit_xy)))
speed_mean = glimpse.helpers.weighted_nanmean(speed, weights=w, axis=2)
speed_sigma = glimpse.helpers.weighted_nanstd(speed, weights=w, axis=2,
    means=speed_mean)
speed_min = np.nanmin(speed, axis=2)
speed_max = np.nanmax(speed, axis=2)

# ---- Plot results ----

# Plot canonical directions
matplotlib.pyplot.figure()
glimpse.Raster(speed_mean, template.x, template.y).plot()
matplotlib.pyplot.colorbar()
matplotlib.pyplot.gca().set_aspect(1)
mask = ~np.isnan(vx_mean)
matplotlib.pyplot.quiver(template.X[mask], template.Y[mask],
    100 * vx_mean[mask] / speed_mean[mask], 100 * vy_mean[mask] / speed_mean[mask],
    angles='xy', scale_units='xy', scale=1)

# Magnitudes (sigma)
glimpse.Raster(speed_sigma, template.x, template.y).plot()
matplotlib.pyplot.colorbar()
matplotlib.pyplot.gca().set_aspect(1)

# Orientation (sigma)
glimpse.Raster(theta_sigma, template.x, template.y).plot()
matplotlib.pyplot.colorbar()
matplotlib.pyplot.gca().set_aspect(1)
