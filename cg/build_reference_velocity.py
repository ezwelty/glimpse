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
# Sort by time
velocity_keys.sort(key=lambda x: x[0])

# ---- Build paths ----

velocity_paths = [os.path.join(root, 'velocities-' + tag, 'data', datestr)
    for datestr, tag in velocity_keys]
velocity_xpaths = [path + '_vx.tif' for path in velocity_paths]
velocity_ypaths = [path + '_vy.tif' for path in velocity_paths]

# ---- Parse dates from paths ----

starts, ends = tuple(zip(*[re.findall(r'([0-9]{8})_([0-9]{8})', key[0])[0]
    for key in velocity_keys]))
ranges = np.column_stack((
    [datetime.datetime.strptime(x, '%Y%m%d') for x in starts],
    [datetime.datetime.strptime(x, '%Y%m%d') for x in ends]))

# ---- Read and sample at template ----
# t: start, end, mid
# vx, vy: [x, y, t]

t, vx, vy = [], [], []
for i, key in enumerate(velocity_keys):
    print(key)
    # HACK: Deal with inconsistent NoDataValues
    nan = -9999 if key[1] == 'krimmel' else 0
    try:
        x = glimpse.Raster.read(velocity_xpaths[i], xlim=template.xlim,
            ylim=template.ylim, nan=nan)
    except ValueError:
        continue
    y = glimpse.Raster.read(velocity_ypaths[i], xlim=template.xlim,
        ylim=template.ylim, nan=nan)
    # NOTE: Avoiding faster grid sampling because of NaN
    vx.append(x.sample(points, order=1, bounds_error=False).reshape(template.shape))
    vy.append(y.sample(points, order=1, bounds_error=False).reshape(template.shape))
    t.append(ranges[i])
# Stack results
t = np.vstack(t)
vx = np.dstack(vx)
vy = np.dstack(vy)
# Append midtimes
t = np.column_stack((t, t[:, 0] + 0.5 * np.diff(t, axis=1).ravel()))

# ---- Compute weights ----
# Use sum of distances to neighbors
# Saturate at 0.5 years because of seasonal variability

dts = np.column_stack((
    np.concatenate(([datetime.timedelta(0)], np.diff(t[:, 2]))),
    np.concatenate((np.diff(t[:, 2]), [datetime.timedelta(0)])))).sum(axis=1)
weights = np.array([dt.total_seconds() / (3600 * 24 * 365) for dt in dts])
weights[weights > 0.5] = 0.5
w = glimpse.helpers.tile_axis(weights, vx.shape, axis=(0, 1))

# ---- Compute summary statistics (cartesian) ----

vx_mean = glimpse.helpers.weighted_nanmean(vx, weights=w, axis=2)
vy_mean = glimpse.helpers.weighted_nanmean(vy, weights=w, axis=2)
vx_sigma = glimpse.helpers.weighted_nanstd(vx, weights=w, axis=2,
    means=vx_mean)
vy_sigma = glimpse.helpers.weighted_nanstd(vy, weights=w, axis=2,
    means=vy_mean)
# vx_max = np.nanmax(vx, axis=2)
# vy_max = np.nanmax(vy, axis=2)
# vx_min = np.nanmin(vx, axis=2)
# vy_min = np.nanmin(vy, axis=2)
# vx_sigmas = np.maximum(np.abs(vx_max - vx_mean), np.abs(vx_min - vx_mean)) / vx_sigma
# vy_sigmas = np.maximum(np.abs(vy_max - vy_mean), np.abs(vy_min - vy_mean)) / vy_sigma

# Plot
# matplotlib.pyplot.imshow(vy_sigmas)
# matplotlib.pyplot.colorbar()

# ---- Save to file ----

names = ('vx', 'vy', 'vx_stderr', 'vy_stderr')
Zs = (vx_mean, vy_mean, vx_sigma, vy_sigma)
for name, Z in zip(names, Zs):
    glimpse.Raster(Z, template.x, template.y).write(
        os.path.join('velocity', name + '.tif'), nan=-9999, crs=32606)

# ---- Compute summary statistics (polar) ----

# speed = np.hypot(vx, vy)
# theta = np.arctan2(vy, vx)
# theta[theta < 0] += 2 * np.pi
# unit_xy = (
#     glimpse.helpers.weighted_nanmean(np.sin(theta), weights=w, axis=2),
#     glimpse.helpers.weighted_nanmean(np.cos(theta), weights=w, axis=2))
# theta_mean = np.arctan2(*unit_xy)
# theta_mean[theta_mean < 0] += 2 * np.pi
# theta_sigma = np.sqrt(-2 * np.log(np.hypot(*unit_xy)))
# speed_mean = glimpse.helpers.weighted_nanmean(speed, weights=w, axis=2)
# speed_sigma = glimpse.helpers.weighted_nanstd(speed, weights=w, axis=2,
#     means=speed_mean)
# speed_min = np.nanmin(speed, axis=2)
# speed_max = np.nanmax(speed, axis=2)

# ---- Plot results ----

# Plot canonical directions
# mask = ~np.isnan(vx_mean)
# glimpse.Raster(speed_mean, template.x, template.y).plot()
# matplotlib.pyplot.colorbar()
# matplotlib.pyplot.quiver(template.X[mask], template.Y[mask],
#     100 * vx_mean[mask] / speed_mean[mask], 100 * vy_mean[mask] / speed_mean[mask],
#     angles='xy', scale_units='xy', scale=1)
# matplotlib.pyplot.gca().set_aspect(1)

# ---- Tests ----

# idx = np.unravel_index(np.count_nonzero(~np.isnan(vx), axis=2)[0].argsort()[::-1][:10], template.shape)
# i, j = tuple(zip(*idx))[0]
# dx = vx[i, j, :]
# dx = dx[~np.isnan(dx)]
# dy = vy[i, j, :]
# dy = dy[~np.isnan(dy)]
# dx_mean, dx_sigma = dx.mean(), np.std(dx)
# dy_mean, dy_sigma = dy.mean(), np.std(dy)
# mag = np.hypot(dx, dy)
# rot = np.arctan2(dy, dx)
# rot[rot < 0] += 2 * np.pi
# mag_mean, mag_sigma = mag.mean(), np.std(mag)
# mag2 = np.hypot(
#     dx_mean + np.random.randn(10000) * dx_sigma,
#     dy_mean + np.random.randn(10000) * dy_sigma)
# mag_mean2, mag_sigma2 = mag2.mean(), np.std(mag2)
# uxy = (
#     np.sin(rot).mean(),
#     np.cos(rot).mean())
# rot_mean = np.arctan2(*uxy)
# if rot_mean < 0:
#     rot_mean += 2 * np.pi
# rot_sigma = np.sqrt(-2 * np.log(np.hypot(*uxy)))
# rot2 = np.arctan2(
#     dy_mean + np.random.randn(10000) * dy_sigma,
#     dx_mean + np.random.randn(10000) * dx_sigma)
# rot2[rot2 < 0] += 2 * np.pi
# uxy2 = (
#     np.sin(rot2).mean(),
#     np.cos(rot2).mean())
# rot_mean2 = np.arctan2(*uxy2)
# if rot_mean2 < 0:
#     rot_mean2 += 2 * np.pi
# rot_sigma2 = np.sqrt(-2 * np.log(np.hypot(*uxy2)))
