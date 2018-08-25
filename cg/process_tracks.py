import cg
from cg import glimpse
from glimpse.imports import (np, os, matplotlib, collections)
root = '/volumes/science/data/columbia'
cg.IMAGE_PATH = os.path.join(root, 'timelapse')
cg.FLAT_IMAGE_PATH = False

observer_json = glimpse.helpers.read_json('observers.json',
    object_pairs_hook=collections.OrderedDict)

# ---- Load result -----

i_obs = 1
date_str = min([cg.parse_image_path(basenames[0])['date_str']
    for basenames in observer_json[i_obs].values()])
basename = os.path.join('tracks', date_str + '-' + str(i_obs))
suffixes = ('f', 'fv', 'r', 'rv')
tracks = [glimpse.helpers.read_pickle(basename + '-' + suffix + '.pkl')
    for suffix in suffixes]
# Use same tracker for all tracks
for i in range(1, len(tracks)):
    tracks[i].tracker = tracks[0].tracker
# Reset paths
for observer in tracks[0].tracker.observers:
    for img in observer.images:
        basename = glimpse.helpers.strip_path(img.path)
        img.path = cg.find_image(basename)
# Multi-camera mask
is_multi = tracks[0].params['observer_mask'].sum(axis=1) > 1
# Load dems
# dem = glimpse.Raster.read(os.path.join('dems', date_str + '-' + str(i_obs) + '.tif'))
# dem.plot(dem.hillshade(), cmap='gray')

# ---- Plot result ----

# Map
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
tracks[0].plot_xy(mean=dict(color='green'), start=False)
# tracks_r.plot_xy(mean=dict(color='red'), start=False)
matplotlib.pyplot.gca().set_aspect(1)
for i in np.nonzero(is_multi)[0]:
    _ = matplotlib.pyplot.text(tracks.params['xy'][i, 0], tracks.params['xy'][i, 1], str(i))

# Input-output correlation
idx = is_multi
mask, first, last = tracks.endpoints(idx)
x = tracks.params['vxyz'][idx, 1][mask]
dx = tracks.params['vxyz_sigma'][idx, 1][mask]
y = tracks.vxyz[idx][mask, last, 1]
dy = tracks.vxyz_sigma[idx][mask, last, 1]
matplotlib.pyplot.figure()
# matplotlib.pyplot.plot(x, y, linestyle='none', marker='.', color='black')
matplotlib.pyplot.errorbar(x, x, yerr=dx, color='gray', alpha=1, linestyle='none', linewidth=2)
matplotlib.pyplot.errorbar(x, y, yerr=dy, color='black', alpha=1, linestyle='none', linewidth=2)
matplotlib.pyplot.xlim(matplotlib.pyplot.xlim())
matplotlib.pyplot.ylim(matplotlib.pyplot.ylim())
matplotlib.pyplot.plot((-1e3, 1e3), (-1e3, 1e3), linestyle='dotted', color='gray')
matplotlib.pyplot.title('2 cameras')
matplotlib.pyplot.xlabel('Input')
matplotlib.pyplot.ylabel('Output')

# Elevation
track = 4524
matplotlib.pyplot.figure()
tracks.plot_v1d(dim=2, tracks=track, sigma=True)
tracks_r.plot_v1d(dim=2, tracks=track, sigma=True)

# Animation
track = 4400
obs = 1
reverse = False
subplots = dict(figsize=(8, 4), tight_layout=True)
ani = (tracks_r if reverse else tracks).animate(track=track, obs=obs, subplots=subplots)
ani.save(os.path.join('tracks', date_str + '-' + str(i_obs) + ('_r' if reverse else '') + '-' + str(obs) + '-' + str(track) + '.mp4'))

# ---- Reverse run strategies ----

track = 5
dim = 0
titles = ('Forward', 'Forward: last vxy', 'Reverse', 'Reverse: last vxy')
fig, axes = matplotlib.pyplot.subplots(2, 2, tight_layout=True)
axes = axes.ravel()
limits = []
for i in range(len(tracks)):
    matplotlib.pyplot.sca(axes[i])
    tracks[i].plot_v1d(dim=dim, tracks=track, sigma=True)
    matplotlib.pyplot.title(titles[i])
    axes[i].set_xlim(axes[0].get_xlim())
    axes[i].set_ylim(axes[0].get_ylim())

# Overall sigma
idx = ~is_multi
fig, axes = matplotlib.pyplot.subplots(2, 2, tight_layout=True)
axes = axes.ravel()
for i in range(len(suffixes)):
    matplotlib.pyplot.sca(axes[i])
    temp = glimpse.helpers.read_pickle(basename + suffixes[i])
    track_vxy_sigma = np.nanmean(temp.vxyz_sigma[idx, :, 0:2], axis=1)
    mask = ~np.isnan(track_vxy_sigma[:, 0])
    matplotlib.pyplot.hist(track_vxy_sigma[mask, 0], color='gray', alpha=0.5)
    matplotlib.pyplot.hist(track_vxy_sigma[mask, 1], color='gray', alpha=0.5)
    matplotlib.pyplot.title(titles[i])
    # overall_speed_sigma = np.nanmean(np.sqrt(np.sum(track_vxy_sigma**2, axis=1)))
    # print(overall_speed_sigma)
    axes[i].set_xlim(axes[0].get_xlim())
    axes[i].set_ylim(axes[0].get_ylim())

# ---- Notes -----

# matplotlib.pyplot.figure()
# matplotlib.pyplot.gca().set_aspect(1)
# track = 4524
# dim = [0, 1]
# i = 0
# xyz = tracks[i].xyz[track, :, dim].T
# matplotlib.pyplot.plot(xyz[:, 0], xyz[:, 1], color = 'green')
# xyz = tracks[i + 1].xyz[track, :, dim].T
# matplotlib.pyplot.plot(xyz[:, 0], xyz[:, 1], color = 'green')
# xyz = tracks[i + 2].xyz[track, :, dim].T
# matplotlib.pyplot.plot(xyz[:, 0], xyz[:, 1], color = 'red')
# xyz = tracks[i + 3].xyz[track, :, dim].T
# matplotlib.pyplot.plot(xyz[:, 0], xyz[:, 1], color = 'red')
# #
# i = 0
# xyz = tracks[i].xyz[track, :, dim].T
# matplotlib.pyplot.plot(xyz[:, 0], xyz[:, 1], color = 'green')
# xyz = tracks[i + 2].xyz[track, :, dim].T
# matplotlib.pyplot.plot(xyz[:, 0], xyz[:, 1], color = 'red')
# #
# i = 0
# xyz = tracks[i].xyz[track, :, dim].T
# matplotlib.pyplot.plot(xyz[:, 0], xyz[:, 1], color = 'black')
#
# matplotlib.pyplot.figure()
# x = np.arange(means.shape[1])
# track = 4524
# dim = 1
# i = 0
# u, s = tracks[i].vxyz[track, :, dim], tracks[i].vxyz_sigma[track, :, dim]
# matplotlib.pyplot.fill_between(x, y1=u - s, y2=u + s, color = 'green', alpha=0.1)
# matplotlib.pyplot.plot(u, color = 'green')
# u, s = tracks[i + 1].vxyz[track, :, dim], tracks[i + 1].vxyz_sigma[track, :, dim]
# matplotlib.pyplot.fill_between(x, y1=u - s, y2=u + s, color = 'green', alpha=0.1)
# matplotlib.pyplot.plot(u, color = 'green')
# u, s = tracks[i + 2].vxyz[track, ::-1, dim], tracks[i + 2].vxyz_sigma[track, ::-1, dim]
# matplotlib.pyplot.fill_between(x, y1=u - s, y2=u + s, color = 'red', alpha=0.1)
# matplotlib.pyplot.plot(u, color = 'red')
# u, s = tracks[i + 3].vxyz[track, ::-1, dim], tracks[i + 3].vxyz_sigma[track, ::-1, dim]
# matplotlib.pyplot.fill_between(x, y1=u - s, y2=u + s, color = 'red', alpha=0.1)
# matplotlib.pyplot.plot(u, color = 'red')
# #
# i = 0
# u, s = tracks[i].vxyz[track, :, dim], tracks[i].vxyz_sigma[track, :, dim]
# matplotlib.pyplot.fill_between(x, y1=u - s, y2=u + s, color = 'green', alpha=0.25)
# matplotlib.pyplot.plot(u, color = 'green')
# u, s = tracks[i + 2].vxyz[track, ::-1, dim], tracks[i + 2].vxyz_sigma[track, ::-1, dim]
# matplotlib.pyplot.fill_between(x, y1=u - s, y2=u + s, color = 'red', alpha=0.25)
# matplotlib.pyplot.plot(u, color = 'red')
# #
# i = 0
# u, s = tracks[i].vxyz[track, :, dim], tracks[i].vxyz_sigma[track, :, dim]
# matplotlib.pyplot.fill_between(x, y1=u - s, y2=u + s, color = 'black', alpha=0.25)
# matplotlib.pyplot.plot(u, color = 'black')
#
# # Using weighted average with uncertainty propagation yield smaller than best uncertainty
# # Reasonable choice seems to take best result for each time step, where best is sqrt(sum(variance) of velocity?)
# # Alternative is to take the best run overall
# # Is this done per component or globally?
#
# # Combining reverse runs (f, r)
# # Using weighted average with uncertainty propagation yield smaller than best uncertainty
# # Reasonable choice seems to take best result for each time step, where best is sqrt(sum(variance) of velocity?)
# # Alternative is to take the best run overall
# # Is this done per component or globally?
#
# x = (10, 1)
# y = (10, 1)
# # sample = np.concatenate((
# #     x[0] + x[1] * np.random.randn(10000),
# #     y[0] + y[1] * np.random.randn(10000)))
# # sample.mean(), sample.var()
# # sample = np.concatenate((
# #     x[0] + x[1] * np.random.randn(10000),
# #     x[0] + y[1] * np.random.randn(10000)))
# # sample.mean(), sample.var()
# w = 1 / np.array((x[1], y[1]))
# w /= w.sum()
# mean = x[0] * w[0] + y[0] * w[1]
# var = w[0]**2 * x[1] + w[1]**2 * y[1] + 2 * w[0] * w[1] * 1
# mean, var, np.sqrt(var)
#
# # Use same tracker for all tracks
# for i in range(1, len(tracks)):
#     tracks[i].tracker = tracks[0].tracker
# # Reset paths
# for observer in tracks[0].tracker.observers:
#     for img in observer.images:
#         basename = glimpse.helpers.strip_path(img.path)
#         img.path = cg.find_image(basename)
# # Multi-camera mask
# is_multi = tracks[0].params['observer_mask'].sum(axis=1) > 1
# # Load dems
# # dem = glimpse.Raster.read(os.path.join('dems', date_str + '-' + str(i_obs) + '.tif'))
# # dem.plot(dem.hillshade(), cmap='gray')
#
# # ---- Plot result ----
#
# # Map
# matplotlib.pyplot.figure()
# matplotlib.pyplot.plot(cg.Glacier()[:, 0], cg.Glacier()[:, 1])
# tracks[0].plot_xy(mean=dict(color='green'), start=False)
# # tracks_r.plot_xy(mean=dict(color='red'), start=False)
# matplotlib.pyplot.gca().set_aspect(1)
# for i in np.nonzero(is_multi)[0]:
#     _ = matplotlib.pyplot.text(tracks[0].params['xy'][i, 0], tracks[0].params['xy'][i, 1], str(i))
#
#
# # ---- Reverse run strategies ----
#
# track = 4524
# obs = 0
# i = 3
# subplots = dict(figsize=(8, 4), tight_layout=True)
# ani = tracks[i].animate(track=track, obs=obs, subplots=subplots)
# ani.save(os.path.join('tracks', date_str + '-' + str(i_obs) + '-' + suffixes[i] + '-' + str(obs) + '-' + str(track) + '.mp4'))
#
#
# track = 4524
# track = 5488
# dim = 1
# titles = ('Forward', 'Forward: last vxy', 'Reverse', 'Reverse: last vxy')
# fig, axes = matplotlib.pyplot.subplots(2, 2, tight_layout=True)
# axes = axes.ravel()
# limits = []
# for i in range(len(tracks)):
#     matplotlib.pyplot.sca(axes[i])
#     tracks[i].plot_v1d(dim=dim, tracks=track, sigma=True)
#     matplotlib.pyplot.title(titles[i])
#     axes[i].set_xlim(axes[0].get_xlim())
#     axes[i].set_ylim(axes[0].get_ylim())
#
# track = 4524
# dim = 1
# means = np.column_stack([tracks[i].vxyz[track, :, dim] for i in (0, 1)])
# sigmas = np.column_stack([tracks[i].vxyz_sigma[track, :, dim] for i in (0, 1)])**2
#
# # smallest sigma
# ux = np.where(sigmas[:, 0] < sigmas[:, 1], means[:, 0], means[:, 1])
# sx = np.where(sigmas[:, 0] < sigmas[:, 1], sigmas[:, 0], sigmas[:, 1])
# # smallest overall sigma
# # mean weighted by inverse of the variance
# # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
# weights = 1 / sigmas**2
# weights /= weights.sum()
# uy = np.sum(weights * means, axis=1)
# # uncertainty for uncorrelated
# sy = np.sqrt(np.sum(weights**2 * sigmas**2, axis=1))
# # uncertainty for correlated (covariance of means over time)
# sy2 = np.sqrt(np.sum(weights**2 * sigmas**2, axis=1) + 2 * np.product(weights, axis=1) * np.cov(means[:, 0], means[:, 1])[0, 1])
# # uncertainty of weighted mean
# # sy3 = np.sqrt(1 / np.sum(weights, axis=1))
#
# # u = np.column_stack((means, ux, uy, sy, sy2, sy3))
# # s = np.column_stack((sigmas, sx, sy, sy, sy2, sy3))
# x = np.arange(len(means))
# matplotlib.pyplot.figure()
# # matplotlib.pyplot.plot(means)
# # matplotlib.pyplot.plot(ux)
# # matplotlib.pyplot.plot(uy)
# # matplotlib.pyplot.legend(('f', 'fv', 'best', 'weighted'))
# matplotlib.pyplot.plot(sigmas)
# matplotlib.pyplot.plot(sx)
# matplotlib.pyplot.plot(sy)
# matplotlib.pyplot.plot(sy2)
# # matplotlib.pyplot.plot(sy3)
# matplotlib.pyplot.legend(('f', 'fv', 'best', 'uncorrelated', 'correlated'))
#
# # matplotlib.pyplot.fill_between(x, y1=means[:, 0] - sigmas[:, 0], y2=means[:, 0] + sigmas[:, 0], alpha=0.5)
# # matplotlib.pyplot.fill_between(x, y1=means[:, 1] - sigmas[:, 1], y2=means[:, 1] + sigmas[:, 1], alpha=0.5)
# # matplotlib.pyplot.fill_between(x, y1=ux - sx, y2=ux + sx, alpha=0.5)
# matplotlib.pyplot.fill_between(x, y1=uy - sy, y2=uy + sy, alpha=0.5)
# matplotlib.pyplot.fill_between(x, y1=uy - sy2, y2=uy + sy2, alpha=0.5)
# matplotlib.pyplot.fill_between(x, y1=uy - sy3, y2=uy + sy3, alpha=0.5)
#
# # Overall sigma
# idx = ~is_multi
# fig, axes = matplotlib.pyplot.subplots(2, 2, tight_layout=True)
# axes = axes.ravel()
# for i in range(len(suffixes)):
#     matplotlib.pyplot.sca(axes[i])
#     temp = glimpse.helpers.read_pickle(basename + suffixes[i])
#     track_vxy_sigma = np.nanmean(temp.vxyz_sigma[idx, :, 0:2], axis=1)
#     mask = ~np.isnan(track_vxy_sigma[:, 0])
#     matplotlib.pyplot.hist(track_vxy_sigma[mask, 0], color='gray', alpha=0.5)
#     matplotlib.pyplot.hist(track_vxy_sigma[mask, 1], color='gray', alpha=0.5)
#     matplotlib.pyplot.title(titles[i])
#     # overall_speed_sigma = np.nanmean(np.sqrt(np.sum(track_vxy_sigma**2, axis=1)))
#     # print(overall_speed_sigma)
#     axes[i].set_xlim(axes[0].get_xlim())
#     axes[i].set_ylim(axes[0].get_ylim())
