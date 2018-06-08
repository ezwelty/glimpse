import cg
from cg import (glimpse, glob)
from glimpse.imports import (sys, os, re, np, matplotlib, datetime)
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

# ---- Functions ---- #

def load_model(camera, svgs=None, keys=None, step=None, group_params=dict(),
    station_calib=False, camera_calib=False, fixed=None):
    # Gather motion control
    motion_images, motion_controls, motion_cam_params = cg.camera_motion_matches(
        camera, station_calib=station_calib, camera_calib=camera_calib)
    # Gather svg control
    svg_images, svg_controls, svg_cam_params = cg.camera_svg_controls(
        camera, keys=keys, svgs=svgs, correction=True,
        station_calib=station_calib, camera_calib=camera_calib, step=step)
    # Standardize image sizes
    imgszs = np.unique([img.cam.imgsz for img in (motion_images + svg_images)], axis=0)
    if len(imgszs) > 1:
        i_max = np.argmax(imgszs[:, 0])
        print('Resizing images and controls to', imgszs[i_max])
        for control in motion_controls + svg_controls:
            control.resize(size=imgszs[i_max], force=True)
        # Set new imgsz as original camera imgsz
        for img in motion_images + svg_images:
            img.cam.original_vector[6:8] = imgszs[i_max]
    # Determine whether xyz can be optimized
    stations = [cg.parse_image_path(img.path)['station'] for img in svg_images]
    if fixed is None:
        if len(stations) > 0 and (np.array(stations) == stations[0]).all():
            fixed = cg.Stations()[stations[0]]['properties']['fixed']
        else:
            fixed = True
    station = None if fixed else stations[0]
    if station:
        group_params = glimpse.helpers.merge_dicts(group_params, dict(xyz=True))
    model = glimpse.optimize.Cameras(
        cams=[img.cam for img in motion_images + svg_images],
        controls=motion_controls + svg_controls,
        cam_params=motion_cam_params + svg_cam_params,
        group_params=group_params)
    return motion_images, svg_images, model, station

def write_calibration(camera, model, fit, group=0, suffix='', station=None):
    # model.set_cameras(fit.params)
    i = model.group_indices[group][0]
    # Save means
    means = [item.value for item in fit.params.values()]
    model.set_cameras(means)
    # (fmm, cmm, k, p, sensorsz)
    model.cams[i].write(
        path=os.path.join('cameras', camera + suffix + '.json'),
        attributes=('fmm', 'cmm', 'k', 'p', 'sensorsz'),
        indent=4, flat_arrays=True)
    # (xyz)
    if station:
        xyz0 = cg.load_calibrations(station_estimate=station, merge=True)['xyz']
        print('xyz deviation:', model.cams[i].xyz - xyz0)
        model.cams[i].write(
            path=os.path.join('stations', station + suffix + '.json'),
            attributes=['xyz'], indent=4, flat_arrays=True)
    # Save standard errors
    stderrs = [item.stderr for item in fit.params.values()]
    if fit.errorbars:
        model.set_cameras(stderrs)
        # (fmm, cmm, k, p)
        model.cams[i].write(
            path=os.path.join('cameras', camera + suffix + '_stderr.json'),
            attributes=('fmm', 'cmm', 'k', 'p'),
            indent=4, flat_arrays=True)
        # (xyz)
        if station:
            model.cams[i].write(
                path=os.path.join('stations', station + suffix + '_stderr.json'),
                attributes=['xyz'], indent=4, flat_arrays=True)
    model.reset_cameras()

# ---- Calibrate cameras ---- #

svgs = (
	'AK01_20070817_200132', # 2
	'AK01_20070922_230210', # 4
	'AK01b_20080811_193452', # 4
	# 'AK01b_20080922_193508', # 2
	# 'AK01b_20090319_213755', # 2
	# 'AK01b_20090502_193824', # 2
	'AK01b_20090827_202939', # 4
	# 'AK01b_20100202_220056', # 1
	# 'AK01b_20100403_200128', # 2
	# 'AK01b_20100509_200138', # 2
	# 'AK01b_20100525_200710', # 2
	'AK01b_20100602_200707', # 2
	'AK01b_20100915_200427', # 2
	'AK01b_20110711_195216', # 2
	'AK01b_20120731_192459', # 2
	# 'AK01b_20130609_184325', # 2
	# 'AK01b_20130612_162325', # 2
	# 'AK01b_20130615_220325', # 2
	# 'AK01b_20130620_204325', # 2
	# 'AK01b_20130623_010325', # 1
	# 'AK01b_20130625_012325', # 1
	# 'AK01b_20131106_192014', # 2
	# 'AK01b_20150820_212106', # 0
    # NOTE: Skip AK02
	# 'AK02_20070706_195704', # 1
	# 'AK02_20070811_195526', # 1
	# 'AK02_20070922_195333', # 1
	'AK03_20071007_185318', # 2
	'AK03_20080208_205507', # 2
	# 'AK03_20080608_181855', # 2
	'AK03b_20080621_201611', # 2
	'AK03b_20080811_191645', # 4
	'AK03b_20090503_162210', # 2
    # NOTE: Skip AK04
	# 'AK04_20070610_180021', # 2
	# 'AK04_20070619_180030', # 2
	# 'AK09_20090516_200409', # 2
	'AK09_20090803_200320', # 4
	'AK09_20090827_200153', # 4
	'AK09b_20090827_200153', # 4
	# 'AK09b_20091102_195909', # 0
	# 'AK09b_20100101_195948', # 0
	'AK09b_20100525_200819', # 3
	'AK09b_20100602_200818', # 3
	'AK10_20090803_195410', # 3
	'AK10_20090827_192655', # 3
    'AK10b_20120605_203759', # 2
	'AK10b_20120629_205821', # 2
	'AK10b_20121012_160709', # 4
	# 'AK10b_20130615_013704', # 1
	# 'AK10b_20131106_200636', # 2
	'AK10b_20150724_172806', # 2
	'AK10b_20160918_201825', # 2
	'AK12_20100906_214611', # 2
	'AK12_20100924_204619', # 3
	# 'AK12_20100925_231620', # 0
	# 'AK12_20101118_194702', # 0
	# 'AK12_20101130_211712', # 0
	# 'AK12_20110217_204832', # 0
	# 'AK12_20110606_195154', # 0
	'AK12_20110721_200018', # 3
	'AK12_20110812_210007', # 3
	'AKJNC_20120605_223513', # 4
	'AKJNC_20120813_205325', # 4
	'AKJNC_20121001_155719', # 4
	'AKST03A_20100525_224800', # 3
	'AKST03A_20100602_224800', # 3
	'AKST03A_20100807_184800', # 3
	'AKST03B_20100525_224800', # 3
	'AKST03B_20100602_224800', # 3
	'CG04_20040624_200016', # 2
	'CG04_20040707_200052', # 4
	'CG04_20041008_200016', # 2
	'CG05_20050610_150000', # 2
	'CG05_20050623_200000', # 2
	'CG05_20050811_130000', # 2
	'CG05_20050811_190000', # 2
	# 'CG05_20050822_130000', # 1
	# 'CG05_20050827_130000', # 1
	'CG05_20050827_190000', # 2
	'CG05_20050914_190000', # 2
	'CG06_20060627_195956', # 2
	'CG06_20060712_195953', # 3
	'CG06_20060727_195951' # 3
	# 'CG06_20060913_215944', # 1
)
group_params = [
    dict(),
    dict(f=True),
    dict(f=True, k=[0]),
    dict(f=True, k=[0, 1]),
    dict(f=True, k=[0, 1], p=True),
    dict(f=True, k=[0, 1], p=True, c=True)]
group_params_xyz = group_params[:4] + [
    dict(f=True, k=[0, 1], xyz=True),
    dict(f=True, k=[0, 1], xyz=True, p=True),
    dict(f=True, k=[0, 1], xyz=True, p=True, c=True)]
keys = ('moraines', 'gcp', 'horizon', 'terminus', 'coast')
step = 20 # pixels
jobs = (
    dict(camera='nikon-e8700', keys=keys, group_params=group_params), # CG04
    dict(camera='nikon-d2x', keys=('moraines', 'gcp', 'horizon', 'terminus'), group_params=group_params), # CG05
    dict(camera='canon-20d', keys=keys, group_params=group_params_xyz), # CG06 (not fixed)
    dict(camera='nikon-d200-04-24', keys=('moraines', 'gcp', 'horizon', 'terminus'), group_params=group_params_xyz), # AK01-2 (not fixed)
    dict(camera='nikon-d200-08-24', keys=keys, group_params=group_params), # AK01b NOTE: Large c
    dict(camera='nikon-d200-03-20', keys=keys, group_params=group_params), # AK03, AK03b NOTE: Foreground control suspect
    dict(camera='nikon-d200-13-20', keys=('gcp', 'horizon', 'terminus'), group_params=group_params_xyz), # AK09 (not fixed)
    dict(camera='nikon-d200-14-20', keys=('gcp', 'horizon', 'terminus'), group_params=group_params_xyz), # AK09b (not fixed) NOTE: cy very large
    dict(camera='nikon-d200-10-24', keys=keys, group_params=group_params), # AK10
    dict(camera='nikon-d200-17-20', keys=keys, group_params=group_params), # AK10b-1
    dict(camera='nikon-d200-18-20', keys=keys, group_params=group_params), # AK10b-2
    dict(camera='nikon-d300s', keys=keys, group_params=group_params), # AK12
    dict(camera='canon-40d-01', keys=('gcp', 'horizon', 'coast', 'moraines'), group_params=group_params), # AKJNC
    dict(camera='nikon-d200-11-28', keys=keys, group_params=group_params), # AKST0XA
    dict(camera='nikon-d200-12-28', keys=keys, group_params=group_params) # AKST0XB
)

# sys.stdout = open(os.path.join('logs', 'calibrate-cameras' + suffix + '.log'), 'w')
# sys.stdout = sys.__stdout__
for job in jobs:
    print(job['camera'])
    motion_images, svg_images, model, station = load_model(
        job['camera'], keys=job['keys'], group_params=job['group_params'][-3],
        svgs=svgs, step=step, camera_calib=False, fixed=None)
    xyz_added = station and 'xyz' not in job['group_params'][-1]
    fit = model.fit(
        full=True, method='leastsq',
        group_params=job['group_params']
        if xyz_added else job['group_params'][:-3])
    suffix = datetime.datetime.now().strftime('-%Y%m%d%H%M%S')
    write_calibration(job['camera'],
        model=model, fit=fit, station=station, suffix=suffix)

# ---- Multi-camera calibration: AK09, AK09b with equal xyz ---- #

cameras = ('nikon-d200-13-20', 'nikon-d200-14-20')
stations = ('AK09', 'AK09b')
motion_images, svg_images = [], []
cams, controls, cam_params, ncams = [], [], [], [0]
for camera in cameras:
    motion_imgs, svg_imgs, model, station = load_model(
        camera, keys=('gcp', 'horizon', 'terminus'), group_params=group_params[-1],
        svgs=svgs, step=step, camera_calib=False, fixed=True)
    motion_images.extend(motion_imgs)
    svg_images.extend(svg_imgs)
    cams.extend(model.cams)
    controls.extend(model.controls)
    cam_params.extend(model.cam_params)
    ncams.append(len(model.cams))
group_indices = [range(start, start + ncams[i + 1]) for i, start in enumerate(ncams[:-1])]
model = glimpse.optimize.Cameras(cams, controls, cam_params, group_indices + [range(len(cams))],
    group_params=[group_params[-1], group_params[-1], dict(xyz=True)])
fit = model.fit(method='leastsq', full=True,
    group_params=list(zip(group_params[:-1], group_params[:-1], [dict()] * 4 + [dict(xyz=True)])))
suffix = datetime.datetime.now().strftime('-%Y%m%d%H%M%S')
for group, camera in enumerate(cameras):
    write_calibration(camera, model, fit, group=group,
        station=stations[group], suffix=suffix)

# ---- Check calibration ---- #

camera = 'nikon-d200-04-24'
keys = ('moraines', 'gcp', 'horizon', 'terminus')
motion_images, svg_images, model, _ = load_model(camera, keys=keys, svgs=svgs,
    camera_calib=True, station_calib=True, fixed=True, step=step)
fit = model.fit(full=True)
model.set_cameras(fit.params)

# Print errors
model.errors().mean(), model.errors().std()
motion_errors = np.linalg.norm(np.vstack([control.observed() - control.predicted()
    for control in model.controls
    if isinstance(control, glimpse.optimize.Matches)]), axis=1)
motion_errors.mean(), motion_errors.std()
svg_errors = np.linalg.norm(np.vstack([control.observed() - control.predicted()
    for control in model.controls
    if isinstance(control, (glimpse.optimize.Points, glimpse.optimize.Lines))]), axis=1)
svg_errors.mean(), svg_errors.std()

# Plot svg images
for i, img in enumerate(svg_images):
    matplotlib.pyplot.figure()
    img.plot()
    model.plot(cam=model.cams.index(img.cam))
    matplotlib.pyplot.title(glimpse.helpers.strip_path(img.path))
    img.set_plot_limits()

# Plot match images
matches = [control for control in model.controls
    if isinstance(control, glimpse.optimize.Matches)]
scale = 0.5
for m in matches:
    matplotlib.pyplot.figure()
    img = motion_images[model.cams.index(m.cams[0])]
    imgB = motion_images[model.cams.index(m.cams[1])]
    img.cam.resize(scale)
    imgB.cam.resize(scale)
    m.resize(scale)
    img.plot()
    I = imgB.project(img.cam)
    matplotlib.pyplot.imshow(I, alpha=0.5)
    m.plot(scale=20, width=1, selected='yellow')
    errors = (1 / scale) * np.linalg.norm(m.observed() - m.predicted(), axis=1)
    matplotlib.pyplot.title(
        glimpse.helpers.strip_path(img.path) + ' - ' + glimpse.helpers.strip_path(imgB.path) + '\n'
        + str(round(errors.mean(), 2)) + ', ' + str(round(errors.std(), 2)))
    img.set_plot_limits()
    img.cam.resize(1)
    imgB.cam.resize(1)
    m.resize(1)

# ---- Check single image (svg) ---- #

basename = 'AK10b_20120605_203759'
img = glimpse.Image(
    path=cg.find_image(basename),
    cam=cg.load_calibrations(basename, station_estimate=True, merge=True))
controls = cg.svg_controls(img)
svg_model = glimpse.optimize.Cameras(img.cam, controls,
    cam_params=dict(viewdir=True), group_params=group_params[-1])
svg_fit = svg_model.fit(full=True, group_params=group_params[:-1])
matplotlib.pyplot.figure()
img.plot()
svg_model.plot(svg_fit.params)
img.set_plot_limits()

# ---- Check undistorted image ---- #

basename = 'AKJNC_20120508_191103C'
img = glimpse.Image(
    path=cg.find_image(basename),
    cam=cg.load_calibrations(basename, camera=True, merge=True))
ideal_cam = img.cam.copy()
ideal_cam.idealize()
I = img.project(ideal_cam)
img.write(path='test.jpg', I=I)
