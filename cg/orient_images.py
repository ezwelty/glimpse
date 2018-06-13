import cg
from cg import (glimpse, glob)
from glimpse.imports import (os, re, np, matplotlib)
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

# ---- Constants ----

keys = {'gcp', 'horizon', 'terminus', 'moraines', 'coast'}
camera_keys = {
    'nikon-d2x': keys - {'coast'},
    'nikon-d200-04-24': keys - {'coast'},
    'nikon-d200-08-24': keys - {'coast'},
    'nikon-d200-13-20': keys - {'coast', 'moraines'},
    'nikon-d200-14-20': keys - {'coast', 'moraines'},
    'canon-40d-01': keys - {'terminus', 'coast'}
}
step = 20 # pixels
suffixes = ['', '-calib', '-ideal']
stations = (
    'AK01', 'AK01b', 'AK03', 'AK03b', 'AK09', 'AK09b', 'AK10', 'AK10b',
    'AK12', 'AKJNC', 'AKST03A', 'AKST03B', 'CG04', 'CG05', 'CG06')

# ---- Orient anchor images ----

paths = glob.glob(os.path.join('svg', '*.svg'))
paths += glob.glob(os.path.join('svg-synth', '*.svg'))
metas = [cg.parse_image_path(path) for path in paths]
selected = [meta['station'] in stations for meta in metas]
paths = [path for path, meta in zip(paths, metas)
    if meta['station'] in stations]

for path in paths:
    meta = cg.parse_image_path(path, sequence=True)
    svg_keys = camera_keys.get(meta['camera'], keys)
    for suffix in suffixes:
        if not os.path.isfile(os.path.join(
            'cameras', meta['camera'] + suffix + '.json')):
            continue
        basename = os.path.join('images', meta['basename'] + suffix)
        if os.path.isfile(basename + '.json'):
            continue
        print(meta['basename'] + suffix)
        # TODO: Use station xyz estimated for calib for non-fixed stations
        calibration = cg.load_calibrations(path,
            station_estimate=meta['station'], station=meta['station'],
            camera=meta['camera'] + suffix, merge=True, file_errors=False)
        img_path = cg.find_image(path)
        img = glimpse.Image(img_path, cam=calibration)
        controls = cg.svg_controls(img, keys=svg_keys, step=step)
        controls += cg.synth_controls(img, step=step)
        if not controls:
            print("No controls found")
            continue
        model = glimpse.optimize.Cameras(
            cams=img.cam, controls=controls, cam_params=dict(viewdir=True))
        fit = model.fit(full=True)
        model.set_cameras(fit.params)
        img.cam.write(basename + '.json',
            attributes=('xyz', 'viewdir', 'fmm', 'cmm', 'k', 'p', 'sensorsz'),
            indent=4, flat_arrays=True)
        # Plot image with markup
        fig = matplotlib.pyplot.figure(
            figsize=tuple(img.cam.imgsz / 100), dpi=100 * 0.25, frameon=False)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis('off')
        img.plot()
        model.plot(
            lines_observed=dict(color='yellow', linewidth=3),
            lines_predicted=dict(color='red', linewidth=2))
        img.set_plot_limits()
        matplotlib.pyplot.savefig(basename + '-markup.jpg', dpi=100)
        matplotlib.pyplot.close()
