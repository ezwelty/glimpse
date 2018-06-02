import cg
from cg import (glimpse, glob)
from glimpse.imports import (os, re, np, matplotlib)
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

IMG_SIZE = 0.5
FIGURE_SIZE = 0.25
MAX_RATIO = 0.5

# For each motion sequence...
motion = glimpse.helpers.read_json('motion.json')
for d in motion:
    paths = np.asarray(d['paths'])
    # Skip if all files already exist
    basenames = [os.path.join('motion', paths[i] + '-' + paths[i + 1])
        for i in range(len(paths) - 1)]
    nexists = np.sum([os.path.isfile(basename + '.pkl') for basename in basenames])
    if nexists == len(paths) - 1:
        continue
    # Load images
    images = [glimpse.Image(cg.find_image(path)) for path in paths]
    # Compute sequential matches
    for img in images:
        img.cam.resize(IMG_SIZE)
    matches = cg.build_sequential_matches(images, match=dict(max_ratio=MAX_RATIO))
    # For each motion pair...
    for i, control in enumerate(matches):
        # Skip if file exists
        if os.path.isfile(basenames[i] + '.pkl'):
            continue
        print(basenames[i])
        # Initialize control
        control.resize(1)
        # Filter with RANSAC
        model = glimpse.optimize.Cameras(
            control.cams, control,
            cam_params=[dict(), dict(viewdir=True)], group_params=dict(k=0),
            sparsity=False)
        params, selected = glimpse.optimize.ransac(
            model, max_error=0.002 * control.imgszs[0][0], iterations=100,
            sample_size=min(50, round(control.size * 0.25)),
            min_inliers=min(50, round(control.size * 0.25)))
        # Plot results
        imgs = images[i:(i + 2)]
        model.set_cameras(params)
        control.resize(FIGURE_SIZE)
        fig = matplotlib.pyplot.figure()
        imgs[0].plot()
        matplotlib.pyplot.imshow(imgs[1].project(imgs[0].cam), alpha=0.5)
        control.plot(
            index=selected, selected='yellow', unselected='red',
            scale=5, width=2)
        matplotlib.pyplot.gca().xaxis.set_visible(False)
        matplotlib.pyplot.gca().yaxis.set_ticks([])
        matplotlib.pyplot.tight_layout(h_pad=0.1)
        matplotlib.pyplot.savefig(basenames[i] + '.png', dpi=100, bbox_inches='tight')
        matplotlib.pyplot.close(fig)
        control.resize(1)
        model.reset_cameras()
        # Write results
        control.uvs = [uv[selected, :] for uv in control.uvs]
        glimpse.helpers.write_pickle(control, basenames[i] + '.pkl')
