import cg
from cg import (glimpse, glob)
from glimpse.imports import (os, re, np, matplotlib)
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

IMG_SIZE = 0.5
FIGURE_SIZE = 0.25

motion = glimpse.helpers.read_json('motion.json')
[cg.parse_image_path(d['paths'][0], sequence=True)['camera'] for d in motion]

# For each motion sequence...
for d in motion:
    # Load images
    img_paths = d['paths']
    images = [glimpse.Image(cg.find_image(path)) for path in img_paths]
    # Compute sequential matches
    for img in images:
        img.cam.resize(IMG_SIZE)
    matches = cg.build_sequential_matches(images, match=dict(max_ratio=0.5))
    # For each motion pair...
    for control in matches:
        control.resize(1)
        # Initialize control
        img = images[[img.cam for img in images].index(control.cams[0])]
        img2 = images[[img.cam for img in images].index(control.cams[1])]
        basename = os.path.join('motion', glimpse.helpers.strip_path(img.path) + '-' + glimpse.helpers.strip_path(img2.path))
        if os.path.isfile(basename + '.pkl'):
            continue
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
        model.set_cameras(params)
        control.resize(FIGURE_SIZE)
        fig = matplotlib.pyplot.figure()
        img.plot()
        matplotlib.pyplot.imshow(img2.project(img.cam), alpha=0.5)
        control.plot(
            index=selected, selected='yellow', unselected='red',
            scale=5, width=2)
        matplotlib.pyplot.gca().xaxis.set_visible(False)
        matplotlib.pyplot.gca().yaxis.set_ticks([])
        matplotlib.pyplot.tight_layout(h_pad=0.1)
        matplotlib.pyplot.savefig(basename + '.png', dpi=100, bbox_inches='tight')
        matplotlib.pyplot.close(fig)
        control.resize(1)
        model.reset_cameras()
        # Write results
        control.uvs = [uv[selected, :] for uv in control.uvs]
        glimpse.helpers.write_pickle(control, basename + '.pkl')
