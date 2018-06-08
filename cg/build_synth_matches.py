import cg
from cg import (glimpse, glob)
from glimpse.imports import (np, cv2, os, matplotlib)
cg.IMAGE_PATH = '/volumes/science-b/data/columbia/timelapse'

MAX_DISTANCE_SCALE = 0.025 # max match distance (fraction of image width)
MAX_ERROR_SCALE = 0.005 # max RANSAC error (fraction of image width)
CAM_DXYZ = 25 # max camera displacement (meters)
MAX_PARALLAX = 0.5 # pixels

images = [glimpse.helpers.strip_path(path)
    for path in glob.glob(os.path.join('svg-synth', '*.svg'))]
for image in images:
    basename = os.path.join(cg.CG_PATH, 'svg-synth', image)
    # Skip if output exists
    if os.path.isfile(basename + '.png'):
        continue
    print(image)
    # Prepare image
    cam = cg.load_calibrations(image, station=True, camera=True, image=True, viewdir=True, merge=True, file_errors=False)
    img = glimpse.Image(basename + '.JPG', cam=cam)
    I = img.read()
    if I.ndim > 2:
        I = glimpse.helpers.rgb_to_gray(I).astype(np.uint8)
    # Prepare synthetic image
    cam = glimpse.helpers.read_json(basename + '-synth.json')
    simg = glimpse.Image(basename + '-synth.JPG', cam=cam)
    sI = simg.read()
    if sI.ndim > 2:
        smask = (sI[:, :, 0] != 127).astype(np.uint8)
        sI = glimpse.helpers.rgb_to_gray(sI).astype(np.uint8)
    else:
        smask = (sI != 127).astype(np.uint8)
    depth = glimpse.Raster.read(basename + '-depth.tif')
    depth_sigma = glimpse.Raster.read(basename + '-depth_stderr.tif')
    depth_sigma.Z[np.isnan(depth_sigma.Z)] = 0
    # # Equalize images
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))
    # I = clahe.apply(I)
    # sI = clahe.apply(sI)
    # Match keypoints
    k = glimpse.optimize.detect_keypoints(I)
    sk = glimpse.optimize.detect_keypoints(sI, mask=smask)
    uv, suv, ratio = glimpse.optimize.match_keypoints(k, sk, max_ratio=0.75,
        max_distance=MAX_DISTANCE_SCALE * img.cam.imgsz[0], return_ratios=True)
    print(len(uv), 'matches')
    # matplotlib.pyplot.hist(ratio)
    # Enforce max parallax
    d = depth.sample(suv)
    d_sigma = depth_sigma.sample(suv)
    theta = np.arctan((d + d_sigma * 2) / CAM_DXYZ) - np.arctan(d / CAM_DXYZ)
    theta_per_pixel = 2 * np.arctan(img.cam.imgsz[0] / img.cam.f[0]) / img.cam.imgsz[0]
    pixels = theta / theta_per_pixel
    keep = pixels < MAX_PARALLAX
    uv, suv, ratio = uv[keep], suv[keep], ratio[keep]
    # print('Max parallax:', round(np.nanmax(pixels), 2), 'pixels')
    print(len(uv), 'matches (after parallax filter)')
    # Build "world" points
    dxyz = simg.cam.invproject(suv)
    xyz = simg.cam.xyz + dxyz * depth.sample(suv).reshape(-1, 1)
    points = glimpse.optimize.Points(cam=img.cam, uv=uv, xyz=xyz, directions=False)
    # Filter with RANSAC
    # NOTE: 'AKJNC_20120813_205325' needed dict(viewdir=True, k=0, f=True)
    model = glimpse.optimize.Cameras(cams=[img.cam], controls=[points], cam_params=[dict(viewdir=True, k=0)])
    try:
        params, selected = glimpse.optimize.ransac(model,
            sample_size=5, max_error=MAX_ERROR_SCALE * img.cam.imgsz[0],
            min_inliers=5, iterations=100)
        model.set_cameras(params)
    except:
        selected = []
    print(len(selected), 'matches (after ransac filter)')
    # Plot results
    matplotlib.pyplot.figure(figsize=(12, 8))
    img.plot(cmap='gray')
    points.plot(index=selected, selected='yellow', unselected='red', scale=1, width=20)
    matplotlib.pyplot.gca().xaxis.set_visible(False)
    matplotlib.pyplot.gca().yaxis.set_ticks([])
    matplotlib.pyplot.tight_layout(h_pad=0.1)
    matplotlib.pyplot.savefig(os.path.join('svg-synth', image + '.png'), dpi=200, bbox_inches='tight')
    matplotlib.pyplot.clf()
    model.reset_cameras()
    # # Save to file
    # filtered_points = glimpse.optimize.Points(cam=img.cam, uv=uv[selected], xyz=xyz[selected])
    # glimpse.helpers.write_pickle(filtered_points, basename + '.pkl')
    # Add to svg file
    svg_path = basename + '.svg'
    if os.path.isfile(svg_path):
        svg = glimpse.svg._read_svg(svg_path)
        scale = glimpse.svg._parse_svg_size(svg) / img.cam.imgsz
        lines = zip(uv[selected] * scale, suv[selected] * scale)
        group = glimpse.svg._g(
            *[glimpse.svg._line(start=start, end=end) for start, end in lines],
            id='points-auto'
        )
        svg.append(group)
        glimpse.svg._write_svg(svg, path=svg_path, pretty_print=True)
