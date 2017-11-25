import datetime
import pytest
import numpy as np
import scipy.misc
import cv2
import image
import optimize
import dem as DEM
import svg
# HACK: Needed for pytest from command line (?)
reload(svg)

# ---- Camera ----

def test_camera_init_with_fmm(fmm=[10, 10], sensorsz=[30, 20]):
    cam = image.Camera(fmm=fmm, sensorsz=sensorsz)
    assert np.array_equal(cam.f, fmm * cam.imgsz / sensorsz)

def test_camera_resize(imgsz=[100, 100]):
    cam = image.Camera(imgsz=imgsz)
    cam.resize(0.5)
    assert np.array_equiv(cam.imgsz * 2, imgsz)
    cam.resize(2)
    assert np.array_equiv(cam.imgsz, imgsz)

def test_camera_idealize(c=1, k=1, p=1):
    cam = image.Camera(c=c, k=k, p=p)
    cam.idealize()
    assert (cam.c == 0).all()
    assert (cam.k == 0).all()
    assert (cam.p == 0).all()

def pixel_centers(cam):
    """Return image coordinates of all pixel centers (Nx2)."""
    u = np.linspace(0.5, cam.imgsz[0] - 0.5, int(cam.imgsz[0]))
    v = np.linspace(0.5, cam.imgsz[1] - 0.5, int(cam.imgsz[1]))
    U, V = np.meshgrid(u, v)
    return np.column_stack((U.flatten(), V.flatten()))

def reprojection_errors(cam):
    """Compute reprojection errors at all pixel centers."""
    uv = pixel_centers(cam)
    dxyz = cam.invproject(uv)
    puv = cam.project(dxyz, directions=True)
    return np.sqrt(np.sum((puv - uv)**2, axis=1))

def test_camera_reprojection_ideal(tol=1e-14):
    cam = image.Camera()
    err = reprojection_errors(cam)
    assert err.max() < tol

def test_camera_reprojection_distorted(tol=1e-12):
    # Positive k1
    cam = image.Camera(k=0.1)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Negative k1
    cam = image.Camera(k=-0.1)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Radial distortion only
    cam = image.Camera(k=[0.1] * 6)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Tangential distortion only
    cam = image.Camera(p=[0.01] * 2)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # All distortion
    cam = image.Camera(k=[0.1] * 6, p=[0.01] * 2)
    err = reprojection_errors(cam)
    assert err.max() < tol

def test_camera_reprojection_extreme(tol=1e-12):
    # Positive k1
    cam = image.Camera(k = 2)
    err = reprojection_errors(cam)
    assert err.max() < tol
    # Negative k1
    cam = image.Camera(k = -2)
    err = reprojection_errors(cam)
    assert err.max() < tol

# ---- Exif ----

def test_exif_test_image():
    path = "tests/AK10b_20141013_020336.JPG"
    exif = image.Exif(path)
    assert np.array_equiv(exif.size, scipy.misc.imread(path).shape[0:2][::-1])
    assert exif.fmm == 20
    assert exif.make == "NIKON CORPORATION"
    assert exif.model == "NIKON D200"
    assert exif.iso == 200
    assert exif.shutter == 0.0125
    assert exif.aperture == 8
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36, 28)

def test_exif_subsecond():
    path = "tests/AK10b_20141013_020336.JPG"
    exif = image.Exif(path)
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36, 28)
    exif.set_tag('SubSecTimeOriginal', None)
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36)

# ---- Image ----

def test_image_init():
    path = "tests/AK10b_20141013_020336.JPG"
    # Defaults
    img = image.Image(path)
    assert img.path == path
    assert img.datetime == img.exif.datetime
    assert all(img.cam.imgsz == img.exif.size)
    sensorsz = image.get_sensor_size(img.exif.make, img.exif.model)
    assert all(img.cam.f == img.exif.fmm * img.exif.size / sensorsz)
    # Override defaults
    img_time = datetime.datetime(2014, 10, 13)
    camera_args = {'imgsz': [100, 100], 'sensorsz': [10, 10]}
    img = image.Image(path, datetime=img_time, camera_args=camera_args)
    assert img.datetime == img_time
    assert all(img.cam.imgsz == camera_args['imgsz'])
    assert all(img.cam.f == img.exif.fmm * np.divide(camera_args['imgsz'], camera_args['sensorsz']))

def test_image_read():
    path = "tests/AK10b_20141013_020336.JPG"
    # Default size
    img = image.Image(path)
    I = img.read()
    assert all(I.shape[0:2][::-1] == img.cam.imgsz)
    # Resize camera
    img.cam.resize(0.5)
    I = img.read()
    assert all(I.shape[0:2][::-1] == img.cam.imgsz)

# ---- DEM ----

def test_dem_defaults():
    Z = np.zeros([3, 3])
    dem = DEM.DEM(Z)
    assert np.array_equal(dem.xlim, np.array([0., 3.]))
    assert np.array_equal(dem.ylim, np.array([3., 0.]))
    assert np.array_equal(dem.x, np.array([0.5,  1.5,  2.5]))
    assert np.array_equal(dem.y, np.array([2.5,  1.5,  0.5]))
    assert np.array_equal(dem.X, np.array([dem.x, dem.x, dem.x]))
    assert np.array_equal(dem.Y, np.array([dem.y, dem.y, dem.y]).transpose())
    assert np.array_equal(dem.zlim, np.array([0., 0.]))
    assert np.array_equal(dem.n, np.array([3, 3]))
    assert np.array_equal(dem.d, np.array([1., -1.]))
    assert np.array_equal(dem.min, np.array([0., 0., 0.]))
    assert np.array_equal(dem.max, np.array([3., 3., 0.]))

def test_dem_xy():
    xlim = [0, 3]
    ylim = [3, 0]
    x = [0.5, 1.5, 2.5]
    y = [2.5, 1.5, 0.5]
    X = [x, x, x]
    Y = np.asarray([y, y, y]).transpose()
    Z = np.zeros([3, 3])
    # Initialize from xlim, ylim
    dem = DEM.DEM(Z, x=xlim, y=ylim)
    assert np.array_equiv(dem.xlim, xlim)
    assert np.array_equiv(dem.ylim, ylim)
    assert np.array_equiv(dem.x, x)
    assert np.array_equiv(dem.y, y)
    assert np.array_equiv(dem.X, X)
    assert np.array_equiv(dem.Y, Y)
    # Initialize from x, y
    dem = DEM.DEM(Z, x=x, y=y)
    assert np.array_equiv(dem.xlim, xlim)
    assert np.array_equiv(dem.ylim, ylim)
    # Initialize from X, Y
    dem = DEM.DEM(Z, x=X, y=Y)
    assert np.array_equiv(dem.xlim, xlim)
    assert np.array_equiv(dem.ylim, ylim)
    assert np.array_equiv(dem.x, x)
    assert np.array_equiv(dem.y, y)

def test_dem_sample(tol=1e-13):
    Z = np.reshape(range(0, 16, 1), [4, 4])
    dem = DEM.DEM(Z, [-0.5, 3.5], [-0.5, 3.5])
    # Sample cells centers along diagonal
    xy_diagonal = np.array([dem.x, dem.y]).transpose()
    dz_points = dem.sample(xy_diagonal) - dem.Z.diagonal()
    assert np.all(dz_points < tol)
    # Sample cell center grid
    dz_grid = dem.sample_grid(dem.x, dem.y) - dem.Z
    assert np.all(dz_grid < tol)
    # Sample transect through cell centers
    x_transect = np.arange(0, 3.5, 0.5)
    dz_transect = dem.sample_grid(x_transect, 0).flatten() - x_transect
    assert np.all(dz_transect < tol)

def test_dem_crop_ascending():
    Z = np.arange(1, 10, 1).reshape([3, 3])
    dem = DEM.DEM(Z, [0, 3], [0, 3])
    # Out of bounds
    with pytest.raises(Exception):
        dem.crop([3, 5])
    # Equal bounds
    cdem = dem.crop(xlim=[0, 3], ylim=[0, 3])
    assert np.array_equal(dem.Z, cdem.Z)
    # Crop bounds: x left edge
    cdem = dem.crop(xlim=[0, 2])
    assert np.array_equiv(cdem.xlim, [0, 2])
    assert np.array_equiv(cdem.Z, Z[:, 0:2])
    # Crop bounds: x right edge (w/ overshoot)
    cdem = dem.crop(xlim=[2, 4])
    assert np.array_equiv(cdem.xlim, [2, 3])
    assert np.array_equiv(cdem.Z, Z[:, 2:3])
    # Crop bounds: y top edge
    cdem = dem.crop(ylim=[0, 2])
    assert np.array_equiv(cdem.ylim, [0, 2])
    assert np.array_equiv(cdem.Z, Z[0:2, :])
    # Crop bounds: y bottom edge (w/ overshoot)
    cdem = dem.crop(ylim=[2, 4])
    assert np.array_equiv(cdem.ylim, [2, 3])
    assert np.array_equiv(cdem.Z, Z[2:3, :])
    # Crop bounds: x, y interior
    cdem = dem.crop(xlim=[1, 2], ylim=[1, 2])
    assert np.array_equiv(cdem.xlim, [1, 2])
    assert np.array_equiv(cdem.ylim, [1, 2])
    assert np.array_equiv(cdem.Z, Z[1:2, 1:2])
    # Crop bounds: x, y interior (non-edges)
    cdem = dem.crop(xlim=[1.5, 1.9], ylim=[1, 1.9])
    assert np.array_equiv(cdem.xlim, [1, 2])
    assert np.array_equiv(cdem.ylim, [1, 2])
    assert np.array_equiv(cdem.Z, Z[1:2, 1:2])

def test_dem_crop_descending():
    Z = np.arange(1, 10, 1).reshape([3, 3])
    dem = DEM.DEM(Z, [3, 0], [3, 0])
    # Equal bounds
    cdem = dem.crop(xlim=[0, 3], ylim=[0, 3])
    assert np.array_equal(dem.xlim, cdem.xlim)
    assert np.array_equal(dem.Z, cdem.Z)
    # Crop bounds: x, y interior (non-edges)
    cdem = dem.crop(xlim=[1.5, 1.9], ylim=[1, 1.9])
    assert np.array_equiv(cdem.xlim, [2, 1])
    assert np.array_equiv(cdem.ylim, [2, 1])
    assert np.array_equiv(cdem.Z, Z[1:2, 1:2])

def test_dem_resize():
    Z = np.zeros([10, 10])
    dem = DEM.DEM(Z)
    # Downsize
    rdem = dem.resize(0.5)
    assert np.array_equal(rdem.d, dem.d * 2)
    assert np.array_equal(rdem.xlim, dem.xlim)
    # Upsize
    rdem = dem.resize(2)
    assert np.array_equal(rdem.d, dem.d / 2)
    assert np.array_equal(rdem.xlim, dem.xlim)

# ---- RANSAC (Polynomial) ----

def test_ransac_polynomial():
    data = np.array([
        [0, 0],
        [1.1, 1.0],
        [1.9, 2.0],
        [3.1, 3.1],
        [3.0, 0.1],
        [0.1, 3.0],
        [4.1, 4.0]])
    model = optimize.Polynomial(data, deg=1)
    inliers = [0, 1, 2, 3, 6]
    rvalues, rindex = optimize.ransac(model, sample_size=2, max_error=0.5, min_inliers=2, iterations=10)
    assert np.isin(rindex, inliers).all()

# ---- RANSAC (Cameras) ----

# Rotate camera
imgA = image.Image("tests/AK10b_20141013_020336.JPG")
imgA.cam.resize(0.5)
imgB = imgA.copy()
viewdir = [2, 2, 2]
imgB.cam.viewdir = viewdir
imgB.I = imgA.project(imgB.cam)
# Match features
matches = optimize.sift_matches([imgA, imgB], ratio=0.8)

def test_ransac_camera_viewdir(tol=0.1):
    model = optimize.Cameras(cams=imgB.cam, controls=matches, cam_params={'viewdir': True})
    assert np.all(np.abs(model.fit() - viewdir) > tol)
    rvalues, rindex = optimize.ransac(model, sample_size=12, max_error=5, min_inliers=10, iterations=10)
    assert np.all(np.abs(rvalues - viewdir) < tol)

# ---- SVG ----

def test_svg_parse_polyline():
    points = "20,100 40,60 70,80"
    expected = np.array([[20, 100], [40, 60], [70, 80]])
    assert np.array_equal(expected, svg.parse_polyline(points))

def test_svg_parse_polygon():
    points = "20,100 40,60 70,80"
    expected = np.array([[20, 100], [40, 60], [70, 80], [20, 100]])
    assert np.array_equal(expected, svg.parse_polygon(points, closed=True))

def test_svg_parse_line():
    args = {'x1': 20, 'y1': 100, 'x2': 40, 'y2': 60}
    expected = np.array([[20, 100], [40, 60]])
    assert np.array_equal(expected, svg.parse_line(**args))

def test_svg_parse_path():
    d = "M 100 100 L 300 100 L 200 300 z"
    expected = np.array([[100, 100], [300, 100], [200, 300], [100, 100]])
    assert np.array_equal(expected, svg.parse_path(d))
    d = "M100,200 C100,100 250,100 250,200 S400,300 400,200"
    expected = np.array([[100, 200], [250, 200], [400, 200]])
    assert np.array_equal(expected, svg.parse_path(d))
    d = "M600,350 l 50,-25 a25,25 -30 0,1 50,-25"
    expected = np.array([[600, 350], [600 + 50, 350 - 25], [600 + 50 + 50, 350 - 25 - 25]])
    assert np.array_equal(expected, svg.parse_path(d))
