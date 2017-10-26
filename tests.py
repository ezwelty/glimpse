import datetime
import pytest
import numpy as np
import scipy.misc
import image
import dem as DEM
import ransac
import matplotlib.pyplot as plt
import cv2

# ---- Camera ----

reload(image)

def test_camera_init_with_fmm(fmm=[10, 10], sensorsz=[30, 20]):
    cam = image.Camera(fmm=fmm, sensorsz=sensorsz)
    assert np.array_equal(cam.f, fmm * cam.imgsz / sensorsz)

def test_camera_resize(imgsz=[100, 100]):
    cam = image.Camera(imgsz=imgsz)
    cam.resize(0.5)
    assert np.array_equiv(cam.imgsz * 2, imgsz)
    cam = cam.resize(2)
    assert np.array_equiv(cam.imgsz, imgsz)

def test_camera_idealize(k=1, c=1, p=1):
    cam = image.Camera(k=k, c=c, p=p)
    cam.idealize()
    assert cam.k[0] == 0
    cam = image.Camera(k=k, c=c, p=p).idealize()
    assert cam.k[0] == 0

def test_camera_reprojection_ideal(tol=1e-13):
    cam = image.Camera(xyz=[1, 2, -3], viewdir=[10, 20, -30])
    uv = np.random.rand(1000, 2) * cam.imgsz
    dxyz = cam.invproject(uv)
    uv2 = cam.project(dxyz, directions=True)
    assert np.abs(uv - uv2).max() < tol
    
def test_camera_reprojection_distorted(tol=0.2):
    cam = image.Camera(xyz=[1, 2, -3], viewdir=[10, 20, -30], k = [0.1, -0.1] * 3, p = [0.01, -0.01])
    uv = np.random.rand(1000, 2) * cam.imgsz
    dxyz = cam.invproject(uv)
    uv2 = cam.project(dxyz, directions=True)
    assert np.abs(uv - uv2).max() < tol

# ---- Exif ----

reload(image)

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

def text_exif_subecond():
    path = "tests/AK10b_20141013_020336.JPG"
    exif = image.Exif(path)
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36, 28)
    exif.tags['EXIF SubSecTimeOriginal'] = None
    assert exif.datetime == datetime.datetime(2014, 10, 13, 2, 3, 36)

# ---- Image ----

reload(image)

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

reload(DEM)

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

def test_dem_sample(tol = 1e-13):
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

# ---- RANSAC (test models) ----

def test_ransac_polynomial():
    model = ransac.Polynomial(1)
    data = np.array([
        [0, 0],
        [1.1, 1.0],
        [1.9, 2.0],
        [3.1, 3.1],
        [3.0, 0.1],
        [0.1, 3.0],
        [4.1, 4.0]])
    inliers = [0, 1, 2, 3, 6]
    rparams, idx = ransac.ransac(data, model, sample_size=2, max_error=0.5, min_inliers=2, iterations=10)
    assert np.isin(idx, inliers).all()
    # # Plot
    # plt.figure()
    # plt.scatter(data[:, 0], data[:, 1], c='grey')
    # params = model.fit(data)
    # plt.plot(data[:, 0], model.predict(params, data), c='grey')
    # rparams, idx = ransac.ransac(data, model, sample_size=2, max_error=0.5, min_inliers=2, iterations=10)
    # plt.scatter(data[idx, 0], data[idx, 1], c='red')
    # plt.plot(data[:, 0], model.predict(rparams, data), c='red')

# ---- RANSAC (camera models) ----

# Rotate camera
img = image.Image("tests/AK10b_20141013_020336.JPG")
cam = img.cam.copy()
cam.viewdir = [2, 2, 2]
# Project image
Ia = img.read()
Ib = img.project(cam)
# Match features
sift = cv2.SIFT()
Ka, Da = sift.detectAndCompute(Ia, None)
Kb, Db = sift.detectAndCompute(Ib, None)
index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(Da, Db, k=2)
A = np.array([Ka[m.queryIdx].pt for m,n in matches])
B = np.array([Kb[m.trainIdx].pt for m,n in matches])

def test_ransac_camera_viewdir(tol=0.001):
    # Optimize viewdir with ransac.ransac
    data = np.column_stack((B, img.cam.invproject(A)))
    model = ransac.Camera(cam=img.cam.copy(), directions=True, params={'viewdir': True})
    rparams, idx = ransac.ransac(data, model, sample_size=8, max_error=2, min_inliers=10, iterations=1e3)
    assert np.all((rparams - cam.viewdir) < abs(tol))
    err = model.errors(rparams, data)[idx]
    rmse = (np.sum(err ** 2) / len(err)) ** 0.5
    assert rmse < 0.5
    # # Plot
    # plt.figure(figsize=(12,8))
    # plt.imshow(Ia)
    # plt.imshow(Ib, alpha=0.5)
    # plt.quiver(A[:, 0], A[:, 1], B[:, 0] - A[:, 0], B[:, 1] - A[:, 1], color = 'green', scale=1, scale_units='xy', angles='xy')
    # plt.quiver(A[idx, 0], A[idx, 1], B[idx, 0] - A[idx, 0], B[idx, 1] - A[idx, 1], color = 'red', scale=1, scale_units='xy', angles='xy')

def test_camera_optimize_viewdir(tol=0.001):
    # Optimize viewdir with Camera.optimize
    uv = B
    dxyz = img.cam.invproject(A)
    ransac_cam = img.cam.optimize(uv, dxyz, directions=True, params={'viewdir': True}, use_ransac=True, iterations=1e3)
    assert np.all((ransac_cam.viewdir - cam.viewdir) < abs(tol))