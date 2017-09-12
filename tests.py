import pytest
import numpy as np

# ---- Camera ----

import Camera
reload(Camera)

def test_reprojection_ideal():
    cam = Camera.Camera(xyz=[1, 2, -3], viewdir=[10, 20, -30])
    uv = np.random.rand(1000, 2) * cam.imgsz
    dxyz = cam.invproject(uv)
    uv2 = cam.project(dxyz, directions=True)
    assert np.abs(uv - uv2).max() < 1e-13
    
def test_reprojection_distorted():
    cam = Camera.Camera(xyz=[1, 2, -3], viewdir=[10, 20, -30], k = [0.1, -0.1] * 3, p = [0.01, -0.01])
    uv = np.random.rand(1000, 2) * cam.imgsz
    dxyz = cam.invproject(uv)
    uv2 = cam.project(dxyz, directions=True)
    assert np.abs(uv - uv2).max() < 0.2

# ---- DEM ----

import DEM
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
