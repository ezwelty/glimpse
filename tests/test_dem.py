from .context import *
from glimpse.imports import (np)
import pytest

def test_dem_defaults():
    Z = np.zeros((3, 3))
    dem = glimpse.DEM(Z)
    assert all(dem.xlim == (0, Z.shape[1]))
    assert all(dem.ylim == (0, Z.shape[0]))
    assert all(dem.zlim == (Z.min(), Z.max()))
    assert all(dem.n == Z.shape)
    assert all(dem.d == (1, 1))
    assert all(dem.min == (0, 0))
    assert all(dem.max == Z.shape[::-1])
    assert all(dem.x == (0.5, 1.5, 2.5))
    assert all(dem.y == (0.5, 1.5, 2.5))
    assert (dem.X == ((dem.x, dem.x, dem.x))).all()
    assert (dem.Y.T == ((dem.y, dem.y, dem.y))).all()

def test_dem_xy():
    xlim = (0, 3)
    ylim = (3, 0)
    x = (0.5, 1.5, 2.5)
    y = (2.5, 1.5, 0.5)
    X = (x, x, x)
    Y = np.asarray((y, y, y)).T
    Z = np.zeros((3, 3))
    # Initialize from xlim, ylim
    dem = glimpse.DEM(Z, x=xlim, y=ylim)
    assert all(dem.xlim == xlim)
    assert all(dem.ylim == ylim)
    assert all(dem.x == x)
    assert all(dem.y == y)
    assert (dem.X == X).all()
    assert (dem.Y == Y).all()
    # Initialize from x, y
    dem = glimpse.DEM(Z, x=x, y=y)
    assert all(dem.xlim == xlim)
    assert all(dem.ylim == ylim)
    # Initialize from X, Y
    dem = glimpse.DEM(Z, x=X, y=Y)
    assert all(dem.xlim == xlim)
    assert all(dem.ylim == ylim)
    assert all(dem.x == x)
    assert all(dem.y == y)

def test_dem_sample(tol=1e-13):
    Z = np.arange(16).reshape(4, 4)
    dem = glimpse.DEM(Z, (-0.5, 3.5), (-0.5, 3.5))
    # Sample cells centers along diagonal
    xy_diagonal = np.column_stack((dem.x, dem.y))
    dz_points = dem.sample(xy_diagonal) - dem.Z.diagonal()
    assert all(dz_points < tol)

def test_dem_crop_ascending():
    Z = np.arange(9).reshape(3, 3)
    dem = glimpse.DEM(Z, (0, 3), (0, 3))
    # Out of bounds
    with pytest.raises(Exception):
        dem.crop(xlim=(3, 5))
    # Equal bounds
    cdem = dem.copy()
    cdem.crop(xlim=(0, 3), ylim=(0, 3))
    assert (dem.Z == cdem.Z).all()
    # Crop bounds: x left edge
    cdem = dem.copy()
    cdem.crop(xlim=(0, 2))
    assert all(cdem.xlim == (0, 2))
    assert (cdem.Z == Z[:, 0:2]).all()
    # Crop bounds: x right edge (w/ overshoot)
    cdem = dem.copy()
    cdem.crop(xlim=(2, 4))
    assert all(cdem.xlim == (2, 3))
    assert (cdem.Z == Z[:, 2:3]).all()
    # Crop bounds: y top edge
    cdem = dem.copy()
    cdem.crop(ylim=(0, 2))
    assert all(cdem.ylim == (0, 2))
    assert (cdem.Z == Z[0:2, :]).all()
    # Crop bounds: y bottom edge (w/ overshoot)
    cdem = dem.copy()
    cdem.crop(ylim=(2, 4))
    assert all(cdem.ylim == (2, 3))
    assert (cdem.Z == Z[2:3, :]).all()
    # Crop bounds: x, y interior
    cdem = dem.copy()
    cdem.crop(xlim=(1, 2), ylim=(1, 2))
    assert all(cdem.xlim == (1, 2))
    assert all(cdem.ylim == (1, 2))
    assert (cdem.Z == Z[1:2, 1:2]).all()
    # Crop bounds: x, y interior (non-edges)
    cdem = dem.copy()
    cdem.crop(xlim=(1.5, 1.9), ylim=(1, 1.9))
    assert all(cdem.xlim == (1, 2))
    assert all(cdem.ylim == (1, 2))
    assert (cdem.Z == Z[1:2, 1:2]).all()

def test_dem_crop_descending():
    Z = np.arange(9).reshape(3, 3)
    dem = glimpse.DEM(Z, (3, 0), (3, 0))
    # Equal bounds
    cdem = dem.copy()
    cdem.crop(xlim=(0, 3), ylim=(0, 3))
    assert all(dem.xlim == cdem.xlim)
    assert (dem.Z == cdem.Z).all()
    # Crop bounds: x, y interior (non-edges)
    cdem = dem.copy()
    cdem.crop(xlim=(1.5, 1.9), ylim=(1, 1.9))
    assert all(cdem.xlim == (2, 1))
    assert all(cdem.ylim == (2, 1))
    assert (cdem.Z == Z[1:2, 1:2]).all()

def test_dem_resize():
    Z = np.zeros((10, 10))
    dem = glimpse.DEM(Z)
    # Downsize
    rdem = dem.copy()
    rdem.resize(0.5)
    assert all(rdem.d == dem.d * 2)
    assert all(rdem.xlim == dem.xlim)
    # Upsize
    rdem = dem.copy()
    rdem.resize(2)
    assert all(rdem.d == dem.d / 2)
    assert all(rdem.xlim == dem.xlim)
