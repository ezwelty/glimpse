"""Tests of the raster module."""
import datetime
import itertools
import os

import numpy as np
import osgeo.osr
import pytest

import glimpse


def test_initializes_default_raster() -> None:
    """Initializes Raster with default coordinate system."""
    Z = np.zeros((4, 3))
    dem = glimpse.Raster(Z)
    assert all(dem.xlim == (0, Z.shape[1]))
    assert all(dem.ylim == (0, Z.shape[0]))
    assert all(dem.zlim == (Z.min(), Z.max()))
    assert all(dem.size == Z.shape[::-1])
    assert all(dem.d == (1, 1))
    assert all(dem.min == (0, 0))
    assert all(dem.max == Z.shape[::-1])
    assert all(dem.x == (0.5, 1.5, 2.5))
    assert all(dem.y == (0.5, 1.5, 2.5, 3.5))
    assert (dem.X == [dem.x] * Z.shape[0]).all()
    assert (dem.Y.T == [dem.y] * Z.shape[1]).all()


def test_initializes_custom_raster() -> None:
    """Initializes Raster with custom coordinate system."""
    xlim = (0, 3)
    ylim = (3, 0)
    x = (0.5, 1.5, 2.5)
    y = (2.5, 1.5, 0.5)
    X = (x, x, x)
    Y = np.asarray((y, y, y)).T
    Z = np.zeros((3, 3))
    # Initialize from xlim, ylim
    dem = glimpse.Raster(Z, x=xlim, y=ylim)
    assert all(dem.xlim == xlim)
    assert all(dem.ylim == ylim)
    assert all(dem.x == x)
    assert all(dem.y == y)
    assert (dem.X == X).all()
    assert (dem.Y == Y).all()
    # Initialize from x, y
    dem = glimpse.Raster(Z, x=x, y=y)
    assert all(dem.xlim == xlim)
    assert all(dem.ylim == ylim)
    # Initialize from X, Y
    dem = glimpse.Raster(Z, x=X, y=Y)
    assert all(dem.xlim == xlim)
    assert all(dem.ylim == ylim)
    assert all(dem.x == x)
    assert all(dem.y == y)


def test_samples_raster(tol: float = 1e-13) -> None:
    """Samples Raster at points."""
    Z = np.arange(16).reshape(4, 4)
    dem = glimpse.Raster(Z, (-0.5, 3.5), (-0.5, 3.5))
    # Sample cells centers along diagonal
    xy_diagonal = np.column_stack((dem.x, dem.y))
    dz_points = dem.sample(xy_diagonal) - dem.array.diagonal()
    assert all(dz_points < tol)


def test_crops_raster_with_ascending_y() -> None:
    """Crops Raster with ascending y coordinates."""
    Z = np.arange(9).reshape(3, 3)
    dem = glimpse.Raster(Z, (0, 3), (0, 3))
    # Out of bounds
    with pytest.raises(Exception):
        dem.crop(xlim=(3, 5))
    # Equal bounds
    cdem = dem.copy()
    cdem.crop(xlim=(0, 3), ylim=(0, 3))
    assert (dem.array == cdem.array).all()
    # Crop bounds: x left edge
    cdem = dem.copy()
    cdem.crop(xlim=(0, 2))
    assert all(cdem.xlim == (0, 2))
    assert (cdem.array == Z[:, 0:2]).all()
    # Crop bounds: x right edge (w/ overshoot)
    cdem = dem.copy()
    cdem.crop(xlim=(2, 4))
    assert all(cdem.xlim == (2, 3))
    assert (cdem.array == Z[:, 2:3]).all()
    # Crop bounds: y top edge
    cdem = dem.copy()
    cdem.crop(ylim=(0, 2))
    assert all(cdem.ylim == (0, 2))
    assert (cdem.array == Z[0:2, :]).all()
    # Crop bounds: y bottom edge (w/ overshoot)
    cdem = dem.copy()
    cdem.crop(ylim=(2, 4))
    assert all(cdem.ylim == (2, 3))
    assert (cdem.array == Z[2:3, :]).all()
    # Crop bounds: x, y interior
    cdem = dem.copy()
    cdem.crop(xlim=(1, 2), ylim=(1, 2))
    assert all(cdem.xlim == (1, 2))
    assert all(cdem.ylim == (1, 2))
    assert (cdem.array == Z[1:2, 1:2]).all()
    # Crop bounds: x, y interior (non-edges)
    cdem = dem.copy()
    cdem.crop(xlim=(1.5, 1.9), ylim=(1, 1.9))
    assert all(cdem.xlim == (1, 2))
    assert all(cdem.ylim == (1, 2))
    assert (cdem.array == Z[1:2, 1:2]).all()


def test_crops_raster_with_descending_y() -> None:
    """Crops Raster with descending y coordinates."""
    Z = np.arange(9).reshape(3, 3)
    dem = glimpse.Raster(Z, (3, 0), (3, 0))
    # Equal bounds
    cdem = dem.copy()
    cdem.crop(xlim=(0, 3), ylim=(0, 3))
    assert all(dem.xlim == cdem.xlim)
    assert (dem.array == cdem.array).all()
    # Crop bounds: x, y interior (non-edges)
    cdem = dem.copy()
    cdem.crop(xlim=(1.5, 1.9), ylim=(1, 1.9))
    assert all(cdem.xlim == (2, 1))
    assert all(cdem.ylim == (2, 1))
    assert (cdem.array == Z[1:2, 1:2]).all()


def test_resizes_raster() -> None:
    """Resizes Raster smaller and larger."""
    Z = np.zeros((10, 10))
    dem = glimpse.Raster(Z)
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


def test_writes_and_reads_raster() -> None:
    """Writes Raster to file and reads it back from file."""
    old = glimpse.Raster(
        np.array([(0, 0, 0), (0, np.nan, 0), (1, 1, 1)], dtype=float),
        x=np.array((1, 2, 3), dtype=float),
        y=np.array((3, 2, 1), dtype=float),
        crs="+init=epsg:4326",
    )
    # Write to file and read
    tempfile = "temp.tif"
    old.write(tempfile)
    new = glimpse.Raster.read(tempfile)
    np.testing.assert_equal(old.array, new.array)
    np.testing.assert_equal(old.x, new.x)
    np.testing.assert_equal(old.y, new.y)
    old_crs = osgeo.osr.SpatialReference()
    old_crs.ImportFromProj4(old.crs)
    new_crs = osgeo.osr.SpatialReference()
    new_crs.ImportFromWkt(new.crs)
    assert old_crs.IsSame(new_crs)
    # Delete file
    os.remove(tempfile)


def test_interpolates_rasters() -> None:
    """Interpolates Rasters."""
    # Read rasters
    mean_paths = [
        os.path.join("tests", "000nan.tif"),
        os.path.join("tests", "11-1nan.tif"),
    ]
    means = [glimpse.Raster.read(path) for path in mean_paths]
    Zs = [mean.array for mean in means]
    sigma_paths = mean_paths
    sigmas = means
    # Define tests
    # x
    xs = [
        (0, 1),
        (datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 3)),
        (0.0, 1.0),
    ]
    # means, sigmas
    means_sigmas = [
        (means, sigmas),
        (means, None),
        (means, [0] * len(means)),
        (mean_paths, sigma_paths),
        (mean_paths, None),
        (mean_paths, [0] * len(means)),
    ]
    # scale, extrapolate
    samples = [(0.5, False), (1.5, True)]
    tests = tuple(itertools.product(xs, means_sigmas, samples))
    # Run tests
    for test in tests:
        x = test[0]
        means, sigmas = test[1]
        scale, extrapolate = test[2]
        interpolant = glimpse.RasterInterpolant(means=means, sigmas=sigmas, x=x)
        xi = x[0] + (x[1] - x[0]) * scale
        imean, isigma = interpolant(xi, extrapolate=extrapolate, return_sigma=True)
        mean = Zs[0] + (Zs[1] - Zs[0]) * scale
        np.testing.assert_equal(imean.array, mean)
        if isinstance(xi, datetime.datetime):
            # Test whether Raster.datetime set when appropriate
            assert imean.datetime == xi
            assert isigma.datetime == xi
