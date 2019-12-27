[![Build Status](https://travis-ci.com/ezwelty/glimpse.svg?branch=master)](https://travis-ci.com/ezwelty/glimpse)
[![codecov](https://codecov.io/gh/ezwelty/glimpse/branch/master/graph/badge.svg)](https://codecov.io/gh/ezwelty/glimpse)

# `glimpse` Glacier Image Particle Sequencer

`glimpse` is a Python package built for quickly and precisely analyzing time-lapse photographs of glaciers. Some useful things one can do with glimpse:

#### Camera calibration

  - `glimpse.Camera`: Model a distorted camera and perform incoming and outgoing ray projections, accounting for earth curvature, atmospheric refraction, and uncertainties in camera parameters.
  - `glimpse.convert`: Convert between Matlab, Agisoft PhotoScan, PhotoModeler, and glimpse camera formats, accounting for uncertainties in camera parameters.
  - `glimpse.optimize`: Perform single and multi-camera calibrations (controlling which parameters are fixed, varied, or synched across multiple cameras) using points and lines in image and world coordinates.
  - `glimpse.svg` and `glimpse.geojson`: Load manual vector image annotations (SVG) and spatial data (GeoJSON) as inputs for camera calibration.
  - `glimpse.Raster.horizon()`: Compute the visible horizon from a position as input for camera calibration.
  - `glimpse.Camera.project_dem()`: Generate photorealistic synthetic images (and depth maps) from a camera model, gridded elevations, and optional orthophoto for automated control point collection and camera model validation.
  - `glimpse.optimize.KeypointMatcher`: Stabilize photographic sequences of arbitrary length using automatic image matching and globally optimal orientation estimates.

#### Velocity tracking

  - `glimpse.Tracker`: Compute the velocity (and corresponding uncertainty) of points visible in photographs taken from one or more positions.
  - `glimpse.Raster.viewshed()`: Compute the viewshed of a position, for tracking point selection.


## Installation

`glimpse` has not yet been released for distribution, but can still be installed with `pip` from the source code on GitHub:

```bash
pip install https://github.com/ezwelty/glimpse/tarball/master#egg=glimpse[io]
```

To install without `gdal` support (used for reading and writing raster data), use:

```bash
pip install https://github.com/ezwelty/glimpse/tarball/master#egg=glimpse
```

## Documentation

### Compilation

Documentation files are compiled using [Sphinx](http://www.sphinx-doc.org/en/master/usage/installation.html) using the [Read the Docs theme](https://sphinx-rtd-theme.readthedocs.io/en/latest/). Once these are installed, simply run:

```bash
cd docs
make html
```

Open `docs/build/html/index.html` with a browser to check the output.

### Docstring format

Docstrings should follow the [Khan Academy style guide](https://github.com/Khan/style-guides/blob/master/style/python.md#docstrings), ending with the following special sections:

- `Arguments:` (or `Attributes:` for classes) List each parameter in the format "name (class): description". The description can span several lines using a hanging indent.
- `Returns:` (or `Yields:` for generators) Describe the return value(s) in the format "class: description" or skip if returns `None`.

Examples should be formatted for testing by [doctest](https://docs.pytest.org).

```python
"""
Project world coordinates to image coordinates.

Arguments:
    xyz (numpy.ndarray): World coordinates (n, 3) or camera coordinates (n, 2)
    directions (bool): Whether absolute coordinates (`False`) or ray directions
      (`True`)

Returns:
    numpy.ndarray: Image coordinates (n, 2)

>>> xyz = np.array([[0., 1., 0.]])
>>> cam = Camera(xyz = [0, 0, 0], viewdir = [0, 0, 0])
>>> cam.project(xyz)
array([[ 50.,  50.]])
"""
```

## Tests

### Unit tests

Unit tests are written for [pytest](https://docs.pytest.org/en/latest/getting-started.html) and live in `tests/test_*.py`. Once installed, run all unit tests with:

```bash
pytest tests
```

### Doctests

To run all [doctests](https://docs.python.org/3/library/doctest.html), use:

```bash
python run_doctests.py
```
