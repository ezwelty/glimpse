`glimpse` Glacier Image Particle Sequencer
==========================================

[![tests](https://github.com/ezwelty/glimpse/workflows/tests/badge.svg)](https://github.com/ezwelty/glimpse/actions?workflow=tests)
[![coverage](https://codecov.io/gh/ezwelty/glimpse/branch/master/graph/badge.svg)](https://codecov.io/gh/ezwelty/glimpse)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`glimpse` is a Python package built for quickly and precisely analyzing time-lapse photographs of glaciers.
Some useful things one can do with `glimpse`:

#### Camera calibration

  - `glimpse.Camera`: Model a distorted camera and perform incoming and outgoing ray projections, accounting for earth curvature and atmospheric refraction.
  - `glimpse.convert`: Convert between MATLAB, OpenCV, Agisoft PhotoScan, PhotoModeler, and glimpse camera formats, accounting for uncertainties in camera parameters.
  - `glimpse.optimize`: Perform single and multi-camera calibrations (controlling which parameters are fixed, varied, or synched across multiple cameras) using points and lines in image and world coordinates.
  - `glimpse.svg`: Load manual vector image annotations as inputs for camera calibration.
  - `glimpse.Raster.horizon()`: Compute the visible horizon from a position as input for camera calibration.
  - `glimpse.Camera.project_dem()`: Generate photorealistic synthetic images (and depth maps) from a camera model, gridded elevations, and optional orthophoto for automated control point collection and camera model validation.
  - `glimpse.optimize.KeypointMatcher`: Stabilize photographic sequences of arbitrary length using automatic image matching and globally optimal orientation estimates.

#### Velocity tracking

  - `glimpse.Tracker`: Compute the velocity (and corresponding uncertainty) of points visible in photographs taken from one or more positions.
  - `glimpse.Raster.viewshed()`: Compute the viewshed of a position, for tracking point selection.

### References

The methods implemented in this software are described in great detail across two PhD dissertations.

- Douglas Brinkerhoff (2017): *[Bayesian methods in glaciology](http://hdl.handle.net/11122/8113)*, chapter 4. Uses particle filtering to extract velocities from one year of oblique time-lapse images of Columbia Glacier, Alaska.

- Ethan Welty (2018): *[High-precision photogrammetry for glaciology](https://doi.org/10.13140/RG.2.2.20751.64164)*. Calibrates and stabilizes time-lapse cameras using landscape cues, extends the particle filter from 2-d to 3-d (to account for uncertain surface elevations), and uses the methods on thirteen years of oblique time-lapse images of Columbia Glacier, Alaska.

# Installation

`glimpse` has not yet been released for distribution, but can still be installed with [`pip`](https://pip.pypa.io/en/stable/installing) from the source code on GitHub:

```bash
pip install https://github.com/ezwelty/glimpse/tarball/master#egg=glimpse
```

The installation requires [`gdal`](https://gdal.org/download.html#binaries) to be present. The simplest way of achieving this is to install `gdal` into a [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install) environment:

```bash
conda create -n glimpse -c conda-forge python=3.8 gdal=3.2.0
conda activate glimpse
pip install https://github.com/ezwelty/glimpse/tarball/master#egg=glimpse
```

# Contribute

Thank you for considering contributing to `glimpse`!

What follows is intended to make contribution more accessible by codifying conventions.
Don’t be afraid to open unfinished pull requests or to ask questions if something is unclear!

- No contribution is too small.
- Try to limit each pull request to one change only (one bug fix, one new feature).
- Always add tests and docstrings for your code.
- Make sure your changes pass the continuous integration tests.

## Code & docstrings

Code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/).
It is formatted (automatically, if you use the pre-commit hooks)
to conform to the [`black`](https://github.com/psf/black) code style and import order
with a maximum line length of 88 characters.

Docstrings follow [PEP 257](https://www.python.org/dev/peps/pep-0257/) and the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), with the exception of using section title `"Arguments:"` instead of `"Args:"`.
Argument and return types are specified as type annotations and not included in the docstrings.
Examples are formatted for testing by [`doctest`](https://docs.pytest.org).
Unless the docstring fits in a single line, the `"""` are on separate lines.

```python
def xyz_to_uv(xyz: np.ndarray) -> np.ndarray:
    """
    Convert world coordinates to image coordinates.

    Arguments:
        xyz: World coordinates (n, [x, y, z]).

    Returns:
        Image coordinates (n, 2).

    Examples:
        >>> xyz = np.array([[0., 1., 0.]])
        >>> xyz_to_uv(xyz)
        array([[ 50.,  50.]])
    """
```

## Tests

Unit tests are written for [pytest](https://docs.pytest.org/en/latest/getting-started.html).
As with the rest of the code, they should include type annotations and [good docstrings](https://jml.io/test-docstrings).

To run the full test suite on the current Python version, simply run:

```bash
make test
```

if you have `make` installed, or run the equivalent:

```bash
poetry run pytest --doctest-modules src tests
```

The package manager [`poetry`](https://python-poetry.org) (see below) will manage the virtual environment and dependencies for you.
To easily install and switch between different Python versions,
consider using [`pyenv`](https://github.com/pyenv/pyenv).

## Development environment

Before you begin, you will need to have [`gdal`](https://gdal.org/download.html#binaries),
a modern stable release of Python 3,
and the package manager [`poetry`](https://python-poetry.org).
For example, using [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install):

```bash
conda activate base
conda install -c conda-forge python=3.8 gdal=3.2.0 poetry=1
```

First, clone the `glimpse` repository and change into the newly created directory:

```bash
git clone https://github.com/ezwelty/glimpse
cd glimpse
```

Use `poetry` to install `glimpse` (and its dependencies) into a new virtual environment
linked to your current Python version:

```bash
poetry install
```

Run commands on this virtual environment using `poetry run`.
For example, run unit tests with:

```
poetry run pytest
```

or open a Python console with:

```bash
poetry run python
```

To avoid committing code that breaks tests or violates the style guide,
consider installing [`pre-commit`](https://pre-commit.com) (if needed)
and installing the hooks:

```bash
pre-commit install
```

You can run the pre-commit hooks anytime using:

```bash
pre-commit run --all-files
```

Other useful commands are listed in the [`Makefile`](Makefile).
For example, to build the documentation:

```bash
make docs
# Equivalent to:
# poetry run sphinx-build docs docs/build
```
