# glimpse: Glacier Image Particle Sequencer

## Development

### Documentation

Docstrings should follow the [Khan Academy style guide](https://github.com/Khan/style-guides/blob/master/style/python.md#docstrings), ending with the following special sections:

- `Arguments:` (or `Attributes:` for classes) List each parameter in the format "name (type): description". The description can span several lines using a hanging indent.
- `Returns:` (or `Yields:` for generators) Describe the return value in the format "type: description" or skip if returns `None`.

Examples should be formatted for testing by [doctest](https://docs.pytest.org).

```python
"""
Project world coordinates to image coordinates.

Arguments:
    xyz (array): World coordinates (Nx3) or camera coordinates (Nx2)
    directions (bool): Whether absolute coordinates (False) or ray directions (True)

Returns:
    array: Image coordinates (Nx2)

>>> xyz = np.array([[0., 1., 0.]])
>>> cam = Camera(xyz = [0, 0, 0], viewdir = [0, 0, 0])
>>> cam.project(xyz)
array([[ 50.,  50.]])
"""
```

### Testing

Tests are written using [pytest](https://docs.pytest.org), and currently live in `tests.py`. To run tests:

```bash
pip install pytest
pytest tests.py
```

To run doctests, use:

```bash
python -m doctest Camera.py # one module
pytest --doctest-modules # all modules
```

## Classes

An overview of the core attributes and methods of the object classes.

### Camera

Once calibrated and oriented in 3-dimensional (3D) space, a `Camera` can transform 3D world coordinates to their corresponding 2D image coordinates. The reverse transformation results in a 3D ray, and distance from the camera (e.g., from a digital elevation model or a 3D ray pointing to the same point from another camera) is needed to recover the 3D coordinates of the original point.

**Properties**

- `xyz` - Position [x, y, z]
- `viewdir` - View direction in degrees [yaw, pitch, roll]
- `imgsz` - Image size in pixels [nx, ny]
- `f` - Focal length in pixels [fx, fy]
- `c` - Principal point offset in pixels [dx, dy]
- `k` - Radial distortion coefficients [k1, ..., k6]
- `p` - Tangential distortion coefficients [p1, p2]

**Methods**

- `Camera` - Create a new `Camera`.
- `optimize` - Adjust `Camera` properties to fit surveyed image-world point pairs.
- `project` - Project world coordinates to image coordinates.
- `invproject` - Project image coordinates to world ray directions.

### Image

While a `Camera` describes the geometry of a camera, an `Image` describes the camera settings and resulting image at a particular moment in time. Each `Image` has an associated `Camera` object.

**Properties**

- `Image` - Create a new `Image`.
- `file` - Path to the image file
- `datetime` - Capture date and time
- `size` - Size of the original image in pixels [nx, ny]
- `cam` - `Camera` object

**Methods**

- `read` - Read image from file.
- `plot` - Plot image.
- `project` - Project image into a `Camera`.

### DEM (Digital Elevation Model)

A `DEM` describes elevations on a regular 2D grid.

**Properties**

- `Z` - Grid of values on a regular xy grid
- `xlim` - Outer bounds of the grid in x [left, right]
- `ylim` - Outer bounds of the grid in y [top, bottom]
- `datetime` - Capture date and time

**Methods**

- `DEM` - Create a new `DEM`.
- `crop` - Crop a `DEM`.
- `resize` - Resize a `DEM`.
- `plot` - Plot a `DEM`.
- `sample` - Sample a `DEM` at a point [x, y].
- `horizon` - Compute the horizon from a point [x, y, z].

### Observer

An `Observer` is a sequence of `Image` objects captured from the same position. It computes perturbations in image coordinates (e.g., from wind and thermal expansion), the cross-correlation statistics between images, and finally the log-likelihood function for each particle.

**Properties**

- `images` - Array of `Image` objects.
- `ref_template` - [?]
- `pca` - [?]
- `sigma_0` - Pixel accuracy when delcorr/std is 1 [?]
- `sigma_1` - Pixel accuracy (e.g. std deviation) when correlation is perfect [?]
- `B` - Shape parameter [?]
- `outlier_tol_pixels` [peak_tolerance] - Distance in pixels beyond which peak is regarded as an outlier

**Methods**

- `Observer` - Create a new `Observer`.
- `align`, [`optimize`] - Adjust `Image.Camera` properties to align images to each other based on shared static regions.
- `compute_likelihood` - Compute the likelihood of particles between two images.
- ...

### ParticleTracker [Tracker]

The `ParticleTracker` implements a constant velocity model particle filter.

**Properties**

- `observers` - Array of `Observer` objects
- `dems` - Array of `DEM` objects

**Methods**

- `ParticleTracker` - Create a new `ParticleTracker`.
- `predict` - Predict future particle locations.
- `track` - Track particles through time.
