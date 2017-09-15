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

**Attributes**

- [x] `xyz` - Position [x, y, z]
- [x] `viewdir` - View direction in degrees [yaw, pitch, roll]
- [x] `imgsz` - Image size in pixels [nx, ny]
- [x] `f` - Focal length in pixels [fx, fy]
- [x] `c` - Principal point offset in pixels [dx, dy]
- [x] `k` - Radial distortion coefficients [k1, ..., k6]
- [x] `p` - Tangential distortion coefficients [p1, p2]

**Methods**

- [x] `Camera` - Create a new `Camera`.
- [x] `project` - Project world coordinates to image coordinates.
- [x] `invproject` - Project image coordinates to world ray directions.
- [x] `resize` - Resize by a scale factor.
- [x] `idealize` - Set distortion to zero.
- [ ] `optimize` - Adjust parameters to fit surveyed image-world point and line correspondences.

### Exif

`Exif` is a container and parser for image file metadata.

**Attributes**

- [x] `tags` - Image file metadata
- [x] `size` - Image size in pixels [nx, ny]
- [x] `datetime` - Capture date and time
- [x] `fmm` - Focal length in millimeters
- [x] `shutter` - Shutter speed in seconds
- [x] `aperture` - Aperture size as f-number
- [x] `iso` - Film speed
- [x] `make` - Camera make
- [x] `model` - Camera model

**Methods**

- [x] `Exif` - Create a new `Exif`.
- [x] `parse_tag` - Parse an exif tag and return its value.

### Image

While a `Camera` describes the geometry of a camera, an `Image` describes the camera settings and resulting image at a particular moment in time. Each `Image` has an associated `Camera` object.

**Attributes**

- [x] `path` - Path to the image file
- [x] `exif` - `Exif` object
- [x] `datetime` - Capture date and time
- [x] `cam` - `Camera` object
- [ ] `fixed` - Fixed regions to use for alignment [Observer attribute?]

**Methods**

- [x] `Image` - Create a new `Image`.
- [x] `read` - Read image from file.
- [ ] `set_fixed` - Define fixed regions to use for alignment.
- [ ] `project` - Project image into a `Camera`.
- [ ] `plot` - Plot image.

### DEM (Digital Elevation Model)

A `DEM` describes elevations on a regular 2-dimensional grid.

**Attributes**

- [x] `Z` - Grid of values on a regular xy grid
- [x] `xlim` - Outer bounds of the grid in x [left, right]
- [x] `ylim` - Outer bounds of the grid in y [top, bottom]
- [x] `datetime` - Capture date and time

**Methods**

- [x] `DEM` - Create a new `DEM`.
- [x] `crop` - Crop to xy bounds.
- [x] `resize` - Resize by a scale factor.
- [x] `sample` / `sample_grid` - Sample elevation at points [x, y].
- [x] `fill_crevasses_simple` / `fill_crevasses_complex` - Fill crevasses.
- [x] `visible` - Compute visibility from a point [x, y, z].
- [ ] `horizon` - Compute horizon from a point [x, y, z].
- [ ] `plot` - Plot a `DEM`.

### Observer

An `Observer` is a sequence of `Image` objects captured from the same position. It computes perturbations in image coordinates (e.g., from wind and thermal expansion), the cross-correlation statistics between images, and finally the log-likelihood function for each particle.

**Attributes**

- [ ] `images` - List of `Image` objects.
- [ ] `ref_template` - [?]
- [ ] `pca` - [?]
- [ ] `sigma_0` - Pixel accuracy when delcorr/std is 1 [?]
- [ ] `sigma_1` - Pixel accuracy (e.g. std deviation) when correlation is perfect [?]
- [ ] `B` - Shape parameter [?]
- [ ] `outlier_tol_pixels` [peak_tolerance] - Distance in pixels beyond which peak is regarded as an outlier

**Methods**

- [ ] `Observer` - Create a new `Observer`.
- [ ] `align`, [`optimize`?] - Align images to each other based on shared static regions.
- [ ] `compute_likelihood` - Compute the likelihood of particles between two images.

### ParticleTracker [Tracker?]

The `ParticleTracker` implements a constant velocity model particle filter.

**Attributes**

- [ ] `observers` - List of `Observer` objects
- [ ] `dems` - List of `DEM` objects

**Methods**

- [ ] `ParticleTracker` - Create a new `ParticleTracker`.
- [ ] `predict` - Predict future particle locations.
- [ ] `track` - Track particles through time.
