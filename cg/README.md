# Columbia Glacier: Camera Calibrations

A variety of map data, image annotations, and metadata are wired together into a series of least-squares optimizations to solve for camera, then station, then image-level camera calibrations.

Module [`cgcalib.py`](cgcalib.py) provides a convenient interface to the time-lapse file and metadata structures.

### Inputs

- `geojson/`
  - `horizons/<station>.geojson` - Precomputed horizons for each station (used as line control)
  - `moraines/<yyyymmdd>.geojson` - Distinctive moraines traced from map data (used as line control for same-day images)
  - `coast.geojson` - Fjord coastline (used as line control)
  - `gcp.geojson` - Named ground control points (used as point control)
  - `stations.geojson` - List of stations with nominal camera positions and orientations
  - `termini.geojson` - Glacier terminus traced from map data (used as line control for same-day images)
- `svg/<image>.svg` - Image coordinates of line and point control (any of 'gcp', 'coast', 'terminus', 'horizon', 'moraines')
- `motion.json` - Lists of images separated by large motion (used for matches control)
- `sequences.csv` - List of image sequences with station, camera, image time range, and other metadata

### Outputs

- `cameras/<camera>.json` - Internal camera parameters ('fmm', 'cmm', 'k', 'p', 'sensorsz')
- `stations/<station>.json` - External camera parameters ('xyz', average 'viewdir')
- `images/`
  - `<image>.json` - Complete camera solution ('xyz', 'viewdir', 'fmm', 'cmm', 'k', 'p', 'sensorsz')
  - `<image>-markup.jpg` - Image with control markup and errors overlaid
  - `<image>-distorted.jpg` - Same-day orthophoto projected into the camera
  - `<image>-oriented.jpg` - Same-day orthophoto projected into the same camera if oriented but not calibrated
  - `<image>-original.jpg` - Original image (for comparison to above)

`*_stderr.json` are standard errors for the optimized parameters, as computed by [lmfit](https://lmfit.github.io/lmfit-py/parameters.html?highlight=stderr#lmfit.parameter.Parameter.stderr).
