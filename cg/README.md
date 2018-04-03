# Columbia Glacier

Module [`cg.py`](cg.py) provides functions to access Columbia Glacier timelapse images, metadata, and spatial data needed for camera calibration and feature tracking.

### Calibration inputs

- `geojson/`
  - `horizons/<station>.geojson` - Precomputed horizons for each station (used as line control)
  - `moraines/<yyyymmdd>.geojson` - Distinctive moraines traced from map data (used as line control for same-day images)
  - `coast.geojson` - Fjord coastline (used as line control)
  - `gcp.geojson` - Named ground control points (used as point control)
  - `stations.geojson` - List of stations with nominal camera positions and orientations
  - `termini.geojson` - Glacier terminus traced from map data (used as line control for same-day images)
- `svg/<image>.svg` - Image coordinates of control and polygon regions parsed from SVG layer groups:
  - `gcp`: Ground control points, as one or more `<path>` (single vertex) or `<circle>` whose `id` matches a feature `id` in `gcp.geojson`
  - `horizon`: Mountain horizon segments, as one or more `<path>`, `<line>`, or `<polyline>`
  - `terminus`: Glacier terminus segments (traced at the water line), as one or more `<path>`, `<line>`, or `<polyline>`
  - `coast`: Coastline segments (traced at the water line), as one or more `<path>`, `<line>`, or `<polyline>`
  - `moraines`: Glacier medial moraines, as one or more `<path>`, `<line>`, or `<polyline>` whose `id` matches a feature `id` in `moraines/<yyyymmdd>.geojson`
  - `land`: Static land areas, as one or more `<polygon>`
- `motion.json` - Lists of images separated by large motion (used for matches control)
- `sequences.csv` - List of image sequences with station, camera, image time range, and other metadata

### Calibration outputs

- `cameras/<camera>.json` - Internal camera parameters ('fmm', 'cmm', 'k', 'p', 'sensorsz')
- `stations/<station>.json` - External camera parameters ('xyz', average 'viewdir')
- `images/<image>.json` - Complete camera solution ('xyz', 'viewdir', 'fmm', 'cmm', 'k', 'p', 'sensorsz')
- `viewdirs/<image>.json` - Refined camera view directions ('viewdir')

`*_stderr.json` are standard errors for the optimized parameters, as computed by [lmfit](https://lmfit.github.io/lmfit-py/parameters.html?highlight=stderr#lmfit.parameter.Parameter.stderr).

### Image archive (`cg.IMAGE_PATH`)

Functions in `cg` expect timelapse images to be filed as `IMAGE_PATH/<station>/<station_service>/*/<image>.JPG`, with images at most two levels below `<station_service>`. More specifically, the archive is organized as follows:

- `<station>/` - Each station has a top-level directory
  - `<station_service>/` - which contains one or more service directories
    - `<image>.JPG` - which contain one or more time-lapse images.
    - ...
    A service directory may also contain:
    - `other/` - Additional time-lapse images not considered part of the sequence (e.g. test images taken during servicing).
    - `clock/` - Images taken for camera clock calibration.
    - `clock_reset/` - Images taken for camera clock calibration following a reset of the clock.
    - `calib/` - Images taken for camera lens calibration.
    - `pano/` - Images taken as part of a panorama.
  - `<dir>/` - Non-service subdirectories contain images not taken with the time-lapse camera (e.g. to document the station installation) and thus kept separate.
- `<station>/`
  - ...

All images taken with a timelapse camera (`<image>.JPG`) are named `<station>_<yyyymmdd>_<hhmmss>[A-Z].JPG`.

### DEM archive (`cg.DEM_PATHS`)

Digital elevation models (DEMs) are stored as GeoTIFF files in the directories listed in `cg.DEM_PATHS` with names `*<yyyymmdd>*.tif`.

### Keypoints file cache (`cg.KEYPOINT_PATH`)

Keypoints - restricted to the static regions of the scene - are cached as tuples (list of `cv2.KeyPoint` objects, `numpy.ndarry` of descriptors) and saved as a binary pickle in `cg.KEYPOINT_PATH` with names `<image>.pkl`.

### Matches file cache (`cg.MATCH_PATH`)

Keypoint matches are cached as `glimpse.optimize.Matches` and saved as a binary pickle in `cg.MATCH_PATH` with names `<imageA>-<imageB>.pkl`.
