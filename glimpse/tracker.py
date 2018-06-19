from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, cv2, warnings, datetime, scipy, matplotlib, sys, traceback)
from . import (helpers, raster, config)

class Tracker(object):
    """
    A `Tracker' estimates the trajectory of world points through time.

    Attributes:
        observers (list): Observer objects
        time_unit (timedelta): Length of time unit for temporal arguments
        dem: Elevation of the surface on which to track points, as either a scalar or
             a Raster
        dem_sigma: Elevation standard deviations, as either a scalar or
             a Raster with the same extent as `dem`. `0` means particles stay glued to
             `dem` and weighing particles by their distance from the `dem` is disabled.
        viewshed (Raster): Binary viewshed with the same extent as `dem`
        resample_method (str): Particle resampling method
            ('systematic', 'stratified', 'residual', 'choice': np.random.choice with replacement)
        grayscale (dict): Grayscale conversion
            (arguments to glimpse.helpers.rgb_to_gray)
        highpass (dict): Median high-pass filter
            (arguments to scipy.ndimage.filters.median_filter)
        interpolation (dict): Subpixel interpolation
            (arguments to scipy.interpolate.RectBivariateSpline)
        particles (array): Positions and velocities of particles (n, 6) [[x, y, z, vx, vy vz], ...]
        weights (array): Particle likelihoods (n, )
        particle_mean (array): Weighted mean of `particles` (6, ) [x, y, z, vx, vy, vz]
        particle_covariance (array): Weighted covariance matrix of `particles` (6, 6)
        templates (list): For each Observer, a template extracted from the first image
            matching a `datetimes` centered around the `particle_mean` at that time.
            Templates are dictionaries that include at least

            - 'tile': Image tile used as a template for image cross-correlation
            - 'histogram': Histogram (values, quantiles) of the 'tile' used for histogram matching
            - 'duv': Subpixel offset of 'tile' (desired - sampled)
    """
    def __init__(self, observers, time_unit, dem, dem_sigma=0, viewshed=None, resample_method='systematic',
        grayscale=dict(method='average'), highpass=dict(size=(5, 5)), interpolation=dict(kx=3, ky=3)):
        self.observers = observers
        self.dem = dem
        self.dem_sigma = dem_sigma
        self.viewshed = viewshed
        self.time_unit = time_unit
        self.resample_method = resample_method
        self.grayscale = grayscale
        self.highpass = highpass
        self.interpolation = interpolation
        # Placeholders
        self.particles = None
        self.weights = None
        self.templates = None

    @property
    def particle_mean(self):
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def particle_covariance(self):
        return np.cov(self.particles.T, aweights=self.weights)

    def _test_particles(self):
        """
        Particle test run after each particle initialization or evolution.
        """
        if self.viewshed is not None:
            is_visible = self.viewshed.sample(self.particles[:, 0:2], order=0)
            if not all(is_visible):
                raise ValueError('Some particles are on non-visible viewshed cells')
        if any(np.isnan(self.particles[:, 2])):
            raise ValueError('Some particles are on NaN dem cells')

    def _sample_dem(self, xy, sigma=False):
        obj = self.dem_sigma if sigma else self.dem
        if isinstance(obj, raster.Raster):
            return obj.sample(xy)
        else:
            return np.full(len(self.particles), obj)

    def initialize_particles(self, xy, n=1000, xy_sigma=(0, 0), vxyz=(0, 0, 0), vxyz_sigma=(0, 0, 0)):
        """
        Initialize particles given an initial normal distribution.

        Temporal arguments (`vxyz`, `vxyz_sigma`) are assumed to be in
        `self.time_unit` time units.

        Arguments:
            xy (iterable): Mean position (x, y)
            n (int): Number of particles
            xy_sigma (iterable): Standard deviation of position (x, y)
            vxyz (iterable): Mean velocity (x, y, z)
            vxyz_sigma (iterable): Standard deviation of velocity (x, y, z)
        """
        if self.particles is None or len(self.particles) != n:
            self.particles = np.zeros((n, 6), dtype=float)
        self.particles[:, 0:2] = xy + xy_sigma * np.random.randn(n, 2)
        z = self._sample_dem(self.particles[:, 0:2])
        z_sigma = self._sample_dem(self.particles[:, 0:2], sigma=True)
        self.particles[:, 2] = z + z_sigma * np.random.randn(n)
        self.particles[:, 3:6] = vxyz + vxyz_sigma * np.random.randn(n, 3)
        self._test_particles()
        self.weights = np.full(n, 1 / n)

    def evolve_particles(self, dt, axyz=(0, 0, 0), axyz_sigma=(0, 0, 0)):
        """
        Evolve particles through time by stochastic differentiation.

        Accelerations (`axyz`, `axyz_sigma`) are assumed to be with respect to
        `self.time_unit`.

        Arguments:
            dt (timedelta): Time difference to evolve particles forward or backward
            axyz (iterable): Mean of random accelerations (x, y, z)
            axyz_sigma (iterable): Standard deviation of random accelerations (x, y, z)
        """
        n = len(self.particles)
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        daxyz = axyz_sigma * np.random.randn(n, 3)
        self.particles[:, 0:3] += (time_units * self.particles[:, 3:6]
            + 0.5 * (axyz + daxyz) * time_units**2)
        self.particles[:, 3:6] += time_units * (axyz + daxyz)
        self._test_particles()

    def update_weights(self, likelihoods):
        """
        Update particle weights based on their likelihoods.

        Arguments:
            likelihoods (array): Likelihood of each particle
        """
        self.weights = likelihoods + 1e-300
        self.weights *= 1 / self.weights.sum()

    def resample_particles(self, method=None):
        """
        Prune unlikely particles and reproduce likely ones.

        Arguments:
            method (str): Optional override of `self.resample_method`
        """
        n = len(self.particles)
        # Systematic resample (vectorized)
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def systematic():
            positions = (np.arange(n) + np.random.random()) * (1 / n)
            cumulative_weight = np.cumsum(self.weights)
            return np.searchsorted(cumulative_weight, positions)
        # Stratified resample (vectorized)
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def stratified():
            positions = (np.arange(n) + np.random.random(n)) * (1 / n)
            cumulative_weight = np.cumsum(self.weights)
            return np.searchsorted(cumulative_weight, positions)
        # Residual resample (vectorized)
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def residual():
            repetitions = (n * self.weights).astype(int)
            initial_indexes = np.repeat(np.arange(n), repetitions)
            residuals = self.weights - repetitions
            residuals *= 1 / residuals.sum()
            cumulative_sum = np.cumsum(residuals)
            cumulative_sum[-1] = 1.0
            additional_indexes = np.searchsorted(
                cumulative_sum, np.random.random(n - len(initial_indexes)))
            return np.hstack((initial_indexes, additional_indexes))
        # Random choice
        def choice():
            return np.random.choice(np.arange(n), size=(n, ),
                replace=True, p=self.weights)
        if method is None:
            method = self.resample_method
        if method == 'systematic':
            indexes = systematic()
        elif method == 'stratified':
            indexes = stratified()
        elif method == 'residual':
            indexes = residual()
        elif method == 'choice':
            indexes = choice()
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights *= 1 / self.weights.sum()

    def track(self, xy, n=1000, xy_sigma=(0, 0), vxyz=(0, 0, 0), vxyz_sigma=(0, 0, 0),
        axyz=(0, 0, 0), axyz_sigma=(0, 0, 0), datetimes=None, maxdt=datetime.timedelta(0),
        tile_size=(15, 15), parallel=False, return_particles=False, observer_mask=None):
        """
        Track particles through time.

        Velocities and accelerations (`vxyz`, `vxyz_sigma`, `axyz`, `axyz_sigma`)
        are assumed to be in `self.time_unit` time units.

        If `len(xy) > 1`, errors and warnings are caught silently,
        and matching images from Observers with `cache = True` are cached.

        Arguments:
            xy (iterable): Single (x, y) or multiple ((xi, yi), ...) initial positions
            n (int): Number of particles
            xy_sigma (iterable): Standard deviation of initial position (x, y)
            vxyz (iterable): Mean velocity (x, y, z)
            vxyz_sigma (iterable): Standard deviation of velocity (x, y, z)
            axyz (iterable): Mean acceleration (x, y, z)
            axyz_sigma (iterable) Standard deviation of acceleration (x, y, z)
            datetimes (iterable): Monotonic sequence of datetimes at which to
                track particles. If `None`, defaults to all unique datetimes in
                `self.observers`.
            maxdt (timedelta): Maximum timedelta for an image to match `datetimes`
            tile_size (iterable): Size of reference tiles in pixels (width, height)
            parallel: Number of initial positions to track in parallel (int),
                or whether to track in parallel (bool). If `True`,
                all available CPU cores are used.
            return_particles (bool): Whether to return all particles and weights
                at each timestep
            observer_mask (array): Boolean mask of Observers
                to use for each `xy` (len(xy), len(self.observers)). If `None`,
                all Observers are used.

        Returns:
            `Tracks`: Tracks object
        """
        # Save function arguments for Tracks
        # NOTE: Must be called first
        params = locals().copy()
        # Clear any previous tracking state
        self.reset()
        # Enforce defaults
        xy = np.atleast_2d(xy)
        errors = len(xy) <= 1
        parallel = helpers._parse_parallel(parallel)
        if datetimes is None:
            datetimes = np.unique(np.concatenate([
                obs.datetimes for obs in self.observers]))
        else:
            datetimes = self.parse_datetimes(datetimes=datetimes, maxdt=maxdt)
        if observer_mask is None:
            observer_mask = np.ones((len(xy), len(self.observers)), dtype=bool)
        # Compute matching images
        matching_images = self.match_datetimes(datetimes=datetimes, maxdt=maxdt)
        template_images = (matching_images != None).argmax(axis=0)
        # Cache matching images
        if len(xy) > 1:
            for i, observer in enumerate(self.observers):
                if observer.cache:
                    index = [img for img in matching_images[:, i] if img is not None]
                    observer.cache_images(index=index)
        # Define parallel process
        bar = helpers._progress_bar(max=len(xy))
        ntimes = len(datetimes)
        dts = np.diff(datetimes)
        def process(xyi, mask):
            means = np.full((ntimes, 6), np.nan)
            covariances = np.full((ntimes, 6, 6), np.nan)
            if return_particles:
                particles = np.full((ntimes, n, 6), np.nan)
                weights = np.full((ntimes, n), np.nan)
            error = None
            all_warnings = None
            try:
                with warnings.catch_warnings(record=True) as caught:
                    self.initialize_particles(n=n, xy=xyi, xy_sigma=xy_sigma,
                        vxyz=vxyz, vxyz_sigma=vxyz_sigma)
                    # Initialize templates for Observers starting at datetimes[0]
                    for obs, img in enumerate(template_images):
                        if img == 0:
                            self.initialize_template(obs=obs, img=img, tile_size=tile_size)
                    means[0] = self.particle_mean
                    covariances[0] = self.particle_covariance
                    if return_particles:
                        particles[0] = self.particles
                        weights[0] = self.weights
                    for i, dt in enumerate(dts, start=1):
                        self.evolve_particles(dt=dt, axyz=axyz, axyz_sigma=axyz_sigma)
                        # Initialize templates for Observers starting at datetimes[i]
                        for obs, img in enumerate(template_images):
                            if img == i and mask[obs]:
                                self.initialize_template(obs=obs, img=img, tile_size=tile_size)
                        imgs = [img if m else None
                            for img, m in zip(matching_images[i], mask)]
                        likelihoods = self.compute_likelihoods(imgs=imgs)
                        self.update_weights(likelihoods)
                        self.resample_particles()
                        means[i] = self.particle_mean
                        covariances[i] = self.particle_covariance
                        if return_particles:
                            particles[i] = self.particles
                            weights[i] = self.weights
                if caught:
                    warns = tuple(caught)
            except Exception as e:
                # traceback object cannot be pickled, so include in message
                # TODO: Use tblib instead (https://stackoverflow.com/a/26096355)
                if errors:
                    raise e
                elif parallel:
                    error = e.__class__(''.join(
                        traceback.format_exception(*sys.exc_info())))
                else:
                    error = e
            results = [means, covariances, error, all_warnings]
            if return_particles:
                results += [particles, weights]
            return results
        def reduce(results):
            bar.next()
            return results
        # Run process in parallel
        with config._MapReduce(np=parallel) as pool:
            results = pool.map(func=process, reduce=reduce, star=True,
                sequence=tuple(zip(xy, observer_mask)))
        bar.finish()
        # Return results as Tracks
        if return_particles:
            means, covariances, errors, all_warnings, particles, weights = zip(*results)
        else:
            means, covariances, errors, all_warnings = zip(*results)
            particles, weights = None, None
        return Tracks(
            datetimes=datetimes, means=means, covariances=covariances,
            particles=particles, weights=weights,
            tracker=self, images=matching_images, params=params,
            errors=errors, warnings=all_warnings)

    def reset(self):
        """
        Reset to initial state.

        Resets tracking attributes to `None`.
        """
        self.particles = None
        self.weights = None
        self.templates = None

    def parse_datetimes(self, datetimes, maxdt=datetime.timedelta(0)):
        """
        Parse track datetimes.

        Datetimes are tested to be monotonic, duplicates are dropped,
        those matching no Observers are dropped,
        and an error is thrown if fewer than two remain.

        Arguments:
            datetimes (iterable): Monotonically increasing sequence of
                datetimes at which to track particles.
            maxdt (timedelta): Maximum timedelta for an image to match `datetimes`
        """
        datetimes = np.asarray(datetimes)
        # Datetimes must be monotonic
        monotonic = (
            (datetimes[1:] >= datetimes[:-1]).all() or
            (datetimes[1:] <= datetimes[:-1]).all())
        if not monotonic:
            raise ValueError('Datetimes must be monotonic')
        # Datetimes must be unique
        selected = np.concatenate(((True, ), datetimes[1:] != datetimes[:-1]))
        if not all(selected):
            warnings.warn('Dropping duplicate datetimes')
            datetimes = datetimes[selected]
        # Datetimes must match at least one Observer
        distances = helpers.pairwise_distance_datetimes(datetimes, observed_datetimes)
        selected = distances.min(axis=1) <= abs(maxdt.total_seconds())
        if not all(selected):
            warnings.warn('Dropping datetimes not matching any Observers')
            datetimes = datetimes[selected]
        if len(datetimes) < 2:
            raise ValueError('Fewer than two valid datetimes')
        return datetimes

    def match_datetimes(self, datetimes, maxdt=datetime.timedelta(0)):
        """
        Return matching image indices for each Observer and datetime.

        Arguments:
            datetimes (iterable): Datetime objects
            maxdt (timedelta): Maximum timedelta for an image to match `datetimes`

        Returns:
            array: Grid of matching image indices ([i, j] for datetimes[i], observers[j]),
                or `None` for no match
        """
        matches = np.full((len(datetimes), len(self.observers)), None)
        for i, observer in enumerate(self.observers):
            distances = helpers.pairwise_distance_datetimes(datetimes, observer.datetimes)
            nearest_index = np.argmin(distances, axis=1)
            matches[:, i] = nearest_index
            nearest_distance = distances[np.arange(distances.shape[0]), nearest_index]
            not_selected = nearest_distance > abs(maxdt.total_seconds())
            matches[not_selected, i] = None
        return matches

    def _extract_tile(self, obs, img, box, histogram=None, return_histogram=False):
        """
        Extract image tile.

        The tile is converted to grayscale, normalized to mean 0, variance 1,
        matched to a histogram (if `histogram`), and passed through a
        median low pass filer.

        Arguments:
            obs (int): Observer index
            img (int): Observer image index
            box (iterable): Tile boundaries (see `Observer.extract_tile()`)
            histogram (iterable): Template for histogram matching (see `helpers.match_histogram`)
            return_histogram (bool): Whether to return a tile histogram.
                The histogram is computed for the low pass filter.
        """
        tile = self.observers[obs].extract_tile(box=box, img=img)
        if tile.ndim > 2:
            tile = helpers.rgb_to_gray(tile, **self.grayscale)
        tile = helpers.normalize(tile)
        if histogram is not None:
            tile = helpers.match_histogram(tile, template=histogram)
        if return_histogram:
            returned_histogram = helpers.compute_cdf(tile, return_inverse=False)
        tile_low = scipy.ndimage.filters.median_filter(tile, **self.highpass)
        tile -= tile_low
        if return_histogram:
            return tile, returned_histogram
        else:
            return tile

    def initialize_template(self, obs, img, tile_size):
        """
        Initialize an observer template from the current particle state.

        Arguments:
            obs (int): Observer index
            img (int): Observer image index
            tile_size (iterable): Size of template tile in pixels (nx, ny)
        """
        if self.templates is None:
            self.templates = [None] * len(self.observers)
        # Compute image box centered around particle mean
        xyz = self.particle_mean[None, 0:3]
        uv = self.observers[obs].project(xyz, img=img).ravel()
        box = self.observers[obs].tile_box(uv, size=tile_size)
        # Build template
        template = dict(
            obs=obs, img=img, box=box,
            duv=uv - box.reshape(2, -1).mean(axis=0))
        template['tile'], template['histogram'] = self._extract_tile(
            obs=obs, img=img, box=box, return_histogram=True)
        self.templates[obs] = template

    def compute_likelihoods(self, imgs):
        """
        Compute the particle likelihoods summed across all observers.

        Arguments:
            imgs (iterable): Image index for each Observer, or `None` to skip
        """
        log_likelihoods_observer = [self._compute_observer_log_likelihoods(obs, img)
            for obs, img in enumerate(imgs)]
        z = self._sample_dem(self.particles[:, 0:2])
        z_sigma = self._sample_dem(self.particles[:, 0:2], sigma=True)
        # Avoid division by zero
        nonzero = np.nonzero(z_sigma)[0]
        log_likelihoods_dem = np.zeros(len(self.particles), dtype=float)
        log_likelihoods_dem[nonzero] = (1 / (2 * z_sigma[nonzero]**2) *
            (z[nonzero] - self.particles[nonzero, 2])**2)
        return np.exp(-sum(log_likelihoods_observer) - log_likelihoods_dem)

    def _compute_observer_log_likelihoods(self, obs, img):
        """
        Compute the log likelihoods of each particle for an Observer.

        Arguments:
            t (datetime): Date and time at which to query Observer
            observer (Observer): Observer object
        """
        constant_log_likelihood = np.array([0.0])
        if img is None:
            return constant_log_likelihood
        # Build image box around all particles, with a buffer for template matching
        size = np.asarray(self.templates[obs]['tile'].shape[0:2][::-1])
        uv = self.observers[obs].project(self.particles[:, 0:3], img=img)
        halfsize = size * 0.5
        box = np.row_stack((
            uv.min(axis=0) - halfsize,
            uv.max(axis=0) + halfsize))
        # Enlarge box to ensure SSE has cols, rows (ky + 1, kx + 1) for interpolation
        ky = self.interpolation.get('ky', 3)
        ncols = ky - (np.diff(box[:, 0]) - size[0])
        if ncols > 0:
            # Widen box in 2nd ('y') dimension (x|cols)
            box[:, 0] += np.hstack((-ncols, ncols)) * 0.5
        kx = self.interpolation.get('kx', 3)
        nrows = kx - (np.diff(box[:, 1]) - size[1])
        if nrows > 0:
            # Widen box in 1st ('x') dimension (y|rows)
            box[:, 1] += np.hstack((-nrows, nrows)) * 0.5
        box = np.vstack((np.floor(box[0, :]), np.ceil(box[1, :]))).astype(int)
        # Check that box is within image bounds
        if not all(self.observers[obs].grid.inbounds(box)):
            warnings.warn('Particles too close to or beyond image bounds, skipping image')
            return constant_log_likelihood
        # Flatten box
        box = box.ravel()
        # Extract search tile
        search_tile = self._extract_tile(
            obs=obs, img=img, box=box, histogram=self.templates[obs]['histogram'])
        # Compute area-averaged sum of squares error (SSE)
        sse = cv2.matchTemplate(search_tile.astype(np.float32),
            templ=self.templates[obs]['tile'].astype(np.float32), method=cv2.TM_SQDIFF)
        sse *= 1 / (size[0] * size[1])
        # Compute SSE bounding box
        # (relative to search tile: shrunk by halfsize of template tile - 0.5 pixel)
        box_edge = halfsize - 0.5
        sse_box = box + np.concatenate((box_edge, -box_edge))
        # (shift by subpixel offset of template tile)
        sse_box += np.tile(self.templates[obs]['duv'], 2)
        # Sample at projected particles
        sampled_sse = self.observers[obs].sample_tile(uv, tile=sse,
            box=sse_box, grid=False, **self.interpolation)
        return sampled_sse * (1 / (2 * self.observers[obs].sigma**2))

class Tracks(object):
    """
    A `Tracks' contains the estimated trajectories of world points.

    In the array dimensions below, n: number of tracks, m: number of datetimes,
    p: number of particles.

    Attributes:
        datetimes (array): Datetimes at which particles were estimated (m, )
        means (array): Mean particle positions and velocities (x, y, z, vx, vy, vz) (n, m, 6)
        covariances (array): Covariance of particle positions and velocities (n, m, 6, 6)
        particles (array): Particle positions and velocities (n, m, p, 6)
        weights (array): Particle weights (n, m, p)
        tracker (Tracker): Tracker object used for tracking
        images (array): Grid of image indices ([i, j] for `datetimes[i]`, `tracker.observers`[j]`).
            `None` indicates no image from `tracker.observers[j]` matched `datetimes[i]`.
        params (dict): Arguments to `Tracker.track`
        errors (array): The error, if any, caught for each track (n, ).
            An error indicates that the track did not complete successfully.
        warnings (array): Warnings, if any, caught for each track (n, ).
            Warnings indicate the track completed but may not be valid.
    """

    def __init__(self, datetimes, means, covariances=None, particles=None, weights=None,
        tracker=None, images=None, params=None, errors=None, warnings=None):
        self.datetimes = np.asarray(datetimes)
        if np.iterable(means) and not isinstance(means, np.ndarray):
            means = np.stack(means, axis=0)
        self.means = means
        if np.iterable(covariances) and not isinstance(covariances, np.ndarray):
            covariances = np.stack(covariances, axis=0)
        self.covariances = covariances
        if np.iterable(particles) and not isinstance(particles, np.ndarray):
            particles = np.stack(particles, axis=0)
        self.particles = particles
        if np.iterable(weights) and not isinstance(weights, np.ndarray):
            weights = np.stack(weights, axis=0)
        self.weights = weights
        self.tracker = tracker
        self.images = np.asarray(images)
        self.params = params
        self.errors = np.asarray(errors)
        self.warnings = np.asarray(warnings)

    @property
    def xyz(self):
        """
        array: Mean particle positions (n, m, [x, y, z])
        """
        return self.means[:, :, 0:3]

    @property
    def vxyz(self):
        """
        array: Mean particle velocities (n, m, [vx, vy, vz])
        """
        return self.means[:, :, 3:6]

    @property
    def xyz_sigma(self):
        """
        array: Standard deviation of particle positions (n, m, [x, y, z])
        """
        if self.covariances is not None:
            return np.sqrt(self.covariances[:, :, (0, 1, 2), (0, 1, 2)])

    @property
    def vxyz_sigma(self):
        """
        array: Standard deviation of particle velocities (n, m, [vx, vy, vz])
        """
        if self.covariances is not None:
            return np.sqrt(self.covariances[:, :, (3, 4, 5), (3, 4, 5)])

    @property
    def success(self):
        """
        array: Whether track completed without errors (n, )
        """
        if self.errors is not None:
            return np.array([error is None for error in self.errors])

    def plot_xy(self, tracks=None, start=True, mean=True, sigma=False):
        """
        Plot tracks on the x-y plane.

        Arguments:
            tracks: Slice object or iterable of indices of tracks to include.
                If `None`, all tracks are included.
            start: Whether to plot starting x, y (bool) or arguments to
                `matplotlib.pyplot.plot()` (dict)
            mean: Whether to plot mean x, y (bool) or arguments to
                `matplotlib.pyplot.plot()` (dict)
            sigma: Whether to plot sigma x, y (bool) or arguments to
                `matplotlib.pyplot.plot()` (dict)
        """
        if tracks is None:
            tracks = slice(None)
        if mean:
            if mean is True:
                mean = dict()
            default = dict(color='black')
            mean = helpers.merge_dicts(default, mean)
            matplotlib.pyplot.plot(self.xyz[tracks, :, 0].T, self.xyz[tracks, :, 1].T, **mean)
        if start:
            if start is True:
                start = dict()
            default = dict(color='black', marker='.', linestyle='none')
            if isinstance(mean, dict) and 'color' in mean:
                default['color'] = mean['color']
            start = helpers.merge_dicts(default, start)
            matplotlib.pyplot.plot(self.xyz[tracks, 0, 0], self.xyz[tracks, 0, 1], **start)
        if sigma:
            if sigma is True:
                sigma = dict()
            default = dict(color='black', alpha=0.25)
            if isinstance(mean, dict) and 'color' in mean:
                default['color'] = mean['color']
            sigma = helpers.merge_dicts(default, sigma)
            for i in np.atleast_1d(np.arange(len(self.xyz))[tracks]):
                matplotlib.pyplot.errorbar(
                    self.xyz[i, :, 0], self.xyz[i, :, 1],
                    xerr=self.xyz_sigma[i, :, 0], yerr=self.xyz_sigma[i, :, 1],
                    **sigma)

    def plot_vxy(self, tracks=None, **kwargs):
        """
        Plot velocities as vector fields on the x-y plane.

        Arguments:
            tracks: Slice object or iterable of indices of tracks to include.
                If `None`, all tracks are included.
            **kwargs: Additional arguments to `matplotlib.pyplot.quiver()`
        """
        if tracks is None:
            tracks = slice(None)
        default = dict(angles='xy')
        kwargs = helpers.merge_dicts(default, kwargs)
        for i in np.atleast_1d(np.arange(len(self.xyz))[tracks]):
            matplotlib.pyplot.quiver(
                self.xyz[i, :, 0], self.xyz[i, :, 1],
                self.vxyz[i, :, 0], self.vxyz[i, :, 1], **kwargs)

    def plot_v1d(self, dim, tracks=None, mean=True, sigma=False):
        """
        Plot velocity for one dimension.

        Arguments:
            tracks: Slice object or iterable of indices of tracks to include.
                If `None`, all tracks are included.
            mean: Whether to plot mean vx (bool) or arguments to
                `matplotlib.pyplot.plot()` (dict)
            sigma: Whether to plot sigma vx (bool) or arguments to
                `matplotlib.pyplot.fill_between()` (dict)
        """
        if tracks is None:
            tracks = slice(None)
        if mean:
            if mean is True:
                mean = dict()
            default = dict(color='black')
            mean = helpers.merge_dicts(default, mean)
            matplotlib.pyplot.plot(self.datetimes, self.vxyz[tracks, :, dim].T, **mean)
        if sigma:
            if sigma is True:
                sigma = dict()
            default = dict(facecolor='black', edgecolor='none', alpha=0.25)
            if isinstance(mean, dict) and 'color' in mean:
                default['facecolor'] = mean['color']
            sigma = helpers.merge_dicts(default, sigma)
            for i in np.atleast_1d(np.arange(len(self.xyz))[tracks]):
                matplotlib.pyplot.fill_between(
                    self.datetimes,
                    y1=self.vxyz[i, :, dim] + self.vxyz_sigma[i, :, dim],
                    y2=self.vxyz[i, :, dim] - self.vxyz_sigma[i, :, dim],
                    **sigma)

    def animate(self, track, obs=0, frames=None, images=None, particles=None,
        map_size=(20, 20), img_size=(100, 100), subplots=dict(), animation=dict()):
        """
        Animate track.

        Arguments:
            track (int): Track index
            obs (int): Observer index
            frames (iterable): Datetime index.
                Any requested datetimes with missing data are skipped.
                If `None`, all times with a result for `track`, `obs` are used.
            images (bool): Whether to plot images,
                or `None` to plot only if `self.tracker` is set
            particles (bool): Whether to plot particles,
                or `None` to plot only if `self.particles` and `self.weights` are set
            map_size (iterable): Size of map window in world units
            img_size (iterable): Size of image window in pixels
            subplots (dict): Optional arguments to `matplotlib.pyplot.subplots()`
            animation (dict): Optional arguments to `matplotlib.animation.FuncAnimation`
        """
        if images is None:
            images = self.tracker is not None
        if particles is None:
            particles = self.particles is not None and self.weights is not None
        if images:
            fig, axes = matplotlib.pyplot.subplots(ncols=2, **subplots)
        else:
            fig, axes = matplotlib.pyplot.subplots(ncols=1, **subplots)
            axes = [axes]
        # Select frames for which track has a solution and observer has an image
        if frames is None:
            frames = np.arange(len(self.datetimes))
        has_frame = np.where(
            ~np.isnan(self.xyz[track, :, 0]) &
            self.images[:, obs] != None)[0]
        frames = np.intersect1d(frames, has_frame)
        # Initialize plot
        i = frames[0]
        img = self.images[i, obs]
        # Map: Track
        track_xyz = self.xyz[track, :(i + 1)]
        map_track = axes[0].plot(track_xyz[:, 0], track_xyz[:, 1], color='black', marker='.')[0]
        if images:
            # Image: Track
            track_uv = self.tracker.observers[obs].project(track_xyz, img=img)
            image_track = axes[1].plot(track_uv[:, 0], track_uv[:, 1], color='black', marker='.')[0]
            # Image: Mean
            image_mean = axes[1].plot(track_uv[-1, 0], track_uv[-1, 1], color='red', marker='.')[0]
            # Image: Tile
            box = self.tracker.observers[obs].tile_box(track_uv[-1], size=img_size)
            tile = self.tracker.observers[obs].extract_tile(img=img, box=box)
            image_tile = self.tracker.observers[obs].plot_tile(tile=tile, box=box, axes=axes[1])
        # Map: Basename
        if images:
            basename = helpers.strip_path(self.tracker.observers[obs].images[img].path)
        else:
            basename = str(obs) + ' : ' + str(img)
        map_txt = axes[0].text(0.5, 0.9, basename, color='black',
            horizontalalignment='center', transform=axes[0].transAxes)
        if particles:
            # Compute quiver scales
            scales = np.diff(self.datetimes[frames]) / self.tracker.time_unit
            # Compute weight limits for static colormap
            clim = (
                self.weights[track, :].ravel().min(),
                self.weights[track, :].ravel().max())
        elif self.tracker is not None:
            scales = np.diff(self.datetimes[frames]) / self.tracker.time_unit
        else:
            scales = np.ones(len(frames) - 1)
        # Discard last frame
        frames = frames[:-1]
        def update_plot(i):
            # PathCollections cannot set x, y, so new objects have to be created
            for ax in axes:
                ax.collections = []
            img = self.images[i, obs]
            if particles:
                # Map: Particles
                particle_xyz = self.particles[track, i, :, 0:3]
                particle_vxy = self.particles[track, i, :, 3:5] * scales[i]
                axes[0].quiver(
                    particle_xyz[:, 0], particle_xyz[:, 1],
                    particle_vxy[:, 0], particle_vxy[:, 1],
                    self.weights[track, i],
                    cmap=matplotlib.pyplot.cm.gnuplot2, alpha=0.25,
                    angles='xy', scale=1, scale_units='xy', units='xy', clim=clim)
                    # matplotlib.pyplot.colorbar(quivers, ax=axes[0], label='Weight')
            if images and particles:
                # Image: Particles
                particle_uv = self.tracker.observers[obs].project(particle_xyz, img=img)
                axes[1].scatter(
                    particle_uv[:, 0], particle_uv[:, 1],
                    c=self.weights[track, i], marker='.',
                    cmap=matplotlib.pyplot.cm.gnuplot2, alpha=0.25, edgecolors='none',
                    vmin=clim[0], vmax=clim[1])
                # matplotlib.pyplot.colorbar(image_particles, ax=axes[1], label='Weight')
            # Map: Track
            track_xyz = self.xyz[track, :(i + 1)]
            map_track.set_data(track_xyz[:, 0], track_xyz[:, 1])
            axes[0].set_xlim(track_xyz[-1, 0] - map_size[0] / 2, track_xyz[-1, 0] + map_size[0] / 2)
            axes[0].set_ylim(track_xyz[-1, 1] - map_size[1] / 2, track_xyz[-1, 1] + map_size[1] / 2)
            # Map: Mean
            axes[0].quiver(
                self.xyz[track, i, 0], self.xyz[track, i, 1],
                self.vxyz[track, i, 0] * scales[i], self.vxyz[track, i, 1] * scales[i],
                color='red', alpha=1,
                angles='xy', scale=1, scale_units='xy', units='xy')
            if images:
                # Image: Track
                track_uv = self.tracker.observers[obs].project(track_xyz, img=img)
                image_track.set_data(track_uv[:, 0], track_uv[:, 1])
                axes[1].set_xlim(track_uv[-1, 0] - img_size[0] / 2, track_uv[-1, 0] + img_size[0] / 2)
                axes[1].set_ylim(track_uv[-1, 1] + img_size[1] / 2, track_uv[-1, 1] - img_size[1] / 2)
                # Image: Mean
                image_mean.set_data(track_uv[-1, 0], track_uv[-1, 1])
                # Image: Tile
                box = self.tracker.observers[obs].tile_box(uv=track_uv[-1, :], size=img_size)
                tile = self.tracker.observers[obs].extract_tile(box=box, img=img)
                image_tile.set_data(tile)
                image_tile.set_extent((box[0], box[2], box[3], box[1]))
            # Map: Basename
            if images:
                basename = helpers.strip_path(self.tracker.observers[obs].images[img].path)
            else:
                basename = str(obs) + ' : ' + str(img)
            basename = helpers.strip_path(self.tracker.observers[obs].images[img].path)
            map_txt.set_text(basename)
            if images:
                return map_track, map_txt, image_track, image_tile, image_mean
            else:
                return map_track, map_txt
        return matplotlib.animation.FuncAnimation(fig, update_plot, frames=frames, blit=True, **animation)
