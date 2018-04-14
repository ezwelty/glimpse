from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, cv2, warnings, datetime, scipy, sharedmem, matplotlib)
from . import (helpers, dem as DEM)

class Tracker(object):
    """
    A `Tracker' estimates the trajectory of world points through time.

    Attributes:
        observers (list): Observer objects
        dem (DEM): Digital elevation model of the surface on which to track points
        time_unit (timedelta): Length of time unit for temporal arguments
        viewshed (DEM): `DEM` object of a binary viewshed.
            Can also be an array, in which case it must be the same shape as `dem.Z`.
        resample_method (str): Particle resampling method
            ('systematic', 'stratified', 'residual', 'choice': np.random.choice with replacement)
        grayscale (dict): Grayscale conversion
            (arguments to glimpse.helpers.rgb_to_gray)
        highpass (dict): Median high-pass filter
            (arguments to scipy.ndimage.filters.median_filter)
        interpolation (dict): Subpixel interpolation
            (arguments to scipy.interpolate.RectBivariateSpline)
        particles (array): Positions and velocities of particles (n, 5) [[x, y, z, vx, vy], ...]
        weights (array): Particle likelihoods (n, )
        particle_mean (array): Weighted mean of `particles` (5, ) [x, y, z, vx, vy]
        particle_covariance (array): Weighted covariance matrix of `particles` (5, 5)
        templates (list): For each Observer, a template extracted from the first image
            matching a `datetimes` centered around the `particle_mean` at that time.
            Templates are dictionaries that include at least

            - 'tile': Image tile used as a template for image cross-correlation
            - 'histogram': Histogram (values, quantiles) of the 'tile' used for histogram matching
            - 'duv': Subpixel offset of 'tile' (desired - sampled)
    """
    def __init__(self, observers, dem, time_unit, viewshed=None, resample_method='systematic',
        grayscale=dict(method='average'), highpass=dict(size=(5, 5)), interpolation=dict(kx=3, ky=3)):
        self.observers = observers
        self.dem = dem
        if isinstance(viewshed, np.ndarray):
            viewshed = DEM.DEM(Z=viewshed, x=self.dem.x, y=self.dem.y)
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
            is_visible = self.viewshed.sample(self.particles[:, 0:2], method='nearest')
            if not all(is_visible):
                raise ValueError('Some particles are on non-visible viewshed cells')
        if any(np.isnan(self.particles[:, 2])):
            raise ValueError('Some particles are on NaN dem cells')

    def initialize_particles(self, xy, n=1000, xy_sigma=(0, 0), vxy=(0, 0), vxy_sigma=(0, 0)):
        """
        Initialize particles given an initial normal distribution.

        Temporal arguments (`vxy`, `vxy_sigma`) are assumed to be in
        `self.time_unit` time units.

        Arguments:
            xy (iterable): Mean position (x, y)
            n (int): Number of particles
            xy_sigma (iterable): Standard deviation of position (x, y)
            vxy (iterable): Mean velocity (x, y)
            vxy_sigma (iterable): Standard deviation of velocity (x, y)
        """
        if self.particles is None or len(self.particles) != n:
            self.particles = np.zeros((n, 5))
        self.particles[:, 0:2] = xy + xy_sigma * np.random.randn(n, 2)
        self.particles[:, 2] = self.dem.sample(self.particles[:, 0:2])
        self.particles[:, 3:5] = vxy + vxy_sigma * np.random.randn(n, 2)
        self._test_particles()
        self.weights = np.full(n, 1.0 / n)

    def evolve_particles(self, dt, axy=(0, 0), axy_sigma=(0, 0)):
        """
        Evolve particles through time by stochastic differentiation.

        Accelerations (`axy`, `axy_sigma`) are assumed to be with respect to
        `self.time_unit`.

        Arguments:
            dt (timedelta): Time difference to evolve particles forward or backward
            axy (iterable): Mean of random accelerations (x, y)
            axy_sigma (iterable): Standard deviation of random accelerations (x, y)
        """
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        daxy = axy_sigma * np.random.randn(len(self.particles), 2)
        self.particles[:, 0:2] += (time_units * self.particles[:, 3:5]
            + 0.5 * (axy + daxy) * time_units**2)
        self.particles[:, 2] = self.dem.sample(self.particles[:, 0:2])
        self.particles[:, 3:5] += time_units * (axy + daxy)
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

    def track(self, xy, n=1000, xy_sigma=(0, 0), vxy=(0, 0), vxy_sigma=(0, 0),
        axy=(0, 0), axy_sigma=(0, 0), datetimes=None, maxdt=datetime.timedelta(0),
        tile_size=(15, 15), parallel=False):
        """
        Track particles through time.

        Velocities and accelerations (`vxy`, `vxy_sigma`, `axy`, `axy_sigma`)
        are assumed to be in `self.time_unit` time units.

        If `parallel == True`, errors are caught silently and included in the result.
        If `len(xy) > 1`, matching images from Observers with `cache = True` are cached.

        Arguments:
            xy (iterable): Single (x, y) or multiple ((xi, yi), ...) initial positions
            n (int): Number of particles
            xy_sigma (iterable): Standard deviation of initial position (x, y)
            vxy (iterable): Mean velocity (x, y)
            vxy_sigma (iterable): Standard deviation of velocity (x, y)
            axy (iterable): Mean acceleration (x, y)
            axy_sigma (iterable) Standard deviation of acceleration (x, y)
            datetimes (iterable): Monotonic sequence of datetimes at which to
                track particles. If `None`, defaults to all unique datetimes in
                `self.observers`.
            maxdt (timedelta): Maximum timedelta for an image to match `datetimes`
            tile_size (iterable): Size of reference tiles in pixels (width, height)
            parallel: Number of initial positions to track in parallel (int),
                or whether to track in parallel (bool). If `True`,
                all available CPU cores are used.

        Returns:
            `Tracks`: Tracks object
        """
        # Save function arguments for Tracks
        # NOTE: Must be called first
        params = locals().copy()
        # Clear any previous tracking state
        self.reset()
        # Enforce defaults
        errors = not parallel
        if parallel is True:
            parallel = sharedmem.cpu_count()
        elif parallel is False:
            parallel = 0
        xy = np.atleast_2d(xy)
        if datetimes is None:
            datetimes = np.unique(np.concatenate([
                obs.datetimes for obs in self.observers]))
        else:
            datetimes = self.parse_datetimes(datetimes=datetimes, maxdt=maxdt)
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
        def process(xyi):
            means = np.full((len(datetimes), 5), np.nan)
            covariances = np.full((len(datetimes), 5, 5), np.nan)
            error = None
            all_warnings = None
            try:
                with warnings.catch_warnings(record=True) as caught:
                    self.initialize_particles(n=n, xy=xyi, xy_sigma=xy_sigma, vxy=vxy, vxy_sigma=vxy_sigma)
                    # Initialize templates for Observers starting at datetimes[0]
                    for obs, img in enumerate(template_images):
                        if img == 0:
                            self.initialize_template(obs=obs, img=img, tile_size=tile_size)
                    means[0] = self.particle_mean
                    covariances[0] = self.particle_covariance
                    dts = np.diff(datetimes)
                    for i, dt in enumerate(dts, start=1):
                        self.evolve_particles(dt=dt, axy=axy, axy_sigma=axy_sigma)
                        # Initialize templates for Observers starting at datetimes[i]
                        for obs, img in enumerate(template_images):
                            if img == i:
                                self.initialize_template(obs=obs, img=img, tile_size=tile_size)
                        likelihoods = self.compute_likelihoods(imgs=matching_images[i])
                        self.update_weights(likelihoods)
                        self.resample_particles()
                        means[i] = self.particle_mean
                        covariances[i] = self.particle_covariance
                if caught:
                    warns = tuple(caught)
            except Exception as e:
                error = e
                if errors:
                    raise e
            return means, covariances, error, all_warnings
        # Run process in parallel
        with sharedmem.MapReduce(np=parallel) as pool:
            results = pool.map(process, xy)
        # Return results as Tracks
        means, covariances, errors, all_warnings = zip(*results)
        return Tracks(
            datetimes=datetimes, means=means, covariances=covariances,
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
        log_likelihoods = [self._compute_observer_log_likelihoods(obs, img)
            for obs, img in enumerate(imgs)]
        return np.exp(-sum(log_likelihoods))

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
        return sampled_sse * (1 / self.observers[obs].sigma**2)

class Tracks(object):
    """
    A `Tracks' contains the estimated trajectories of world points.

    In the array dimensions below, n: number of tracks, m: number of datetimes.

    Attributes:
        datetimes (array): Datetimes at which particles were estimated (m, )
        means (array): Mean particle positions and velocities (x, y, z, vx, vy) (n, m, 5)
        covariances (array): Covariance of particle positions and velocities (n, m, 5, 5)
        tracker (Tracker): Tracker object used for tracking
        images (array): Grid of image indices ([i, j] for `datetimes[i]`, `tracker.observers`[j]`).
            `None` indicates no image from `tracker.observers[j]` matched `datetimes[i]`.
        params (dict): Arguments to `Tracker.track`
        errors (array): The error, if any, caught for each track (n, ).
            An error indicates that the track did not complete successfully.
        warnings (array): Warnings, if any, caught for each track (n, ).
            Warnings indicate the track completed but may not be valid.
    """

    def __init__(self, datetimes, means, covariances=None,
        tracker=None, images=None, params=None, errors=None, warnings=None):
        self.datetimes = np.asarray(datetimes)
        if not isinstance(means, np.ndarray):
            means = np.stack(means, axis=0)
        self.means = means
        if not isinstance(covariances, np.ndarray):
            covariances = np.stack(covariances, axis=0)
        self.covariances = covariances
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
    def vxy(self):
        """
        array: Mean particle velocities (n, m, [vx, vy])
        """
        return self.means[:, :, 3:5]

    @property
    def xyz_sigma(self):
        """
        array: Standard deviation of particle positions (n, m, [x, y, z])
        """
        if self.covariances is not None:
            return np.sqrt(self.covariances[:, :, (0, 1, 2), (0, 1, 2)])

    @property
    def vxy_sigma(self):
        """
        array: Standard deviation of particle velocities (n, m, [vx, vy])
        """
        if self.covariances is not None:
            return np.sqrt(self.covariances[:, :, (3, 4), (3, 4)])

    @property
    def success(self):
        """
        array: Whether track completed without errors (n, )
        """
        if self.errors is not None:
            return np.array([error is None for error in self.errors])

    def plot_xy(self, **kwargs):
        matplotlib.pyplot.plot(self.xyz[:, :, 0].T, self.xyz[:, :, 1].T, **kwargs)
        matplotlib.pyplot.plot(self.xyz[:, 0, 0], self.xyz[:, 0, 1], marker='.', linestyle='None', **kwargs)

    def plot_vx(self, **kwargs):
        matplotlib.pyplot.plot(self.datetimes, self.vxy[:, :, 0].T, **kwargs)

    def plot_vy(self, **kwargs):
        matplotlib.pyplot.plot(self.datetimes, self.vxy[:, :, 1].T, **kwargs)
