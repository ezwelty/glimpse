from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, cv2, warnings, datetime, scipy)
from . import (helpers, dem as DEM)

class Tracker(object):
    """
    A `Tracker' estimates the trajectory of world points through time.

    Attributes:
        observers (list): Observer objects
        dem (DEM): Digital elevation model of the surface on which to track points
        viewshed (DEM): `DEM` object of a binary viewshed.
            Can also be an array, in which case it must be the same shape as `dem.Z`.
        time_unit (float): Length of time unit for temporal arguments, in seconds
            (e.g., 1 minute = 60, 1 hour = 3600)
        resample_method (str): Particle resampling method
            ('systematic', 'stratified', 'residual', 'choice': np.random.choice with replacement)
        grayscale (dict): Grayscale conversion
            (arguments to glimpse.helpers.rgb_to_gray)
        highpass (dict): Median high-pass filter
            (arguments to scipy.ndimage.filters.median_filter)
        interpolation (dict): Subpixel interpolation
            (arguments to scipy.interpolate.RectBivariateSpline)
        n (int): Number of particles (more particles gives better results at higher expense)
        particles (array): Positions and velocities of particles (n, 5) [[x, y, z, vx, vy], ...]
        weights (array): Particle likelihoods (n, )
        particle_mean (array): Weighted mean of `particles` (1, 5) [[x, y, z, vx, vy]]
        particle_covariance (array): Weighted covariance matrix of `particles` (5, 5)
        datetimes (array): Date and times at which particle positions were estimated
        means (list): `particle_mean` at each `datetimes`
        covariances (list): `particle_covariance` at each `datetimes`
        tiles (list): For each Observer, a tile extracted from the first
            Image matching a `datetimes` centered around the `particle_mean`
        histograms (list): Histogram (values, quantiles) of each `tiles`
            for histogram matching
    """
    def __init__(self, observers, dem, viewshed=None, time_unit=1, resample_method='systematic',
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
        # Placeholders: particles
        self.particles = None
        self.weights = None
        # Placeholders: observers
        self.tiles = None
        self.histograms = None
        # Placeholders: track
        self.datetimes = None
        self.means = None
        self.covariances = None

    @property
    def n(self):
        return len(self.particles)

    @property
    def particle_mean(self):
        return np.average(self.particles, weights=self.weights, axis=0).reshape(1, -1)

    @property
    def particle_covariance(self):
        return np.cov(self.particles.T, aweights=self.weights)

    @property
    def initialized(self):
        return self.particles is not None

    def initialize_particles(self, n, xy, xy_sigma, vxy=(0, 0), vxy_sigma=(0, 0)):
        """
        Initialize particles given an initial normal distribution.

        Temporal arguments (`vxy`, `vxy_sigma`) are assumed to be in
        `self.time_unit` time units.

        Arguments:
            n (int): Number of particles
            xy (array-like): Mean position (x, y)
            xy_sigma (array-like): Standard deviation of positions (x, y)
            vxy (array-like): Mean velocity (x, y)
            vxy_sigma (array-like): Standard deviation of velocities (x, y)
        """
        if self.particles is None or self.n != n:
            self.particles = np.zeros((n, 5))
        self.particles[:, 0:2] = xy + xy_sigma * np.random.randn(n, 2)
        self.particles[:, 2] = self.dem.sample(self.particles[:, 0:2])
        self.particles[:, 3:5] = vxy + vxy_sigma * np.random.randn(n, 2)
        self.weights = np.full(n, 1.0 / n)

    def advance_particles(self, dt=1, axy=(0, 0), axy_sigma=(0, 0)):
        """
        Advance particles forward in time by stochastic differentiation.

        Temporal arguments (`dt`, `axy`, `axy_sigma`) are assumed to be in
        `self.time_unit` time units.

        Arguments:
            dt (float): Time step
            axy (array-like): Mean of random accelerations (x, y)
            axy_sigma (array-like): Standard deviation of random accelerations (x, y)
        """
        daxy = axy_sigma * np.random.randn(self.n, 2)
        self.particles[:, 0:2] += dt * self.particles[:, 3:5] + 0.5 * (axy + daxy) * dt**2
        if self.viewshed is not None:
            is_visible = self.viewshed.sample(self.particles[:, 0:2], method='nearest')
            if not all(is_visible):
                raise ValueError('Some particles are not visible')
        self.particles[:, 2] = self.dem.sample(self.particles[:, 0:2])
        if any(np.isnan(self.particles[:, 2])):
            raise ValueError('Some particles have missing elevations')
        self.particles[:, 3:5] += dt * (axy + daxy)

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
        # Systematic resample
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def systematic():
            positions = (np.arange(self.n) + np.random.random()) * (1.0 / self.n)
            cumulative_weight = np.cumsum(self.weights)
            indexes = np.zeros(self.n, dtype=int)
            i, j = 0, 0
            while i < self.n:
                if positions[i] < cumulative_weight[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            return indexes
        # Stratified resample
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def stratified():
            positions = (np.arange(self.n) + np.random.random(self.n)) * (1.0 / self.n)
            cumulative_weight = np.cumsum(self.weights)
            indexes = np.zeros(self.n, dtype=int)
            i, j = 0, 0
            while i < self.n:
                if positions[i] < cumulative_weight[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            return indexes
        # Residual resample (vectorized)
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def residual():
            repetitions = (self.n * self.weights).astype(int)
            initial_indexes = np.repeat(np.arange(self.n), repetitions)
            residuals = self.weights - repetitions
            residuals *= 1 / residuals.sum()
            cumulative_sum = np.cumsum(residuals)
            cumulative_sum[-1] = 1.0
            additional_indexes = np.searchsorted(
                cumulative_sum, np.random.random(self.n - len(initial_indexes)))
            return np.hstack((initial_indexes, additional_indexes))
        # Random choice
        def choice():
            return np.random.choice(np.arange(self.n), size=(self.n, ),
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

    def track(self, datetimes=None, maxdt=0, tile_size=(15, 15),
        axy=(0, 0), axy_sigma=(0, 0)):
        """
        Track particles through time.

        Temporal arguments (`maxdt`, `axy`, `axy_sigma`) are assumed to be in
        `self.time_unit` time units.
        Tracking results are saved in
        `self.datetimes`, `self.means`, and `self.covariances`.

        Arguments:
            datetimes (array-like): Monotonically increasing sequence of
                datetimes at which to track particles.
                If `None`, defaults to all unique datetimes in `self.observers`.
            maxdt (float): Maximum time delta for an image to match `datetimes`
            tile_size (array-like): Size of reference tiles in pixels (width, height)
            axy (array-like): Mean of random accelerations (x, y)
            axy_sigma (array-like) Standard deviation of random accelerations (x, y)
        """
        if self.particles is None:
            raise Exception('Particles are not initialized')
        self._initialize_track(datetimes=datetimes, maxdt=maxdt, tile_size=tile_size)
        self.means = [self.particle_mean]
        self.covariances = [self.particle_covariance]
        time_deltas = np.array([dt.total_seconds() for dt in np.diff(self.datetimes)])
        time_deltas *= 1.0 / self.time_unit
        for t, dt in zip(self.datetimes[1:], time_deltas):
            self.advance_particles(dt=dt, axy=axy, axy_sigma=axy_sigma)
            likelihoods = self.compute_likelihoods(t, maxdt=maxdt)
            self.update_weights(likelihoods)
            self.resample_particles()
            self.means.append(self.particle_mean)
            self.covariances.append(self.particle_covariance)

    def _initialize_track(self, datetimes=None, maxdt=0, tile_size=(15, 15)):
        """
        Initialize track properties.

        `self.datetimes`: Datetimes are tested to be monotonically increasing,
        duplicates are dropped, those matching no Observers are dropped,
        and an error is thrown if fewer than two remain.

        `self.tiles` and `self.histograms`: For each Observer, the earliest image
        matching a datetime is selected and a reference image tile is extracted
        centered around the weighted mean of the current particles.

        Arguments:
            datetimes (array-like): Monotonically increasing sequence of
                datetimes at which to track particles.
                If `None`, defaults to all unique datetimes in `self.observers`.
            maxdt (float): Maximum time delta for an image to match `datetimes`
            tile_size (array-like): Size of reference tiles in pixels (width, height)
        """
        if datetimes is None:
            datetimes = np.sort(np.hstack((obs.datetimes for obs in self.observers)))
        datetimes = np.unique(datetimes)
        # Datetimes must be stricly increasing
        # NOTE: Not necessary if default datetimes
        time_deltas = np.array([dt.total_seconds() for dt in np.diff(datetimes)])
        if any(time_deltas < 0):
            raise ValueError('Datetimes are not monotonically increasing')
        # Drop datetimes not matching any Observer
        # NOTE: Not necessary if default datetimes
        nearest = []
        for observer in self.observers:
            indices = np.searchsorted(observer.datetimes, datetimes, side='left')
            indices = np.where(indices < 0, 1, indices)
            n = len(observer.datetimes)
            indices = np.where(indices > n - 1, n - 1, indices)
            nearest.append(np.column_stack((
                observer.datetimes[indices - 1],
                observer.datetimes[indices])))
        dt = abs(np.hstack(nearest) - datetimes.reshape(-1, 1))
        has_match = dt <= datetime.timedelta(seconds=maxdt * self.time_unit)
        has_any_match = np.any(has_match, axis=1)
        if not all(has_any_match):
            warnings.warn('Dropping datetimes not matching any Observers')
            datetimes = datetimes[has_any_match]
            has_match = has_match[has_any_match, :]
        if len(datetimes) < 2:
            raise ValueError('Fewer than two valid datetimes')
        self.datetimes = datetimes
        # Initialize each Observer at first matching datetime
        self.tiles = [None] * len(self.observers)
        self.histograms = [None] * len(self.observers)
        self.duvs = [None] * len(self.observers)
        center_xyz = self.particle_mean[:, 0:3]
        for i, observer in enumerate(self.observers):
            # NOTE: May be faster to retrieve index of match in earlier loop
            try:
                ti = np.nonzero(np.any(has_match[:, (i * 2):(i * 2) + 2], axis=1))[0][0]
            except IndexError:
                # Skip Observer if no matching image
                continue
            img = observer.index(self.datetimes[ti], max_seconds=maxdt * self.time_unit)
            center_uv = observer.project(center_xyz, img=img)
            box = observer.tile_box(center_uv, size=tile_size)
            self.duvs[i] = center_uv[0] - box.reshape(2, -1).mean(axis=0)
            self.tiles[i], self.histograms[i] = self._prepare_tile_histogram(
                obs=i, img=img, box=box)

    def _prepare_tile_histogram(self, obs, img, box):
        tile = self.observers[obs].extract_tile(box=box, img=img)
        if tile.ndim > 2:
            tile = helpers.rgb_to_gray(tile, **self.grayscale)
        tile = helpers.normalize(tile)
        histogram = helpers.compute_cdf(tile, return_inverse=False)
        tile_low = scipy.ndimage.filters.median_filter(tile, **self.highpass)
        tile -= tile_low
        return tile, histogram

    def _prepare_test_tile(self, obs, img, box, template):
        tile = self.observers[obs].extract_tile(box=box, img=img)
        if tile.ndim > 2:
            tile = helpers.rgb_to_gray(tile, **self.grayscale)
        tile = helpers.normalize(tile)
        tile = helpers.match_histogram(tile, template=template)
        tile_low = scipy.ndimage.filters.median_filter(tile, **self.highpass)
        tile -= tile_low
        return tile

    def compute_likelihoods(self, t, maxdt=0):
        """
        Compute the likelihoods of the particles summed across all observers.

        Arguments:
            t (datetime): Date and time at which to query Observers
        """
        log_likelihoods = [self._compute_observer_log_likelihoods(observer, t, maxdt)
            for observer in self.observers]
        return np.exp(-sum(log_likelihoods))

    def _compute_observer_log_likelihoods(self, observer, t, maxdt=0):
        """
        Compute the log likelihoods of each particle for an Observer.

        Arguments:
            t (datetime): Date and time at which to query Observer
            obsberver (Observer): Observer object
        """
        i = self.observers.index(observer)
        # Select image
        try:
            img = observer.index(t, max_seconds=maxdt * self.time_unit)
        except IndexError:
            # If no image, return a constant log likelihood
            return np.array([0.0])
        # Build image box around all particles, with a buffer for template matching
        uv = observer.project(self.particles[:, 0:3], img=img)
        halfsize = np.multiply(self.tiles[i].shape[0:2][::-1], 0.5)
        box = np.vstack((
            uv.min(axis=0) - halfsize,
            uv.max(axis=0) + halfsize))
        # Enlarge box to ensure SSE has cols, rows (ky + 1, kx + 1) for interpolation
        ky = self.interpolation.get('ky', 3)
        ncols = ky - (np.diff(box[:, 0]) - self.tiles[i].shape[1])
        if ncols > 0:
            # Widen box in 2nd ('y') dimension (x|cols)
            box[:, 0] += np.hstack((-ncols, ncols)) / 2
        kx = self.interpolation.get('kx', 3)
        nrows = kx - (np.diff(box[:, 1]) - self.tiles[i].shape[0])
        if nrows > 0:
            # Widen box in 1st ('x') dimension (y|rows)
            box[:, 1] += np.hstack((-nrows, nrows)) / 2
        box = np.vstack((np.floor(box[0, :]), np.ceil(box[1, :]))).astype(int)
        # Check that box is within image bounds
        if not all(observer.grid.inbounds(box)):
            raise IndexError('Particle bounding box extends past image bounds')
        # Flatten box
        box = box.ravel()
        # Extract test tile
        test_tile = self._prepare_test_tile(
            obs=i, img=img, box=box, template=self.histograms[i])
        # Compute area-averaged sum of squares error (SSE)
        sse = cv2.matchTemplate(test_tile.astype(np.float32),
            templ=self.tiles[i].astype(np.float32), method=cv2.TM_SQDIFF)
        sse *= 1.0 / (self.tiles[i].shape[0] * self.tiles[i].shape[1])
        # Compute SSE bounding box
        # (relative to test tile: shrunk by halfsize of reference tile - 0.5 pixel)
        box_edge = halfsize - 0.5
        sse_box = box + np.hstack((box_edge, -box_edge))
        # (shift by subpixel offset of reference tile)
        sse_box += np.tile(self.duvs[i], 2)
        # Sample at projected particles
        sampled_sse = observer.sample_tile(uv, tile=sse,
            box=sse_box, grid=False, **self.interpolation)
        return sampled_sse * (1.0 / observer.sigma**2)
