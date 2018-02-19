from .imports import (np, cv2, warnings, datetime, matplotlib)
from . import (helpers)

class Tracker(object):
    """
    A `Tracker' estimates the trajectory of world points through time.

    Attributes:
        observers (list): Observer objects
        dem (DEM): Digital elevation model of the surface on which to track points
        time_unit (float): Length of time unit for temporal arguments, in seconds
            (e.g., 1 minute = 60, 1 hour = 3600)
        n (int): Number of particles (more particles gives better results at higher expense)
        particles (array): Positions and velocities of particles (n, 5) [[x, y, z, vx, vy], ...]
        weights (array): Particle likelihoods (n, )
        particle_mean (array): Weighted mean of `particles` (1, 5) [[x, y, z, vx, vy]]
        particle_covariance (array): Weighted covariance matrix of `particles` (5, 5)
        datetimes (array): Date and times at which particle positions were estimated.
        means (list): `particle_mean` at each `datetimes`
        covariances (list): `particle_covariance` at each `datetimes`
    """
    def __init__(self, observers, dem, time_unit=1):
        self.observers = observers
        self.dem = dem
        self.time_unit = time_unit
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
        # Plotting only
        self.test_tiles = None

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
        self.particles[:, 2] = self.dem.sample(self.particles[:, 0:2])
        self.particles[:, 3:5] += dt * (axy + daxy)

    def update_weights(self, likelihoods):
        """
        Update particle weights based on their likelihoods.

        Arguments:
            likelihoods (array): Likelihood of each particle
        """
        # TODO: Make faster?
        self.weights.fill(1)
        self.weights *= likelihoods
        self.weights += 1e-300
        self.weights *= 1 / self.weights.sum()

    def resample_particles(self):
        """
        Prune unlikely particles and reproduce likely ones.
        """
        # Systematic resample
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def f():
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
        # Residual resample (vectorized)
        # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
        def f1a():
            repetitions = (self.n * self.weights).astype(int)
            initial_indexes = np.repeat(np.arange(self.n), repetitions)
            residuals = self.weights - repetitions
            residuals *= 1 / residuals.sum()
            cumulative_sum = np.cumsum(residuals)
            cumulative_sum[-1] = 1.0
            additional_indexes = np.searchsorted(
                cumulative_sum, np.random.random(self.n - len(initial_indexes)))
            return np.hstack((initial_indexes, additional_indexes))
        def f1b():
            repetitions = (self.n * self.weights).astype(int)
            initial_indexes = np.repeat(np.arange(self.n), repetitions)
            residuals = self.weights - repetitions
            residuals += residuals.min()
            residuals *= 1 / residuals.sum()
            additional_indexes = np.random.choice(
                np.arange(self.n), size=(self.n - len(initial_indexes), ), replace=True, p=residuals)
            return np.hstack((initial_indexes, additional_indexes))
        # Random sample
        def f3():
            return np.random.choice(np.arange(self.n), size=(self.n, ), replace=True, p=self.weights)
        indexes = f()
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights *= 1 / self.weights.sum()

    def track(self, datetimes=None, maxdt=0, tile_size=(15, 15),
        axy=(0, 0), axy_sigma=(0, 0), plot=False):
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
            plot (bool): Whether to plot results in realtime
        """
        if self.particles is None:
            raise Exception("Particles are not initialized")
        self._initialize_track(datetimes=datetimes, maxdt=maxdt, tile_size=tile_size)
        self.means = [self.particle_mean]
        self.covariances = [self.particle_covariance]
        if plot:
            self.initialize_plot()
            matplotlib.pyplot.pause(2)
        time_deltas = np.array([dt.total_seconds() for dt in np.diff(self.datetimes)])
        time_deltas *= 1.0 / self.time_unit
        for t, dt in zip(self.datetimes[1:], time_deltas):
            self.test_tiles = [None] * len(self.observers)
            self.advance_particles(dt=dt, axy=axy, axy_sigma=axy_sigma)
            likelihoods = self.compute_likelihoods(t, maxdt=maxdt)
            self.update_weights(likelihoods)
            self.resample_particles()
            self.means.append(self.particle_mean)
            self.covariances.append(self.particle_covariance)
            if plot:
                self.update_plot()
                matplotlib.pyplot.pause(1)

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
            raise ValueError("Datetimes are not monotonically increasing")
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
            warnings.warn("Dropping datetimes not matching any Observers")
            datetimes = datetimes[has_any_match]
            has_match = has_match[has_any_match, :]
        if len(datetimes) < 2:
            raise ValueError("Fewer than two valid datetimes")
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
            # TODO: Process outside Observer.extract_tile
            self.tiles[i] = observer.extract_tile(
                box, img=img, gray=dict(method='pca'), highpass=dict(size=(5, 5)),
                subpixel=False, uv=center_uv)
            temp = np.atleast_3d(observer.extract_tile(box, img=img))
            self.histograms[i] = [helpers.compute_cdf(temp[:, :, b], return_inverse=False)
                for b in range(temp.shape[2])]

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
            # TODO: Control
            img = observer.index(t, max_seconds=maxdt * self.time_unit)
        except IndexError:
            # If no image, return a constant log likelihood
            return np.array([0.0])
        # Build image box around all particles, with a buffer for template matching
        # -1, +1 adjustment ensures SSE size > 3 pixels (5+) for cubic spline interpolation
        # TODO: Adjust minimum size based on interpolation order
        uv = observer.project(self.particles[:, 0:3], img=img)
        halfsize = np.multiply(self.tiles[i].shape[0:2][::-1], 0.5)
        box = np.vstack((
            np.floor(uv.min(axis=0) - halfsize) - 1,
            np.ceil(uv.max(axis=0) + halfsize) + 1)).astype(int).flatten()
        # TEMP: Test if even tile size is the problem
        # center_uv = observer.project(self.particle_mean[:, 0:3], img=img)
        # box = observer.tile_box(center_uv, size=(71, 71))
        if any(~observer.grid.inbounds(box.reshape(2, -2))):
            # Tile extends beyond image bounds
            raise IndexError("Particle bounding box extends beyond image bounds: " + str(box))
        # Extract test tile
        self.test_tiles[i] = observer.extract_tile(box=box, template=self.histograms[i],
            img=img, gray=dict(method='pca'), highpass=dict(size=(5, 5)), subpixel=False)
        # Compute area-averaged sum of squares error
        sse = cv2.matchTemplate(self.test_tiles[i].astype(np.float32),
            templ=self.tiles[i].astype(np.float32), method=cv2.TM_SQDIFF)
        sse *= 1.0 / (self.tiles[i].shape[0] * self.tiles[i].shape[1])
        # Sample at projected particles
        # SSE tile box is shrunk by halfsize of reference tile - 0.5 pixel
        box_edge = halfsize - 0.5
        sse_box = box + np.hstack((box_edge, -box_edge))
        # Shift box by subpixel offset
        sse_box += np.tile(self.duvs[i], 2)
        sampled_sse = observer.sample_tile(uv, tile=sse,
            box=sse_box, grid=False, **dict(kx=3, ky=3))
        return sampled_sse * (1.0 / observer.sigma**2)

    def initialize_plot(self):
        # useful plot:
        # map of mean position and particles
        # test_tile with reference tile as box
        # particles projected
        """
        Initialize animation plot.
        Warning: Do not use with multiprocessing!
        """
        matplotlib.pyplot.ion()
        nplots = len(self.observers) + 1
        self.fig, self.ax = matplotlib.pyplot.subplots(
            nrows=1, ncols=nplots, figsize=(10 * nplots, 10))
        self.fig.tight_layout()
        self.meplot = self.ax[0].scatter(
            self.means[0][0, 0], self.means[0][0, 1],
            color='red', s=50, label='Mean position')
        v = np.hypot(self.particles[:, 3], self.particles[:, 4])
        self.pa_plot = self.ax[0].quiver(
            self.particles[:, 0], self.particles[:, 1],
            self.particles[:, 3] / v, self.particles[:, 4] / v, v,
            cmap=matplotlib.pyplot.cm.gnuplot2, clim=(0, 15), alpha=0.2, linewidths=0)
        self.ax[0].legend()
        self.ax[0].set_xlabel('X')
        self.ax[0].set_ylabel('Y')
        self.ax[0].axis('equal')
        self.ax[0].set_xlim(self.means[0][0, 0] - 50, self.means[0][0, 0] + 50)
        self.ax[0].set_ylim(self.means[0][0, 1] - 50, self.means[0][0, 1] + 50)
        self.cb = matplotlib.pyplot.colorbar(self.pa_plot, ax=self.ax[0],
            orientation='horizontal', aspect=30, pad=0.07)
        self.cb.set_label('Speed')
        self.cb.solids.set_edgecolor('face')
        self.cb.solids.set_alpha(1)
        # for i, tile in enumerate(self.tiles):
        #     if tile is not None:
        #         self.ax[i + 1].imshow(tile, cmap='Greys_r')

    def update_plot(self):
        """
        Update animation plot.
        """
        self.meplot.remove()
        means = np.vstack(self.means)
        self.meplot = self.ax[0].scatter(means[:, 0], means[:, 1], s=50, c='red')
        v = np.hypot(self.particles[:, 3], self.particles[:, 4])
        self.pa_plot.remove()
        self.pa_plot = self.ax[0].quiver(
            self.particles[:, 0], self.particles[:, 1],
            self.particles[:, 3] / v, self.particles[:, 4] / v, v,
            scale=50, cmap=matplotlib.pyplot.cm.gnuplot2, clim=[0, 15], alpha=0.2)
        self.ax[0].set_xlim(means[-1, 0] - 50, means[-1, 0] + 50)
        self.ax[0].set_ylim(means[-1, 1] - 50, means[-1, 1] + 50)
        for i, tile in enumerate(self.test_tiles):
            if tile is not None:
                self.ax[i + 1].imshow(tile, cmap='Greys_r')
