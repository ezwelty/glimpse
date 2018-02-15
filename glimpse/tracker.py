from .imports import (np, cv2)
from . import (helpers)

class Tracker(object):
    """
    A `Tracker' estimates the trajectory of world points through time.

    Attributes:
        observers (list): Observer objects
        dem (DEM): Digital elevation model of the surface on which to track points
        time_unit (float): Length of time unit for particle evolution arguments, in seconds
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
    def __init__(self, observers, dem, tile_size=(15, 15), time_unit=1):
        self.observers = observers
        self.dem = dem
        self.tile_size = tile_size
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

    @property
    def n(self):
        return len(self.particles)

    @property
    def particle_mean(self):
        return np.average(self.particles, weights=self.weights, axis=0).reshape(1, -1)

    @property
    def particle_covariance(self):
        return np.cov(self.particles.T, aweights=self.weights)

    # TODO: initialized property

    def initialize(self, n, xy, xy_sigma, vxy=(0, 0), vxy_sigma=(0, 0)):
        """
        Initialize particles given an initial normal distribution.

        Temporal arguments (`vxy`, `vxy_sigma`) are assumed to be in
        `self.time_unit` time units.
        Reference image tiles are loaded from each observer centered around
        the weighted mean of the particles positions (`self.tiles` and `self.histograms`).

        Arguments:
            n (int): Number of particles
            xy (array-like): Mean position (x, y)
            xy_sigma (array-like): Standard deviation of positions (x, y)
            vxy (array-like): Mean velocity (x, y)
            vxy_sigma (array-like): Standard deviation of velocities (x, y)
        """
        # TODO: Tile operations to properties: Observer or Tracker?
        # TODO: Initialize observers based on start time?
        # Initialize particles with equal weights
        if self.particles is None or self.n != n:
            self.particles = np.zeros((n, 5))
        self.particles[:, 0:2] = xy + xy_sigma * np.random.randn(n, 2)
        self.particles[:, 2] = self.dem.sample(self.particles[:, 0:2])
        self.particles[:, 3:5] = vxy + vxy_sigma * np.random.randn(n, 2)
        self.weights = np.ones(n).astype(float) / n
        # Initialize reference tiles and histograms
        center_xyz = self.particle_mean[:, 0:3]
        self.tiles = []
        self.histograms = []
        for i, observer in enumerate(self.observers):
            center_uv = observer.project(center_xyz)
            box = observer.tile_box(center_uv, size=self.tile_size)
            self.tiles.append(observer.extract_tile(
                box, gray=dict(method='pca'), highpass=dict(size=(5, 5)),
                subpixel=dict(kx=3, ky=3), uv=center_uv))
            self.histograms.append(helpers.compute_cdf(
                self.tiles[-1], return_inverse=False))

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
        self.weights.fill(1)
        self.weights *= likelihoods
        self.weights += 1e-300
        self.weights *= 1 / self.weights.sum()

    def resample_particles(self):
        """
        Prune unlikely particles and reproduce likely ones.
        """
        relative_position = (np.arange(self.n) + np.random.random()) / self.n
        cumulative_weight = np.cumsum(self.weights)
        indexes = np.zeros(self.n, dtype=int)
        i, j = 0, 0
        while i < self.n:
            if relative_position[i] < cumulative_weight[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights *= 1 / self.weights.sum()

    def track(self, datetimes=None, axy=(0, 0), axy_sigma=(0, 0), plot=False):
        """
        Track particles through time.

        Temporal arguments (`axy`, `axy_sigma`) are assumed to be in
        `self.time_unit` time units.
        Results are saved as
        `self.datetimes`, `self.means`, and `self.covariances`.

        Arguments:
            datetimes (array-like): Datetime.datetime objects at which to track particles.
                If `None`, defaults to all unique datetimes in `self.observers`.
            axy (array-like): Mean of random accelerations (x, y)
            axy_sigma (array-like) Standard deviation of random accelerations (x, y)
            plot (bool): Whether to plot results in realtime
        """
        if self.particles is None:
            raise Exception("Particles are not initialized")
        if datetimes:
            self.datetimes = datetimes
        else:
            obs_datetimes = np.hstack((obs.datetimes for obs in self.observers))
            self.datetimes = np.sort(np.unique(obs_datetimes))
        self.means = [self.particle_mean]
        self.covariances = [self.particle_covariance]
        if plot:
            self.initialize_plot()
        delta_times = np.array([dt.total_seconds() for dt in np.diff(self.datetimes)])
        delta_times /= self.time_unit
        for t, dt in zip(self.datetimes[1:], delta_times):
            self.advance_particles(dt=dt, axy=axy, axy_sigma=axy_sigma)
            likelihoods = self.compute_likelihoods(t)
            self.update_weights(likelihoods)
            self.resample_particles()
            self.means.append(self.particle_mean)
            self.covariances.append(self.particle_covariance)
            if plot:
                self.update_plot(t)

    def compute_likelihoods(self, t):
        """
        Compute the likelihoods of the particles summed across all observers.

        Arguments:
            t (datetime): Date and time at which to query Observers
        """
        log_likelihoods = [self.compute_observer_log_likelihoods(t, observer)
            for observer in self.observers]
        return np.exp(-sum(log_likelihoods))

    def compute_observer_log_likelihoods(self, t, obs=0):
        """
        Compute the log likelihoods of each particle for an Observer.

        Arguments:
            t (datetime): Date and time at which to query Observer
            obs: Either index of Observer or Observer object
        """
        # Select observer
        if isinstance(obs, int):
            observer = self.observers[obs]
            i = obs
        else:
            i = self.observers.index(obs)
            observer = obs
        # Select image
        try:
            img = observer.index(t)
        except IndexError:
            # If no image, return a constant log likelihood
            return np.array([0.0])
        # Build image box around all particles, with a buffer for template matching
        # -1, +1 adjustment ensures SSE size > 3 pixels (5+) for cubic spline interpolation
        # TODO: Adjust minimum size based on interpolation order
        uv = observer.project(self.particles[:, 0:3])
        halfsize = np.multiply(self.tiles[i].shape[-2::-1], 0.5)
        box = np.vstack((
            np.floor(uv.min(axis=0) - halfsize) - 1,
            np.ceil(uv.max(axis=0) + halfsize) + 1)).astype(int)
        if any(~observer.grid.inbounds(box)):
            # Tile extends beyond image bounds
            raise IndexError("Particle bounding box extends beyond image bounds: " + str(box))
        # Extract test tile
        test_tile = observer.extract_tile(box=box.flatten(), template=self.histograms[i],
            gray=dict(method='pca'), highpass=dict(size=(5, 5)))
        # Compute area-averaged sum of squares error
        sse = cv2.matchTemplate(test_tile.astype(np.float32),
            templ=self.tiles[i].astype(np.float32), method=cv2.TM_SQDIFF)
        sse *= 1.0 / (self.tiles[i].shape[0] * self.tiles[i].shape[1])
        # Sample at projected particles
        # SSE tile box is shrunk by halfsize of reference tile - 0.5 pixel
        box_edge = halfsize - 0.5
        sse_box = box + np.vstack((box_edge, -box_edge))
        sampled_sse = observer.sample_tile(uv, tile=sse,
            box=sse_box.flatten(), grid=False, **dict(kx=3, ky=3))
        return sampled_sse * (1.0 / observer.sigma**2)

    def initialize_plot(self):
        """
        Initialize animation plot.

        Warning: Do not use with multiprocessing!
        """
        matplotlib.pyplot.ion()
        nplots = 1 + len(self.observers)
        self.fig, self.ax = matplotlib.pyplot.subplots(
            nrows=1, ncols=nplots, figsize=(10 * nplots, 10))
        self.fig.tight_layout()
        #self.ax[0].contourf(self.dem.X, self.dem.Y, self.dem.hillshade(), 31, cmap=matplotlib.pyplot.cm.gray)
        self.meplot = self.ax[0].scatter(
            self.means[0][0, 0], self.means[0][0, 1],
            c='red', s=50, label='Mean position')
        v = np.hypot(self.particles[:, 3], self.particles[:, 4])
        self.pa_plot = self.ax[0].quiver(
            self.particles[:, 0], self.particles[:, 1],
            self.particles[:, 3] / v, self.particles[:, 4] / v, v,
            cmap=matplotlib.pyplot.cm.gnuplot2, clim=(0, 15), alpha=0.2, linewidths=0)
        self.ax[0].legend()
        self.cb = matplotlib.pyplot.colorbar(self.pa_plot, ax=self.ax[0],
            orientation='horizontal', aspect=30, pad=0.07)
        self.cb.set_label('Speed')
        self.cb.solids.set_edgecolor('face')
        self.cb.solids.set_alpha(1)
        self.ax[0].set_xlabel('X')
        self.ax[0].set_ylabel('Y')
        self.ax[0].axis('equal')
        self.ax[0].set_xlim(self.means[0][0, 0] - 50, self.means[0][0, 0] + 50)
        self.ax[0].set_ylim(self.means[0][0, 1] - 50, self.means[0][0, 1] + 50)
        for axis, observer in zip(self.ax[1:], self.observers):
            observer.initialize_plot(axis)
        matplotlib.pyplot.pause(2.0)

    def update_plot(self, t):
        """
        Update animation plot.

        Arguments:
            t (datetime): Date and time
        """
        self.meplot.remove()
        self.pa_plot.remove()
        self.meplot = self.ax[0].scatter(
            [m.squeeze()[0] for m in self.means],
            [m.squeeze()[1] for m in self.means], s=50, c='red')
        v = np.hypot(self.particles[:, 3], self.particles[:, 4])
        self.pa_plot = self.ax[0].quiver(
            self.particles[:, 0], self.particles[:, 1],
            self.particles[:, 3] / v, self.particles[:, 4] / v, v,
            scale=50, cmap=matplotlib.pyplot.cm.gnuplot2, clim=[0, 15], alpha=0.2)
        xmed = np.median(self.particles[:, 0])
        ymed = np.median(self.particles[:, 1])
        self.ax[0].set_xlim(xmed - 50, xmed + 50)
        self.ax[0].set_ylim(ymed - 50, ymed + 50)
        for observer in self.observers:
            observer.update_plot(t)
        #self.fig.savefig('./particle_tracker_imgs/images_{0:03d}.jpg'.format(self.counter), bbox_inches='tight', dpi=300)
        matplotlib.pyplot.pause(0.00001)
