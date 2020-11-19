"""Track particle trajectories through time."""
import datetime
import sys
import traceback
import warnings
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import scipy.ndimage
from typing_extensions import Literal

from .. import config, helpers
from ..raster import Raster
from .motion import Motion
from .observer import Observer
from .tracks import Tracks

Number = Union[int, float]


class Tracker:
    """
    Estimate the trajectory of world points through time.

    Attributes:
        observers (iterable): Observers.
        viewshed (Raster): Binary viewshed.
        resample_method (str): Particle resampling method.
            See :meth:`resample_particles`.
        highpass (dict): Median high-pass filter
            (arguments to :meth:`scipy.ndimage.filters.median_filter`).
        interpolation (dict): Subpixel interpolation
            (arguments to :class:`scipy.interpolate.RectBivariateSpline`).
        particles (numpy.ndarray): Positions and velocities of particles (n, 6)
            [(x, y, z, vx, vy vz), ...].
        weights (numpy.ndarray): Particle likelihoods (n,).
        particle_mean (numpy.ndarray): Weighted mean of `particles` (6,)
            [x, y, z, vx, vy, vz].
        particle_covariance (numpy.ndarray): Weighted covariance matrix of `particles`
            (6, 6).
        templates (list): For each Observer, a template extracted from the first image
            centered around the `particle_mean` at the time of image capture.
            Templates are dictionaries that include at least:

            - 'tile': Image tile used as a template for image cross-correlation.
            - 'histogram': Histogram (values, quantiles) of the 'tile' used for
              histogram matching.
            - 'duv': Subpixel offset of 'tile' (desired - sampled).
    """

    def __init__(
        self,
        observers: Iterable[Observer],
        viewshed: Raster = None,
        resample_method: Literal[
            "systematic", "stratified", "residual", "choice"
        ] = "systematic",
        highpass: dict = {"size": (5, 5)},
        interpolation: dict = {"kx": 3, "ky": 3},
    ) -> None:
        self.observers = observers
        self.viewshed = viewshed
        self.resample_method = resample_method
        self.highpass = highpass
        self.interpolation = interpolation
        # Placeholders
        self.particles = None
        self.weights = None
        self.templates = None

    @property
    def particle_mean(self) -> np.ndarray:
        """Weighted particle mean [x, y, z, vx, vy, vz]."""
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def particle_covariance(self) -> np.ndarray:
        """Weighted (biased) particle covariance matrix (6, 6)."""
        return np.cov(self.particles.T, aweights=self.weights, ddof=0)

    @property
    def datetimes(self) -> np.ndarray:
        """Sorted list of unique observation datetimes."""
        return np.unique(np.concatenate([obs.datetimes for obs in self.observers]))

    def compute_particle_sigma(self, mean: Iterable[Number] = None) -> np.ndarray:
        """
        Return the weighted particle standard deviation.

        Works faster by using a precomputed weighted particle mean.

        Arguments:
            mean: Weighted particle mean. By default, uses :attr:`particle_mean`.
        """
        if mean is None:
            mean = self.particle_mean
        variance = np.average(
            (self.particles - mean) ** 2, weights=self.weights, axis=0
        )
        return np.sqrt(variance)

    def test_particles(self) -> None:
        """
        Test particle validity.

        Raises:
            ValueError: Some particles on non-visible viewshed cells.
            ValueError: Some particles have missing (NaN) values.
        """
        if self.viewshed is not None:
            is_visible = self.viewshed.sample(self.particles[:, 0:2], order=0)
            if not all(is_visible):
                raise ValueError("Some particles are on non-visible viewshed cells")
        if np.isnan(self.particles).any():
            raise ValueError("Some particles have missing (NaN) values")

    def initialize_weights(self) -> None:
        """Initialize particle weights."""
        n = len(self.particles)
        self.weights = np.full(n, 1 / n)

    def update_weights(
        self, imgs: Iterable[Optional[int]], motion_model: Motion = None
    ) -> None:
        """
        Update particle weights.

        Particle log likelihoods are summed across all Observers and,
        optionally, a motion model.

        Arguments:
            imgs: For each Observer, either the Image index or `None` to skip.
            motion_model: Motion model.
        """
        log_likelihoods = [
            self.compute_observer_log_likelihoods(obs, img)
            for obs, img in enumerate(imgs)
        ]
        if motion_model:
            log_likelihoods.append(motion_model.compute_log_likelihoods(self.particles))
        # Remove empty elements
        log_likelihoods = [x for x in log_likelihoods if x is not None]
        likelihoods = np.exp(-sum(log_likelihoods))
        self.weights = likelihoods + 1e-300
        self.weights *= 1 / self.weights.sum()

    def resample_particles(
        self, method: Literal["systematic", "stratified", "residual", "choice"] = None
    ) -> None:
        """
        Prune unlikely particles and reproduce likely ones.

        Arguments:
            method: Resampling method. By default, uses :attr:`resample_method`.

                - 'systematic': Systematic resampling.
                - 'stratified': Stratified resampling.
                - 'residual': Residual resampling
                  (Liu & Chen 1998: https://doi.org/10.1080/01621459.1998.10473765).
                - 'choice': Random choice with replacement
                  (:func:`numpy.random.choice` with `replace=True`).
        """
        n = len(self.particles)

        def systematic() -> np.ndarray:
            """Systematic resampling."""
            # Vectorized version of FilterPy
            # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
            positions = (np.arange(n) + np.random.random()) * (1 / n)
            cumulative_weight = np.cumsum(self.weights)
            return np.searchsorted(cumulative_weight, positions)

        def stratified() -> np.ndarray:
            """Stratified resampling."""
            # Vectorized version of FilterPy
            # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
            positions = (np.arange(n) + np.random.random(n)) * (1 / n)
            cumulative_weight = np.cumsum(self.weights)
            return np.searchsorted(cumulative_weight, positions)

        def residual() -> np.ndarray:
            """Residual resampling."""
            # Vectorized version of FilterPy
            # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
            repetitions = (n * self.weights).astype(int)
            initial_indexes = np.repeat(np.arange(n), repetitions)
            residuals = self.weights - repetitions
            residuals *= 1 / residuals.sum()
            cumulative_sum = np.cumsum(residuals)
            cumulative_sum[-1] = 1.0
            additional_indexes = np.searchsorted(
                cumulative_sum, np.random.random(n - len(initial_indexes))
            )
            return np.hstack((initial_indexes, additional_indexes))

        def choice() -> np.ndarray:
            """Random choice with replacement."""
            return np.random.choice(
                np.arange(n), size=(n,), replace=True, p=self.weights
            )

        if method is None:
            method = self.resample_method
        if method == "systematic":
            indexes = systematic()
        elif method == "stratified":
            indexes = stratified()
        elif method == "residual":
            indexes = residual()
        elif method == "choice":
            indexes = choice()
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights *= 1 / self.weights.sum()

    def track(
        self,
        motion_models: Iterable[Motion],
        datetimes: Iterable[datetime.datetime] = None,
        maxdt: datetime.timedelta = datetime.timedelta(0),
        tile_size: Iterable[int] = (15, 15),
        observer_mask: np.ndarray = None,
        return_covariances: bool = False,
        return_particles: bool = False,
        parallel: Union[bool, int] = False,
    ) -> Tracks:
        """
        Track particles through time.

        If more than one motion models are passed (`motion_models`),
        errors and warnings are caught silently,
        and any images from Observers with :attr:`cache``=True` are cached.

        Arguments:
            motion_models: Motion models specifying which particles to track.
            datetimes: Monotonic sequence of datetimes at which to track particles.
                Defaults to :attr:`datetimes`.
            maxdt: Maximum timedelta for an image to match `datetimes`.
            tile_size: Size of reference tiles in pixels (width, height).
            observer_mask: Boolean mask of Observers to use for each `motion_models`
                (n motion models, m observers).
                By default, uses all Observers for all motion models.
            return_covariances: Whether to return particle covariance
                matrices or just particle standard deviations.
            return_particles: Whether to return all particles and weights
                at each timestep.
            parallel: Number of motion models to track in parallel (int),
                or whether to track in parallel (bool). If `True`,
                defaults to :func:`os.cpu_count`.

        Returns:
            Motion tracks.
        """
        # Save original function arguments (stored in result)
        # NOTE: Must be called first
        params = locals().copy()
        time_unit = motion_models[0].time_unit
        for model in motion_models[1:]:
            if model.time_unit != time_unit:
                raise ValueError("Motion models must have equal time units")
        # Clear previous tracking state
        self.reset()
        # Enforce defaults
        ntracks = len(motion_models)
        errors = ntracks < 2
        parallel = helpers._parse_parallel(parallel)
        if datetimes is None:
            datetimes = self.datetimes
        else:
            datetimes = self.parse_datetimes(datetimes=datetimes, maxdt=maxdt)
        if observer_mask is None:
            observer_mask = np.ones((ntracks, len(self.observers)), dtype=bool)
        # Compute matching images
        matching_images = self.match_datetimes(datetimes=datetimes, maxdt=maxdt)
        template_indices = np.not_equal(matching_images, None).argmax(axis=0)
        # Cache matching images
        if ntracks > 1:
            for i, observer in enumerate(self.observers):
                if observer.cache:
                    index = [img for img in matching_images[:, i] if img is not None]
                    observer.cache_images(index=index)
        # Define parallel process
        bar = helpers._progress_bar(max=ntracks)
        ntimes = len(datetimes)
        dts = np.diff(datetimes)

        def process(motion_model: Motion, observer_mask: np.ndarray) -> list:
            means = np.full((ntimes, 6), np.nan)
            if return_covariances:
                sigmas = np.full((ntimes, 6, 6), np.nan)
            else:
                sigmas = np.full((ntimes, 6), np.nan)
            if return_particles:
                particles = np.full((ntimes, motion_model.n, 6), np.nan)
                weights = np.full((ntimes, motion_model.n), np.nan)
            error = None
            all_warnings = None
            try:
                with warnings.catch_warnings(record=True) as caught:
                    # Skip datetimes before first and after last available image
                    # NOTE: Track thus starts from initial particle state at first
                    # available image
                    observed = np.not_equal(
                        matching_images[:, observer_mask], None
                    ).any(axis=1)
                    first = np.argmax(observed)
                    last = len(observed) - 1 - np.argmax(observed[::-1])
                    for i in range(first, last + 1):
                        if i == first:
                            self.particles = motion_model.initialize_particles()
                            self.test_particles()
                            self.initialize_weights()
                        else:
                            dt = dts[i - 1]
                            motion_model.evolve_particles(self.particles, dt=dt)
                            self.test_particles()
                        # Initialize templates for Observers starting at datetimes[i]
                        at_template = observer_mask & (template_indices == i)
                        for obs in np.nonzero(at_template)[0]:
                            self.initialize_template(
                                obs=obs,
                                img=matching_images[i][obs],
                                tile_size=tile_size,
                            )
                        if i > first:
                            imgs = [
                                img if m else None
                                for img, m in zip(matching_images[i], observer_mask)
                            ]
                            self.update_weights(imgs=imgs, motion_model=motion_model)
                            self.resample_particles()
                        means[i] = self.particle_mean
                        if return_covariances:
                            sigmas[i] = self.particle_covariance
                        else:
                            sigmas[i] = self.compute_particle_sigma(mean=means[i])
                        if return_particles:
                            particles[i] = self.particles
                            weights[i] = self.weights
                if caught:
                    all_warnings = tuple(caught)
            except Exception as e:
                # traceback object cannot be pickled, so include in message
                # TODO: Use tblib instead (https://stackoverflow.com/a/26096355)
                if errors:
                    raise e
                elif parallel:
                    error = e.__class__(
                        "".join(traceback.format_exception(*sys.exc_info()))
                    )
                else:
                    error = e
            results = [means, sigmas, error, all_warnings]
            if return_particles:
                results += [particles, weights]
            return results

        def reduce(results: list) -> list:
            bar.next()
            return results

        # Run process in parallel
        with config.backend(np=parallel) as pool:
            results = pool.map(
                func=process,
                reduce=reduce,
                star=True,
                sequence=tuple(zip(motion_models, observer_mask)),
            )
        bar.finish()
        # Return results as Tracks
        if return_particles:
            means, sigmas, errors, all_warnings, particles, weights = zip(*results)
        else:
            means, sigmas, errors, all_warnings = zip(*results)
            particles, weights = None, None
        kwargs = {
            "time_unit": time_unit,
            "datetimes": datetimes,
            "means": means,
            "particles": particles,
            "weights": weights,
            "tracker": self,
            "images": matching_images,
            "params": params,
            "errors": errors,
            "warnings": all_warnings,
        }
        if return_covariances:
            kwargs["covariances"] = sigmas
        else:
            kwargs["sigmas"] = sigmas
        return Tracks(**kwargs)

    def reset(self) -> None:
        """Reset to initial state."""
        self.particles = None
        self.weights = None
        self.templates = None

    def parse_datetimes(
        self,
        datetimes: Iterable[datetime.datetime],
        maxdt: datetime.timedelta = datetime.timedelta(0),
    ) -> np.ndarray:
        """
        Parse datetimes for tracking.

        Datetimes are tested to be monotonic, duplicates are dropped,
        and those not matching any Observers are dropped.

        Arguments:
            datetimes: Monotonically-increasing datetimes at which to track particles.
            maxdt: Maximum timedelta for an image to match `datetimes`.

        Raises:
            ValueError: Datetimes must be monotonic.
            ValueError: Fewer than two valid datetimes.
        """
        datetimes = np.asarray(datetimes)
        # Datetimes must be monotonic
        monotonic = (datetimes[1:] >= datetimes[:-1]).all() or (
            datetimes[1:] <= datetimes[:-1]
        ).all()
        if not monotonic:
            raise ValueError("Datetimes must be monotonic")
        # Datetimes must be unique
        selected = np.concatenate(((True,), datetimes[1:] != datetimes[:-1]))
        if not all(selected):
            warnings.warn("Dropping duplicate datetimes")
            datetimes = datetimes[selected]
        # Datetimes must match at least one Observer
        distances = helpers.pairwise_distance_datetimes(datetimes, self.datetimes)
        selected = distances.min(axis=1) <= abs(maxdt.total_seconds())
        if not all(selected):
            warnings.warn("Dropping datetimes not matching any Observers")
            datetimes = datetimes[selected]
        if len(datetimes) < 2:
            raise ValueError("Fewer than two valid datetimes")
        return datetimes

    def match_datetimes(
        self,
        datetimes: Iterable[datetime.datetime],
        maxdt: datetime.timedelta = datetime.timedelta(0),
    ) -> np.ndarray:
        """
        Return matching image indices for each Observer and datetime.

        Arguments:
            datetimes: Datetimes.
            maxdt: Maximum timedelta for an image to match `datetimes`.

        Returns:
            Grid of matching image indices (i, j) for datetimes[i] and observers[j].
            `None` represents no match.
        """
        matches = np.full((len(datetimes), len(self.observers)), None)
        for i, observer in enumerate(self.observers):
            distances = helpers.pairwise_distance_datetimes(
                datetimes, observer.datetimes
            )
            nearest_index = np.argmin(distances, axis=1)
            matches[:, i] = nearest_index
            nearest_distance = distances[np.arange(distances.shape[0]), nearest_index]
            not_selected = nearest_distance > abs(maxdt.total_seconds())
            matches[not_selected, i] = None
        return matches

    def extract_tile(
        self,
        obs: int,
        img: int,
        box: Iterable[Number],
        histogram: Tuple[np.ndarray, np.ndarray] = None,
        return_histogram: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Extract image tile.

        The tile is converted to grayscale, normalized to mean 0, variance 1,
        matched to a histogram (if `histogram`), and passed through a
        median low-pass filer.

        Arguments:
            obs: Observer index.
            img: Observer Image index.
            box: Tile boundaries (see :meth:`Observer.extract_tile`).
            histogram: Template for histogram matching
                (see :func:`helpers.compute_cdf`).
            return_histogram: Whether to return a tile histogram.
                The histogram is computed before the low-pass filter.

        Returns:
            An image tile and, if `return_histogram=True`,
            a histogram (values, quantiles) computed before the low-pass filter.
        """
        tile = self.observers[obs].extract_tile(box=box, img=img)
        if tile.ndim > 2:
            tile = tile.mean(axis=2)
        tile = helpers.normalize(tile)
        if histogram is not None:
            tile = helpers.match_cdf(tile, histogram)
        if return_histogram:
            returned_histogram = helpers.compute_cdf(tile, return_inverse=False)
        tile_low = scipy.ndimage.filters.median_filter(tile, **self.highpass)
        tile -= tile_low
        if return_histogram:
            return tile, returned_histogram
        return tile

    def initialize_template(self, obs: int, img: int, tile_size: Iterable[int]) -> None:
        """
        Initialize an observer template from the current particle state.

        Arguments:
            obs: Observer index.
            img: Observer Image index.
            tile_size: Size of template tile in pixels (nx, ny).
        """
        if self.templates is None:
            self.templates = [None] * len(self.observers)
        # Compute image box centered around particle mean
        xyz = self.particle_mean[None, 0:3]
        uv = self.observers[obs].xyz_to_uv(xyz, img=img).ravel()
        box = self.observers[obs].tile_box(uv, size=tile_size, img=img)
        # Build template
        template = {
            "obs": obs,
            "img": img,
            "box": box,
            "duv": uv - box.reshape(2, -1).mean(axis=0),
        }
        template["tile"], template["histogram"] = self.extract_tile(
            obs=obs, img=img, box=box, return_histogram=True
        )
        self.templates[obs] = template

    def compute_observer_log_likelihoods(
        self, obs: int, img: int
    ) -> Optional[np.ndarray]:
        """
        Compute particle log likelihoods for one Observer.

        Arguments:
            obs: Observer index.
            img: Observer image index.

        Returns:
            Particle log likelihoods.
        """
        constant_log_likelihood = None
        if img is None:
            return constant_log_likelihood
        # Build image box around all particles, with a buffer for template matching
        size = np.asarray(self.templates[obs]["tile"].shape[0:2][::-1])
        uv = self.observers[obs].xyz_to_uv(self.particles[:, 0:3], img=img)
        halfsize = size * 0.5
        box = np.row_stack((uv.min(axis=0) - halfsize, uv.max(axis=0) + halfsize))
        # Enlarge box to ensure SSE has cols, rows (ky + 1, kx + 1) for interpolation
        ky = self.interpolation.get("ky", 3)
        ncols = ky - (np.diff(box[:, 0]) - size[0])
        if np.all(ncols > 0):
            # Widen box in 2nd ('y') dimension (x|cols)
            box[:, 0] += np.hstack((-ncols, ncols)) * 0.5
        kx = self.interpolation.get("kx", 3)
        nrows = kx - (np.diff(box[:, 1]) - size[1])
        if np.all(nrows > 0):
            # Widen box in 1st ('x') dimension (y|rows)
            box[:, 1] += np.hstack((-nrows, nrows)) * 0.5
        box = np.vstack((np.floor(box[0, :]), np.ceil(box[1, :]))).astype(int)
        # Check that box is within image bounds
        if not all(self.observers[obs].images[img].inbounds(box)):
            warnings.warn(
                "Particles too close to or beyond image bounds, skipping image"
            )
            return constant_log_likelihood
        # Flatten box
        box = box.ravel()
        # Extract search tile
        search_tile = self.extract_tile(
            obs=obs, img=img, box=box, histogram=self.templates[obs]["histogram"]
        )
        # Compute area-averaged sum of squares error (SSE)
        sse = cv2.matchTemplate(
            search_tile.astype(np.float32),
            templ=self.templates[obs]["tile"].astype(np.float32),
            method=cv2.TM_SQDIFF,
        )
        sse *= 1 / (size[0] * size[1])
        # Compute SSE bounding box
        # (relative to search tile: shrunk by halfsize of template tile - 0.5 pixel)
        box_edge = halfsize - 0.5
        sse_box = box + np.concatenate((box_edge, -box_edge))
        # (shift by subpixel offset of template tile)
        sse_box += np.tile(self.templates[obs]["duv"], 2)
        # Sample at projected particles
        sampled_sse = self.observers[obs].sample_tile(
            uv, tile=sse, box=sse_box, grid=False, **self.interpolation
        )
        return sampled_sse * (1 / (2 * self.observers[obs].sigma ** 2))
