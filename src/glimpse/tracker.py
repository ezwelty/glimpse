"""Track particle trajectories through time."""
import datetime
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings

import cv2
import matplotlib.animation
import matplotlib.pyplot
import numpy as np
import scipy.ndimage
from typing_extensions import Literal

from . import config, helpers
from .observer import Observer
from .raster import Raster

Index = Union[slice, Iterable[int]]
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
        self, imgs: Iterable[Optional[int]], motion_model: MotionModel = None
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
        motion_models: Iterable[MotionModel],
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

        def process(motion_model: MotionModel, observer_mask: np.ndarray) -> list:
            means = np.full((ntimes, 6), np.nan)
            if return_covariances:
                sigmas = np.full((ntimes, 6, 6), np.nan)
            else:
                sigmas = np.full((ntimes, 6), np.nan)
            if return_particles:
                particles = np.full((ntimes, ntracks, 6), np.nan)
                weights = np.full((ntimes, ntracks), np.nan)
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
        box = self.observers[obs].tile_box(uv, size=tile_size)
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
        if ncols > 0:
            # Widen box in 2nd ('y') dimension (x|cols)
            box[:, 0] += np.hstack((-ncols, ncols)) * 0.5
        kx = self.interpolation.get("kx", 3)
        nrows = kx - (np.diff(box[:, 1]) - size[1])
        if nrows > 0:
            # Widen box in 1st ('x') dimension (y|rows)
            box[:, 1] += np.hstack((-nrows, nrows)) * 0.5
        box = np.vstack((np.floor(box[0, :]), np.ceil(box[1, :]))).astype(int)
        # Check that box is within image bounds
        if not all(self.observers[obs].grid.inbounds(box)):
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


class Tracks:
    """
    Estimated trajectories of world points.

    In the argument and attribute descriptions:

    - n: number of tracks
    - m: number of datetimes
    - p: number of particles

    Attributes:
        datetimes (array): Datetimes at which particles were estimated (m, )
        means (array): Mean particle positions and velocities
            (x, y, z, vx, vy, vz) (n, m, 6)
        sigmas (array): Standard deviations of particle positions and velocities
            (x, y, z, vx, vy, vz) (n, m, 6)
        covariances (array): Covariance of particle positions and velocities
            (n, m, 6, 6)
        particles (array): Particle positions and velocities (n, m, p, 6)
        weights (array): Particle weights (n, m, p)
        tracker (Tracker): Tracker object used for tracking
        images (array): Grid of image indices ([i, j] for `datetimes[i]`,
            `tracker.observers`[j]`). `None` indicates no image from
            `tracker.observers[j]` matched `datetimes[i]`.
        params (dict): Arguments to `Tracker.track`
        errors (array): The error, if any, caught for each track (n, ).
            An error indicates that the track did not complete successfully.
        warnings (array): Warnings, if any, caught for each track (n, ).
            Warnings indicate the track completed but may not be valid.
    """

    def __init__(
        self,
        datetimes: Iterable[datetime.datetime],
        means: Iterable[Iterable[Iterable[Number]]],
        sigmas: Iterable[Iterable[Iterable[Number]]] = None,
        covariances: Iterable[Iterable[Iterable[Iterable[Number]]]] = None,
        particles: Iterable[Iterable[Iterable[Iterable[Number]]]] = None,
        weights: Iterable[Iterable[Iterable[Number]]] = None,
        tracker: Tracker = None,
        images: Iterable[Iterable[Optional[int]]] = None,
        params: dict = None,
        errors: Iterable = None,
        warnings: Iterable = None,
    ) -> None:
        self.datetimes = np.asarray(datetimes)
        if np.iterable(means) and not isinstance(means, np.ndarray):
            means = np.stack(means, axis=0)
        self.means = means
        if np.iterable(sigmas) and not isinstance(sigmas, np.ndarray):
            sigmas = np.stack(sigmas, axis=0)
        self.sigmas = sigmas
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
        self.images = images if images is None else np.asarray(images)
        self.params = params
        self.errors = errors if errors is None else np.asarray(errors)
        self.warnings = warnings if warnings is None else np.asarray(warnings)

    @property
    def xyz(self) -> np.ndarray:
        """Mean particle positions (n, m, [x, y, z])."""
        return self.means[:, :, 0:3]

    @property
    def vxyz(self) -> np.ndarray:
        """Mean particle velocities (n, m, [vx, vy, vz])."""
        return self.means[:, :, 3:6]

    @property
    def xyz_sigma(self) -> np.ndarray:
        """Standard deviation of particle positions (n, m, [x, y, z])."""
        if self.sigmas is not None:
            return self.sigmas[:, :, 0:3]
        if self.covariances is not None:
            return np.sqrt(self.covariances[:, :, (0, 1, 2), (0, 1, 2)])

    @property
    def vxyz_sigma(self) -> np.ndarray:
        """Standard deviation of particle velocities (n, m, [vx, vy, vz])."""
        if self.sigmas is not None:
            return self.sigmas[:, :, 3:6]
        if self.covariances is not None:
            return np.sqrt(self.covariances[:, :, (3, 4, 5), (3, 4, 5)])

    @property
    def success(self) -> np.ndarray:
        """Whether track completed without errors (n,)."""
        if self.errors is not None:
            return np.array([error is None for error in self.errors])

    def endpoints(
        self, tracks: Index = slice(None)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return endpoints of each track.

        Arguments:
            tracks: Indices of tracks to return.

        Returns:
            For each selected track, the indices of the first and last valid datetimes.
        """
        valid = ~np.isnan(self.means[tracks, :, 0])
        first = np.argmax(valid, axis=1)
        last = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
        first_valid = valid[np.arange(len(first)), first]
        return first_valid, first[first_valid], last[first_valid]

    # @property
    # def first(self):
    #     valid = ~np.isnan(self.means[:, :, 0])
    #     idx = np.argmax(valid, axis=1)
    #     still_valid = valid[np.arange(len(idx)), idx]
    #     return np.nonzero(still_valid)[0], idx[idx]
    #
    # @property
    # def last(self):
    #     valid = ~np.isnan(self.means[:, :, 0])
    #     idx = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    #     still_valid = valid[np.arange(len(idx)), idx]
    #     return np.nonzero(still_valid)[0], idx[idx]

    def plot_xy(
        self,
        tracks: Index = slice(None),
        start: Union[bool, dict] = True,
        mean: Union[bool, dict] = True,
        sigma: Union[bool, dict] = False,
    ) -> Dict[
        Literal["mean", "start", "sigma"],
        Union[
            List[matplotlib.lines.Line2D], List[matplotlib.container.ErrorbarContainer]
        ],
    ]:
        """
        Plot tracks on the x-y plane.

        Arguments:
            tracks: Indices of tracks to include.
            start: Whether to plot starting x, y (bool) or arguments to
                :func:`matplotlib.pyplot.plot` (dict).
            mean: Whether to plot mean x, y (bool) or arguments to
                :func:`matplotlib.pyplot.plot` (dict).
            sigma: Whether to plot sigma x, y (bool) or arguments to
                :func:`matplotlib.pyplot.errorbar` (dict).
        """
        results = {}
        if mean:
            if mean is True:
                mean = {}
            default = {"color": "black"}
            mean = {**default, **mean}
            results["mean"] = matplotlib.pyplot.plot(
                self.xyz[tracks, :, 0].T, self.xyz[tracks, :, 1].T, **mean
            )
        if start:
            if start is True:
                start = {}
            default = {"color": "black", "marker": ".", "linestyle": "none"}
            if isinstance(mean, dict) and "color" in mean:
                default["color"] = mean["color"]
            start = {**default, **start}
            results["start"] = matplotlib.pyplot.plot(
                self.xyz[tracks, 0, 0], self.xyz[tracks, 0, 1], **start
            )
        if sigma:
            if sigma is True:
                sigma = {}
            default = {"color": "black", "alpha": 0.25}
            if isinstance(mean, dict) and "color" in mean:
                default["color"] = mean["color"]
            sigma = {**default, **sigma}
            results["sigma"] = []
            for i in np.atleast_1d(np.arange(len(self.xyz))[tracks]):
                results["sigma"].append(
                    matplotlib.pyplot.errorbar(
                        self.xyz[i, :, 0],
                        self.xyz[i, :, 1],
                        xerr=self.xyz_sigma[i, :, 0],
                        yerr=self.xyz_sigma[i, :, 1],
                        **sigma
                    )
                )
        return results

    def plot_vxy(
        self, tracks: Index = slice(None), **kwargs: Any
    ) -> List[matplotlib.quiver.Quiver]:
        """
        Plot velocities as vector fields on the x-y plane.

        Arguments:
            tracks: Indices of tracks to include.
            **kwargs: Optional arguments to :func:`matplotlib.pyplot.quiver`.
        """
        default = {"angles": "xy"}
        kwargs = {**default, **kwargs}
        results = []
        for i in np.atleast_1d(np.arange(len(self.xyz))[tracks]):
            results.append(
                matplotlib.pyplot.quiver(
                    self.xyz[i, :, 0],
                    self.xyz[i, :, 1],
                    self.vxyz[i, :, 0],
                    self.vxyz[i, :, 1],
                    **kwargs
                )
            )
        return results

    def plot_v1d(
        self,
        dim: int,
        tracks: Index = slice(None),
        mean: Union[bool, dict] = True,
        sigma: Union[bool, dict] = False,
    ) -> Dict[
        Literal["mean", "sigma"],
        Union[
            List[matplotlib.lines.Line2D, List[matplotlib.collections.PolyCollection]]
        ],
    ]:
        """
        Plot velocity for one dimension.

        Arguments:
            dim: Dimension to plot (0: vx, 1: vy, 2: vz).
            tracks: Indices of tracks to include.
            mean: Whether to plot mean vx (bool) or arguments to
                :func:`matplotlib.pyplot.plot` (dict).
            sigma: Whether to plot sigma vx (bool) or arguments to
                :func:`matplotlib.pyplot.fill_between` (dict).
        """
        results = {}
        if mean:
            if mean is True:
                mean = {}
            default = {"color": "black"}
            mean = {**default, **mean}
            results["mean"] = matplotlib.pyplot.plot(
                self.datetimes, self.vxyz[tracks, :, dim].T, **mean
            )
        if sigma:
            if sigma is True:
                sigma = {}
            default = {"facecolor": "black", "edgecolor": "none", "alpha": 0.25}
            if isinstance(mean, dict) and "color" in mean:
                default["facecolor"] = mean["color"]
            sigma = {**default, **sigma}
            results["sigma"] = []
            for i in np.atleast_1d(np.arange(len(self.xyz))[tracks]):
                results["sigma"].append(
                    matplotlib.pyplot.fill_between(
                        self.datetimes,
                        y1=self.vxyz[i, :, dim] + self.vxyz_sigma[i, :, dim],
                        y2=self.vxyz[i, :, dim] - self.vxyz_sigma[i, :, dim],
                        **sigma
                    )
                )
        return results

    def animate(
        self,
        track: int,
        obs: int = 0,
        frames: Iterable[datetime.datetime] = None,
        images: bool = None,
        particles: bool = None,
        map_size: Tuple[Number, Number] = (20, 20),
        img_size: Tuple[int, int] = (100, 100),
        subplots: dict = {},
        animation: dict = {},
    ) -> matplotlib.animation.FuncAnimation:
        """
        Animate track.

        Arguments:
            track: Track index.
            obs: Observer index.
            frames: Datetime index. Datetimes with no observations are skipped.
                By default, all times with observations are used.
            images: Whether to plot images.
                By default, images are plotted if :attr:`tracker` is set.
            particles: Whether to plot particles. By default, particles are plotted
                if :attr:`particles` and :attr:`weights` are set.
            map_size: Size of map window in world units.
            img_size: Size of image window in pixels.
            subplots: Optional arguments to :func:`matplotlib.pyplot.subplots`.
            animation: Optional arguments to :func:`matplotlib.animation.FuncAnimation`.
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
            ~np.isnan(self.xyz[track, :, 0]) & np.not_equal(self.images[:, obs], None)
        )[0]
        frames = np.intersect1d(frames, has_frame)
        # Initialize plot
        i = frames[0]
        img = self.images[i, obs]
        # Map: Track
        track_xyz = self.xyz[track, : (i + 1)]
        map_track = axes[0].plot(
            track_xyz[:, 0], track_xyz[:, 1], color="black", marker="."
        )[0]
        if images:
            # Image: Track
            track_uv = self.tracker.observers[obs].xyz_to_uv(track_xyz, img=img)
            image_track = axes[1].plot(
                track_uv[:, 0], track_uv[:, 1], color="black", marker="."
            )[0]
            # Image: Mean
            image_mean = axes[1].plot(
                track_uv[-1, 0], track_uv[-1, 1], color="red", marker="."
            )[0]
            # Image: Tile
            box = self.tracker.observers[obs].tile_box(track_uv[-1], size=img_size)
            tile = self.tracker.observers[obs].extract_tile(img=img, box=box)
            image_tile = self.tracker.observers[obs].plot_tile(
                tile=tile, box=box, axes=axes[1]
            )
        # Map: Basename
        if images:
            basename = helpers.strip_path(self.tracker.observers[obs].images[img].path)
        else:
            basename = str(obs) + " : " + str(img)
        map_txt = axes[0].text(
            0.5,
            0.9,
            basename,
            color="black",
            horizontalalignment="center",
            transform=axes[0].transAxes,
        )
        if particles:
            # Compute quiver scales
            scales = np.diff(self.datetimes[frames]) / self.tracker.time_unit
            # Compute weight limits for static colormap
            clim = (
                self.weights[track, :].ravel().min(),
                self.weights[track, :].ravel().max(),
            )
        elif self.tracker is not None:
            scales = np.diff(self.datetimes[frames]) / self.tracker.time_unit
        else:
            scales = np.ones(len(frames) - 1)
        # Discard last frame
        frames = frames[:-1]

        def update_plot(i: int) -> tuple:
            # PathCollections cannot set x, y, so new objects have to be created
            for ax in axes:
                ax.collections = []
            img = self.images[i, obs]
            if particles:
                # Map: Particles
                particle_xyz = self.particles[track, i, :, 0:3]
                particle_vxy = self.particles[track, i, :, 3:5] * scales[i]
                axes[0].quiver(
                    particle_xyz[:, 0],
                    particle_xyz[:, 1],
                    particle_vxy[:, 0],
                    particle_vxy[:, 1],
                    self.weights[track, i],
                    cmap=matplotlib.pyplot.cm.gnuplot2,
                    alpha=0.25,
                    angles="xy",
                    scale=1,
                    scale_units="xy",
                    units="xy",
                    clim=clim,
                )
                # matplotlib.pyplot.colorbar(quivers, ax=axes[0], label="Weight")
            if images and particles:
                # Image: Particles
                particle_uv = self.tracker.observers[obs].xyz_to_uv(
                    particle_xyz, img=img
                )
                axes[1].scatter(
                    particle_uv[:, 0],
                    particle_uv[:, 1],
                    c=self.weights[track, i],
                    marker=".",
                    cmap=matplotlib.pyplot.cm.gnuplot2,
                    alpha=0.25,
                    edgecolors="none",
                    vmin=clim[0],
                    vmax=clim[1],
                )
                # matplotlib.pyplot.colorbar(
                #   image_particles, ax=axes[1], label="Weight"
                # )
            # Map: Track
            track_xyz = self.xyz[track, : (i + 1)]
            map_track.set_data(track_xyz[:, 0], track_xyz[:, 1])
            axes[0].set_xlim(
                track_xyz[-1, 0] - map_size[0] / 2, track_xyz[-1, 0] + map_size[0] / 2
            )
            axes[0].set_ylim(
                track_xyz[-1, 1] - map_size[1] / 2, track_xyz[-1, 1] + map_size[1] / 2
            )
            # Map: Mean
            axes[0].quiver(
                self.xyz[track, i, 0],
                self.xyz[track, i, 1],
                self.vxyz[track, i, 0] * scales[i],
                self.vxyz[track, i, 1] * scales[i],
                color="red",
                alpha=1,
                angles="xy",
                scale=1,
                scale_units="xy",
                units="xy",
            )
            if images:
                # Image: Track
                track_uv = self.tracker.observers[obs].xyz_to_uv(track_xyz, img=img)
                image_track.set_data(track_uv[:, 0], track_uv[:, 1])
                axes[1].set_xlim(
                    track_uv[-1, 0] - img_size[0] / 2, track_uv[-1, 0] + img_size[0] / 2
                )
                axes[1].set_ylim(
                    track_uv[-1, 1] + img_size[1] / 2, track_uv[-1, 1] - img_size[1] / 2
                )
                # Image: Mean
                image_mean.set_data(track_uv[-1, 0], track_uv[-1, 1])
                # Image: Tile
                box = self.tracker.observers[obs].tile_box(
                    uv=track_uv[-1, :], size=img_size
                )
                tile = self.tracker.observers[obs].extract_tile(box=box, img=img)
                image_tile.set_data(tile)
                image_tile.set_extent((box[0], box[2], box[3], box[1]))
            # Map: Basename
            if images:
                basename = helpers.strip_path(
                    self.tracker.observers[obs].images[img].path
                )
            else:
                basename = str(obs) + " : " + str(img)
            basename = helpers.strip_path(self.tracker.observers[obs].images[img].path)
            map_txt.set_text(basename)
            if images:
                return map_track, map_txt, image_track, image_tile, image_mean
            return map_track, map_txt

        return matplotlib.animation.FuncAnimation(
            fig, update_plot, frames=frames, blit=True, **animation
        )


class MotionModel:
    """
    Illustration of the motion model interface required by :class:`Tracker`.

    :class:`Tracker` requires a motion model to have the following methods:

        - `initialize_particles()`: Initializes particles, typically around an
          initial position.
        - `evolve_particles(particles, dt)`: Evolves particles forward or backward
          in time.
        - `compute_log_likelihoods(particles)`: Computes particle log likelihoods
          (optional). If provided, these are added to the Observer log
          likelihoods computed by a `Tracker` to calculate the final particle
          likelihoods.

    This minimal example initializes all particles at the same position
    (x, y, 0) with velocity components in x, y, z drawn from normal
    distributions with zero mean. Particles are evolved based only on their
    initial velocities since accelerations are absent.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        n (int): Number of particles.
        vxyz_sigma (iterable): Standard deviation of velocity
            (dx/dt, dy/dt, dz/dt) in `time_unit` time units.
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        n: int = 1000,
        vxyz_sigma: Iterable[Number] = (0, 0, 0),
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        self.n = n
        self.vxyz_sigma = vxyz_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy
        particles[:, 3:6] = self.vxyz_sigma * np.random.randn(self.n, 3)
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        particles[:, 0:3] += time_units * particles[:, 3:6]

    def compute_log_likelihoods(self, particles: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute particle log likelihoods.

        If specified, these are added to the Observer log likelihoods computed
        by a :class:`Tracker` to calculate the final particle likelihoods.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).

        Returns:
            Particle log likelihoods, or `None`.
        """
        return None


class CartesianMotionModel(MotionModel):
    """
    Evolves particles following a Cartesian motion model.

    Initial particle positions and velocities, and random accelerations, are
    specified by independent and normally distributed x, y, z components.
    Temporal arguments (e.g. :attr:`vxyz`, :attr:`axyz`) are assumed to be
    in :attr:`time_unit` time units.

    Particle heights (z) are initialized based on a mean surface (:attr:`dem`) and its
    associated uncertainty (:attr:`dem_sigma`), then evolved following the motion
    model. Particles are weighted based on their distance (dz) from the surface
    and its uncertainty.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        dem: Elevation of the surface on which to track points (scalar or Raster).
        dem_sigma: Elevation standard deviations, as either a scalar or
             a Raster with the same extent as `dem`. `0` means particles stay
             glued to `dem` and weighing particles by their offset from `dem`
             is disabled.
        n (int): Number of particles.
        xy_sigma (iterable): Standard deviation of initial position (x, y).
        vxyz (iterable): Mean initial velocity (dx/dt, dy/dt, dz/dt).
        vxyz_sigma (iterable): Standard deviation of initial velocity
            (dx/dt, dy/dt, dz/dt).
        axyz (iterable): Mean acceleration (d^2x/dt^2, d^2y/dt^2, d^2z/dt^2).
        axyz_sigma (iterable): Standard deviation of acceleration
            (d^2x/dt^2, d^2y/dt^2, d^2z/dt^2).
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        dem: Union[Number, Raster],
        dem_sigma: Union[Number, Raster] = 0,
        n: int = 1000,
        xy_sigma: Iterable[Number] = (0, 0),
        vxyz: Iterable[Number] = (0, 0, 0),
        vxyz_sigma: Iterable[Number] = (0, 0, 0),
        axyz: Iterable[Number] = (0, 0, 0),
        axyz_sigma: Iterable[Number] = (0, 0, 0),
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        self.dem = dem
        self.dem_sigma = dem_sigma
        self.n = n
        self.xy_sigma = xy_sigma
        self.vxyz = vxyz
        self.vxyz_sigma = vxyz_sigma
        self.axyz = axyz
        self.axyz_sigma = axyz_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy + self.xy_sigma * np.random.randn(self.n, 2)
        z = self._sample_dem(particles[:, 0:2])
        z_sigma = self._sample_dem(particles[:, 0:2], sigma=True)
        particles[:, 2] = z + z_sigma * np.random.randn(self.n)
        particles[:, 3:6] = self.vxyz + self.vxyz_sigma * np.random.randn(self.n, 3)
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        n = len(particles)
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        axyz = self.axyz + self.axyz_sigma * np.random.randn(n, 3)
        particles[:, 0:3] += (
            time_units * particles[:, 3:6] + 0.5 * axyz * time_units ** 2
        )
        particles[:, 3:6] += time_units * axyz

    def compute_log_likelihoods(self, particles: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute particle log likelihoods.

        Particles are weighted based on their distance from the mean surface
        (:attr:`dem`) and its associated uncertainty (:attr:`dem_sigma`).

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).

        Returns:
            Particle log likelihoods, or `None`.
        """
        if self.dem_sigma == 0:
            return None
        z = self._sample_dem(particles[:, 0:2])
        z_sigma = self._sample_dem(particles[:, 0:2], sigma=True)
        # Avoid division by zero
        nonzero = np.nonzero(z_sigma)[0]
        log_likelihoods = np.zeros(len(particles), dtype=float)
        log_likelihoods[nonzero] = (
            1 / (2 * z_sigma[nonzero] ** 2) * (z[nonzero] - particles[nonzero, 2]) ** 2
        )
        return log_likelihoods

    def _sample_dem(self, xy: np.ndarray, sigma: bool = False) -> np.ndarray:
        """
        Sample DEM at points.

        Arguments:
            xy: Points [(x, y), ...].
            sigma: Whether to sample :attr:`dem_sigma` (True) or :attr:`dem` (False).
        """
        obj = self.dem_sigma if sigma else self.dem
        if isinstance(obj, Raster):
            return obj.sample(xy)
        return np.full(len(xy), obj)


class CylindricalMotionModel(CartesianMotionModel):
    """
    Evolves particles following a cylindrical motion model.

    Identical to :class:`CartesianMotionModel`, except that particle motion is
    specified by independent and normally distributed components of magnitude
    (radius), direction (theta), and elevation (z). Angular arguments are
    assumed to be in radians counterclockwise from the +x axis.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        dem: Elevation of the surface on which to track points
            (scalar or Raster).
        dem_sigma: Elevation standard deviations, as either a scalar or
             a Raster with the same extent as :attr:`dem`. `0` means particles stay
             glued to `dem` and weighing particles by their offset from `dem`
             is disabled.
        n (int): Number of particles.
        xy_sigma (iterable): Standard deviation of initial position (x, y).
        vrthz (iterable): Mean initial velocity (d radius/dt, theta, dz/dt).
        vrthz_sigma (iterable): Standard deviation of initial velocity
            (d radius/dt, theta, dz/dt).
        arthz (iterable): Mean acceleration
            (d^2 radius/dt^2, d theta/dt, d^2z/dt^2).
        arthz_sigma (iterable): Standard deviation of acceleration
            (d^2 radius/dt^2, d theta/dt, d^2z/dt^2).
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        dem: Union[Number, Raster],
        dem_sigma: Union[Number, Raster] = 0,
        n: int = 1000,
        xy_sigma: Iterable[Number] = (0, 0),
        vrthz: Iterable[Number] = (0, 0, 0),
        vrthz_sigma: Iterable[Number] = (0, 0, 0),
        arthz: Iterable[Number] = (0, 0, 0),
        arthz_sigma: Iterable[Number] = (0, 0, 0),
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        self.dem = dem
        self.dem_sigma = dem_sigma
        self.n = n
        self.xy_sigma = xy_sigma
        self.vrthz = vrthz
        self.vrthz_sigma = vrthz_sigma
        self.arthz = arthz
        self.arthz_sigma = arthz_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy + self.xy_sigma * np.random.randn(self.n, 2)
        z = self._sample_dem(particles[:, 0:2])
        z_sigma = self._sample_dem(particles[:, 0:2], sigma=True)
        particles[:, 2] = z + z_sigma * np.random.randn(self.n)
        v = self.vrthz + self.vrthz_sigma * np.random.randn(self.n, 3)
        particles[:, 3:6] = np.column_stack(
            (
                # r' * cos(th)
                v[:, 0] * np.cos(v[:, 1]),
                # r' * sin(th)
                v[:, 0] * np.sin(v[:, 1]),
                v[:, 2],
            )
        )
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        n = len(particles)
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        vx = particles[:, 3]
        vy = particles[:, 4]
        vr = np.sqrt(vx ** 2 + vy ** 2)
        arthz = self.arthz + self.arthz_sigma * np.random.randn(n, 3)
        axyz = np.column_stack(
            (
                # r'' * cos(th) - r' * sin(th) * th'
                arthz[:, 0] * (vx / vr) - vy * arthz[:, 1],
                # r'' * sin(th) - r' * cos(th) * th'
                arthz[:, 0] * (vy / vr) + vx * arthz[:, 1],
                arthz[:, 2],
            )
        )
        particles[:, 0:3] += (
            time_units * particles[:, 3:6] + 0.5 * axyz * time_units ** 2
        )
        particles[:, 3:6] += time_units * axyz
