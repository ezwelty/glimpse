"""Manipulate and plot tracked particle trajectories."""
import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.animation
import matplotlib.pyplot
import numpy as np
from typing_extensions import Literal

if TYPE_CHECKING:
    # Prevent circular import (see https://stackoverflow.com/a/39757388)
    from .tracker import Tracker

from .. import helpers

Index = Union[slice, Iterable[int]]
Number = Union[int, float]


class Tracks:
    """
    Estimated trajectories of world points.

    In the argument and attribute descriptions:

    - n: number of tracks
    - m: number of datetimes
    - p: number of particles

    Attributes:
        datetimes (numpy.ndarray): Datetimes at which particles were estimated (m,).
        time_unit (datetime.timedelta): Time unit for the velocities.
        means (numpy.ndarray): Mean particle positions and velocities
            (x, y, z, vx, vy, vz) (n, m, 6).
        sigmas (numpy.ndarray): Standard deviations of particle positions and velocities
            (x, y, z, vx, vy, vz) (n, m, 6).
        covariances (numpy.ndarray): Covariance of particle positions and velocities
            (n, m, 6, 6).
        particles (numpy.ndarray): Particle positions and velocities (n, m, p, 6).
        weights (numpy.ndarray): Particle weights (n, m, p).
        tracker (Tracker): Tracker object used for tracking.
        images (numpy.ndarray): Grid of image indices ([i, j] for `datetimes[i]`,
            `tracker.observers`[j]`). `None` indicates no image from
            `tracker.observers[j]` matched `datetimes[i]`.
        params (dict): Arguments to :meth:`Tracker.track`.
        errors (numpy.ndarray): The error, if any, caught for each track (n,).
            An error indicates that the track did not complete successfully.
        warnings (numpy.ndarray): Warnings, if any, caught for each track (n,).
            Warnings indicate the track completed but may not be valid.
    """

    def __init__(
        self,
        datetimes: Iterable[datetime.datetime],
        time_unit: datetime.timedelta,
        means: Iterable[Iterable[Iterable[Number]]],
        sigmas: Iterable[Iterable[Iterable[Number]]] = None,
        covariances: Iterable[Iterable[Iterable[Iterable[Number]]]] = None,
        particles: Iterable[Iterable[Iterable[Iterable[Number]]]] = None,
        weights: Iterable[Iterable[Iterable[Number]]] = None,
        tracker: "Tracker" = None,
        images: Iterable[Iterable[Optional[int]]] = None,
        params: dict = None,
        errors: Iterable = None,
        warnings: Iterable = None,
    ) -> None:
        self.datetimes = np.asarray(datetimes)
        self.time_unit = time_unit
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
            List[matplotlib.lines.Line2D], List[matplotlib.collections.PolyCollection]
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
            box = self.tracker.observers[obs].tile_box(
                track_uv[-1], size=img_size, img=img
            )
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
            scales = np.diff(self.datetimes[frames]) / self.time_unit
            # Compute weight limits for static colormap
            clim = (
                self.weights[track, :].ravel().min(),
                self.weights[track, :].ravel().max(),
            )
        elif self.tracker is not None:
            scales = np.diff(self.datetimes[frames]) / self.time_unit
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
                    uv=track_uv[-1, :], size=img_size, img=img
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
