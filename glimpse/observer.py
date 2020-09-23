"""Represent a sequence of images taken from the same camera position."""
import datetime
from typing import Any, cast, Iterable, List, Tuple, Union

import matplotlib.animation
import matplotlib.patches
import matplotlib.pyplot
import numpy as np
import scipy.interpolate

from . import helpers
from .image import Image
from .raster import Grid


class Observer:
    """
    A sequence of images taken from the same camera position.

    Attributes:
        xyz (np.ndarray): Camera position in world coordinates.
        images (List[Image]): Images with equal camera position (xyz),
            focal length (f), image size (imgsz) and
            strictly increasing in time (datetime).
        datetimes (np.ndarray): Image capture times.
        sigma (float): Standard deviation of pixel values between images
            due to changes in illumination, deformation, or unresolved camera motion.
        cache (bool): Whether to cache images on read.

    Raises:
        ValueError: Images are not two or greater.
        ValueError: Image is missing datetime.
        ValueError: Image datetimes are not stricly increasing.
        ValueError: Camera image sizes are not equal.
        ValueError: Camera positions are not equal.
        ValueError: Camera focal lengths are not equal.
    """

    def __init__(
        self, images: Iterable[Image], sigma: float = 0.3, cache: bool = True,
    ) -> None:
        if len(images) < 2:
            raise ValueError("Images are not two or greater")
        datetimes = []
        for i, img in enumerate(images):
            if img.datetime is None:
                raise ValueError(f"Image {i} is missing datetime")
            datetimes.append(img.datetime)
            if np.linalg.norm(img.cam.xyz - images[0].cam.xyz) > 1e-3:
                raise ValueError("Camera positions (xyz) are not equal")
            if any(img.cam.f != images[0].cam.f):
                raise ValueError("Camera focal lengths (f) are not equal")
            if any(img.cam.imgsz != images[0].cam.imgsz):
                raise ValueError("Camera image sizes (imgsz) are not equal")
        time_deltas = np.array([dt.total_seconds() for dt in np.diff(datetimes)])
        if any(time_deltas <= 0):
            raise ValueError("Image datetimes are not stricly increasing")
        self.images = list(images)
        self.xyz = images[0].cam.xyz
        self.datetimes = np.array(datetimes)
        self.sigma = sigma
        self.cache = cache
        n = self.images[0].cam.imgsz
        self._grid = Grid(n=n, x=(0, n[0]), y=(0, n[1]))

    def index(
        self,
        value: Union[Image, datetime.datetime],
        maxdt: datetime.timedelta = datetime.timedelta(0),
    ) -> int:
        """
        Return the index of an image.

        Arguments:
            value: Either an image to find in :attr:`images` or
                a datetime to match against :attr:`datetimes`.
            maxdt: Maximum timedelta for a datetime `value` to be
                considered a match. If `None`, no limit is placed on the match.

        Returns:
            Image index.

        Raises:
            ValueError: Value is not in list.
            ValueError: Nearest image out of range.
        """
        if isinstance(value, datetime.datetime):
            dts = np.abs(value - self.datetimes)
            index = np.argmin(dts)
            if maxdt is not None and dts[index] > abs(maxdt):
                raise ValueError(
                    "Nearest image out of range by " + str(dts[index] - abs(maxdt))
                )
            return index
        return self.images.index(value)

    def xyz_to_uv(
        self, xyz: np.ndarray, img: int, directions: bool = False
    ) -> np.ndarray:
        """
        Project world coordinates to image coordinates.

        Arguments:
            xyz: World coordinates (n, [x, y, z]).
            img: Index of image to project into.
            directions: Whether `xyz` are absolute coordinates (`False`)
                or ray directions (`True`).

        Returns:
            Image coordinates (n, [u, v]).
        """
        return self.images[img].cam.xyz_to_uv(xyz, directions=directions)

    def tile_box(self, uv: Iterable[float], size: Iterable[int] = (1, 1)) -> np.ndarray:
        """
        Compute a grid-aligned box centered around a point.

        Arguments:
            uv: Desired box center in image coordinates (u, v).
            size: Size of the box in pixels (nx, ny).

        Returns:
            Integer (pixel edge) boundaries (left, top, right, bottom).
        """
        return self._grid.snap_box(uv, size, centers=False, edges=True).astype(int)

    def extract_tile(self, box: Iterable[int], img: int) -> np.ndarray:
        """
        Extract rectangular image region.

        Cached results (:attr:`cache`) are slow the first time (the full image is read),
        but fastest on subsequent reads.
        Non-cached results are read with a speed proportional to the size of the box.

        Arguments:
            box: Boundaries of tile in image coordinates (left, top, right, bottom).
            img: Index of image to read.
        """
        return self.images[img].read(box=box, cache=self.cache)

    def shift_tile(
        self, tile: np.ndarray, duv: Iterable[float], **kwargs: Any
    ) -> np.ndarray:
        """
        Shift tile by a subpixel offset.

        Useful for centering an image tile over an arbitrary center point.

        Arguments:
            tile: 2-D or 3-D image tile.
            duv: Shift in image coordinates (du, dv).
                Must be 0.5 pixels or smaller in each dimension.
            **kwargs: Optional arguments to
                :class:`scipy.interpolate.RectBivariateSpline`.

        Raises:
            ValueError: Shift larger than 0.5 pixels.
        """
        if any(np.abs(duv) > 0.5):
            raise ValueError("Shift larger than 0.5 pixels")
        # Cell center coordinates (arbitrary origin)
        cu = self._grid.x[0 : tile.shape[0]]  # x|cols
        cv = self._grid.y[0 : tile.shape[1]]  # y|rows
        # Interpolate at shifted center coordinates
        tile = np.atleast_3d(tile)
        for i in range(tile.shape[2]):
            f = scipy.interpolate.RectBivariateSpline(cv, cu, tile[:, :, i], **kwargs)
            tile[:, :, i] = f(cv + duv[1], cu + duv[0], grid=True)
        if tile.shape[2] == 1:
            return tile.squeeze(axis=2)
        return tile

    def sample_tile(
        self,
        uv: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        tile: np.ndarray,
        box: Iterable[float],
        grid: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Sample tile at image coordinates.

        Arguments:
            uv: Image coordinates as either points (n, [u, v]) if `grid` is `False`
                or grid coordinate arrays [(nu, ), (nv, )] if `grid` is `True`.
            tile: Image tile (ny, nx, 2).
            box: Boundaries of tile in image coordinates (left, top, right, bottom).
            grid: See `uv`.
            **kwargs: Optional arguments to
                :class:`scipy.interpolate.RectBivariateSpline`.

        Raises:
            ValueError: Some sampling points are outside box.
        """
        if not np.all(helpers.in_box(uv, box)):
            raise ValueError("Some sampling points are outside box")
        # Cell sizes
        du = (box[2] - box[0]) / tile.shape[1]
        dv = (box[3] - box[1]) / tile.shape[0]
        # Cell center coordinates
        cu = np.arange(box[0] + du * 0.5, box[2])  # x|cols
        cv = np.arange(box[1] + dv * 0.5, box[3])  # y|rows
        # Interpolate at arbitrary coordinates
        f = scipy.interpolate.RectBivariateSpline(cv, cu, tile, **kwargs)
        if grid:
            return f(uv[1], uv[0], grid=grid)
        uv = cast(np.ndarray, uv)
        return f(uv[:, 1], uv[:, 0], grid=grid)

    def plot_tile(
        self,
        tile: np.ndarray,
        box: Iterable[float] = None,
        axes: matplotlib.axes.Axes = None,
        **kwargs: Any,
    ) -> matplotlib.image.AxesImage:
        """
        Plot image tile.

        Arguments:
            tile: Image tile (ny, nx, 2 or 3).
            box: Boundaries of tile in image coordinates (left, top, right, bottom).
                If `None`, the upper-left corner of the
                upper-left pixel is placed at (0, 0).
            axes: Axes to plot to. If `None`, uses the current axes.
            **kwargs: Optional arguments to :func:`matplotlib.pyplot.imshow`.
        """
        if box is None:
            box = (0, 0, tile.shape[0], tile.shape[1])
        extent = (box[0], box[2], box[3], box[1])
        if axes is None:
            axes = matplotlib.pyplot.gca()
        return axes.imshow(tile, origin="upper", extent=extent, **kwargs)

    def plot_box(
        self, box: Iterable[float], axes: matplotlib.axes.Axes = None, **kwargs: Any
    ) -> matplotlib.patches.Rectangle:
        """
        Plot bounding box.

        Arguments:
            box: Box in image coordinates (left, top, right, bottom).
            axes: Axes to plot to. If `None`, uses the current axes.
            **kwargs: Optional arguments to :class:`matplotlib.patches.Rectangle`.
        """
        if axes is None:
            axes = matplotlib.pyplot.gca()
        return axes.add_patch(
            matplotlib.patches.Rectangle(
                xy=box[0:2], width=box[2] - box[0], height=box[3] - box[1], **kwargs
            )
        )

    def set_plot_limits(self, box: Iterable[float] = None) -> None:
        """
        Set limits of current plot axes.

        Arguments:
            box: Plot limits in image coordinates (left, top, right, bottom).
                If `None`, uses the full extent of the images.
        """
        if box is None:
            box = (0, 0, self._grid.n[0], self._grid.n[1])
        matplotlib.pyplot.xlim(box[0::2])
        matplotlib.pyplot.ylim(box[1::2])

    def cache_images(self, index: Union[Iterable[int], slice] = slice(None)) -> None:
        """
        Cache image data.

        Arguments:
            index: Index of images.
        """
        for img in np.asarray(self.images)[index]:
            img.read(cache=True)

    def clear_images(self, index: Union[Iterable[int], slice] = slice(None)) -> None:
        """
        Clear cached image data.

        Arguments:
            index: Index of images.
        """
        for img in np.asarray(self.images)[index]:
            img.I = None

    def animate(
        self,
        uv: Iterable[float] = None,
        frames: Iterable[int] = None,
        size: Iterable[int] = (100, 100),
        interval: float = 200,
        subplots: dict = {},
        animation: dict = {},
    ) -> matplotlib.animation.FuncAnimation:
        """
        Animate image tiles centered around a target point.

        The left subplot shifts tiles based on the projected position of the
        point (marked as a red dot); this represents the corrected image alignment.
        The right subplot does not shift tiles; this represents the original
        uncorrected image alignment.

        Arguments:
            uv: Image coordinate (u, v) of the center of the tile in
                in the first image. If `None`, the image center is used.
            frames: Integer indices of the images to include.
                If `None`, includes all images.
            size: Size of the image tiles to plot (nx, ny).
            interval: Delay between frames in milliseconds.
            subplots: Additional arguments to :func:`matplotlib.pyplot.subplots`.
            animation: Additional arguments to
                :class:'matplotlib.animation.FuncAnimation'.
        """
        if uv is None:
            uv = self.images[0].cam.imgsz / 2
        if frames is None:
            frames = np.arange(len(self.images))
        dxyz = self.images[frames[0]].cam.uv_to_xyz(np.atleast_2d(uv))
        halfsize = (size[0] * 0.5, size[1] * 0.5)
        # Initialize plot
        fig, ax = matplotlib.pyplot.subplots(ncols=2, **subplots)
        box = self.tile_box(uv, size=size)
        tile = self.extract_tile(img=frames[0], box=box)
        im = [self.plot_tile(tile=tile, box=box, axes=axes) for axes in ax]
        pt = [axis.plot(uv[0], uv[1], marker=".", color="red")[0] for axis in ax]
        # NOTE: Frame label drawn inside axes due to limitations with blit=True
        # https://stackoverflow.com/questions/17558096/animated-title-in-matplotlib.
        txt = ax[0].text(
            0.5,
            0.95,
            "",
            color="white",
            horizontalalignment="center",
            transform=ax[0].transAxes,
        )
        ax[1].set_xlim(uv[0] - halfsize[0], uv[0] + halfsize[0])
        ax[1].set_ylim(uv[1] + halfsize[1], uv[1] - halfsize[0])

        # Update plot
        def update_plot(i: int) -> list:
            puv = self.images[i].cam.xyz_to_uv(dxyz, directions=True)[0]
            box = np.vstack([puv - halfsize, puv + halfsize]).ravel()
            inbounds = self.images[i].cam.inframe(helpers.box_to_polygon(box))
            if np.any(inbounds):
                if not np.all(inbounds):
                    # Intersect box with image bounds
                    box = helpers.intersect_boxes(
                        (box, np.concatenate(([0, 0], self.images[i].cam.imgsz)))
                    )
                box = self._grid.snap_xy(
                    helpers.unravel_box(box), centers=False, edges=True
                ).ravel()
                tile = self.extract_tile(img=i, box=box)
            else:
                # Use white tile
                tile = np.zeros((size[1], size[0], 3), dtype=np.uint8) + 255
            for j in range(2):
                im[j].set_array(tile)
                im[j].set_extent((box[0], box[2], box[3], box[1]))
                pt[j].set_xdata(puv[0])
                pt[j].set_ydata(puv[1])
            ax[0].set_xlim(puv[0] - halfsize[0], puv[0] + halfsize[0])
            ax[0].set_ylim(puv[1] + halfsize[1], puv[1] - halfsize[0])
            basename = helpers.strip_path(self.images[i].path)
            txt.set_text(str(i) + " : " + basename)
            return im + pt + [txt]

        # Build animation
        return matplotlib.animation.FuncAnimation(
            fig, update_plot, frames=frames, interval=interval, blit=True, **animation
        )

    def track(
        self,
        xyz: Iterable[float],
        frames: Iterable[int] = None,
        size: Iterable[int] = (100, 100),
        interval: float = 200,
        subplots: dict = {},
        animation: dict = {},
    ) -> matplotlib.animation.FuncAnimation:
        """
        Animate image tiles tracking a moving point.

        The left subplot shows the first image centered on the first point position
        (marked as a red dot). The right subplot shows the nth image centered on the nth
        point position (marked as a red dot) alongside previous positions (marked as a
        yellow line with dots).

        Arguments:
            xyz: World coordinates (n, [x, y, z]).
            frames: Integer indices of the images to include.
                If `None`, includes all images.
            size: Size of the image tiles to plot (nx, ny).
            interval: Delay between frames in milliseconds.
            subplots: Additional arguments to :func:`matplotlib.pyplot.subplots`.
            animation: Additional arguments to
                :class:`matplotlib.animation.FuncAnimation`.
        """
        if frames is None:
            frames = np.arange(len(xyz))
        halfsize = (size[0] * 0.5, size[1] * 0.5)
        # Initialize plot
        fig, ax = matplotlib.pyplot.subplots(ncols=2, **subplots)
        track_uv: np.ndarray = self.images[frames[0]].cam.xyz_to_uv(xyz[0:1])
        uv = track_uv[-1]
        box = self.tile_box(uv, size=size)
        tile = self.extract_tile(img=frames[0], box=box)
        im = [self.plot_tile(tile=tile, box=box, axes=axes, zorder=1) for axes in ax]
        track = ax[1].plot(track_uv[:, 0], track_uv[:, 1], "y.-", alpha=0.5, zorder=2)[
            0
        ]
        pt = [
            axis.plot(uv[0], uv[1], marker=".", color="red", zorder=3)[0] for axis in ax
        ]
        basename = helpers.strip_path(self.images[frames[0]].path)
        ax[0].text(
            0.5,
            0.95,
            "0 : " + basename,
            color="white",
            horizontalalignment="center",
            zorder=4,
            transform=ax[0].transAxes,
        )
        txt = ax[1].text(
            0.5,
            0.95,
            "",
            color="white",
            horizontalalignment="center",
            zorder=4,
            transform=ax[1].transAxes,
        )

        # Update plot
        def update_plot(i: int) -> list:
            j = np.where(frames == i)[0][0]
            track_uv: np.ndarray = self.images[i].cam.xyz_to_uv(xyz[: j + 1])
            uv = track_uv[-1]
            box = self.tile_box(uv, size=size)
            tile = self.extract_tile(img=i, box=box)
            im[1].set_array(tile)
            im[1].set_extent((box[0], box[2], box[3], box[1]))
            track.set_xdata(track_uv[:, 0])
            track.set_ydata(track_uv[:, 1])
            pt[1].set_xdata(uv[0])
            pt[1].set_ydata(uv[1])
            basename = helpers.strip_path(self.images[i].path)
            txt.set_x(uv[0])
            txt.set_y(uv[1] - (halfsize[1] - 10))
            txt.set_text(str(i) + " : " + basename)
            return im + [track] + pt + [txt]

        # Build animation
        return matplotlib.animation.FuncAnimation(
            fig, update_plot, frames=frames, interval=interval, blit=True, **animation
        )

    def subset(self, **kwargs: Any) -> "Observer":
        """
        Return a new Observer with a subset of the original images.

        Arguments:
            **kwargs: Optional arguments to :func:`helpers.select_datetimes`.
        """
        mask = helpers.select_datetimes(self.datetimes, **kwargs)
        images = np.asarray(self.images)[mask]
        return self.__class__(images, sigma=self.sigma, cache=self.cache)

    def split(
        self, n: Union[int, Iterable[datetime.datetime]], overlap: int = 1
    ) -> List["Observer"]:
        """
        Split into multiple Observers.

        Arguments:
            n: Number of equal-length Observers (int) or datetime breaks (iterable).
            overlap: Number of images from previous Observer to append to start
                of following Observer.
        """
        if np.iterable(n):
            breaks = np.unique(np.hstack((n, self.datetimes[[0, -1]])))
        else:
            dt = (self.datetimes[-1] - self.datetimes[0]) / n
            breaks = helpers.datetime_range(self.datetimes[0], self.datetimes[-1], dt)
        observers = []
        start = breaks[0]
        for i in range(len(breaks) - 1):
            observer = self.subset(start=start, end=breaks[i + 1])
            if overlap:
                lag = min(overlap, len(observer.datetimes))
                start = observer.datetimes[-lag]
            else:
                # HACK: Prevent overlap by moving start by smallest timedelta
                start = observer.datetimes[-1] + datetime.timedelta(microseconds=1)
            observers.append(observer)
        return observers
