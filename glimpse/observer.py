import datetime

import matplotlib.patches
import matplotlib.pyplot
import numpy as np
import scipy.interpolate

from . import helpers
from .raster import Grid


class Observer(object):
    """
    An `Observer` contains a sequence of `Image` objects and the methods to compute
    the misfit between image subsets.

    Attributes:
        xyz (array): Position in world coordinates (`images[0].cam.xyz`)
        images (list): Image objects with equal camera position (xyz),
            focal length (f), image size (imgsz) and
            strictly increasing in time (`datetime`)
        datetimes (array): Image capture times,
            by default read from `images[i].datetime`
        sigma (float): Standard deviation of pixel values between images
            due to changes in illumination, deformation, or unresolved camera motion
        correction: Curvature and refraction correction (see `Camera.project()`)
        cache (bool): Whether to cache images on read
        grid (glimpse.raster.Grid): Grid object for operations on image coordinates
    """

    def __init__(self, images, datetimes=None, sigma=0.3, correction=True, cache=True):
        if len(images) < 2:
            raise ValueError("Observer must have two or more images")
        self.xyz = images[0].cam.xyz
        self.test_images(images)
        self.images = images
        if datetimes is None:
            datetimes = np.array([img.datetime for img in self.images])
        else:
            datetimes = np.asarray(datetimes)
        time_deltas = np.array([dt.total_seconds() for dt in np.diff(datetimes)])
        if any(time_deltas <= 0):
            raise ValueError("Image datetimes are not stricly increasing")
        self.datetimes = datetimes
        self.sigma = sigma
        self.correction = correction
        self.cache = cache
        n = self.images[0].cam.imgsz.astype(int)
        if (n != self.images[0].cam.imgsz).any():
            raise ValueError("Image sizes (imgsz) are not integer")
        self.grid = Grid(n=n, x=(0, n[0]), y=(0, n[1]))

    @staticmethod
    def test_images(images, cam_tol=1e-3):
        for img in images[1:]:
            if np.linalg.norm(img.cam.xyz - images[0].cam.xyz) > cam_tol:
                raise ValueError("Positions (xyz) are not equal")
            if any(img.cam.f != images[0].cam.f):
                raise ValueError("Focal lengths (f) are not equal")
            if any(img.cam.imgsz != images[0].cam.imgsz):
                raise ValueError("Image sizes (imgsz) are not equal")

    def index(self, value, maxdt=datetime.timedelta(0)):
        """
        Retrieve the index of an image.

        Arguments:
            value: Either Image object to find in `self.images` or
                Date and time to match against `self.datetimes` (datetime)
            maxdt (timedelta): Maximum timedelta for `value` (datetime) to be
                considered a match. If `None`, no limit is placed on the match.
        """
        if isinstance(value, datetime.datetime):
            dts = np.abs(value - self.datetimes)
            index = np.argmin(dts)
            if maxdt is not None and dts[index] > abs(maxdt):
                raise IndexError(
                    "Nearest image out of range by " + str(dts[index] - abs(maxdt))
                )
            return index
        else:
            return self.images.index(value)

    def project(self, xyz, img, directions=False):
        """
        Project world coordinates to image coordinates.

        Arguments:
            xyz (array): World coordinates (Nx3) or camera coordinates (Nx2)
            img: Index of Image to project into
            directions (bool): Whether `xyz` are absolute coordinates (False)
                or ray directions (True)
        """
        return self.images[img].cam.project(
            xyz, directions=directions, correction=self.correction
        )

    def tile_box(self, uv, size=(1, 1)):
        """
        Compute a grid-aligned box centered around a point.

        Arguments:
            uv (array-like): Desired box center in image coordinates (u, v)
            size (array-like): Size of box in pixels (width, height)

        Returns:
            array: Integer (pixel edge) boundaries (left, top, right, bottom)
        """
        return self.grid.snap_box(uv, size, centers=False, edges=True).astype(int)

    def extract_tile(self, box, img, cache=None):
        """
        Extract rectangular image region.

        Cached results are slowest the first time (the full image is read),
        but fastest on subsequent reads.
        Non-cached results are read with a speed proportional to the size of the box.

        Arguments:
            box (array-like): Boundaries of tile in image coordinates
                (left, top, right, bottom)
            img: Index of Image to read
            cache (bool): Optional override of `self.cache`
        """
        if cache is None:
            cache = self.cache
        return self.images[img].read(box=box, cache=cache)

    def shift_tile(self, tile, duv, **kwargs):
        """
        Shift tile by a half-pixel (or smaller) offset.

        Useful for centering a tile over an arbitrary center point.

        Arguments:
            tile (array): 2-d or 3-d array
            duv (array-like): Shift in image coordinates (du, dv).
                Must be 0.5 pixels or smaller in each dimension.
            **kwargs: Optional arguments to scipy.interpolate.RectBivariateSpline
        """
        if any(np.abs(duv) > 0.5):
            raise ValueError("Shift larger than 0.5 pixels")
        # Cell center coordinates (arbitrary origin)
        cu = self.grid.x[0 : tile.shape[0]]  # x|cols
        cv = self.grid.y[0 : tile.shape[1]]  # y|rows
        # Interpolate at shifted center coordinates
        tile = np.atleast_3d(tile)
        for i in range(tile.shape[2]):
            f = scipy.interpolate.RectBivariateSpline(cv, cu, tile[:, :, i], **kwargs)
            tile[:, :, i] = f(cv + duv[1], cu + duv[0], grid=True)
        if tile.shape[2] == 1:
            return tile.squeeze(axis=2)
        else:
            return tile

    def sample_tile(self, uv, tile, box, grid=False, **kwargs):
        """
        Sample tile at image coordinates.

        Arguments:
            uv (array-like): Image coordinates as either points (Nx2) if `grid=False`
                or an iterable of grid coordinate arrays (u, v) if `grid=True`
            tile (array): 2-d array
            box (array-like): Boundaries of tile in image coordinates
                (left, top, right, bottom)
            grid (bool): See `uv`
            **kwargs: Optional arguments to scipy.interpolate.RectBivariateSpline
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
        else:
            return f(uv[:, 1], uv[:, 0], grid=grid)

    def plot_tile(self, tile, box=None, axes=None, **kwargs):
        """
        Draw tile on current matplotlib axes.

        Arguments:
            tile (array): 2-d or 3-d array
            box (array-like): Boundaries of tile in image coordinates
                (left, top, right, bottom). If `None`, the upper-left corner of the
                upper-left pixel is placed at (0, 0).
            axes (`matplotlib.axes.Axes`): Matplotlib axes to plot on
            **kwargs: Optional arguments to matplotlib.pyplot.imshow

        Returns:
            `matplotlib.image.AxesImage`
        """
        if box is None:
            box = (0, 0, tile.shape[0], tile.shape[1])
        extent = (box[0], box[2], box[3], box[1])
        if axes is None:
            axes = matplotlib.pyplot
        return axes.imshow(tile, origin="upper", extent=extent, **kwargs)

    def plot_box(self, box, fill=False, axes=None, **kwargs):
        """
        Draw box on current matplotlib axes.

        Arguments:
            box (array-like): Box in image coordinates (left, top, right, bottom)
            fill (bool): Whether to fill the box
            axes (`matplotlib.axes.Axes`): Matplotlib axes to plot on
            **kwargs: Optional arguments to matplotlib.patches.Rectangle

        Returns:
            `matplotlib.patches.Rectangle`
        """
        if axes is None:
            axes = matplotlib.pyplot.gca()
        return axes.add_patch(
            matplotlib.patches.Rectangle(
                xy=box[0:2],
                width=box[2] - box[0],
                height=box[3] - box[1],
                fill=fill,
                **kwargs
            )
        )

    def set_plot_limits(self, box=None):
        """
        Set the x,y limits of the current matplotlib axes.

        Arguments:
            box (array-like): Plot limits in image coordinates
                (left, top, right, bottom).
                If `None`, uses the full extent of the images.
        """
        if box is None:
            box = (0, 0, self.grid.n[0], self.grid.n[1])
        matplotlib.pyplot.xlim(box[0::2])
        matplotlib.pyplot.ylim(box[1::2])

    def cache_images(self, index=None):
        """
        Cache image data.

        Arguments:
            index (iterable or slice): Index of images, or all if `None`
        """
        if index is None:
            index = slice(None)
        for img in np.array(self.images)[index]:
            img.read(cache=True)

    def clear_images(self, index=None):
        """
        Clear cached image data.

        Arguments:
            index (iterable or slice): Index of images, or all if `None`
        """
        if index is None:
            index = slice(None)
        for img in np.array(self.images)[index]:
            img.I = None

    def animate(
        self,
        uv=None,
        frames=None,
        size=(100, 100),
        interval=200,
        subplots={},
        animation={},
    ):
        """
        Animate image tiles centered around a target point.

        The left subplot shifts tiles based on the projected position of the
        point (marked as a red dot); this represents the corrected image alignment.
        The right subplot does not shift tiles; this represents the original
        uncorrected image alignment.

        NOTE: The frame label ('<image index>: <image basename>') is drawn inside the
        axes due to limitations of 'matplotlib.animation.FuncAnimation(blit=True)'. See
        https://stackoverflow.com/questions/17558096/animated-title-in-matplotlib.

        Arguments:
            uv (iterable): Image coordinate (u, v) of the center of the tile in
                in the first image (`frames[0]`). If `None`, the image center is used.
            frames (iterable): Integer indices of the images to include
            size (iterable): Size of the image tiles to plot
            interval (number): Delay between frames in milliseconds
            subplots (dict): Additional arguments to `matplotlib.pyplot.subplots()`
            animation (dict): Additional arguments to
                'matplotlib.animation.FuncAnimation()'

        Returns:
            `matplotlib.animation.FuncAnimation`
        """
        if uv is None:
            uv = self.images[0].cam.imgsz / 2
        if frames is None:
            frames = np.arange(len(self.images))
        dxyz = self.images[frames[0]].cam.invproject(np.atleast_2d(uv))
        halfsize = (size[0] * 0.5, size[1] * 0.5)
        # Initialize plot
        fig, ax = matplotlib.pyplot.subplots(ncols=2, **subplots)
        box = self.tile_box(uv, size=size)
        tile = self.extract_tile(img=frames[0], box=box)
        im = [self.plot_tile(tile=tile, box=box, axes=axes) for axes in ax]
        pt = [axis.plot(uv[0], uv[1], marker=".", color="red")[0] for axis in ax]
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
        def update_plot(i):
            puv = self.images[i].cam.project(dxyz, directions=True)[0]
            box = np.vstack([puv - halfsize, puv + halfsize]).ravel()
            inbounds = self.images[i].cam.inframe(helpers.box_to_polygon(box))
            if np.any(inbounds):
                if not np.all(inbounds):
                    # Intersect box with image bounds
                    box = helpers.intersect_boxes(
                        (box, np.concatenate(([0, 0], self.images[i].cam.imgsz)))
                    )
                box = self.grid.snap_xy(
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
        xyz,
        frames=None,
        size=(100, 100),
        interval=200,
        subplots={},
        animation={},
    ):
        """
        Animate image tiles tracking a moving point.

        The left subplot shows the first image centered on the first point position
        (marked as a red dot). The right subplot shows the nth image centered on the nth
        point position (marked as a red dot) alongside previous positions (marked as a
        yellow line with dots).

        NOTE: The frame labels (('<image index>: <image basename>')) are drawn inside
        the axes due to limitations of 'matplotlib.animation.FuncAnimation(blit=True)'.
        See https://stackoverflow.com/questions/17558096/animated-title-in-matplotlib.

        Arguments:
            xyz (array): World coordinates (x, y, z)
            frames (iterable): Integer indices of the images to include.
                If `None`, defaults to `range(len(xyz))`.
            size (iterable): Size of the image tiles to plot
            interval (number): Delay between frames in milliseconds
            subplots (dict): Additional arguments to `matplotlib.pyplot.subplots()`
            animation (dict): Additional arguments to
                'matplotlib.animation.FuncAnimation()'

        Returns:
            `matplotlib.animation.FuncAnimation`
        """
        if frames is None:
            frames = np.arange(len(xyz))
        halfsize = (size[0] * 0.5, size[1] * 0.5)
        # Initialize plot
        fig, ax = matplotlib.pyplot.subplots(ncols=2, **subplots)
        track_uv = self.images[frames[0]].cam.project(xyz[0:1])
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
        def update_plot(i):
            j = np.where(frames == i)[0][0]
            track_uv = self.images[i].cam.project(xyz[: (j + 1)])
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

    def subset(self, **kwargs):
        """
        Return a new Observer with a subset of the original images.

        Arguments:
            **kwargs: Arguments to `helpers.select_datetimes()`
        """
        index = helpers.select_datetimes(self.datetimes, **kwargs)
        images = [self.images[i] for i in index]
        params = {key: getattr(self, key) for key in ("sigma", "correction", "cache")}
        return self.__class__(images, datetimes=self.datetimes[index], **params)

    def split(self, n, overlap=1):
        """
        Split into multiple Observers.

        Arguments:
            n: Number of equal-length Observers (int) or datetime breaks (iterable)
            overlap (int): Number of images from previous Observer to append to start
                of following Observer
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
