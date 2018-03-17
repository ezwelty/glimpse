from .imports import (np, scipy, datetime, matplotlib, os)
from . import (helpers, dem)

class Observer(object):
    """
    An `Observer` contains a sequence of `Image` objects and the methods to compute
    the misfit between image subsets.

    Attributes:
        xyz (array): Position in world coordinates (`images[0].cam.xyz`)
        images (list): Image objects with equal camera position (xyz),
            focal length (f), image size (imgsz) and
            strictly increasing in time (`datetime`)
        datetimes (array): Image capture times
        sigma (float): Standard deviation of pixel values between images
            due to changes in illumination, deformation, or unresolved camera motion
        correction: Curvature and refraction correction (see `Camera.project()`)
        cache (bool): Whether to cache images on read
        grid (glimpse.dem.Grid): Grid object for operations on image coordinates
    """

    def __init__(self, images, sigma=0.3, correction=True, cache=True):
        self.xyz = images[0].cam.xyz
        for img in images[1:]:
            if any(img.cam.xyz != self.xyz):
                raise ValueError("Positions (xyz) are not equal")
            if any(img.cam.f != images[0].cam.f):
                raise ValueError("Focal lengths (f) are not equal")
            if any(img.cam.imgsz != images[0].cam.imgsz):
                raise ValueError("Image sizes (imgsz) are not equal")
        self.images = images
        self.datetimes = np.array([img.datetime for img in self.images])
        time_deltas = np.array([dt.total_seconds() for dt in np.diff(self.datetimes)])
        if any(time_deltas <= 0):
            raise ValueError("Image datetimes are not stricly increasing")
        self.sigma = sigma
        self.correction = correction
        self.cache = cache
        n = self.images[0].cam.imgsz
        self.grid = dem.Grid(n=n, xlim=(0, n[0]), ylim=(0, n[1]))

    def index(self, value, max_seconds=1):
        """
        Retrieve the index of an image.

        Arguments:
            value: Either Image object to find in `self.images` or
                Date and time to match against `self.datetimes` (datetime)
            max_seconds (float): If `value` is datetime,
                maximum time delta in seconds to be considered a match.
        """
        if isinstance(value, datetime.datetime):
            time_deltas = np.abs(value - self.datetimes)
            index = np.argmin(time_deltas)
            seconds = time_deltas[index].total_seconds()
            if seconds > max_seconds:
                raise IndexError("Nearest image out of range by " + str(seconds - max_seconds) + " s")
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
        return self.images[img].cam.project(xyz, directions=directions, correction=self.correction)

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
        cu = self.grid.x[0:tile.shape[0]] # x|cols
        cv = self.grid.y[0:tile.shape[1]] # y|rows
        # Interpolate at shifted center coordinates
        tile = np.atleast_3d(tile)
        for i in range(tile.shape[2]):
            f = scipy.interpolate.RectBivariateSpline(cv, cu, tile[:, :, i], **kwargs)
            tile[:, :, i] = f(cv + duv[1], cu + duv[0], grid=True)
        if tile.shape[2] is 1:
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
        cu = np.arange(box[0] + du * 0.5, box[2]) # x|cols
        cv = np.arange(box[1] + dv * 0.5, box[3]) # y|rows
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
            box (array-like): Boundaries of tile in image coordinates (left, top, right, bottom).
                If `None`, the upper-left corner of the upper-left pixel is placed at (0, 0).
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
        return axes.imshow(tile, origin='upper', extent=extent, **kwargs)

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
        return axes.add_patch(matplotlib.patches.Rectangle(
            xy=box[0:2], width=box[2] - box[0], height=box[3] - box[1],
            fill=fill, **kwargs))

    def set_plot_limits(self, box=None):
        """
        Set the x,y limits of the current matplotlib axes.

        Arguments:
            box (array-like): Plot limits in image coordinates (left, top, right, bottom).
                If `None`, uses the full extent of the images.
        """
        if box is None:
            box = (0, 0, self.grid.n[0], self.grid.n[1])
        matplotlib.pyplot.xlim(box[0::2])
        matplotlib.pyplot.ylim(box[1::2])

    def clear_cache(self):
        """
        Clear cached image data from all images.
        """
        for img in self.images:
            img.I = None

    def animate(self, uv, frames=None, size=(100, 100), interval=200, subplots=dict(), animation=dict()):
        """
        Animate image tiles centered around a target point.

        The left subplot shifts tiles based on the projected position of the
        point (marked as a red dot); this represents the corrected image alignment.
        The right subplot does not shift tiles; this represents the original
        uncorrected image alignment.

        NOTE: The frame label (index, image basenmae) is drawn inside the axes
        due to limitations of 'matplotlib.animation.FuncAnimation(blit=True)'.
        See https://stackoverflow.com/questions/17558096/animated-title-in-matplotlib.

        Arguments:
            uv (iterable): Image coordinate (u, v) of the center of the tile in
                in the first image (`frames[0]`)
            frames (iterable): Integer indices of the images to include
            size (iterable): Size of the image tiles to plot
            interval (number): Delay between frames in milliseconds
            subplots (dict): Additional arguments to `matplotlib.pyplot.subplots()`
            animation (dict): Additional arguments to 'matplotlib.animation.FuncAnimation()'

        Returns:
            `matplotlib.animation.FuncAnimation`
        """
        if frames is None:
            frames = range(len(self.images))
        dxyz = self.images[frames[0]].cam.invproject(np.atleast_2d(uv))
        halfsize = (size[0] * 0.5, size[1] * 0.5)
        # Initialize plot
        fig, ax = matplotlib.pyplot.subplots(ncols=2, **subplots)
        box = self.tile_box(uv, size=size)
        tile = self.extract_tile(img=frames[0], box=box)
        im = [self.plot_tile(tile=tile, box=box, axes=axes) for axes in ax]
        pt = [axis.plot(uv[0], uv[1], marker='.', color='red')[0] for axis in ax]
        txt = ax[1].text(uv[0], uv[1] - (halfsize[1] - 10), '', color='white',
            horizontalalignment='center')
        # Update plot
        def update_plot(i):
            puv = self.images[i].cam.project(dxyz, directions=True)[0]
            tile = self.extract_tile(img=i, box=box)
            for j in range(2):
                im[j].set_array(tile)
                pt[j].set_xdata(puv[0])
                pt[j].set_ydata(puv[1])
            ax[0].set_xlim(puv[0] - halfsize[0], puv[0] + halfsize[0])
            ax[0].set_ylim(puv[1] + halfsize[1], puv[1] - halfsize[0])
            basename = os.path.splitext(os.path.basename(self.images[i].path))[0]
            txt.set_text(str(i) + ' : ' + basename)
            return im + pt + [txt]
        # Build animation
        return matplotlib.animation.FuncAnimation(fig, update_plot, frames=frames, interval=interval, blit=True, **animation)
