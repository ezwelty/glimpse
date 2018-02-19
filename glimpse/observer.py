from .imports import (np, scipy, datetime, matplotlib)
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
        correction: Curvature and refraction correction (see `glimpse.Camera.project()`)
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

    def extract_tile(self, box, img=None, gray=False, highpass=False,
        template=False, subpixel=False, uv=None, cache=None):
        """
        Extract rectangular image region.

        Cached results are slowest the first time (the full image is read),
        but fastest on subsequent reads. A subset of the cached image is returned,
        so modifications are applied to the cached image.
        Non-cached results are read with a speed proportional to the size of the box.

        Optional operations are applied in the following order:
        match histogram to template (`template`), convert to grayscale (`gray`),
        correct subpixel offset (`subpixel`), apply high-pass filter (`highpass`).

        Arguments:
            box (array-like): Boundaries of tile in image coordinates
                (left, top, right, bottom)
            img: Index of Image to read
            gray (bool or dict): Whether to convert tile to grayscale.
                Either arguments to `helpers.rgb_to_gray` (dict),
                `True` for default arguments, or `False` to skip.
            template (bool or array-like): Histogram matching template
                (see `helpers.match_histogram`) or `False` to skip.
            highpass (bool or dict): Whether to apply a median high-pass filter.
                Either arguments to `scipy.ndimage.filters.median_filter`,
                `True` for default arguments, or `False` to skip.
            subpixel (bool or dict): Whether to correct for subpixel offset
                between desired and actual center of tile.
                Either arguments to `scipy.interpolate.RectBivariateSpline` (dict),
                `True` for default arguments, or `False` to skip.
            uv (array-like): Desired center of tile in image coordinates (u, v)
            cache (bool): Optional override of `self.cache`
        """
        # Apply defaults
        if cache is None:
            cache = self.cache
        if gray is True:
            gray = dict()
        if highpass is True:
            highpass = dict()
        if subpixel is True:
            subpixel = dict()
        # Extract image region
        # NOTE: Copy not necessary here in cases when copied later
        I = self.images[img].read(box=box, cache=cache).copy()
        # NOTE: Move operations to tracker? Different trackers may have different needs.
        if template is not False:
            # Match histogram to template
            if I.ndim < 3:
                I = helpers.match_histogram(I, template=template)
            else:
                if isinstance(template, np.ndarray):
                    template = np.dsplit(template, template.shape[2])
                for i in xrange(I.shape[2]):
                    I[:, :, i] = helpers.match_histogram(I[:, :, i], template[i])
        if gray is not False:
            # Convert to grayscale
            I = helpers.rgb_to_gray(I, **gray)
        if subpixel is not False and uv is not None:
            # FIXME: Works only for grayscale
            # Correct subpixel offset from desired center
            duv = uv - np.reshape(box, (2, -1)).mean(axis=0)
            I = self.shift_tile(I, duv=duv.flatten(), **subpixel)
        if highpass is not False:
            # Apply median high-pass filter
            I_low = scipy.ndimage.filters.median_filter(I, **highpass)
            I -= I_low
        return I

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

    def plot_tile(self, tile, box=None, **kwargs):
        """
        Draw tile on current matplotlib axes.

        Arguments:
            tile (array): 2-d or 3-d array
            box (array-like): Boundaries of tile in image coordinates (left, top, right, bottom).
                If `None`, the upper-left corner of the upper-left pixel is placed at (0, 0).
            **kwargs: Optional arguments to matplotlib.pyplot.imshow
        """
        if box is None:
            box = (0, 0, tile.shape[0], tile.shape[1])
        extent = (box[0], box[2], box[1], box[3])
        matplotlib.pyplot.imshow(tile, origin='upper', extent=extent, **kwargs)

    def plot_box(self, box, fill=False, **kwargs):
        """
        Draw box on current matplotlib axes.

        Arguments:
            box (array-like): Box in image coordinates (left, top, right, bottom)
            fill (bool): Whether to fill the box
            **kwargs: Optional arguments to matplotlib.patches.Rectangle
        """
        axis = matplotlib.pyplot.gca()
        axis.add_patch(matplotlib.patches.Rectangle(
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

    def initialize_plot(self, ax):
        self.ax = ax
        self.im_plot = ax.imshow(self.images[0].I, interpolation='none')
        self.le0, self.re0, self.be0, self.te0 = self.im_plot.get_extent()
        #self.spprd = ax.scatter(self.cols_predicted, self.rows_predicted, s=50, c='green', label='Prior Prediction')
        self.sppnts = self.ax.scatter(self.rc[:, 1], self.rc[:, 0], s=25, c=-self.log_like, cmap=matplotlib.pyplot.cm.gnuplot2, linewidths=0, alpha=0.2, vmin=-3., vmax=-1, label='Particle Position/Log-Likelihood')
        self.ax.legend()
        self.cb = matplotlib.pyplot.colorbar(self.sppnts, ax=self.ax, orientation='horizontal', aspect=30, pad=0.07)
        self.cb.set_label('Log-likelihood')
        self.cb.solids.set_edgecolor('face')
        self.cb.solids.set_alpha(1)
        self.row_0 = np.mean(self.rc[:, 0])
        self.col_0 = np.mean(self.rc[:, 1])
        self.re = ax.add_patch(matplotlib.patches.Rectangle((self.col_0 - self.hw_col, self.row_0 - self.hw_row), self.hw_col * 2 + 1, self.hw_row * 2 + 1, fill=False))
        ax.set_xlim(self.col_0 - 50, self.col_0 + 50)
        ax.set_ylim(self.row_0 + 50, self.row_0 - 50)

    def update_plot(self, t):
        try:
            image_index = self.image_index(t, max_seconds=0.1)
        except IndexError:
            return
        self.sppnts.remove()
        self.sppnts = self.ax.scatter(self.rc[:, 1], self.rc[:, 0], s=25, c=-self.log_like, cmap=matplotlib.pyplot.cm.gnuplot2, linewidths=0, alpha=0.2, vmin=-3., vmax=-1)
        self.row_1 = np.mean(self.rc[:, 0])
        self.col_1 = np.mean(self.rc[:, 1])
        self.re.set_bounds(self.col_1 - self.hw_col, self.row_1 - self.hw_row, 2 * self.hw_col + 1, 2 * self.hw_row + 1)
        col_offset = self.col_1 - self.col_0
        row_offset = self.row_1 - self.row_0
        self.im_plot.set_data(self.images[image_index].read())
        self.ax.set_xlim(self.col_1 - 50, self.col_1 + 50)
        self.ax.set_ylim(self.row_1 + 50, self.row_1 - 50)
        #self.im_plot.set_extent((self.le0 + col_offset, self.re0 + col_offset, self.be0 + row_offset, self.te0 + row_offset))
