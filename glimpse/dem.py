from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, scipy, gdal, matplotlib, datetime)
from . import (helpers)

class Grid(object):

    def __init__(self, n, xlim=None, ylim=None):
        self.n = np.atleast_1d(n).astype(int)
        if len(self.n) < 2:
            self.n = np.repeat(self.n, 2)
        if xlim is None:
            xlim = (0, self.n[0])
        self.xlim = np.atleast_1d(xlim).astype(float)
        if ylim is None:
            ylim = (0, self.n[1])
        self.ylim = np.atleast_1d(ylim).astype(float)
        # Placeholders
        self._x, self._y = None, None
        self._X, self._Y = None, None

    # ---- Properties (dependent) ---- #

    @property
    def d(self):
        return np.hstack((np.diff(self.xlim), np.diff(self.ylim))) / self.n

    @property
    def min(self):
        return np.array((min(self.xlim), min(self.ylim)))

    @property
    def max(self):
        return np.array((max(self.xlim), max(self.ylim)))

    @property
    def x(self):
        if self._x is None:
            value = np.linspace(self.min[0] + abs(self.d[0]) / 2, self.max[0] - abs(self.d[0]) / 2, self.n[0])
            if self.d[0] < 0:
                self._x = value[::-1]
            else:
                self._x = value
        return self._x

    @property
    def X(self):
        if self._X is None:
            self._X = np.tile(self.x, (self.n[1], 1))
        return self._X

    @property
    def y(self):
        if self._y is None:
            value = np.linspace(self.min[1] + abs(self.d[1]) / 2, self.max[1] - abs(self.d[1]) / 2, self.n[1])
            if self.d[1] < 0:
                self._y = value[::-1]
            else:
                self._y = value
        return self._y

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.tile(self.y, (self.n[0], 1)).T
        return self._Y

    @classmethod
    def read(cls, path, d=None, xlim=None, ylim=None):
        """
        Read Grid from raster file.

        Arguments:
            path (str): Path to file
            d (float): Target grid cell size
            xlim (array-like): Crop bounds in x.
                If `None` (default), read from file.
            ylim (array-like): Crop bounds in y.
                If `None` (default), read from file.
        """
        raster = gdal.Open(path, gdal.GA_ReadOnly)
        transform = raster.GetGeoTransform()
        grid = cls(n=(raster.RasterXSize, raster.RasterYSize),
            xlim=transform[0] + transform[1] * np.array([0, raster.RasterXSize]),
            ylim=transform[3] + transform[5] * np.array([0, raster.RasterYSize]))
        xlim, ylim, rows, cols = grid.crop_extent(xlim=xlim, ylim=ylim)
        win_xsize = (cols[1] - cols[0]) + 1
        win_ysize = (rows[1] - rows[0]) + 1
        if d:
            buf_xsize = int(np.ceil(abs(win_xsize * grid.d[0] / d)))
            buf_ysize = int(np.ceil(abs(win_ysize * grid.d[1] / d)))
        else:
            buf_xsize = win_xsize
            buf_ysize = win_ysize
        grid.xlim, grid.ylim = xlim, ylim
        grid.n = (buf_xsize, buf_ysize)
        return grid

    # ---- Methods (private) ----

    def _clear_cache(self, attributes=['x', 'X', 'y', 'Y']):
        for attr in attributes:
            setattr(self, '_' + attr, None)

    def _parse_x(self, obj):
        """
        Parse object into xlim, x, and X attributes.

        Arguments:
            obj (object): Either xlim, x, or X
        """
        if obj is None:
            obj = [0, self.n[0]]
        if not isinstance(obj, np.ndarray):
            obj = np.asarray(obj, float)
        is_X = len(obj.shape) > 1 and all(n > 1 for n in obj.shape[0:2])
        if is_X:
            # TODO: Check if all rows equal
            X = obj
            obj = obj[0, :]
        else:
            X = None
        is_x = any(n > 2 for n in obj.shape[0:2])
        if is_x:
            x = obj
            # TODO: Check if equally spaced monotonic
            dx = abs(np.diff(obj[0:2]))
            xlim = np.append(obj[0] - dx / 2, obj[-1] + dx / 2)
        else:
            x = None
            xlim = obj
        return [xlim, x, X]

    def _parse_y(self, obj):
        """
        Parse object into ylim, y, and Y attributes.

        Arguments:
            obj (object): Either ylim, y, or Y
        """
        if obj is None:
            obj = [0, self.n[1]]
        if not isinstance(obj, np.ndarray):
            obj = np.asarray(obj, float)
        is_Y = len(obj.shape) > 1 and all(n > 1 for n in obj.shape[0:2])
        if is_Y:
            # TODO: Check if all rows equal
            Y = obj
            obj = obj[:, 0]
        else:
            Y = None
        is_y = any(n > 2 for n in obj.shape[0:2])
        if is_y:
            y = obj
            # TODO: Check if equally spaced monotonic
            dy = abs(np.diff(obj[0:2]))
            ylim = np.append(obj[0] + dy / 2, obj[-1] - dy / 2)
        else:
            y = None
            ylim = obj
        return [ylim, y, Y]

    # ---- Methods ---- #

    def copy(self):
        return Grid(n=self.n, xlim=self.xlim, y=self.ylim)

    def resize(self, scale):
        """
        Resize grid.

        Grid cell aspect ratio may not be preserved due to integer rounding
        of grid dimensions.

        Arguments:
            scale (float): Fraction of current size
        """
        self.n = np.floor(self.n * scale + 0.5).astype(int)

    def inbounds(self, xy):
        """
        Test whether points are in (or on) bounds.

        Arguments:
            xy (array): Point coordinates (Nx2)

        Returns:
            array: Whether each point is inbounds (Nx1)
        """
        return np.logical_and(xy >= self.min[0:2], xy <= self.max[0:2]).all(axis = 1)

    def snap_xy(self, xy, centers=False, edges=False, inbounds=True):
        """
        Snap x,y coordinates to nearest grid position.

        When snapping to cell centers, points on edges snap to higher grid indices.
        If `inbounds=True`, points on right or bottom edges snap down to stay in bounds.

        Arguments:
            xy (array): Point coordinates (Nx2)
            centers (bool): Whether to snap to nearest cell centers
            edges (bool): Whether to snap to nearest cell edges
            inbounds (bool): Whether to snap points on right and bottom edges
                to interior cell centers
        """
        # TODO: Faster version for image grid
        if not centers and not edges:
            raise ValueError('centers and edges cannot both be false')
        origin = np.append(self.xlim[0], self.ylim[0])
        nxy = (xy - origin) / self.d
        if centers and not edges:
            nxy -= 0.5
        elif centers and edges:
            nxy *= 2
        nxy = np.floor(nxy + 0.5)
        if not edges and inbounds:
            # Points on right or bottom edges snap down to stay in bounds
            is_outer_edge = xy == np.append(self.xlim[1], self.ylim[1])
            nxy[is_outer_edge] -= 1
        if centers and not edges:
            nxy += 0.5
        elif centers and edges:
            nxy /= 2
        return nxy * self.d + origin

    def snap_box(self, xy, size, centers=False, edges=True, inbounds=True):
        """
        Snap x,y box boundaries to nearest grid positions.

        Arguments:
            xy (array-like): Point coordinates of desired box center (x, y)
            size (array-like): Size of desired box in `xy` units (nx, ny)
            centers (bool): Whether to snap to nearest cell centers
            edges (bool): Whether to snap to nearest cell edges
            inbounds (bool): See `self.snap_xy()`

        Returns:
            array: x,y box boundaries (xmin, ymin, xmax, ymax)
        """
        halfsize = np.multiply(size, 0.5)
        xy_box = np.vstack((xy - halfsize, xy + halfsize))
        if any(~self.inbounds(xy_box)):
            raise IndexError('Sample extends beyond grid bounds')
        return self.snap_xy(xy_box, centers=centers, edges=edges, inbounds=inbounds).flatten()

    def rowcol_to_xy(self, rowcol):
        """
        Return x,y coordinates of row,col indices.

        Places integer indices at the centers of each cell.
        Therefore, the upper left corner is [-0.5, -0.5]
        and the center of that cell is [0, 0].

        Arguments:
            rowcol (array): Array indices (Nx2)
        """
        xy_origin = np.array((self.xlim[0], self.ylim[0]))
        return (rowcol + 0.5)[:, ::-1] * self.d + xy_origin

    def xy_to_rowcol(self, xy, snap=False, inbounds=True):
        """
        Return row,col indices of x,y coordinates.

        Arguments:
            xy (array): Spatial coordinates (Nx2)
            snap (bool): Whether to snap indices to nearest cell centers
                (see `self.snap_xy()`)
            inbounds (bool): See `self.snap_xy()`

        Returns:
            array: row,col indices as either float (`snap=False`)
                or int (`snap=True`)
        """
        # TODO: Remove snapping from function (now a seperate operation)
        if snap:
            xy = self.snap_xy(xy, centers=True, edges=False, inbounds=inbounds)
        origin = np.append(self.xlim[0], self.ylim[0])
        colrow = (xy - origin) / self.d - 0.5
        if snap:
            colrow = colrow.astype(int)
        return colrow[:, ::-1]

    def rowcol_to_idx(self, rowcol):
        return np.ravel_multi_index((rowcol[:, 0], rowcol[:, 1]), self.n[::-1])

    def idx_to_rowcol(self, idx):
        return np.column_stack(np.unravel_index(idx, self.n[::-1]))

    def crop_extent(self, xlim=None, ylim=None):
        # Calculate x,y limits
        if xlim is None:
            xlim = self.xlim
        if ylim is None:
            ylim = self.ylim
        box = helpers.intersect_boxes(np.vstack((
            np.hstack((min(xlim), min(ylim), max(xlim), max(ylim))),
            np.hstack((self.min[0:2], self.max[0:2]))
        )))
        xlim = box[0::2]
        if self.xlim[0] > self.xlim[1]:
             xlim = xlim[::-1]
        ylim = box[1::2]
        if self.ylim[0] > self.ylim[1]:
             ylim = ylim[::-1]
        # Convert xy limits to grid indices
        xy = np.column_stack((xlim, ylim))
        rowcol = self.xy_to_rowcol(xy, snap=True)
        # Snap down bottom-right, non-outer edges
        # see .xy_to_rowcol()
        bottom_right = np.append(self.xlim[1], self.ylim[1])
        is_edge = (bottom_right - xy[1, :]) % self.d == 0
        is_outer_edge = xy[1, :] == bottom_right
        snap_down = (is_edge & ~is_outer_edge)
        rowcol[1, snap_down[::-1]] -= 1
        new_xy = self.rowcol_to_xy(rowcol)
        new_xlim = new_xy[:, 0] + np.array([-0.5, 0.5]) * self.d[0]
        new_ylim = new_xy[:, 1] + np.array([-0.5, 0.5]) * self.d[1]
        return new_xlim, new_ylim, rowcol[:, 0], rowcol[:, 1]

    def set_plot_limits(self):
        matplotlib.pyplot.xlim(self.xlim[0], self.xlim[1])
        matplotlib.pyplot.ylim(self.ylim[0], self.ylim[1])

class DEM(Grid):
    """
    A `DEM` describes elevations on a regular 2-dimensional grid.

    Arguments:
        x (object): Either `xlim`, `x`, or `X`
        y (object): Either `ylim`, `y`, or `Y`

    Attributes:
        Z (array): Grid of values on a regular xy grid
        zlim (array): Limits of `Z` [min, max]
        xlim,ylim (array): Outer bounds of the grid [left, right], [top, bottom]
        x,y (array): Cell center coordinates as row vectors [left to right], [top to bottom]
        X,Y (array): Cell center coordinates as grids equivalent to `Z`
        min (array): Minimum bounding box coordinates [x, y, z]
        max (array): Maximum bounding box coordinates [x, y, z]
        n (array): Dimensions of `Z` [nx|cols, ny|rows]
        d (array): Grid cell size [dx, dy]
        datetime (datetime): Capture date and time
    """

    def __init__(self, Z, x=None, y=None, datetime=None):
        self.Z = Z
        self._Zf = None
        self.xlim, self._x, self._X = self._parse_x(x)
        self.ylim, self._y, self._Y = self._parse_y(y)
        self.datetime = datetime

    @classmethod
    def read(cls, path, band=1, d=None, xlim=None, ylim=None, datetime=None):
        """
        Read DEM from raster file.

        See `gdal.Open()` for details.
        If raster is float and has a defined no-data value,
        no-data values are replaced with `np.nan`.
        Otherwise, the raster data is unchanged.

        Arguments:
            path (str): Path to file
            band (int): Raster band to read (1 = first band)
            d (float): Target grid cell size
            xlim (array-like): Crop bounds in x.
                If `None` (default), read from file.
            ylim (array-like): Crop bounds in y.
                If `None` (default), read from file.
            datetime (datetime): Capture date and time
        """
        raster = gdal.Open(path, gdal.GA_ReadOnly)
        transform = raster.GetGeoTransform()
        grid = Grid(
            n=(raster.RasterXSize, raster.RasterYSize),
            xlim=transform[0] + transform[1] * np.array([0, raster.RasterXSize]),
            ylim=transform[3] + transform[5] * np.array([0, raster.RasterYSize]))
        xlim, ylim, rows, cols = grid.crop_extent(xlim=xlim, ylim=ylim)
        win_xsize = (cols[1] - cols[0]) + 1
        win_ysize = (rows[1] - rows[0]) + 1
        if d:
            buf_xsize = np.ceil(abs(win_xsize * grid.d[0] / d))
            buf_ysize = np.ceil(abs(win_ysize * grid.d[1] / d))
        else:
            buf_xsize = win_xsize
            buf_ysize = win_ysize
        band = raster.GetRasterBand(band)
        Z = band.ReadAsArray(
            # ReadAsArray() requires int, not numpy.int#
            xoff=int(cols[0]), yoff=int(rows[0]),
            win_xsize=int(win_xsize), win_ysize=int(win_ysize),
            buf_xsize=int(buf_xsize), buf_ysize=int(buf_ysize))
        # FIXME: band.GetNoDataValue() not equal to read values due to rounding
        nan_value = band.GetNoDataValue()
        if np.issubdtype(Z.dtype, np.floating) and nan_value:
            Z[Z == nan_value] = np.nan
        return cls(Z, x=xlim, y=ylim, datetime=datetime)

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, value):
        if hasattr(self, '_Z'):
            original_shape = self._Z.shape
            self._Z = np.asarray(value, float)
            self._clear_cache(['Zf'])
            if self._Z.shape != original_shape:
                self._clear_cache(['x', 'X', 'y', 'Y'])
        else:
            self._Z = np.asarray(value, float)

    # ---- Properties (dependent) ----

    @property
    def zlim(self):
        value = [np.nanmin(self.Z), np.nanmax(self.Z)]
        return np.array(value)

    @property
    def n(self):
        return np.array(self.Z.shape[0:2][::-1])

    # ---- Properties (cached) ----

    @property
    def Zf(self):
        if self._Zf is None:
            sign = np.sign(self.d).astype(int)
            self._Zf = scipy.interpolate.RegularGridInterpolator(
                (self.x[::sign[0]], self.y[::sign[1]]), self.Z.T[::sign[0], ::sign[1]],
                method='linear', bounds_error=False)
        return self._Zf

    # ---- Methods (public) ----

    def copy(self):
        return DEM(self.Z, x=self.xlim, y=self.ylim, datetime=self.datetime)

    def sample(self, xy, indices=False, method='linear'):
        """
        Sample `Z` at points.

        Arguments:
            xy (array): Point coordinates (Nx2)
            method (str): Interpolation method,
                either 'linear' (default) or 'nearest'
        """
        if indices:
            return self.Z.flat[self.rowcol_to_idx(xy)]
        else:
            return self.Zf(xy, method=method)

    def resample(self, dem, method='linear'):
        xy = np.column_stack((dem.X.ravel(), dem.Y.ravel()))
        Z = self.sample(xy, method=method)
        self.Z = Z.reshape(dem.Z.shape)
        self.xlim, self._x, self._X = self._parse_x(dem.X)
        self.ylim, self._y, self._Y = self._parse_y(dem.Y)

    def plot(self, array=None, cmap='gray', **kwargs):
        """
        Plot a DEM.

        Arguments:
            array (array): Values to plot. If `None`, `.Z` is used.
            kwargs (dict): Arguments passed to `matplotlib.pyplot.imshow()`
        """
        if array is None:
            array = self.Z
        matplotlib.pyplot.imshow(array,
            extent=(self.xlim[0], self.xlim[1], self.ylim[1], self.ylim[0]),
            cmap=cmap, **kwargs)

    def hillshade(self, azimuth=315, altitude=45, **kwargs):
        """
        Return the illumination intensity of the surface.

        Arguments:
            azimuth (number): Azimuth angle of the light source
                (0-360, degrees clockwise from North)
            altitude (number): Altitude angle of the light source
                (0-90, degrees up from horizontal)
            kwargs (dict): Arguments passed to `matplotlib.colors.LightSource.hillshade()`
        """
        light = matplotlib.colors.LightSource(azdeg=azimuth, altdeg=altitude)
        return light.hillshade(self.Z, dx=self.d[0], dy=self.d[1], **kwargs)

    def rasterize(self, xy, values, fun=np.mean):
        """
        Convert points to a raster image.

        Arguments:
            xy (array): Point coordinates (Nx2)
            values (array): Point values
            fun (function): Aggregate function to apply to values of overlapping points
        """
        is_in = self.inbounds(xy)
        rowcol = self.xy_to_rowcol(xy[is_in, :], snap=True)
        return helpers.rasterize_points(rowcol[:, 0], rowcol[:, 1],
            values[is_in], self.Z.shape, fun=fun)

    def crop(self, xlim=None, ylim=None, zlim=None):
        """
        Crop a `DEM`.

        Arguments:
            xlim (array_like): Crop bounds in x
            ylim (array_like): Crop bounds in y
            zlim (array_like): Crop bounds in z.
                Values outside range are set to `np.nan`.
        """
        if xlim is not None or ylim is not None:
            xlim, ylim, rows, cols = self.crop_extent(xlim=xlim, ylim=ylim)
            Z = self.Z[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1]
            self.Z = Z
            self.xlim = xlim
            self.ylim = ylim
        if zlim is not None:
            self.Z[(self.Z < min(zlim)) | (self.Z > max(zlim))] = np.nan

    def resize(self, scale):
        """
        Resize a `DEM`.

        Arguments:
            scale (float): Fraction of current size
        """
        self.Z = scipy.ndimage.zoom(self.Z, zoom=float(scale), order=1)

    def fill_crevasses(self, maximum_filter_size=5, gaussian_filter_sigma=5,
        mask=None, fill=False):
        """
        Apply a maximum filter to `Z`, then perform Gaussian smoothing.

        Arguments:
            maximum_filter_size (int): Kernel size of maximum filter in pixels
            gaussian_filter_sigma (float): Standard deviation of Gaussian filter
            mask (array): Boolean mask of cells to include (True) or exclude (False).
                If `None`, all cells are included.
            fill (bool): Whether to fill cells excluded by `mask` with interpolated values
        """
        self.Z = helpers.gaussian_filter(
            helpers.maximum_filter(self.Z, size=maximum_filter_size, mask=mask, fill=fill),
            sigma=gaussian_filter_sigma, mask=mask, fill=fill)

    def viewshed(self, origin, correction=False):
        """
        Return the binary viewshed from a point within the DEM.

        Arguments:
            origin (iterable): World coordinates of viewing position (x, y, z)
            correction (dict or bool): Either arguments to `helpers.elevation_corrections()`,
                `True` for default arguments, or `None` or `False` to skip.

        Returns:
            array: Boolean array of the same shape as `self.Z`
                with visible cells tagged as `True`
        """
        assert abs(self.d).all()
        assert self.inbounds(np.atleast_2d(origin[0:2]))
        # Compute distance to all cell centers
        dx = np.tile(self.x - origin[0], self.n[1])
        dy = np.repeat(self.y - origin[1], self.n[0])
        dz = self.Z.ravel() - origin[2]
        dxy = dx**2 + dy**2 # wait to square root
        if correction is True:
            correction = dict()
        if isinstance(correction, dict):
            dz += helpers.elevation_corrections(
                squared_distances=dxy, **correction)
        dxy = np.sqrt(dxy)
        dxy_cell = (dxy * (1 / abs(self.d[0])) + 0.5).astype(int)
        # Compute heading (-pi to pi CW from -y axis)
        heading = np.arctan2(dy, dx)
        # Sort cells by distance, then heading
        ix = np.lexsort((heading, dxy_cell))
        dxy_cell_sorted = dxy_cell[ix]
        # Compute start and end indices of each ring
        rings = np.flatnonzero(np.diff(dxy_cell_sorted)) + 1
        if len(rings):
            if dxy_cell_sorted[0]:
                # Include first ring
                rings = np.hstack((0, rings))
        else:
            if dxy_cell_sorted[0]:
                # Single ring starting at 0
                rings = np.array([0])
            else:
                # Single co-located pixel, return all visible
                return np.ones(self.Z.shape, dtype=bool)
        rings = np.append(rings, len(ix))
        # Compute elevation ratio
        first_ring = ix[rings[0]:rings[1]]
        is_zero = np.where(dxy[first_ring] == 0)[0]
        dxy[first_ring[is_zero]] = np.nan
        elevation = dz / dxy
        # Compute max number of points on most distant ring
        N = int(np.ceil(2 * np.pi * dxy_cell_sorted[-1]))
        # Initialize result raster
        vis = np.zeros(self.Z.size, dtype=bool)
        # Loop through rings
        period = 2 * np.pi
        for k in range(len(rings) - 1):
            rix = ix[rings[k]:rings[k + 1]]
            rheading = heading[rix]
            relev = elevation[rix]
            # Test visibility
            if k > 0:
                # Interpolate max_elevations to current headings
                max_elevations = np.interp(rheading, previous_headings, max_elevations, period=period)
                # NOTE: Throws warning if np.nan in relev
                is_visible = relev > max_elevations
                if max_elevations_has_nan:
                    is_nan_max_elevation = np.isnan(max_elevations)
                    new_visible = is_nan_max_elevation & ~np.isnan(relev)
                    is_visible |= new_visible
                    if np.count_nonzero(is_nan_max_elevation) == np.count_nonzero(new_visible):
                        max_elevations_has_nan = False
                max_elevations[is_visible] = relev[is_visible]
            else:
                # First ring is always visible (if not NaN)
                is_visible = ~np.isnan(relev)
                max_elevations = relev
                max_elevations_has_nan = any(np.isnan(relev))
            vis[rix] = is_visible
            previous_headings = rheading
        return vis.reshape(self.Z.shape)

    def horizon(self, origin, headings=range(360), correction=False):
        """
        Return the horizon from an arbitrary viewing position.

        Missing values (`numpy.nan`) are ignored. A cell which is the last
        non-missing cell along a sighting is not considered part of the horizon.

        Arguments:
            origin (iterable): World coordinates of viewing position (x, y, z)
            headings (iterable):
            correction (dict or bool): Either arguments to `helpers.elevation_corrections()`,
                `True` for default arguments, or `None` or `False` to skip.

        Returns:
            list: List of world coordinate arrays (n, 3) each tracing an unbroken
                segment of the horizon
        """
        n = len(headings)
        if correction is True:
            correction = dict()
        # Compute ray directions (2d)
        headings = np.array(headings, dtype=float)
        thetas = - (headings - 90) * (np.pi / 180)
        directions = np.column_stack((np.cos(thetas), np.sin(thetas)))
        # Intersect with box (2d)
        box = np.concatenate((self.min[0:2], self.max[0:2]))
        xy_starts, xy_ends = helpers.intersect_rays_box(origin[0:2], directions, box)
        # Convert spatial coordinates (x, y) to grid indices (xi, yi)
        inside = self.inbounds(np.atleast_2d(origin[0:2]))[0]
        if inside:
            # If inside, start at origin
            rowcol = self.xy_to_rowcol(np.atleast_2d(origin[0:2]), snap=True)
            starts = np.repeat(rowcol[:, ::-1], n, axis=0)
        else:
            rowcol = self.xy_to_rowcol(xy_starts)
            starts = rowcol[:, ::-1]
        rowcol = self.xy_to_rowcol(xy_ends, snap=True)
        ends = rowcol[:, ::-1]
        # Iterate over line in each direction
        hxyz = np.full((n, 3), np.nan)
        for i in range(n):
            rowcol = helpers.bresenham_line(starts[i, :], ends[i, :])[:, ::-1]
            if inside:
                # Skip start cell
                rowcol = rowcol[1:]
            idx = self.rowcol_to_idx(rowcol)
            # TODO: Precompute Z.flatten()?
            dz = self.Z.flat[idx] - origin[2]
            xy = self.rowcol_to_xy(rowcol)
            dxy = np.sum((xy - origin[0:2])**2, axis=1) # wait to take square root
            if isinstance(correction, dict):
                delta = helpers.elevation_corrections(
                    squared_distances=dxy, **correction)
                maxi = np.nanargmax((dz + delta) / np.sqrt(dxy))
            else:
                maxi = np.nanargmax(dz / np.sqrt(dxy))
            # Save point if not last non-nan value
            if maxi < (len(dz) - 1) and np.any(~np.isnan(dz[maxi + 1:])):
                hxyz[i, 0:2] = xy[maxi, :]
                hxyz[i, 2] = dz[maxi]
        hxyz[:, 2] += origin[2]
        # Split at NaN
        mask = np.isnan(hxyz[:, 0])
        splits = helpers.boolean_split(hxyz, mask, axis=0, circular=True)
        if mask[0]:
            # Starts with isnan group
            return splits[1::2]
        else:
            # Starts with not-isnan group
            return splits[0::2]

    def fill_circle(self, center, radius, value=np.nan):
        # Circle indices
        rowcol = self.xy_to_rowcol(np.atleast_2d(center[0:2]), snap=True)
        r = np.round(radius / self.d[0])
        xyi = helpers.bresenham_circle(rowcol[0, ::-1], r).astype(int)
        # Filled circle indices
        ind = []
        y = np.unique(xyi[:, 1])
        yin = (y > -1) & (y < self.n[1])
        for yi in y[yin]:
            xb = xyi[xyi[:, 1] == yi, 0]
            xi = range(max(xb.min(), 0), min(xb.max(), self.n[0] - 1) + 1)
            if xi:
                rowcols = np.column_stack((np.repeat(yi, len(xi)), xi))
                ind.extend(self.rowcol_to_idx(rowcols))
        # Apply
        self.Z.flat[ind] = value

class DEMInterpolant(object):
    """
    A `DEMInterpolant` predicts DEMs for arbitrary times by interpolating between observed DEMs.

    Attributes:
        paths (iterable): Paths to DEM files
        datetimes (iterable): Capture datetimes
    """

    def __init__(self, paths, datetimes):
        assert len(paths) == len(datetimes)
        assert len(paths) > 1
        assert len(paths) == len(set(paths))
        assert len(datetimes) == len(set(datetimes))
        self.paths = paths
        self.datetimes = datetimes

    def __call__(self, t, d=None, xlim=None, ylim=None, zlim=None, extrapolate=False, fun=None, **kwargs):
        """
        Return a DEM time-interpolated from two nearby DEM.

        Arguments:
            t (datetime.datetime): Datetime of interpolated DEM
            d (float): Target grid cell size.
                If `None`, the largest DEM cell size is used.
            xlim (iterable): Crop bounds in x.
                If `None`, the intersection of the DEMs is used.
            ylim (iterable): Crop bounds in y.
                If `None`, the intersection of the DEMs is used.
            extrapolate (bool): Whether to interpolate from the two nearest DEMs (True)
                or only from DEMs on either side of `t`
            zlim (iterable): Crop bounds in z.
                Values outside range are set to `np.nan`.
            fun (callable): Function to apply to each DEM before interpolation,
                with signature `fun(dem, **kwargs)`. Must modify DEM in place.
            **kwargs (dict): Additional arguments passed to `fun`

        Returns:
            DEM: An interpolated DEM for time `t`
        """
        dt = np.atleast_1d(self.datetimes) - t
        if extrapolate:
            # Get two nearest DEMs
            i, j = abs(dt).argsort()[:2]
        else:
            # Get nearest DEM on either side of t
            i = np.argmin(abs(dt))
            i_sign = np.sign(dt[i].total_seconds())
            if i_sign == 0:
                j = i
            else:
                on_j_side = int(i_sign) * dt < datetime.timedelta()
                if np.count_nonzero(on_j_side):
                    j_side_nearest = np.argmin(abs(dt[on_j_side]))
                    j = np.arange(len(dt))[on_j_side][j_side_nearest]
                else:
                    raise ValueError('Sample time not bounded on both sides by a DEM')
        if self.datetimes[i] < self.datetimes[j]:
            ij = i, j
        elif self.datetimes[i] > self.datetimes[j]:
            ij = j, i
        else:
            ij = i,
        if d is None or xlim is None or ylim is None:
            grids = [Grid.read(self.paths[k]) for k in ij]
        if d is None:
            d = np.max(np.abs(np.stack([grid.d for grid in grids])))
        if xlim is None:
            xlim = helpers.intersect_ranges([grid.xlim for grid in grids])
        if ylim is None:
            ylim = helpers.intersect_ranges([grid.ylim for grid in grids])
        dems = [DEM.read(self.paths[k], d=d, xlim=xlim, ylim=ylim) for k in ij]
        if zlim is not None:
            for dem in dems:
                dem.crop(zlim=zlim)
        if fun:
            for dem in dems:
                fun(dem, **kwargs)
        if len(dems) > 1:
            different_grids = (any(dems[0].d != dems[1].d) or any(dems[0].n != dems[1].n)
                or any(dems[0].xlim != dems[1].xlim) or any(dems[0].ylim != dems[1].ylim))
            if different_grids:
                dems[1].resample(dems[0], method='linear')
            total_seconds = np.sum(abs(dt[list(ij)])).total_seconds()
            i_ratio = abs(dt[i]).total_seconds() / total_seconds
            j_ratio = abs(dt[j]).total_seconds() / total_seconds
            dems[0].Z = i_ratio * dems[0].Z + j_ratio * dems[1].Z
        dems[0].datetime = t
        return dems[0]
