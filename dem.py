import numpy as np
import scipy.interpolate
import scipy.ndimage
import gdal
import matplotlib
import helper

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
            self._X = np.tile(self.x, [self.n[1], 1])
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
            self._Y = np.tile(self.y, [self.n[0], 1]).transpose()
        return self._Y

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

    def xy_to_rowcol(self, xy, snap=False):
        """
        Return row,col indices of x,y coordinates.

        Arguments:
            xy (array): Spatial coordinates (Nx2)
            snap (bool): Whether to snap indices to nearest integers (cell centers).
                Points on cell edges snap to higher indices, except
                points on right or bottom edges snap down to stay in bounds.
        """
        origin = np.append(self.xlim[0], self.ylim[0])
        colrow = ((xy - origin) / self.d - 0.5)
        if snap:
            temp = np.floor(colrow + 0.5).astype(int)
            # Points on right or bottom edges snap down to stay in bounds
            is_outer_edge = xy == np.append(self.xlim[1], self.ylim[1])
            temp[is_outer_edge] -= 1
            return temp[:, ::-1]
        else:
            return colrow[:, ::-1]

    def rowcol_to_idx(self, rowcol):
        return np.ravel_multi_index((rowcol[:, 0], rowcol[:, 1]), self.n[::-1])

    def crop_extent(self, xlim=None, ylim=None):
        # Calculate x,y limits
        if xlim is None:
            xlim = self.xlim
        if ylim is None:
            ylim = self.ylim
        box = helper.intersect_boxes(np.vstack((
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
        """
        Create a `DEM`.

        Arguments:
            Z (array): Grid of values on a regular xy grid
            x (object): Either `xlim`, `x`, or `X`
            y (object): Either `ylim`, `y`, or `Y`
            datetime (datetime): Capture date and time
        """
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
        no-data values are replaced with NaN.
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
            buf_xsize = int(np.ceil(abs(win_xsize * grid.d[0] / d)))
            buf_ysize = int(np.ceil(abs(win_ysize * grid.d[1] / d)))
        else:
            buf_xsize = win_xsize
            buf_ysize = win_ysize
        band = raster.GetRasterBand(band)
        Z = band.ReadAsArray(
            xoff=cols[0], yoff=rows[0],
            win_xsize=win_xsize, win_ysize=win_ysize,
            buf_xsize=buf_xsize, buf_ysize=buf_ysize
        )
        # FIXME: band.GetNoDataValue() not equal to read values due to rounding
        # HACK: Use < -9998 since most common are -9999 and -3.4e38
        nan_value = band.GetNoDataValue()
        if np.issubdtype(Z.dtype, float) and nan_value:
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
                method="linear", bounds_error=False)
        return self._Zf

    # ---- Methods (public) ----

    def copy(self):
        return DEM(self.Z, x=self.xlim, y=self.ylim, datetime=self.datetime)

    def sample(self, xy, indices=False, method="linear"):
        """
        Sample `Z` at points.

        Arguments:
            xy (array): Point coordinates (Nx2)
            method (str): Interpolation method,
                either "linear" (default) or "nearest"
        """
        if indices:
            return self.Z.flat[self.rowcol_to_idx(xy)]
        else:
            return self.Zf(xy, method=method)

    def resample(self, dem, method="linear"):
        xy = np.column_stack((dem.X.flatten(), dem.Y.flatten()))
        Z = self.sample(xy, method=method)
        self.Z = Z.reshape(dem.Z.shape)
        self.xlim, self._x, self._X = self._parse_x(dem.X)
        self.ylim, self._y, self._Y = self._parse_y(dem.Y)

    def plot(self, array=None, cmap="gray", **kwargs):
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
        return helper.rasterize_points(rowcol[:, 0], rowcol[:, 1],
            values[is_in], self.Z.shape, fun=fun)

    def crop(self, xlim=None, ylim=None, zlim=None):
        """
        Crop a `DEM`.

        Arguments:
            xlim (array_like): Cropping bounds in x
            ylim (array_like): Cropping bounds in y
            zlim (array_like): Cropping bounds in z.
                Values outside range are set to `nan`.
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

    def fill_crevasses_simple(self, maximum_filter_size=5, gaussian_filter_sigma=5):
        """
        Apply a maximum filter to `Z`, then perform Gaussian smoothing (fast).

        Arguments:
            maximum_filter_size (int): Kernel size of maximum filter in pixels
            gaussian_filter_sigma (float): Standard deviation of Gaussian filter
        """
        self.Z = scipy.ndimage.filters.gaussian_filter(
            scipy.ndimage.filters.maximum_filter(self.Z, size=maximum_filter_size),
            sigma=gaussian_filter_sigma)

    def fill_crevasses_complex(self, maximum_filter_size=5, gaussian_filter_sigma=5):
        """
        Find the local maxima of `Z`, fit a surface through them, then perform Gaussian smoothing (slow).

        Arguments:
            maximum_filter_size (int): Kernel size of maximum filter in pixels
            gaussian_filter_sigma (float): Standard deviation of Gaussian filter
        """
        Z_maxima = scipy.ndimage.filters.maximum_filter(self.Z, size=maximum_filter_size)
        is_max = (Z_maxima == self.Z).ravel()
        Xmax = self.X.ravel()[is_max]
        Ymax = self.Y.ravel()[is_max]
        Zmax = self.Z.ravel()[is_max]
        max_interpolant = scipy.interpolate.LinearNDInterpolator(np.vstack((Xmax, Ymax)).T, Zmax)
        Z_fmax = max_interpolant(self.X.ravel(), self.Y.ravel()).reshape(self.Z.shape)
        self.Z = scipy.ndimage.filters.gaussian_filter(Z_fmax, sigma=gaussian_filter_sigma)

    def visible(self, xyz):
        X = self.X.flatten() - xyz[0]
        Y = self.Y.flatten() - xyz[1]
        Z = self.Z.flatten() - xyz[2]
        # NOTE: Compute dx, dz, then elevation angle instead?
        d = np.sqrt(X**2 + Y**2 + Z**2)
        x = (np.arctan2(Y, X) + np.pi) / (np.pi * 2) # ???
        y = Z / d
        # Slow:
        ix = np.lexsort((x,
            np.round(((X / abs(self.d[0])) ** 2 + (Y / abs(self.d[1])) ** 2) ** 0.5)))
        loopix = np.argwhere(np.diff(x[ix]) < 0).flatten()
        vis = np.ones(len(x), dtype=bool)
        maxd = np.nanmax(d)
        # Number of points in voxel horizon:
        N = np.ceil(2 * np.pi / (abs(self.d[0]) / maxd))
        voxx = np.linspace(0, 1, int(N) + 1)
        voxy = np.zeros(len(voxx)) - np.inf
        for k in range(len(loopix) - 1):
            lp = ix[(loopix[k] + 1):(loopix[k + 1] + 1)]
            lp = np.hstack(([lp[-1]], lp, [lp[0]]))
            yy = y[lp]
            xx = x[lp]
            # Why?:
            xx[0] -= 1
            xx[-1] += 1
            # ---
            end = len(lp) - 1
            if k: # Skip on first iteration (all visible)
                vis[lp[1:end]] = np.interp(xx[1:end], voxx, voxy) < yy[1:end]
            voxy = np.maximum(voxy, np.interp(voxx, xx, yy))
        return vis.reshape(self.Z.shape)

    def visible2(self, xyz):
        # NOTE: Assumes square grid
        # Compute distance to all cell centers
        dx = self.X.flatten() - xyz[0]
        dy = self.Y.flatten() - xyz[1]
        dz = self.Z.flatten() - xyz[2]
        dxy2 = (dx**2 + dy**2)
        dxy =  np.sqrt(dxy2)
        # dxyz = np.sqrt((dxy2 + dz**2))
        dxy_cell = (dxy / abs(self.d[0]) + 0.5).astype(int)
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
        rings = np.hstack((rings, len(ix)))
        # Compute elevation
        dxy[dxy == 0] = np.nan
        # NOTE: Arctan necessary?
        elevation = np.arctan(dz / dxy)
        # Compute max number of points on most distant ring
        # N = np.ceil(2 * np.pi * (dxy[ix[-1]] / abs(self.d[0]))).astype(int)
        N = int(np.ceil(2 * np.pi * dxy_cell_sorted[-1]))
        # Initialize loop
        # NOTE: N or N + 1 ?
        vis = np.zeros(self.Z.size, dtype=bool)
        headings = np.linspace(-np.pi, np.pi, N)
        headings_f = scipy.interpolate.interp1d(headings, np.arange(N), kind='nearest', assume_sorted=True)
        max_elevations = np.full(N, -np.inf)
        max_elevation_ix = np.full(N, np.nan)
        # Loop through rings
        # NOTE: Why len(rings) - 1 ?
        period = 2 * np.pi
        for k in range(len(rings) - 1):
            rix = ix[rings[k]:rings[k + 1]]
            rheading = heading[rix]
            relev = elevation[rix]
            # Visibility
            if k > 0:
                # NOTE: Throws warning if np.nan in relev
                is_visible = relev >= np.interp(rheading, headings, max_elevations, period=period)
            else:
                # First ring is always visible
                is_visible = np.ones(relev.shape, dtype=bool)
            vis[rix] = is_visible
            max_elevations = np.fmax(max_elevations, np.interp(headings, rheading, relev, period=period)) # ignores nan
        return vis.reshape(self.Z.shape)

    def horizon(self, origin, headings=np.arange(360), corrected=True):
        # TODO: Radius min, max (by slicing bresenham line)
        n = len(headings)
        # Compute ray directions (2d)
        thetas = - (headings - 90) * np.pi / 180
        directions = np.column_stack((np.cos(thetas), np.sin(thetas)))
        # Intersect with box (2d)
        box = np.concatenate((self.min[0:2], self.max[0:2]))
        xy_starts, xy_ends = helper.intersect_rays_box(origin[0:2], directions, box)
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
            rowcol = helper.bresenham_line(starts[i, :], ends[i, :])[:, ::-1]
            if inside:
                # Skip start cell
                rowcol = rowcol[1:]
            idx = self.rowcol_to_idx(rowcol)
            # TODO: Precompute Z.flatten()?
            dz = self.Z.flat[idx] - origin[2]
            xy = self.rowcol_to_xy(rowcol)
            dxy = np.sqrt(np.sum((xy - origin[0:2])**2, axis=1))
            maxi = np.nanargmax(dz / dxy)
            # Save point it not last non-nan value
            if maxi < (len(dz) - 1) and np.any(~np.isnan(dz[maxi + 1:])):
                hxyz[i, 0:2] = xy[maxi, :]
                hxyz[i, 2] = dz[maxi]
        hxyz[:, 2] += origin[2]
        if corrected:
            # Correct for earth curvature and refraction
            # e.g. http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?topicname=how_viewshed_works
            # z_actual = z_surface - 0.87 * distance^2 / diameter_earth
            hxyz[:, 2] -= 0.87 * np.sum((origin[0:2] - hxyz[:, 0:2])**2, axis=1) / 12.74e6
        # Split at NaN
        mask = np.isnan(hxyz[:, 0])
        splits = helper.boolean_split(hxyz, mask, axis=0, circular=True)
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
        xyi = helper.bresenham_circle(rowcol[0, ::-1], r).astype(int)
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
