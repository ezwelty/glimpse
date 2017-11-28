import numpy as np
import scipy.interpolate
import scipy.ndimage.filters
import scipy.misc
import gdal

class DEM(object):
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
    def read(cls, path, x=None, y=None, datetime=None):
        """
        Read DEM from raster file.

        See `gdal.Open()` for details.

        Arguments:
            path (str): Path to file
            x (object): Either `xlim`, `x`, or `X`.
                If `None` (default), read from file.
            y (object): Either `ylim`, `y`, or `Y`.
                If `None` (default), read from file.
            datetime (datetime): Capture date and time
        """
        path = "cg/arcticdem_10m.tif"
        raster = gdal.Open(path, gdal.GA_ReadOnly)
        band = raster.GetRasterBand(1)
        Z = band.ReadAsArray()
        # FIXME: band.GetNoDataValue() not equal to read values due to rounding
        # HACK: Use < -9998 since most common are -9999 and -3.4e38
        Z[Z < -9998] = np.nan
        transform = raster.GetGeoTransform()
        if x is None:
            x = transform[0] + transform[1] * np.array([0, raster.RasterXSize])
        if y is None:
            y = transform[3] + transform[5] * np.array([0, raster.RasterYSize])
        return cls(Z, x=x, y=y, datetime=datetime)

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
    def min(self):
        value = [min(self.xlim), min(self.ylim), min(self.zlim)]
        return np.array(value)

    @property
    def max(self):
        value = [max(self.xlim), max(self.ylim), max(self.zlim)]
        return np.array(value)

    @property
    def n(self):
        value = [self.Z.shape[1], self.Z.shape[0]]
        return np.array(value, dtype=int)

    @property
    def d(self):
        return np.append(np.diff(self.xlim), np.diff(self.ylim)) / self.n

    # ---- Properties (cached) ----

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

    @property
    def Zf(self):
        if self._Zf is None:
            sign = np.sign(self.d).astype(int)
            self._Zf = scipy.interpolate.RectBivariateSpline(
                self.x[::sign[0]], self.y[::sign[1]], np.nan_to_num(self.Z[::sign[1], ::sign[0]]).T, kx=3, ky=3, s=0)
        return self._Zf

    # ---- Methods (public) ----

    def inbounds(self, xy):
        """
        Test whether points are within (or on) the xy bounds of the `DEM`.

        Arguments:
            xy (array): Point coordinates (Nx2)

        Returns:
            array: Whether each point is inbounds (Nx1)
        """
        return np.logical_and(xy >= self.min[0:2], xy <= self.max[0:2]).all(axis = 1)

    def xy_to_rowcol(self, xy, snap=False):
        """
        Return row,col indices of x,y coordinates.

        See `.rowcol_to_xy()` for the inverse.

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

    def rowcol_to_xy(self, rowcol):
        """
        Return x,y coordinates of row,col indices.

        Places integer indices at the centers of each cell.
        Therefore, the upper left corner is [-0.5, -0.5]
        and the center of that cell is [0, 0].

        See `.xy_to_rowcol()` for the inverse.

        Arguments:
            rowcol (array): Array indices (Nx2)
        """
        origin = np.append(self.xlim[0], self.ylim[0])
        return (rowcol + 0.5)[:, ::-1] * self.d + origin

    def sample(self, xy):
        """
        Sample `Z` at points.

        Interpolation is performed using a 3rd order spline.
        Missing values are currently replaced with zeros.

        Arguments:
            xy (array): Point coordinates (Nx2)

        Returns:
            array: Value of `Z` interpolated at each point (Nx1).
        """
        return self.Zf(xy[:, 0], xy[:, 1], grid=False)

    def sample_grid(self, x, y):
        """
        Sample `Z` at points.

        Interpolation is performed using a 3rd order spline.
        Missing values are currently replaced with zeros.

        Arguments:
            x, y (array_like): Coordinates specifying points on a grid.
                Arrays must be sorted in increasing order.

        Returns:
            array: Value of `Z` interpolated at each point, with x as columns and y as rows.
        """
        return self.Zf(x, y, grid=True).transpose()

    def crop(self, xlim=None, ylim=None, copy=True):
        """
        Crop a `DEM`.

        Arguments:
            xlim (array_like): Cropping bounds in x
            ylim (array_like): Cropping bounds in y
            copy (bool): Whether to return result as new `DEM`

        Returns:
            DEM: New object (if `copy=True`)
        """
        # Intersect xlim
        if xlim is None:
            xlim = self.xlim
        else:
            xlim = np.asarray([
                max(min(self.xlim), min(xlim)),
                min(max(self.xlim), max(xlim))
                ], float)
            dx = np.diff(xlim)
            if dx <= 0:
                raise ValueError("Crop bounds (xlim) do not intersect DEM.")
            if np.diff(self.xlim) < 0:
                xlim = xlim[::-1]
        # Intersect ylim
        if ylim is None:
            ylim = self.ylim
        else:
            ylim = np.asarray([
                max(min(self.ylim), min(ylim)),
                min(max(self.ylim), max(ylim))
                ], float)
            dy = np.diff(ylim)
            if dy <= 0:
                raise ValueError("Crop bounds (ylim) do not intersect DEM.")
            if np.diff(self.ylim) < 0:
                ylim = ylim[::-1]
        # Test for equality
        if all(xlim == self.xlim) and all(ylim == self.ylim):
            if copy:
                return DEM(self.Z, self.xlim, self.ylim)
            else:
                return None
        # Convert xy limits to grid indices
        xy = np.column_stack((xlim, ylim))
        rc = self.xy_to_rowcol(xy, snap=True)
        # Snap down bottom-right, non-outer edges
        # see .xy_to_rowcol()
        bottom_right = np.append(self.xlim[1], self.ylim[1])
        is_edge = (bottom_right - xy[1, :]) % self.d == 0
        is_outer_edge = xy[1, :] == bottom_right
        snap_down = (is_edge & ~is_outer_edge)
        rc[1, snap_down[::-1]] -= 1
        # Crop DEM
        Z = self.Z[rc[0, 0]:rc[1, 0] + 1, rc[0, 1]:rc[1, 1] + 1]
        new_xy = self.rowcol_to_xy(rc)
        new_xlim = new_xy[:, 0] + np.array([-0.5, 0.5]) * self.d[0]
        new_ylim = new_xy[:, 1] + np.array([-0.5, 0.5]) * self.d[0]
        if copy:
            return DEM(Z, new_xlim, new_ylim)
        else:
            self.Z = Z
            self.xlim = new_xlim
            self.ylim = new_ylim

    def resize(self, scale, copy=True):
        """
        Resize a `DEM`.

        Arguments:
            scale (float): Fraction of current size
            copy (bool): Whether to return result as new `DEM`

        Returns:
            DEM: New object (if `copy=True`)
        """
        Z = scipy.misc.imresize(self.Z, size=float(scale), interp='bilinear')
        if copy:
            return DEM(Z, self.xlim, self.ylim)
        else:
            self.Z = Z

    def fill_crevasses_simple(self, maximum_filter_size=5, gaussian_filter_sigma=5, copy=True):
        """
        Apply a maximum filter to `Z`, then perform Gaussian smoothing (fast).

        Arguments:
            maximum_filter_size (int): Kernel size of maximum filter in pixels
            gaussian_filter_sigma (float): Standard deviation of Gaussian filter
            copy (bool): Whether to return result as new `DEM`

        Returns:
            DEM: New object (if `copy=True`)
        """
        Z = scipy.ndimage.filters.gaussian_filter(
            scipy.ndimage.filters.maximum_filter(self.Z, size=maximum_filter_size),
            sigma=gaussian_filter_sigma
            )
        if copy:
            xs = [self._X, self._x, self.xlim]
            x = next(item for item in xs if item is not None)
            ys = [self._Y, self._y, self.ylim]
            y = next(item for item in ys if item is not None)
            return DEM(Z, x, y)
        else:
            self.Z = Z

    def fill_crevasses_complex(self, maximum_filter_size=5, gaussian_filter_sigma=5, copy=True):
        """
        Find the local maxima of `Z`, fit a surface through them, then perform Gaussian smoothing (slow).

        Arguments:
            maximum_filter_size (int): Kernel size of maximum filter in pixels
            gaussian_filter_sigma (float): Standard deviation of Gaussian filter
            copy (bool): Whether to return result as new `DEM`

        Returns:
            DEM: New object (if `copy=True`)
        """
        Z_maxima = scipy.ndimage.filters.maximum_filter(self.Z, size=maximum_filter_size)
        is_max = (Z_maxima == self.Z).ravel()
        Xmax = self.X.ravel()[is_max]
        Ymax = self.Y.ravel()[is_max]
        Zmax = self.Z.ravel()[is_max]
        max_interpolant = scipy.interpolate.LinearNDInterpolator(np.vstack((Xmax, Ymax)).T, Zmax)
        Z_fmax = max_interpolant(self.X.ravel(), self.Y.ravel()).reshape(self.Z.shape)
        Z = scipy.ndimage.filters.gaussian_filter(Z_fmax, sigma=gaussian_filter_sigma)
        if copy:
            xs = [self._X, self._x, self.xlim]
            x = next(item for item in xs if item is not None)
            ys = [self._Y, self._y, self.ylim]
            y = next(item for item in ys if item is not None)
            return DEM(Z, x, y)
        else:
            self.Z = Z

    def visible(self, xyz):
        X = self.X.flatten() - xyz[0]
        Y = self.Y.flatten() - xyz[1]
        Z = self.Z.flatten() - xyz[2]
        # NOTE: Compute dx, dz, then elevation angle instead?
        d = (X ** 2 + Y ** 2 + Z ** 2) ** 0.5
        x = (np.arctan2(Y, X) + np.pi) / (np.pi * 2) # ???
        y = Z / d
        # Slow:
        ix = np.lexsort((
                x,
                np.round(((X / abs(self.d[0])) ** 2 + (Y / abs(self.d[1])) ** 2) ** 0.5),
            ))
        loopix = np.argwhere(np.diff(x[ix]) < 0).flatten()
        vis = np.ones(len(x), dtype=bool)
        maxd = d.max()
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

    # ---- Methods (private) ----

    def _clear_cache(self, attributes=['x', 'X', 'y', 'Y', 'Zf']):
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
            obj = [self.n[1], 0]
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
