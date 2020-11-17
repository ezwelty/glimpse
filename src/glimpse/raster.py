"""Read, write, and manipulate orthorectified images."""
import copy
import datetime
import numbers
import warnings
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import matplotlib.colors
import matplotlib.pyplot
import numpy as np
import osgeo.gdal
import scipy.interpolate
import scipy.ndimage
from typing_extensions import Literal

from . import helpers

Number = Union[int, float]


class Grid:
    """
    Regular rectangular 2-dimensional grid.

    Arguments:
        n: Grid dimensions (nx, ny).
        x: X coordinates as either :attr:`xlim`, :attr:`x`, or :attr:`X`.
        y: Y coordinates as either :attr:`ylim`, :attr:`y`, or :attr:`Y`.
        crs: Coordinate reference system as int (EPSG) or str (Proj4 or WKT).

    Attributes:
        xlim (numpy.ndarray): Outer x limits of the grid (left, right).
        ylim (numpy.ndarray): Outer y limits of the grid (top, bottom).
        size (numpy.ndarray): Grid dimensions (nx, ny).
        d (numpy.ndarray): Grid cell size (dx, dy).
        x (numpy.ndarray): Cell center x coordinates from left to right (nx,).
        y (numpy.ndarray): Cell center y coordinates from top to bottom (ny,).
        X (numpy.ndarray): Cell center x coordinates for each cell (ny, nx).
        Y (numpy.ndarray): Cell center y coordinates for each cell (ny, nx).
        min (numpy.ndarray): Minimum bounding box coordinates (xmin, ymin).
        max (numpy.ndarray): Maximum bounding box coordinates (xmax, ymax).
        box2d (numpy.ndarray): Bounding box (xmin, ymin, xmax, ymax).
        crs: Coordinate reference system as int (EPSG) or str (Proj4 or WKT).
    """

    def __init__(
        self,
        size: Tuple[int, int],
        x: Iterable[Union[Number, Iterable[Number]]] = None,
        y: Iterable[Union[Number, Iterable[Number]]] = None,
        crs: Union[int, str] = None,
    ) -> None:
        self._size = size
        self.xlim, self._x, self._X = self._parse_xy(x, dim=0)
        self.ylim, self._y, self._Y = self._parse_xy(y, dim=1)
        self.crs = crs

    def __eq__(self, other: "Grid") -> bool:
        """Consider equal if coordinate system is equal."""
        return (
            (self.shape == other.shape)
            and (self.xlim == other.xlim).all()
            and (self.ylim == other.ylim).all()
        )

    # ---- Properties ---- #

    @property
    def size(self) -> np.ndarray:
        """Grid dimensions (nx, ny)."""
        return self._size

    @size.setter
    def size(self, value: Iterable[int]) -> None:
        value = np.atleast_1d(value)
        if value.shape == (1,):
            value = np.concatenate((value, value))
        if value.shape != (2,):
            raise ValueError("Grid dimensions must be scalar or (2,)")
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError("Grid dimensions must be integer")
        if (value <= 0).any():
            raise ValueError("Grid dimensions must be positive")
        self._size = value

    @property
    def xlim(self) -> np.ndarray:
        """Outer x limits of the grid (left, right)."""
        return self._xlim

    @xlim.setter
    def xlim(self, value: Iterable[Number]) -> None:
        value = self._parse_limits(value)
        if not hasattr(self, "xlim") or not np.array_equal(self.xlim, value):
            self._xlim = value
            self._clear_cache(["x", "X"])

    @property
    def ylim(self) -> np.ndarray:
        """Outer y limits of the grid (top, bottom)."""
        return self._ylim

    @ylim.setter
    def ylim(self, value: Iterable[Number]) -> None:
        value = self._parse_limits(value)
        if not hasattr(self, "ylim") or not np.array_equal(self.ylim, value):
            self._ylim = value
            self._clear_cache(["y", "Y"])

    # ---- Properties (dependent) ---- #

    @property
    def shape(self) -> Tuple[int, int]:
        """Array shape (ny, nx)."""  # noqa: D402
        return self.size[1], self.size[0]

    @property
    def d(self) -> np.ndarray:
        """Grid cell size (dx, dy)."""
        return np.hstack((np.diff(self.xlim), np.diff(self.ylim))) / self.size

    @property
    def min(self) -> np.ndarray:
        """Minimum bounding box coordinates (xmin, ymin)."""
        return np.array((min(self.xlim), min(self.ylim)))

    @property
    def max(self) -> np.ndarray:
        """Maximum bounding box coordinates (xmax, ymax)."""
        return np.array((max(self.xlim), max(self.ylim)))

    @property
    def box2d(self) -> np.ndarray:
        """Bounding box (xmin, ymin, xmax, ymax)."""
        return np.hstack((self.min, self.max))

    @property
    def x(self) -> np.ndarray:
        """Cell center x coordinates from left to right (nx,)."""
        if self._x is None:
            value = np.linspace(
                start=self.min[0] + abs(self.d[0]) / 2,
                stop=self.max[0] - abs(self.d[0]) / 2,
                num=self.size[0],
            )
            if self.d[0] < 0:
                self._x = value[::-1]
            else:
                self._x = value
        return self._x

    @property
    def X(self) -> np.ndarray:
        """Cell center x coordinates for each cell (ny, nx)."""
        if self._X is None:
            self._X = np.tile(self.x, (self.size[1], 1))
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Cell center y coordinates from top to bottom (ny,)."""
        if self._y is None:
            value = np.linspace(
                start=self.min[1] + abs(self.d[1]) / 2,
                stop=self.max[1] - abs(self.d[1]) / 2,
                num=self.size[1],
            )
            if self.d[1] < 0:
                self._y = value[::-1]
            else:
                self._y = value
        return self._y

    @property
    def Y(self) -> np.ndarray:
        """Cell center y coordinates for each cell (ny, nx)."""
        if self._Y is None:
            self._Y = np.tile(self.y, (self.size[0], 1)).T
        return self._Y

    @classmethod
    def read(
        cls,
        path: str,
        d: Number = None,
        xlim: Iterable[Number] = None,
        ylim: Iterable[Number] = None,
    ) -> "Grid":
        """
        Read Grid from raster file.

        Arguments:
            path: Path to file.
            d: Target grid cell size.
            xlim: Target outer bounds of crop in x.
            ylim: Target outer bounds of crop in y.
        """
        raster = osgeo.gdal.Open(path, osgeo.gdal.GA_ReadOnly)
        transform = raster.GetGeoTransform()
        size = (raster.RasterXSize, raster.RasterYSize)
        crs = raster.GetProjection()
        grid = cls(
            size,
            x=transform[0] + transform[1] * np.array([0, size[0]]),
            y=transform[3] + transform[5] * np.array([0, size[1]]),
            crs=crs if crs else None,
        )
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

    def _clear_cache(self, attributes: Iterable[str] = ["x", "X", "y", "Y"]) -> None:
        """Clear cached attributes."""
        attributes = tuple(attributes)
        for attr in attributes:
            setattr(self, "_" + attr, None)

    def _parse_limits(self, value: Iterable[Number]) -> np.ndarray:
        """Check and parse limits."""
        value = np.atleast_1d(value)
        if value.shape != (2,):
            raise ValueError("Grid limits must be (2,)")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError("Grid limits must be numeric")
        if value[0] == value[1]:
            raise ValueError("Grid limits cannot be equal")
        return value

    def _parse_xy(
        self, value: Iterable[Union[Number, Iterable[Number]]], dim: Literal[0, 1]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse limits, coordinate vector, and coordinate matrix."""
        if value is None:
            value = (0, self.size[dim])
        if not isinstance(value, np.ndarray):
            value = np.atleast_1d(value)
        is_X = value.shape[0:2] == self.shape[0:2]
        if is_X:
            # TODO: Check if all columns equal
            X = value
            value = value[:, 0] if dim else value[0]
        else:
            X = None
        is_x = any(n > 2 for n in value.shape[0:2])
        if is_x:
            x = value
            # TODO: Check if equally spaced monotonic
            dx = np.diff(value[0:2])
            xlim = np.append(value[0] - dx / 2, value[-1] + dx / 2)
        else:
            x = None
            xlim = value
        if len(xlim) != 2:
            raise ValueError("Could not parse limits from x, y inputs")
        return xlim, x, X

    def _shift_xy(self, dx: Number = None, dy: Number = None) -> None:
        """Shift grid position."""
        if dx is not None:
            self._xlim += dx
            if self._x is not None:
                self._x += dx
            if self._X is not None:
                self._X += dx
        if dy is not None:
            self._ylim += dy
            if self._y is not None:
                self._y += dy
            if self._Y is not None:
                self._Y += dy

    # ---- Methods ---- #

    def copy(self) -> "Grid":
        """Copy grid."""
        return Grid(self.size.copy(), x=self.xlim.copy(), y=self.ylim.copy())

    def resize(self, scale: Number) -> None:
        """
        Resize grid.

        Grid cell aspect ratio may not be preserved due to integer rounding
        of grid dimensions.

        Arguments:
            scale: Fraction of current size.
        """
        self.size = np.floor(self.size * scale + 0.5).astype(int)

    def shift(self, dx: Number = None, dy: Number = None) -> None:
        """
        Shift grid position.

        Arguments:
            dx: Shift in x.
            dy: Shift in y.
        """
        self._shift_xy(dx=dx, dy=dy)

    def inbounds(
        self, xy: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], grid: bool = False
    ) -> np.ndarray:
        """
        Test whether points are in (or on) bounds.

        Arguments:
            xy: Input coordinates x and y.

                - If `grid=False`, as point coordinates (n, 2).
                - If `grid=True`, as coordinate vectors x (n,) and y (m,).

            grid: Whether `xy` defines a grid or invidual points.

        Returns:
            Whether each point is inbounds (n, 1) if `grid` is False,
                or whether each grid column or row is inbounds (n,), (m,)
                if `grid` is True.
        """
        if grid:
            return (
                (xy[0] >= self.min[0]) & (xy[0] <= self.max[0]),
                (xy[1] >= self.min[1]) & (xy[1] <= self.max[1]),
            )
        return np.all((xy >= self.min[0:2]) & (xy <= self.max[0:2]), axis=1)

    def snap_xy(
        self,
        xy: np.ndarray,
        centers: bool = False,
        edges: bool = False,
        inbounds: bool = True,
    ) -> np.ndarray:
        """
        Snap points to nearest grid positions.

        When snapping to cell centers, points on edges snap to higher grid indices.
        If `inbounds=True`, points on the right and bottom outer edges snap to interior
        cell centers to stay in bounds.

        Arguments:
            xy: Point coordinates (n, [x, y]).
            centers: Whether to snap points to the nearest cell centers.
            edges: Whether to snap points to nearest cell edges.
            inbounds: Whether to snap points on right and bottom bounds
                to interior cell centers (if `edges=False` and `centers=True`).

        Returns:
            Snapped point coordinates (n, [x, y]).

        Raises:
            ValueError: Arguments centers and edges cannot both be False.
        """
        # TODO: Faster version for image grid
        if not centers and not edges:
            raise ValueError("Arguments centers and edges cannot both be False")
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

    def snap_box(
        self,
        xy: Iterable[Number],
        size: Iterable[Number],
        centers: bool = False,
        edges: bool = True,
        inbounds: bool = True,
    ) -> np.ndarray:
        """
        Snap box to nearest grid positions.

        Arguments:
            xy: Coordinates of box center (x, y).
            size: Dimensions of box (nx, ny).
            centers: Whether to snap to nearest cell centers.
            edges: Whether to snap to nearest cell edges.
            inbounds: Whether to snap right and bottom bounds to interior cell centers
                (if `edges=False` and `centers=True`).

        Returns:
            Snapped box boundaries (xmin, ymin, xmax, ymax).

        Raises:
            IndexError: Box extends beyond grid bounds.
        """
        halfsize = np.multiply(size, 0.5)
        xy_box = np.vstack((xy - halfsize, xy + halfsize))
        if any(~self.inbounds(xy_box)):
            raise IndexError("Box extends beyond grid bounds")
        return self.snap_xy(
            xy_box, centers=centers, edges=edges, inbounds=inbounds
        ).flatten()

    def xyz_to_uv(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to image coordinates.

        Arguments:
            xyz: World coordinates (n, [x, y, z]). Dimension z is optional and unused.

        Returns:
            Image coordinates (n, [u, v]).

        Examples:
            >>> grid = Grid((3, 2), x=(0, 30), y=(4, 0))
            >>> xyz = [(5, 3), (15, 2), (30, 0)]
            >>> uv = grid.xyz_to_uv(xyz)
            >>> uv
            array([[0.5, 0.5],
                   [1.5, 1. ],
                   [3. , 2. ]])
            >>> (grid.uv_to_xyz(uv)[:, 0:2] == xyz).all()
            True
        """
        xyz = np.asarray(xyz)
        return (xyz[:, 0:2] - (self.xlim[0], self.ylim[0])) / self.d

    def uv_to_xyz(self, uv: np.ndarray) -> np.ndarray:
        """
        Convert image coordinates to world coordinates.

        Arguments:
            uv: Image coordinates (n, [u, v]).

        Returns:
            World coordinates (n, [x, y, z]). Dimension z is NaN.
        """
        uv = np.asarray(uv)
        xy = uv * self.d + (self.xlim[0], self.ylim[0])
        return np.column_stack((xy, np.full((xy.shape[0], 1), np.nan)))

    def rowcol_to_xy(self, rowcol: np.ndarray) -> np.ndarray:
        """
        Convert array indices to map coordinates.

        Places integer array indices at the centers of each cell.
        Therefore, the upper left corner is [-0.5, -0.5]
        and the center of that cell is [0, 0].

        Arguments:
            Array indices (n, [row, col]).

        Returns:
            Map coordinates (n, [x, y]).
        """
        xy_origin = np.array((self.xlim[0], self.ylim[0]))
        return (rowcol + 0.5)[:, ::-1] * self.d + xy_origin

    def xy_to_rowcol(
        self, xy: np.ndarray, snap: bool = False, inbounds: bool = True
    ) -> np.ndarray:
        """
        Convert map coordinates to array indices.

        Arguments:
            xy: Map coordinates (n, [x, y]).
            snap: Whether to snap to nearest cell centers.
            inbounds: Whether to snap right and bottom bounds to interior cell centers
                (see :meth:`snap_xy`).

        Returns:
            Array indices as either float (`snap=False`) or int (`snap=True`).
        """
        # TODO: Remove snapping from function (now a seperate operation)
        if snap:
            xy = self.snap_xy(xy, centers=True, edges=False, inbounds=inbounds)
        origin = np.append(self.xlim[0], self.ylim[0])
        colrow = (xy - origin) / self.d - 0.5
        if snap:
            colrow = colrow.astype(int)
        return colrow[:, ::-1]

    def rowcol_to_idx(self, rowcol: np.ndarray) -> np.ndarray:
        """
        Convert 2-dimensional array indices to flat array indices.

        Arguments:
            rowcol: Array indices (n, [row, col]).

        Returns:
            Flat array indices (n, ).
        """
        return np.ravel_multi_index((rowcol[:, 0], rowcol[:, 1]), self.size[::-1])

    def idx_to_rowcol(self, idx: np.ndarray) -> np.ndarray:
        """
        Convert flat array indices to 2-dimensional array indices.

        Arguments:
            idx: Flat array indices (n, ).

        Returns:
            Array indices (n, [row, col]).
        """
        return np.column_stack(np.unravel_index(idx, self.size[::-1]))

    def crop_extent(
        self, xlim: Iterable[Number] = None, ylim: Iterable[Number] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute crop extent.

        Arguments:
            xlim: Target outer bounds of crop in x.
            ylim: Target outer bounds of crop in y.

        Returns:
            Outer bounds of crop in x (left, right).
            Outer bounds of crop in y (top, bottom).
            Outer bounds of crop as row indices (top, bottom).
            Outer bounds of crop as column indices (left, right).
        """
        # Calculate x,y limits
        if xlim is None:
            xlim = self.xlim
        if ylim is None:
            ylim = self.ylim
        box = helpers.intersect_boxes(
            np.vstack(
                (
                    np.hstack((min(xlim), min(ylim), max(xlim), max(ylim))),
                    np.hstack((self.min[0:2], self.max[0:2])),
                )
            )
        )
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
        snap_down = is_edge & ~is_outer_edge
        rowcol[1, snap_down[::-1]] -= 1
        new_xy = self.rowcol_to_xy(rowcol)
        new_xlim = new_xy[:, 0] + np.array([-0.5, 0.5]) * self.d[0]
        new_ylim = new_xy[:, 1] + np.array([-0.5, 0.5]) * self.d[1]
        return new_xlim, new_ylim, rowcol[:, 0], rowcol[:, 1]

    def set_plot_limits(self) -> None:
        """Set limits of current plot axis to grid limits."""
        matplotlib.pyplot.xlim(self.xlim[0], self.xlim[1])
        matplotlib.pyplot.ylim(self.ylim[1], self.ylim[0])

    def tile_indices(
        self, size: Iterable[int], overlap: Iterable[int] = (0, 0)
    ) -> Tuple[slice, slice]:
        """
        Return slice objects that chop the grid into tiles.

        Arguments:
            size: Target tile size (nx, ny).
            overlap: Number of overlapping grid cells between tiles (nx, ny).

        Returns:
            Pairs of slice objects (rows, cols) with which to subset grid.
        """
        n = np.round(self.size / size).astype(int)
        # Ignore divide by zero
        with np.errstate(divide="ignore"):
            xi = np.floor(np.arange(self.size[0]) / np.ceil(self.size[0] / n[0]))
            yi = np.floor(np.arange(self.size[1]) / np.ceil(self.size[1] / n[1]))
        xends = np.insert(np.searchsorted(xi, np.unique(xi), side="right"), 0, 0)
        yends = np.insert(np.searchsorted(yi, np.unique(yi), side="right"), 0, 0)
        # HACK: Achieves overlap by increasing tile size
        xstarts = xends.copy()
        xstarts[1:-1] -= overlap[0]
        ystarts = yends.copy()
        ystarts[1:-1] -= overlap[1]
        return tuple(
            (slice(ystarts[i], yends[i + 1]), slice(xstarts[j], xends[j + 1]))
            for i in range(len(ystarts) - 1)
            for j in range(len(xstarts) - 1)
        )


class Raster(Grid):
    """
    Values on a regular rectangular 2-dimensional grid.

    For rasters with dimension of length 2, `x` (`y`) is assumed to be `xlim` (`ylim`)
    if a vector.
    For rasters with dimensions of length 1, `x` (`y`) must be `xlim` (`ylim`)
    since cell size cannot be determined from adjacent cell coordinates.

    Arguments:
        array: Raster values.
        x: Either `xlim`, `x`, or `X`.
        y: Either `ylim`, `y`, or `Y`.
        crs: Coordinate reference system as int (EPSG) or str (Proj4 or WKT).

    Attributes (in addition to those inherited from `Grid`):
        array (numpy.ndarray): Raster values.
        zlim (numpy.ndarray): Limits of raster values (nanmin, nanmax).
        box3d (numpy.ndarray): Bounding box (xmin, ymin, zmin, xmax, ymax, zmax).
        datetime (datetime.datetime): Capture date and time.
        crs (int, str): Coordinate reference system as int (EPSG) or str (Proj4 or WKT).
    """

    def __init__(
        self,
        array: Union[Number, Iterable[Union[Number, Iterable[Number]]]],
        x: Iterable[Union[Number, Iterable[Number]]] = None,
        y: Iterable[Union[Number, Iterable[Number]]] = None,
        datetime: datetime.datetime = None,
        crs: Union[int, str] = None,
    ) -> None:
        self.array = array
        self.xlim, self._x, self._X = self._parse_xy(x, dim=0)
        self.ylim, self._y, self._Y = self._parse_xy(y, dim=1)
        self.datetime = datetime
        self.crs = crs
        # Placeholders
        self._Zf = None

    def __eq__(self, other: "Raster") -> bool:
        """Consider equal if coordinate system and values are equal."""
        return (
            np.array_equiv(self.array, other.array)
            and (self.xlim == other.xlim).all()
            and (self.ylim == other.ylim).all()
        )

    def __getitem__(
        self, indices: Tuple[Union[int, slice], Union[int, slice]]
    ) -> "Raster":
        """Extract raster subset with array indices."""
        if not isinstance(indices, tuple):
            indices = (indices, slice(None))
        if not all((isinstance(idx, (int, slice)) for idx in indices)):
            raise IndexError("Only integers and slices are valid indices")
        i, j = indices[0], indices[1]
        if not isinstance(i, slice):
            i = slice(i, i + 1)
        if not isinstance(j, slice):
            j = slice(j, j + 1)
        d = self.d
        if i.step and i.step > 1:
            d[1] *= i.step
        if j.step and j.step > 1:
            d[0] *= j.step
        x, y = self.x[j], self.y[i]
        if len(x) < 3:
            x = x[[0, -1]] + (-0.5, 0.5) * d[0:1]
        if len(y) < 3:
            y = y[[0, -1]] + (-0.5, 0.5) * d[1:2]
        return self.__class__(self.array[i, j], x=x, y=y, datetime=self.datetime)

    @classmethod
    def read(
        cls,
        path: str,
        band: int = 1,
        d: float = None,
        xlim: Iterable[Number] = None,
        ylim: Iterable[Number] = None,
        datetime: datetime.datetime = None,
        nan: Any = None,
    ) -> "Raster":
        """
        Read Raster from gdal raster file.

        See `gdal.Open()` for details.
        If raster is float and has a defined no-data value,
        no-data values are replaced with `np.nan`.
        Otherwise, the raster data is unchanged.

        Arguments:
            path: Path to file.
            band: Raster band to read (1 = first band).
            d: Target grid cell size.
            xlim: Crop bounds in x.
            ylim: Crop bounds in y.
            datetime: Capture date and time.
            nan: Raster value to replace with `null`. If provided, raster values are
                cast to :class:`float`.
        """
        raster = osgeo.gdal.Open(path, osgeo.gdal.GA_ReadOnly)
        transform = raster.GetGeoTransform()
        grid = Grid(
            (raster.RasterXSize, raster.RasterYSize),
            x=transform[0] + transform[1] * np.array([0, raster.RasterXSize]),
            y=transform[3] + transform[5] * np.array([0, raster.RasterYSize]),
        )
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
            xoff=int(cols[0]),
            yoff=int(rows[0]),
            win_xsize=int(win_xsize),
            win_ysize=int(win_ysize),
            buf_xsize=int(buf_xsize),
            buf_ysize=int(buf_ysize),
        )
        # FIXME: band.GetNoDataValue() not equal to read values due to rounding
        default_nan = band.GetNoDataValue()
        is_float = np.issubdtype(Z.dtype, np.floating)
        if nan is not None or (is_float and default_nan):
            if nan is None:
                nan = default_nan
            if not is_float:
                Z = Z.astype(float)
            Z[Z == nan] = np.nan
        crs = raster.GetProjection()
        return cls(Z, x=xlim, y=ylim, datetime=datetime, crs=crs if crs else None)

    @property
    def array(self) -> np.ndarray:
        """Raster values (ny, nx)."""
        return self._array

    @array.setter
    def array(
        self, value: Union[Number, Iterable[Union[Number, Iterable[Number]]]]
    ) -> None:
        value = np.atleast_2d(value)
        if hasattr(self, "_array"):
            self._clear_cache(["Zf"])
            if value.shape != self._array.shape:
                self._clear_cache(["x", "X", "y", "Y"])
        self._array = value

    # ---- Properties (dependent) ----

    @property
    def zlim(self) -> np.ndarray:
        """Raster value limits (nanmin, nanmax)."""
        value = [np.nanmin(self.array), np.nanmax(self.array)]
        return np.array(value)

    @property
    def size(self) -> np.ndarray:
        """Grid dimensions (nx, ny)."""
        return np.array(self.array.shape[0:2][::-1]).astype(int)

    @property
    def box3d(self) -> np.ndarray:
        """Bounding box (xmin, ymin, xmax, ymax)."""
        zlim = self.zlim
        return np.hstack((self.min, zlim.min(), self.max, zlim.max()))

    @property
    def grid(self) -> "Grid":
        """Raster grid."""
        return Grid(self.size, x=self.xlim, y=self.ylim)

    # ---- Properties (cached) ----

    @property
    def Zf(self) -> scipy.interpolate.RegularGridInterpolator:
        """Regular grid interpolator."""
        if self._Zf is None:
            sign = np.sign(self.d).astype(int)
            self._Zf = scipy.interpolate.RegularGridInterpolator(
                (self.x[:: sign[0]], self.y[:: sign[1]]),
                self.array.T[:: sign[0], :: sign[1]],
            )
        return self._Zf

    # ---- Methods (public) ----

    def copy(self) -> "Raster":
        """Copy raster."""
        return self.__class__(
            self.array.copy(),
            x=self.xlim.copy(),
            y=self.ylim.copy(),
            datetime=copy.copy(self.datetime),
        )

    def sample(
        self,
        xy: Iterable[Iterable[Number]],
        grid: bool = False,
        order: int = 1,
        bounds_error: bool = True,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        Sample at points.

        If `grid=False`:

            - Uses cached :class:`scipy.interpolate.RegularGridInterpolator`
              (:attr:`Zf`).
            - Supports interpolation `order` 0 and 1.
            - Faster for small sets of points.

        If `grid=True`:

            - Uses :class:`scipy.interpolate.RectBivariateSpline`.
            - Supports interpolation `order` 1 through 5.
            - Much faster for larger grids.

        If any dimension has length 1, the value of the singleton dimension(s)
        is returned directly and :class:`scipy.interpolate.interp1d` is used for the
        remaining dimension.

        Arguments:
            xy: Input coordinates x and y.

                - If `grid=False`, point coordinates (n, 2).
                - If `grid=True`, coordinate vectors (n,), (m,).

            grid: Whether `xy` defines a grid (`True`) or invidual points (`False`).
            order: Interpolation order
                (0: nearest, 1: linear, 2: quadratic, 3: cubic, 4: quartic, 5: quintic).
            bounds_error: Whether an error is raised if `xy` are outside bounds.
            fill_value: Value to use for points outside bounds.
                If `None`, values outside bounds are extrapolated.

        Returns:
            Raster value at each point,
            either as (n,) if `grid=False` or (m, n) if `grid=True`.

        Raises:
            ValueError: Some of the sampling coordinates are out of bounds.
        """
        error = ValueError("Some of the sampling coordinates are out of bounds")
        methods = ("nearest", "linear", "quadratic", "cubic", "quartic", "quintic")
        if bounds_error or fill_value is not None:
            # Test whether sampling points are in bounds
            xyin = self.inbounds(xy, grid=grid)
            if grid:
                xout, yout = ~xyin[0], ~xyin[1]
                if bounds_error and (xout.any() or yout.any()):
                    raise error
            else:
                xyout = ~xyin
                if bounds_error and xyout.any():
                    raise error
        has_fill = not bounds_error and fill_value is not None
        # Test which dimensions are non-singleton
        dims = np.where(self.size > 1)[0]
        ndims = len(dims)
        # Take samples
        if grid:
            # Sample at grid points
            if ndims == 2:
                # 2D: Use RectBivariateSpline
                kx, ky = order, order
                samples = self._sample_grid(xy, kx=kx, ky=ky)
            elif ndims == 1:
                # 1D: Use interp1d
                dim = dims[0]
                z = self._sample_1d(xy[dim], dim=dim, kind=methods[order])
                samples = np.tile(
                    z.reshape(-1 if dim else 1, 1 if dim else -1),
                    reps=(1 if dim else len(z), len(z) if dim else 1),
                )
            else:
                # 0D: Return constant
                samples = np.full((len(xy[0]), len(xy[1])), self.array.flat[0])
            if has_fill:
                # Fill out of bounds with value
                samples[yout, :] = fill_value
                samples[:, xout] = fill_value
        else:
            # Sample at points
            if has_fill:
                samples = np.full(len(xy), fill_value)
            if ndims == 2:
                # 2D: Use RegularGridInterpolator
                self.Zf.bounds_error = False
                self.Zf.fill_value = None
                if has_fill:
                    samples[xyin] = self.Zf(xy[xyin], method=methods[order])
                else:
                    samples = self.Zf(xy, method=methods[order])
            elif ndims == 1:
                # 1D: Use interp1d
                dim = dims[0]
                if has_fill:
                    samples[xyin] = self._sample_1d(
                        xy[xyin, dim], dim=dim, kind=methods[order]
                    )
                else:
                    samples = self._sample_1d(xy[:, dim], dim=dim, kind=methods[order])
            else:
                # 0D: Return constant
                if has_fill:
                    samples[xyin] = self.array.flat[0]
                else:
                    samples = np.full(len(xy), self.array.flat[0])
        return samples

    def _sample_1d(
        self, x: Iterable[Number], dim: Literal[0, 1], kind: Union[str, int] = "linear"
    ) -> np.ndarray:
        """Sample raster values with singleton dimension."""
        xdir = np.sign(self.d[dim]).astype(int)
        xi = (self.y if dim else self.x)[::xdir]
        zi = (self.array[:, 0] if dim else self.array[0])[::xdir]
        zxfun = scipy.interpolate.interp1d(
            x=xi, y=zi, kind=kind, assume_sorted=True, fill_value="extrapolate"
        )
        samples = zxfun(x)
        return samples

    def _sample_grid(
        self,
        xy: Tuple[Iterable[Number], Iterable[Number]],
        kx: int = 1,
        ky: int = 1,
        s: Number = 0,
    ) -> np.ndarray:
        """Sample raster values on regular grid."""
        x, y = xy
        signs = np.sign(self.d).astype(int)
        # HACK: scipy.interpolate.RectBivariateSpline does not support NAN
        Zmin = np.nanmin(self.array)
        is_nan = np.isnan(self.array)
        self.array[is_nan] = helpers.numpy_dtype_minmax(self.array.dtype)[0]
        fun = scipy.interpolate.RectBivariateSpline(
            self.y[:: signs[1]],
            self.x[:: signs[0]],
            self.array[:: signs[1], :: signs[0]],
            bbox=(min(self.ylim), max(self.ylim), min(self.xlim), max(self.xlim)),
            kx=kx,
            ky=ky,
            s=s,
        )
        xdir = 1 if (len(x) < 2) or x[1] > x[0] else -1
        ydir = 1 if (len(y) < 2) or y[1] > y[0] else -1
        samples = fun(y[::ydir], x[::xdir], grid=True)[::ydir, ::xdir]
        samples[samples < Zmin] = np.nan
        self.array[is_nan] = np.nan
        return samples

    def resample(self, grid: Grid, **kwargs: Any) -> None:
        """
        Resample to match coordinate system of other raster.

        Arguments:
            grid: Regular grid cell centers at which to sample.
            **kwargs: Optional arguments to :meth:`sample`.
        """
        array = self.sample((grid.x, grid.y), grid=True, **kwargs)
        self.array = array
        self.xlim, self.ylim = grid.xlim, grid.ylim
        self._x, self._y = grid.x, grid.y

    def plot(
        self, array: np.ndarray = None, **kwargs: Any
    ) -> matplotlib.image.AxesImage:
        """
        Plot.

        Arguments:
            array: Values to plot. If `None`, :attr:`array` is used.
            **kwargs: Optional arguments to :func:`matplotlib.pyplot.imshow`.
        """
        if array is None:
            array = self.array
        return matplotlib.pyplot.imshow(
            array,
            extent=(self.xlim[0], self.xlim[1], self.ylim[1], self.ylim[0]),
            **kwargs
        )

    def rasterize(self, xy: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Convert points to a raster image.

        Arguments:
            xy: Point coordinates (n, [x, y]).
            values: Point values (n, ).

        Returns:
            Image array (float) of mean values of the same dimensions as :attr:`array`.
            Pixels without points are `NaN`.
        """
        mask = self.inbounds(xy)
        rowcol = self.xy_to_rowcol(xy[mask, :], snap=True)
        array = self.array.copy()
        helpers.rasterize_points(rowcol[:, 0], rowcol[:, 1], values[mask], a=array)
        return array

    def rasterize_poygons(
        self,
        polygons: Iterable[Iterable[Iterable[Number]]],
        holes: Iterable[Iterable[Iterable[Number]]] = None,
    ) -> np.ndarray:
        """
        Convert polygons to a raster image.

        Returns a boolean array of the grid cells inside the polygons.

        Arguments:
            polygons: Polygons [[(xi, yi), ...], ...].
            holes: Polygons representing holes in `polygons` [[(xi, yi), ...], ...].
        """
        size = self.shape[0:2][::-1]
        polygons = [self.xy_to_rowcol(xy)[:, ::-1] + 0.5 for xy in polygons]
        if holes is not None:
            holes = [self.xy_to_rowcol(xy)[:, ::-1] + 0.5 for xy in holes]
        return helpers.polygons_to_mask(polygons, size=size, holes=holes)

    def crop(
        self,
        xlim: Iterable[Number] = None,
        ylim: Iterable[Number] = None,
        zlim: Iterable[Number] = None,
    ) -> None:
        """
        Crop.

        Arguments:
            xlim: Crop bounds in x.
            ylim: Crop bounds in y.
            zlim: Crop bounds in z.
                Values outside range are set to `np.nan` (casting to float as needed).
        """
        if xlim is not None or ylim is not None:
            xlim, ylim, rows, cols = self.crop_extent(xlim=xlim, ylim=ylim)
            self.array = self.array[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1]
            self.xlim = xlim
            self.ylim = ylim
        if zlim is not None:
            outbounds = (self.array < min(zlim)) | (self.array > max(zlim))
            if np.count_nonzero(outbounds) and not issubclass(
                self.array.dtype.type, np.floating
            ):
                warnings.warn("array cast to float to accommodate NaN")
                self.array = self.array.astype(float)
            self.array[outbounds] = np.nan

    def resize(self, scale: Number, order: int = 1) -> None:
        """
        Resize.

        Arguments:
            scale: Fraction of current size.
            order: Interpolation order
                (0: nearest, 1: linear, 2: quadratic, 3: cubic, 4: quartic, 5: quintic).
        """
        self.array = scipy.ndimage.zoom(self.array, zoom=float(scale), order=order)

    def shift(self, dx: Number = None, dy: Number = None, dz: Number = None) -> None:
        """
        Shift position.

        Arguments:
            dx: Shift in x.
            dy: Shift in y.
            dz: Shift in z.
        """
        self._shift_xy(dx=dx, dy=dy)
        if dz is not None:
            # Prevent reset of cached interpolants
            self._Z += dz
        if self._Zf is not None:
            # Shift cached interpolants
            if dx is not None:
                self._Zf.grid[0][:] += dx
            if dy is not None:
                self._Zf.grid[1][:] += dy
            if dy is not None:
                self._Zf.values += dz

    def fill_circle(
        self, center: Iterable[Number], radius: Number, value: Any = np.nan
    ) -> None:
        """
        Fill a circle with a fixed value.

        Arguments:
            center: Circle center (x, y).
            radius: Circle radius.
            value: Fill value.
        """
        # Circle indices
        rowcol = self.xy_to_rowcol(np.atleast_2d(center[0:2]), snap=True)
        r = np.round(radius / self.d[0])
        xyi = helpers.bresenham_circle(rowcol[0, ::-1], r).astype(int)
        # Filled circle indices
        ind = []
        y = np.unique(xyi[:, 1])
        yin = (y > -1) & (y < self.size[1])
        for yi in y[yin]:
            xb = xyi[xyi[:, 1] == yi, 0]
            xi = range(max(xb.min(), 0), min(xb.max(), self.size[0] - 1) + 1)
            if xi:
                rowcols = np.column_stack((np.repeat(yi, len(xi)), xi))
                ind.extend(self.rowcol_to_idx(rowcols))
        # Apply
        self.array.flat[ind] = value

    def hillshade(
        self, azimuth: Number = 315, altitude: Number = 45, **kwargs: Any
    ) -> np.ndarray:
        """
        Return the illumination intensity of the surface.

        Arguments:
            azimuth: Azimuth angle of the light source
                (0-360, degrees clockwise from North).
            altitude: Altitude angle of the light source
                (0-90, degrees up from horizontal).
            **kwargs: Optional arguments to
                :meth:`matplotlib.colors.LightSource.hillshade`.
        """
        light = matplotlib.colors.LightSource(azdeg=azimuth, altdeg=altitude)
        return light.hillshade(self.array, dx=self.d[0], dy=self.d[1], **kwargs)

    def fill_crevasses(
        self,
        maximum: dict = {"size": 5},
        gaussian: dict = {"sigma": 5},
        mask: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]] = None,
        fill: bool = False,
    ) -> None:
        """
        Apply a maximum filter to values, then perform Gaussian smoothing.

        Arguments:
            maximum: Optional arguments to :func:`helpers.maximum_filter`.
            gaussian: Optional arguments to :func:`helpers.gaussian_filter`.
            mask: Boolean array of cells to include (True) or exclude (False),
                or callable that generates the mask from :attr:`array`.
                If `None`, all cells are included.
            fill: Whether to fill cells excluded by `mask` with interpolated values.
        """
        if callable(mask):
            mask = mask(self.array)
        self.array = helpers.gaussian_filter(
            helpers.maximum_filter(self.array, **maximum, mask=mask, fill=fill),
            **gaussian,
            mask=mask,
            fill=fill
        )

    def viewshed(
        self, origin: Iterable[Number], correction: Optional[Union[bool, dict]] = False
    ) -> np.ndarray:
        """
        Return the binary viewshed from a point within the raster.

        Arguments:
            origin: World coordinates of viewing position (x, y, z).
            correction: Either arguments to :func:`helpers.elevation_corrections`,
                `True` for default arguments, or `None` or `False` to skip.

        Returns:
            Boolean array of the same shape as :attr:`array`
            with visible cells tagged as `True`.
        """
        if not all(abs(self.d[0]) == abs(self.d)):
            warnings.warn(
                "DEM cells not square "
                + str(tuple(abs(self.d)))
                + " - "
                + "may lead to unexpected results"
            )
        if not self.inbounds(np.atleast_2d(origin[0:2])):
            warnings.warn("Origin not in DEM - may lead to unexpected results")
        # Compute distance to all cell centers
        dx = np.tile(self.x - origin[0], self.size[1])
        dy = np.repeat(self.y - origin[1], self.size[0])
        dz = self.array.ravel() - origin[2]
        dxy = dx ** 2 + dy ** 2  # wait to square root
        if correction is True:
            correction = {}
        if isinstance(correction, dict):
            dz += helpers.elevation_corrections(dxy, **correction)
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
                return np.ones(self.array.shape, dtype=bool)
        rings = np.append(rings, len(ix))
        # Compute elevation ratio
        first_ring = ix[rings[0] : rings[1]]
        is_zero = np.where(dxy[first_ring] == 0)[0]
        dxy[first_ring[is_zero]] = np.nan
        elevation = dz / dxy
        # Compute max number of points on most distant ring
        # N = int(np.ceil(2 * np.pi * dxy_cell_sorted[-1]))
        # Initialize result raster
        vis = np.zeros(self.array.size, dtype=bool)
        # Loop through rings
        period = 2 * np.pi
        previous_headings = None
        max_elevations = False
        max_elevations_has_nan = False
        for k in range(len(rings) - 1):
            rix = ix[rings[k] : rings[k + 1]]
            rheading = heading[rix]
            relev = elevation[rix]
            # Test visibility
            if k > 0:
                # Interpolate max_elevations to current headings
                max_elevations = np.interp(
                    rheading, previous_headings, max_elevations, period=period
                )
                # NOTE: Throws warning if np.nan in relev
                is_visible = relev > max_elevations
                if max_elevations_has_nan:
                    is_nan_max_elevation = np.isnan(max_elevations)
                    new_visible = is_nan_max_elevation & ~np.isnan(relev)
                    is_visible |= new_visible
                    if np.count_nonzero(is_nan_max_elevation) == np.count_nonzero(
                        new_visible
                    ):
                        max_elevations_has_nan = False
                max_elevations[is_visible] = relev[is_visible]
            else:
                # First ring is always visible (if not NaN)
                is_visible = ~np.isnan(relev)
                max_elevations = relev
                max_elevations_has_nan = any(np.isnan(relev))
            vis[rix] = is_visible
            previous_headings = rheading
        return vis.reshape(self.array.shape)

    def horizon(
        self,
        origin: Iterable[Number],
        headings: Iterable[Number] = range(360),
        correction: Optional[Union[bool, dict]] = False,
    ) -> List[np.ndarray]:
        """
        Return the horizon from an arbitrary viewing position.

        Null values (`numpy.nan`) are ignored. A cell which is the last
        non-missing cell along a sighting is not considered part of the horizon.

        Arguments:
            origin: World coordinates of viewing position (x, y, z).
            headings: Headings at which to compute horizon,
                in degrees clockwise from north.
            correction: Either arguments to :func:`helpers.elevation_corrections()`,
                `True` for default arguments, or `None` or `False` to skip.

        Returns:
            List of world coordinate arrays (n, [x, y, z]) each tracing an unbroken
            segment of the horizon.
        """
        n = len(headings)
        if correction is True:
            correction = {}
        # Compute ray directions (2d)
        headings = np.array(headings, dtype=float)
        thetas = -(headings - 90) * (np.pi / 180)
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
            dz = self.array.flat[idx] - origin[2]
            xy = self.rowcol_to_xy(rowcol)
            dxy = np.sum((xy - origin[0:2]) ** 2, axis=1)  # wait to take square root
            if isinstance(correction, dict):
                delta = helpers.elevation_corrections(dxy, **correction)
                maxi = np.nanargmax((dz + delta) / np.sqrt(dxy))
            else:
                maxi = np.nanargmax(dz / np.sqrt(dxy))
            # Save point if not last non-nan value
            if maxi < (len(dz) - 1) and np.any(~np.isnan(dz[maxi + 1 :])):
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

    def gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return gradients in x and y.

        Returns:
            array: Derivative of :attr:`array` with respect to x.
            array: Derivative of :attr:`array` with respect to y.
        """
        dzdy, dzdx = np.gradient(self.array, self.d[1], self.d[0])
        return dzdx, dzdy

    def write(self, path: str, **kwargs: Any) -> None:
        """
        Write to file.

        Arguments:
            path: Path to file.
            **kwargs: Optional arguments to :func:`helpers.write_raster`.
        """
        kwargs = {
            # top-left x, dx, rotation, top-left y, rotation, dy
            "transform": (self.xlim[0], self.d[0], 0, self.ylim[0], 0, self.d[1]),
            "crs": self.crs,
            **kwargs,
        }
        helpers.write_raster(a=self.array, path=path, **kwargs)

    def data_extent(self) -> Tuple[slice, slice]:
        """
        Return slices for the region bounding all non-missing values.

        Returns:
            Row slice and column slice.

        Raises:
            ValueError: No non-missing values present.
        """
        data = ~np.isnan(self.array)
        data_row = np.any(data, axis=1)
        first_data_row = np.argmax(data_row)
        if first_data_row == 0 and not data_row[0]:
            raise ValueError("No non-missing values present")
        last_data_row = data_row.size - np.argmax(data_row[::-1])
        data_col = np.any(data, axis=0)
        first_data_col = np.argmax(data_col)
        last_data_col = data_col.size - np.argmax(data_col[::-1])
        return (
            slice(first_data_row, last_data_row),
            slice(first_data_col, last_data_col),
        )

    def crop_to_data(self) -> None:
        """Crop to bounds of non-missing values."""
        slices = self.data_extent()
        x = self.x[slices[1]]
        y = self.y[slices[0]]
        self.xlim = x[[0, -1]] + (-0.5, 0.5) * self.d[0:1]
        self.ylim = y[[0, -1]] + (-0.5, 0.5) * self.d[1:2]
        self.array = self.array[slices]
        self._x = x
        self._y = y


class RasterInterpolant:
    """
    Interpolation of a raster timeseries.

    Attributes:
        means (iterable): Mean values as Rasters, paths to raster files,
            or numbers (interpreted as infinite rasters).
        sigmas (iterable): Standard deviations as Rasters, paths to raster files,
            or numbers (interpreted as infinite rasters).
            If `None`, defaults to zero.
        x (numpy.ndarray): 1-dimensional coordinates of the observations,
            as either numbers or :class:`datetime.datetime`.
            If `None`, tries to read datetimes from `means`.
    """

    def __init__(
        self,
        means: Iterable[Union[Raster, str, Number]],
        sigmas: Iterable[Union[Raster, str, Number]] = None,
        x: Iterable[Union[Number, datetime.datetime]] = None,
    ) -> None:
        self.means = means
        if x is None:
            x = [raster.datetime for raster in means]
        self.x = np.asarray(x)
        self.sigmas = sigmas

    def _parse_as_raster(
        self,
        obj: Union[Raster, str, Number],
        xi: Union[Number, datetime.datetime] = None,
        d: Number = None,
        xlim: Iterable[Number] = None,
        ylim: Iterable[Number] = None,
    ) -> Raster:
        """Parse object as a Raster."""
        t = xi if isinstance(xi, datetime.datetime) else None
        if isinstance(obj, numbers.Number):
            # Scalar
            # Infinite bounds unless specified
            if xlim is None:
                xlim = (-np.inf, np.inf)
            if ylim is None:
                ylim = (-np.inf, np.inf)
            return Raster(obj, x=xlim, y=ylim, datetime=t)
        if isinstance(obj, Raster):
            # Raster
            # Copy and reshape to specified bounds
            # NOTE: Adjust to match grid when read from file
            d_change = d is not None and d != np.abs(obj.d).mean()
            xlim_change = xlim is not None and sorted(xlim) != sorted(obj.xlim)
            ylim_change = ylim is not None and sorted(ylim) != sorted(obj.ylim)
            if any((d_change, xlim_change, ylim_change)):
                obj = obj.copy()
            if xlim_change or ylim_change:
                obj.crop(xlim=xlim, ylim=ylim)
            if d_change:
                scale = d / np.abs(obj.d).mean()
                obj.resize(scale)
            return obj
        if isinstance(obj, str):
            # Path to raster
            # Read from file
            return Raster.read(obj, d=d, xlim=xlim, ylim=ylim, datetime=t)
        raise ValueError("Cannot cast as Raster: " + str(type(obj)))

    def _read_mean(
        self,
        index: int,
        d: Number = None,
        xlim: Iterable[Number] = None,
        ylim: Iterable[Number] = None,
        zlim: Iterable[Number] = None,
        fun: Callable = None,
        **kwargs: Any
    ) -> Raster:
        """Parse mean raster."""
        xi = self.x[index]
        obj = self.means[index]
        raster = self._parse_as_raster(obj, xi, d=d, xlim=xlim, ylim=ylim)
        if (zlim is not None or fun is not None) and raster is obj:
            raster = raster.copy()
        if zlim is not None:
            raster.crop(zlim=zlim)
        if fun is not None:
            fun(raster, **kwargs)
        return raster

    def _read_sigma(
        self,
        index: int,
        d: Number = None,
        xlim: Iterable[Number] = None,
        ylim: Iterable[Number] = None,
    ) -> Raster:
        """Parse sigma raster."""
        xi = self.x[index]
        if self.sigmas is None:
            obj = 0
        else:
            obj = self.sigmas[index]
        return self._parse_as_raster(obj, xi, d=d, xlim=xlim, ylim=ylim)

    def _read_mean_grid(self, index: int) -> Grid:
        """Parse mean raster grid."""
        obj = self.means[index]
        if isinstance(obj, Raster):
            return obj.grid
        if isinstance(obj, str):
            return Grid.read(obj)
        if isinstance(obj, numbers.Number):
            return Grid((1, 1), x=(-np.inf, np.inf), y=(-np.inf, np.inf))
        raise ValueError("Cannot cast as Grid: " + str(type(obj)))

    def nearest(
        self, xi: Union[Number, datetime.datetime], extrapolate: bool = False
    ) -> Tuple[int, int]:
        """
        Return the indices of the two nearest Rasters.

        Arguments:
            xi: 1-dimensional coordinate.
            extrapolate: Whether to return the two nearest Rasters (True)
                or only if the Rasters are on either side of `xi` (False).

        Raises:
            ValueError: Not bounded on both sides by a Raster (`extrapolate=False`).
        """
        dx = self.x - xi
        zero = type(dx[0])(0)
        if extrapolate:
            # Get two nearest DEMs
            i, j = abs(dx).argsort()[:2]
        else:
            # Get nearest DEM on either side of t
            before = np.where(dx <= zero)[0]
            after = np.where(dx >= zero)[0]
            if not before.size or not after.size:
                raise ValueError("Not bounded on both sides by a Raster")
            i = before[np.argmin(abs(dx[before]))]
            j = after[np.argmin(dx[after])]
        ij = [i, j]
        ij.sort(key=lambda index: self.x[index])
        return tuple(ij)

    def _interpolate(
        self,
        means: Iterable[Raster],
        x: Iterable[Union[Number, datetime.datetime]],
        xi: Union[Number, datetime.datetime],
        sigmas: Iterable[Raster] = None,
    ) -> Union[Raster, Tuple[Raster, Raster]]:
        """Interpolate between two rasters."""
        dz = means[1].array - means[0].array
        dx = x[1] - x[0]
        scale = (xi - x[0]) / dx
        z = means[0].array + dz * scale
        t = xi if isinstance(xi, datetime.datetime) else None
        raster = means[0].__class__(z, x=means[0].xlim, y=means[0].ylim, datetime=t)
        if sigmas is not None:
            # Bounds uncertainty: error propagation of z above
            # NOTE: 'a * (1 - scale) + b * scale' form underestimates uncertainty
            z_var = sigmas[0].array ** 2 + scale ** 2 * (
                sigmas[0].array ** 2 + sigmas[1].array ** 2
            )
            # Interpolation uncertainty: nearest bound at 99.7%
            nearest_dx = np.min(np.abs(np.subtract(xi, x)))
            zi_var = ((1 / 3) * dz * (nearest_dx / dx)) ** 2
            sigma = raster.__class__(
                np.sqrt(z_var + zi_var), x=means[0].xlim, y=means[0].ylim, datetime=t
            )
            return raster, sigma
        return raster

    def __call__(
        self,
        xi: Union[Number, datetime.datetime],
        d: Number = None,
        xlim: Iterable[Number] = None,
        ylim: Iterable[Number] = None,
        zlim: Iterable[Number] = None,
        return_sigma: bool = False,
        extrapolate: bool = False,
        fun: Callable = None,
        **kwargs: Any
    ) -> Union[Raster, Tuple[Raster, Raster]]:
        """
        Return the interpolated Raster.

        Arguments:
            xi: 1-dimensional coordinate of the interpolated Raster.
            d: Target grid cell size.
                If `None`, the largest Raster cell size is used.
            xlim: Crop bounds in x.
                If `None`, the intersection of the Rasters is used.
            ylim: Crop bounds in y.
                If `None`, the intersection of the Rasters is used.
            zlim: Crop bounds in z (means only).
                Values outside range are set to `np.nan`.
            return_sigma: Whether to return sigmas.
            extrapolate: Whether to use the two nearest Rasters (True)
                or only if the Rasters are on either side of `xi` (False)
            fun: Function to apply to each Raster (means only) before
                interpolation, with signature `fun(raster, **kwargs)`.
                Must modify Raster in place.
            **kwargs: Additional arguments passed to `fun`.

        Returns:
            Interpolated mean raster and
            (if `return_sigma=True`) standard deviation raster.
        """
        ij = self.nearest(xi, extrapolate=extrapolate)
        # Determine common grid (lowest resolution, smallest extent)
        grids = [self._read_mean_grid(k) for k in ij]
        if d is None:
            d = np.max(np.abs(np.stack([grid.d for grid in grids])))
        if xlim is None:
            xlim = (-np.inf, np.inf)
        if ylim is None:
            ylim = (-np.inf, np.inf)
        boxes = [grid.box2d for grid in grids]
        boxes.append([min(xlim), min(ylim), max(xlim), max(ylim)])
        box = helpers.intersect_boxes(boxes)
        xlim, ylim = box[0::2], box[1::2]
        # Read mean rasters
        means = [
            self._read_mean(k, d=d, xlim=xlim, ylim=ylim, zlim=zlim, fun=fun, **kwargs)
            for k in ij
        ]
        if means[0].grid != means[1].grid:
            if means[1] is self.means[ij[1]]:
                means[1] = means[1].copy()
            means[1].resample(means[0])
        # Read sigma rasters
        if return_sigma:
            sigmas = [self._read_sigma(k, d=d, xlim=xlim, ylim=ylim) for k in ij]
            if sigmas[0].grid != sigmas[1].grid:
                if sigmas[1] is self.sigmas[ij[1]]:
                    sigmas[1] = sigmas[1].copy()
                sigmas[1].resample(sigmas[0])
        else:
            sigmas = None
        # Interpolate
        return self._interpolate(means=means, sigmas=sigmas, x=self.x[list(ij)], xi=xi)
