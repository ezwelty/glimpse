"""Convert between world and image coordinates using a distorted camera model."""
import copy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas
import scipy.interpolate
import scipy.ndimage
import scipy.optimize

from . import config, helpers
from .raster import Raster

Number = Union[int, float]
Array = Union[Sequence[Number], np.ndarray]
Vector = Union[Number, Array]


class Camera:
    """
    Distorted camera model.

    A `Camera` converts between 3-D world coordinates and their corresponding 2-D image
    coordinates.

    By default, cameras are initialized at the origin (0, 0, 0), parallel with the
    horizon (xy-plane), and pointed north (+y). All attributes are coerced to
    `numpy.ndarray` during initialization or when individually set. The focal length in
    pixels (:attr:`f`) is calculated from :attr:`fmm` and :attr:`sensorsz` if both are
    provided. The principal point offset in pixels (:attr:`c`) is calculated from
    :attr:`cmm` and :attr:`sensorsz` if both are provided.

    Attributes:
        vector (numpy.ndarray): Vector of the core camera attributes
            (:attr:`xyz`, :attr:`viewdir`, :attr:`imgsz`, :attr:`f`, :attr:`c`,
            :attr:`k`, :attr:`p`)
        xyz (numpy.ndarray): Position in world coordinates (x, y, z)
        viewdir (numpy.ndarray): View direction in degrees (yaw, pitch, roll)

            - yaw: clockwise rotation about z-axis (0 = look north)
            - pitch: rotation from horizon (+ look up, - look down)
            - roll: rotation about optical axis (+ down right, - down left, from behind)

        imgsz (numpy.ndarray): Image size in pixels (nx, ny)
        f (numpy.ndarray): Focal length in pixels (fx, fy)
        c (numpy.ndarray): Principal point offset from the image center in pixels
            (dx, dy)
        k (numpy.ndarray): Radial distortion coefficients (k1, ..., k6)
        p (numpy.ndarray): Tangential distortion coefficients (p1, p2)
        sensorsz (numpy.ndarray): Sensor size in millimeters (nx, ny)
        fmm (numpy.ndarray): Focal length in millimeters (fx, fy).
        cmm (numpy.ndarray): Principal point offset from the image center in millimeters
            (dx, dy).
        R (numpy.ndarray): Rotation matrix equivalent of :attr:`viewdir` (3, 3).
            Assumes an initial camera orientation with +z pointing up, +x east,
            and +y north.
        Rprime (numpy.ndarray): Derivative of :attr:`R` with respect to :attr:`viewdir`.
            Used for fast Jacobian (gradient) calculations by
            :class:`optimize.ObserverCameras`.
        original_vector (numpy.ndarray): Value of :attr:`vector` when first initialized
    """

    def __init__(
        self,
        imgsz: Vector,
        f: Vector = None,
        c: Vector = None,
        sensorsz: Vector = None,
        fmm: Vector = None,
        cmm: Vector = None,
        k: Vector = (0, 0, 0, 0, 0, 0),
        p: Vector = (0, 0),
        xyz: Vector = (0, 0, 0),
        viewdir: Vector = (0, 0, 0),
    ) -> None:
        if (fmm is not None or cmm is not None) and sensorsz is None:
            raise ValueError("'fmm' or 'cmm' provided without 'sensorsz'")
        if f is not None and fmm is not None:
            raise ValueError("'f' and 'fmm' cannot both be provided")
        if c is not None and cmm is not None:
            raise ValueError("'c' and 'cmm' cannot both be provided")
        self.vector = np.full(20, np.nan, dtype=float)
        self.xyz = xyz
        self.viewdir = viewdir
        self.imgsz = imgsz
        self.sensorsz = sensorsz
        if fmm is not None:
            f = helpers.format_list(fmm, length=2) * self.imgsz / self.sensorsz
        if f is None:
            raise ValueError("Either 'f' or 'fmm' must be provided")
        self.f = f
        if cmm is not None:
            c = helpers.format_list(cmm, length=2) * self.imgsz / self.sensorsz
        if c is None:
            c = (0, 0)
        self.c = c
        self.k = k
        self.p = p
        self.original_vector = self.vector.copy()

    # ---- Properties (dependent) ----

    @property
    def xyz(self) -> np.ndarray:
        """Position in world coordinates (x, y, z)."""
        return self.vector[0:3]

    @xyz.setter
    def xyz(self, value: Vector) -> None:
        self.vector[0:3] = helpers.format_list(value, length=3, default=0)

    @property
    def viewdir(self) -> np.ndarray:
        """
        View direction in degrees (yaw, pitch, roll).

        - yaw: clockwise rotation about z-axis (0 = look north).
        - pitch: rotation from horizon (+ look up, - look down).
        - roll: rotation about optical axis (+ down right, - down left, from behind).
        """
        return self.vector[3:6]

    @viewdir.setter
    def viewdir(self, value: Vector) -> None:
        self.vector[3:6] = helpers.format_list(value, length=3, default=0)

    @property
    def imgsz(self) -> np.ndarray:
        """Image size in pixels (nx, ny)."""
        return self.vector[6:8]

    @imgsz.setter
    def imgsz(self, value: Vector) -> None:
        self.vector[6:8] = helpers.format_list(value, length=2)

    @property
    def f(self) -> np.ndarray:
        """Focal length in pixels (fx, fy)."""
        return self.vector[8:10]

    @f.setter
    def f(self, value: Vector) -> None:
        self.vector[8:10] = helpers.format_list(value, length=2)

    @property
    def c(self) -> np.ndarray:
        """Principal point offset from the image center in pixels (dx, dy)."""
        return self.vector[10:12]

    @c.setter
    def c(self, value: Vector) -> None:
        self.vector[10:12] = helpers.format_list(value, length=2, default=0)

    @property
    def k(self) -> np.ndarray:
        """Radial distortion coefficients (k1, k2, k3, k4, k5, k6)."""
        return self.vector[12:18]

    @k.setter
    def k(self, value: Vector) -> None:
        self.vector[12:18] = helpers.format_list(value, length=6, default=0)

    @property
    def p(self) -> np.ndarray:
        """Tangential distortion coefficients (p1, p2)."""
        return self.vector[18:20]

    @p.setter
    def p(self, value: Vector) -> None:
        self.vector[18:20] = helpers.format_list(value, length=2, default=0)

    @property
    def sensorsz(self) -> Optional[np.ndarray]:
        """Sensor size in millimeters (nx, ny)."""
        return self._sensorsz

    @sensorsz.setter
    def sensorsz(self, value: Vector = None) -> np.ndarray:
        if value is not None:
            value = np.array(helpers.format_list(value, length=2), dtype=float)
        self._sensorsz = value

    @property
    def fmm(self) -> Optional[np.ndarray]:
        """Focal length in millimeters (fx, fy)."""
        if self.sensorsz is None:
            return None
        return self.f * self.sensorsz / self.imgsz

    @property
    def cmm(self) -> Optional[np.ndarray]:
        """Principal point offset from the image center in millimeters (dx, dy)."""
        if self.sensorsz is None:
            return None
        return self.c * self.sensorsz / self.imgsz

    @property
    def R(self) -> np.ndarray:
        """
        Rotation matrix equivalent of :attr:`viewdir` (3, 3).

        Assumes an initial camera orientation with
        +z pointing up, +x pointing east, and +y pointing north.
        """
        # Initial rotations of camera reference frame
        # (camera +z pointing up, with +x east and +y north)
        # Point camera north: -90 deg counterclockwise rotation about x-axis
        #   ri = [1 0 0; 0 cosd(-90) sind(-90); 0 -sind(-90) cosd(-90)];
        # (camera +z now pointing north, with +x east and +y down)
        # yaw: counterclockwise rotation about y-axis
        # (relative to north, from above: +cw, - ccw)
        #   ry = [C1 0 -S1; 0 1 0; S1 0 C1];
        # pitch: counterclockwise rotation about x-axis
        # (relative to horizon: + up, - down)
        #   rp = [1 0 0; 0 C2 S2; 0 -S2 C2];
        # roll: counterclockwise rotation about z-axis
        # (from behind camera: + ccw, - cw)
        #   rr = [C3 S3 0; -S3 C3 0; 0 0 1];
        # Apply all rotations in order
        #   R = rr * rp * ry * ri;
        radians = np.deg2rad(self.viewdir)
        C = np.cos(radians)
        S = np.sin(radians)
        return np.array(
            [
                [
                    C[0] * C[2] + S[0] * S[1] * S[2],
                    C[0] * S[1] * S[2] - C[2] * S[0],
                    -C[1] * S[2],
                ],
                [
                    C[2] * S[0] * S[1] - C[0] * S[2],
                    S[0] * S[2] + C[0] * C[2] * S[1],
                    -C[1] * C[2],
                ],
                [C[1] * S[0], C[0] * C[1], S[1]],
            ]
        )

    @property
    def Rprime(self) -> np.ndarray:
        """
        Derivative of :attr:`R` with respect to :attr:`viewdir`.

        Used for fast Jacobian (gradient) calculations by
        :class:`optimize.ObserverCameras`.
        """
        radians = np.deg2rad(self.viewdir)
        C = np.cos(radians)
        S = np.sin(radians)
        Rprime = np.stack(
            (
                [
                    [
                        C[0] * S[1] * S[2] - S[0] * C[2],
                        S[0] * S[2] + C[0] * S[1] * C[2],
                        C[0] * C[1],
                    ],
                    [
                        -S[0] * S[1] * S[2] - C[0] * C[2],
                        C[0] * S[2] - S[0] * S[1] * C[2],
                        -S[0] * C[1],
                    ],
                    [0, 0, 0],
                ],
                [
                    [S[0] * C[1] * S[2], S[0] * C[1] * C[2], -S[0] * S[1]],
                    [C[0] * C[1] * S[2], C[0] * C[1] * C[2], -C[0] * S[1]],
                    [S[1] * S[2], S[1] * C[2], C[1]],
                ],
                [
                    [
                        S[0] * S[1] * C[2] - C[0] * S[2],
                        -S[0] * S[1] * S[2] - C[0] * C[2],
                        0,
                    ],
                    [
                        S[0] * S[2] + C[0] * S[1] * C[2],
                        S[0] * C[2] - C[0] * S[1] * S[2],
                        0,
                    ],
                    [-C[1] * C[2], C[1] * S[2], 0],
                ],
            ),
            axis=1,
        )
        return Rprime * (np.pi / 180)

    @property
    def original_imgsz(self) -> np.ndarray:
        """Original image size in pixels (nx, ny)."""
        return self.original_vector[6:8]

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape in pixels (ny, nx)."""
        return int(self.imgsz[1]), int(self.imgsz[0])

    # ----- Methods (class) ----

    @classmethod
    def from_json(cls, path: str, **kwargs: Any) -> "Camera":
        """
        Read Camera from JSON.

        See :meth:`to_json` for the reverse.

        Arguments:
            path (str): Path to JSON file
            **kwargs: Additional parameters passed to :meth:`Camera`.
                These override any parameters read from **path**.

        Returns:
            A :class:`Camera` object
        """
        json_args = helpers.read_json(path)
        for key in json_args:
            # Conversion to float converts None to nan
            value = np.array(json_args[key], dtype=float)
            if np.isnan(value).all():
                value = None
            json_args[key] = value
        args = {**json_args, **kwargs}
        return cls(**args)

    # ---- Methods (static) ----

    @staticmethod
    def get_scale_from_size(
        old_size: Sequence[int], new_size: Sequence[int]
    ) -> Optional[float]:
        """
        Return the scale factor that achieves a target image size.

        Arguments:
            old_size: Initial image size (nx, ny)
            new_size: Target image size (nx, ny)

        Returns:
            Scale factor, or `None` if **new_size** cannot be achieved exactly
        """
        old = np.asarray(old_size)
        new = np.asarray(new_size)
        if all(new == old):
            return 1.0
        scale_bounds = new / old
        if scale_bounds[0] == scale_bounds[1]:
            return scale_bounds[0]

        def err(scale: float) -> float:
            return np.sum(np.abs(np.floor(scale * old + 0.5) - new_size))

        fit = scipy.optimize.minimize(
            err, x0=scale_bounds.mean(), bounds=[scale_bounds]
        )
        if err(fit["x"]) == 0:
            return fit["x"]
        return None

    # ---- Methods (public) ----

    def copy(self) -> "Camera":
        """
        Return a copy of this camera.

        The original state of the copy (to which :meth:`reset` reverts)
        will be the current state of this camera, not its original state.

        Example:
            Create a camera `cam`, modify it, then make a reference `rcam`
            and a copy `ccam`.

            >>> cam = Camera(imgsz=1, f=1)
            >>> cam.f[0] = 2
            >>> rcam = cam
            >>> ccam = cam.copy()

            If we modify `cam`, only the reference `rcam` changes.

            >>> cam.f[0] = 3
            >>> cam.f[0] == rcam.f[0]
            True
            >>> cam.f[0] == ccam.f[0]
            False

            If we modify, then reset, the copy `ccam`,
            it resets to its original state, not the original state of `cam`.

            >>> ccam.f[0] = 4
            >>> ccam.reset()
            >>> ccam.f[0] == 2
            True
            >>> cam.reset()
            >>> cam.f[0] == 1
            True
        """
        cam = copy.deepcopy(self)
        cam.original_vector = cam.vector.copy()
        return cam

    def reset(self) -> None:
        """
        Reset this camera to its original state.

        Example:
            >>> cam = Camera(imgsz=1, f=1)
            >>> cam.f[0] += 1
            >>> cam.reset()
            >>> cam.f[0] == 1
        """
        self.vector = self.original_vector.copy()

    def to_dict(
        self,
        attributes: Sequence[str] = ("xyz", "viewdir", "imgsz", "f", "c", "k", "p"),
    ) -> Dict[str, tuple]:
        """
        Return this camera as a dictionary.

        Arguments:
            attributes: Names of attributes to include.

        Returns:
            Attribute names and values.

        Example:
            >>> cam = Camera(imgsz=(8, 6), f=(7.9, 6.1))
            >>> cam.to_dict()
            {..., 'imgsz': (8.0, 6.0), 'f': (7.9, 6.1), ...}
            >>> cam.to_dict(('imgsz', 'f'))
            {'imgsz': (8.0, 6.0), 'f': (7.9, 6.1)}
        """
        return {key: tuple(getattr(self, key)) for key in attributes}

    def to_json(
        self,
        path: str = None,
        attributes: Sequence[str] = ("xyz", "viewdir", "imgsz", "f", "c", "k", "p"),
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Write or return this camera as JSON.

        See :meth:`from_json` for the reverse.

        Arguments:
            path: Path of file to write to.
                If `None`, a JSON-formatted string is returned.
            attributes: Attributes to include.
            **kwargs: Additional arguments to :func:`helpers.write_json()`

        Returns:
            Attribute names and values as a JSON-formatted string,
                or `None` if **path** is specified.

        Example:
            >>> cam = Camera(imgsz=(8, 6), f=(7.9, 6.1))
            >>> print(cam.to_json())
            {..., "imgsz": [8.0, 6.0], "f": [7.9, 6.1], ...}
            >>> print(cam.to_json(indent=4, flat_arrays=True))
            {
                "xyz": [0.0, 0.0, 0.0],
                "viewdir": [0.0, 0.0, 0.0],
                "imgsz": [8.0, 6.0],
                "f": [7.9, 6.1],
                "c": [0.0, 0.0],
                "k": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "p": [0.0, 0.0]
            }
        """
        obj = self.to_dict(attributes=attributes)
        return helpers.write_json(obj, path=path, **kwargs)

    def idealize(self) -> None:
        """
        Make this camera ideal (remove all distortions).

        Sets the principal point offset (:attr:`c`) and radial and tangential distortion
        coefficients (:attr:`k` and :attr:`p`) to zero.

        Example:
            >>> cam = Camera(imgsz=1, f=1, c=(0.1, 0.2), k=(0.1, 0.2), p=(0.1, 0.2))
            >>> cam.idealize()
            >>> all(cam.c == 0)
            True
            >>> all(cam.k == 0)
            True
            >>> all(cam.p == 0)
            True
        """
        self.k = np.zeros(6, dtype=float)
        self.p = np.zeros(2, dtype=float)
        self.c = np.zeros(2, dtype=float)

    def resize(self, size: Vector = 1, force: bool = False) -> None:
        """
        Resize this camera.

        Scales image size (:attr:`imgsz`), focal length (:attr:`f`),
        and principal point offset (:attr:`c`) accordingly.

        Arguments:
            size: Target image size (ny, ny) or factor of current size (float).
            force: Whether to allow the target image size even if it does not preserve
                the original aspect ratio.

        Raises:
            ValueError: Target image size does not preserve the original aspect ratio.

        Example:
            >>> cam = Camera(imgsz=(10, 20), f=(1, 2), c=(0.1, 0.2))
            >>> cam.resize(2)
            >>> cam.imgsz
            array([20., 40.])
            >>> cam.f
            array([2., 4.])
            >>> cam.c
            array([0.2, 0.4])

            If the target image size does not preserve the original aspect ratio,
            it is rejected by default.

            >>> cam.resize((11, 20))
            Traceback (most recent call last):
              ...
            ValueError: Target image size does not preserve the original aspect ratio
            >>> cam.resize((11, 20), force=True)
            >>> cam.imgsz
            array([11., 20.])
        """
        scale1d = np.atleast_1d(size)
        if len(scale1d) > 1 and force:
            # Use target size without checking for scalar scale factor
            new_size = scale1d
        else:
            if len(scale1d) > 1:
                # Compute scalar scale factor (if one exists)
                scale1d = Camera.get_scale_from_size(self.original_imgsz, scale1d)
                if scale1d is None:
                    raise ValueError(
                        "Target image size does not preserve the original aspect ratio"
                    )
            new_size = np.floor(scale1d * self.original_imgsz + 0.5)
        scale2d = new_size / self.imgsz
        # Ensure whole numbers
        self.imgsz = np.round(new_size)
        self.f *= scale2d
        self.c *= scale2d

    def project(
        self,
        xyz: np.ndarray,
        directions: bool = False,
        correction: Union[bool, dict] = False,
        return_depth: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Project world coordinates to image coordinates.

        Arguments:
            xyz: World coordinates (n, [x, y, z]).
            directions: Whether `xyz` are absolute coordinates (`False`)
                or ray directions (`True`).
            correction: Optional arguments to `helpers.elevation_corrections()` (dict),
                `True` for default arguments, or `False` to skip.
                Only applies if `directions` is `False`.
            return_depth: Whether to return the distance of each world point
                along the camera's optical axis.

        Returns:
            Image coordinates of the world points (n, 2).
            (optional) Distances of the world points (n, ).

        Example:
            By default, cameras are initialized at the origin (0, 0, 0),
            parallel with the horizon (xy-plane), pointed north (+y),
            and distortion-free. For such a camera, the world point (0, 10, 0),
            which is on the +y axis and a distance of 10 from the
            origin, will be projected onto the image center with a depth of 10.

            >>> cam = Camera(imgsz=10, f=10)
            >>> xyz = np.array([(0, 10, 0)])
            >>> cam.project(xyz)
            array([[5., 5.]])
            >>> cam.project(xyz, return_depth=True)
            (array([[5., 5.]]), array([10.]))
        """
        xy = self._world2camera(
            xyz, directions=directions, correction=correction, return_depth=return_depth
        )
        if return_depth:
            xy, depth = xy
        uv = self._camera2image(xy)
        if return_depth:
            return uv, depth
        return uv

    def invproject(
        self, uv: np.ndarray, directions: bool = True, depth: Vector = 1,
    ) -> np.ndarray:
        """
        Project image coordinates to world coordinates or ray directions.

        Arguments:
            uv: Image coordinates (n, [u, v]).
            directions: Whether to return world ray directions relative
                to the camera position (True) or absolute coordinates by adding
                on the camera position (False).
            depth: Distance of rays along the camera's optical axis, as either a
                scalar or a vector (n, ).

        Returns:
            World coordinates or ray directions (n, [x, y, z]).

        Example:
            By default, cameras are initialized at the origin (0, 0, 0),
            parallel with the horizon (xy-plane), pointed north (+y),
            and distortion-free. For such a camera, the image point (5, 5),
            at the center of the image, will be projected out the camera along the
            +y axis.

            >>> cam = Camera(imgsz=10, f=10)
            >>> uv = np.array([(5, 5)])
            >>> cam.invproject(uv)
            array([[0., 1., 0.]])
            >>> cam.invproject(uv, depth=10)
            array([[ 0., 10., 0.]])
        """
        xy = self._image2camera(uv)
        xyz = self._camera2world(xy, directions=directions, depth=depth)
        return xyz

    def infront(self, xyz: np.ndarray, directions: bool = False) -> np.ndarray:
        """
        Test whether world coordinates are in front of the camera.

        Arguments:
            xyz: World coordinates (n, [x, y, y]).
            directions: Whether `xyz` are ray directions (True)
                or absolute coordinates (False).

        Returns:
            Boolean mask (n, ).

        Example:
            For a default camera at the origin (0, 0, 0),
            parallel with the horizon (xy-plane), pointed north (+y),
            any points with +y coordinates are in front of the camera,
            regardless of whether they are actually in the image frame.

            >>> cam = Camera(imgsz=10, f=10)
            >>> xyz = np.array([(1000, 10, 0), (0, 10, 0), (0, 0, 0), (0, -10, 0)])
            >>> cam.infront(xyz)
            array([ True, True, False, False])
            >>> uv = cam.project(xyz)
            >>> uv
            array([[1005.,    5.],
                   [   5.,    5.],
                   [  nan,   nan],
                   [  nan,   nan]])
            >>> cam.inframe(uv)
            array([False, True, False, False])
        """
        dxyz = xyz if directions else xyz - self.xyz
        z = np.dot(dxyz, self.R.T[:, 2])
        return z > 0

    def inframe(self, uv: np.ndarray) -> np.ndarray:
        """
        Test whether image coordinates are in (or on) the image frame.

        Arguments:
            uv: Image coordinates (n, [u, v]).

        Returns:
            Boolean mask (n, ).

        Example:
            >>> cam = Camera(imgsz=(10, 12), f=10)
            >>> uv = np.array([(-1, 1), (0, 0), (9, 11), (10, 15)])
            >>> cam.inframe(uv)
            array([False,  True,  True, False])
        """
        # Ignore comparisons to NaN
        with np.errstate(invalid="ignore"):
            return np.all((uv >= 0) & (uv <= self.imgsz), axis=1)

    def inview(self, xyz: np.ndarray, directions: bool = False) -> np.ndarray:
        """
        Test whether world coordinates are within view of the camera.

        Arguments:
            xyz: World coordinates (n, [x, y, z]).
            directions: Whether `xyz` are ray directions (True)
                or absolute coordinates (False).

        Returns:
            Boolean mask (n, ).
        """
        uv = self.project(xyz, directions=directions)
        return self.inframe(uv)

    def grid(
        self, step: Vector = 1, snap: Sequence[float] = (0.5, 0.5), mode: str = "points"
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return grid of image coordinates.

        Arguments:
            step: Grid spacing for all (float) or each (iterable) dimension.
            snap: Point (x, y) to align grid to.
            mode: Return format:
                - 'vectors' (tuple of np.ndarray): x (nx, ) and y (ny, ) coordinates
                - 'grids' (tuple of np.ndarray): x (ny, nx) and y (ny, nx) coordinates
                - 'points' (np.ndarray): x, y coordinates (ny * nx, [x, y])

        Raises:
            ValueError: Unsupported mode.

        Example:
            >>> cam = Camera(imgsz=3, f=1)
            >>> cam.grid()
            array([[0.5, 0.5],
                   [1.5, 0.5],
                   [2.5, 0.5],
                   [0.5, 1.5],
                   [1.5, 1.5],
                   [2.5, 1.5],
                   [0.5, 2.5],
                   [1.5, 2.5],
                   [2.5, 2.5]])
            >>> cam.grid(mode='vectors')
            (array([0.5, 1.5, 2.5]), array([0.5, 1.5, 2.5]))
            >>> cam.grid(mode='grids')
            (array([[0.5, 1.5, 2.5],
                    [0.5, 1.5, 2.5],
                    [0.5, 1.5, 2.5]]),
             array([[0.5, 0.5, 0.5],
                    [1.5, 1.5, 1.5],
                    [2.5, 2.5, 2.5]]))
            >>> cam.grid(mode='unknown')
            Traceback (most recent call last):
              ...
            ValueError: Unsupported mode: unknown
        """
        box = (0, 0, self.imgsz[0], self.imgsz[1])
        vectors = helpers.box_to_grid(box, step=step, snap=snap, mode="vectors")
        if mode == "vectors":
            return vectors
        grid = np.meshgrid(*vectors)
        if mode == "grids":
            return tuple(grid)
        if mode == "points":
            return helpers.grid_to_points(grid)
        raise ValueError(f"Unsupported mode: {mode}")

    def edges(self, step: Vector = 1) -> np.ndarray:
        """
        Return coordinates of image edges.

        Points are ordered clockwise from the origin (0, 0).

        Arguments:
            step: Point spacing for all (float) or each (iterable) dimension.

        Returns:
            Image coordinates (n, 2).

        Example:
            >>> cam = Camera(imgsz=2, f=1)
            >>> cam.edges()
            array([[0., 0.],
                   [1., 0.],
                   [2., 0.],
                   [2., 1.],
                   [2., 2.],
                   [1., 2.],
                   [0., 2.],
                   [0., 1.]])
        """
        if isinstance(step, (int, float)):
            step = (step, step)
        nu = self.imgsz[0] / step[0] + 1
        nv = self.imgsz[1] / step[1] + 1
        u = np.linspace(0, self.imgsz[0], int(nu))
        v = np.linspace(0, self.imgsz[1], int(nv))
        return np.vstack(
            (
                np.column_stack((u, np.repeat(0, len(u)))),
                np.column_stack((np.repeat(u[-1], len(v) - 2), v[1:-1])),
                np.column_stack((u[::-1], np.repeat(v[-1], len(u)))),
                np.column_stack((np.repeat(0, len(v) - 2), v[::-1][1:-1])),
            )
        )

    def viewbox(self, depth: Number) -> np.ndarray:
        """
        Return bounding box of the camera viewshed.

        The camera viewshed is built by projecting out edge pixels to a fixed depth.

        Arguments:
            depth: Distance of point projections along the camera's optical axis.

        Returns:
            Bounding box (min x, min y, min z, max x, max y, max z).

        Example:
            >>> cam = Camera(imgsz=3, f=3)
            >>> cam.viewbox(depth=1)
            array([-0.5, 0. , -0.5, 0.5, 1. , 0.5])
            >>> cam.viewbox(depth=2)
            array([-1., 0., -1., 1., 2., 1.])
        """
        uv = self.edges()
        dxyz = self.invproject(uv, depth=depth, directions=False)
        vertices = np.vstack((self.xyz, dxyz))
        return helpers.bounding_box(vertices)

    def viewpoly(self, depth: Number) -> np.ndarray:
        """
        Return bounding polygon of the camera viewshed.

        The polygon is built by projecting out the edges of the row passing
        through the principal point and appending the camera position.

        Arguments:
            depth: Distance of point projections along the camera's optical axis.

        Returns:
            Bounding polygon (nx, [x, y, z]).

        Example:
            >>> cam = Camera(imgsz=100, f=100)
            >>> cam.viewpoly(depth=2)
            array([[ 0.,  0.,  0.],
                   [-1.,  2.,  0.],
                   [ 1.,  2.,  0.],
                   [ 0.,  0.,  0.]])
            >>> cam.viewdir = (90, 0, 0)
            >>> cam.viewpoly(depth=2)
            array([[ 0.,  0.,  0.],
                   [ 2.,  1.,  0.],
                   [ 2., -1.,  0.],
                   [ 0.,  0.,  0.]])
        """
        cy = self.imgsz[1] / 2 + self.c[1]
        uv = np.array([(0, cy), (self.imgsz[0], cy)])
        xyz = self.invproject(uv, directions=False, depth=depth)
        return np.row_stack([self.xyz, xyz, self.xyz])

    def rasterize(
        self,
        uv: np.ndarray,
        values: np.ndarray,
        fun: Callable[[np.ndarray], np.ndarray] = np.mean,
    ) -> np.ndarray:
        """
        Convert points to a raster image.

        Arguments:
            uv: Image point coordinates (n, [u, v]).
            values: Point values (n, ).
            fun: Aggregate function to apply to the values of the points in each pixel.

        Returns:
            Image of aggregated values of the same dimensions as :attr:`imgsz` (ny, nx).
                Pixels without points are NaN.

        Example:
            >>> cam = Camera(imgsz=(3, 2), f=1)
            >>> uv = np.array([(0.5, 0.5), (2.5, 1.5), (2.5, 1.5)])
            >>> values = np.array([1, 2, 4])
            >>> cam.rasterize(uv=uv, values=values, fun=np.mean)
            array([[ 1., nan, nan],
                   [nan, nan,  3.]])
        """
        inframe = self.inframe(uv)
        return helpers.rasterize_points(
            # astype(int) equivalent to floor()
            uv[inframe, 1].astype(int),
            uv[inframe, 0].astype(int),
            values[inframe],
            shape=self.shape,
            fun=fun,
        )

    def spherical_to_xyz(self, angles: np.ndarray) -> np.ndarray:
        """
        Convert world spherical coordinates to euclidean.

        Arguments:
            angles: Spherical coordinates (n, [azimuth, altitude(, distance)]).

                - azimuth: Degrees clockwise from north (+y axis).
                - altitude: Degrees above horizon (xy-plane).
                - distance (optional): Distance from camera.

        Returns:
            World coordinates (n, [x, y, z]),
                either absolute (if distances were provided)
                or relative (if distances were not).

        Example:
            >>> cam = Camera(imgsz=1, f=1, xyz=(0, 0, 0))
            >>> angles = np.array([(0, 0, 1), (90, 0, 2), (0, 45, 3)])
            >>> xyz = cam.spherical_to_xyz(angles)
            >>> np.round(xyz, 1)
            array([[0. , 1. , 0. ],
                   [2. , 0. , 0. ],
                   [0. , 2.1, 2.1]])
            >>> cam.xyz = (1, 2, 3)
            >>> np.all(cam.spherical_to_xyz(angles) == xyz + cam.xyz)
            True
        """
        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        azimuth_iso = (np.pi / 2 - angles[:, 0] * np.pi / 180) % (2 * np.pi)
        altitude_iso = (np.pi / 2 - angles[:, 1] * np.pi / 180) % (2 * np.pi)
        xyz = np.column_stack(
            (
                np.sin(altitude_iso) * np.cos(azimuth_iso),
                np.sin(altitude_iso) * np.sin(azimuth_iso),
                np.cos(altitude_iso),
            )
        )
        if angles.shape[1] > 2:
            xyz *= angles[:, 2:3]
            xyz += self.xyz
        return xyz

    def xyz_to_spherical(self, xyz: np.ndarray, directions: bool = False) -> np.ndarray:
        """
        Convert world coordinates to spherical coordinates.

        Arguments:
            xyz: World coordinates (n, [x, y, z]).
            directions: Whether `xyz` are absolute coordinates (False)
                or ray directions (True).

        Returns:
            Spherical coordinates (n, [azimuth, altitude(, distance)]).
                - azimuth: Degrees clockwise from north (+y).
                - altitude: Degrees above horizon (xy-plane).
                - distance: Distance from camera (only if `directions` is False).

        Example:
            >>> cam = Camera(imgsz=1, f=1, xyz=(1, 2, 3))
            >>> angles = np.array([(0, 0, 1), (90, 0, 2), (0, 45, 3)])
            >>> xyz = cam.spherical_to_xyz(angles)
            >>> angles2 = cam.xyz_to_spherical(xyz)
            >>> np.all(np.isclose(angles, angles2))
            True
        """
        if not directions:
            xyz = xyz - self.xyz
        r = np.sqrt(np.sum(xyz ** 2, axis=1))
        azimuth_iso = np.arctan2(xyz[:, 1], xyz[:, 0])
        altitude_iso = np.arccos(xyz[:, 2] / r)
        angles = np.column_stack(
            (
                (90 - (azimuth_iso * 180 / np.pi)) % 360,
                90 - (altitude_iso * 180 / np.pi),
            )
        )
        if not directions:
            angles = np.column_stack((angles, r))
        return angles

    def project_dem(
        self,
        dem: Raster,
        values: np.ndarray = None,
        mask: np.ndarray = None,
        tile_size: Sequence[int] = (256, 256),
        tile_overlap: Sequence[int] = (1, 1),
        scale: Number = 1,
        scale_limits: Sequence[Number] = (1, 1),
        aggregate: Callable[[np.ndarray], np.ndarray] = np.mean,
        parallel: Union[bool, int] = False,
        correction: Union[bool, dict] = False,
        return_depth: bool = False,
    ) -> np.ndarray:
        """
        Return an image simulated from a digital elevation model.

        If `parallel` is True and inputs are large, ensure that `dem`, `values`,
        and `mask` are in shared memory (see `sharedmem.copy()`).

        Arguments:
            dem: `Raster` object containing elevations.
            values: Values to use in building the image.
                Must have the same 2-dimensional shape as `dem.Z` but can have
                multiple layers stacked along the 3rd dimension.
                Cannot be `None` unless `return_depth` is True.
            mask: Boolean mask of cells of `dem` to include.
                Must have the same shape as `dem.Z`.
                If `None`, only NaN cells in `dem.Z` are skipped.
            tile_size: Target size of `dem` tiles.
            tile_overlap: Overlap between `dem` tiles.
            scale: Target `dem` cells per image pixel.
                Each tile is rescaled based on the average distance from the camera.
            scale_limits: Min and max values of `scale`.
            aggregate: Passed as `func` to `pandas.DataFrame.aggregate()`
                to aggregate values projected onto the same image pixel.
                Each layer of `values`, and depth if `return_depth` is True,
                are named by their integer position in the stack (e.g. 0, 1, ...).
            parallel: Number of parallel processes (int),
                or whether to work in parallel (bool). If `True`,
                defaults to `os.cpu_count()`.
            correction: Whether or how to apply elevation corrections
                (see `helpers.elevation_corrections()`).
            return_depth: Whether to return a depth map - the distance of the
                `dem` surface measured along the camera's optical axis.

        Returns:
            Array with 2-dimensional shape (`self.imgsz[1]`, `self.imgsz[0]`)
                and 3rd dimension corresponding to each layer in `values`.
                If `return_depth` is True, it is appended as an additional layer.

        Raises:
            ValueError: `values` does not have same 2-d shape as `dem`.
            ValueError: `mask` does not have the same 2-d shape as `dem.
            ValueError: `values` is missing and `return_depth` is false.

        Example:
            For simplicity, assume the camera is looking straight down at the ground.
            The raster is positioned such that each cell of its 3 x 3 grid is projected
            into its own cell of the 3 x 3 image. In this situation, the values of the
            image are equal to the values of the raster, and the depth of each raster
            cell is equal to the elevation of the camera minus the elevation of the
            cell.

            >>> cam = Camera(imgsz=3, f=3, xyz=(0, 0, 3), viewdir=(0, -90, 0))
            >>> Z = np.array([(0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)])
            >>> values = np.random.randn(*cam.shape)
            >>> dem = Raster(Z, x=(-1, 0, 1), y=(1, 0, -1))
            >>> img = cam.project_dem(dem, values=values, return_depth=True)
            >>> np.all(img[:, :, 0] == values)
            True
            >>> np.all(img[:, :, 1] == cam.xyz[2] - Z)
            True
        """
        has_values = False
        if values is not None:
            has_values = True
            values = np.atleast_3d(values)
            if values.shape[0:2] != dem.shape:
                raise ValueError("values does not have the same 2-d shape as dem")
        elif not return_depth:
            raise ValueError("values cannot be missing if return_depth is False")
        if mask is None:
            mask = ~np.isnan(dem.Z)
        if mask.shape != dem.shape:
            raise ValueError("mask does not have the same 2-d shape as dem")
        parallel = helpers._parse_parallel(parallel)
        # Generate DEM block indices
        tile_indices = dem.tile_indices(size=tile_size, overlap=tile_overlap)
        ntiles = len(tile_indices)
        # Initialize array
        # HACK: Use dummy DataFrame to predict output size of aggregate
        nbands_in = (values.shape[2] if has_values else 0) + return_depth
        df = pandas.DataFrame(
            data=np.zeros((2, nbands_in + 2)),
            columns=["row", "col"] + [str(x) for x in range(nbands_in)],
        )
        nbands_out = df.groupby(["row", "col"]).aggregate(aggregate).shape[1]
        I = np.full(self.shape + (nbands_out,), np.nan)
        # Define parallel process
        bar = helpers._progress_bar(max=ntiles)

        def process(
            ij: Tuple[slice, slice]
        ) -> Tuple[Tuple[Sequence[int], Sequence[int]], Sequence[Number]]:
            tile_mask = mask[ij]
            if not np.count_nonzero(tile_mask):
                # No cells selected
                return None
            tile = dem[ij]
            if has_values:
                tile_values = values[ij]
            # Scale tile based on distance from camera
            mean_xyz = tile.xlim.mean(), tile.ylim.mean(), np.nanmean(tile.Z[tile_mask])
            if np.isnan(mean_xyz[2]):
                # No cells with elevations
                return None
            _, mean_depth = self._world2camera(
                np.atleast_2d(mean_xyz), return_depth=True, correction=correction
            )
            tile_scale = scale * np.abs(tile.d).mean() / (mean_depth / self.f.mean())
            tile_scale = min(max(tile_scale, min(scale_limits)), max(scale_limits))
            if tile_scale != 1:
                tile.resize(tile_scale)
                tile_mask = scipy.ndimage.zoom(
                    tile_mask, zoom=float(tile_scale), order=0
                )
                tile_values = np.dstack(
                    scipy.ndimage.zoom(
                        tile_values[:, :, i], zoom=float(tile_scale), order=1
                    )
                    for i in range(tile_values.shape[2])
                )
            # Project tile
            xyz = helpers.grid_to_points(
                (tile.X[tile_mask], tile.Y[tile_mask], tile.Z[tile_mask])
            )
            if return_depth:
                xy, depth = self._world2camera(
                    xyz, correction=correction, return_depth=True
                )
                uv = self._camera2image(xy)
            else:
                uv = self.project(xyz, correction=correction)
            is_in = self.inframe(uv)
            if not np.count_nonzero(is_in):
                # No cells in image
                return None
            rc = uv[is_in, ::-1].astype(int)
            # Compile values
            if has_values:
                tile_values = tile_values[tile_mask][is_in]
            if return_depth:
                if has_values:
                    tile_values = np.column_stack((tile_values, depth[is_in, None]))
                else:
                    tile_values = depth[is_in, None]
            # Build DataFrame for fast groupby operation
            df = pandas.DataFrame({"row": rc[:, 0], "col": rc[:, 1]})
            for i in range(tile_values.shape[1]):
                df.insert(df.shape[1], i, tile_values[:, i])
            # Aggregate values
            groups = df.groupby(["row", "col"], sort=False, as_index=False).aggregate(
                aggregate
            )
            idx = (groups["row"].values.astype(int), groups["col"].values.astype(int))
            return idx, groups.iloc[:, 2:].values

        def reduce(
            idx: Tuple[Sequence[int], Sequence[int]] = None,
            values: Sequence[Number] = None,
        ) -> None:
            bar.next()
            if idx is not None:
                I[idx] = values

        with config._MapReduce(np=parallel) as pool:
            pool.map(func=process, reduce=reduce, sequence=tile_indices)
        bar.finish()
        return I

    # ---- Methods (private) ----

    def _radial_distortion(self, r2: np.ndarray) -> np.ndarray:
        """
        Compute the radial distortion multiplier `dr`.

        Arguments:
            r2: Squared radius of camera coordinates (n, 1)
        """
        # dr = (1 + k1 * r^2 + k2 * r^4 + k3 * r^6)/(1 + k4 * r^2 + k5 * r^4 + k6 * r^6)
        dr: np.ndarray = 1
        if self.k[0]:
            dr += self.k[0] * r2
        if self.k[1]:
            dr += self.k[1] * r2 * r2
        if self.k[2]:
            dr += self.k[2] * r2 * r2 * r2
        if any(self.k[3:6]):
            temp = 1
            if self.k[3]:
                temp += self.k[3] * r2
            if self.k[4]:
                temp += self.k[4] * r2 * r2
            if self.k[5]:
                temp += self.k[5] * r2 * r2 * r2
            dr /= temp
        # Return as column
        return dr[:, None]

    def _tangential_distortion(self, xy: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Compute tangential distortion additive `[dtx, dty]`.

        Arguments:
            xy: Camera coordinates (n, 2)
            r2: Squared radius of `xy` (n, 1)
        """
        # dtx = 2xy * p1 + p2 * (r^2 + 2x^2)
        # dty = p1 * (r^2 + 2y^2) + 2xy * p2
        xty = xy[:, 0] * xy[:, 1]
        dtx = 2 * xty * self.p[0] + self.p[1] * (r2 + 2 * xy[:, 0] ** 2)
        dty = self.p[0] * (r2 + 2 * xy[:, 1] ** 2) + 2 * xty * self.p[1]
        return np.column_stack((dtx, dty))

    def _distort(self, xy: np.ndarray) -> np.ndarray:
        """
        Apply distortion to camera coordinates.

        Arguments:
            xy: Camera coordinates (n, 2)
        """
        # X' = dr * X + dt
        if not any(self.k) and not any(self.p):
            return xy
        dxy = xy.copy()
        r2 = np.sum(xy ** 2, axis=1)
        if any(self.k):
            dxy *= self._radial_distortion(r2)
        if any(self.p):
            dxy += self._tangential_distortion(xy, r2)
        return dxy

    def _undistort(
        self, xy: np.ndarray, method: str = "oulu", **kwargs: Any
    ) -> np.ndarray:
        """
        Remove distortion from camera coordinates.

        TODO: Quadtree 2-D bisection
        https://stackoverflow.com/questions/3513660/multivariate-bisection-method.

        Arguments:
            xy: Camera coordinates (n, 2).
            method: Undistort method to use if undistorted coordinates must be estimated
                numerically ("lookup", "oulu", or "regulafalsi").
            **kwargs: Optional arguments to the undistort method.

        Returns:
            Undistorted camera coordinates (n, 2).

        Raises:
            ValueError: Undistort method not supported.
        """
        # X = (X' - dt) / dr
        if not any(self.k) and not any(self.p):
            return xy
        if self.k[0] and not any(self.k[1:]) and not any(self.p):
            return self._undistort_k1(xy)
        if method == "lookup":
            return self._undistort_lookup(xy, **kwargs)
        if method == "oulu":
            return self._undistort_oulu(xy, **kwargs)
        if method == "regulafalsi":
            return self._undistort_regulafalsi(xy, **kwargs)
        raise ValueError(f"Undistort method not supported: {method}")

    def _undistort_k1(self, xy: np.ndarray) -> np.ndarray:
        """
        Remove 1st order radial distortion.

        Uses the closed-form solution to the cubic equation when
        the only non-zero distortion coefficient is k1 (self.k[0]).

        Arguments:
            xy: Camera coordinates (n, 2)
        """
        # Cubic roots solution from Numerical Recipes in C 2nd Edition:
        # http://apps.nrbook.com/c/index.html (pages 183-185)
        # Solve for undistorted radius in polar coordinates
        # r^3 + r / k1 - r'/k1 = 0
        phi = np.arctan2(xy[:, 1], xy[:, 0])
        Q = -1 / (3 * self.k[0])
        # r' = y / cos(phi)
        R = -xy[:, 0] / (2 * self.k[0] * np.cos(phi))
        has_three_roots = R ** 2 < Q ** 3
        r = np.full(len(xy), np.nan)
        if np.any(has_three_roots):
            th = np.arccos(R[has_three_roots] * Q ** -1.5)
            r[has_three_roots] = -2 * np.sqrt(Q) * np.cos((th - 2 * np.pi) / 3)
        has_one_root = ~has_three_roots
        if np.any(has_one_root):
            A = -np.sign(R[has_one_root]) * (
                np.abs(R[has_one_root]) + np.sqrt(R[has_one_root] ** 2 - Q ** 3)
            ) ** (1.0 / 3)
            B = np.zeros(A.shape)
            not_zero = A != 0
            B[not_zero] = Q / A[not_zero]
            r[has_one_root] = A + B
        return np.column_stack((np.cos(phi), np.sin(phi))) * r[:, None]

    def _undistort_lookup(self, xy: np.ndarray, density: Number = 1) -> np.ndarray:
        """
        Remove distortion by table lookup.

        Creates a grid of test coordinates and applies distortion,
        then interpolates undistorted coordinates from the result
        with scipy.interpolate.LinearNDInterpolator().

        NOTE: Remains stable in extreme distortion, but slow for large lookup tables.

        Arguments:
            xy: Camera coordinates (n, 2)
            density: Grid points per pixel (approximate)
        """
        # Estimate undistorted camera coordinate bounds
        uv_edges = self.imgsz * np.array(
            [[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 1], [0, 1], [0, 0.5]]
        )
        xyu_edges = (uv_edges - (self.imgsz / 2 + self.c)) / self.f
        xyd_edges = self._distort(xyu_edges)
        # Build undistorted camera coordinates on regular grid
        ux = np.linspace(
            min(xyu_edges[:, 0].min(), xyd_edges[:, 0].min()),
            max(xyu_edges[:, 0].max(), xyd_edges[:, 0].max()),
            int(density * self.imgsz[0]),
        )
        uy = np.linspace(
            min(xyu_edges[:, 1].min(), xyd_edges[:, 1].min()),
            max(xyu_edges[:, 1].max(), xyd_edges[:, 1].max()),
            int(density * self.imgsz[1]),
        )
        UX, UY = np.meshgrid(ux, uy)
        uxy = np.column_stack((UX.flatten(), UY.flatten()))
        # Distort grid
        dxy = self._distort(uxy)
        # Interpolate distortion removal from gridded results
        # NOTE: Cannot use faster grid interpolation because dxy is not regular
        return scipy.interpolate.griddata(dxy, uxy, xy, method="linear")

    def _undistort_oulu(
        self, xy: np.ndarray, iterations: int = 20, tolerance: Number = 0
    ) -> np.ndarray:
        """
        Remove distortion by the iterative Oulu University method.

        See http://www.vision.caltech.edu/bouguetj/calib_doc/ (comp_distortion_oulu.m)

        NOTE: Converges very quickly in normal cases, but fails for extreme distortion.

        Arguments:
            xy: Camera coordinates (n, 2)
            iterations: Maximum number of iterations
            tolerance: Approximate pixel displacement in x and y below which
                to exit early, or `0` to disable early exit
        """
        # Initial guess
        uxy = xy
        for _ in range(iterations):
            r2 = np.sum(uxy ** 2, axis=1)
            if any(self.p) and not any(self.k):
                uxy = xy - self._tangential_distortion(uxy, r2)
            elif any(self.k) and not any(self.k):
                uxy = xy * (1 / self._radial_distortion(r2))
            else:
                uxy = (xy - self._tangential_distortion(uxy, r2)) * (
                    1 / self._radial_distortion(r2)
                )
            if tolerance > 0 and np.all(
                (np.abs(self._distort(uxy) - xy)) < tolerance / self.f.mean()
            ):
                break
        return uxy

    def _undistort_regulafalsi(
        self, xy: np.ndarray, iterations: int = 100, tolerance: Number = 0
    ) -> np.ndarray:
        """
        Remove distortion by iterative regula falsi (false position) method.

        See https://en.wikipedia.org/wiki/False_position_method

        NOTE: Almost always converges, but may require many iterations for extreme
        distortion.

        Arguments:
            xy: Camera coordinates (n, 2)
            iterations: Maximum number of iterations
            tolerance: Approximate pixel displacement in x and y
                (for all points) below which to exit early,
                or `0` to disable early exit (default).
        """
        # Start at center of image (distortion free)
        x1 = np.zeros(xy.shape, dtype=float)
        y1 = -xy
        # Then try halfway towards distorted coordinate
        # (more stable to approach solution from image center)
        x2 = xy / 2
        y2 = self._distort(x2) - xy
        uxy = np.full(xy.shape, np.nan)
        for n in range(iterations):
            dy = y2 - y1
            not_converged = np.all(dy != 0, axis=1)
            if tolerance > 0:
                not_converged &= np.any(np.abs(y2) > tolerance / self.f.mean())
            if n == 0:
                mask = np.ones(len(xy), dtype=bool)
            converged = np.zeros(mask.shape, dtype=bool)
            converged[mask] = ~not_converged
            uxy[converged] = x2[~not_converged]
            mask[mask] = not_converged
            x1 = x1[not_converged]
            y1 = y1[not_converged]
            x2 = x2[not_converged]
            y2 = y2[not_converged]
            if not np.any(not_converged):
                break
            x3 = (x1 * y2 - x2 * y1) / dy[not_converged]
            y3 = self._distort(x3) - xy[mask]
            x1 = x2
            y1 = y2
            x2 = x3
            y2 = y3
        uxy[mask] = x2
        return uxy

    def _reversible(self) -> bool:
        """
        Test whether the camera model is reversible.

        Checks whether distortion produces a monotonically increasing result.
        If not, distorted coordinates are non-unique and cannot be reversed.

        TODO: Derive this explicitly from the distortion parameters.
        """
        xy_row = np.column_stack(
            (
                np.linspace(
                    -self.imgsz[0] / (2 * self.f[0]),
                    self.imgsz[0] / (2 * self.f[0]),
                    int(self.imgsz[0]),
                ),
                np.zeros(int(self.imgsz[0])),
            )
        )
        dxy = self._distort(xy_row)
        continuous_row = np.all(dxy[1:, 0] >= dxy[:-1, 0])
        xy_col = np.column_stack(
            (
                np.zeros(int(self.imgsz[1])),
                np.linspace(
                    -self.imgsz[1] / (2 * self.f[1]),
                    self.imgsz[1] / (2 * self.f[1]),
                    int(self.imgsz[1]),
                ),
            )
        )
        dxy = self._distort(xy_col)
        continuous_col = np.all(dxy[1:, 1] >= dxy[:-1, 1])
        return continuous_row and continuous_col

    def _world2camera(
        self,
        xyz: np.ndarray,
        directions: bool = False,
        correction: Union[bool, dict] = False,
        return_depth: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Project world coordinates to camera coordinates.

        Arguments:
            xyz: World coordinates (n, 3)
            directions: Whether `xyz` are absolute coordinates (False)
                or ray directions (True)
            correction: Arguments to `helpers.elevation_corrections()` (dict),
                `True` for default arguments, or `False` to skip.
                Only applies if `directions` is `False`.
            return_depth: Whether to return the distance of each point
                along the camera's optical axis
        """
        if directions:
            dxyz = xyz
        else:
            dxyz = xyz - self.xyz
            if correction is True:
                correction = {}
            if isinstance(correction, dict):
                # Apply elevation correction
                dxyz[:, 2] += helpers.elevation_corrections(
                    squared_distances=np.sum(dxyz[:, 0:2] ** 2, axis=1), **correction
                )
        # Convert coordinates to ray directions
        if config._UseMatMul:
            xyz_c = np.matmul(self.R, dxyz.T).T
        else:
            xyz_c = np.dot(dxyz, self.R.T)
        # Normalize by perspective division
        # Ignore divide by zero
        with np.errstate(invalid="ignore"):
            xy = xyz_c[:, 0:2] / xyz_c[:, 2:3]
        # Set points behind camera to NaN
        behind = xyz_c[:, 2] <= 0
        xy[behind] = np.nan
        if return_depth:
            return xy, xyz_c[:, 2]
        return xy

    def _camera2world(
        self, xy: np.ndarray, directions: bool = True, depth: Vector = 1
    ) -> np.ndarray:
        """
        Project camera coordinates to world coordinates or ray directions.

        Arguments:
            xy: Camera coordinates (n, 2)
            directions: Whether to return world ray directions relative
                to the camera position (True) or absolute coordinates by adding
                on the camera position (False)
            depth: Distance of rays along the camera's optical axis, as either a
                scalar or a vector (n, )
        """
        # Multiply 2-d coordinates
        if config._UseMatMul:
            xyz = np.matmul(self.R.T[:, 0:2], xy.T).T
        else:
            xyz = np.dot(xy, self.R[0:2, :])
        # Simulate z = 1 by adding 3rd column of rotation matrix
        xyz += self.R.T[:, 2]
        if depth != 1:
            xyz *= np.atleast_1d(depth).reshape(-1, 1)
        if not directions:
            xyz += self.xyz
        return xyz

    def _camera2image(self, xy: np.ndarray) -> np.ndarray:
        """
        Project camera to image coordinates.

        Arguments:
            xy: Camera coordinates (n, 2)
        """
        xy = self._distort(xy)
        uv = xy * self.f + (self.imgsz / 2 + self.c)
        return uv

    def _image2camera(self, uv: np.ndarray) -> np.ndarray:
        """
        Project image to camera coordinates.

        Arguments:
            uv: Image coordinates (n, 2)
        """
        xy = (uv - (self.imgsz * 0.5 + self.c)) * (1 / self.f)
        xy = self._undistort(xy)
        return xy

    def _image2camera_grid_ideal(self, uv: Sequence[np.ndarray]) -> np.ndarray:
        """
        Project image to camera coordinates.

        Faster version for an ideal camera and regularly gridded image coordinates.

        Arguments:
            uv: Vectors (u, v) of regularly gridded image coordinates
        """
        x = (uv[0] - (self.imgsz[0] * 0.5 + self.c[0])) * (1 / self.f[0])
        y = (uv[1] - (self.imgsz[1] * 0.5 + self.c[1])) * (1 / self.f[1])
        return helpers.grid_to_points(np.meshgrid(x, y))
