"""Optimize camera models to fit observations taken from images and the world."""
import datetime
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import warnings

import cv2
import lmfit
import matplotlib.pyplot
import numpy as np
import scipy.optimize
import scipy.sparse

from . import config, helpers
from .camera import Camera

Index = Union[slice, Sequence[int]]
CamIndex = Union[int, Camera]
Color = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]
ColorArgs = Optional[Union[dict, Color]]

# ---- Controls ----

# Controls (within Cameras) support RANSAC with the following API:
# .size
# .observed(index)
# .predicted(index)


class Points:
    """
    Image-world point correspondences.

    World coordinates (`xyz`) are projected to image coordinates,
    then compared to their expected image coordinates (`uv`).

    Attributes:
        cam (Camera): Camera.
        uv (array): Image coordinates (n, [u, v]).
        xyz (array): World coordinates (n, [x, y, z]).
        directions (bool): Whether `xyz` are absolute coordinates (False)
            or ray directions (True). If the latter, camera position cannot change.
        size (int): Number of point pairs (n).

    Raises:
        ValueError: Image and world coordinates have different length.

    Example:
        Say we have a camera looking down onto three world points on the xy-plane.

        >>> cam = Camera(imgsz=10, f=1, xyz=(0, 0, 1), viewdir=(0, -90, 0))
        >>> xyz = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
        >>> uv = [(3, 5), (5, 5), (7, 5)]
        >>> points = Points(cam=cam, uv=uv, xyz=xyz)
        >>> points.size
        3

        The image coordinates predicted from the world coordinates and the camera model
        do not match the observed image coordinates.

        >>> points.predicted() - points.observed()
        array([[ 1., 0.],
               [ 0., 0.],
               [-1., 0.]])

        Plotting the reprojection errors can help diagnose the problem.

        >>> import matplotlib.pyplot as plt
        >>> points.plot()
        {'unselected': None, 'selected': <matplotlib.quiver.Quiver ...>}
        >>> plt.show()  # doctest: +SKIP
        >>> plt.close()

        In this case, the errors are symmetric and increasing from the image center,
        suggesting that the camera focal length needs to be adjusted.

        >>> cam.f = 2
        >>> points.predicted() - points.observed()
        array([[0., 0.],
               [0., 0.],
               [0., 0.]])
    """

    def __init__(
        self, cam: Camera, uv: np.ndarray, xyz: np.ndarray, directions: bool = False,
    ) -> None:
        if len(uv) != len(xyz):
            raise ValueError("Image and world coordinates have different length")
        self.cam = cam
        self.uv = np.asarray(uv, dtype=float)
        self.xyz = np.asarray(xyz, dtype=float)
        self.directions = directions
        self._position = cam.xyz.copy()
        self._imgsz = cam.imgsz.copy()

    @property
    def size(self) -> int:
        """Number of points pairs."""
        return len(self.uv)

    def observed(self, index: Index = slice(None)) -> np.ndarray:
        """
        Return observed image coordinates.

        Arguments:
            index: Indices of points to return.
        """
        return self.uv[index]

    def _test_position(self) -> None:
        if self.directions and any(self.cam.xyz != self._position):
            raise ValueError(
                "Camera position has changed and world coordinates are ray directions"
            )

    def predicted(self, index: Index = slice(None)) -> np.ndarray:
        """
        Predict image coordinates from world coordinates.

        Arguments:
            index: Indices of world points to project.

        Raises:
            ValueError: Camera position has changed and world coordinates are
                ray directions.
        """
        self._test_position()
        return self.cam.xyz_to_uv(self.xyz[index], directions=self.directions)

    def plot(
        self,
        index: Index = slice(None),
        selected: ColorArgs = "red",
        unselected: ColorArgs = "gray",
        **kwargs: Any
    ) -> Dict[str, Optional[matplotlib.quiver.Quiver]]:
        """
        Plot reprojection errors as quivers.

        Arrows point from observed to predicted image coordinates.

        Arguments:
            index: Indices of points to select.
            selected: For selected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            unselected: For unselected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            **kwargs: Optional arguments to matplotlib.pyplot.quiver for all points.
        """
        new_plot = not matplotlib.pyplot.get_fignums()
        defaults = {
            "scale": 1,
            "scale_units": "xy",
            "angles": "xy",
            "units": "xy",
            "width": self.cam.imgsz[0] * 0.005,
            **kwargs,
        }
        uv = self.observed()
        duv = self.predicted() - uv
        full = np.arange(self.size)
        index, unindex = full[index], np.delete(full, index)
        # Plot selected points on top
        result: Dict[str, Optional[matplotlib.quiver.Quiver]] = {}
        for idx, args, label in [
            (unindex, unselected, "unselected"),
            (index, selected, "selected"),
        ]:
            if not len(idx) or args is None:
                result[label] = None
                continue
            if not isinstance(args, dict):
                args = {"color": args}
            args = {**defaults, **args}
            result[label] = matplotlib.pyplot.quiver(
                uv[idx, 0], uv[idx, 1], duv[idx, 0], duv[idx, 1], **args
            )
        if new_plot:
            self.cam.set_plot_limits()
        return result

    def _scale(self, scale: np.ndarray) -> None:
        if np.any(scale != 1):
            self.uv = self.uv * scale

    def resize(
        self, size: Union[float, Sequence[int]] = None, force: bool = False
    ) -> None:
        """
        Resize to new image size.

        Resizes both the camera and image coordinates.

        Arguments:
            size: Scale factor relative to the camera's original size (float)
                or target image size in pixels (nx, ny).
                If `None`, image coordinates are resized to fit the current
                camera image size.
            force: Whether to use `size` even if it does not preserve
                the original aspect ratio.

        Example:
            >>> cam = Camera(imgsz=10, f=1)
            >>> xyz = [(0, 1, 0)]
            >>> uv = [(5, 5)]
            >>> points = Points(cam=cam, uv=uv, xyz=xyz)
            >>> points.resize(0.5)
            >>> cam.imgsz
            array([5, 5])
            >>> points.uv
            array([[2.5, 2.5]])
            >>> cam.resize(1)
            >>> points.resize()
            >>> points.uv
            array([[5., 5.]])
        """
        if size is not None:
            self.cam.resize(size=size, force=force)
        self._scale(self.cam.imgsz / self._imgsz)
        self._imgsz = self.cam.imgsz.copy()


class Lines(Points):
    """
    Image-world line correspondences.

    Image polylines (`uvs`) are reduced to a single array of image points (`uv`).
    World polylines (`xyzs`) are projected onto the image with a target pixel density,
    and each image point is matched to the nearest projected world point.

    Since each image point is matched to a projected world point
    (but not every projected world point is necessarily matched to an image point),
    the image lines should be a subset of the world lines.
    The opposite will not yield correct results.

    Attributes:
        cam (Camera): Camera.
        uvs (list of array): Image line vertices [(ni, [u, v]), ...].
        xyzs (list of array): World line vertices [(mi, [x, y, z]), ...].
        directions (bool): Whether `xyzs` are absolute coordinates (False)
            or ray directions (True).
        density (float): Target
        uv (array): Merged image line vertices (n, [u, v]).
        size (int): Number of image points (n).

    Example:
        Say we have a camera looking north towards a distant horizon.
        Only a portion of the horizon is traced in the image.

        >>> cam = Camera(imgsz=10, f=1)
        >>> xyzs = [[(-10, 1, 0), (0, 1, 0), (10, 1, 0)]]
        >>> uvs = [[(2, 4), (4, 4)], [(6, 4), (8, 4)]]
        >>> lines = Lines(cam=cam, uvs=uvs, xyzs=xyzs, density=10)
        >>> lines.size
        4

        The image coordinates predicted from the world lines and the camera model
        do not match the observed image coordinates.

        >>> lines.predicted() - lines.observed()
        array([[0., 1.],
               [0., 1.],
               [0., 1.],
               [0., 1.]])

        Plotting the reprojection errors can help diagnose the problem.

        >>> import matplotlib.pyplot as plt
        >>> lines.plot()
        {'observed': [<matplotlib.lines.Line2D ...>, <matplotlib.lines.Line2D ...>],
        'predicted': [<matplotlib.lines.Line2D ...>],
        'unselected': None, 'selected': <matplotlib.quiver.Quiver ...>}
        >>> plt.show()  # doctest: +SKIP
        >>> plt.close()

        The errors all point straight down,
        suggesting that the camera needs to be rotated downward.

        >>> cam.viewdir[1] -= 45
        >>> lines.predicted() - lines.observed()
        array([[0., 0.],
               [0., 0.],
               [0., 0.],
               [0., 0.]])
    """

    def __init__(
        self,
        cam: Camera,
        uvs: Sequence[np.ndarray],
        xyzs: Sequence[np.ndarray],
        directions: bool = False,
        density: float = 1,
    ) -> None:
        self.cam = cam
        self.uvs = [np.asarray(uv, dtype=float) for uv in uvs]
        self.uv = np.row_stack(self.uvs)
        self.xyzs = xyzs
        self.directions = directions
        self.density = density
        self._position = cam.xyz.copy()
        self._imgsz = cam.imgsz.copy()

    def _xyzs_to_uvs(self) -> List[np.ndarray]:
        """
        Project world lines onto the image.

        Returns:
            Arrays of image coordinates [(ni, [u, v]), ...].
        """
        xy_step = (1 / self.density) / self.cam.f.max()
        uv_edges = self.cam.edges(step=self.cam.imgsz / 2)
        xy_edges = self.cam._uv_to_xy(uv_edges)
        xy_box = np.hstack((np.min(xy_edges, axis=0), np.max(xy_edges, axis=0)))
        puvs = []
        inlines = []
        for xyz in self.xyzs:
            # TODO: Instead, clip lines to 3D polar viewbox before projecting
            # Project world lines to camera
            xy = self.cam._xyz_to_xy(xyz, directions=self.directions)
            # Discard nan values (behind camera)
            lines = helpers.boolean_split(xy, np.isnan(xy[:, 0]), include="false")
            for line in lines:
                inlines.append(line)
                # Clip lines in view
                # Resolves coordinate wrap around with large distortion
                for cline in helpers.clip_polyline_box(line, xy_box):
                    # Interpolate clipped lines to target resolution
                    puvs.append(
                        self.cam._xy_to_uv(
                            helpers.interpolate_line(np.array(cline), dx=xy_step)
                        )
                    )
        if puvs:
            return puvs
        # If no lines inframe, project line vertices infront
        return [self.cam._xy_to_uv(line) for line in inlines]

    def predicted(self, index: Index = slice(None)) -> np.ndarray:
        """
        Predict image coordinates from world coordinates.

        Arguments:
            index: Indices of image points to include in nearest-neighbor search.

        Returns:
            Image coordinates (n, [u, v]) on the projected world lines nearest
            the observed image coordinates.

        Raises:
            ValueError: Camera position has changed and world coordinates are
                ray directions.
        """
        self._test_position()
        puv = np.row_stack(self._xyzs_to_uvs())
        distances = helpers.pairwise_distance(self.observed(index=index), puv)
        min_index = np.argmin(distances, axis=1)
        return puv[min_index, :]

    def plot(
        self,
        index: Index = slice(None),
        selected: ColorArgs = "red",
        unselected: ColorArgs = "gray",
        observed: ColorArgs = "green",
        predicted: ColorArgs = "yellow",
        **kwargs: Any
    ) -> Dict[
        str, Optional[Union[matplotlib.quiver.Quiver, List[matplotlib.lines.Line2D]]]
    ]:
        """
        Plot reprojection errors as quivers.

        Arrows point from observed to predicted image coordinates.

        Arguments:
            index: Indices of points to select.
            selected: For selected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            unselected: For unselected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            observed: For image lines, optional arguments to
                matplotlib.pyplot.plot (dict), color, or `None` to hide.
            predicted: For world lines, optional arguments to
                matplotlib.pyplot.plot (dict), color, or `None` to hide.
            **kwargs: Optional arguments to matplotlib.pyplot.quiver for all points.
        """
        new_plot = not matplotlib.pyplot.get_fignums()
        result: Dict[
            str,
            Optional[Union[matplotlib.quiver.Quiver, List[matplotlib.lines.Line2D]]],
        ] = {}
        # Plot lines
        for uvs, args, label in [
            (self.uvs, observed, "observed"),
            (self._xyzs_to_uvs(), predicted, "predicted"),
        ]:
            if args is None:
                result[label] = None
                continue
            if not isinstance(args, dict):
                args = {"color": args}
            # matplotlib.pyplot.plot returns a list even for a single line
            result[label] = [
                matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], **args)[0] for uv in uvs
            ]
        # Plot errors
        defaults = {
            "scale": 1,
            "scale_units": "xy",
            "angles": "xy",
            "units": "xy",
            "width": self.cam.imgsz[0] * 0.005,
            **kwargs,
        }
        uv = self.observed()
        duv = self.predicted() - uv
        full = np.arange(self.size)
        index, unindex = full[index], np.delete(full, index)
        for idx, args, label in [
            (unindex, unselected, "unselected"),
            (index, selected, "selected"),
        ]:
            if not len(idx) or args is None:
                result[label] = None
                continue
            if not isinstance(args, dict):
                args = {"color": args}
            args = {**defaults, **args}
            result[label] = matplotlib.pyplot.quiver(
                uv[idx, 0], uv[idx, 1], duv[idx, 0], duv[idx, 1], **args
            )
        if new_plot:
            self.cam.set_plot_limits()
        return result

    def _scale(self, scale: np.ndarray) -> None:
        if np.any(scale != 1):
            for i, uv in enumerate(self.uvs):
                self.uvs[i] = uv * scale
            self.uv *= scale


class Matches:
    """
    Image-image point correspondences.

    The image coordinates (`uvs[i]`) of one camera (`cams[i]`) are projected into the
    other camera (`cams[j]`), then compared to the expected image coordinates for that
    camera (`uvs[j]`).

    Since the world coordinates of the points are not known,
    both cameras must have the same position.

    Attributes:
        cams (list of Camera): Pair of cameras.
        uvs (list of array): Image point coordinates for each camera
            [(n, [ui, vi]), (n, [uj, vj])].
        weights (array): Relative weight of each point pair (n, ).
        size (int): Number of point pairs (n).

    Raises:
        ValueError: Both cameras are the same object.
        ValueError: Cameras and point coordinates do not have two elements each.
        ValueError: Camera point coordinates do not have the same length.
        ValueError: Cameras have different positions.

    Example:
        Say we have two identical cameras with some points matched between them.

        >>> cams = Camera(imgsz=10, f=1), Camera(imgsz=10, f=1)
        >>> uvs = [(4, 5), (5, 5), (6, 5)], [(4.1, 5), (5.1, 5), (6.1, 5)]
        >>> matches = Matches(cams=cams, uvs=uvs)
        >>> matches.size
        3

        The image coordinates predicted for the first camera
        do not match the image coordinates observed by the first camera.

        >>> matches.predicted() - matches.observed()
        array([[0.1, 0. ],
               [0.1, 0. ],
               [0.1, 0. ]])

        Plotting the reprojection errors can help diagnose the problem.

        >>> import matplotlib.pyplot as plt
        >>> matches.plot(scale=0.5)
        {'unselected': None, 'selected': <matplotlib.quiver.Quiver ...>}
        >>> plt.show()  # doctest: +SKIP
        >>> plt.close()

        The errors all point right. If we assume the first camera is fixed,
        this suggests that the second camera needs to be rotated slightly left.
        Since errors are still not zero, it may be possible to further reduce them by
        adjusting other camera parameters.

        >>> cams[1].viewdir[0] = -3
        >>> matches.predicted() - matches.observed()
        array([[ 0.0..., 0. ],
               [ 0.0..., 0. ],
               [-0.0..., 0. ]])
    """

    def __init__(
        self,
        cams: Sequence[Camera],
        uvs: Sequence[np.ndarray],
        weights: np.ndarray = None,
    ) -> None:
        self.cams = cams
        self.uvs = [np.asarray(uv, dtype=float) for uv in uvs]
        self.weights = weights
        self._test_matches()
        self._test_position()
        self._imgszs = [cam.imgsz.copy() for cam in cams]

    @property
    def size(self) -> int:
        """Number of points pairs."""
        return len(self.uvs[0])

    def _test_matches(self) -> None:
        if self.cams[0] is self.cams[1]:
            raise ValueError("Both cameras are the same object")
        # HACK: Support subclasses with xyzs and optional uvs
        uvs = self.uvs or self.xyzs
        if len(self.cams) != 2 or len(uvs) != 2:
            raise ValueError(
                "Cameras and point coordinates do not have two elements each"
            )
        if len(uvs[0]) != len(uvs[1]):
            raise ValueError("Camera point coordinates do not have the same length")

    def _test_position(self) -> None:
        if any(self.cams[0].xyz != self.cams[1].xyz):
            raise ValueError("Cameras have different positions")

    def _cam_index(self, cam: CamIndex) -> int:
        if isinstance(cam, int):
            if cam >= len(self.cams):
                raise IndexError("Camera index out of range")
            return cam
        return self.cams.index(cam)

    def observed(self, cam: CamIndex = 0, index: Index = slice(None)) -> np.ndarray:
        """
        Return observed image coordinates.

        Arguments:
            cam: Camera whose points to return.
            index: Indices of points to return.
        """
        c = self._cam_index(cam)
        return self.uvs[c][index]

    def predicted(self, cam: CamIndex = 0, index: Index = slice(None)) -> np.ndarray:
        """
        Predict image coordinates for a camera from those of the other camera.

        Arguments:
            cam: Camera for which to predict image coordinates.
            index: Indices of points to predict.

        Raises:
            ValueError: Cameras have different positions.
        """
        self._test_position()
        ci = self._cam_index(cam)
        co = 0 if ci else 1
        dxyz = self.cams[co].uv_to_xyz(self.uvs[co][index])
        return self.cams[ci].xyz_to_uv(dxyz, directions=True)

    def plot(
        self,
        cam: CamIndex = 0,
        index: Index = slice(None),
        selected: ColorArgs = "red",
        unselected: ColorArgs = "gray",
        **kwargs: Any
    ) -> Dict[str, matplotlib.quiver.Quiver]:
        """
        Plot reprojection errors as quivers.

        Arrows point from observed to predicted image coordinates.

        Arguments:
            cam: Camera to plot.
            index: Indices of points to select.
            selected: For selected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            unselected: For unselected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            **kwargs: Optional arguments to matplotlib.pyplot.quiver for all points.
        """
        new_plot = not matplotlib.pyplot.get_fignums()
        c = self._cam_index(cam)
        defaults = {
            "scale": 1,
            "scale_units": "xy",
            "angles": "xy",
            "units": "xy",
            "width": self.cams[c].imgsz[0] * 0.005,
            **kwargs,
        }
        uv = self.observed(cam=cam)
        duv = self.predicted(cam=cam) - uv
        full = np.arange(self.size)
        index, unindex = full[index], np.delete(full, index)
        # Plot selected points on top
        result: Dict[str, matplotlib.quiver.Quiver] = {}
        for idx, args, label in [
            (unindex, unselected, "unselected"),
            (index, selected, "selected"),
        ]:
            if not len(idx) or args is None:
                result[label] = None
                continue
            if not isinstance(args, dict):
                args = {"color": args}
            args = {**defaults, **args}
            result[label] = matplotlib.pyplot.quiver(
                uv[idx, 0], uv[idx, 1], duv[idx, 0], duv[idx, 1], **args
            )
        if new_plot:
            self.cams[c].set_plot_limits()
        return result

    def to_type(self, mtype: Type["Matches"]) -> "Matches":
        """Return as matches of a different type."""
        if mtype is type(self):
            return self
        return mtype(cams=self.cams, uvs=self.uvs, weights=self.weights)

    def resize(
        self, size: Union[float, Sequence[int]] = None, force: bool = False
    ) -> None:
        """
        Resize to new image size.

        Resizes both the cameras and their image coordinates.

        Arguments:
            size: Scale factor relative to the cameras' original sizes (float)
                or target image size (iterable).
                If `None`, image coordinates are resized to fit current
                camera image sizes.
            force: Whether to use `size` even if it does not preserve
                the original aspect ratio
        """
        for i, cam in enumerate(self.cams):
            if size is not None:
                cam.resize(size=size, force=force)
            scale = cam.imgsz / self._imgszs[i]
            if np.any(scale != 1):
                self.uvs[i] = self.uvs[i] * scale
                self._imgszs[i] = cam.imgsz.copy()

    def filter(
        self,
        n_best: int = None,
        min_weight: float = None,
        cam: CamIndex = 0,
        max_error: float = None,
        max_distance: float = None,
        scaled: bool = False,
    ) -> None:
        """
        Filter matches.

        Arguments:
            n_best: Maximum number of matches to keep, by descending :attr:`weights`.
            min_weight: Minimum value for :attr:`weights`.
            cam: Camera to use as reference for the following filters.
            max_error: Maximum pixel distance between observed and predicted
                image coordinates, as seen by `cam`.
            max_distance: Maximum pixel distance between image point pairs.
                If the cameras have different image sizes, the image coordinates of the
                other camera are scaled to match the image size of `cam`.
            scaled: Whether `max_error` and `max_distance` are
                absolute pixel distances (False) or
                relative to the image width of `cam` (True).

        Raises:
            ValueError: Filtering on weights failed since these are missing.
        """
        selected = np.ones(self.size, dtype=bool)
        if self.weights is not None:
            if n_best or min_weight:
                raise ValueError("Filtering on weights failed since these are missing")
            if n_best:
                order = np.argsort(-self.weights)
                selected[order[min(n_best, self.size) :]] = False
            if min_weight:
                selected &= self.weights >= min_weight
        ci = self._cam_index(cam)
        co = 0 if ci else 1
        if max_error:
            if scaled:
                max_error = max_error * self.cams[ci].imgsz[0]
            errors = np.linalg.norm(
                self.observed(ci, index=selected) - self.predicted(ci, index=selected),
                axis=1,
            )
            selected[selected] &= errors <= max_error
        if max_distance:
            if scaled:
                max_distance = max_distance * self.cams[ci].imgsz[0]
            scale = self.cams[ci].imgsz / self.cams[co].imgsz
            distances = np.linalg.norm(
                self.predicted(co, index=selected) * scale
                - self.predicted(ci, index=selected),
                axis=1,
            )
            selected[selected] &= distances <= max_distance
        # HACK: Support for RotationMatches
        if self.uvs:
            self.uvs = [uv[selected] for uv in self.uvs]
        else:
            self.xys = [xy[selected] for xy in self.xys]
        if self.weights is not None:
            self.weights = self.weights[selected]


class RotationMatches(Matches):
    """
    Image-image point correspondences for cameras separated by a pure rotation.

    Unlike :class:`Matches`, normalized camera coordinates are pre-computed for speed,
    so internal camera parameters cannot change after initialization.

    Attributes:
        cams (list of Camera): Pair of cameras.
        uvs (list of array): Image point coordinates for each camera
            [(n, [ui, vi]), (n, [uj, vj])].
        xys (list of array): Normalized camera coordinates for each camera
            [(n, [xi, yi]), (n, [xj, yj])].
        weights (array): Relative weight of each point pair (n, ).
        size (int): Number of point pairs (n).

    Raises:
        ValueError: Both :attr:`uvs` and :attr:`xys` are missing.
    """

    def __init__(
        self,
        cams: Sequence[Camera],
        uvs: Sequence[np.ndarray] = None,
        xys: Sequence[np.ndarray] = None,
        weights: np.ndarray = None,
    ) -> None:
        self.cams = cams
        self.uvs, self.xys = self._initialize_uvs_xys(uvs, xys)
        self.uvs = self._build_uvs()
        self.xys = self._build_xys()
        self.weights = weights
        self._test_matches()
        # [imgsz, f, c, k, p]
        self._internals = [cam.to_array()[6:] for cam in self.cams]

    def _initialize_uvs_xys(
        self, uvs: Sequence[np.ndarray] = None, xys: Sequence[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if uvs is None and xys is None:
            raise ValueError("Both uvs and xys are missing")
        if uvs is not None:
            uvs = [np.asarray(uv, dtype=float) for uv in uvs]
        if xys is not None:
            xys = [np.asarray(xy, dtype=float) for xy in xys]
        return uvs, xys

    def _build_xys(self) -> List[np.ndarray]:
        if self.xys is None:
            return [cam._uv_to_xy(uv) for cam, uv in zip(self.cams, self.uvs)]
        return self.xys

    def _build_uvs(self) -> List[np.ndarray]:
        if self.uvs is None:
            return [cam._xy_to_uv(xy) for cam, xy in zip(self.cams, self.xys)]
        return self.uvs

    def _test_internals(self) -> None:
        """Test whether camera internal parameters are unchanged."""
        if any(
            (cam._vector[6:] != v).any() for cam, v in zip(self.cams, self._internals)
        ):
            raise ValueError(
                "Camera internal parameters (imgsz, f, c, k, p) have changed"
            )

    def predicted(self, cam: CamIndex = 0, index: Index = slice(None)) -> np.ndarray:
        """
        Predict image coordinates for a camera from those of the other camera.

        Arguments:
            cam: Camera for which to predict image coordinates.
            index: Indices of points to predict.

        Raises:
            ValueError: Cameras have different positions.
            ValueError: Camera internal parameters (imgsz, f, c, k, p) have changed.
        """
        self._test_position()
        self._test_internals()
        ci = self._cam_index(cam)
        co = 0 if ci else 1
        dxyz = self.cams[co]._xy_to_xyz(self.xys[co][index])
        return self.cams[ci].xyz_to_uv(dxyz, directions=True)

    def to_type(self, mtype: Type["Matches"]) -> "Matches":
        """Return as a matches object of a different type."""
        if mtype is type(self):
            return self
        return mtype(cams=self.cams, uvs=self.uvs, weights=self.weights)


class RotationMatchesXY(RotationMatches):
    """
    Image-image point correspondences for cameras separated by a pure rotation.

    Unlike :class:`Matches`, normalized camera coordinates are pre-computed for speed,
    so internal camera parameters cannot change after initialization.
    Unlike :class:`RotationMatches`, image coordinates may be discarded to save memory.
    :meth:`predicted` and :meth:`observed` return normalized camera coordinates
    rather than image coordinates.

    Attributes:
        cams (list of Camera): Pair of cameras.
        uvs (list of array): Image point coordinates for each camera
            [(n, [ui, vi]), (n, [uj, vj])].
        xys (list of array): Normalized camera coordinates for each camera
            [(n, [xi, yi]), (n, [xj, yj])].
        weights (array): Relative weight of each point pair (n, ).
        size (int): Number of point pairs (n).

    Raises:
        ValueError: Both :attr:`uvs` and :attr:`xys` are missing.
    """

    _EXCLUDED = ["plot"]

    def __init__(
        self,
        cams: Sequence[Camera],
        uvs: Sequence[np.ndarray] = None,
        xys: Sequence[np.ndarray] = None,
        weights: np.ndarray = None,
    ) -> None:
        self.cams = cams
        self.uvs, self.xyz = self._initialize_uvs_xys(uvs, xys)
        self.xys = self._build_xys()
        self.weights = weights
        self._test_matches()
        # [imgsz, f, c, k, p]
        self._internals = [cam.to_array()[6:] for cam in self.cams]

    def __dir__(self) -> list:
        return sorted(
            (set(dir(self.__class__)) | set(self.__dict__.keys())) - set(self._EXCLUDED)
        )

    def __getattribute__(self, name: str) -> Any:
        if name in self._EXCLUDED:
            raise AttributeError(name)
        return super(RotationMatches, self).__getattribute__(name)

    @property
    def size(self) -> int:
        """Number of points pairs."""
        return len(self.xys[0])

    def observed(self, cam: CamIndex = 0, index: Index = slice(None)) -> np.ndarray:
        """
        Return observed camera coordinates.

        Arguments:
            cam: Camera whose points to return.
            index: Indices of points to return.
        """
        c = self._cam_index(cam)
        return self.xys[c][index]

    def predicted(self, cam: CamIndex = 0, index: Index = slice(None)) -> np.ndarray:
        """
        Predict camera coordinates for a camera from those of the other camera.

        Arguments:
            cam: Camera for which to predict camera coordinates.
            index: Indices of points to predict.

        Raises:
            ValueError: Cameras have different positions.
            ValueError: Camera internal parameters (imgsz, f, c, k, p) have changed.
        """
        self._test_position()
        self._test_internals()
        ci = self._cam_index(cam)
        co = 0 if ci else 1
        dxyz = self.cams[co]._xy_to_xyz(self.xys[co][index])
        return self.cams[ci]._xyz_to_xy(dxyz, directions=True)

    def to_type(self, mtype: Type[Matches]) -> Matches:
        """Return as a matches object of a different type."""
        if mtype is type(self):
            return self
        if mtype is Matches:
            uvs = self._build_uvs()
            return mtype(cams=self.cams, uvs=uvs, weights=self.weights)
        return mtype(cams=self.cams, uvs=self.uvs, xys=self.xys, weights=self.weights)


class RotationMatchesXYZ(RotationMatchesXY):
    """
    Image-image point correspondences for cameras separated by a pure rotation.

    Unlike :class:`Matches`, normalized camera coordinates are pre-computed for speed,
    so internal camera parameters cannot change after initialization.
    Unlike :class:`RotationMatches`, image coordinates may be discarded to save memory.
    :meth:`predicted` returns world ray directions and :meth:`observed` is disabled.
    Exclusively for use with :class:`ObserverCameras`.

    Attributes:
        cams (list of Camera): Pair of cameras.
        uvs (list of array): Image point coordinates for each camera
            [(n, [ui, vi]), (n, [uj, vj])].
        xys (list of array): Normalized camera coordinates for each camera
            [(n, [xi, yi]), (n, [xj, yj])].
        weights (array): Relative weight of each point pair (n, ).
        size (int): Number of point pairs (n).

    Raises:
        ValueError: Both :attr:`uvs` and :attr:`xys` are missing.
    """

    _EXCLUDED = ["observed"]

    def __init__(
        self,
        cams: Sequence[Camera],
        uvs: Sequence[np.ndarray] = None,
        xys: Sequence[np.ndarray] = None,
        weights: np.ndarray = None,
    ) -> None:
        super().__init__(cams=cams, uvs=uvs, xys=xys, weights=weights)

    def __dir__(self) -> list:
        return sorted(
            (set(dir(self.__class__)) | set(self.__dict__.keys())) - set(self._EXCLUDED)
        )

    def __getattribute__(self, name: str) -> Any:
        if name in self._EXCLUDED:
            raise AttributeError(name)
        return super(RotationMatchesXY, self).__getattribute__(name)

    def predicted(self, cam: CamIndex = 0, index: Index = slice(None)) -> np.ndarray:
        """
        Predict world coordinates for a camera.

        Returns world coordinates as ray directions with unit length.

        Arguments:
            cam: Camera for which to predict world coordinates.
            index: Indices of points to predict.
        """
        self._test_position()
        self._test_internals()
        c = self._cam_index(cam)
        dxyz = self.cams[c]._xy_to_xyz(self.xys[c][index])
        # Normalize world coordinates to unit sphere
        dxyz *= 1 / np.linalg.norm(dxyz, ord=2, axis=1, keepdims=True)
        return dxyz


# ---- Models ----

# Models support RANSAC with the following API:
# .size
# .fit(index)
# .errors(params, index)


class Polynomial:
    """
    Least-square polynomial model.

    Fits a polynomial to 2-dimensional points and
    returns the coefficients that minimize the squared error.

    Attributes:
        xy: Observed point coordinates (n, [x, y]).
        deg: Degree of the polynomial.
        size: Number of observations (n ).

    Example:
        Add noisy data to an initial set of points lying close to the line `y = x + 0`.

        >>> xy = [(0, 0), (1.1, 1), (1.9, 2), (3.1, 3), (3.9, 4)]
        >>> xy += [(3, 0.1), (0.1, 3)]

        Fitting a least-squares line to noisy points will not result in a good fit.

        >>> model = Polynomial(xy, deg=1)
        >>> model.fit()
        array([0.41631292, 1.09232868])

        Instead, we can use RANSAC to find the inliers among the noise.
        The polynomial coefficients of the fit are now much closer to the ideal (1, 0).

        >>> params, inliers = ransac(
        ...     model, n=2, max_error=0.2, min_inliers=2, iterations=100)
        >>> inliers
        array([0, 1, 2, 3, 4])
        >>> params
        array([ 1.01659751, -0.03319502])

        Plotting the result confirms a successful fit.

        >>> import matplotlib.pyplot as plt
        >>> model.plot(params=params, index=inliers)
        {'unselected': <matplotlib.collections.PathCollection ...>,
        'selected': <matplotlib.collections.PathCollection ...>, 
        'predicted': [<matplotlib.lines.Line2D ...>]}
        >>> plt.show()  # doctest: +SKIP
        >>> plt.close()
    """

    def __init__(self, xy: np.ndarray, deg: int = 1) -> None:
        self.xy = np.asarray(xy)
        self.deg = deg

    @property
    def size(self) -> int:
        """Number of observations."""
        return len(self.xy)

    def predict(self, params: np.ndarray, index: Index = slice(None)) -> np.ndarray:
        """
        Predict the values of a polynomial.

        Arguments:
            params: Values of the polynomial, from highest to lowest degree component.
            index: Indices of points for which to predict y from x.
        """
        return np.polyval(params, self.xy[index, 0])

    def errors(self, params: np.ndarray, index: Index = slice(None)) -> np.ndarray:
        """
        Compute the errors of a polynomial prediction.

        Arguments:
            params: Values of the polynomial, from highest to lowest degree component.
            index: Indices of points for which to predict y from x.
        """
        prediction = self.predict(params, index)
        return np.abs(prediction - self.xy[index, 1])

    def fit(self, index: Index = slice(None)) -> np.ndarray:
        """
        Fit a polynomial to the points.

        Arguments:
            index: Indices of points to use for fitting.

        Returns:
            Values of the polynomial, from highest to lowest degree component.
        """
        return np.polyfit(self.xy[index, 0], self.xy[index, 1], deg=self.deg)

    def plot(
        self,
        params: np.ndarray = None,
        index: Index = slice(None),
        selected: ColorArgs = "red",
        unselected: ColorArgs = "gray",
        predicted: ColorArgs = "red",
        **kwargs: Any
    ) -> Dict[
        str,
        Optional[
            Union[matplotlib.collections.PathCollection, List[matplotlib.lines.Line2D]]
        ],
    ]:
        """
        Plot observations and the polynomial fit.

        Arguments:
            params: Values of the polynomial, from highest to lowest degree component.
            index: Indices of points to select.
            selected: For selected points, optional arguments to
                matplotlib.pyplot.scatter (dict), color, or `None` to hide.
            unselected: For unselected points, optional arguments to
                matplotlib.pyplot.scatter (dict), color, or `None` to hide.
            predicted: For polynomial fit, optional arguments to
                matplotlib.pyplot.plot (dict), color, or `None` to hide.
            **kwargs: Optional arguments to matplotlib.pyplot.scatter for all points.
        """
        if params is None:
            params = self.fit(index)
        defaults = {}
        result = {}
        full = np.arange(self.size)
        index, unindex = full[index], np.delete(full, index)
        for idx, args, label in [
            (unindex, unselected, "unselected"),
            (index, selected, "selected"),
        ]:
            if not len(idx) or args is None:
                result = None
                continue
            if not isinstance(args, dict):
                args = {"c": args}
            result[label] = matplotlib.pyplot.scatter(
                self.xy[idx, 0], self.xy[idx, 1], **{**args, **kwargs}
            )
        if predicted is None:
            result["predicted"] = None
        else:
            if not isinstance(predicted, dict):
                predicted = {"color": predicted}
            result["predicted"] = matplotlib.pyplot.plot(
                self.xy[:, 0], self.predict(params), **predicted
            )
        return result


Control = Union[Points, Lines, Matches, RotationMatches]
Params = Dict[str, Union[bool, int, Sequence[int]]]
FitParams = Union[Sequence[float], lmfit.parameter.Parameters]


class Cameras(object):
    """
    Multi-camera optimization.

    Finds the camera parameter values that minimize the reprojection errors of camera
    control:

        - image-world point coordinates (Points)
        - image-world line coordinates (Lines)
        - image-image point coordinates (Matches)

    If used with RANSAC (see :func:`ransac`) with multiple control objects,
    results may be unstable since samples are drawn randomly from all observations,
    and computation will be slow since errors are calculated for all points,
    then subset.

    Arguments:
        scales: Whether to compute and use scale factors for each parameter
            (can be more stable).
        sparsity: Whether to compute and use a sparsity structure for the
            estimation of the Jacobian matrix (much faster for large, sparse systems).

    Attributes:
        cams (list of Camera): Cameras.
        controls (list of Control): Camera control.
        cam_params (list of dict): Parameters to optimize separately for each camera
            (see :meth:`Cameras.parse_params`).
        group_indices (np.ndarray): Integer index of `cams` belonging to each group.
        group_params (list of dict): Parameters to optimize together for all cameras in
            each group (see :meth:`Cameras.parse_params`).
        weights (np.ndarray): Weights for each control point.
        scales (np.ndarray): Scale factors for each parameter (see `camera_scales()`).
        sparsity (scipy.sparse.spmatrix): Sparsity structure for the estimation of the
            Jacobian matrix.
        vectors (list of np.ndarray): Original camera vectors.
        params (lmfit.Parameters): Parameter initial values and bounds.
    """

    def __init__(
        self,
        cams: Sequence[Camera],
        controls: Sequence[Control],
        cam_params: Sequence[Params] = None,
        group_indices: Sequence[int] = None,
        group_params: Sequence[Params] = None,
        weights: np.ndarray = None,
        scales: bool = True,
        sparsity: bool = True,
    ):
        (
            cams,
            controls,
            cam_params,
            group_indices,
            group_params,
        ) = self.__class__._as_lists(
            cams, controls, cam_params, group_indices, group_params
        )
        # Core attributes
        self.cams = cams
        controls = self.prune_controls(controls, cams=self.cams)
        self.controls = controls
        ncams = len(self.cams)
        if cam_params is None:
            cam_params = [{}] * ncams
        self.cam_params = cam_params
        if group_indices is None:
            group_indices = [range(ncams)]
        self.group_indices = group_indices
        if group_params is None:
            group_params = [{}] * len(self.group_indices)
        self.group_params = group_params
        self.weights = weights
        # Build lmfit parameters
        # params, cam_masks, group_masks, cam_breaks, group_breaks for set_cameras()
        self.update_params()
        # Test for errors
        self._test()
        # Save original camera vectors for reset_cameras()
        self.vectors = [cam.to_array() for cam in self.cams]
        # Parameter scale factors
        self.scales = None
        if scales:
            self._build_scales()
        # Sparse Jacobian
        self.sparsity = None
        if sparsity:
            self._build_sparsity()

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray) -> None:
        if value is None:
            self._weights = value
        else:
            value = np.atleast_2d(value).reshape(-1, 1)
            self._weights = value * len(value) / sum(value)

    @staticmethod
    def _as_lists(cams, controls, cam_params, group_indices, group_params):
        if isinstance(cams, Camera):
            cams = [cams]
        if isinstance(controls, (Points, Lines, Matches)):
            controls = [controls]
        if isinstance(cam_params, dict):
            cam_params = [cam_params]
        if isinstance(group_indices, int):
            group_indices = [group_indices]
        if group_indices is not None and isinstance(group_indices[0], int):
            group_indices = [group_indices]
        if isinstance(group_params, dict):
            group_params = [group_params]
        return cams, controls, cam_params, group_indices, group_params

    @staticmethod
    def _lmfit_labels(mask, cam=None, group=None):
        attributes = ("xyz", "viewdir", "imgsz", "f", "c", "k", "p")
        lengths = [3, 3, 2, 2, 2, 6, 2]
        base_labels = np.array(
            [
                attribute + str(i)
                for attribute, length in zip(attributes, lengths)
                for i in range(length)
            ]
        )
        labels = base_labels[mask]
        if cam is not None:
            labels = ["cam" + str(cam) + "_" + label for label in labels]
        if group is not None:
            labels = ["group" + str(group) + "_" + label for label in labels]
        return labels

    @staticmethod
    def _get_control_cams(control):
        if isinstance(control, (Points, Lines)):
            return [control.cam]
        return control.cams

    @classmethod
    def prune_controls(
        cls, controls: List[Control], cams: List[Params]
    ) -> List[Control]:
        """
        Return the controls which reference the specified cameras.

        Arguments:
            controls: Control objects.
            cams: Camera objects.

        Returns:
            Controls which reference one or more cameras in `cams`.

        Example:
            >>> cams = [Camera(imgsz=100, f=10), Camera(imgsz=100, f=10)]
            >>> controls = [
            ...     Points(cam=cams[0], uv=[(0, 0)], xyz=[(0, 0, 0)]),
            ...     Lines(cam=cams[1], uvs=[[(0, 0)]], xyzs=[[(0, 0, 0)]]),
            ...     Matches(cams=cams, uvs=[[(0, 0)], [(0, 0)]])
            ... ]
            >>> Cameras.prune_controls(controls, cams)
            [<...Points...>, <...Lines...>, <...Matches...>]
            >>> Cameras.prune_controls(controls, cams[0:1])
            [<...Points...>, <...Matches...>]
            >>> Cameras.prune_controls(controls, cams[1:2])
            [<...Lines...>, <...Matches...>]
        """
        return [
            control
            for control in controls
            if len(set(cams) & set(cls._get_control_cams(control))) > 0
        ]

    @staticmethod
    def camera_scales(
        cam: Camera, controls: List[Union[Points, Lines]] = None
    ) -> np.ndarray:
        """
        Return camera parameter scale factors.

        These represent the estimated change in each camera parameter needed
        to displace the image coordinates of a point by one pixel.

        Arguments:
            cam: Camera object.
            controls: World control (:class:`Points`, :class:`Lines`),
                used to estimate the impact of changing camera position (`cam.xyz`).
        """
        # Compute pixels per unit change for each variable
        dpixels = np.ones(20, dtype=float)
        # Compute average distance from image center
        # https://math.stackexchange.com/a/100823
        mean_r_uv = (cam.imgsz.mean() / 6) * (np.sqrt(2) + np.log(1 + np.sqrt(2)))
        mean_r_xy = mean_r_uv / cam.f.mean()
        # xyz (if f is not descaled)
        # Compute mean distance to world features
        if controls:
            xyz = []
            for control in controls:
                if (
                    isinstance(control, (Points, Lines))
                    and cam is control.cam
                    and not control.directions
                ):
                    if hasattr(control, "xyz"):
                        xyz.append(control.xyz)
                    elif hasattr(control, "xyzs"):
                        xyz.append(control.xyz)
            if xyz:
                # NOTE: Upper bound (assumes motion perpendicular to feature direction)
                dpixels[0:3] = (
                    cam.f.mean() / np.linalg.norm(np.vstack(xyz) - cam.xyz).mean()
                )
        # viewdir[0, 1]
        # First angle rotates camera left-right
        # Second angle rotates camera up-down
        imgsz_degrees = (2 * np.arctan(cam.imgsz / (2 * cam.f))) * (180 / np.pi)
        dpixels[3:5] = cam.imgsz / imgsz_degrees  # pixels per degree
        # viewdir[2]
        # Third angle rotates camera around image center
        theta = np.pi / 180
        dpixels[5] = 2 * mean_r_uv * np.sin(theta / 2)  # pixels per degree
        # imgsz
        # NOTE: Correct
        dpixels[6:8] = 0.5
        # f (if not descaled)
        # NOTE: Correct for f += (1, 1), not f += (1, 0) or (0, 1)
        dpixels[8:10] = mean_r_xy
        # c
        # NOTE: Correct
        dpixels[10:12] = 1
        # k (if f is not descaled)
        # Approximate at mean radius
        dpixels[12:18] = [
            mean_r_xy ** 3 * cam.f.mean() * 2 ** (1 / 2),
            mean_r_xy ** 5 * cam.f.mean() * 2 ** (3 / 2),
            mean_r_xy ** 7 * cam.f.mean() * 2 ** (5 / 2),
            mean_r_xy ** 3
            / (1 + cam.k[3] * mean_r_xy ** 2)
            * cam.f.mean()
            * 2 ** (1 / 2),
            mean_r_xy ** 5
            / (1 + cam.k[4] * mean_r_xy ** 4)
            * cam.f.mean()
            * 2 ** (3 / 2),
            mean_r_xy ** 7
            / (1 + cam.k[5] * mean_r_xy ** 6)
            * cam.f.mean()
            * 2 ** (5 / 2),
        ]
        # p (if f is not descaled)
        # Approximate at mean radius at 45 degree angle
        dpixels[18:20] = np.sqrt(5) * mean_r_xy ** 2 * cam.f.mean()
        # Convert pixels per change to change per pixel (the inverse)
        return 1 / dpixels

    @staticmethod
    def camera_bounds(cam: Camera) -> np.ndarray:
        """
        Return default camera parameter bounds.

        Bounds for distortion coefficients are based on tested limits of the
        default :class:`Camera` undistort routine.

        Arguments:
            cam: Camera object.

        Returns:
            Bounds for each parameter, in the order of :meth:`Camera.to_array` (n, 2).
        """
        k = cam.f.mean() / 4000
        p = cam.f.mean() / 40000
        return np.array(
            [
                # xyz
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                # viewdir
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                # imgsz
                [0, np.inf],
                [0, np.inf],
                # f
                [0, np.inf],
                [0, np.inf],
                # c
                [-0.5, 0.5] * cam.imgsz[0:1],
                [-0.5, 0.5] * cam.imgsz[1:2],
                # k
                [-k, k],
                [-k / 2, k / 2],
                [-k / 2, k / 2],
                [-k, k],
                [-k, k],
                [-k, k],
                # p
                [-p, p],
                [-p, p],
            ],
            dtype=float,
        )

    @staticmethod
    def parse_params(
        params: Params = None, default_bounds: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a mask of selected camera parameters and associated bounds.

        Arguments:
            params: Parameters to select by name and indices. For example:

                    - {'viewdir': True} : All `viewdir` elements
                    - {'viewdir': 0} : First `viewdir` element
                    - {'viewdir': [0, 1]} : First and second `viewdir` elements

                Bounds can be specified inside a tuple (indices, min, max).
                Singletons are expanded as needed, and `np.inf` with the
                appropriate sign can be used to indicate no bound:

                    - {'viewdir': ([0, 1], -np.inf, 180)}
                    - {'viewdir': ([0, 1], -np.inf, [180, 180])}

                `None` or `np.nan` may also be used. These are replaced by the
                values in `default_bounds` (for example, from `camera_bounds()`),
                or (-)`np.inf` if `None`.

        Returns:
            Parameter boolean mask (20, ) and parameter min and max bounds (20, 2).
        """
        if params is None:
            params = {}
        attributes = ("xyz", "viewdir", "imgsz", "f", "c", "k", "p")
        indices = (0, 3, 6, 8, 10, 12, 18, 20)
        mask = np.zeros(20, dtype=bool)
        bounds = np.full((20, 2), np.nan)
        for key, value in params.items():
            if key in attributes:
                if isinstance(value, tuple):
                    selection = value[0]
                else:
                    selection = value
                if selection or selection == 0:
                    i = attributes.index(key)
                    if selection is True:
                        positions = range(indices[i], indices[i + 1])
                    else:
                        positions = indices[i] + np.atleast_1d(selection)
                    mask[positions] = True
                if isinstance(value, tuple):
                    min_bounds = np.atleast_1d(value[1])
                    if len(min_bounds) == 1:
                        min_bounds = np.repeat(min_bounds, len(positions))
                    max_bounds = np.atleast_1d(value[2])
                    if len(max_bounds) == 1:
                        max_bounds = np.repeat(max_bounds, len(positions))
                    bounds[positions] = np.column_stack((min_bounds, max_bounds))
        if default_bounds is not None:
            missing_min = (bounds[:, 0] is None) | (np.isnan(bounds[:, 0]))
            missing_max = (bounds[:, 1] is None) | (np.isnan(bounds[:, 1]))
            bounds[missing_min, 0] = default_bounds[missing_min, 0]
            bounds[missing_max, 1] = default_bounds[missing_max, 1]
        missing_min = (bounds[:, 0] is None) | (np.isnan(bounds[:, 0]))
        missing_max = (bounds[:, 1] is None) | (np.isnan(bounds[:, 1]))
        bounds[missing_min, 0] = -np.inf
        bounds[missing_max, 1] = np.inf
        return mask, bounds

    def _test(self):
        """
        Test for scenarios leading to unexpected results.
        """
        # Error: No controls reference the cameras
        if not len(self.controls):
            raise ValueError("No controls reference the cameras")
        # Error: 'f' or 'c' in `group_params` but image sizes not equal
        for i, idx in enumerate(self.group_indices):
            fc = "f" in self.group_params[i] or "c" in self.group_params[i]
            sizes = np.unique(np.row_stack([self.cams[i].imgsz for i in idx]), axis=0)
            if fc and len(sizes) > 1:
                raise ValueError(
                    "Group "
                    + str(i)
                    + ": 'f' or 'c' in parameters but image sizes not equal"
                )
        # Error: Cameras appear in multiple groups with overlapping masks
        # Test: indices of groups with overlapping masks non-unique
        M = np.row_stack(self.group_masks)
        overlaps = np.nonzero(np.count_nonzero(M, axis=0) > 1)[0]
        for i in overlaps:
            groups = np.nonzero(M[:, i])[0]
            idx = np.concatenate([self.group_indices[group] for group in groups])
            if len(np.unique(idx)) < len(idx):
                raise ValueError(
                    "Some cameras are in multiple groups with overlapping masks"
                )
        # Error: Some cameras with params do not appear in controls
        control_cams = [
            cam for control in self.controls for cam in self._get_control_cams(control)
        ]
        cams_with_params = [
            cam
            for i, cam in enumerate(self.cams)
            if self.cam_params[i]
            or any(
                [
                    self.group_params[j]
                    for j, idx in enumerate(self.group_indices)
                    if i in idx
                ]
            )
        ]
        if set(cams_with_params) - set(control_cams):
            raise ValueError("Not all cameras with params appear in controls")

    def _build_scales(self):
        # TODO: Weigh each camera by number of control points (sum of weights)
        scales = [self.__class__.camera_scales(cam, self.controls) for cam in self.cams]
        cam_scales = [scale[mask] for scale, mask in zip(scales, self.cam_masks)]
        group_scales = [
            np.nanmean(np.row_stack([scales[i][mask] for i in idx]), axis=0)
            for mask, idx in zip(self.group_masks, self.group_indices)
        ]
        self.scales = np.hstack((np.hstack(group_scales), np.hstack(cam_scales)))

    def _build_sparsity(self):
        # Number of observations
        m_control = [2 * control.size for control in self.controls]
        m = sum(m_control)
        # Number of parameters
        n_groups = [np.count_nonzero(mask) for mask in self.group_masks]
        n_cams = [np.count_nonzero(mask) for mask in self.cam_masks]
        n = sum(n_groups) + sum(n_cams)
        # Group lookup
        groups = np.zeros((len(self.cams), len(self.group_indices)), dtype=bool)
        for i, idx in enumerate(self.group_indices):
            groups[idx, i] = True
        # Initialize sparse matrix with zeros
        S = scipy.sparse.lil_matrix((m, n), dtype=int)
        # Build matrix
        control_breaks = np.cumsum([0] + m_control)
        for i, control in enumerate(self.controls):
            ctrl_slice = slice(control_breaks[i], control_breaks[i + 1])
            for cam in self._get_control_cams(control):
                try:
                    j = self.cams.index(cam)
                except ValueError:
                    continue
                cam_slice = slice(self.cam_breaks[j], self.cam_breaks[j + 1])
                # Camera parameters
                S[ctrl_slice, cam_slice] = 1
                # Group parameters
                for group in np.nonzero(groups[j])[0]:
                    group_slice = slice(
                        self.group_breaks[group], self.group_breaks[group + 1]
                    )
                    S[ctrl_slice, group_slice] = 1
        self.sparsity = S

    def update_params(self) -> None:
        """
        Update parameter bounds and initial values from current state.
        """
        self.params = lmfit.Parameters()
        # Camera parameters
        cam_bounds = [self.__class__.camera_bounds(cam) for cam in self.cams]
        self.cam_masks, cam_bounds = zip(
            *[
                self.__class__.parse_params(params, default_bounds=bounds)
                for params, bounds in zip(self.cam_params, cam_bounds)
            ]
        )
        cam_labels = [
            self.__class__._lmfit_labels(mask, cam=i, group=None)
            for i, mask in enumerate(self.cam_masks)
        ]
        cam_values = [
            self.cams[i]._vector[mask] for i, mask in enumerate(self.cam_masks)
        ]
        # Group parameters
        self.group_masks = []
        for group, idx in enumerate(self.group_indices):
            bounds = np.column_stack(
                (
                    np.column_stack([cam_bounds[i][:, 0] for i in idx]).max(axis=1),
                    np.column_stack([cam_bounds[i][:, 1] for i in idx]).min(axis=1),
                )
            )
            mask, bounds = self.__class__.parse_params(
                self.group_params[group], default_bounds=bounds
            )
            labels = self.__class__._lmfit_labels(mask, cam=None, group=group)
            # NOTE: Initial group values as mean of cameras
            values = np.nanmean(
                np.row_stack([self.cams[i]._vector[mask] for i in idx]), axis=0
            )
            for label, value, bound in zip(labels, values, bounds[mask]):
                self.params.add(
                    name=label, value=value, vary=True, min=bound[0], max=bound[1]
                )
            self.group_masks.append(mask)
        # Add camera parameters after group parameters
        for i in range(len(self.cams)):
            for label, value, bound in zip(
                cam_labels[i], cam_values[i], cam_bounds[i][self.cam_masks[i]]
            ):
                self.params.add(
                    name=label, value=value, vary=True, min=bound[0], max=bound[1]
                )
        # Pre-compute index breaks for set_cameras()
        self.group_breaks = np.cumsum(
            [0] + [np.count_nonzero(mask) for mask in self.group_masks]
        )
        self.cam_breaks = np.cumsum(
            [self.group_breaks[-1]]
            + [np.count_nonzero(mask) for mask in self.cam_masks]
        )

    def set_cameras(self, params: FitParams, save: bool = False) -> None:
        """
        Set camera parameter values.

        The operation can be reversed with :meth:`Cameras.reset_cameras()`.

        Arguments:
            params: Parameter values ordered first
                by group or camera [group0 | group1 | cam0 | cam1 | ...],
                then ordered by position in :meth:`Camera.to_array`.
            save: Whether to save the new state of the cameras as the fallback for
                :meth:`Cameras.reset_cameras()`.
        """
        if isinstance(params, lmfit.parameter.Parameters):
            params = list(params.valuesdict().values())
        for i, idx in enumerate(self.group_indices):
            for j in idx:
                self.cams[j]._vector[self.group_masks[i]] = params[
                    self.group_breaks[i] : self.group_breaks[i + 1]
                ]
                self.cams[j]._vector[self.cam_masks[j]] = params[
                    self.cam_breaks[j] : self.cam_breaks[j + 1]
                ]
        if save:
            self.vectors = [cam.to_array() for cam in self.cams]

    def reset_cameras(self) -> None:
        """
        Reset camera parameters to their previously saved state.
        """
        for cam, vector in zip(self.cams, self.vectors):
            cam._vector = vector.copy()

    @property
    def size(self) -> int:
        """
        Return the total number of data points.
        """
        return np.sum([control.size for control in self.controls])

    def observed(self, index: Index = slice(None)):
        """
        Return the observed image coordinates for all camera control.

        See control `observed()` method for more details.

        Arguments:
            index: Indices of points to return.
        """
        if len(self.controls) == 1:
            return self.controls[0].observed(index=index)
        return np.vstack([control.observed() for control in self.controls])[index]

    def predicted(
        self, params: FitParams = None, index: Index = slice(None)
    ) -> np.ndarray:
        """
        Return the predicted image coordinates for all camera control.

        See control `predicted()` method for more details.

        Arguments:
            params: Parameter values (see :meth:`Cameras.set_cameras`).
            index: Indices of points to return.
        """
        if params is not None:
            vectors = [cam.to_array() for cam in self.cams]
            self.set_cameras(params)
        if len(self.controls) == 1:
            result = self.controls[0].predicted(index=index)
        else:
            # TODO: Map index to subindices for each control
            result = np.vstack([control.predicted() for control in self.controls])[
                index
            ]
        if params is not None:
            for cam, vector in zip(self.cams, vectors):
                cam._vector = vector
        return result

    def residuals(
        self, params: FitParams = None, index: Index = slice(None)
    ) -> np.ndarray:
        """
        Return the reprojection residuals for all camera control.

        Residuals are the difference between :meth:`Cameras.predicted` and
        :meth:`Cameras.observed`.

        Arguments:
            params: Parameter values (see :meth:`Cameras.set_cameras`).
            index: Indices of points to return.
        """
        d = self.predicted(params=params, index=index) - self.observed(index=index)
        if self.weights is None:
            return d
        return d * self.weights[index]

    def errors(
        self, params: FitParams = None, index: Index = slice(None)
    ) -> np.ndarray:
        """
        Return the reprojection errors for all camera control.

        Errors are the Euclidean distance between :meth:`Cameras.predicted` and
        :meth:`Cameras.observed`.

        Arguments:
            params: Parameter values (see :meth:`Cameras.set_cameras`).
            index: Indices of points to return.
        """
        return np.linalg.norm(self.residuals(params=params, index=index), axis=1)

    def fit(
        self,
        index: Index = slice(None),
        cam_params: List[List[Params]] = None,
        group_params: List[List[Params]] = None,
        full: bool = False,
        method: str = "least_squares",
        **kwargs: Any
    ) -> FitParams:
        """
        Return optimal camera parameter values.

        Find the camera parameter values that minimize the reprojection residuals
        or a derivative objective function across all control.
        See `lmfit.minimize()` (https://lmfit.github.io/lmfit-py/fitting.html).

        Arguments:
            index: Indices of residuals to include.
            cam_params: Sequence of `cam_params` to fit iteratively
                before the final run. Must be `None` or same length as `group_params`.
            group_params: Sequence of `group_params` to fit iteratively
                before the final run. Must be `None` or same length as `cam_params`.
            full: Whether to return the full result of `lmfit.Minimize()`.
            **kwargs: Additional arguments to `lmfit.minimize()`.
                `self.scales` and `self.jac_sparsity` (if computed) are applied
                to the following arguments based on `method`:
                `diag=self.scales` for 'leastsq' and
                `x_scale=self.scales` and `jac_sparsity=self.sparsity`
                for 'least_squares'.

        Returns:
            Parameter values ordered first
                by group or camera (group, cam0, cam1, ...),
                then ordered by position in :meth:`Camera.to_array()`.
        """
        kwargs = {"nan_policy": "omit", **kwargs}
        if method == "leastsq":
            if self.scales is not None and "diag" not in kwargs:
                kwargs["diag"] = self.scales
        if method == "least_squares":
            if self.scales is not None and "x_scale" not in kwargs:
                kwargs["x_scale"] = self.scales
            if self.sparsity is not None and "jac_sparsity" not in kwargs:
                if isinstance(index, slice) and index == slice(None):
                    kwargs["jac_sparsity"] = self.sparsity
                else:
                    if isinstance(index, slice):
                        jac_index = np.arange(self.size)[index]
                    else:
                        jac_index = np.asarray(index)
                    jac_index = np.dstack((2 * jac_index, 2 * jac_index + 1)).ravel()
                    kwargs["jac_sparsity"] = self.sparsity[jac_index]

        def callback(params, iter, resid, *args, **kwargs):
            err = np.linalg.norm(resid.reshape(-1, 2), ord=2, axis=1).mean()
            sys.stdout.write("\r" + str(err))
            sys.stdout.flush()

        iterations = max(
            len(cam_params) if cam_params else 0,
            len(group_params) if group_params else 0,
        )
        if iterations:
            for n in range(iterations):
                iter_cam_params = cam_params[n] if cam_params else self.cam_params
                iter_group_params = (
                    group_params[n] if group_params else self.group_params
                )
                model = Cameras(
                    cams=self.cams,
                    controls=self.controls,
                    cam_params=iter_cam_params,
                    group_params=iter_group_params,
                )
                values = model.fit(index=index, method=method, **kwargs)
                if values is not None:
                    model.set_cameras(params=values)
            self.update_params()
        result = lmfit.minimize(
            params=self.params,
            fcn=self.residuals,
            kws={"index": index},
            iter_cb=callback,
            method=method,
            **kwargs
        )
        sys.stdout.write("\n")
        if iterations:
            self.reset_cameras()
            self.update_params()
        if not result.success:
            print(result.message)
        if full:
            return result
        elif result.success:
            return np.array(list(result.params.valuesdict().values()))

    def plot(
        self,
        params: FitParams = None,
        cam: CamIndex = 0,
        index: Index = slice(None),
        selected: ColorArgs = "red",
        unselected: ColorArgs = "gray",
        lines_observed: ColorArgs = "green",
        lines_predicted: ColorArgs = "yellow",
        **kwargs: Any
    ) -> List[
        Dict[
            str,
            Optional[Union[matplotlib.quiver.Quiver, List[matplotlib.lines.Line2D]]],
        ]
    ]:
        """
        Plot reprojection errors as quivers.

        Arrows point from observed to predicted image coordinates.

        Arguments:
            params: Parameter values [group | cam0 | cam1 | ...].
                If `None` (default), cameras are used unchanged.
            cam: Camera to plot in (as object or position in `self.cams`).
            index: Indices of points to plot.
                By default, all points are plotted.
                Other values are only supported for a single control.
            selected: For selected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            unselected: For unselected points, optional arguments to
                matplotlib.pyplot.quiver (dict), color, or `None` to hide.
            lines_observed: For image lines, optional arguments to
                matplotlib.pyplot.plot (dict), color, or `None` to hide.
            lines_predicted: For world lines, optional arguments to
                matplotlib.pyplot.plot (dict), color, or `None` to hide.
            **kwargs: Optional arguments to matplotlib.pyplot.quiver for all points.
        """
        if index != slice(None) and len(self.controls) > 1:
            # TODO: Map index to subindices for each control
            raise ValueError(
                "Plotting with `index` not yet supported with multiple controls"
            )
        if params is not None:
            vectors = [cam.to_array() for cam in self.cams]
            self.set_cameras(params)
        cam = self.cams[cam] if isinstance(cam, int) else cam
        cam_controls = self.prune_controls(self.controls, cams=[cam])
        results = []
        for control in cam_controls:
            if isinstance(control, Lines):
                result = control.plot(
                    index=index,
                    selected=selected,
                    unselected=unselected,
                    observed=lines_observed,
                    predicted=lines_predicted,
                    **kwargs
                )
            elif isinstance(control, Points):
                result = control.plot(
                    index=index, selected=selected, unselected=unselected, **kwargs
                )
            elif isinstance(control, Matches):
                result = control.plot(
                    cam=cam,
                    index=index,
                    selected=selected,
                    unselected=unselected,
                    **kwargs
                )
            results.append(result)
        if params is not None:
            for cam, vector in zip(self.cams, vectors):
                cam._vector = vector
        return results

    def plot_weights(
        self, index: Index = slice(None), scale: float = 1, cmap: str = None
    ) -> None:
        weights = np.ones(self.size) if self.weights is None else self.weights
        uv = self.observed(index=index)
        matplotlib.pyplot.scatter(
            uv[:, 0], uv[:, 1], c=weights[index], s=scale * weights[index], cmap=cmap
        )
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.gca().invert_yaxis()


class ObserverCameras(object):
    """
    `ObserverCameras` finds the optimal view directions of the cameras in an `Observer`.

    Attributes:
        observer (`glimpse.Observer`): Observer with the cameras to orient
        anchors (iterable): Integer indices of `observer.images` to use as anchors.
            If `None`, the first image is used.
        matches (array): Grid of `RotationMatchesXYZ` objects.
        matcher (KeypointMatcher): KeypointMatcher object used by
            `self.build_keypoints()` and `self.build_matches()`
        viewdirs (array): Original camera view directions
    """

    def __init__(self, observer, matches=None, anchors=None):
        self.observer = observer
        if anchors is None:
            is_anchor = [img.anchor for img in self.observer.images]
            anchors = np.where(is_anchor)[0]
            if len(anchors) == 0:
                warnings.warn("No anchor image found, using first image as anchor")
                anchors = (0,)
        self.anchors = anchors
        self.matches = matches
        self.matcher = KeypointMatcher(images=self.observer.images)
        # Placeholders
        self.viewdirs = np.vstack(
            [img.cam.viewdir.copy() for img in self.observer.images]
        )

    def set_cameras(self, viewdirs):
        for i, img in enumerate(self.observer.images):
            img.cam.viewdir = viewdirs[i]

    def reset_cameras(self):
        self.set_cameras(viewdirs=self.viewdirs.copy())

    def build_keypoints(self, *args, **kwargs):
        self.matcher.build_keypoints(*args, **kwargs)

    def build_matches(self, *args, **kwargs):
        self.matcher.build_matches(*args, **kwargs)
        self.matcher.convert_matches(RotationMatchesXYZ)
        self.matches = self.matcher.matches

    def fit(self, anchor_weight=1e6, method="bfgs", **params):
        """
        Return optimal camera view directions.

        The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is used to find
        the camera view directions that minimize the sum of the absolute differences
        (L1-norm). See `scipy.optimize.minimize(method='bfgs')`.

        Arguments:
            anchor_weight (float): Weight on anchor image view directions being correct
            **params: Additional arguments to `scipy.optimize.minimize()`

        Returns:
            `scipy.optimize.OptimizeResult`: The optimization result.
                Attributes include solution array `x`, boolean `success`, and `message`.
        """
        # Ensure matches are in COO sparse matrix format
        if isinstance(self.matches, scipy.sparse.coo.coo_matrix):
            matches = self.matches
        else:
            matches = scipy.sparse.coo_matrix(matches)

        # Define combined objective, jacobian function
        def fun(viewdirs):
            viewdirs = viewdirs.reshape(-1, 3)
            self.set_cameras(viewdirs=viewdirs)
            objective = 0
            gradients = np.zeros(viewdirs.shape)
            for i in self.anchors:
                objective += (anchor_weight / 2.0) * np.sum(
                    (viewdirs[i] - self.viewdirs[i]) ** 2
                )
                gradients[i] += anchor_weight * (viewdirs[i] - self.viewdirs[i])
            for m, i, j in zip(matches.data, matches.row, matches.col):
                # Project matches
                dxyz = m.predicted(cam=0) - m.predicted(cam=1)
                objective += np.sum(np.abs(dxyz))
                # i -> j
                xy_hat = np.column_stack((m.xys[0], np.ones(m.size)))
                dD_dw = np.matmul(m.cams[0].Rprime, xy_hat.T)
                delta = np.sign(dxyz).reshape(-1, 3, 1)
                gradient = np.sum(np.matmul(dD_dw.T, delta).T, axis=2).squeeze()
                gradients[i] += gradient
                # j -> i
                gradients[j] -= gradient
            # Update console output
            sys.stdout.write("\r" + str(objective))
            sys.stdout.flush()
            return objective, gradients.ravel()

        # Optimize camera view directions
        viewdirs_0 = [img.cam.viewdir for img in self.observer.images]
        result = scipy.optimize.minimize(
            fun=fun, x0=viewdirs_0, jac=True, method=method, **params
        )
        self.reset_cameras()
        if not result.success:
            sys.stdout.write("\n")  # new line
            print(result.message)
        return result


# ---- RANSAC ----

Model = Union[Polynomial, Cameras]


def ransac(
    model: Model,
    n: int,
    max_error: float,
    min_inliers: int,
    iterations: int = 100,
    **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit model parameters using the Random Sample Consensus (RANSAC) algorithm.
    
    Samples are drawn without replacement and are guaranteed to be non-repeating.
    See Schattschneider & Green 2012 (https://doi.org/10.1145/2425836.2425878).

    Arguments:
        model: Object with the following attributes and methods.

            - `size`: Maximum sample size.
            - `fit(index)`: Accepts sample indices and returns model parameters.
            - `errors(params, index)`: Accepts sample indices and model parameters and
                returns an error for each sample member.

        n: Size of sample used to fit the model in each iteration.
        max_error: Error below which a sample member is considered an inlier.
        min_inliers: Number of inliers (in addition to `n`) for a sample to be valid.
        iterations: Maximum number of different samples to fit.
        **kwargs: Additional arguments to `model.fit()`.

    Returns:
        Values of model parameters.
        Indices of model inliers.
    """
    params = None
    err = np.inf
    inliers = None
    full = np.arange(model.size)
    for maybe_idx in _ransac_samples(n=n, size=model.size, iterations=iterations):
        maybe_params = model.fit(maybe_idx, **kwargs)
        if maybe_params is None:
            continue
        test_idx = np.delete(full, maybe_idx)
        test_errs = model.errors(maybe_params, test_idx)
        also_idx = test_idx[test_errs < max_error]
        if len(also_idx) > min_inliers:
            better_idx = np.concatenate((maybe_idx, also_idx))
            better_params = model.fit(better_idx, **kwargs)
            if better_params is None:
                continue
            better_errs = model.errors(better_params, better_idx)
            this_err = np.mean(better_errs)
            if this_err < err:
                params = better_params
                err = this_err
                inliers = better_idx
    if params is None:
        raise ValueError("Best fit does not meet acceptance criteria")
    # HACK: Recompute inlier index on best params
    inliers = np.where(model.errors(params) <= max_error)[0]
    return params, inliers


def _ransac_samples(n: int, size: int, iterations: int = 100) -> List[int]:
    """
    Generate non-repeating combinations of indices for random samples.

    Arguments:
        n: Number of items in each sample.
        size: Number of items to sample from.
        iterations: Maximum number of iterations.

    Returns:
        Sample indices.

    Example:
        >>> sorted(_ransac_samples(n=2, size=4))
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    """
    if n >= size:
        raise ValueError("Sample size is larger or equal to total size")
    # Estimate factorial with log(gamma) to avoid overflow
    log = (
        np.math.lgamma(size + 1) - np.math.lgamma(n + 1) - np.math.lgamma(size - n + 1)
    )
    if log:
        # Compute max iterations if no float overflow
        max_iterations = np.floor(np.exp(log))
        iterations = min(iterations, max_iterations)
    samples = set()
    indices = np.arange(size)
    while len(samples) < iterations:
        np.random.shuffle(indices)
        sample = frozenset(indices[:n])
        if sample not in samples:
            yield list(sample)
            samples.add(sample)


# ---- Keypoints ----


def detect_keypoints(
    array: np.ndarray,
    mask: np.ndarray = None,
    method: str = "sift",
    root: bool = True,
    **kwargs: Any
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Return keypoints and descriptors for an image.

    Arguments:
        array: 2 or 3-dimensional image array (cast to uint8).
        mask: Pixel regions in which to detect keypoints (cast to uint8).
        method: The keypoint detection algorithm to use.

            - 'sift': Scale-invariant feature transform (SIFT) by
                Lowe 2004 (https://doi.org/10.1023/B:VISI.0000029664.99615.94.
                Uses :class:`cv2.xfeatures2d.SIFT`
                (https://docs.opencv.org/4.1.1/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html).
            - 'surf': Speeded-up robust features (SURF) by
                Bay et al. 2006 (https://doi.org/10.1016/j.cviu.2007.09.014).
                Uses :class:`cv2.xfeatures2d.SURF`
                (https://docs.opencv.org/4.1.1/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html).

        root: Whether to return square-root L1-normalized descriptors, as described by
             Arandjelović & Zisserman 2012 (https://doi.org/10.1109/CVPR.2012.6248018).
        **kwargs: Additional arguments passed to
            :class:`cv2.xfeatures2d.SIFT` or :class:`cv2.xfeatures2d.SURF`.

    Returns:
        Keypoints as :class:`cv2.KeyPoint`.
        Keypoint descriptors as array rows.
    """
    array = np.asarray(array, dtype=np.uint8)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.uint8)
    if method == "sift":
        try:
            detector = cv2.xfeatures2d.SIFT_create(**kwargs)
        except AttributeError:
            detector = cv2.SIFT(**kwargs)
    elif method == "surf":
        try:
            detector = cv2.xfeatures2d.SURF_create(**kwargs)
        except AttributeError:
            detector = cv2.SURF(**kwargs)
    keypoints, descriptors = detector.detectAndCompute(array, mask=mask)
    # Empty result: ([], None)
    if root and descriptors is not None:
        descriptors = np.sqrt(
            descriptors / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        )
    return keypoints, descriptors


def match_keypoints(
    ka: Tuple[List[cv2.KeyPoint], np.ndarray],
    kb: Tuple[List[cv2.KeyPoint], np.ndarray],
    mask: np.ndarray = None,
    cross_check: bool = False,
    max_ratio: float = None,
    max_distance: float = None,
    return_ratios: bool = False,
    matcher: cv2.DescriptorMatcher = cv2.FlannBasedMatcher(),
):
    """
    Return the image coordinates of matched keypoint pairs.

    Arguments:
        ka: Keypoints of the first image (keypoints, descriptors).
        kb: Keypoints of the second image (keypoints, descriptors).
        mask: Pixel regions in which to retain keypoints (cast to uint8).
        cross_check: Whether to only return matches for which the keypoint from
            `kb` is the best match among `kb` for the keypoint in `ka`,
            and vice versa.
        max_ratio: Maximum descriptor-distance ratio between the best and
            second best match.
            See Lowe 2004 (http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20).
        max_distance: Maximum coordinate-distance of matched keypoints.
        return_ratios: Whether to return the ratio of each match.
        matcher: Keypoint descriptor matcher to use for matching
            (https://docs.opencv.org/4.1.1/db/d39/classcv_1_1DescriptorMatcher.html).
            Default is to use :class:`cv2.FlannBasedMatcher`
            (https://docs.opencv.org/4.1.1/dc/de2/classcv_1_1FlannBasedMatcher.html).

    Returns:
        Coordinates of matches in first image (n, [ua, va]).
        Coordinates of matches in second image (n, [ub, vb]).
        (optional) Ratio of each match (n, ).
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=np.uint8)
    compute_ratios = max_ratio or return_ratios
    n = 2 if compute_ratios else 1
    if len(ka[0]) >= n and len(kb[0]) >= n:
        matches = matcher.knnMatch(ka[1], kb[1], k=n, mask=mask)
        if cross_check:
            matches_ba = matcher.knnMatch(kb[1], ka[1], k=n, mask=mask)
            ba = [(m[0].trainIdx, m[0].queryIdx) for m in matches_ba]
            matches = [m for m in matches if (m[0].queryIdx, m[0].trainIdx) in ba]
        if max_ratio:
            matches = [m for m in matches if m[0].distance / m[1].distance < max_ratio]
        uva = np.asarray([ka[0][m[0].queryIdx].pt for m in matches])
        uvb = np.asarray([kb[0][m[0].trainIdx].pt for m in matches])
        if return_ratios:
            ratios = np.array([m.distance / n.distance for m, n in matches])
        if max_distance:
            valid = np.linalg.norm(uva - uvb, axis=1) < max_distance
            uva, uvb = uva[valid], uvb[valid]
            if return_ratios:
                ratios = ratios[valid]
    else:
        # Not enough keypoints to match
        empty = np.array([], dtype=float).reshape(0, 2)
        uva, uvb = empty, empty.copy()
        ratios = np.array([], dtype=float)
    if return_ratios:
        return uva, uvb, ratios
    return uva, uvb


class KeypointMatcher(object):
    """
    `KeypointMatcher` detects and matches image keypoints.

    - Build (and save to file) keypoint descriptors for each image with
        `self.build_keypoints()`.
    - Build (and save to file) keypoints matches between image pairs with
        `self.build_matches()`.

    Arguments:
        clahe: Arguments to `cv2.createCLAHE()` (dict: clipLimit, tileGridSize)
            or whether to use CLAHE (bool).

    Attributes:
        images (array): Image objects in ascending temporal order
        clahe (cv2.CLAHE): CLAHE object
        matches (scipy.sparse.coo.coo_matrix): Sparse matrix of image-image `Matches`
        keypoints (list): List of image keypoints
            (see `optimize.detect_keypoints()`).
    """

    def __init__(self, images, clahe=False):
        dts = np.diff([img.datetime for img in images])
        if np.any(dts < datetime.timedelta(0)):
            raise ValueError("Images are not in ascending temporal order")
        self.images = np.asarray(images)
        if clahe is False:
            self.clahe = None
        else:
            if clahe is True:
                clahe = {}
            self.clahe = cv2.createCLAHE(**clahe)
        # Placeholders
        self.keypoints = None
        self.matches = None

    def _prepare_image_basenames(self):
        basenames = [helpers.strip_path(img.path) for img in self.images]
        if len(basenames) != len(set(basenames)):
            raise ValueError("Image basenames are not unique")
        return basenames

    def _prepare_image(self, I):
        """
        Prepare image data for keypoint detection.
        """
        if I.ndim > 2:
            I = rgb.mean(axis=2)
        I = I.astype(np.uint8, copy=False)
        if self.clahe is not None:
            I = self.clahe.apply(I)
        return I

    def build_keypoints(
        self,
        masks=None,
        path=None,
        overwrite=False,
        clear_images=True,
        clear_keypoints=False,
        parallel=False,
        **params
    ):
        """
        Build image keypoints and their descriptors.

        Results are stored in `self.keypoints` and/or written to a binary
        `pickle` file with name `basename.pkl`.

        Arguments:
            masks (iterable): Boolean array(s) (uint8) indicating regions in which to
                detect keypoints
            path (str): Directory path for keypoint files.
                If `None`, no files are written.
            overwrite (bool): Whether to recompute and overwrite existing file or
                in-memory keypoints.
            clear_images (bool): Whether to clear cached image data
                (`self.images[i].I`).
            clear_keypoints (bool): Whether to clear cached keypoints
                (`self.keypoints[i]`).
            parallel: Number of image keypoints to detect in parallel (int),
                or whether to detect in parallel (bool). If `True`,
                defaults to `os.cpu_count()`.
            **params: Additional arguments to `optimize.detect_keypoints()`
        """
        if clear_keypoints and not path:
            raise ValueError("path is required when clear_keypoints is True")
        if path and os.path.isfile(path):
            raise ValueError("path must be a directory")
        basenames = self._prepare_image_basenames()
        # Enforce defaults
        if masks is None or isinstance(masks, np.ndarray):
            masks = (masks,) * len(self.images)
        parallel = helpers._parse_parallel(parallel)
        if not self.keypoints:
            self.keypoints = [None] * len(self.images)

        # Define parallel process
        def process(i, img):
            print(img.path)
            if path:
                outpath = os.path.join(path, basenames[i] + ".pkl")
                written = os.path.isfile(outpath)
            else:
                written = False
            keypoints = self.keypoints[i]
            read = keypoints is not None
            if not read and written and not clear_keypoints:
                # Read keypoints from file
                keypoints = helpers.read_pickle(outpath)
            elif read and not written and path:
                # Write keypoints to file
                helpers.write_pickle(keypoints, path=outpath)
            elif (not read and not written) or overwrite:
                # Detect keypoints
                I = self._prepare_image(img.read())
                keypoints = detect_keypoints(I, mask=masks[i], **params)
                if path:
                    # Write keypoints to file
                    helpers.write_pickle(keypoints, path=outpath)
                if clear_images:
                    img.I = None
            if clear_keypoints:
                keypoints = None
            return keypoints

        # Run process in parallel
        with config.backend(np=parallel) as pool:
            self.keypoints = pool.map(
                func=process, sequence=tuple(enumerate(self.images)), star=True
            )

    def build_matches(
        self,
        maxdt=None,
        seq=None,
        imgs=None,
        keypoints_path=None,
        path=None,
        overwrite=False,
        clear_keypoints=True,
        clear_matches=False,
        parallel=False,
        weights=False,
        as_type=None,
        filter=None,
        **params
    ):
        """
        Build matches between each image and its nearest neighbors.

        Results are stored in `self.matches` as an (n, n) upper-triangular sparse matrix
        of `Matches`, and the result for each `Image` pair (i, j) optionally written to
        a binary `pickle` file with name `basenames[i]-basenames[j].pkl`. If
        `clear_matches` is `True`, missing files are written but results are not stored
        in memory.

        Arguments:
            maxdt (`datetime.timedelta`): Maximum time separation between
                pairs of images to match.
                If `None` and `seq` is `None`, all pairs are matched.
            seq (iterable): Positive index of neighbors to match to each image
                (relative to 0) in addition to `maxdt`.
            imgs (iterable): Index of images to require at least one of
                in each matched image pair. If `None`, all image pairs meeting
                the criteria are matched.
            keypoints_path (str): Directory with keypoint files.
            path (str): Directory for match files. If `None`, no files are written.
            overwrite (bool): Whether to recompute and overwrite existing match files
            clear_keypoints (bool): Whether to clear cached keypoints
                (`self.keypoints`)
            clear_matches (bool): Whether to clear matches rather than return them
                (requires `path`). Useful for avoiding memory overruns when
                processing very large image sets.
            parallel: Number of image keypoints to detect in parallel (int),
                or whether to detect in parallel (bool). If `True`,
                defaults to `os.cpu_count()`.
            weights (bool): Whether to include weights in `Matches` objects,
                computed as the inverse of the maximum descriptor-distance ratio
            filter (dict): Arguments to `optimize.Matches.filter()`.
                If truthy, `Matches` are filtered before being saved to `self.matches`.
                Ignored if `clear_matches=True`.
            **params: Additional arguments to `optimize.match_keypoints()`
        """
        if clear_matches and not path:
            raise ValueError("path is required when clear_keypoints is True")
        if path and os.path.isfile(path):
            raise ValueError("path must be a directory")
        parallel = helpers._parse_parallel(parallel)
        params = {**params, **{"return_ratios": weights}}
        basenames = self._prepare_image_basenames()
        if self.keypoints is None:
            self.keypoints = [None] * len(self.images)
        # Match images
        n = len(self.images)
        if maxdt is None and seq is None:
            matching_images = [np.arange(i + 1, n) for i in range(n)]
        elif maxdt is not None:
            datetimes = np.array([img.datetime for img in self.images])
            ends = np.searchsorted(datetimes, datetimes + maxdt, side="right")
            matching_images = [np.arange(i + 1, end) for i, end in enumerate(ends)]
        elif seq is not None:
            matching_images = [np.array([], dtype=int) for i in range(n)]
        # Add match sequence
        if seq is not None:
            seq = np.asarray(seq)
            seq = np.unique(seq[seq > 0])
            for i, m in enumerate(matching_images):
                iseq = seq + i
                iseq = iseq[: np.searchsorted(iseq, n)]
                matching_images[i] = np.unique(np.concatenate((m, iseq)))
        # Filter matched image pairs
        if imgs is not None:
            for i, m in enumerate(matching_images):
                matching_images[i] = m[np.isin(m, imgs)]

        # Define parallel process
        def process(i, js):
            if len(js) > 0:
                print("Matching", i, "->", ", ".join(js.astype(str)))
            matches = []
            imgA = self.images[i]
            if self.keypoints[i] is None and keypoints_path:
                self.keypoints[i] = helpers.read_pickle(
                    os.path.join(keypoints_path, basenames[i] + ".pkl")
                )
            for j in js:
                imgB = self.images[j]
                if self.keypoints[j] is None and keypoints_path:
                    self.keypoints[j] = helpers.read_pickle(
                        os.path.join(keypoints_path, basenames[j] + ".pkl")
                    )
                if path:
                    outfile = os.path.join(
                        path, basenames[i] + "-" + basenames[j] + ".pkl"
                    )
                if path and not overwrite and os.path.exists(outfile):
                    if not clear_matches:
                        match = helpers.read_pickle(outfile)
                        # Point matches to existing Camera objects
                        match.cams = (imgA.cam, imgB.cam)
                        if as_type:
                            match = match.as_type(as_type)
                        matches.append(match)
                else:
                    result = match_keypoints(
                        self.keypoints[i], self.keypoints[j], **params
                    )
                    match = Matches(
                        cams=(imgA.cam, imgB.cam),
                        uvs=result[0:2],
                        weights=(1 / result[2]) if weights else None,
                    )
                    if path is not None:
                        helpers.write_pickle(match, outfile)
                    if not clear_matches:
                        if as_type:
                            match = match.as_type(as_type)
                        matches.append(match)
            if clear_keypoints:
                self.keypoints[i] = None
            return None if clear_matches else matches

        def reduce(matches):
            if filter:
                for match in matches:
                    if match:
                        match.filter(**filter)
            return matches

        # Run process in parallel
        with config.backend(np=parallel) as pool:
            matches = pool.map(
                func=process,
                reduce=reduce,
                star=True,
                sequence=tuple(enumerate(matching_images)),
            )
            if not clear_matches:
                # Build Compressed Sparse Row (CSR) matrix
                matches = scipy.sparse.csr_matrix(
                    (
                        np.concatenate(matches),  # data
                        np.concatenate(matching_images),  # column indices
                        np.cumsum([0] + [len(row) for row in matching_images]),
                    )
                )  # row ranges
                # Convert to Coordinate Format (COO) matrix
                matches = matches.tocoo()
        if clear_matches:
            self.matches = None
        else:
            self.matches = matches
            if parallel:
                self._assign_cameras()

    def _test_matches(self):
        if self.matches is None:
            raise ValueError("Matches have not been initialized. Run build_matches()")

    def _assign_cameras(self):
        for m, i, j in zip(self.matches.data, self.matches.row, self.matches.col):
            m.cams = self.images[i].cam, self.images[j].cam

    def convert_matches(self, mtype, clear_uvs=False, parallel=False):
        self._test_matches()
        parallel = helpers._parse_parallel(parallel)

        def process(i, m):
            m = m.as_type(mtype)
            if clear_uvs and mtype in (RotationMatchesXY, RotationMatchesXYZ):
                m.uvs = None
            return i, m

        def reduce(i, m):
            self.matches.data[i] = m

        with config.backend(np=parallel) as pool:
            _ = pool.map(
                func=process,
                reduce=reduce,
                star=True,
                sequence=tuple(zip(range(self.matches.data.size), self.matches.data)),
            )
        if parallel:
            self._assign_cameras()

    def filter_matches(self, clear_weights=False, parallel=False, **params):
        self._test_matches()
        parallel = helpers._parse_parallel(parallel)

        def process(i, m):
            if params:
                m.filter(**params)
            if clear_weights:
                m.weights = None
            return i, m

        def reduce(i, m):
            self.matches.data[i] = m

        with config.backend(np=parallel) as pool:
            _ = pool.map(
                func=process,
                reduce=reduce,
                star=True,
                sequence=tuple(zip(range(self.matches.data.size), self.matches.data)),
            )
        if parallel:
            self._assign_cameras()

    def _images_mask(self, imgs):
        if np.iterable(imgs):
            return np.isin(self.matches.row, imgs) | np.isin(self.matches.col, imgs)
        else:
            return (self.matches.row == imgs) | (self.matches.col == imgs)

    def _images_matches(self, imgs):
        mask = self._images_mask(imgs)
        return self.matches.data[mask]

    def matches_per_image(self):
        self._test_matches()
        image_matches = [self._images_matches(i) for i in range(len(self.images))]
        # n_images = np.array([
        #   np.sum([mi.size > 0 for mi in m]) for m in image_matches
        # ])
        return np.array([np.sum([mi.size for mi in m]) for m in image_matches])

    def images_per_image(self):
        self._test_matches()
        image_matches = [self._images_matches(i) for i in range(len(self.images))]
        return np.array([np.sum([mi.size > 0 for mi in m]) for m in image_matches])

    def drop_images(self, imgs):
        self._test_matches()
        mask = self._images_mask(imgs)
        self.matches.data[mask] = False
        self.matches.eliminate_zeros()
        # Find all images with no matches
        all = np.arange(len(self.images))
        keep = np.union1d(self.matches.row, self.matches.col)
        drop = np.setdiff1d(all, keep)
        # Remove row and column of each dropped image
        _, new_row = np.unique(
            np.concatenate((self.matches.row, keep)), return_inverse=True
        )
        self.matches.row = new_row[: -len(keep)]
        _, new_col = np.unique(
            np.concatenate((self.matches.col, keep)), return_inverse=True
        )
        self.matches.col = new_col[: -len(keep)]
        # Resize matches matrix
        n = len(self.images) - len(drop)
        self.matches._shape = (n, n)
        # Remove dropped images
        self.images = np.delete(self.images, drop)

    def match_breaks(self, min_matches=0):
        self._test_matches()
        all_starts = np.arange(len(self.images) - 1)
        starts, counts = np.unique(self.matches.row, return_counts=True)
        breaks = np.setdiff1d(all_starts, starts)
        if min_matches:
            min_matches = np.minimum(
                min_matches, len(self.images) - np.arange(len(self.images))
            )
            breaks = np.sort(
                np.concatenate((breaks, np.where(counts < min_matches)[0]))
            )
        return breaks
