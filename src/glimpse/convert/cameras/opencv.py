"""OpenCV camera models."""
import re
import warnings
import xml.etree.ElementTree
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, cast

import numpy as np

from ...camera import Camera
from ..converter import Converter

Parameters = Dict[str, Union[bool, int, Iterable[int]]]
Optimize = Union[bool, Parameters]


class OpenCV:
    """
    Frame camera model used by OpenCV.

    See https://docs.opencv.org/master/d9/d0c/group__calib3d.html#details.
    Distortion coefficients τx and τy are not supported.

    Attributes:
        imgsz (tuple of int): Image size in pixels (nx, ny)
        fx (float): Focal length in pixels (x)
        fy (float): Focal length in pixels (y)
        cx (float): Principal point in pixels (x). Defaults to the image center.
        cy (float): Principal point in pixels (y). Defaults to the image center.
        k1 (float): Radial distortion coefficient #1
        k2 (float): Radial distortion coefficient #2
        k3 (float): Radial distortion coefficient #3
        k4 (float): Radial distortion coefficient #4
        k5 (float): Radial distortion coefficient #5
        k6 (float): Radial distortion coefficient #6
        p1 (float): Tangential distortion coefficient #1
        p2 (float): Tangential distortion coefficient #2
        s1 (float): Thin prism distortion coefficient #1
        s2 (float): Thin prism distortion coefficient #2
        s3 (float): Thin prism distortion coefficient #3
        s4 (float): Thin prism distortion coefficient #4
    """

    def __init__(
        self,
        imgsz: Tuple[int, int],
        fx: float,
        fy: float,
        cx: float = None,
        cy: float = None,
        k1: float = 0,
        k2: float = 0,
        k3: float = 0,
        k4: float = 0,
        k5: float = 0,
        k6: float = 0,
        p1: float = 0,
        p2: float = 0,
        s1: float = 0,
        s2: float = 0,
        s3: float = 0,
        s4: float = 0,
    ) -> None:
        self.imgsz = imgsz
        self.fx, self.fy = fx, fy
        self.cx = imgsz[0] / 2 if cx is None else cx
        self.cy = imgsz[1] / 2 if cy is None else cy
        self.k1, self.k2 = k1, k2
        self.p1, self.p2 = p1, p2
        self.k3, self.k4, self.k5, self.k6 = k3, k4, k5, k6
        self.s1, self.s2, self.s3, self.s4 = s1, s2, s3, s4

    @property
    def cameraMatrix(self) -> List[Tuple[float]]:
        """
        Camera matrix.

        Follows the OpenCV format [(fx 0 cx), (0 fy cy), (0 0 1)].
        """
        return [(self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)]

    @property
    def distCoeffs(self) -> List[float]:
        """
        Distortion coefficients vector.

        Follows the OpenCV format (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4).
        """
        return [
            self.k1,
            self.k2,
            self.p1,
            self.p2,
            self.k3,
            self.k4,
            self.k5,
            self.k6,
            self.s1,
            self.s2,
            self.s3,
            self.s4,
        ]

    @staticmethod
    def _parse_camera_matrix(x: Iterable[Iterable[float]]) -> Dict[str, float]:
        return {"fx": x[0][0], "fy": x[1][1], "cx": x[0][2], "cy": x[1][2]}

    @staticmethod
    def _parse_distortion_coefficients(x: Iterable[float]) -> Dict[str, float]:
        keys = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")
        if len(x) > len(keys):
            warnings.warn(
                f"Coefficients past {keys[-1]} are not supported and were ignored"
            )
            x = x[: len(keys)]
        return {keys[i]: xi for i, xi in enumerate(x)}

    @classmethod
    def from_arrays(
        cls,
        cameraMatrix: Iterable[Iterable[float]],
        distCoeffs: Iterable[float],
        imgsz: Tuple[int, int],
    ) -> "OpenCV":
        """
        From OpenCV camera matrix and distortion coefficients vector.

        Arguments:
            cameraMatrix: OpenCV camera matrix [(fx 0 cx), (0 fy cy), (0 0 1)].
            distCoeffs: OpenCV distortion coefficients, including all or a subset of
                (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, ...).
            imgsz: Image size in pixels (nx, ny).
        """
        kwargs = {
            **cls._parse_camera_matrix(cameraMatrix),
            **cls._parse_distortion_coefficients(distCoeffs),
        }
        return cls(imgsz=imgsz, **kwargs)

    @classmethod
    def from_xml(cls, path: Union[str, Path], imgsz: Tuple[int, int]) -> "OpenCV":
        """
        From Agisoft XML file.

        Arguments:
            path: Path or file object pointing to the XML file.
            imgsz: Image size in pixels (nx, ny).

        Raises:
            ValueError: No camera matrix found.
        """
        tree = xml.etree.ElementTree.parse(path)
        kwargs: Dict[str, Any] = {"imgsz": imgsz}
        elements = tree.findall(".//camera_matrix/data")
        if elements and elements[0].text:
            x = np.asarray(
                [float(xi) for xi in re.findall(r"([0-9\-\.e\+]+)", elements[0].text)]
            ).reshape(3, 3)
            kwargs = {**kwargs, **cls._parse_camera_matrix(x)}
        else:
            raise ValueError("No camera matrix found")
        elements = tree.findall(".//distortion_coefficients/data")
        if elements and elements[0].text:
            x = [float(xi) for xi in re.findall(r"([0-9\-\.e\+]+)", elements[0].text)]
            kwargs = {**kwargs, **cls._parse_distortion_coefficients(x)}
        return cls(**kwargs)

    @classmethod
    def _from_camera_initial(cls, cam: Camera) -> "OpenCV":
        return cls(
            imgsz=(cam.imgsz[0], cam.imgsz[1]),
            fx=cam.f[0],
            fy=cam.f[1],
            cx=cam.c[0] + cam.imgsz[0] / 2,
            cy=cam.c[1] + cam.imgsz[1] / 2,
            k1=cam.k[0],
            k2=cam.k[1],
            k3=cam.k[2],
            k4=cam.k[3],
            k5=cam.k[4],
            k6=cam.k[5],
            p1=cam.p[0],
            p2=cam.p[1],
        )

    @classmethod
    def from_camera(cls, cam: Camera) -> "OpenCV":
        """
        Convert from :class:`~glimpse.Camera` object.

        Since the OpenCV camera model is a superset of the :class:`~glimpse.Camera`
        model, the conversion is exact and no optimization is needed.

        Arguments:
            cam: :class:`~glimpse.Camera` object.
        """
        return cls._from_camera_initial(cam)

    def _xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        # Compute lens distortion
        r2 = np.sum(xy ** 2, axis=1)
        dr = (1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3) / (
            1 + self.k4 * r2 + self.k5 * r2 ** 2 + self.k6 * r2 ** 3
        )
        xty = xy[:, 0] * xy[:, 1]
        dtx = self.p2 * (r2 + 2 * xy[:, 0] ** 2) + 2 * self.p1 * xty
        dty = self.p1 * (r2 + 2 * xy[:, 1] ** 2) + 2 * self.p2 * xty
        # Apply lens distortion
        dxy = np.column_stack(
            (
                dr * xy[:, 0] + dtx + self.s1 * r2 + self.s2 * r2 ** 2,
                dr * xy[:, 1] + dty + self.s3 * r2 + self.s4 * r2 ** 2,
            )
        )
        # Project to image
        return np.column_stack(
            ((self.fx * dxy[:, 0] + self.cx), (self.fy * dxy[:, 1] + self.cy))
        )

    def _to_camera_initial(self) -> Camera:
        return Camera(
            imgsz=self.imgsz,
            f=(self.fx, self.fy),
            c=(self.cx - self.imgsz[0] / 2, self.cy - self.imgsz[1] / 2),
            k=(self.k1, self.k2, self.k3, self.k4, self.k5, self.k6),
            p=(self.p1, self.p2),
        )

    def to_camera(
        self,
        optimize: Optimize = True,
        uv: Union[np.ndarray, int] = 1000,
        **kwargs: Any,
    ) -> Camera:
        """
        Convert to :class:`Camera` object.

        Arguments:
            optimize: Whether and which :class:`Camera` parameters to optimize to
                minimize the residuals between the cameras. If :attr:`s1`, :attr:`s2`,
                :attr:`s3`, and :attr:`s4` are zero, the conversion is exact and
                no optimization is performed. Otherwise:

                    - If `True`, optimizes :attr:`~glimpse.Camera.k` and
                      :attr:`~glimpse.Camera.p`.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                      format as :meth:`Converter.optimize_cam`.

            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to :func:`scipy.optimize.least_squares`.

        Returns:
            Exact or best-fitting Camera object.
        """
        cam = self._to_camera_initial()
        if not optimize or not any((self.s1, self.s2, self.s3, self.s4)):
            return cam
        if optimize is True:
            optimize = {"k": True, "p": True}
        converter = Converter(xcam=self, cam=cam, uv=uv)
        converter.optimize_cam(params=cast(Parameters, optimize), **kwargs)
        return converter.cam
