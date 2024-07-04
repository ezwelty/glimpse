"""MATLAB camera models."""
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union, cast

import numpy as np

from ...camera import Camera
from ..converter import Converter

Parameters = Dict[str, Union[bool, int, Iterable[int]]]
Optimize = Union[bool, Parameters]


class Matlab:
    """
    Camera model used by the Camera Calibration Toolbox for MATLAB.

    See http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html.

    Attributes:
        imgsz (tuple of int): Image size in pixels (nx, ny)
        fc (tuple of float): Focal length in pixels (x, y)
        cc (tuple of float): Principal point in pixels (x, y), in an image coordinate
            system where the center of the top left pixel is (0, 0)
        kc (tuple of float): Image distortion coefficients (k1, k2, p1, p2, k3)
        alpha_c (float): Skew coefficient defining the angle between the x and y
            pixel axes
    """

    def __init__(
        self,
        imgsz: Tuple[int, int],
        fc: Tuple[float, float],
        cc: Tuple[float, float] = None,
        kc: Tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
        alpha_c: float = 0,
    ) -> None:
        self.imgsz = imgsz
        self.fc = fc
        if cc is None:
            cc = (imgsz[0] - 1) / 2, (imgsz[1] - 1) / 2
        self.cc = cc
        self.kc = kc
        self.alpha_c = alpha_c

    @classmethod
    def from_report(cls, path: Union[str, Path], sigmas: bool = False) -> "Matlab":
        """
        Read from calibration report (Calib_Result.m).

        Arguments:
            path: Path to report
            sigmas: Whether to read parameter means (False)
                or standard deviations (True)
        """
        with open(path, mode="r") as fp:
            txt = fp.read()

        def parse(key: str, length: int = 1) -> Tuple[float, ...]:
            if length == 1:
                pattern = fr"{key} = (.*);"
            else:
                groups = " ; ".join(["(.*)"] * length)
                pattern = fr"{key} = \[ {groups} \];"
            values = re.findall(pattern, txt)
            if length > 1:
                # Ensure result is always a sequence of strings
                values = values[0]
            # Error bounds are ~3 times standard deviations
            scale = 1 / 3 if sigmas else 1
            return tuple(float(x) * scale for x in values)

        return cls(
            imgsz=(0, 0) if sigmas else (int(parse("nx")[0]), int(parse("ny")[0])),
            fc=cast(Tuple[float, float], parse("fc_error" if sigmas else "fc", 2)),
            cc=cast(Tuple[float, float], parse("cc_error" if sigmas else "cc", 2)),
            kc=cast(
                Tuple[float, float, float, float, float],
                parse("kc_error" if sigmas else "kc", 5),
            ),
            alpha_c=parse("alpha_c_error" if sigmas else "alpha_c")[0],
        )

    @classmethod
    def _from_camera_initial(cls, cam: Camera) -> "Matlab":
        return cls(
            imgsz=(cam.imgsz[0], cam.imgsz[1]),
            fc=(cam.f[0], cam.f[1]),
            cc=(
                (cam.c[0] + 0.5 * cam.imgsz[0]) - 0.5,
                (cam.c[1] + 0.5 * cam.imgsz[1]) - 0.5,
            ),
            kc=(cam.k[0], cam.k[1], cam.p[0], cam.p[1], cam.k[2]),
        )

    @classmethod
    def from_camera(
        cls,
        cam: Camera,
        optimize: Optimize = True,
        uv: Union[np.ndarray, int] = 1000,
        **kwargs: Any,
    ) -> "Matlab":
        """
        Convert from :class:`~glimpse.Camera` object.

        Arguments:
            cam: Camera object.
            optimize: Whether and which :class:`Matlab` parameters to optimize to
                minimize the residuals between the cameras. If `cam.k[3:6]` are zero,
                the conversion is exact and no optimization is performed. Otherwise:

                    - If `True`, optimizes :attr:`kc`.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                      format as :meth:`Converter.optimize_cam`.

            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to :func:`scipy.optimize.least_squares`.
        """
        xcam = cls._from_camera_initial(cam)
        if not optimize or (cam.k[3:6] == 0).all():
            return xcam
        if optimize is True:
            optimize = {"kc": True}
        converter = Converter(xcam=xcam, cam=cam, uv=uv)
        converter.optimize_xcam(params=cast(Parameters, optimize), **kwargs)
        return cast("Matlab", converter.xcam)

    def _xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        # Compute lens distortion
        r2 = np.sum(xy ** 2, axis=1)
        dr = self.kc[0] * r2 + self.kc[1] * r2 ** 2 + self.kc[4] * r2 ** 3
        xty = xy[:, 0] * xy[:, 1]
        dtx = 2 * self.kc[2] * xty + self.kc[3] * (r2 + 2 * xy[:, 0] ** 2)
        dty = self.kc[2] * (r2 + 2 * xy[:, 1] ** 2) + 2 * self.kc[3] * xty
        # Apply lens distortion
        dxy = xy.copy()
        dxy[:, 0] += dxy[:, 0] * dr + dtx
        dxy[:, 1] += dxy[:, 1] * dr + dty
        # Project to image
        uv = np.column_stack(
            (
                self.fc[0] * (dxy[:, 0] + self.alpha_c * dxy[:, 1]) + self.cc[0],
                self.fc[1] * dxy[:, 1] + self.cc[1],
            )
        )
        # Top left corner of top left pixel is (-0.5, -0.5)
        uv += (0.5, 0.5)
        return uv

    def _to_camera_initial(self) -> Camera:
        return Camera(
            imgsz=self.imgsz,
            f=self.fc,
            c=(
                (self.cc[0] + 0.5) - self.imgsz[0] / 2,
                (self.cc[1] + 0.5) - self.imgsz[1] / 2,
            ),
            k=(self.kc[0], self.kc[1], self.kc[4]),
            p=(self.kc[2], self.kc[3]),
        )

    def to_camera(
        self,
        optimize: Optimize = True,
        uv: Union[np.ndarray, int] = 1000,
        **kwargs: Any,
    ) -> Camera:
        """
        Convert to :class:`~glimpse.Camera` object.

        Arguments:
            optimize: Whether and which :class:`Camera` parameters to optimize to
                minimize the residuals between the cameras. If :attr:`alpha_c` is zero,
                the conversion is exact and no optimization is performed. Otherwise:

                    - If `True`, optimizes :attr:`~glimpse.Camera.f`,
                      :attr:`~glimpse.Camera.c`, :attr:`~glimpse.Camera.k`,
                      and :attr:`~glimpse.Camera.p`.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                      format as :meth:`Converter.optimize_cam`.

            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to :func:`scipy.optimize.least_squares`.

        Returns:
            Exact or best-fitting camera object.
        """
        cam = self._to_camera_initial()
        if not optimize or not self.alpha_c:
            return cam
        if optimize is True:
            optimize = {"f": True, "c": True, "k": True, "p": True}
        converter = Converter(xcam=self, cam=cam, uv=uv)
        converter.optimize_cam(params=cast(Parameters, optimize), **kwargs)
        return converter.cam
