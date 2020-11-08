"""PhotoModeler camera models."""
import re
from typing import Any, cast, Dict, Iterable, Tuple, Union

import numpy as np

from ..converter import Converter
from ...camera import Camera

Parameters = Dict[str, Union[bool, int, Iterable[int]]]
Optimize = Union[bool, Parameters]


class PhotoModeler:
    """
    Camera model used by EOS Systems PhotoModeler.

    See "Lens Distortion Formulation" in the software help.

    Attributes:
        imgsz (tuple of int): Image size in pixels (nx, ny)
        focal (float): Focal length in mm
        xp (float): Principal point in mm (x)
        yp (float): Principal point in mm (y)
        fw (float): Format (sensor) width in mm (x)
        fh (float): Format (sensor) height in mm (y)
        k1 (float): Radial distortion coefficient #1
        k2 (float): Radial distortion coefficient #2
        k3 (float): Radial distortion coefficient #3
        p1 (float): Decentering distortion coefficient #1
        p2 (float): Decentering distortion coefficient #2
    """

    def __init__(
        self,
        imgsz: Tuple[int, int],
        focal: float,
        xp: float,
        yp: float,
        fw: float,
        fh: float,
        k1: float = 0,
        k2: float = 0,
        k3: float = 0,
        p1: float = 0,
        p2: float = 0,
    ) -> None:
        self.imgsz = imgsz
        self.focal = focal
        self.xp, self.yp = xp, yp
        self.fw, self.fh = fw, fh
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.p1, self.p2 = p1, p2

    @classmethod
    def from_report(
        cls, path: str, imgsz: Tuple[int, int], sigmas: bool = False
    ) -> "PhotoModeler":
        """
        Read from camera calibration project report.

        Arguments:
            path: Path to calibration report.
            imgsz: Image size in pixels (nx, ny).
            sigmas: Whether to read parameter means (False) or
                standard deviations (True).
        """
        params = {
            "focal": "Focal Length",
            "xp": "Xp",
            "yp": "Yp",
            "fw": "Fw",
            "fh": "Fh",
            "k1": "K1",
            "k2": "K2",
            "k3": "K3",
            "p1": "P1",
            "p2": "P2",
        }
        with open(path, mode="r") as fp:
            txt = fp.read()
        if sigmas:
            pattern = r".*\s.*\s*Deviation: .*: ([0-9\-\+\.e]+)"
        else:
            pattern = r".*\s*Value: ([0-9\-\+\.e]+)"
        matches = [re.findall(label + pattern, txt) for label in params.values()]
        kwargs = {k: float(v[0]) if v else 0.0 for k, v in zip(params.keys(), matches)}
        return cls(imgsz=imgsz, **kwargs)

    @classmethod
    def _from_camera_initial(cls, cam: Camera) -> "PhotoModeler":
        if cam.sensorsz is None:
            raise ValueError("Camera does not have a sensor size")
        return cls(
            imgsz=(cam.imgsz[0], cam.imgsz[1]),
            focal=(cam.fmm[0] + cam.fmm[1]) / 2,  # type: ignore
            xp=cam.cmm[0] + cam.sensorsz[0] / 2,  # type: ignore
            yp=cam.cmm[1] + cam.sensorsz[1] / 2,  # type: ignore
            fw=cam.sensorsz[0],
            fh=cam.sensorsz[1],
        )

    @classmethod
    def from_camera(
        cls,
        cam: Camera,
        optimize: Optimize = True,
        uv: Union[np.ndarray, int] = 1000,
        **kwargs: Any,
    ) -> "PhotoModeler":
        """
        Convert from :class:`~glimpse.Camera` object.

        Arguments:
            cam: Camera object with `cam.sensorsz`.
            optimize: Whether and which :class:`PhotoModeler` parameters to
                optimize to minimize the residuals between the cameras. If `cam.fmm`
                are equal and all `cam.k` and `cam.p` are zero,
                the conversion is exact and no optimization is performed. Otherwise:

                    - If `True`, optimizes :attr:`focal`, :attr:`fw`, :attr:`fh`,
                      :attr:`xp`, and :attr:`yp` if `cam.fmm` are not equal,
                      :attr:`k1`, :attr:`k2`, and :attr:`k3` if any `cam.k`
                      are non-zero,
                      and :attr:`p1` and :attr:`p2` if any `cam.p` are non-zero.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                      format as :meth:`Converter.optimize_cam`.

            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to :func:`scipy.optimize.least_squares`.

        Raises:
            ValueError: Camera does not have a sensor size.
        """
        xcam = cls._from_camera_initial(cam)
        if not optimize or (
            cam.fmm[0] == cam.fmm[1]  # type: ignore
            and np.all(cam.k == 0)
            and np.all(cam.p == 0)
        ):
            return xcam
        if optimize is True:
            optimize = {}
            if cam.fmm[0] != cam.fmm[1]:  # type: ignore
                optimize = {
                    "focal": True,
                    "fw": True,
                    "fh": True,
                    "xp": True,
                    "yp": True,
                }
            if np.any(cam.k != 0):
                optimize = {**optimize, "k1": True, "k2": True, "k3": True}
            if np.any(cam.p != 0):
                optimize = {**optimize, "p1": True, "p2": True}
        converter = Converter(xcam=xcam, cam=cam, uv=uv)
        converter.optimize_xcam(params=cast(Parameters, optimize), **kwargs)
        return cast("PhotoModeler", converter.xcam)

    def _uv_to_xy(self, uv: np.ndarray) -> np.ndarray:
        # Convert image coordinates to mm relative to principal point
        xy = np.column_stack(
            (
                uv[:, 0] * self.fw / self.imgsz[0] - self.xp,
                uv[:, 1] * self.fh / self.imgsz[1] - self.yp,
            )
        )
        # Flip y (+y is down in image, but up in PM "photo space")
        xy[:, 1] *= -1
        # Remove lens distortion
        r2 = np.sum(xy ** 2, axis=1)
        dr = self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
        xty = xy[:, 0] * xy[:, 1]
        # NOTE: p1 and p2 are reversed
        dtx = self.p1 * (r2 + 2 * xy[:, 0] ** 2) + 2 * self.p2 * xty
        dty = self.p2 * (r2 + 2 * xy[:, 1] ** 2) + 2 * self.p1 * xty
        xy[:, 0] += xy[:, 0] * dr + dtx
        xy[:, 1] += xy[:, 1] * dr + dty
        # Flip y back
        xy[:, 1] *= -1
        # Normalize
        xy *= 1 / self.focal
        return xy

    def _to_camera_initial(self) -> Camera:
        return Camera(
            imgsz=self.imgsz,
            sensorsz=(self.fw, self.fh),
            fmm=self.focal,
            cmm=(self.xp - self.fw / 2, self.yp - self.fh / 2),
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
            optimize: Whether and which :class:`~glimpse.Camera` parameters to optimize
                to minimize the residuals between the cameras.
                If :attr:`k1`, :attr:`k2`, :attr:`k3`, :attr:`p1`, and :attr:`p2` is
                zero, the conversion is exact and no optimization is performed.
                Otherwise:

                    - If `True`, optimizes :attr:`~glimpse.Camera.k` if :attr:`k1`,
                      :attr:`k2`, or :attr:`k3` is non-zero,
                      and `~glimpse.Camera.p` if :attr:`p1` or :attr:`p2` is non-zero.
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
        k = self.k1, self.k2, self.k3
        p = self.p1, self.p2
        if not optimize or not any(k + p):
            return cam
        if optimize is True:
            optimize = {}
            if any(k):
                optimize["k"] = True
            if any(p):
                optimize["p"] = True
        converter = Converter(xcam=self, cam=cam, uv=uv)
        converter.optimize_cam(params=cast(Parameters, optimize), **kwargs)
        return converter.cam
