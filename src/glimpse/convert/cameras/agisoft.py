"""Agisoft camera models."""
import xml.etree.ElementTree
from typing import Any, Dict, Iterable, Tuple, Union, cast

import numpy as np

from ...camera import Camera
from ..converter import Converter

Parameters = Dict[str, Union[bool, int, Iterable[int]]]
Optimize = Union[bool, Parameters]


class Agisoft:
    """
    Frame camera model used by Agisoft software (PhotoScan, Metashape, Lens).

    See https://www.agisoft.com/pdf/metashape-pro_1_6_en.pdf (Appendix C).

    Attributes:
        imgsz (tuple of int): Image size in pixels (width, height)
        f (float): Focal length in pixels
        cx (float): Principal point offset in pixels (x)
        cy (float): Principal point offset in pixels (y)
        k1 (float): Radial distortion coefficient #1
        k2 (float): Radial distortion coefficient #2
        k3 (float): Radial distortion coefficient #3
        k4 (float): Radial distortion coefficient #4
        p1 (float): Tangential distortion coefficient #1
        p2 (float): Tangential distortion coefficient #2
        b1 (float): Affinity coefficient
        b2 (float): Non-orthogonality (skew) coefficient
    """

    def __init__(
        self,
        imgsz: Tuple[int, int],
        f: float,
        cx: float = 0,
        cy: float = 0,
        k1: float = 0,
        k2: float = 0,
        k3: float = 0,
        k4: float = 0,
        p1: float = 0,
        p2: float = 0,
        b1: float = 0,
        b2: float = 0,
    ) -> None:
        self.imgsz = imgsz
        self.f = f
        self.cx, self.cy = cx, cy
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.p1, self.p2 = p1, p2
        self.b1, self.b2 = b1, b2

    @classmethod
    def from_xml(cls, path: str) -> "Agisoft":
        """
        Read from Agisoft XML file.

        Arguments:
            path: Path or file object pointing to the XML file.

        Raises:
            ValueError: No <calibration> element found.
            ValueError: Unsupported camera model type.
        """
        tree = xml.etree.ElementTree.parse(path)
        calibration = next((e for e in tree.iter("calibration")), None)
        if not calibration:
            raise ValueError("No <calibration> element found")
        kwargs: Dict[str, Any] = {}
        for child in calibration:
            if child.tag == "projection" and child.text != "frame":
                raise ValueError(f"Unsupported camera model type: {child.text}")
            if child.text and child.tag in (
                "width",
                "height",
                "f",
                "cx",
                "cy",
                "k1",
                "k2",
                "k3",
                "k4",
                "p1",
                "p2",
                "b1",
                "b2",
            ):
                value = float(child.text)
                kwargs[child.tag] = value
        kwargs["imgsz"] = int(kwargs.pop("width")), int(kwargs.pop("height"))
        return cls(**kwargs)  # type: ignore

    @classmethod
    def _from_camera_initial(cls, cam: Camera) -> "Agisoft":
        return cls(
            imgsz=(cam.imgsz[0], cam.imgsz[1]),
            f=cam.f[1],
            cx=cam.c[0],
            cy=cam.c[1],
            k1=cam.k[0],
            k2=cam.k[1],
            k3=cam.k[2],
            p1=cam.p[1],
            p2=cam.p[0],
            b1=cam.f[0] - cam.f[1],
        )

    @classmethod
    def from_camera(
        cls,
        cam: Camera,
        optimize: Optimize = True,
        uv: Union[np.ndarray, int] = 1000,
        **kwargs: Any,
    ) -> "Agisoft":
        """
        Convert from :class:`~glimpse.Camera` object.

        Arguments:
            cam: Camera object.
            optimize: Whether and which :class:`Agisoft` parameters to optimize to
                minimize the residuals between the cameras. If `cam.k[3:6]` are zero,
                the conversion is exact and no optimization is performed. Otherwise:

                    - If `True`, optimizes :attr:`k1`, :attr:`k2`, and :attr:`k3`.
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
            optimize = {"k1": True, "k2": True, "k3": True}
        converter = Converter(xcam=xcam, cam=cam, uv=uv)
        converter.optimize_xcam(params=cast(Parameters, optimize), **kwargs)
        return cast("Agisoft", converter.xcam)

    def _xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        # Compute lens distortion
        r2 = np.sum(xy ** 2, axis=1)
        dr = self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3 + self.k4 * r2 ** 4
        xty = xy[:, 0] * xy[:, 1]
        dtx = self.p1 * (r2 + 2 * xy[:, 0] ** 2) + 2 * self.p2 * xty
        dty = self.p2 * (r2 + 2 * xy[:, 1] ** 2) + 2 * self.p1 * xty
        # Apply lens distortion
        dxy = xy.copy()
        dxy[:, 0] += dxy[:, 0] * dr + dtx
        dxy[:, 1] += dxy[:, 1] * dr + dty
        # Project to image
        return np.column_stack(
            (
                (
                    self.imgsz[0] * 0.5
                    + self.cx
                    + dxy[:, 0] * (self.f + self.b1)
                    + dxy[:, 1] * self.b2
                ),
                self.imgsz[1] * 0.5 + self.cy + dxy[:, 1] * self.f,
            )
        )

    def _to_camera_initial(self) -> Camera:
        return Camera(
            imgsz=self.imgsz,
            f=(self.f + self.b1, self.f),
            c=(self.cx, self.cy),
            k=(self.k1, self.k2, self.k3),
            p=(self.p2, self.p1),
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
                If :attr:`k4` and :attr:`b2` are zero,
                the conversion is exact and no optimization is performed.
                Otherwise:

                    - If `True`, optimizes :attr:`~glimpse.Camera.k` if :attr:`k4` is
                      non-zero and :attr:`~glimpse.Camera.f`, :attr:`~glimpse.Camera.c`,
                      and :attr:`~glimpse.Camera.k` if :attr:`b2` is non-zero.
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
        if not optimize or not any((self.k4, self.b2)):
            return cam
        if optimize is True:
            optimize = {}
            if self.k4:
                optimize["k"] = True
            if self.b2:
                optimize["f"] = True
                optimize["c"] = True
                optimize["k"] = True
        converter = Converter(xcam=self, cam=cam, uv=uv)
        converter.optimize_cam(params=cast(Parameters, optimize), **kwargs)
        return converter.cam
