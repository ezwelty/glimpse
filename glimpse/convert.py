"""Convert between external camera models and the glimpse camera model."""
import re
from typing import Any, cast, Dict, Sequence, Tuple, Union
import warnings
import xml.etree.ElementTree

import matplotlib.pyplot
import numpy as np
import scipy.optimize

from . import optimize
from .camera import Camera

Parameters = Dict[str, Union[bool, int, Sequence[int]]]
Optimize = Union[bool, Parameters]


class MatlabCamera:
    """
    Camera model used by the Camera Calibration Toolbox for Matlab.

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
    def from_report(cls, path: str, sigmas: bool = False) -> "MatlabCamera":
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
    def _from_camera_initial(cls, cam: Camera) -> "MatlabCamera":
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
    ) -> "MatlabCamera":
        """
        Convert from `:class:Camera` object.

        Arguments:
            cam: Camera object.
            optimize: Whether and which `:class:MatlabCamera` parameters to optimize to
                minimize the residuals between the cameras. If `Camera.k[3:6]` are zero,
                the conversion is exact and no optimization is performed. Otherwise:
                    - If `True`, optimizes `:attr:kc`.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                        format as `Converter.optimize_cam()`.
            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.
        """
        xcam = cls._from_camera_initial(cam)
        if not optimize or (cam.k[3:6] == 0).all():
            return xcam
        if optimize is True:
            optimize = {"kc": True}
        converter = Converter(xcam=xcam, cam=cam, uv=uv)
        converter.optimize_xcam(params=cast(Parameters, optimize), **kwargs)
        return cast("MatlabCamera", converter.xcam)

    def _camera2image(self, xy: np.ndarray) -> np.ndarray:
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
        Convert to `:class:Camera` object.

        Arguments:
            optimize: Whether and which `:class:Camera` parameters to optimize to
                minimize the residuals between the cameras. If `:attr:alpha_c` is zero,
                the conversion is exact and no optimization is performed. Otherwise:
                    - If `True`, optimizes `Camera.f`, `c`, `k`, and `p`.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                        format as `Converter.optimize_cam()`.
            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.

        Returns:
            Exact or best-fitting Camera object.
        """
        cam = self._to_camera_initial()
        if not optimize or not self.alpha_c:
            return cam
        if optimize is True:
            optimize = {"f": True, "c": True, "k": True, "p": True}
        converter = Converter(xcam=self, cam=cam, uv=uv)
        converter.optimize_cam(params=cast(Parameters, optimize), **kwargs)
        return converter.cam


class AgisoftCamera:
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
    def from_xml(cls, path: str) -> "AgisoftCamera":
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
    def _from_camera_initial(cls, cam: Camera) -> "AgisoftCamera":
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
    ) -> "AgisoftCamera":
        """
        Convert from `:class:Camera` object.

        Arguments:
            cam: Camera object.
            optimize: Whether and which `:class:AgisoftCamera` parameters to optimize to
                minimize the residuals between the cameras. If `Camera.k[3:6]` are zero,
                the conversion is exact and no optimization is performed. Otherwise:
                    - If `True`, optimizes `:attr:k1`, `:attr:k2`, and `:attr:k3`.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                        format as `Converter.optimize_cam()`.
            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.
        """
        xcam = cls._from_camera_initial(cam)
        if not optimize or (cam.k[3:6] == 0).all():
            return xcam
        if optimize is True:
            optimize = {"k1": True, "k2": True, "k3": True}
        converter = Converter(xcam=xcam, cam=cam, uv=uv)
        converter.optimize_xcam(params=cast(Parameters, optimize), **kwargs)
        return cast("AgisoftCamera", converter.xcam)

    def _camera2image(self, xy: np.ndarray) -> np.ndarray:
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
        Convert to `:class:Camera` object.

        Arguments:
            optimize: Whether and which `:class:Camera` parameters to optimize to
                minimize the residuals between the cameras. If `:attr:k4` and `:attr:b2`
                are zero, the conversion is exact and no optimization is performed.
                Otherwise:
                    - If `True`, optimizes `Camera.k` if `:attr:k4` is non-zero and
                        `Camera.f`, `c`, and `k` if `:attr:b2` is non-zero.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                        format as `Converter.optimize_cam()`.
            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.

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


class OpenCVCamera:
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
        self.k1, self.k2, = k1, k2
        self.p1, self.p2 = p1, p2
        self.k3, self.k4, self.k5, self.k6 = k3, k4, k5, k6
        self.s1, self.s2, self.s3, self.s4 = s1, s2, s3, s4

    @property
    def cameraMatrix(self) -> Sequence[Sequence[float]]:
        """
        Camera matrix.

        Follows the OpenCV format [(fx 0 cx), (0 fy cy), (0 0 1)].
        """
        return [(self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)]

    @property
    def distCoeffs(self) -> Sequence[float]:
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
    def _parse_camera_matrix(x: Sequence[Sequence[float]]) -> Dict[str, float]:
        return {"fx": x[0][0], "fy": x[1][1], "cx": x[0][2], "cy": x[1][2]}

    @staticmethod
    def _parse_distortion_coefficients(x: Sequence[float]) -> Dict[str, float]:
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
        cameraMatrix: Sequence[Sequence[float]],
        distCoeffs: Sequence[float],
        imgsz: Tuple[int, int],
    ) -> "OpenCVCamera":
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
    def from_xml(cls, path: str, imgsz: Tuple[int, int]) -> "OpenCVCamera":
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
    def _from_camera_initial(cls, cam: Camera) -> "OpenCVCamera":
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
    def from_camera(cls, cam: Camera) -> "OpenCVCamera":
        """
        Convert from `:class:Camera` object.

        Since the OpenCV camera model is a superset of the glimpse camera model,
        the conversion is exact and no optimization is needed.

        Arguments:
            cam: Camera object.
        """
        return cls._from_camera_initial(cam)

    def _camera2image(self, xy: np.ndarray) -> np.ndarray:
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
        Convert to `:class:Camera` object.

        Arguments:
            optimize: Whether and which `:class:Camera` parameters to optimize to
                minimize the residuals between the cameras. If `:attr:s1`, `:attr:s2`,
                `:attr:s3`, and `:attr:s4` are zero, the conversion is exact and
                no optimization is performed. Otherwise:
                    - If `True`, optimizes `Camera.k` and `p`.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                        format as `Converter.optimize_cam()`.
            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.

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


class PhotoModelerCamera:
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
    ) -> "PhotoModelerCamera":
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
    def _from_camera_initial(cls, cam: Camera) -> "PhotoModelerCamera":
        if cam.sensorsz is None:
            raise ValueError("Camera does not have a sensor size")
        return cls(
            imgsz=(cam.imgsz[0], cam.imgsz[1]),
            focal=(cam.fmm[0] + cam.fmm[1]) / 2,
            xp=cam.cmm[0] + cam.sensorsz[0] / 2,
            yp=cam.cmm[1] + cam.sensorsz[1] / 2,
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
    ) -> "PhotoModelerCamera":
        """
        Convert from `:class:Camera` object.

        `Camera.sensorsz` is required.

        Arguments:
            cam: Camera object.
            optimize: Whether and which `:class:PhotoModelerCamera` parameters to
                optimize to minimize the residuals between the cameras. If `Camera.fmm`
                are equal and all `Camera.k` and `Camera.p` are zero,
                the conversion is exact and no optimization is performed. Otherwise:
                    - If `True`, optimizes `:attr:focal`, `:attr:fw`, `:attr:fh`,
                        `:attr:xp`, and `:attr:yp` if `Camera.fmm` are not equal,
                        `:attr:k1`, `:attr:k2`, and `:attr:k3` if any `Camera.k`
                        are non-zero,
                        and `:attr:p1` and `:attr:p2` if any `Camera.p` are non-zero.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                        format as `Converter.optimize_cam()`.
            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.

        Raises:
            ValueError: Camera does not have a sensor size.
        """
        xcam = cls._from_camera_initial(cam)
        if not optimize or (
            cam.fmm[0] == cam.fmm[1] and np.all(cam.k == 0) and np.all(cam.p == 0)
        ):
            return xcam
        if optimize is True:
            optimize = {}
            if cam.fmm[0] != cam.fmm[1]:
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
        return cast("PhotoModelerCamera", converter.xcam)

    def _image2camera(self, uv: np.ndarray) -> np.ndarray:
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
        Convert to `:class:Camera` object.

        Arguments:
            optimize: Whether and which `:class:Camera` parameters to optimize to
                minimize the residuals between the cameras. If `:attr:k1`, `:attr:k2`,
                `:attr:k3`, `:attr:p1`, and `:attr:p2` is zero,
                the conversion is exact and no optimization is performed. Otherwise:
                    - If `True`, optimizes `Camera.k` if `:attr:k1`, `:attr:k2`, or
                        `:attr:k3` is non-zero, and `Camera.p` if `:attr:p1` or
                        `:attr:p2` is non-zero.
                    - If `False`, no optimization is performed.
                    - Alternatively, choose the parameters to optimize using the same
                        format as `Converter.optimize_cam()`.
            uv: Image point coordinates or number of evenly-spaced image points (int)
                at which to compute the residuals between the cameras.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.

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


ExternalCamera = Union[MatlabCamera, AgisoftCamera, OpenCVCamera, PhotoModelerCamera]
InverseCamera = Union[PhotoModelerCamera]


class Converter:
    """
    Convert between an external camera and a glimpse camera.

    Camera parameters are optimized as needed to minimize the residuals between them.
    Both cameras must have the same image size, which cannot change after
    initialization.

    Arguments:
        uv (array-like or int): Image point coordinates or number of image points (int)
            at which to compute the residuals between the cameras.
            If an integer, uses a regular rectangular grid of points
            which closely matches the target number of points.

    Attributes:
        xcam (ExternalCamera): External camera model
        cam (Camera): Camera model
        uv (numpy.ndarray): Image point coordinates

    Raises:
        ValueError: Cameras have different image sizes.
    """

    def __init__(
        self, xcam: ExternalCamera, cam: Camera, uv: Union[np.ndarray, int] = 1000
    ) -> None:
        if any(xcam.imgsz != cam.imgsz):
            raise ValueError("Cameras have different image sizes.")
        self.xcam = xcam
        self.cam = cam
        if isinstance(uv, int):
            uv = self._grid(n=uv)
        self.uv = np.atleast_2d(uv)

    def _grid(self, n: int) -> np.ndarray:
        imgsz = self.cam.imgsz
        area = imgsz[0] * imgsz[1]
        d = np.sqrt(area / n)
        # Place points such that spacing to edge is half of interpoint spacing
        # - | -- | -- | -- | -
        dx = imgsz[0] / round(imgsz[0] / d)
        x = np.arange(0.5 * dx, imgsz[0], dx)
        dy = imgsz[1] / round(imgsz[1] / d)
        y = np.arange(0.5 * dy, imgsz[1], dy)
        return np.reshape(np.meshgrid(x, y), (2, -1)).T

    def residuals(self) -> np.ndarray:
        """
        Returns the image coordinate residuals between the two cameras.

        Residuals are calculated as `:attr:cam` - `:attr:xcam`. For `:attr:xcam` with an
        outgoing distortion model, the original points `:attr:uv` are projected out of
        `:attr:xcam` and into `:attr:cam`. For `:attr:xcam` with an incoming distortion
        model, the original points `:attr:uv` are inverse projected out of `:attr:cam`
        (a numerical estimate that is not exact if distortion coefficients are large),
        then projected into both `:attr:cam` and `:attr:xcam`.

        Returns:
            numpy.ndarray: Image coordinate residuals (n, 2).
        """
        if isinstance(self.xcam, InverseCamera):
            # Project out of xcam and into cam
            return self.cam._camera2image(self.xcam._image2camera(self.uv)) - self.uv
        # Inverse project out of cam, then into both cam and xcam
        # NOTE: Roundtrip out of and into cam avoids counting inversion errors
        xy = self.cam._image2camera(self.uv)
        return self.cam._camera2image(xy) - self.xcam._camera2image(xy)

    def optimize_cam(self, params: Parameters, **kwargs: Any) -> None:
        """
        Optimize `:attr:cam` parameters to best fit `:attr:xcam`.

        Arguments:
            params: Parameters to optimize by name and indices. For example:
                - {'viewdir': True} : All `viewdir` elements
                - {'viewdir': 0} : First `viewdir` element
                - {'viewdir': [0, 1]} : First and second `viewdir` elements
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.
        """
        mask, _ = optimize.Cameras.parse_params(params)

        def fun(x: np.ndarray) -> np.ndarray:
            self.cam.vector[mask] = x
            return self.residuals().ravel()

        fit = scipy.optimize.least_squares(fun=fun, x0=self.cam.vector[mask], **kwargs)
        self.cam.vector[mask] = fit.x

    def optimize_xcam(self, params: Parameters, **kwargs: Any) -> None:
        """
        Optimize `:attr:xcam` parameters to best fit `:attr:cam`.

        Arguments:
            params: Same as in `optimize_cam()`.
            **kwargs: Optional arguments to `scipy.optimize.least_squares()`.
        """
        # Drop empty params and normalize value as a numpy index
        indices = {k: slice(None) if v is True else v for k, v in params.items() if v}
        x0 = []
        for key, idx in indices.items():
            value = np.atleast_1d(getattr(self.xcam, key))
            x0.extend(value[idx])

        def apply(x: np.ndarray) -> None:
            i = 0
            for key, idx in indices.items():
                value = np.atleast_1d(getattr(self.xcam, key)).astype(float)
                n = len(value) if isinstance(idx, slice) else len(np.atleast_1d(idx))
                value[idx] = x[i : i + n]
                setattr(self.xcam, key, tuple(value) if len(value) > 1 else value[0])
                i += n

        def fun(x: np.ndarray) -> np.ndarray:
            apply(x)
            return self.residuals().ravel()

        fit = scipy.optimize.least_squares(fun=fun, x0=x0, **kwargs)
        apply(fit.x)

    def plot(self, **kwargs: Any) -> matplotlib.quiver.Quiver:
        """
        Plot image reprojection errors as quivers.

        Quivers point from `:attr:xcam` image coordinates to `:attr:cam` image
        coordinates.

        Arguments:
            **kwargs: Arguments to matplotlib.pyplot.quiver. Defaults to
                {"scale": 1, "width": 5, "color": "red", "scale_units": "xy",
                "angles": "xy", "units": "xy"}.
        """
        kwargs = {
            "scale": 1,
            "width": 5,
            "color": "red",
            "scale_units": "xy",
            "angles": "xy",
            "units": "xy",
            **kwargs,
        }
        duv = kwargs["scale"] * self.residuals()
        return matplotlib.pyplot.quiver(
            self.uv[:, 0], self.uv[:, 1], duv[:, 0], duv[:, 1], **kwargs
        )
