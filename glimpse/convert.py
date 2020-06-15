import inspect
import re
import xml

import numpy as np

import scipy.optimize

from . import optimize
from .camera import Camera


class _IncomingPoints(object):
    """
    Points control for external cameras with camera-to-image distortion models.

    Projects normalized camera coordinates into `xcam` (observed) and `cam`
    (predicted) image coordinates. The normalized camera coordinates are
    generated from `cam`, so it is important that it have reasonable initial
    parameters.

    Attributes:
        xcam (_IncomingCamera): External camera model
        cam (Camera): Camera model (initial guess)
        step (int): Pixel grid spacing for control points
    """

    def __init__(self, xcam, cam, step=10):
        self.xcam = xcam
        self.cam = cam
        # Cache grid in normalized camera coordinates to avoid inversion errors
        # NOTE: Requires that xcam and cam have the same image size
        self.xy = cam._image2camera(cam.grid(step=step, mode="points"))

    @property
    # HACK: Required for optimize.Cameras
    def size(self):
        return len(self.xy)

    @property
    # HACK: Required for optimize.Cameras
    def cams(self):
        return [self.cam]

    # HACK: index required for optimize.Cameras
    def observed(self, index=None):
        if index is None:
            index = slice(None)
        return self.xcam._camera2image(self.xy[index])

    # HACK: index required for optimize.Cameras
    def predicted(self, index=None):
        if index is None:
            index = slice(None)
        return self.cam._camera2image(self.xy[index])


class _OutgoingPoints(_IncomingPoints):
    """
    Points control for external cameras with image-to-camera distortion models.

    Projects image coordinates from `xcam` (observed) to normalized camera
    coordinates, then into `cam` image coordinates (predicted).

    Attributes:
        xcam (_ExternalCamera): External camera model
        cam (Camera): Camera model (initial guess)
        step (int): Pixel grid spacing for control points
    """

    def __init__(self, xcam, cam, step=10):
        self.xcam = xcam
        self.cam = cam
        # NOTE: Requires that xcam and cam have the same image size
        self.uv = cam.grid(step=step, mode="points")
        self.xy = xcam._image2camera(self.uv)

    # HACK: index required for optimize.Cameras
    def observed(self, index=None):
        if index is None:
            index = slice(None)
        return self.uv[index]


class _ExternalCamera(object):
    """
    Template for an external camera model.
    """

    def __init__(self):
        pass

    @property
    def _points(self):
        pass

    def _as_camera_initial(self):
        """
        Return initial camera model.
        """
        return Camera()

    def _as_camera_estimate(self, params, step=10):
        cam = self._as_camera_initial()
        points = self._points(self, cam, step=step)
        mask, _ = optimize.Cameras.parse_params(params)

        def fun(x):
            cam.vector[mask] = x
            return (points.predicted() - points.observed()).ravel()

        fit = scipy.optimize.least_squares(fun=fun, x0=cam.vector[mask])
        cam.vector[mask] = fit.x
        return cam

    def _cameras(self, params, step=10):
        """
        Get Cameras model for optimization.

        Arguments:
            params (dict): Camera parameters to optimize
            step (int): Pixel grid spacing for control points
        """
        cam = self._as_camera_initial()
        control = self._points(self, cam, step)
        return optimize.Cameras([cam], [control], [params])

    def as_camera(self, step=10):
        """
        Return equivalent camera model.

        Arguments:
            step (int): Pixel grid spacing for control points
        """
        pass


class _IncomingCamera(_ExternalCamera):
    """
    Template for an external camera with a camera-to-image distortion model.
    """

    def __init__(self):
        pass

    @property
    def _points(self):
        return _IncomingPoints

    def _camera2image(self, xy):
        """
        Project camera to image coordinates.

        Arguments:
            xy (array): Camera coordinates (Nx2)

        Returns:
            array: Image coordinates (Nx2), in a system where the top left
                corner of the top left pixel is (0, 0).
        """
        pass


class _OutgoingCamera(_ExternalCamera):
    """
    Template for an external camera with a image-to-camera distortion model.
    """

    @property
    def _points(self):
        return _OutgoingPoints

    def _image2camera(self, uv):
        """
        Project image to camera coordinates.

        Arguments:
            uv (array): Image coordinates (Nx2), in a system where the top left
                corner of the top left pixel is (0, 0).

        Returns:
            array: Normalized camera coordinates (Nx2)
        """
        pass


class MatlabCamera(_IncomingCamera):
    """
    Camera model used by the Camera Calibration Toolbox for Matlab.

    See http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html.

    Attributes:
        nx (float): Image size in pixels (x)
        ny (float): Image size in pixels (y)
        fc (iterable): Focal length in pixels (x, y)
        cc (iterable): Principal point in pixels (x, y), in an image coordinate
            system where the center of the top left pixel is (0, 0)
        kc (iterable): Image distortion coefficients (k1, k2, p1, p2, k3)
        alpha_c (float): Skew coefficient defining the angle between the x and y
            pixel axes
    """

    def __init__(self, nx, ny, fc, cc=None, kc=(0, 0, 0, 0, 0), alpha_c=0):
        self.nx, self.ny = nx, ny
        self.fc = fc
        if cc is None:
            cc = (nx - 1) / 2, (ny - 1) / 2
        self.cc = cc
        self.kc = kc
        self.alpha_c = alpha_c

    @classmethod
    def from_report(cls, path, sigmas=False):
        """
        Read from calibration report (Calib_Result.m).

        Arguments:
            path (str): Path to report
            sigmas (bool): Whether to read parameter means (False)
                or standard deviations (True)
        """
        with open(path, mode="r") as fp:
            txt = fp.read()

        def parse_param(param, length):
            if length == 1:
                pattern = r"^{param} = (.*);".format(param=param)
            else:
                pattern = r"^{param} = \[ {groups} \];".format(
                    param=param, groups=" ; ".join(["(.*)"] * length)
                )
            values = re.findall(pattern, txt, flags=re.MULTILINE)[0]
            # Error bounds are ~3 times standard deviations
            scale = 1 / 3 if sigmas else 1
            if length == 1:
                return float(values) * scale
            else:
                return [float(value) * scale for value in values]

        if sigmas:
            lengths = {"fc_error": 2, "cc_error": 2, "alpha_c_error": 1, "kc_error": 5}
        else:
            lengths = {"fc": 2, "cc": 2, "alpha_c": 1, "kc": 5, "nx": 1, "ny": 1}
        kwargs = {
            param: parse_param(param, length) for param, length in lengths.items()
        }
        if sigmas:
            kwargs = {key.split("_error")[0]: kwargs[key] for key in kwargs}
            kwargs["nx"], kwargs["ny"] = None, None
        return cls(**kwargs)

    def _camera2image(self, xy):
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
                self.fc[0] * dxy[:, 1] + self.cc[1],
            )
        )
        # Top left corner of top left pixel is (-0.5, -0.5)
        uv += (0.5, 0.5)
        return uv

    def _as_camera_initial(self):
        return Camera(
            imgsz=(self.nx, self.ny),
            f=self.fc,
            c=((self.cc[0] + 0.5) - self.nx / 2, (self.cc[1] + 0.5) - self.ny / 2),
            k=(self.kc[0], self.kc[1], self.kc[4]),
            p=(self.kc[2], self.kc[3]),
        )

    def as_camera(self, step=10):
        """
        Return equivalent `Camera` object.

        If `alpha_c` is non-zero, the conversion is estimated numerically.
        Otherwise, the conversion is exact.
        """
        if self.alpha_c:
            params = {"f": True, "c": True, "k": True}
            return self._as_camera_estimate(params=params, step=step)
        else:
            return self._as_camera_initial()


class AgisoftCamera(_IncomingCamera):
    """
    Frame camera model used by Agisoft software (PhotoScan, Metashape, Lens).

    See https://www.agisoft.com/pdf/metashape-pro_1_6_en.pdf (Appendix C).

    Attributes:
        width (float): Image size in pixels (x)
        height (float): Image size in pixels (y)
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
        self, width, height, f, cx, cy, k1=0, k2=0, k3=0, k4=0, p1=0, p2=0, b1=0, b2=0
    ):
        self.width, self.height = width, height
        self.f = f
        self.cx, self.cy = cx, cy
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.p1, self.p2 = p1, p2
        self.b1, self.b2 = b1, b2

    @classmethod
    def from_xml(cls, path):
        tree = xml.etree.ElementTree.parse(path)
        calibration = next((e for e in tree.iter("calibration")), None)
        if not calibration:
            raise ValueError("No camera model found")
        params = {}
        for child in calibration:
            params[child.tag] = child.text
        if params["projection"] != "frame":
            raise ValueError(
                "Found unsupported camera model type: " + params["projection"]
            )
        kwargs = {
            key: float(params[key])
            for key in inspect.getfullargspec(cls).args[1:]
            if key in params
        }
        return cls(**kwargs)

    def _camera2image(self, xy):
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
                    self.width * 0.5
                    + self.cx
                    + dxy[:, 0] * (self.f + self.b1)
                    + dxy[:, 1] * self.b2
                ),
                self.height * 0.5 + self.cy + dxy[:, 1] * self.f,
            )
        )

    def _as_camera_initial(self):
        return Camera(
            imgsz=(self.width, self.height),
            f=(self.f + self.b1, self.f),
            c=(self.cx, self.cy),
            k=(self.k1, self.k2, self.k3),
            p=(self.p2, self.p1),
        )

    def as_camera(self, step=10):
        """
        Return equivalent `Camera` object.

        If either `k4` or `b2` is non-zero, the conversion is estimated
        numerically. Otherwise, the conversion is exact.

        Arguments:
            step: Sample grid spacing for all (float) or each (iterable) dimension
        """
        if any((self.k4, self.b2)):
            params = {}
            if self.k4:
                params["k"] = True
            if self.b2:
                params["f"] = True
                params["c"] = True
                params["k"] = True
            return self._as_camera_estimate(params=params, step=step)
        else:
            return self._as_camera_initial()


class PhotoModelerCamera(_OutgoingCamera):
    """
    Camera model used by EOS Systems PhotoModeler.

    See "Lens Distortion Formulation" in the software help.

    Attributes:
        imgsz (iterable): Desired image size (nx, ny)
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

    def __init__(self, imgsz, focal, xp, yp, fw, fh, k1=0, k2=0, k3=0, p1=0, p2=0):
        self.imgsz = imgsz
        self.focal = focal
        self.xp, self.yp = xp, yp
        self.fw, self.fh = fw, fh
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.p1, self.p2 = p1, p2

    @classmethod
    def from_report(cls, path, imgsz, sigmas=False):
        """
        Read from camera calibration project report.

        Arguments:
            path (str): Path to report
            imgsz (iterable): Desired image size (nx, ny)
            sigmas (bool): Whether to read parameter means (False) or
                standard deviations (True)
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
            matches = [
                re.findall(label + r".*\s.*\s*Deviation: .*: ([0-9\-\+\.e]+)", txt)
                for label in params.values()
            ]
            kwargs = {
                arg: float(match[0]) if match else None
                for arg, match in zip(params.keys(), matches)
            }
        else:
            matches = [
                re.findall(label + r".*\s*Value: ([0-9\-\+\.e]+)", txt)
                for label in params.values()
            ]
            kwargs = {
                arg: float(match[0]) for arg, match in zip(params.keys(), matches)
            }
        return cls(imgsz=imgsz, **kwargs)

    def _image2camera(self, uv):
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

    def _as_camera_initial(self):
        return Camera(
            imgsz=self.imgsz,
            sensorsz=(self.fw, self.fh),
            fmm=self.focal,
            cmm=(self.xp - self.fw / 2, self.yp - self.fh / 2),
        )

    def as_camera(self, step=10):
        """
        Return equivalent `Camera` object.

        If either `k1`, `k2`, `k3`, `p1`, or `p2` is non-zero, the conversion is
        estimated numerically. Otherwise, the conversion is exact.

        Arguments:
            step: Sample grid spacing for all (float) or each (iterable) dimension
        """
        k = self.k1, self.k2, self.k3
        p = self.p1, self.p2
        if any(k + p):
            params = {}
            if any(k):
                params["k"] = True
            if any(p):
                params["p"] = True
            return self._as_camera_estimate(params=params, step=step)
        else:
            return self._as_camera_initial()


class OpenCVCamera(_IncomingCamera):
    """
    Frame camera model used by OpenCV.

    See https://docs.opencv.org/master/d9/d0c/group__calib3d.html#details.
    Distortion coefficients τx and τy are not supported.

    Attributes:
        imgsz (iterable): Image size in pixels (nx, ny)
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
        imgsz,
        fx,
        fy,
        cx=None,
        cy=None,
        k1=0,
        k2=0,
        k3=0,
        k4=0,
        k5=0,
        k6=0,
        p1=0,
        p2=0,
        s1=0,
        s2=0,
        s3=0,
        s4=0,
    ):
        self.imgsz = imgsz
        self.fx, self.fy = fx, fy
        self.cx = cx = imgsz[0] / 2 if cx is None else cx
        self.cy = cy = imgsz[1] / 2 if cy is None else cy
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.k4, self.k5, self.k6 = k4, k5, k6
        self.p1, self.p2 = p1, p2
        self.s1, self.s2, self.s3, self.s4 = s1, s2, s3, s4

    @staticmethod
    def parse_camera_matrix(x):
        """
        Return fx, fy, cx, and cy from camera matrix.

        Arguments:
            x (array-like): Camera matrix [[fx 0 cx], [0 fy cy], [0 0 1]]

        Returns:
            dict: fx, fy, cx, and cy
        """
        x = np.asarray(x)
        return {"fx": x[0, 0], "fy": x[1, 1], "cx": x[0, 2], "cy": x[1, 2]}

    @staticmethod
    def parse_distortion_coefficients(x):
        """
        Return k*, p*, s*, and τ* from distortion coefficients vector.

        Arguments:
            x (iterable): Distortion coefficients
                [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, τx, τy]
        """
        x = np.asarray(x)
        labels = (
            "k1",
            "k2",
            "p1",
            "p2",
            "k3",
            "k4",
            "k5",
            "k6",
            "s1",
            "s2",
            "s3",
            "s4",
            "τx",
            "τy",
        )
        return {key: x[i] if i < len(x) else 0 for i, key in enumerate(labels)}

    @classmethod
    def from_xml(cls, path, imgsz):
        tree = xml.etree.ElementTree.parse(path)
        params = {"imgsz": imgsz}
        matrix = next((e for e in tree.iter("camera_matrix")), None)
        if matrix:
            txt = matrix.find("data").text
            x = np.asarray(
                [float(xi) for xi in re.findall(r"([0-9\-\.e\+]+)", txt)]
            ).reshape(3, 3)
            params = {**params, **cls.parse_camera_matrix(x)}
        else:
            raise ValueError("No camera matrix found")
        coeffs = next((e for e in tree.iter("distortion_coefficients")), None)
        if coeffs:
            txt = coeffs.find("data").text
            x = np.asarray([float(xi) for xi in re.findall(r"([0-9\-\.e\+]+)", txt)])
            params = {**params, **cls.parse_distortion_coefficients(x)}
        kwargs = {
            key: params[key]
            for key in inspect.getfullargspec(cls).args[1:]
            if key in params
        }
        return cls(**kwargs)

    def _camera2image(self, xy):
        # Compute lens distortion
        r2 = np.sum(xy ** 2, axis=1)
        dr = (1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3) / (
            1 + self.k4 * r2 + self.k5 * r2 ** 2 + self.k6 * r2 ** 2
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

    def _as_camera_initial(self):
        return Camera(
            imgsz=self.imgsz,
            f=(self.fx, self.fy),
            c=(self.cx - self.imgsz[0] / 2, self.cy - self.imgsz[1] / 2),
            k=(self.k1, self.k2, self.k3, self.k4, self.k5, self.k6),
            p=(self.p1, self.p2),
        )

    def as_camera(self, step=10):
        """
        Return equivalent `Camera` object.

        If either `s1`, `s2`, `s3`, or `s4` is non-zero (`τx` and `τy` are not
        supported), the conversion is estimated numerically. Otherwise, the
        conversion is exact.

        Arguments:
            step: Sample grid spacing for all (float) or each (iterable) dimension
        """
        if any((self.s1, self.s2, self.s3, self.s4)):
            params = {"k": True, "p": True}
            return self._as_camera_estimate(params=params, step=step)
        else:
            return self._as_camera_initial()


def as_camera_sigma(mean, sigma, n=100, **kwargs):
    """
    Convert to `Camera` and propagate uncertainties.

    Arguments:
        mean: `glimpse.convert` camera object with parameter means
        sigma: `glimpse.convert` camera object (same type as `mean`) with
            parameter standard deviations
        n (int): Number of iterations to use for estimating uncertainties
        **kwargs: Arguments to `mean.as_camera()`

    Returns:
        `glimpse.Camera`: Parameter means
        `glimpse.Camera`: Parameter standard deviations
    """
    mean_args = mean.__dict__.copy()
    sigma_args = sigma.__dict__.copy()
    mean_cam = mean.as_camera(**kwargs)
    vectors = []
    for _ in range(n):
        args = {
            key: mean_args[key]
            + (np.random.normal(scale=sigma_args[key]) if sigma_args[key] else 0)
            for key in mean_args
        }
        new_mean = type(mean)(**args)
        vectors.append(new_mean.as_camera(**kwargs).vector)
    vectors = np.array(vectors)
    sigma_cam = Camera(vector=np.std(vectors, axis=0))
    return mean_cam, sigma_cam
