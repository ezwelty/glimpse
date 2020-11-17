"""Convert between external camera models and the glimpse camera model."""
from typing import TYPE_CHECKING, Any, Dict, Iterable, Union

import matplotlib.pyplot
import numpy as np
import scipy.optimize

if TYPE_CHECKING:
    from .cameras import Agisoft, Matlab, OpenCV, PhotoModeler

from .. import optimize
from ..camera import Camera

Parameters = Dict[str, Union[bool, int, Iterable[int]]]
Optimize = Union[bool, Parameters]
ExternalCamera = Union["Agisoft", "Matlab", "OpenCV", "PhotoModeler"]


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
            uv = self._grid(uv)
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

        Residuals are calculated as :attr:`cam` - :attr:`xcam`. For :attr:`xcam` with an
        outgoing distortion model, the original points :attr:`uv` are projected out of
        :attr:`xcam` and into :attr:`cam`. For :attr:`xcam` with an incoming distortion
        model, the original points :attr:`uv` are inverse projected out of :attr:`cam`
        (a numerical estimate that is not exact if distortion coefficients are large),
        then projected into both :attr:`cam` and :attr:`xcam`.

        Returns:
            numpy.ndarray: Image coordinate residuals (n, 2).
        """
        if hasattr(self.xcam, "_uv_to_xy"):
            # Project out of xcam and into cam
            return self.cam._xy_to_uv(self.xcam._uv_to_xy(self.uv)) - self.uv
        # Inverse project out of cam, then into both cam and xcam
        # NOTE: Roundtrip out of and into cam avoids counting inversion errors
        xy = self.cam._uv_to_xy(self.uv)
        return self.cam._xy_to_uv(xy) - self.xcam._xy_to_uv(xy)

    def optimize_cam(self, params: Parameters, **kwargs: Any) -> None:
        """
        Optimize :attr:`cam` parameters to best fit :attr:`xcam`.

        Arguments:
            params: Parameters to optimize by name and indices. For example:

                - {'viewdir': True} : All `viewdir` elements
                - {'viewdir': 0} : First `viewdir` element
                - {'viewdir': [0, 1]} : First and second `viewdir` elements

            **kwargs: Optional arguments to :func:`scipy.optimize.least_squares`.
        """
        mask, _ = optimize.Cameras.parse_params(params)

        def fun(x: np.ndarray) -> np.ndarray:
            self.cam._vector[mask] = x
            return self.residuals().ravel()

        fit = scipy.optimize.least_squares(fun=fun, x0=self.cam._vector[mask], **kwargs)
        self.cam._vector[mask] = fit.x

    def optimize_xcam(self, params: Parameters, **kwargs: Any) -> None:
        """
        Optimize :attr:`xcam` parameters to best fit :attr:`cam`.

        Arguments:
            params: Same as in :meth:`optimize_cam`.
            **kwargs: Optional arguments to :func:`scipy.optimize.least_squares`.
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

        Quivers point from :attr:`xcam` image coordinates to :attr:`cam` image
        coordinates.

        Arguments:
            **kwargs: Arguments to matplotlib.pyplot.quiver. Defaults to
                `{'scale': 1, 'width': 5, 'color': 'red', 'scale_units': 'xy',
                'angles': 'xy', 'units': 'xy'}`.
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
