"""Read, write, and manipulate photographic images."""
import datetime
from typing import Any, Iterable, Optional, Tuple, Union

import matplotlib.pyplot
import numpy as np
import osgeo.gdal
import scipy.interpolate
import sharedmem

from . import helpers
from .camera import Camera
from .exif import Exif


class Image:
    """
    Photographic image and the settings that gave rise to the image.

    Arguments:
        path: Path to image file.
        cam: :class:`Camera` or arguments passed to :class:`Camera`.
            If missing, `imgsz`, `fmm`, and `sensorsz` are read from **exif**.
        datetime: Image capture date and time. If `None`, read from **exif**.
        exif: Image metadata. If `None` and needed for **cam** or **datetime**,
            read from **path**.

    Attributes:
        path (str): Path to image file.
        cam (:class:`Camera`): Camera model.
        exif (:class:`Exif`): Image metadata.
        datetime (datetime.datetime): Image capture date and time.
        I (numpy.ndarray): Cached image content.

    Example:
        By default, the base camera model (:class:`Camera`) and image capture time
        are loaded from image metadata (:class:`Exif`) read from the image file.

        >>> path = 'tests/AK10b_20141013_020336.JPG'
        >>> img = Image(path)
        >>> img.path
        'tests/AK10b_20141013_020336.JPG'
        >>> img.path == path
        True
        >>> img.cam.imgsz
        array([800, 536])
        >>> all(img.cam.imgsz == img.exif.imgsz)
        True
        >>> img.cam.sensorsz
        array([23.6, 15.8])
        >>> all(img.cam.sensorsz == img.exif.sensorsz)
        True
        >>> img.cam.fmm
        array([20., 20.])
        >>> all(img.cam.fmm == img.exif.fmm)
        True
        >>> img.datetime
        datetime.datetime(2014, 10, 13, 2, 3, 36, 280000)
        >>> img.datetime == img.exif.datetime
        True

        If all of these are provided, image metadata is not read, which is faster.

        >>> img = Image(
        ...     path,
        ...     cam={'imgsz': (800, 536), 'sensorsz': (23.6, 15.8), 'fmm': 20},
        ...     datetime=datetime.datetime(2014, 10, 13, 2, 3, 36, 280000))
        >>> img.exif is None
        True

        Custom camera parameters override and supplement those read from metadata.

        >>> fmm = 28
        >>> xyz = (1, 2, 3)
        >>> img = Image(path, cam={'fmm': fmm, 'xyz': xyz})
        >>> img.cam.imgsz
        array([800, 536])
        >>> all(img.cam.fmm == fmm)
        True
        >>> all(img.cam.xyz == xyz)
        True
    """

    def __init__(
        self,
        path: str,
        cam: Union[dict, Camera] = None,
        datetime: datetime.datetime = None,
        exif: Exif = None,
    ) -> None:
        self.path = path
        if not cam:
            cam = {}
        if isinstance(cam, dict):
            if not (
                "imgsz" in cam and "f" in cam or ("fmm" in cam and "sensorsz" in cam)
            ):
                exif = exif or Exif(path)
                cam = {
                    "imgsz": exif.imgsz,
                    "fmm": exif.fmm,
                    "sensorsz": exif.sensorsz,
                    **cam,
                }
            cam = Camera(**cam)
        self.cam = cam
        if not datetime:
            exif = exif or Exif(path)
            datetime = exif.datetime
        self.datetime = datetime
        self.exif = exif
        self.I = None

    @property
    def _path_imgsz(self) -> Tuple[int, int]:
        ds = osgeo.gdal.Open(self.path)
        return ds.RasterXSize, ds.RasterYSize

    @property
    def _cache_imgsz(self) -> Optional[Tuple[int, int]]:
        if self.I is not None:
            return self.I.shape[1], self.I.shape[0]
        return None

    def read(self, box: Iterable[int] = None, cache: bool = True) -> np.ndarray:
        """
        Read image data from file.

        The image is resized as needed to the camera image size
        (:attr:`cam`.imgsz). The result is cached (:attr:`I`) and reused only if
        it matches the camera image size. To clear the cache, set :attr:`I` to
        `None`.

        Arguments:
            box: Crop extent in image coordinates (left, top, right, bottom)
                relative to :attr:`cam`.imgsz. If `None`, the full image is returned.
            cache: Whether to cache image values.
                If `True`, the region is extracted from the cached image.
                If `False`, the region is extracted directly from the file
                (faster than reading the entire image).

        Example:
            The image is read and resized as needed to match the camera image size.

            >>> img = Image('tests/AK10b_20141013_020336.JPG')
            >>> img.cam.resize(0.5)
            >>> img.cam.imgsz
            array([400, 268])
            >>> I = img.read()
            >>> I.shape[1], I.shape[0]
            (400, 268)
            >>> img.cam.resize(1)
            >>> I = img.read()
            >>> I.shape[1], I.shape[0]
            (800, 536)

            Reading a subset of the image is equivalent to
            slicing the original image, even when it is read directly from file.

            >>> box = 0, 5, 100, 94
            >>> tile = img.read(box)
            >>> np.all(tile == I[box[1] : box[3], box[0] : box[2]])
            True
            >>> tile = img.read(box, cache=False)
            >>> np.all(tile == I[box[1] : box[3], box[0] : box[2]])
            True
        """
        size = self._cache_imgsz or self._path_imgsz
        cam_size = tuple(self.cam.imgsz)
        resize = cam_size != size
        new_I = True
        if self.I is not None and not resize:
            I = self.I
            new_I = False
        else:
            ds = osgeo.gdal.Open(self.path)
            args = {}
            if resize:
                args["buf_xsize"], args["buf_ysize"] = cam_size
            if box is not None and not cache:
                # Resize box to actual image size
                xscale, yscale = size[0] / cam_size[0], size[1] / cam_size[1]
                # Read image subset
                args["xoff"] = int(round(box[0] * xscale))
                args["win_xsize"] = int(round((box[2] - box[0]) * xscale))
                args["yoff"] = int(round(box[1] * yscale))
                args["win_ysize"] = int(round((box[3] - box[1]) * yscale))
            I = np.dstack(
                [
                    ds.GetRasterBand(i + 1).ReadAsArray(**args)
                    for i in range(ds.RasterCount)
                ]
            )
            if I.shape[2] == 1:
                I = I.squeeze(axis=2)
            if cache:
                I = sharedmem.copy(I)
                self.I = I
        if box is not None and (cache or not new_I):
            # Caching and cropping: Subset cached array
            I = I[box[1] : box[3], box[0] : box[2]]
        return I

    def write(self, path: str, I: np.ndarray = None, driver: str = None) -> None:
        """
        Write image data to file.

        Arguments:
            path: File path to write to.
            I: Image data.
                If `None`, the original image data is read.
            driver: GDAL drivers to use (see https://gdal.org/drivers/raster).
                If `None`, tries to guess the driver based on the file extension.
        """
        if I is None:
            I = self.read()
        helpers.write_raster(a=I, path=path, driver=driver)

    def plot(self, **kwargs: Any) -> matplotlib.image.AxesImage:
        """
        Plot image data.

        By default, the image is plotted with the upper-left corner of the
        upper-left pixel at (0, 0).

        Arguments:
            **kwargs: Arguments passed to :func:`matplotlib.pyplot.imshow`.

        Example:
            >>> import matplotlib.pyplot as plt
            >>> img = Image('tests/AK10b_20141013_020336.JPG')
            >>> img.plot()
            <matplotlib.image.AxesImage object at ...>
            >>> plt.show()  #doctest: +SKIP
            >>> plt.close()
        """
        I = self.read()
        kwargs = {"origin": "upper", "extent": (0, I.shape[1], I.shape[0], 0), **kwargs}
        return matplotlib.pyplot.imshow(I, **kwargs)

    def set_plot_limits(self) -> None:
        """
        Set limits of current plot axes to image extent.

        Example:
            >>> import matplotlib.pyplot as plt
            >>> img = Image('tests/AK10b_20141013_020336.JPG')
            >>> img.plot()
            <matplotlib.image.AxesImage ...>
            >>> ax = plt.gca()
            >>> ax.set_xlim(0, 1)
            (0.0, 1.0)
            >>> ax.set_ylim(1, 0)
            (1.0, 0.0)
            >>> img.set_plot_limits()
            >>> ax.get_xlim() == (0, img.cam.imgsz[0])
            True
            >>> ax.get_ylim() == (img.cam.imgsz[1], 0)
            True
        """
        self.cam.set_plot_limits()

    def project(self, cam: Camera, method: str = "linear") -> np.ndarray:
        """
        Project image into another camera.

        Arguments:
            cam: Target camera.
            method: Interpolation method (either 'linear' or 'nearest').

        Raises:
            ValueError: Camera positions are not equal.

        Example:
            For example, turn the original camera slightly right and up.
            The image is now projected to the lower-left corner of the image frame.

            >>> import matplotlib.pyplot as plt
            >>> img = Image('tests/AK10b_20141013_020336.JPG')
            >>> cam = img.cam.copy()
            >>> cam.viewdir = (5, 4, 0)
            >>> I = img.project(cam, method="nearest")
            >>> plt.imshow(I)
            <matplotlib.image.AxesImage ...>
            >>> plt.show()  #doctest: +SKIP
            >>> plt.close()
        """
        if not all(cam.xyz == self.cam.xyz):
            raise ValueError(
                "Source and target cameras have different positions ('xyz')"
            )
        # Construct grid in target image
        u = np.linspace(0.5, cam.imgsz[0] - 0.5, cam.imgsz[0])
        v = np.linspace(0.5, cam.imgsz[1] - 0.5, cam.imgsz[1])
        U, V = np.meshgrid(u, v)
        uv = np.column_stack((U.flatten(), V.flatten()))
        # Project grid out target image
        dxyz = cam.uv_to_xyz(uv)
        # Project target grid onto source image (flip for RegularGridInterpolator)
        pvu = np.fliplr(self.cam.xyz_to_uv(dxyz, directions=True))
        # Construct grid in source image
        if cam.imgsz[0] == self.cam.imgsz[0]:
            pu = u
        else:
            pu = np.linspace(0.5, self.cam.imgsz[0] - 0.5, self.cam.imgsz[0])
        if cam.imgsz[1] == self.cam.imgsz[1]:
            pv = v
        else:
            pv = np.linspace(0.5, self.cam.imgsz[1] - 0.5, self.cam.imgsz[1])
        # Prepare source image
        I = self.read()
        if I.ndim < 3:
            I = np.expand_dims(I, axis=2)
        pI = np.full((cam.imgsz[1], cam.imgsz[0], I.shape[2]), np.nan, dtype=I.dtype)
        # Sample source image at target grid
        for i in range(pI.shape[2]):
            f = scipy.interpolate.RegularGridInterpolator(
                (pv, pu), I[:, :, i], method=method, bounds_error=False
            )
            pI[:, :, i] = f(pvu).reshape(pI.shape[0:2])
        return pI
