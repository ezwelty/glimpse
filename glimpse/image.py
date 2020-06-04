"""
Read, write, and manipulate photographic images.
"""
import numpy as np
import scipy.interpolate
import matplotlib.pyplot
import osgeo.gdal
import sharedmem
from . import helpers
from .exif import Exif
from .camera import Camera

class Image(object):
    """
    Photographic image and the settings that gave rise to the image.

    Arguments:
        path (str): Path to image
        cam (:class:`Camera` or dict): Camera or arguments passed to :class:`Camera()`.
            If missing, 'imgsz', 'fmm', and 'sensorsz' are read from **exif**.
        exif (:class:`Exif`): Image metadata. If `None`, read from **path**.
        datetime (datetime.datetime): Capture date and time.
            If `None`, read from **exif**.

    Attributes:
        path (str): Image path
        cam (:class:`Camera`): Camera model
        exif (:class:`Exif`): Image metadata
        datetime (datetime.datetime): Image capture date and time
        I (numpy.ndarray): Cached image content
    """

    def __init__(self, path, cam=None, exif=None, datetime=None):
        self.path = path
        if not cam:
            cam = {}
        if isinstance(cam, dict):
            if not ('imgsz' in cam and 'f' in cam or ('fmm' in cam and 'sensorsz' in cam)):
                exif = exif or Exif(path)
                cam = {
                    'imgsz': exif.imgsz,
                    'fmm': exif.fmm,
                    'sensorsz': exif.sensorsz,
                    **cam
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
    def _path_imgsz(self):
        ds = osgeo.gdal.Open(self.path)
        return ds.RasterXSize, ds.RasterYSize

    @property
    def _cache_imgsz(self):
        if self.I is not None:
            return self.I.shape[1], self.I.shape[0]

    @property
    def _cam_imgsz(self):
        return int(self.cam.imgsz[0]), int(self.cam.imgsz[1])

    def read(self, box=None, cache=True):
        """
        Read image data from file.

        The image is resized as needed to the camera image size
        (`self.cam.imgsz`). The result is cached (`self.I`) and reused only if
        it matches the camera image size. To clear the cache, set `self.I` to
        `None`.

        Arguments:
            box (array-like): Crop extent in image coordinates (left, top, right, bottom)
                relative to `self.cam.imgsz`.
                If `cache=True`, the region is extracted from the cached image.
                If `cache=False`, the region is extracted directly from the file
                (faster than reading the entire image).
            cache (bool): Whether to save image in `self.I`
        """
        size = self._cache_imgsz or self._path_imgsz
        cam_size = self._cam_imgsz
        resize = cam_size != size
        new_I = True
        if self.I is not None and not resize:
            I = self.I
            new_I = False
        else:
            ds = osgeo.gdal.Open(self.path)
            args = {}
            if resize:
                args['buf_xsize'], args['buf_ysize'] = cam_size
            if box and not cache:
                # Resize box to actual image size
                xscale, yscale = size[0] / cam_size[0], size[1] / cam_size[1]
                # Read image subset
                args['xoff'] = int(round(box[0] * xscale))
                args['win_xsize'] = int(round((box[2] - box[0]) * xscale))
                args['yoff'] = int(round(box[1] * yscale))
                args['win_ysize'] = int(round((box[3] - box[1]) * yscale))
            I = np.dstack([ds.GetRasterBand(i + 1).ReadAsArray(**args)
                for i in range(ds.RasterCount)])
            if I.shape[2] == 1:
                I = I.squeeze(axis=2)
            if cache:
                I = sharedmem.copy(I)
                self.I = I
        if box is not None and (cache or not new_I):
            # Caching and cropping: Subset cached array
            I = I[box[1]:box[3], box[0]:box[2]]
        return I

    def write(self, path, I=None, driver=None):
        """
        Write image data to file.

        Arguments:
            path (str): File path to write to.
            I (array): Image data.
                If `None` (default), the original image data is read.
        """
        if I is None:
            I = self.read()
        helpers.write_raster(a=I, path=path, driver=driver)

    def plot(self, **kwargs):
        """
        Plot image data.

        By default, the image is plotted with the upper-left corner of the
        upper-left pixel at (0, 0).

        Arguments:
            **kwargs: Arguments passed to `matplotlib.pyplot.imshow`.
        """
        I = self.read()
        kwargs = {
            'origin': 'upper',
            'extent': (0, I.shape[1], I.shape[0], 0),
            **kwargs
        }
        matplotlib.pyplot.imshow(I, **kwargs)

    def set_plot_limits(self):
        """
        Set limits of current plot to image extent.
        """
        matplotlib.pyplot.xlim(0, self.cam.imgsz[0])
        matplotlib.pyplot.ylim(self.cam.imgsz[1], 0)

    def project(self, cam, method='linear'):
        """
        Project image into another `Camera`.

        Arguments:
            cam (Camera): Target `Camera`
            method (str): Interpolation method, either 'linear' or 'nearest'
        """
        if not all(cam.xyz == self.cam.xyz):
            raise ValueError("Source and target cameras have different positions ('xyz')")
        # Construct grid in target image
        u = np.linspace(0.5, cam.imgsz[0] - 0.5, int(cam.imgsz[0]))
        v = np.linspace(0.5, cam.imgsz[1] - 0.5, int(cam.imgsz[1]))
        U, V = np.meshgrid(u, v)
        uv = np.column_stack((U.flatten(), V.flatten()))
        # Project grid out target image
        dxyz = cam.invproject(uv)
        # Project target grid onto source image (flip for RegularGridInterpolator)
        pvu = np.fliplr(self.cam.project(dxyz, directions=True))
        # Construct grid in source image
        if cam.imgsz[0] == self.cam.imgsz[0]:
            pu = u
        else:
            pu = np.linspace(0.5, self.cam.imgsz[0] - 0.5, int(self.cam.imgsz[0]))
        if cam.imgsz[1] == self.cam.imgsz[1]:
            pv = v
        else:
            pv = np.linspace(0.5, self.cam.imgsz[1] - 0.5, int(self.cam.imgsz[1]))
        # Prepare source image
        I = self.read()
        if I.ndim < 3:
            I = np.expand_dims(I, axis=2)
        pI = np.full((int(cam.imgsz[1]), int(cam.imgsz[0]), I.shape[2]), np.nan, dtype=I.dtype)
        # Sample source image at target grid
        for i in range(pI.shape[2]):
            f = scipy.interpolate.RegularGridInterpolator((pv, pu), I[:, :, i], method=method, bounds_error=False)
            pI[:, :, i] = f(pvu).reshape(pI.shape[0:2])
        return pI
