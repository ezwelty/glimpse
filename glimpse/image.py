"""
Read, write, and manipulate photographic images.
"""
import shutil
import os
import copy
import warnings
import numpy as np
import scipy.interpolate
import matplotlib.pyplot
import osgeo.gdal
import PIL
import sharedmem
from . import helpers
from .exif import Exif
from .camera import Camera

class Image(object):
    """
    Photographic image and the settings that gave rise to the image.

    Arguments:
        path (str): Path to image
        cam (:class:`Camera`, dict, or str): Camera or arguments passed to
            Camera constructors:

                - if `dict`: Arguments passed to :class:`Camera()`.
                - if `str`: File path passed to :meth:`Camera.read`

            If 'imgsz' is missing, it is read from **exif**.
            If 'f', 'fmm', and 'sensorsz' are missing, 'fmm' is read from
            **exif** and 'sensorsz' is gotten from :meth:`Camera.get_sensor_size`
            with 'make' and 'model' from **exif**.
        exif (:class:`Exif`): Image metadata.
            If `None`, it is read from **path** with :class:`Exif()`.
        datetime (datetime.datetime): Capture date and time.
            If `None`, it is read from **exif.datetime**.
        anchor (bool):
        keypoints_path (str):

    Attributes:
        path (str): Path to image
        cam (:class:`Camera`): Camera
        exif (:class:`Exif`): Image metadata
        datetime (datetime.datetime): Capture date and time
        anchor (bool): Whether the camera parameters, especially view direction,
            are known absolutely. "Anchor" images are used as a reference for
            optimizing other images whose camera parameters are not known absolutely.
        keypoints_path (str): Path for caching image keypoints and their descriptors
            to a `pickle` file. Unless specified, defaults to `path` with a '.pkl' extension.
        keypoints: Cached keypoints
        I (numpy.ndarray): Cached image content
    """

    def __init__(self, path, cam=None, exif=None, datetime=None, anchor=False,
        keypoints_path=None):
        self.path = path
        if exif is None:
            exif = Exif(path=path)
        self.exif = exif
        self.anchor = anchor
        if datetime is None:
          datetime = self.exif.datetime
        self.datetime = datetime
        if isinstance(cam, Camera):
            self.cam = cam
        else:
            if isinstance(cam, (bytes, str)):
                cam = helpers.read_json(cam)
            elif isinstance(cam, dict):
                cam = copy.deepcopy(cam)
            elif cam is None:
                cam = dict()
            if 'vector' not in cam:
                if 'imgsz' not in cam:
                    cam['imgsz'] = self.exif.size
                if 'f' not in cam:
                    if 'fmm' not in cam:
                        cam['fmm'] = self.exif.fmm
                    if 'sensorsz' not in cam and self.exif.make and self.exif.model:
                        cam['sensorsz'] = Camera.get_sensor_size(self.exif.make, self.exif.model)
            self.cam = Camera(**cam)
        self.I = None
        self.keypoints = None
        self.keypoints_path = keypoints_path

    def copy(self):
        """
        Return a copy.

        Copies camera, rereads exif from file, and does not copy cached image data (self.I).
        """
        return Image(path=self.path, cam=self.cam.copy())

    def read(self, box=None, cache=True):
        """
        Read image data from file.

        If the camera image size (self.cam.imgsz) differs from the original image size (self.exif.size),
        the image is resized to fit the camera image size.
        The result is cached (`self.I`) and reused only if it matches the camera image size, or,
        if not set, the original image size.
        To clear the cache, set `self.I` to `None`.

        Arguments:
            box (array-like): Crop extent in image coordinates (left, top, right, bottom)
                relative to `self.cam.imgsz`.
                If `cache=True`, the region is extracted from the cached image.
                If `cache=False`, the region is extracted directly from the file
                (faster than reading the entire image).
            cache (bool): Whether to save image in `self.I`
        """
        I = self.I
        if I is not None:
            size = np.flipud(I.shape[0:2])
        has_cam_size = all(~np.isnan(self.cam.imgsz))
        new_I = False
        if ((I is None) or
            (not has_cam_size and any(size != self.exif.size)) or
            (has_cam_size and any(size != self.cam.imgsz))):
            # Wrong size or not cached: Read image from file
            im = osgeo.gdal.Open(self.path)
            args = dict()
            original_size = (im.RasterXSize, im.RasterYSize)
            target_size = self.cam.imgsz.astype(int) if has_cam_size else original_size
            if any(target_size != original_size):
                # Read image into target-sized buffer
                args['buf_xsize'] = target_size[0]
                args['buf_ysize'] = target_size[1]
            if box is not None and not cache:
                # Resize box to image actual size
                scale = np.divide(original_size, target_size)
                # Read image subset
                args['xoff'] = int(round(box[0] * scale[0]))
                args['win_xsize'] = int(round((box[2] - box[0]) * scale[0]))
                args['yoff'] = int(round(box[1] * scale[1]))
                args['win_ysize'] = int(round((box[3] - box[1]) * scale[1]))
            I = np.stack([im.GetRasterBand(i + 1).ReadAsArray(**args)
                for i in range(im.RasterCount)], axis=2)
            if I.shape[2] == 1:
                I = I.squeeze(axis=2)
            if cache:
                # Caching: Cache result
                I = sharedmem.copy(I)
                self.I = I
            new_I = True
        if box is not None and (cache or not new_I):
            # Caching and cropping: Subset cached array
            I = I[box[1]:box[3], box[0]:box[2]]
        return I

    def write(self, path, I=None, **params):
        """
        Write image data to file.

        Arguments:
            path (str): File or directory path to write to.
                If the latter, the original basename is used.
                If the extension is unchanged and `I=None`, the original file is copied.
            I (array): Image data.
                If `None` (default), the original image data is read.
            **params: Additional arguments passed to `PIL.Image.save()`.
                See http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.save
        """
        if os.path.isdir(path):
            # Use original basename
            path = os.path.join(path, os.path.basename(self.path))
        old_ext = os.path.splitext(self.path)[1].lower()
        ext = os.path.splitext(path)[1].lower()
        if ext == old_ext and I is None:
            # Copy original file
            shutil.copyfile(self.path, path)
        else:
            if I is None:
                im = PIL.Image.fromarray(self.read())
            else:
                im = PIL.Image.fromarray(I)
            # For JPEG file extensions, see https://stackoverflow.com/a/23424597/8161503
            if ext in ('.jpg', '.jpeg', '.jpe', '.jif', '.jfif', '.jfi'):
                exif = self.exif.copy()
                exif.set_tag('PixelXDimension', im.size[0])
                exif.set_tag('PixelYDimension', im.size[1])
                im.save(path, exif=exif.dump(), **params)
            else:
                warnings.warn('Writing EXIF to non-JPEG file is not supported')
                im.save(path, **params)

    def read_keypoints(self):
        """
        Return cached keypoints.

        Returns :attr:`keypoints` or reads them from :attr:`keypoints_path` with
        :func:`helpers.read_pickle`.
        Keypoints are expected to be in the form produced by
        :func:`optimize.detect_keypoints`.
        """
        if self.keypoints is None:
            if self.keypoints_path is None:
                warnings.warn('Keypoints path not specified')
                return None
            else:
                try:
                    self.keypoints = helpers.read_pickle(self.keypoints_path)
                except IOError:
                    warnings.warn('No keypoints found at keypoints path')
        return self.keypoints

    def write_keypoints(self):
        """
        Write keypoints to file.

        Writes :attr:`keypoints` to :attr:`keypoints_path` with
        :func:`helpers.write_pickle`.
        """
        if self.keypoints is not None and self.keypoints_path is not None:
            helpers.write_pickle(self.keypoints, path=self.keypoints_path)
        else:
            raise ValueError('No keypoints, or keypoints path not specified')

    def plot(self, origin='upper', extent=None, **params):
        """
        Plot image data.

        By default, the image is plotted with the upper-left corner of the upper-left pixel at (0,0).

        Arguments:
            origin (str): Place the [0, 0] index of the array in either the 'upper' left (default)
                or 'lower' left corner of the axes.
            extent (scalars): Location of the lower-left and upper-right corners (left, right, bottom, top).
                If `None` (default), the corners are positioned at (0, nx, ny, 0).
            **params: Additional arguments passed to `matplotlib.pyplot.imshow`.
        """
        I = self.read()
        if extent is None:
            extent = (0, I.shape[1], I.shape[0], 0)
        matplotlib.pyplot.imshow(I, origin=origin, extent=extent, **params)

    def set_plot_limits(self):
        matplotlib.pyplot.xlim(0, self.cam.imgsz[0])
        matplotlib.pyplot.ylim(self.cam.imgsz[1], 0)

    def project(self, cam, method='linear'):
        """
        Project image into another `Camera`.

        Arguments:
            cam (Camera): Target `Camera`
            method (str): Interpolation method, either 'linear' or 'nearest'
        """
        if not np.all(cam.xyz == self.cam.xyz):
            raise ValueError("Source and target cameras must have the same position ('xyz')")
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
