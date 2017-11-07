import warnings
from datetime import datetime
import numpy as np
import piexif
import PIL.Image
import scipy.interpolate
import shutil
import os.path
import json

class Camera(object):
    """
    A `Camera` converts between 3D world coordinates and 2D image coordinates.
    
    By default, cameras are initialized at the origin (0, 0, 0), parallel with the horizon (xy-plane), and pointed north (+y).
    All attributes are coerced to numpy arrays during initialization or when individually set.
    The focal length in pixels (`f`) is calculated from `fmm` and `sensorsz` if these are both provided.
    If `vector` is provided, all other arguments are ignored. 
    
    Arguments:
        fmm (array_like): Focal length in millimeters [fx, fy]
        sensorsz (array_like): Sensor size in millimters [nx, ny]
        path (str): Path to JSON camera file (see `Camera.write()`).
            Takes precedence if not `None` (default).
    
    Attributes:
        vector (array): Flat vector of all camera attributes [xyz, viewdir, imgsz, f, c, k, p]
        xyz (array): Position in world coordinates [x, y, z]
        viewdir (array): View direction in degrees [yaw, pitch, roll]

            - yaw: clockwise rotation about z-axis (0 = look north)
            - pitch: rotation from horizon (+ look up, - look down)
            - roll: rotation about optical axis (+ down right, - down left, from behind)
        
        imgsz (array): Image size in pixels [nx, ny]
        f (array): Focal length in pixels [fx, fy]
        c (array): Principal point offset from center in pixels [dx, dy]
        k (array): Radial distortion coefficients [k1, ..., k6]
        p (array): Tangential distortion coefficients [p1, p2]
        R (array): Rotation matrix equivalent of `viewdir`.
            Assumes the camera is initially oriented with +z pointing up, +x east, and +y north.
    """
    
    def __init__(self, xyz=[0, 0, 0], viewdir=[0, 0, 0], imgsz=[100, 100], f=[100, 100], c=[0, 0], k=[0, 0, 0, 0, 0, 0], p=[0, 0],
        fmm=None, sensorsz=None, vector=None, path=None):
        self.vector = np.full(20, np.nan, dtype=float)
        if path is not None:
            with open(path, "r") as fp:
                args = json.load(fp)
            for key in args.keys():
                # Replace None with np.nan
                value = [np.nan if item is None else item for item in args[key]]
                setattr(self, key, value)
        elif vector is not None:
            self.vector = np.asarray(vector, dtype=float)[0:20]
        else:
            self.xyz = xyz
            self.viewdir = viewdir
            self.imgsz = imgsz
            if fmm and sensorsz:
                self.f = fmm_to_fpx(fmm, sensorsz, imgsz)
            else:
                self.f = f
            self.c = c
            self.k = k
            self.p = p
    
    # ---- Properties (dependent) ----
    
    @property
    def xyz(self):
        return self.vector[0:3]
    
    @xyz.setter
    def xyz(self, value):
        self.vector[0:3] = _format_list(value, length=3, default=0)
    
    @property
    def viewdir(self):
        return self.vector[3:6]
    
    @viewdir.setter
    def viewdir(self, value):
        self.vector[3:6] = _format_list(value, length=3, default=0)
    
    @property
    def imgsz(self):
        return self.vector[6:8]
    
    @imgsz.setter
    def imgsz(self, value):
        self.vector[6:8] = _format_list(value, length=2)
    
    @property
    def f(self):
        return self.vector[8:10]
    
    @f.setter
    def f(self, value):
        self.vector[8:10] = _format_list(value, length=2)
    
    @property
    def c(self):
        return self.vector[10:12]
    
    @c.setter
    def c(self, value):
        self.vector[10:12] = _format_list(value, length=2, default=0)
    
    @property
    def k(self):
        return self.vector[12:18]
    
    @k.setter
    def k(self, value):
        self.vector[12:18] = _format_list(value, length=6, default=0)
    
    @property
    def p(self):
        return self.vector[18:20]
    
    @p.setter
    def p(self, value):
        self.vector[18:20] = _format_list(value, length=2, default=0)
    
    @property
    def R(self):
        if self.viewdir is not None:
            # Initial rotations of camera reference frame
            # (camera +z pointing up, with +x east and +y north)
            # Point camera north: -90 deg counterclockwise rotation about x-axis
            #   ri = [1 0 0; 0 cosd(-90) sind(-90); 0 -sind(-90) cosd(-90)];
            # (camera +z now pointing north, with +x east and +y down)
            # yaw: counterclockwise rotation about y-axis (relative to north, from above: +cw, - ccw)
            #   ry = [C1 0 -S1; 0 1 0; S1 0 C1];
            # pitch: counterclockwise rotation about x-axis (relative to horizon: + up, - down)
            #   rp = [1 0 0; 0 C2 S2; 0 -S2 C2];
            # roll: counterclockwise rotation about z-axis (from behind camera: + ccw, - cw)
            #   rr = [C3 S3 0; -S3 C3 0; 0 0 1];
            # Apply all rotations in order
            #   R = rr * rp * ry * ri;
            radians = np.deg2rad(self.viewdir)
            C = np.cos(radians)
            S = np.sin(radians)
            return np.array([
                [C[0] * C[2] + S[0] * S[1] * S[2],  C[0] * S[1] * S[2] - C[2] * S[0], -C[1] * S[2]],
                [C[2] * S[0] * S[1] - C[0] * S[2],  S[0] * S[2] + C[0] * C[2] * S[1], -C[1] * C[2]],
                [C[1] * S[0]                     ,  C[0] * C[1]                     ,  S[1]       ]
            ])
        else:
            return None
    
    # ---- Methods (public) ----
    
    def copy(self):
        """
        Return a copy.
        """
        return Camera(vector=self.vector)
    
    def write(self, path=None):
        """
        Write or return Camera as JSON.
        
        Arguments:
            path (str): Path of file to write to.
                if `None` (default), a JSON-formatted string is returned.
        """
        keys = ['xyz', 'viewdir', 'imgsz', 'f', 'c', 'k', 'p']
        key_strings = ['    "' + key + '": ' + str(list(getattr(self, key))).replace("nan", "null") for key in keys]
        json_string = "{\n" + ",\n".join(key_strings) + "\n}"
        if path:
            with open(path, "w") as fp:
                fp.write(json_string)
            return None
        else:
            return json_string
    
    def idealize(self):
        """
        Set distortion to zero.
        
        Radial distortion (`k`), tangential distortion (`p`), and principal point offset (`c`) are set to zero.
        """
        self.k = np.zeros(6, dtype=float)
        self.p = np.zeros(2, dtype=float)
        self.c = np.zeros(2, dtype=float)
    
    def resize(self, scale):
        """
        Resize a camera by a scale factor.
        
        Image size (`imgsz`), focal length (`f`), and principal point offset (`c`) are scaled accordingly.
        
        Arguments:
            scale (scalar): Scale factor
        """
        target_size = np.round(self.imgsz * float(scale))
        scale = target_size / self.imgsz
        self.imgsz *= scale
        self.f *= scale
        self.c *= scale
    
    def project(self, xyz, directions=False):
        """
        Project world coordinates to image coordinates.
        
        Arguments:
            xyz (array): World coordinates (Nx3) or camera coordinates (Nx2)
            directions (bool): Whether absolute coordinates (False) or ray directions (True)
        
        Returns:
            array: Image coordinates (Nx2)
        """
        if xyz.shape[1] == 3:
            xy = self._world2camera(xyz, directions=directions)
        else:
            xy = xyz
        uv = self._camera2image(xy)
        return uv
    
    def invproject(self, uv):
        """
        Project image coordinates to world ray directions.
        
        Arguments:
            uv (array): Image coordinates (Nx2)
        
        Returns:
            array: World ray directions relative to camera position (Nx3)
        """
        xy = self._image2camera(uv)
        xyz = self._camera2world(xy)
        return xyz
    
    # ---- Methods (private) ----
    
    def _infront(self, xyz, directions=False):
        """
        Test whether world coordinates are in front of the camera.
        
        Arguments:
            xyz (array): World coordinates (Nx3)
            directions (bool): Whether `xyz` are absolute coordinates (`False`) or ray directions (`True`)
        """
        if directions:
            dxyz = xyz
        else:
            dxyz = xyz - self.xyz
        z = np.dot(dxyz, self.R.T)[:, 2]
        return z > 0
    
    def _inframe(self, uv):
        """
        Test whether image coordinates are in or on the image frame.
        
        Arguments:
            uv (array) Image coordinates (Nx2)
        """
        return np.all((uv >= 0) & (uv <= self.imgsz), axis=1)
    
    def _radial_distortion(self, r2):
        """
        Compute the radial distortion multipler `dr`.
        
        Arguments:
            r2 (array): Squared radius of camera coordinates (Nx1)
        """
        # dr = (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) / (1 + k4 * r^2 + k5 * r^4 + k6 * r^6)
        dr = 1
        if any(self.k[0:3]):
            dr += self.k[0] * r2 + self.k[1] * r2**2 + self.k[2] * r2**3
        if any(self.k[3:6]):
            dr /= 1 + self.k[3] * r2 + self.k[4] * r2**2 + self.k[5] * r2**3
        return dr[:, None] # column
    
    def _tangential_distortion(self, xy, r2):
        """
        Compute tangential distortion additive `[dtx, dty]`.
        
        Arguments:
            xy (array): Camera coordinates (Nx2)
            r2 (array): Squared radius of `xy` (Nx1)
        """
        # dtx = 2xy * p1 + p2 * (r^2 + 2x^2)
        # dty = p1 * (r^2 + 2y^2) + 2xy * p2
        xty = xy[:, 0] * xy[:, 1]
        dtx = 2 * xty * self.p[0] + self.p[1] * (r2 + 2 * xy[:, 0]**2)
        dty = self.p[0] * (r2 + 2 * xy[:, 1]**2) + 2 * xty * self.p[1]
        return np.column_stack((dtx, dty))
    
    def _distort(self, xy):
        """
        Apply distortion to camera coordinates.
        
        Arguments:
            xy (array): Camera coordinates (Nx2)
        """
        # X' = dr * X + dt
        if any(self.k) or any(self.p):
            r2 = np.sum(xy**2, axis=1)
        if any(self.k):
            xy *= self._radial_distortion(r2)
        if any(self.p):
            xy += self._tangential_distortion(xy, r2)
        return xy
    
    def _undistort(self, xy):
        """
        Remove distortion from camera coordinates.
        
        Arguments:
            xy (array): Camera coordinates (Nx2)
        """
        # X = (X' - dt) / dr
        if any(self.k) or any(self.p):
            if self.k[0] < -0.5:
                # May fail for large negative k1.
                warnings.warn('Large negative k1 (' + str(round(self.k[0], 3)) + '). Undistort may fail.')
            if self.k[0] and all(self.k[1:] == 0) and all(self.p == 0):
                # If only k1 is nonzero, use closed form solution.
                # Cubic roots solution from Numerical Recipes in C 2nd Edition:
                # http://apps.nrbook.com/c/index.html (pages 183-185)
                phi = np.arctan2(xy[:, 1], xy[:, 0])
                Q = -1 / (3 * self.k[0])
                R = -xy[:, 1] / (2 * self.k[0] * np.cos(phi))
                if self.k[0] < 0:
                    # For negative k1
                    th = np.arccos(R / np.sqrt(Q**3))
                    r = -2 * np.sqrt(Q) * np.cos((th - 2 * np.pi) / 3)
                else:
                    # For positive k1
                    A = (np.sqrt(R**2 - Q**3) - R)**(1 / 3)
                    B = Q * (1 / A)
                    r = A + B
                xy = np.column_stack((np.cos(phi), np.sin(phi))) * r[:, None]
                xy = np.real(xy)
            else:
                # Use iterative solution.
                xyi = xy # initial quess
                for n in range(1, 20):
                    r2 = np.sum(xy**2, axis=1)
                    if any(self.p):
                        xy = xyi - self._tangential_distortion(xy, r2)
                    else:
                        xy = xyi
                    if any(self.k):
                        xy /= self._radial_distortion(r2)
        return xy
    
    def _world2camera(self, xyz, directions=False):
        """
        Project world coordinates to camera coordinates.
        
        Arguments:
            xyz (array): World coordinates (Nx3)
            directions (bool): Whether `xyz` are absolute coordinates (False) or ray directions (True)
        """
        if directions:
            dxyz = xyz
        else:
            # Convert coordinates to ray directions
            dxyz = xyz - self.xyz
        xyz_c = np.dot(dxyz, self.R.T)
        # Normalize by perspective division
        xy = xyz_c[:, 0:2] / xyz_c[:, 2][:, None]
        # Set points behind camera to NaN
        behind = xyz_c[:, 2] <= 0
        xy[behind, :] = np.nan
        return xy
    
    def _camera2world(self, xy):
        """
        Project camera coordinates to world ray directions.
        
        Arguments:
            xy (array): Camera coordinates (Nx2)
        """
        ones = np.ones((xy.shape[0], 1))
        xy_z = np.c_[xy, ones]
        dxyz = np.dot(xy_z, self.R)
        return dxyz
    
    def _camera2image(self, xy):
        """
        Project camera to image coordinates.
        
        Arguments:
            xy (array): Camera coordinates (Nx2)
        """
        xy = self._distort(xy)
        uv = xy * self.f + (self.imgsz / 2 + self.c)
        return uv
    
    def _image2camera(self, uv):
        """
        Project image to camera coordinates.
        
        Arguments:
            uv (array): Image coordinates (Nx2)
        """
        xy = (uv - (self.imgsz / 2 + self.c)) / self.f
        xy = self._undistort(xy)
        return xy

class Exif(object):
    """
    `Exif` is a container and parser for image file metadata.

    Arguments:
        path (str): Path to image file.
            If `None` (default), an empty Exif object is returned.
        thumbnail (bool): Whether to retain the image thumbnail
    
    Attributes:
        tags (dict): Image file metadata, as returned by piexif.load()
        size (array): Image size in pixels [nx, ny]
        datetime (datetime): Capture date and time
        fmm (float): Focal length in millimeters
        shutter (float): Shutter speed in seconds
        aperture (float): Aperture size as f-number
        iso (float): Film speed
        make (str): Camera make
        model (str): Camera model
    """
    
    def __init__(self, path=None, thumbnail=False):
        if path:
            self.tags = piexif.load(path, key_is_name=False)
            if not thumbnail:
                self.tags.pop('thumbnail', None)
                self.tags.pop('1st', None)
            self.size = np.array(PIL.Image.open(path).size, dtype=float)
        else:
            self.tags = {}
            self.size = None
    
    @property
    def datetime(self):
        datetime_str = self.get_tag('DateTimeOriginal')
        subsec_str = self.get_tag('SubSecTimeOriginal')
        if datetime_str and not subsec_str:
            return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
        elif datetime_str and subsec_str:
            return datetime.strptime(datetime_str + "." + subsec_str.zfill(6), "%Y:%m:%d %H:%M:%S.%f")
        else:
            return None
    
    @property
    def shutter(self):
        dividend, divisor = self.get_tag('ExposureTime')
        return float(dividend) / divisor
    
    @property
    def aperture(self):
        dividend, divisor = self.get_tag('FNumber')
        return float(dividend) / divisor
    
    @property
    def iso(self):
        return float(self.get_tag('ISOSpeedRatings'))
    
    @property
    def fmm(self):
        dividend, divisor = self.get_tag('FocalLength')
        return float(dividend) / divisor
    
    @property
    def make(self):
        return self.get_tag('Make', group='Image')
    
    @property
    def model(self):
        return self.get_tag('Model', group='Image')
    
    def get_tag(self, tag, group='Exif'):
        """
        Return the value of a tag (or None if missing).
        
        Arguments:
            tag (str): Tag name
            group (str): Group name ('Exif', 'Image', or 'GPS')
        """
        code = getattr(getattr(piexif, group + 'IFD'), tag)
        if group is 'Image':
            group = '0th'
        if not self.tags.has_key(group) or not self.tags[group].has_key(code):
            return None
        else:
            return self.tags[group][code]
    
    def set_tag(self, tag, value, group='Exif'):
        """
        Set the value of a tag, adding it if missing.
        
        Arguments:
            tag (str): Tag name
            value (object): Tag value
            group (str): Group name ('Exif', 'Image', or 'GPS')
        """
        code = getattr(getattr(piexif, group + 'IFD'), tag)
        if group is 'Image':
            group = '0th'
        if not self.tags.has_key(group):
            self.tags[group] = {}
        self.tags[group][code] = value
    
    def dump(self):
        """
        Return exif as bytes.
        """
        return piexif.dump(self.tags)
    
    def copy(self):
        """
        Return a copy.
        """
        exif = Exif()
        exif.tags = self.tags
        exif.size = self.size
        return exif

class Image(object):
    """
    An `Image` describes the camera settings and resulting image captured at a particular time.

    Arguments:
        path (str): Path to image file
        datetime (datetime): Capture date and time. Unless specified, it is read from the metadata.
        camera_args (dict): Arguments passed to `Camera()`.
            If `imgsz` is missing, the actual size of the image is used.
            If `f` is missing, an attempt is made to specify both `fmm` and `sensorsz`.
            If `fmm` is missing, it is read from the metadata, and if `sensorsz` is missing,
            `get_sensor_size()` is called with `make` and `model` read from the metadata.
    
    Attributes:
        path (str): Path to the image file
        exif (Exif): Image metadata object
        datetime (datetime): Capture date and time
        cam (Camera): Camera object
    """
    
    def __init__(self, path, datetime=None, camera_args=None):
        self.path = path
        self.exif = Exif(path=path)
        if datetime:
            self.datetime = datetime
        else:
            self.datetime = self.exif.datetime
        if not camera_args:
            camera_args = {}
        # TODO: Throw warning if `imgsz` has different aspect ratio than file size.
        if not camera_args.has_key('imgsz'):
            camera_args['imgsz'] = self.exif.size
        if not camera_args.has_key('f'):
            if not camera_args.has_key('fmm'):
                camera_args['fmm'] = self.exif.fmm
            if not camera_args.has_key('sensorsz'):
                camera_args['sensorsz'] = get_sensor_size(self.exif.make, self.exif.model)
        self.cam = Camera(**camera_args)
    
    def read(self):
        """
        Read image data from file.
        
        If the camera image size (self.cam.imgsz) differs from the original image size (self.exif.size),
        the image is resized to fit the camera image size.
        """
        im = PIL.Image.open(self.path)
        if self.cam.imgsz is not None and not np.array_equal(self.cam.imgsz, self.exif.size):
            # Resize to match camera model
            im = im.resize(size=self.cam.imgsz.astype(int), resample=PIL.Image.BILINEAR)
        return np.array(im)
    
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
        if ext is old_ext and I is None:
            # Copy original file
            shutil.copyfile(self.path, path)
        else:
            if I is None:
                im = PIL.Image.open(self.path)
            else:
                im = PIL.Image.fromarray(I)
            # For JPEG file extensions, see https://stackoverflow.com/a/23424597/8161503
            if ext in ('.jpg', '.jpeg', '.jpe', '.jif', '.jfif', '.jfi'):
                exif = self.exif.copy()
                exif.set_tag('PixelXDimension', im.size[0])
                exif.set_tag('PixelYDimension', im.size[1])
                im.save(path, exif=exif.dump(), **params)
            else:
                warnings.warn("Writing EXIF to non-JPEG file is not supported.")
                im.save(path, **params)
    
    def project(self, cam, method="linear"):
        """
        Project image into another `Camera`.
        
        Arguments:
            cam (Camera): Target `Camera`
            method (str): Interpolation method, either "linear" or "nearest"
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
        if cam.imgsz[0] is self.cam.imgsz[0]:
            pu = u
        else:
            pu = np.linspace(0.5, self.cam.imgsz[0] - 0.5, int(self.cam.imgsz[0]))
        if cam.imgsz[1] is self.cam.imgsz[1]:
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

# ---- Static methods (public) ----

def get_sensor_size(make, model):
    """
    Get a camera model's CCD sensor width and height in mm.
    
    Data is from Digital Photography Review (https://dpreview.com).
    See also https://www.dpreview.com/articles/8095816568/sensorsizes.
    
    Arguments:
        make (str): Camera make (EXIF Make)
        model (str): Camera model (EXIF Model)
        
    Return:
        list: Camera sensor width and height in mm
    """
    sensor_sizes = { # mm
        'NIKON CORPORATION NIKON D2X': [23.7, 15.7], # https://www.dpreview.com/reviews/nikond2x/2
        'NIKON CORPORATION NIKON D200': [23.6, 15.8], # https://www.dpreview.com/reviews/nikond200/2
        'NIKON CORPORATION NIKON D300S': [23.6, 15.8], # https://www.dpreview.com/reviews/nikond300s/2
        'NIKON E8700': [8.8, 6.6], # https://www.dpreview.com/reviews/nikoncp8700/2
        'Canon Canon EOS 20D': [22.5, 15.0], # https://www.dpreview.com/reviews/canoneos20d/2
        'Canon Canon EOS 40D': [22.2, 14.8], # https://www.dpreview.com/reviews/canoneos40d/2
    }
    if make and model:
        make_model = make.strip() + " " + model.strip()
    else:
        make_model = ""
    if sensor_sizes.has_key(make_model):
        return sensor_sizes[make_model]
    else:
        raise KeyError("No sensor size found for: " + make_model)

def fmm_to_fpx(fmm, sensorsz, imgsz):
    """
    Convert focal length in millimeters to pixels.
    
    Arguments:
        fmm (array-like): Focal length in millimeters [fx, fy]
        sensorsz (array-like): Sensor size in millimeters [nx, ny]
        imgsz (array-like): Image size in pixels [nx, ny]
    """
    fmm, sensorsz, imgsz = (
        _format_list(i, length=2) for i in (fmm, sensorsz, imgsz)
        )
    return fmm * imgsz / sensorsz
    
def fpx_to_fmm(fpx, sensorsz, imgsz):
    """
    Convert focal length in pixels to millimeters.
    
    Arguments:
        fpx (array-like): Focal length in pixels [fx, fy]
        sensorsz (array-like): Sensor size in millimeters [nx, ny]
        imgsz (array-like): Image size in pixels [nx, ny]
    """
    fpx, sensorsz, imgsz = (
        _format_list(i, length=2) for i in (fpx, sensorsz, imgsz)
        )
    return fpx * sensorsz / imgsz

# ---- Static methods (private) ----

def _format_list(obj, length=1, default=None, dtype=float, ltype=np.array):
    """
    Format a list-like object.
    
    Arguments:
        obj (object): Object
        length (int): Output object length
        default (scalar): Default element value.
            If `None`, `obj` is repeated to achieve length `length`.
        dtype (type): Data type to coerce list elements to.
            If `None`, data is left as-is.
        ltype (type): List type to coerce object to.
    """
    if obj is None:
        return obj
    try:
        obj = list(obj)
    except TypeError:
        obj = [obj]
    if len(obj) < length:
        if default is not None:
            # Fill remaining slots with 0
            obj.extend([default] * (length - len(obj)))
        else:
            # Repeat list
            if len(obj) > 0:
                assert length % len(obj) == 0
                obj *= length / len(obj)
    if dtype:
        obj = [dtype(i) for i in obj[0:length]]
    if ltype:
        obj = ltype(obj)
    return obj
