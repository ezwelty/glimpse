import operator
import warnings
from datetime import datetime
import numpy as np
import exifread
import scipy.misc
import scipy.interpolate

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
    
    Attributes:
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
        vector (array): Flat vector of all camera attributes [xyz, viewdir, imgsz, f, c, k, p]
        
    """

    def __init__(self, xyz=[0, 0, 0], viewdir=[0, 0, 0], imgsz=[100, 100], f=[100, 100], c=[0, 0], k=[0, 0, 0, 0, 0, 0], p=[0, 0],
        fmm=None, sensorsz=None, vector=None):
        if vector is not None:
            self.vector = vector
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

    # ---- Properties (independent) ----

    xyz = property(operator.attrgetter('_xyz'))
    viewdir = property(operator.attrgetter('_viewdir'))
    imgsz = property(operator.attrgetter('_imgsz'))
    f = property(operator.attrgetter('_f'))
    c = property(operator.attrgetter('_c'))
    k = property(operator.attrgetter('_k'))
    p = property(operator.attrgetter('_p'))

    @xyz.setter
    def xyz(self, value):
        self._xyz = _format_list(value, length=3, default=0)

    @viewdir.setter
    def viewdir(self, value):
        self._viewdir = _format_list(value, length=3, default=0)

    @imgsz.setter
    def imgsz(self, value):
        self._imgsz = _format_list(value, length=2)

    @f.setter
    def f(self, value):
        self._f = _format_list(value, length=2)

    @c.setter
    def c(self, value):
        self._c = _format_list(value, length=2, default=0)

    @k.setter
    def k(self, value):
        self._k = _format_list(value, length=6, default=0)

    @p.setter
    def p(self, value):
        self._p = _format_list(value, length=2, default=0)
    
    # ---- Properties (dependent) ----
    
    @property
    def _R(self):
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

    @property
    def vector(self):
        return np.concatenate((self.xyz, self.viewdir, self.imgsz, self.f, self.c, self.k, self.p))
    
    @vector.setter
    def vector(self, value):
        temp = np.array(value[0:20], dtype=float)
        self._xyz = temp[0:3]
        self._viewdir = temp[3:6]
        self._imgsz = temp[6:8]
        self._f = temp[8:10]
        self._c = temp[10:12]
        self._k = temp[12:18]
        self._p = temp[18:20]
    
    # ---- Methods (public) ----

    def idealize(self, copy=True):
        """
        Set distortion to zero.
        
        Radial distortion (`k`), tangential distortion (`p`), and principal point offset (`c`) are set to zero.
        
        Arguments:
            copy (bool): Whether to return result as new `Camera`
        
        Returns:
            Camera: New object (if `copy=True`)
        """
        if copy:
            return Camera(xyz=self.xyz, viewdir=self.viewdir, imgsz=self.imgsz, f=self.f)
        else:
            self.k = [0, 0, 0, 0, 0, 0]
            self.p = [0, 0]
            self.c = [0, 0]

    def resize(self, scale, copy=True):
        """
        Resize a camera by a scale factor.
        
        Image size (`imgsz`), focal length (`f`), and principal point offset (`c`) are scaled accordingly.
        
        Arguments:
            scale (scalar): Scale factor
            copy (bool): Whether to return result as new `Camera`
        
        Returns:
            Camera: New object (if `copy=True`)
        """
        target_size = np.round(self.imgsz * float(scale))
        scale = target_size / self.imgsz
        if copy:
            return Camera(
                xyz=self.xyz, viewdir=self.viewdir, imgsz=self.imgsz * scale,
                f=self.f * scale, c=self.c * scale, k=self.k, p=self.p
                )
        else:
            self.imgsz *= scale
            self.f *= scale
            self.c *= scale
    
    def optimize(self, uv, xyz, params={'viewdir': True}, directions=False, copy=True, tol=None, options=None):
        """
        Calibrate a camera from paired image-world coordinates.
        
        Points `uv` and `xyz` are matched by row index.
        The Levenberg-Marquardt algorithm is used to find the camera parameters that minimize the distance between
        `uv` and the projection of `xyz` into the camera.
        
        Arguments:
            uv (array): Image coordinates (Nx2)
            xyz (array): World coordinates (Nx3)
            params (dict): Parameters to optimize. For example:
                
                - {'viewdir': True} : All `viewdir` elements
                - {'viewdir': 0} : First `viewdir` element
                - {'viewdir': [0, 1]} : First and second `viewdir` elements
            
            directions (bool): Whether `xyz` are absolute coordinates (False) or ray directions (True).
                If True, 'xyz' cannot be in `params`.
            copy (bool): Whether to return result as new `Camera`
            tol (float): Tolerance for termination (see scipy.optimize.root)
            options (dict): Solver options (see scipy.optimize.root)
            
        Returns:
            Camera: New object (if `copy=True`)
        """
        uv = np.asarray(uv, dtype = float)
        xyz = np.asarray(xyz, dtype = float)
        if directions and 'xyz' in params:
            raise ValueError("'xyz' cannot be in `params` when `directions` is True")
        mask = self._vector_mask(params)
        cam = Camera(vector=self.vector)
        def minfun(values):
            cam._update_vector(mask, values)
            return cam._projerror_points(uv, xyz, directions=directions).flatten()
        result = scipy.optimize.root(minfun, cam.vector[mask], method='lm', tol=tol, options=options)
        cam._update_vector(mask, result['x'])
        if not result['success']:
            raise RuntimeError(result['message'])
        if copy:
            return cam
        else:
            self.vector = cam.vector
    
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
        z = np.dot(dxyz, self._R.T)[:, 2]
        return z > 0

    def _inframe(self, uv):
        """
        Test whether image coordinates are in or on the image frame.
        
        Arguments:
            uv (array) Image coordinates (Nx2)
        """
        return np.all((uv >= 0) & (uv <= self.imgsz), axis=1)

    def _projerror_points(self, uv, xyz, directions=False, normalize=False):
        """
        Calculate pixel reprojection errors for points.
        
        Points `uv` and `xyz` are matched by row index.
        
        Arguments:
            uv (array): Image coordinates (Nx2)
            xyz: (array): World coordinates (Nx3) or camera coordinates (Nx2)
            directions (bool): Whether `xyz` are absolute coordinates (False) or ray directions (True)
            normalize (bool): Whether to return pixels (False) or normalize by mean focal length (True)
        """
        puv = self.project(xyz, directions=directions)
        duv = puv - uv
        if normalize:
            duv /= self.f.mean()
        return duv

    def _vector_mask(self, params={}):
        names = ['xyz', 'viewdir', 'imgsz', 'f', 'c', 'k', 'p']
        indices = [0, 3, 6, 8, 10, 12, 18, 20]
        selected = np.zeros(20, dtype = bool)
        for name, value in params.items():
            if (value or value == 0) and name in names:
                start = names.index(name)
                if value is True:
                    selected[indices[start]:indices[start + 1]] = True
                else:
                    value = np.array(value)
                    selected[indices[start] + value] = True
        return selected
    
    def _update_vector(self, mask, values):
        vector = self.vector
        vector[mask] = values
        self.vector = vector
    
    # def _clip_line_inview(self, xyz):
    #     # in = cam.inview(xyz);
    #     # lines = splitmat(xyz, in);
    
    # def _projerror_lines(self, luv, lxyz, directions=False, normalize=False):
    #     """
    #     Calculate pixel reprojection errors for lines.
    #     Lines are matched by proximity.
    #     luv: (list:array:float) List of image coordinates [Nx2, ...]
    #     lxyz: (list:array:float) List of world coordinates [Nx3, ...] or camera coordinates [Nx2, ...]
    #     directions: (bool) Whether absolute coordinates (False) or ray directions (True)
    #     normalize: (bool) Whether to return pixels (False) or normalize by mean focal length (True)
    #     """
    #     # lxyz:
    #     # Extract line segments within camera view
    #     # Project to camera
    #     # Resample? by length * max(self.f)
    #     # Project to image
    #     # Convert to points
    #     # Decimate?
    #     # Extract line segments within image frame

    #     # lxyz vs. luv
    #     # Euclidean distance matrix
    #     # Return nearest distance for each point
    #     # -or-
    #     # Try again: Nearest distance to line?

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
                    r = -2 * np.sqrt(Q) * np.cos((th - 2 * pi) / 3)
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
        xyz_c = np.dot(dxyz, self._R.T)
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
        dxyz = np.dot(xy_z, self._R)
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
        path (str): Path to image file
        details (bool): Whether to process makernote tags and extract the thumbnail image
    
    Attributes:
        tags (dict): Image file metadata, as returned by exifread.process_file()
        size (array): Image size in pixels [nx, ny]
        datetime (datetime): Capture date and time
        fmm (float): Focal length in millimeters
        shutter (float): Shutter speed in seconds
        aperture (float): Aperture size as f-number
        iso (float): Film speed
        make (str): Camera make
        model (str): Camera model
    """
    
    def __init__(self, path, details=False):
        f = open(path, 'rb')
        self.tags = exifread.process_file(f, details=details)
    
    @property
    def size(self):
        width = self.parse_tag('EXIF ExifImageWidth')
        height = self.parse_tag('EXIF ExifImageLength')
        if width and height:
            return np.array([width, height])
        else:
            return None
    
    @property
    def datetime(self):
        datetime_str = self.parse_tag('EXIF DateTimeOriginal')
        subsec_str = self.parse_tag('EXIF SubSecTimeOriginal')
        if datetime_str and not subsec_str:
            return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
        elif datetime_str and subsec_str:
            return datetime.strptime(datetime_str + "." + subsec_str.zfill(6), "%Y:%m:%d %H:%M:%S.%f")
        else:
            return None
    
    @property
    def shutter(self):
        return self.parse_tag('EXIF ExposureTime')
        
    @property
    def aperture(self):
        return self.parse_tag('EXIF FNumber')
        
    @property
    def iso(self):
        return self.parse_tag('EXIF ISOSpeedRatings')
    
    @property
    def fmm(self):
        return self.parse_tag('EXIF FocalLength')
    
    @property
    def make(self):
        return self.parse_tag('Image Make')
        
    @property
    def model(self):
        return self.parse_tag('Image Model')

    def parse_tag(self, key):
        """Parse an exif tag and return its value (or None if missing)."""
        if not self.tags.has_key(key):
            return None
        if self.tags[key]: 
            value = self.tags[key].values
        else:
            return None
        if type(value) is list:
            value = value[0]
        if isinstance(value, exifread.Ratio):
            return float(value.num) / value.den
        else:
            return value

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
    
    def read(self, size=None):
        """
        Read image data from file.
        
        Arguments:
            size (scalar or list): Fraction of original size (scalar) or target size [nx, ny].
                If `None` (default), self.cam.imgsz is used (if set) or the image is returned as-is.
        """
        I = scipy.misc.imread(self.path)
        if size is None and self.cam.imgsz is not None:
            size = self.cam.imgsz
        if size is not None:
            size = np.array(size)
            if size.size > 1:
                # Switch from (x, y) to (y, x) for scipy.misc.imresize()
                size = size.astype(int)[::-1]
            else:
                size = float(size)
            # TODO: Throw warning if `imgsz` has different aspect ratio than file size.
            I = scipy.misc.imresize(I, size=size)
        return I
    
    def project(self, cam, method="linear"):
        """
        Project image data into a `Camera`.
        
        Arguments:
            cam (Camera): Target `Camera`
            method (str): Interpolation method, either "linear" or "nearest"
        """
        if not np.all(cam.xyz == self.cam.xyz):
            raise ValueError("Current and target cameras must have the same position ('xyz')")
        # Construct grid in target image
        u = np.linspace(0.5, cam.imgsz[0] - 0.5, int(cam.imgsz[0]))
        v = np.linspace(0.5, cam.imgsz[1] - 0.5, int(cam.imgsz[1]))
        U, V = np.meshgrid(u, v)
        uv = np.column_stack((U.flatten(), V.flatten()))
        # Project grid out target image
        dxyz = cam.invproject(uv)
        # Project grid onto image
        puv = self.cam.project(dxyz, directions=True)
        pvu = np.fliplr(puv)
        # Sample image at grid
        I = self.read()
        if I.ndim < 3:
            I = np.expand_dims(I, axis=2)
        pI = np.full((int(cam.imgsz[1]), int(cam.imgsz[0]), I.shape[2]), np.nan, dtype=I.dtype)
        pu = np.linspace(0.5, self.cam.imgsz[0] - 0.5, int(self.cam.imgsz[0]))
        pv = np.linspace(0.5, self.cam.imgsz[1] - 0.5, int(self.cam.imgsz[1]))
        for i in range(pI.shape[2]):
            f = scipy.interpolate.RegularGridInterpolator((pv, pu), I[:, :, i], method=method, bounds_error=False)
            pI[:, :, i] = f(pvu).reshape(I.shape[0:2])
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
    make_model = make.strip() + " " + model.strip()
    sensor_sizes = { # mm
        'NIKON CORPORATION NIKON D2X': [23.7, 15.7], # https://www.dpreview.com/reviews/nikond2x/2
        'NIKON CORPORATION NIKON D200': [23.6, 15.8], # https://www.dpreview.com/reviews/nikond200/2
        'NIKON CORPORATION NIKON D300S': [23.6, 15.8], # https://www.dpreview.com/reviews/nikond300s/2
        'NIKON E8700': [8.8, 6.6], # https://www.dpreview.com/reviews/nikoncp8700/2
        'Canon Canon EOS 20D': [22.5, 15.0], # https://www.dpreview.com/reviews/canoneos20d/2
        'Canon Canon EOS 40D': [22.2, 14.8], # https://www.dpreview.com/reviews/canoneos40d/2
    }
    if sensor_sizes.has_key(make_model):
        return sensor_sizes[make_model]
    else:
        raise KeyError("No sensor size found for " + make_model)

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
