import warnings
from datetime import datetime
import numpy as np
import piexif
import PIL.Image
import scipy.interpolate
import shutil
import os.path
import matplotlib.pyplot
import helper
import copy

class Camera(object):
    """
    A `Camera` converts between 3D world coordinates and 2D image coordinates.

    By default, cameras are initialized at the origin (0, 0, 0), parallel with the horizon (xy-plane), and pointed north (+y).
    All attributes are coerced to numpy arrays during initialization or when individually set.
    The focal length in pixels (`f`) is calculated from `fmm` and `sensorsz` if both are provided.
    The principal point offset in pixels (`c`) is calculated from `cmm` and `sensorsz` if both are provided.
    If `vector` is provided, all arguments are ignored except `sensorsz`,
    which is saved for later calculation of `fmm` and `cmm`.

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
        sensorsz (array_like): Sensor size in millimters [nx, ny]
        fmm (array_like): Focal length in millimeters [fx, fy]
        cmm (array_like): Principal point offset from center in millimiters [dx, dy]
        R (array): Rotation matrix equivalent of `viewdir`.
            Assumes the camera is initially oriented with +z pointing up, +x east, and +y north.
        cameraMatrix (array): Camera matrix in OpenCV format
        distCoeffs (array): Distortion coefficients (k, p) in OpenCV format
        original_vector (array): Original value of `vector`
    """

    def __init__(self, vector=None, xyz=[0, 0, 0], viewdir=[0, 0, 0], imgsz=[100, 100], f=[100, 100], c=[0, 0], k=[0, 0, 0, 0, 0, 0], p=[0, 0],
        sensorsz=None, fmm=None, cmm=None):
        self.vector = np.full(20, np.nan, dtype=float)
        self.sensorsz = sensorsz
        if vector is not None:
            self.vector = np.asarray(vector, dtype=float)[0:20]
        else:
            self.xyz = xyz
            self.viewdir = viewdir
            self.imgsz = imgsz
            if (fmm is not None or cmm is not None) and sensorsz is None:
                raise ValueError("'fmm' or 'cmm' provided without 'sensorsz'")
            if sensorsz is not None and fmm is not None:
                self.f = helper.format_list(fmm, length=2) * self.imgsz / self.sensorsz
            else:
                self.f = f
            if sensorsz is not None and cmm is not None:
                self.c = helper.format_list(cmm, length=2) * self.imgsz / self.sensorsz
            else:
                self.c = c
            self.k = k
            self.p = p
        self.original_vector = self.vector.copy()

    @classmethod
    def read(cls, path, **kwargs):
        """
        Read Camera from JSON.

        See `.write()` for the reverse.

        Arguments:
            path (str): Path to JSON file
            **kwargs (dict): Additional arguments passed to `Camera()`.
                These take precedence over arguments read from file.
        """
        json_args = helper.read_json(path)
        for key in json_args.keys():
            value = [np.nan if item is None else item for item in json_args[key]]
            if all([item is np.nan for item in value]):
                value = None
            json_args[key] = value
        args = helper.merge_dicts(json_args, kwargs)
        return cls(**args)

    # ---- Properties (dependent) ----

    @property
    def xyz(self):
        return self.vector[0:3]

    @xyz.setter
    def xyz(self, value):
        self.vector[0:3] = helper.format_list(value, length=3, default=0)

    @property
    def viewdir(self):
        return self.vector[3:6]

    @viewdir.setter
    def viewdir(self, value):
        self.vector[3:6] = helper.format_list(value, length=3, default=0)

    @property
    def imgsz(self):
        return self.vector[6:8]

    @imgsz.setter
    def imgsz(self, value):
        self.vector[6:8] = helper.format_list(value, length=2)

    @property
    def f(self):
        return self.vector[8:10]

    @f.setter
    def f(self, value):
        self.vector[8:10] = helper.format_list(value, length=2)

    @property
    def c(self):
        return self.vector[10:12]

    @c.setter
    def c(self, value):
        self.vector[10:12] = helper.format_list(value, length=2, default=0)

    @property
    def k(self):
        return self.vector[12:18]

    @k.setter
    def k(self, value):
        self.vector[12:18] = helper.format_list(value, length=6, default=0)

    @property
    def p(self):
        return self.vector[18:20]

    @p.setter
    def p(self, value):
        self.vector[18:20] = helper.format_list(value, length=2, default=0)

    @property
    def sensorsz(self):
        if self._sensorsz is not None:
            return self._sensorsz
        else:
            return np.full(2, np.nan)

    @sensorsz.setter
    def sensorsz(self, value):
        self._sensorsz = helper.format_list(value, length=2)

    @property
    def fmm(self):
        return self.f * self.sensorsz / self.imgsz

    @property
    def cmm(self):
        return self.c * self.sensorsz / self.imgsz

    @property
    def R(self):
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

    @property
    def cameraMatrix(self):
        """
        OpenCV camera matrix.
        """
        return np.array([
            [self.f[0], 0, self.c[0] + self.imgsz[0] / 2],
            [0, self.f[1], self.c[1] + self.imgsz[1] / 2],
            [0, 0, 1]])

    @property
    def distCoeffs(self):
        """
        OpenCV distortion coefficients.
        """
        return np.hstack((self.k[0:2], self.p[0:2], self.k[2:]))

    # ---- Methods (static) ----

    @staticmethod
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

    @staticmethod
    def get_scale_from_size(old_size, new_size):
        """
        Return the scale factor that achieves the target image size.

        Arguments:
            old_size (array-like): Initial image size [nx, ny]
            new_size (array-like): Target image size [nx, ny]

        Returns:
            float: Scale factor, or `None` if target size cannot be achieved
        """
        old_size = np.floor(np.array(old_size) + 0.5)
        new_size = np.floor(np.array(new_size) + 0.5)
        if all(new_size == old_size):
            return 1.0
        scale_bounds = new_size / old_size
        if scale_bounds[0] == scale_bounds[1]:
            return scale_bounds[0]
        def err(scale):
            return np.sum(np.abs(np.floor(scale * old_size + 0.5) - new_size))
        fit = scipy.optimize.minimize(err, x0=scale_bounds.mean(), bounds=[scale_bounds])
        if err(fit['x']) == 0:
            return fit['x']
        else:
            return None

    # ---- Methods (public) ----

    def copy(self):
        """
        Return a copy.

        The original state of the new object (`original_vector`)
        is set to the current state of the old object.
        """
        return Camera(vector=self.vector.copy(), sensorsz=self.sensorsz)

    def reset(self):
        """
        Reset to original state.
        """
        self.vector = self.original_vector.copy()

    def write(self, path=None, attributes=None):
        """
        Write or return Camera as JSON.

        Arguments:
            path (str): Path of file to write to.
                If `None` (default), a JSON-formatted string is returned.
            attributes (list): Camera attributes to include.
                If `None` (default), all arguments to `Camera()` are included
                other than `self` and `vector`.
        """
        if attributes is None:
            attributes = Camera.__init__.__code__.co_varnames[2:]
        lines = ['    "' + name + '": ' + str(list(getattr(self, name))).replace("nan", "null") for name in attributes]
        json_string = "{\n" + ",\n".join(lines) + "\n}"
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

    def resize(self, size, force=False):
        """
        Resize the camera.

        Image size (`imgsz`), focal length (`f`), and principal point offset (`c`) are scaled accordingly.

        Arguments:
            size: Either scale factor (scalar) or target image size (length-2)
            force: Whether to use the target image size even if it cannot be achieved
                by a scalar scale factor
        """
        scale1d = np.atleast_1d(size)
        if len(scale1d) > 1 and force:
            # Use target size without checking for scalar scale factor
            new_size = scale1d
        else:
            if len(scale1d) > 1:
                # Compute scalar scale factor (if one exists)
                scale1d = Camera.get_scale_from_size(self.imgsz, scale1d)
                if scale1d is None:
                    raise ValueError("Target size cannot be achieved with scalar scale factor")
            new_size = np.floor(scale1d * self.imgsz + 0.5)
        scale2d = new_size / self.imgsz
        self.imgsz = np.round(self.imgsz * scale2d) # ensure whole numbers
        self.f *= scale2d
        self.c *= scale2d

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

    def infront(self, xyz, directions=False):
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

    def inframe(self, uv):
        """
        Test whether image coordinates are in or on the image frame.

        Arguments:
            uv (array) Image coordinates (Nx2)
        """
        return np.all((uv >= 0) & (uv <= self.imgsz), axis=1)

    def inview(self, xyz, directions=False):
        """
        Test whether world coordinates are within view.

        Arguments:
            xyz (array): World coordinates (Nx3) or normalized camera coordinates (Nx2)
            directions (bool): Whether `xyz` are absolute coordinates (`False`) or ray directions (`True`)
        """
        uv = self.project(xyz, directions=directions)
        return self.inframe(uv)

    def centers(self, mode=None):
        """
        Return image coordinates of all grid cell centers.

        Arguments:
            mode (str): Image coordinates, returned as either

                - "vectors": x (nx, ) and y (ny, ) coordinates
                - "grids": x (ny, nx) and y (ny, nx) coordinates
                - otherwise: x, y coordinates (ny * nx, 2)
        """
        u = np.linspace(0.5, self.imgsz[0] - 0.5, int(self.imgsz[0]))
        v = np.linspace(0.5, self.imgsz[1] - 0.5, int(self.imgsz[1]))
        if mode == "vectors":
            return u, v
        U, V = np.meshgrid(u, v)
        if mode == "grids":
            return U, V
        return np.column_stack((U.flatten(), V.flatten()))

    def rasterize(self, uv, values, fun=np.mean):
        """
        Convert points to a raster image.

        Arguments:
            uv (array): Image point coordinates (Nx2)
            values (array): Point values
            fun (function): Aggregate function to apply to values of overlapping points
        """
        is_in = self.inframe(uv)
        shape = (int(self.imgsz[1]), int(self.imgsz[0]))
        return helper.rasterize_points(uv[is_in, 1].astype(int), uv[is_in, 0].astype(int),
            values[is_in], shape, fun=fun)

    # ---- Methods (private) ----

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
        if not any(self.k) and not any(self.p):
            return xy
        else:
            dxy = xy.copy()
            r2 = np.sum(xy**2, axis=1)
            if any(self.k):
                dxy *= self._radial_distortion(r2)
            if any(self.p):
                dxy += self._tangential_distortion(xy, r2)
            return dxy

    def _undistort(self, xy, method="oulu", **params):
        """
        Remove distortion from camera coordinates.

        TODO: Quadtree 2-D bisection
        https://stackoverflow.com/questions/3513660/multivariate-bisection-method

        Arguments:
            xy (array): Camera coordinates (Nx2)
        """
        # X = (X' - dt) / dr
        if not any(self.k) and not any(self.p):
            return xy
        elif self.k[0] and not any(self.k[1:]) and not any(self.p):
            return self._undistort_k1(xy)
        elif method == "lookup":
            return self._undistort_lookup(xy, **params)
        elif method == "oulu":
            return self._undistort_oulu(xy, **params)
        elif method == "regulafalsi":
            return self._undistort_regulafalsi(xy, **params)

    def _undistort_k1(self, xy):
        """
        Remove 1st order radial distortion.

        Uses the closed-form solution to the cubic equation when
        the only non-zero distortion coefficient is k1 (self.k[0]).

        Arguments:
            xy (array): Camera coordinates (Nx2)
        """
        # If only k1 is nonzero, use closed form solution.
        # Cubic roots solution from Numerical Recipes in C 2nd Edition:
        # http://apps.nrbook.com/c/index.html (pages 183-185)
        # Solve for undistorted radius in polar coordinates
        # r^3 + r / k1 - r'/k1 = 0
        phi = np.arctan2(xy[:, 1], xy[:, 0])
        Q = - 1 / (3 * self.k[0])
        R = - xy[:, 0] / (2 * self.k[0] * np.cos(phi)) # r' = y / cos(phi)
        has_three_roots = R**2 < Q**3
        r = np.full(len(xy), np.nan)
        if np.any(has_three_roots):
          th = np.arccos(R[has_three_roots] / Q**1.5)
          r[has_three_roots] = -2 * np.sqrt(Q) * np.cos((th - 2 * np.pi) / 3)
        has_one_root = ~has_three_roots
        if np.any(has_one_root):
          A = - np.sign(R[has_one_root]) * (np.abs(R[has_one_root]) + np.sqrt(R[has_one_root]**2 - Q**3))**(1.0 / 3)
          B = np.where(A == 0, 0, Q / A)
          r[has_one_root] = A + B
        return np.column_stack((np.cos(phi), np.sin(phi))) * r[:, None]

    def _undistort_lookup(self, xy, density=1):
        """
        Remove distortion by table lookup.

        Creates a grid of test coordinates and applies distortion,
        then interpolates undistorted coordinates from the result
        with scipy.interpolate.LinearNDInterpolator().

        NOTE: Remains stable in extreme distortion, but slow
        for large lookup tables.

        Arguments:
            xy (array): Camera coordinates (Nx2)
            density (float): Grid points per pixel (approximate)
        """
        # Estimate undistorted camera coordinate bounds
        uv_edges = self.imgsz * np.array([
            [0, 0], [0.5, 0], [1, 0], [1, 0.5],
            [1, 1], [0.5, 1], [0, 1], [0, 0.5]
        ])
        xyu_edges = (uv_edges - (self.imgsz / 2 + self.c)) / self.f
        xyd_edges = self._distort(xyu_edges)
        # Build undistorted camera coordinates on regular grid
        ux = np.linspace(
            min(xyu_edges[:, 0].min(), xyd_edges[:, 0].min()),
            max(xyu_edges[:, 0].max(), xyd_edges[:, 0].max()),
            int(density * self.imgsz[0]))
        uy = np.linspace(
            min(xyu_edges[:, 1].min(), xyd_edges[:, 1].min()),
            max(xyu_edges[:, 1].max(), xyd_edges[:, 1].max()),
            int(density * self.imgsz[1]))
        UX, UY = np.meshgrid(ux, uy)
        uxy = np.column_stack((UX.flatten(), UY.flatten()))
        # Distort grid
        dxy = self._distort(uxy)
        # Interpolate distortion removal from gridded results
        # NOTE: Cannot use faster grid interpolation because dxy is not regular
        return scipy.interpolate.griddata(dxy, uxy, xy, method="linear")

    def _undistort_oulu(self, xy, iterations=20, tolerance=0):
        """
        Remove distortion by the iterative Oulu University method.

        See http://www.vision.caltech.edu/bouguetj/calib_doc/ (comp_distortion_oulu.m)

        NOTE: Converges very quickly in normal cases, but fails for extreme distortion.

        Arguments:
            xy (array): Camera coordinates (Nx2)
            iterations (int): Maximum number of iterations
            tolerance (float): Approximate pixel displacement in x and y below which
                to exit early, or `0` to disable early exit
        """
        uxy = xy # initial guess
        for n in range(iterations):
            r2 = np.sum(uxy**2, axis=1)
            if any(self.p) and not any(self.k):
                uxy = xy - self._tangential_distortion(uxy, r2)
            elif any(self.k) and not any(self.k):
                uxy = xy / self._radial_distortion(r2)
            else:
                uxy = (xy - self._tangential_distortion(uxy, r2)) / self._radial_distortion(r2)
            if tolerance > 0 and np.all((np.abs(self._distort(uxy) - xy)) < tolerance / self.f.mean()):
                break
        return uxy

    def _undistort_regulafalsi(self, xy, iterations=100, tolerance=0):
        """
        Remove distortion by iterative regula falsi (false position) method.

        See https://en.wikipedia.org/wiki/False_position_method

        NOTE: Almost always converges, but may require many iterations for extreme distortion.

        Arguments:
            xy (array): Camera coordinates (Nx2)
            iterations (int): Maximum number of iterations
            tolerance (float): Approximate pixel displacement in x and y (for all points)
                below which to exit early, or `0` to disable early exit (default)
        """
        # Start at center of image (distortion free)
        x1 = np.zeros(xy.shape, dtype=float)
        y1 = -xy
        # Then try halfway towards distorted coordinate
        # (more stable to approach solution from image center)
        x2 = xy / 2
        y2 = self._distort(x2) - xy
        uxy = np.full(xy.shape, np.nan)
        for n in range(iterations):
            dy = y2 - y1
            not_converged = np.all(dy != 0, axis=1)
            if tolerance > 0:
                not_converged &= np.any(np.abs(y2) > tolerance / self.f.mean())
            if n == 0:
                mask = np.ones(len(xy), dtype=bool)
            converged = np.zeros(mask.shape, dtype=bool)
            converged[mask] = ~not_converged
            uxy[converged] = x2[~not_converged]
            mask[mask] = not_converged
            x1 = x1[not_converged]
            y1 = y1[not_converged]
            x2 = x2[not_converged]
            y2 = y2[not_converged]
            if not np.any(not_converged):
                break
            x3 = (x1 * y2 - x2 * y1) / dy[not_converged]
            y3 = self._distort(x3) - xy[mask]
            x1 = x2
            y1 = y2
            x2 = x3
            y2 = y3
        uxy[mask] = x2
        return uxy

    def _reversible(self):
        """
        Test whether the camera model is reversible.

        Checks whether distortion produces a monotonically increasing result.
        If not, distorted coordinates are non-unique and cannot be reversed.

        TODO: Derive this explicitly from the distortion parameters.
        """
        xy_row = np.column_stack((
            np.linspace(-self.imgsz[0] / (2 * self.f[0]), self.imgsz[0] / (2 * self.f[0]), int(self.imgsz[0])),
            np.zeros(int(self.imgsz[0]))))
        dxy = self._distort(xy_row)
        continuous_row = np.all(dxy[1:, 0] >= dxy[:-1, 0])
        xy_col = np.column_stack((
            np.zeros(int(self.imgsz[1])),
            np.linspace(-self.imgsz[1] / (2 * self.f[1]), self.imgsz[1] / (2 * self.f[1]), int(self.imgsz[1]))))
        dxy = self._distort(xy_col)
        continuous_col = np.all(dxy[1:, 1] >= dxy[:-1, 1])
        return continuous_row and continuous_col

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
            if self.size is None:
                # NOTE: Is this still necessary?
                size = np.array(PIL.Image.open(path).size, dtype=float)
                self.set_tag('PixelXDimension', size[0])
                self.set_tag('PixelYDimension', size[1])
        else:
            self.tags = {}
            self.size = None

    @property
    def size(self):
        width = self.get_tag('PixelXDimension')
        height = self.get_tag('PixelYDimension')
        if width and height:
            return np.array((width, height), dtype=float)
        else:
            return None

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
        tag = self.get_tag('ExposureTime')
        if tag:
            return float(tag[0]) / tag[1]
        else:
            return None

    @property
    def aperture(self):
        tag = self.get_tag('FNumber')
        if tag:
            return float(tag[0]) / tag[1]
        else:
            return None

    @property
    def iso(self):
        tag = self.get_tag('ISOSpeedRatings')
        if tag:
            return float(tag)
        else:
            return None

    @property
    def fmm(self):
        tag = self.get_tag('FocalLength')
        if tag:
            return float(tag[0]) / tag[1]
        else:
            return None

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
        cam (Camera, dict, or str): Camera object or arguments passed to `Camera()`.
            If string, assumes a JSON file path and reads arguments from file.
            If `imgsz` is missing, the actual size of the image is used.
            If `f` is missing, an attempt is made to specify both `fmm` and `sensorsz`.
            If `fmm` is missing, it is read from the metadata, and if `sensorsz` is missing,
            `Camera.get_sensor_size()` is called with `make` and `model` read from the metadata.
        datetime (datetime): Capture date and time. Unless specified, it is read from the metadata.

    Attributes:
        path (str): Path to the image file
        exif (Exif): Image metadata object
        datetime (datetime): Capture date and time
        cam (Camera): Camera object
    """

    def __init__(self, path, cam=None, datetime=None):
        self.path = path
        self.exif = Exif(path=path)
        if datetime:
            self.datetime = datetime
        else:
            self.datetime = self.exif.datetime
        # TODO: Throw warning if `imgsz` has different aspect ratio than file size.
        if isinstance(cam, Camera):
            self.cam = cam
        else:
            if isinstance(cam, str):
                cam = helper.read_json(cam)
            elif isinstance(cam, dict):
                cam = copy.deepcopy(cam)
            elif cam is None:
                cam = dict()
            if not cam.has_key('vector'):
                if not cam.has_key('imgsz'):
                    cam['imgsz'] = self.exif.size
                if not cam.has_key('f'):
                    if not cam.has_key('fmm'):
                        cam['fmm'] = self.exif.fmm
                    if not cam.has_key('sensorsz'):
                        cam['sensorsz'] = Camera.get_sensor_size(self.exif.make, self.exif.model)
            self.cam = Camera(**cam)
        self.I = None

    def copy(self):
        """
        Return a copy.

        Copies camera, rereads exif from file, and does not copy cached image data (self.I).
        """
        return Image(path=self.path, cam=self.cam.copy())

    def read(self):
        """
        Read image data from file.

        If the camera image size (self.cam.imgsz) differs from the original image size (self.exif.size),
        the image is resized to fit the camera image size.
        The result is cached (`self.I`) and reused only if it matches the camera image size, or,
        if not set, the original image size.
        To clear the cache, set `self.I` to `None`.
        """
        if self.I is not None:
            size = np.flipud(self.I.shape[0:2])
        has_cam_size = all(~np.isnan(self.cam.imgsz))
        if ((self.I is None) or
            (not has_cam_size and not np.array_equal(size, self.exif.size)) or
            (has_cam_size and not np.array_equal(size, self.cam.imgsz))):
            # Read file
            im = PIL.Image.open(self.path)
            if has_cam_size and not np.array_equal(self.cam.imgsz, self.exif.size):
                # Resize to match camera model
                im = im.resize(size=self.cam.imgsz.astype(int), resample=PIL.Image.BILINEAR)
            self.I = np.array(im)
        return self.I

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
                im = PIL.Image.fromarray(self.read())
            else:
                im = PIL.Image.fromarray(I)
            # For JPEG file extensions, see https://stackoverflow.com/a/23424597/8161503
            if ext.lower() in ('.jpg', '.jpeg', '.jpe', '.jif', '.jfif', '.jfi'):
                exif = self.exif.copy()
                exif.set_tag('PixelXDimension', im.size[0])
                exif.set_tag('PixelYDimension', im.size[1])
                im.save(path, exif=exif.dump(), **params)
            else:
                warnings.warn("Writing EXIF to non-JPEG file is not supported.")
                im.save(path, **params)

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
            extent=(0, I.shape[1], I.shape[0], 0)
        matplotlib.pyplot.imshow(I, origin=origin, extent=extent, **params)

    def set_plot_limits(self):
        matplotlib.pyplot.xlim(0, self.cam.imgsz[0])
        matplotlib.pyplot.ylim(self.cam.imgsz[1], 0)

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
