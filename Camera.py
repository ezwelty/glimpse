import numpy as np
import operator
import warnings

class Camera(object):
    """
    A `Camera` converts between 3D world coordinates and 2D image coordinates.

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
    """

    def __init__(self, xyz=[0, 0, 0], viewdir=[0, 0, 0], imgsz=[100, 100], f=[100, 100], c=[0, 0], k=[0, 0, 0, 0, 0, 0], p=[0, 0], sensorsz=[]):
        """
        Create a camera.

        All camera properties are coerced to numpy arrays.
        """
        self.xyz = xyz
        self.viewdir = viewdir
        self.imgsz = imgsz
        self.sensorsz = sensorsz
        self.f = f
        self.c = c
        self.k = k
        self.p = p

    # ---- Properties (independent) ----

    xyz = property(operator.attrgetter('_xyz'))
    viewdir = property(operator.attrgetter('_viewdir'))
    imgsz = property(operator.attrgetter('_imgsz'))
    sensorsz = property(operator.attrgetter('_sensorsz'))
    f = property(operator.attrgetter('_f'))
    c = property(operator.attrgetter('_c'))
    k = property(operator.attrgetter('_k'))
    p = property(operator.attrgetter('_p'))

    @xyz.setter
    def xyz(self, value):
        self._xyz = get_float_array(value, n=3, fill=True)

    @viewdir.setter
    def viewdir(self, value):
        self._viewdir = get_float_array(value, n=3, fill=True)

    @imgsz.setter
    def imgsz(self, value):
        self._imgsz = get_float_array(value, n=2, fill=False)

    @sensorsz.setter
    def sensorsz(self, value):
        self._sensorsz = get_float_array(value, n=2, fill=False)

    @f.setter
    def f(self, value):
        self._f = get_float_array(value, n=2, fill=False)

    @c.setter
    def c(self, value):
        self._c = get_float_array(value, n=2, fill=True)

    @k.setter
    def k(self, value):
        self._k = get_float_array(value, n=6, fill=True)

    @p.setter
    def p(self, value):
        self._p = get_float_array(value, n=3, fill=True)

    # ---- Properties (dependent) ----

    @property
    def fmm(self):
        if len(self.f) > 0 and len(self.sensorsz) > 0 and len(self.imgsz) > 0:
            return(self.f * self.sensorsz / self.imgsz)
        else:
            raise AttributeError("Missing required attributes (f, sensorsz, imgsz)")

    @fmm.setter
    def fmm(self, value):
        if len(self.sensorsz) > 0 and len(self.imgsz) > 0:
            fmm = get_float_array(value, n=2, fill=False)
            self.f = fmm * self.imgsz / self.sensorsz
        else:
            raise AttributeError("Missing required attributes (sensorsz, imgsz)")

    @property
    def R(self):
        if len(self.viewdir) > 0:
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
            return(np.array([
                [C[0] * C[2] + S[0] * S[1] * S[2],  C[0] * S[1] * S[2] - C[2] * S[0], -C[1] * S[2]],
                [C[2] * S[0] * S[1] - C[0] * S[2],  S[0] * S[2] + C[0] * C[2] * S[1], -C[1] * C[2]],
                [C[1] * S[0]                     ,  C[0] * C[1]                     ,  S[1]       ]
            ]))
        else:
            raise AttributeError("Missing required attributes (viewdir)")

    # ---- Methods (public) ----

    def idealize(self):
        """
        Set camera distortion to zero.
        """
        self.k = [0, 0, 0, 0, 0, 0]
        self.p = [0, 0]
        cam.c = [0, 0]

    def resize(self, scale):
        """
        Scale a camera model by a factor.
        scale: (int|float) Scale factor
        """
        # Get scale
        target_size = np.round(self.imgsz * scale)
        scale = target_size / self.imgsz
        # Apply scale
        self.f *= scale
        self.c *= scale
        self.imgsz *= scale

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
            xy = self.world2camera(xyz, directions=directions)
        else:
            xy = xyz
        xy = self.world2camera(xyz, directions=directions)
        uv = self.camera2image(xy)
        return(uv)

    def invproject(self, uv):
        """
        Project image coordinates to world ray directions.
        (or the intersection with a surface, if provided)
        uv: (array:float) Image coordinates (Nx2)
        """
        xy = self.image2camera(uv)
        xyz = self.camera2world(xy)
        return(xyz)

    # ---- Methods (optimization) ----

    def infront(self, xyz, directions=False):
        """
        Test whether world coordinates are in front of the camera.
        xyz: (array:float) World coordinates (Nx3)
        directions: (bool) Whether absolute coordinates (False) or ray directions (True)
        """
        if directions:
            dxyz = xyz
        else:
            dxyz = xyz - self.xyz
        z = np.dot(dxyz, self.R.T)[:, 2]
        return(z > 0)

    def inframe(self, uv):
        """
        Test whether image coordinates are in (or on) the image frame.
        uv: (array:float) Image coordinates (Nx2)
        """
        return(np.all((uv >= 0) & (uv <= self.imgsz), axis=1))

    # def clip_line_inview(self, xyz):
    #     # in = cam.inview(xyz);
    #     # lines = splitmat(xyz, in);

    def projerror_points(self, uv, xyz, directions=False, normalize=False):
        """
        Calculate pixel reprojection errors for points.
        Points are matched by index.
        uv: (array:float) Image coordinates (Nx2)
        xyz: (array:float) World coordinates (Nx3) or camera coordinates (Nx2)
        directions: (bool) Whether absolute coordinates (False) or ray directions (True)
        normalize: (bool) Whether to return pixels (False) or normalize by mean focal length (True)
        """
        puv = self.project(xyz, directions=directions)
        duv = puv - uv
        if normalize:
            duv /= self.f.mean()
        return(duv)

    # def projerror_lines(self, luv, lxyz, directions=False, normalize=False):
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

    # ---- Methods (private) ----

    def radial_distortion(self, r2):
        """
        Compute radial distortion multipler [dr].
        r2: (array:float) Squared radius of camera coordinates (Nx1)
        """
        # dr = (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) / (1 + k4 * r^2 + k5 * r^4 + k6 * r^6)
        dr = 1
        if any(self.k[0:3]):
            dr += self.k[0] * r2 + self.k[1] * r2**2 + self.k[2] * r2**3
        if any(self.k[3:6]):
            dr /= 1 + self.k[3] * r2 + self.k[4] * r2**2 + self.k[5] * r2**3
        return(dr[:, None]) # column

    def tangential_distortion(self, xy, r2):
        """
        Compute tangential distortion additive [dtx, dty].
        xy: (array:float) Camera coordinates (Nx2)
        r2: (array:float) Squared radius of camera coordinates (Nx1)
        """
        # dtx = 2xy * p1 + p2 * (r^2 + 2x^2)
        # dty = p1 * (r^2 + 2y^2) + 2xy * p2
        xty = xy[:, 0] * xy[:, 1]
        dtx = 2 * xty * self.p[0] + self.p[1] * (r2 + 2 * xy[:, 0]**2)
        dty = self.p[0] * (r2 + 2 * xy[:, 1]**2) + 2 * xty * self.p[1]
        dt = np.column_stack((dtx, dty))
        return(dt)

    def distort(self, xy):
        """
        Apply distortion to normalized camera coordinates.
        xy: (array:float) Camera coordinates (Nx2)
        """
        # X' = dr * X + dt
        if any(self.k) or any(self.p):
            r2 = np.sum(xy**2, axis=1)
        if any(self.k):
            xy *= self.radial_distortion(r2)
        if any(self.p):
            xy += self.tangential_distortion(xy, r2)
        return(xy)

    def undistort(self, xy):
        """
        Remove distortion from normalized camera coordinates.
        xy: (array:float) Camera coordinates (Nx2)
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
                        xy = xyi - self.tangential_distortion(xy, r2)
                    else:
                        xy = xyi
                    if any(self.k):
                        xy /= self.radial_distortion(r2)
        return(xy)

    def world2camera(self, xyz, directions = False):
        """
        Project world coordinates to camera coordinates.
        xyz: (array:float) World coordinates (Nx3)
        directions: (bool) Whether absolute coordinates (False) or ray directions (True)
        """
        if not directions:
            # Convert coordinates to ray directions
            xyz -= self.xyz
        xyz = np.dot(xyz, self.R.T)
        # Normalize by perspective division
        xy = xyz[:, 0:2] / xyz[:, 2][:, None]
        # Set points behind camera to NaN
        behind = xyz[:, 2] <= 0
        xy[behind, :] = np.nan
        return(xy)

    def camera2world(self, xy):
        """
        Project camera coordinates to world ray directions.
        xy: (array:float) Camera coordinates (Nx2)
        """
        ones = np.ones((xy.shape[0], 1))
        xy_z = np.c_[xy, ones]
        dxyz = np.dot(xy_z, self.R)
        return(dxyz)

    def camera2image(self, xy):
        """
        Project camera to image coordinates
        xy: (array:float) Camera coordinates (Nx2)
        """
        xy = self.distort(xy)
        uv = xy * self.f + (self.imgsz / 2 + self.c)
        return(uv)

    def image2camera(self, uv):
        """
        Project image to camera coordinates
        uv: (array:float) Image coordinates (Nx2)
        """
        xy = (uv - (self.imgsz / 2 + self.c)) / self.f
        xy = self.undistort(xy)
        return(xy)


# ---- Static methods ----

def get_float_array(obj, n=1, fill=True):
    """
    Coerce object to 1-d array of floating point numbers.
    obj: Object
    n: (int) Array length
    fill: (bool) Whether to repeat array (False) or fill with zeroes (True)
    """
    if isinstance(obj, np.ndarray):
        obj = list(obj)
    obj = get_float_list(obj, n=n, fill=fill)
    return(np.array(obj))

def get_float_list(obj, n=1, fill=True):
    """
    Coerce object to a list of floating point numbers.
    obj: Object
    n: (int) List length
    fill: (bool) Whether to repeat list (False) or fill list with zeroes (True)
    """
    if not isinstance(obj, list):
        obj = [obj]
    if len(obj) < n:
        if fill:
            # Fill remaining slots with 0
            obj.extend([0] * (n - len(obj)))
        else:
            # Repeat list
            if len(obj) == 0:
                return([])
            assert n % len(obj) == 0
            obj = obj * ((n) / len(obj))
    return([float(i) for i in obj[0:n]])
