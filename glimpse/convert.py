from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (
    np, lmfit, sys, lxml, pandas)
from . import (helpers, Camera, optimize)

class MatlabCamera(object):
    """
    Camera model used by the Camera Calibration Toolbox for Matlab.

    See http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.htmlself.

    Attributes:
        imgsz (iterable): Image size in pixels (x, y)
        fc (iterable): Focal length in pixels (x, y)
        cc (iterable): Principal point in pixels (x, y),
            in an image coordinate system where the center of the top left pixel
            is (0, 0)
        kc (iterable): Image distortion coefficients (k1, k2, p1, p2, k3)
        alpha_c (float): Skew coefficient defining the angle between the x and y
            pixel axes
    """

    def __init__(self, imgsz, fc, cc=None, kc=[0, 0, 0, 0, 0], alpha_c=0):
        self.imgsz = imgsz
        self.fc = fc
        if cc is None:
            cc = (np.asarray(imgsz) - 1) / 2
        self.cc = cc
        self.kc = kc
        self.alpha_c = alpha_c

    def _camera2image(self, xy):
        # Compute lens distortion
        r2 = np.sum(xy**2, axis=1)
        dr = self.kc[0] * r2 + self.kc[1] * r2 * r2 + self.k[4] * r2 * r2 * r2
        xty = xy[:, 0] * xy[:, 1]
        dtx = 2 * self.kc[2] * xty + self.kc[3] * (r2 + 2 * xy[:, 0]**2)
        dty = self.kc[2] * (r2 + 2 * xy[:, 1]**2) + 2 * self.kc[3] * xty
        # Apply lens distortion
        dxy = xy.copy()
        dxy[:, 0] += dxy[:, 0] * dr + dtx
        dxy[:, 1] += dxy[:, 1] * dr + dty
        # Project to image
        return np.column_stack((
            self.fc[0] * (dxy[:, 0] + self.alpha_d * dxy[:, 1]) + self.cc[0],
            self.fc[0] * dxy[:, 1] + self.cc[1]))

    def residuals(self, cam, uv=None):
        if uv is None:
            uv = cam.grid(step=10, mode='points')
        xy = cam._image2camera(uv)
        return self._camera2image(xy) - uv

    def as_camera(self):
        """
        Return equivalent `Camera` object.

        A non-zero `alpha_c` is not currently supported.
        """
        # Initialize camera
        cam = Camera(
            imgsz=self.imgsz,
            f=self.fc, c=np.asarray(self.cc) + 0.5 - np.asarray(self.imgsz) / 2,
            k=(self.kc[0], self.kc[1], self.kc[4]), p=(self.kc[2], self.kc[3]))
        # Convert camera
        if self.alpha_c:
            raise ValueError('Fitting with non-zero alpha_c not supported')
        return cam

class PhotoScanCamera(object):
    """
    Frame camera model used by Agisoft Photoscan.

    See http://www.agisoft.com/pdf/photoscan-pro_1_4_en.pdf (Appendix C).

    Attributes:
        imgsz (iterable): Image size in pixels (x, y)
        f (float): Focal length in pixels
        cx (float): Principal point offset in pixels (x)
        cy (float): Principal point offset in pixels (y)
        k1 (float): Radial distortion coefficient #1
        k2 (float): Radial distortion coefficient #2
        k3 (float): Radial distortion coefficient #3
        p1 (float): Tangential distortion coefficient #1
        p2 (float): Tangential distortion coefficient #2
        b1 (float): Affinity coefficient
        b2 (float): Non-orthogonality (skew) coefficient
    """

    def __init__(self, imgsz, f, cx, cy, k1=0, k2=0, k3=0, k4=0, b1=0, b2=0, p1=0, p2=0, p3=0, p4=0):
        self.imgsz = imgsz
        self.f = f
        self.cx, self.cy = cx, cy
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.b1, self.b2 = b1, b2
        self.p1, self.p2, self.p3, self.p4 = p1, p2, p3, p4

    def _camera2image(self, xy):
        # Compute lens distortion
        r2 = np.sum(xy**2, axis=1)
        dr = self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2 + self.k4 * r2 * r2 * r2 * r2
        xty = xy[:, 0] * xy[:, 1]
        p34 = 1 + self.p3 * r2 + self.p4 * r2 * r2
        dtx = (self.p1 * (r2 + 2 * xy[:, 0]**2) + 2 * self.p2 * xty) * p34
        dty = (self.p2 * (r2 + 2 * xy[:, 1]**2) + 2 * self.p1 * xty) * p34
        # Apply lens distortion
        dxy = xy.copy()
        dxy[:, 0] += dxy[:, 0] * dr + dtx
        dxy[:, 1] += dxy[:, 1] * dr + dty
        # Project to image
        return np.column_stack((
            self.imgsz[0] * 0.5 + self.cx + dxy[:, 0] * (self.f + self.b1) + dxy[:, 1] * self.b2,
            self.imgsz[1] * 0.5 + self.cy + dxy[:, 1] * self.f))

    def residuals(self, cam, uv=None):
        if uv is None:
            uv = cam.grid(step=10, mode='points')
        xy = cam._image2camera(uv)
        return self._camera2image(xy) - uv

    def as_camera(self, step=10, return_fit=False):
        """
        Return equivalent `Camera` object.

        If either `k4`, `p3`, `p4` or `b2` is non-zero, the conversion is
        estimated numerically. A non-zero `b2` is not currently supported.

        Arguments:
            step: Sample grid spacing for all (float) or each (iterable) dimension
            return_fit (bool): Whether to also return the `lmfit.MinimizerResult`
        """
        # Initialize camera
        cam = Camera(
            imgsz=self.imgsz,
            f=(self.f - self.b1, self.f), c=(self.cx, self.cy),
            k=(self.k1, self.k2, self.k3), p=(self.p2, self.p1))
        # Convert camera
        if any((self.k4, self.p3, self.p4, self.b2)):
            # Initialize image coordinates
            uv = cam.grid(step=step, mode='points')
            # Fit Camera
            params = dict()
            if self.k4:
                params['k'] = [3, 4, 5]
            if self.p3 or self.p4:
                params['p'] = True
            if self.b2:
                raise ValueError('Fitting with non-zero b2 not supported')
            lmfit_params, apply_params = optimize.build_lmfit_params([cam], [params])
            def residuals(params):
                apply_params(params)
                return self.residuals(cam=cam, uv=uv)
            def callback(params, iter, resid, *args, **kwargs):
                err = np.linalg.norm(resid.reshape(-1, 2), ord=2, axis=1).mean()
                sys.stdout.write('\r' + str(err))
                sys.stdout.flush()
            fit = lmfit.minimize(params=lmfit_params, fcn=residuals, iter_cb=callback)
            sys.stdout.write('\n')
            apply_params(fit.params)
        else:
            fit = None
        if return_fit:
            return cam, fit
        else:
            return cam

class PhotoModelerCamera(object):
    """
    Camera model used by EOS Systems PhotoModeler.

    See "Lens Distortion Formulation" in the software help.

    Attributes:
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

    def __init__(self, focal, xp, yp, fw, fh, k1=0, k2=0, k3=0, p1=0, p2=0):
        self.focal = focal
        self.xp, self.yp = xp, yp
        self.fw, self.fh = fw, fh
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.p1, self.p2 = p1, p2

    def _image2camera(self, uv, imgsz):
        # Convert image coordinates to mm relative to principal point
        xy = uv * (self.fw, self.fh) * (1 / imgsz) - (self.xp, self.yp)
        # Flip y (+y is down in image, but up in PM "photo space")
        xy[:, 1] *= -1
        # Remove lens distortion
        r2 = np.sum(xy**2, axis=1)
        dr = self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        xty = xy[:, 0] * xy[:, 1]
        # NOTE: p1 and p2 are reversed
        dtx = self.p1 * (r2 + 2 * xy[:, 0]**2) + 2 * self.p2 * xty
        dty = self.p2 * (r2 + 2 * xy[:, 1]**2) + 2 * self.p1 * xty
        xy[:, 0] += xy[:, 0] * dr + dtx
        xy[:, 1] += xy[:, 1] * dr + dty
        # Flip y back
        xy[:, 1] *= -1
        # Normalize
        xy *= (1 / self.focal)
        return xy

    def residuals(self, cam, uv=None):
        if uv is None:
            uv = cam.grid(step=10, mode='points')
        xy = self._image2camera(uv, cam.imgsz)
        return cam._camera2image(xy) - uv

    def as_camera(self, imgsz, step=10, return_fit=False):
        """
        Return equivalent `Camera` object.

        If either `k1`, `k2`, `k3`, `p1`, or `p2` is non-zero, the conversion is
        estimated numerically.

        Arguments:
            imgsz (iterable): Image size in pixels (x, y)
            step: Sample grid spacing for all (float) or each (iterable) dimension
            return_fit (bool): Whether to also return the `lmfit.MinimizerResult`
        """
        imgsz = np.asarray(imgsz)
        sensorsz = np.array((self.fw, self.fh))
        # Initialize camera
        cam = Camera(
            imgsz=imgsz, sensorsz=sensorsz,
            fmm=self.focal, cmm=(self.xp, self.yp) - sensorsz / 2)
        # Convert camera
        k = (self.k1, self.k2, self.k3)
        p = (self.p1, self.p2)
        if any(k + p):
            # Initialize image coordinates
            uv = cam.grid(step=step, mode='points')
            # Fit Camera
            params = dict()
            if any(k):
                params['k'] = [i for i, k in enumerate((self.k1, self.k2, self.k3)) if k]
            if any(p):
                params['p'] = True
            lmfit_params, apply_params = optimize.build_lmfit_params([cam], [params])
            def residuals(params):
                apply_params(params)
                return self.residuals(cam=cam, uv=uv)
            def callback(params, iter, resid, *args, **kwargs):
                err = np.linalg.norm(resid.reshape(-1, 2), ord=2, axis=1).mean()
                sys.stdout.write('\r' + str(err))
                sys.stdout.flush()
            fit = lmfit.minimize(params=lmfit_params, fcn=residuals, iter_cb=callback)
            sys.stdout.write('\n')
            apply_params(fit.params)
        else:
            fit = None
        if return_fit:
            return cam, fit
        else:
            return cam

def pm_points_to_ps_markers(points_path, camera_labels, imgsz, markers_path=None):
    pm = pandas.read_csv(points_path, skiprows=3).loc[:, ('Object Point ID', 'Photo #', 'X (pixels)', 'Y (pixels)')]
    pm.columns = ('point_id', 'photo_id', 'x', 'y')
    pm['camera_id'] = pm.photo_id - 1
    point_ids = list(pm.point_id.unique())
    pm['marker_id'] = [point_ids.index(id) for id in pm.point_id]
    from lxml.builder import E as e
    cameras = [e.camera(id=str(id), sensor_id='0', label=label, enabled='1')
        for id, label in zip(pm.camera_id.unique(), camera_labels)]
    markers = [e.marker(id=str(row.marker_id), label=str(row.point_id))
        for i, row in pm.loc[:, ('marker_id', 'point_id')].drop_duplicates().iterrows()]
    frame_markers = list()
    for id in pm.marker_id.unique():
        locations = [e.location(camera_id=str(int(row.camera_id)), pinned='1', x=str(row.x), y=str(row.y))
            for i, row in pm[pm.marker_id == id].iterrows()]
        frame_markers.append(e.marker(marker_id=str(id), *locations))
    xml = e.document(
        e.chunk(
            e.sensors(
                e.sensor(
                    e.resolution(width=str(imgsz[0]), height=str(imgsz[1])),
                    id='0'
                )
            ),
            e.cameras(*cameras),
            e.markers(*markers),
            e.frames(
                e.frame(
                    e.markers(*frame_markers),
                    id='0'
                )
            )
        )
    )
    txt = lxml.etree.tostring(xml, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode()
    if markers_path:
        with open(markers_path, mode='w') as fp:
            fp.write(txt)
    else:
        return txt
