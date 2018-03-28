from __future__ import (print_function, division, unicode_literals)
from .backports import *
from .imports import (np, scipy, cv2, lmfit, matplotlib, sys, os, copy, pickle,
    warnings, datetime)
from . import (helpers)

# ---- Controls ----

# Controls (within Cameras) support RANSAC with the following API:
# .size
# .observed(index)
# .predicted(index)

class Points(object):
    """
    `Points` store image-world point correspondences.

    World coordinates (`xyz`) are projected into the camera,
    then compared to the corresponding image coordinates (`uv`).

    Attributes:
        cam (Camera): Camera object
        uv (array): Image coordinates (Nx2)
        xyz (array): World coordinates (Nx3)
        directions (bool): Whether `xyz` are absolute coordinates (False) or ray directions (True)
        original_cam_xyz (array): Original camera position (`cam.xyz`)
        correction (dict or bool): See `cam.project()`
        size (int): Number of point pairs
    """

    def __init__(self, cam, uv, xyz, directions=False, correction=False):
        if len(uv) != len(xyz):
            raise ValueError('`uv` and `xyz` have different number of rows')
        self.cam = cam
        self.uv = uv
        self.xyz = xyz
        self.directions = directions
        self.correction = correction
        self.original_cam_xyz = cam.xyz.copy()
        self.size = len(self.uv)

    def observed(self, index=None):
        """
        Return observed image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
        """
        if index is None:
            index = slice(None)
        return self.uv[index]

    def predicted(self, index=None):
        """
        Predict image coordinates from world coordinates.

        If the camera position (`cam.xyz`) has changed and `xyz` are ray directions (`directions=True`),
        the point correspondences are invalid and an error is raised.

        Arguments:
            index (array_like or slice): Indices of world points to project, or all if `None`
        """
        if index is None:
            index = slice(None)
        if self.directions and not self.is_static():
            raise ValueError('Camera has changed position (xyz) and `directions=True`')
        return self.cam.project(self.xyz[index], directions=self.directions, correction=self.correction)

    def is_static(self):
        """
        Test whether the camera is at its original position.
        """
        return (self.cam.xyz == self.original_cam_xyz).all()

    def plot(self, index=None, scale=1, width=5, selected='red', unselected=None):
        """
        Plot reprojection errors as quivers.

        Arrows point from observed to predicted coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
        """
        if index is None:
            index = slice(None)
            other_index = slice(0)
        else:
            other_index = np.delete(np.arange(self.size), index)
        uv = self.observed()
        puv = self.predicted()
        duv = scale * (puv - uv)
        defaults = dict(scale=1, scale_units='xy', angles='xy', units='xy', width=width, color='red')
        if unselected is not None:
            if not isinstance(unselected, dict):
                unselected=dict(color=unselected)
            unselected = helpers.merge_dicts(defaults, unselected)
            matplotlib.pyplot.quiver(
                uv[other_index, 0], uv[other_index, 1], duv[other_index, 0], duv[other_index, 1], **unselected)
        if selected is not None:
            if not isinstance(selected, dict):
                selected=dict(color=selected)
            selected = helpers.merge_dicts(defaults, selected)
            matplotlib.pyplot.quiver(
                uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1], **selected)

class Lines(object):
    """
    `Lines` store image and world lines believed to overlap.

    Image lines (`uvs`) are interpolated to a single array of image points (`uvi`).
    World lines (`xyzs`) are projected into the camera and the nearest point along
    any such lines is matched to each image point.

    Attributes:
        cam (Camera): Camera object
        uvs (array or list): Arrays of image line vertices (Nx2)
        uvi (array): Image coordinates interpolated from `uvs` by `step`
        xyzs (array or list): Arrays of world line vertices (Nx3)
        directions (bool): Whether `xyzs` are absolute coordinates (False) or ray directions (True)
        correction (dict or bool): See `cam.project()`
        step (float): Along-line distance between image points interpolated from lines `uvs`
        original_cam_xyz (array): Original camera position (`cam.xyz`)
        size (int): Number of image points
    """

    def __init__(self, cam, uvs, xyzs, directions=False, correction=False, step=None):
        self.cam = cam
        # Retain image lines for plotting
        self.uvs = uvs if isinstance(uvs, list) else [uvs]
        self.step = step
        if step:
            self.uvi = np.vstack((interpolate_line(uv, step=step, normalized=False) for uv in self.uvs))
        else:
            self.uvi = np.vstack(self.uvs)
        self.xyzs = xyzs if isinstance(xyzs, list) else [xyzs]
        self.directions = directions
        self.correction = correction
        self.original_cam_xyz = cam.xyz.copy()
        self.size = len(self.uvi)

    def observed(self, index=None):
        """
        Return observed image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
        """
        if index is None:
            index = slice(None)
        return self.uvi[index]

    def project(self):
        """
        Project world lines onto the image.

        If the camera position (`cam.xyz`) has changed and `xyz` are ray directions (`directions=True`),
        the point correspondences are invalid and an error is raised.

        Returns:
            list: Arrays of image coordinates (Nx2)
        """
        if self.directions and not self.is_static():
            raise ValueError('Camera has changed position (xyz) and `directions=True`')
        xy_step = 1 / self.cam.f.mean()
        uv_edges = self.cam.edges(step = self.cam.imgsz / 2)
        xy_edges = self.cam._image2camera(uv_edges)
        xy_box = np.hstack((np.min(xy_edges, axis=0), np.max(xy_edges, axis=0)))
        puvs = []
        for xyz in self.xyzs:
            # TODO: Instead, clip lines to 3D polar viewbox before projecting
            # Project world lines to camera
            xy = self.cam._world2camera(xyz, directions=self.directions, correction=self.correction)
            # Discard nan values (behind camera)
            lines = helpers.boolean_split(xy, np.isnan(xy[:, 0]), include='false')
            for line in lines:
                # Clip lines in view
                # Resolves coordinate wrap around with large distortion
                for cline in helpers.clip_polyline_box(line, xy_box):
                    # Interpolate clipped lines to ~1 pixel density
                    puvs.append(self.cam._camera2image(
                        interpolate_line(np.array(cline), step=xy_step, normalized=False)))
        if puvs:
            return puvs
        else:
            # FIXME: Fails if lines slip out of camera view
            # TODO: Return center of image instead of error?
            raise ValueError('All line vertices are outside camera view')

    def predicted(self, index=None):
        """
        Return the points on the projected world lines nearest the image coordinates.

        Arguments:
            index (array_like or slice): Indices of image points to include in nearest-neighbor search,
                or all if `None`

        Returns:
            array: Image coordinates (Nx2)
        """
        puv = np.row_stack(self.project())
        min_index = find_nearest_neighbors(self.observed(), puv)
        return puv[min_index, :]

    def is_static(self):
        """
        Test whether the camera is at its original position.
        """
        return (self.cam.xyz == self.original_cam_xyz).all()

    def plot(self, index=None, scale=1, width=5, selected='red', unselected=None,
        observed='green', predicted='yellow'):
        """
        Plot the reprojection errors as quivers.

        Arrows point from observed to predicted image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
            observed: For image lines, further arguments to matplotlib.pyplot.plot (dict), `None` to hide, or color
            predicted: For world lines, further arguments to matplotlib.pyplot.plot (dict), `None` to hide, or color
        """
        # Plot image lines
        if observed is not None:
            if not isinstance(observed, dict):
                observed=dict(color=observed)
            observed = helpers.merge_dicts(dict(color='green'), observed)
            for uv in self.uvs:
                matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], **observed)
        # Plot world lines
        if predicted is not None:
            if not isinstance(predicted, dict):
                predicted=dict(color=predicted)
            predicted = helpers.merge_dicts(dict(color='yellow'), predicted)
            puvs = self.project()
            for puv in puvs:
                matplotlib.pyplot.plot(puv[:, 0], puv[:, 1], **predicted)
        # Plot errors
        if selected is not None or unselected is not None:
            if index is None:
                index = slice(None)
                other_index = slice(0)
            else:
                other_index = np.delete(np.arange(self.size), index)
            uv = self.observed()
            if not predicted:
                puvs = self.project()
            puv = np.row_stack(puvs)
            min_index = find_nearest_neighbors(uv, puv)
            duv = scale * (puv[min_index, :] - uv)
            defaults = dict(scale=1, scale_units='xy', angles='xy', units='xy', width=width, color='red')
            if unselected is not None:
                if not isinstance(unselected, dict):
                    unselected=dict(color=unselected)
                unselected = helpers.merge_dicts(defaults, unselected)
                matplotlib.pyplot.quiver(
                    uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1], **unselected)
            if selected is not None:
                if not isinstance(selected, dict):
                    selected=dict(color=selected)
                selected = helpers.merge_dicts(defaults, selected)
                matplotlib.pyplot.quiver(
                    uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1], **selected)

class Matches(object):
    """
    `Matches` store image-image point correspondences.

    The image coordinates (`uvs[i]`) of one camera (`cams[i]`) are projected into the other camera (`cams[j]`),
    then compared to the expected image coordinates for that camera (`uvs[j]`).

    Attributes:
        cams (list): Pair of Camera objects
        uvs (list): Pair of image coordinate arrays (Nx2)
        size (int): Number of point pairs
    """

    def __init__(self, cams, uvs):
        if len(cams) != 2 or len(uvs) != 2:
            raise ValueError('`cams` and `uvs` must each have two elements')
        if cams[0] is cams[1]:
            raise ValueError('Both cameras are the same object')
        if uvs[0].shape != uvs[1].shape:
            raise ValueError('Image coordinate arrays have different shapes')
        self.cams = cams
        self.uvs = uvs
        self.size = len(self.uvs[0])

    def observed(self, index=None, cam=0):
        """
        Return observed image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
            cam (Camera or int): Camera of points to return
        """
        if index is None:
            index = slice(None)
        cam_idx = self.cam_index(cam)
        return self.uvs[cam_idx][index]

    def predicted(self, index=None, cam=0):
        """
        Predict image coordinates for a camera from the coordinates of the other camera.

        If the cameras are not at the same position, the point correspondences cannot be
        projected explicitly and an error is raised.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError('Cameras have different positions (xyz)')
        if index is None:
            index = slice(None)
        cam_in = self.cam_index(cam)
        cam_out = 0 if cam_in else 1
        dxyz = self.cams[cam_out].invproject(self.uvs[cam_out][index])
        return self.cams[cam_in].project(dxyz, directions=True)

    def is_static(self):
        """
        Test whether the cameras are at the same position.
        """
        return (self.cams[0].xyz == self.cams[1].xyz).all()

    def cam_index(self, cam):
        """
        Retrieve the index of a camera.

        Arguments:
            cam (Camera): Camera object
        """
        if isinstance(cam, int):
            if cam >= len(self.cams):
                raise IndexError('Camera index out of range')
            return cam
        else:
            return self.cams.index(cam)

    def plot(self, index=None, cam=0, scale=1, width=5, selected='red', unselected=None):
        """
        Plot the reprojection errors as quivers.

        Arrows point from the observed to the predicted coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            cam (Camera or int): Camera to plot
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
        """
        if index is None:
            index = slice(None)
            other_index = slice(0)
        else:
            other_index = np.delete(np.arange(self.size), index)
        uv = self.observed(cam=cam)
        puv = self.predicted(cam=cam)
        duv = scale * (puv - uv)
        defaults = dict(scale=1, scale_units='xy', angles='xy', units='xy', width=width, color='red')
        if unselected is not None:
            if not isinstance(unselected, dict):
                unselected=dict(color=unselected)
            unselected = helpers.merge_dicts(defaults, unselected)
            matplotlib.pyplot.quiver(
                uv[other_index, 0], uv[other_index, 1], duv[other_index, 0], duv[other_index, 1], **unselected)
        if selected is not None:
            if not isinstance(selected, dict):
                selected=dict(color=selected)
            selected = helpers.merge_dicts(defaults, selected)
            matplotlib.pyplot.quiver(
                uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1], **selected)

class RotationMatches(Matches):
    """
    `RotationMatches` store image-image point correspondences for cameras seperated
    only by a pure rotation.

    Normalized camera coordinates are pre-computed for speed. Therefore,
    the cameras must always have equal `xyz` (as for `Matches`)
    and no internal parameters can change after initialization.

    Attributes:
        cams (list): Pair of Camera objects
        uvs (list): Pair of image coordinate arrays (Nx2)
        xys (list): Pair of normalized coordinate arrays (Nx2)
        original_internals (array): Original camera internal parameters
            (imgsz, f, c, k, p)
        size (int): Number of point pairs
    """

    def __init__(self, cams, uvs):
        if len(cams) != 2 or len(uvs) != 2:
            raise ValueError('`cams` and `uvs` must each have two elements')
        if cams[0] is cams[1]:
            raise ValueError('Both cameras are the same object')
        if uvs[0].shape != uvs[1].shape:
            raise ValueError('Image coordinate arrays have different shapes')
        if (cams[0].vector[6:] != cams[1].vector[6:]).any():
            raise ValueError('Camera internal parameters (imgsz, f, c, k, p) are not equal')
        self.cams = cams
        self.uvs = uvs
        self.xys = (
            self.cams[0]._image2camera(self.uvs[0]),
            self.cams[1]._image2camera(self.uvs[1]))
        # [imgsz, f, c, k, p]
        self.original_internals = self.cams[0].vector.copy()[6:]
        self.size = len(self.uvs[0])

    def predicted(self, index=None, cam=0):
        """
        Predict image coordinates for a camera from the coordinates of the other camera.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError('Cameras have different positions (xyz)')
        if not self.is_original_internals():
            raise ValueError('Camera internal parameters (imgsz, f, c, k, p) have changed')
        if index is None:
            index = slice(None)
        cam_in = self.cam_index(cam)
        cam_out = 0 if cam_in else 1
        dxyz = self.cams[cam_out]._camera2world(self.xys[cam_out][index])
        return self.cams[cam_in].project(dxyz, directions=True)

    def is_original_internals(self):
        """
        Test whether camera internal parameters are unchanged.
        """
        return (
            (self.cams[0].vector[6:] == self.original_internals) &
            (self.cams[1].vector[6:] == self.original_internals)).all()

class RotationMatchesXY(RotationMatches):
    """
    `RotationMatchesXY` store image-image point correspondences for cameras seperated
    only by a pure rotation.

    Normalized camera coordinates are pre-computed for speed,
    and image coordinates are discarded to save memory.
    Unlike `RotationMatches`, `self.predicted()` and `self.observed()` return
    normalized camera coordinates.

    Arguments:
        uvs (list): Pair of image coordinate arrays (Nx2)

    Attributes:
        cams (list): Pair of Camera objects
        xys (list): Pair of normalized coordinate arrays (Nx2)
        original_internals (array): Original camera internal parameters
            (imgsz, f, c, k, p)
        normalized (bool): Whether to normalize ray directions to unit length
        size (int): Number of point pairs
    """

    def __init__(self, cams, uvs, normalized=False):
        if len(cams) != 2 or len(uvs) != 2:
            raise ValueError('`cams` and `uvs` must each have two elements')
        if cams[0] is cams[1]:
            raise ValueError('Both cameras are the same object')
        if uvs[0].shape != uvs[1].shape:
            raise ValueError('Image coordinate arrays have different shapes')
        if (cams[0].vector[6:] != cams[1].vector[6:]).any():
            raise ValueError('Camera internal parameters (imgsz, f, c, k, p) are not equal')
        self.cams = cams
        self.xys = (
            self.cams[0]._image2camera(uvs[0]),
            self.cams[1]._image2camera(uvs[1]))
        # [imgsz, f, c, k, p]
        self.original_internals = self.cams[0].vector.copy()[6:]
        self.normalized = normalized
        self.size = len(self.xys[0])

    def observed(self, index=None, cam=0):
        """
        Return observed camera coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
            cam (Camera or int): Camera of points to return
        """
        if index is None:
            index = slice(None)
        cam_idx = self.cam_index(cam)
        return self.xys[cam_idx][index]

    def predicted(self, index=None, cam=0):
        """
        Predict camera coordinates for a camera from the coordinates of the other camera.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError('Cameras have different positions (xyz)')
        if not self.is_original_internals():
            raise ValueError('Camera internal parameters (imgsz, f, c, k, p) have changed')
        if index is None:
            index = slice(None)
        cam_in = self.cam_index(cam)
        cam_out = 0 if cam_in else 1
        dxyz = self.cams[cam_out]._camera2world(self.xys[cam_out][index])
        return self.cams[cam_in]._world2camera(dxyz, directions=True)

    def plot(self, *args, **kwargs):
        raise AttributeError('plot() not supported by RotationMatchesXY')

class RotationMatchesXYZ(RotationMatches):
    """
    `RotationMatches3D` store image-image point correspondences for cameras seperated
    only by a pure rotation.

    Normalized camera coordinates are pre-computed for speed,
    and image coordinates are discarded to save memory.
    Unlike `RotationMatches`, `self.predicted()` returns
    world ray directions and `self.observed()` is disabled.

    Arguments:
        uvs (list): Pair of image coordinate arrays (Nx2)

    Attributes:
        cams (list): Pair of Camera objects
        xys (list): Pair of normalized coordinate arrays (Nx2)
        original_internals (array): Original camera internal parameters
            (imgsz, f, c, k, p)
        size (int): Number of point pairs
    """

    def __init__(self, cams, uvs):
        if len(cams) != 2 or len(uvs) != 2:
            raise ValueError('`cams` and `uvs` must each have two elements')
        if cams[0] is cams[1]:
            raise ValueError('Both cameras are the same object')
        if uvs[0].shape != uvs[1].shape:
            raise ValueError('Image coordinate arrays have different shapes')
        if (cams[0].vector[6:] != cams[1].vector[6:]).any():
            raise ValueError('Camera internal parameters (imgsz, f, c, k, p) are not equal')
        self.cams = cams
        self.xys = (
            self.cams[0]._image2camera(uvs[0]),
            self.cams[1]._image2camera(uvs[1]))
        # [imgsz, f, c, k, p]
        self.original_internals = self.cams[0].vector.copy()[6:]
        self.size = len(self.xys[0])

    def observed(self, *args, **kwargs):
        raise AttributeError('observed() not supported by RotationMatchesXYZ')

    def predicted(self, index=None, cam=0):
        """
        Predict world coordinates for a camera.

        Returns world coordinates as ray directions normalized with unit length.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError('Cameras have different positions (xyz)')
        if not self.is_original_internals():
            raise ValueError('Camera internal parameters (imgsz, f, c, k, p) have changed')
        if index is None:
            index = slice(None)
        cam_idx = self.cam_index(cam)
        dxyz = self.cams[cam_idx]._camera2world(self.xys[cam_idx][index])
        # Normalize world coordinates to unit sphere
        dxyz *= 1 / np.linalg.norm(dxyz, ord=2, axis=1).reshape(-1, 1)
        return dxyz

    def plot(self, *args, **kwargs):
        raise AttributeError('plot() not supported by RotationMatchesXY')

# ---- Models ----

# Models support RANSAC with the following API:
# .data_size()
# .fit(index)
# .errors(params, index)

class Polynomial(object):
    """
    Least-squares 1-dimensional polynomial model.

    Fits a polynomial of degree `deg` to 2-dimensional points (rows of `data`) and
    returns the coefficients that minimize the squared error (`params`).
    Can be used with RANSAC algorithm (see optimize.ransac).

    Attributes:
        data (array): Point coordinates (x,y) (Nx2)
        deg (int): Degree of the polynomial
    """

    def __init__(self, data, deg=1):
        self.deg = deg
        self.data = data

    def data_size(self):
        """
        Count the number of points.
        """
        return len(self.data)

    def predict(self, params, index=slice(None)):
        """
        Predict the values of a polynomial.

        Arguments:
            params (array): Values of the polynomial, from highest to lowest degree component
            index (array_like or slice): Indices of points for which to predict y from x
        """
        return np.polyval(params, self.data[index, 0])

    def errors(self, params, index=slice(None)):
        """
        Compute the errors of a polynomial prediction.

        Arguments:
            params (array): Values of the polynomial, from highest to lowest degree component
            index (array_like or slice): Indices of points for which to predict y from x
        """
        prediction = self.predict(params, index)
        return np.abs(prediction - self.data[index, 1])

    def fit(self, index=slice(None)):
        """
        Fit a polynomial to the points (using numpy.polyfit).

        Arguments:
            index (array_like or slice): Indices of points to use for fitting

        Returns:
            array: Values of the polynomial, from highest to lowest degree component
        """
        return np.polyfit(self.data[index, 0], self.data[index, 1], deg=self.deg)

    def plot(self, params=None, index=slice(None), selected='red', unselected='grey', polynomial='red'):
        """
        Plot the points and the polynomial fit.

        Arguments:
            params (array): Values of the polynomial, from highest to lowest degree component,
                or computed if `None`
            index (array_like or slice): Indices of points to select
            selected (color): Matplotlib color for selected points, or `None` to hide
            unselected (color): Matplotlib color for unselected points, or `None` to hide
            polynomial (color): Matplotlib color for polynomial fit, or `None` to hide
        """
        if params is None:
            params = self.fit(index)
        other_index = np.delete(np.arange(self.data_size()), index)
        if selected:
            matplotlib.pyplot.scatter(self.data[index, 0], self.data[index, 1], c=selected)
        if unselected:
            matplotlib.pyplot.scatter(self.data[other_index, 0], self.data[other_index, 1], c=unselected)
        if polynomial:
            matplotlib.pyplot.plot(self.data[:, 0], self.predict(params), c=polynomial)

class Cameras(object):
    """
    Multi-camera optimization.

    Finds the camera parameter values that minimize the reprojection errors of camera control:

        - image-world point coordinates (Points)
        - image-world line coordinates (Lines)
        - image-image point coordinates (Matches, RotationMatches)

    If used with RANSAC (see `optimize.ransac`) with multiple control objects,
    results may be unstable since samples are drawn randomly from all observations,
    and computation will be slow since errors are calculated for all points then subset.

    Attributes:
        cams (list): Camera objects
        controls (list): Camera control (Points, Lines, and Matches objects)
        cam_params (list): Parameters to optimize for each camera seperately (see `parse_params()`)
        group_params (dict): Parameters to optimize for all cameras (see `parse_params()`)
        weights (array): Weights for each control point
        vectors (list): Original camera vectors (for resetting camera parameters)
        params (`lmfit.Parameters`): Parameter initial values and bounds
        scales (array): Scale factors for each parameter.
            Pre-computed if `True`, skipped if `False`.
        sparsity (sparse matrix): Sparsity structure for the estimation of the Jacobian matrix.
            Pre-computed if `True`, skipped if `False`.
        cam_masks (list): Boolean masks of parameters being optimized for each camera
        cam_bounds (list): Bounds of parameters optimized for each camera
        group_mask (array): Boolean mask of parameters optimized for all cameras
        group_bounds (array): Bounds of parameters optimized for all cameras
    """

    def __init__(self, cams, controls, cam_params=dict(viewdir=True), group_params=dict(), weights=None,
        scales=True, sparsity=True):
        self.cams = cams if isinstance(cams, (list, tuple)) else [cams]
        self.vectors = [cam.vector.copy() for cam in self.cams]
        controls = controls if isinstance(controls, (list, tuple)) else [controls]
        self.controls = prune_controls(self.cams, controls)
        self.cam_params = cam_params if isinstance(cam_params, (list, tuple)) else [cam_params]
        self.group_params = group_params
        test_cameras(self)
        temp = [parse_params(params) for params in self.cam_params]
        self.cam_masks = [x[0] for x in temp]
        self.cam_bounds = [x[1] for x in temp]
        self.group_mask, self.group_bounds = parse_params(self.group_params)
        # TODO: Avoid computing masks and bounds twice
        self.params, self.apply_params = build_lmfit_params(self.cams, self.cam_params, group_params)
        self.weights = weights
        # Compute optimal variable scale factors
        if scales is True:
            scales = [camera_scale_factors(cam, self.controls) for cam in self.cams]
            # TODO: Weigh each camera by number of control points
            group_scale = np.vstack((scale[self.group_mask] for scale in scales)).mean(axis=0)
            cam_scales = np.hstack((scale[mask] for scale, mask in zip(scales, self.cam_masks)))
            self.scales = np.hstack((group_scale, cam_scales))
        elif scales is not False:
            self.scales = scales
        else:
            self.scales = None
        if sparsity is True:
            self.sparsity = self.build_sparsity()
        elif scales is not False:
            self.sparsity = sparsity
        else:
            self.sparsity = None

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is None:
            self._weights = value
        else:
            value = np.atleast_2d(value).reshape(-1, 1)
            self._weights = value * len(value) / sum(value)

    def set_cameras(self, params):
        """
        Set camera parameter values.

        Saved camera vectors (`.vectors`) are unchanged.
        The operation can be reversed with `.reset_cameras()`.

        Arguments:
            params (array or `lmfit.Parameters`): Parameter values ordered first
                by group or camera [group | cam0 | cam1 | ...],
                then ordered by position in `Camera.vector`.
        """
        self.apply_params(params)

    def reset_cameras(self, vectors=None, save=False):
        """
        Reset camera parameters to their saved values.

        Arguments:
            vectors (list): Camera vectors.
                If `None` (default), the saved vectors are used (`.vectors`).
            save (bool): Whether to save `vectors` as new defaults.
        """
        if vectors is None:
            vectors = self.vectors
        else:
            if save:
                self.vectors = vectors
        for cam, vector in zip(self.cams, vectors):
            cam.vector = vector.copy()

    def update_params(self):
        """
        Update parameter bounds and initial values from current state of cameras.
        """
        self.params, self.apply_params = build_lmfit_params(self.cams, self.cam_params, self.group_params)

    def data_size(self):
        """
        Return the total number of data points.
        """
        return np.array([control.size for control in self.controls]).sum()

    def build_sparsity(self):
        # Number of parameters
        n_group = np.count_nonzero(self.group_mask)
        n_cams = [np.count_nonzero(mask) for mask in self.cam_masks]
        n = n_group + sum(n_cams)
        # Number of observations
        m_control = [2 * control.size for control in self.controls]
        m = sum(m_control)
        # Initialize sparse matrix with zeros
        S = scipy.sparse.lil_matrix((m, n), dtype=int)
        # Group parameters
        S[:, 0:n_group] = 1
        # Camera parameters
        ctrl_ends = np.cumsum([0] + m_control)
        cam_ends = np.cumsum([0] + n_cams) + n_group
        for i, control in enumerate(self.controls):
            ctrl_cams = getattr(control, 'cam', None)
            if ctrl_cams is None:
                ctrl_cams = getattr(control, 'cams')
            if not isinstance(ctrl_cams, (tuple, list)):
                ctrl_cams = (ctrl_cams, )
            for cam in ctrl_cams:
                if cam in self.cams:
                    j = self.cams.index(cam)
                    S[ctrl_ends[i]:ctrl_ends[i + 1], cam_ends[j]:cam_ends[j + 1]] = 1
        return S

    def observed(self, index=None):
        """
        Return the observed image coordinates for all camera control.

        See control `observed()` method for more details.

        Arguments:
            index (array or slice): Indices of points to return, or all if `None`
        """
        if len(self.controls) == 1:
            return self.controls[0].observed(index=index)
        else:
            # TODO: Map index to subindices for each control
            if index is None:
                index = slice(None)
            return np.vstack((control.observed() for control in self.controls))[index]

    def predicted(self, params=None, index=None):
        """
        Return the predicted image coordinates for all camera control.

        See control `predicted()` method for more details.

        Arguments:
            params (array or `lmfit.Parameters`): Parameter values (see `.set_cameras()`)
            index (array or slice): Indices of points to return, or all if `None` (default)
        """
        if params is not None:
            vectors = [cam.vector.copy() for cam in self.cams]
            self.set_cameras(params)
        if len(self.controls) == 1:
            result = self.controls[0].predicted(index=index)
        else:
            # TODO: Map index to subindices for each control
            if index is None:
                index = slice(None)
            result = np.vstack((control.predicted() for control in self.controls))[index]
        if params is not None:
            self.reset_cameras(vectors)
        return result

    def residuals(self, params=None, index=None):
        """
        Return the reprojection residuals for all camera control.

        Residuals are the difference between `.predicted()` and `.observed()`.

        Arguments:
            params (array or `lmfit.Parameters`): Parameter values (see `.set_cameras()`)
            index (array_like or slice): Indices of points to include, or all if `None`
        """
        d = self.predicted(params=params, index=index) - self.observed(index=index)
        if self.weights is None:
            return d
        else:
            if index is None:
                index = slice(None)
            return d * self.weights[index]

    def errors(self, params=None, index=None):
        """
        Return the reprojection errors for all camera control.

        Errors are the Euclidean distance between `.predicted()` and `.observed()`.

        Arguments:
            params (array or `lmfit.Parameters`): Parameter values (see `.set_cameras()`)
            index (array or slice): Indices of points to include, or all if `None`
        """
        return np.linalg.norm(self.residuals(params=params, index=index), axis=1)

    def fit(self, index=None, cam_params=None, group_params=None, full=False,
        method='least_squares', nan_policy='omit', reduce_fcn=None, **kwargs):
        """
        Return optimal camera parameter values.

        Find the camera parameter values that minimize the reprojection residuals
        or a derivative objective function across all control.
        See `lmfit.minimize()` (https://lmfit.github.io/lmfit-py/fitting.html).

        Unless `.update_params()` is called first, `.fit()` will use the
        parameter bounds and initial values computed initially.

        Arguments:
            index (array or slice): Indices of residuals to include, or all if `None`
            cam_params (list): Sequence of independent camera properties to fit (see `Cameras`)
                iteratively before final run. Must be `None` or same length as `group_params`.
            group_params (list): Sequence of group camera properties to fit (see `Cameras`)
                iteratively before final run. Must be `None` or same length as `cam_params`.
            full (bool): Whether to return the full result of `lmfit.Minimize()`
            **kwargs: Additional arguments to `lmfit.minimize()`.
                `self.scales` and `self.jac_sparsity` (if computed) are applied
                to the following arguments if not provided:

                    - `diag=self.scales` for `method='leastsq'`
                    - `x_scale=self.scales` and `jac_sparsity=self.sparsity` for
                    `method='least_squares'`

        Returns:
            array or `lmfit.Parameters` (`full=True`): Parameter values ordered first
                by group or camera (group, cam0, cam1, ...),
                then ordered by position in `Camera.vector`.
        """
        if method == 'leastsq':
            if self.scales is not None and not hasattr(kwargs, 'diag'):
                kwargs['diag'] = self.scales
        if method == 'least_squares':
            if self.scales is not None and not hasattr(kwargs, 'x_scale'):
                kwargs['x_scale'] = self.scales
            if self.sparsity is not None and not hasattr(kwargs, 'jac_sparsity'):
                if index is None:
                    kwargs['jac_sparsity'] = self.sparsity
                else:
                    if isinstance(index, slice):
                        jac_index = np.arange(self.data_size())[index]
                    else:
                        jac_index = index
                    jac_index = np.dstack((2 * jac_index, 2 * jac_index + 1)).ravel()
                    kwargs['jac_sparsity'] = self.sparsity[jac_index]
        def callback(params, iter, resid, *args, **kwargs):
            err = np.linalg.norm(resid.reshape(-1, 2), ord=2, axis=1).mean()
            sys.stdout.write('\r' + str(err))
            sys.stdout.flush()
        iterations = max(
            len(cam_params) if cam_params else 0,
            len(group_params) if group_params else 0)
        if iterations:
            for n in range(iterations):
                iter_cam_params = cam_params[n] if cam_params else self.cam_params
                iter_group_params = group_params[n] if group_params else self.group_params
                model = Cameras(cams=self.cams, controls=self.controls,
                    cam_params=iter_cam_params, group_params=iter_group_params)
                values = model.fit(index=index, method=method, nan_policy=nan_policy, reduce_fcn=reduce_fcn, **kwargs)
                if values is not None:
                    model.set_cameras(params=values)
            self.update_params()
        result = lmfit.minimize(params=self.params, fcn=self.residuals, kws=dict(index=index), iter_cb=callback,
            method=method, nan_policy=nan_policy, reduce_fcn=reduce_fcn, **kwargs)
        sys.stdout.write('\n')
        if iterations:
            self.reset_cameras()
            self.update_params()
        if not result.success:
            print(result.message)
        if full:
            return result
        elif result.success:
            return np.array(list(result.params.valuesdict().values()))

    def plot(self, params=None, cam=0, index=None, scale=1, width=5, selected='red', unselected=None,
        lines_observed='green', lines_predicted='yellow'):
        """
        Plot reprojection errors.

        See control object `plot()` methods for details.

        Arguments:
            params (array): Parameter values [group | cam0 | cam1 | ...].
                If `None` (default), cameras are used unchanged.
            cam (Camera or int): Camera to plot in (as object or position in `self.cams`)
            index (array or slice): Indices of points to plot. If `None` (default), all points are plotted.
                Other values require `self.test_ransac()` to be True.
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to matplotlib.pyplot.quiver (dict), `None` to hide, or color
            lines_observed: For image lines, further arguments to matplotlib.pyplot.plot (dict), `None` to hide, or color
            lines_predicted: For world lines, further arguments to matplotlib.pyplot.plot (dict), `None` to hide, or color
        """
        if params is not None:
            vectors = [camera.vector.copy() for camera in self.cams]
            self.set_cameras(params)
        cam = self.cams[cam] if isinstance(cam, int) else cam
        cam_controls = prune_controls([cam], self.controls)
        if index is not None and len(self.controls) > 1:
            # TODO: Map index to subindices for each control
            raise ValueError('Plotting with `index` not yet supported with multiple controls')
        for control in cam_controls:
            if isinstance(control, Lines):
                control.plot(index=index, scale=scale, width=width, selected=selected, unselected=unselected,
                    observed=lines_observed, predicted=lines_predicted)
            elif isinstance(control, Points):
                control.plot(index=index, scale=scale, width=width, selected=selected, unselected=unselected)
            elif isinstance(control, Matches):
                control.plot(cam=cam, index=index, scale=scale, width=width, selected=selected, unselected=unselected)
        if params is not None:
            self.reset_cameras(vectors)

    def plot_weights(self, index=None, scale=1, cmap=None):
        if index is None:
            index = slice(None)
        weights = np.ones(self.data_size()) if self.weights is None else self.weights
        uv = self.observed(index=index)
        matplotlib.pyplot.scatter(uv[:, 0], uv[:, 1], c=weights[index], s=scale * weights[index], cmap=cmap)
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.gca().invert_yaxis()

class ObserverCameras(object):
    """
    `ObserverCameras` finds the optimal view directions of the cameras in an `Observer`.

    Optimization proceeds in stages:

        - Build (and save to file) keypoint descriptors for each image with `self.build_keypoints()`.
        - Build (and save to file) keypoints matches between image pairs with `self.build_matches()`.
        Matches are made between an image and all others falling within a certain distance in time.
        - Solve for the optimal view directions of all the cameras with `self.fit()`.

    Arguments:
        template: Template for histogram matching (see `helpers.match_histogram()`).
            If `None`, the first anchor image is used, converted to grayscale.

    Attributes:
        observer (`glimpse.Observer`): Observer with the cameras to orient (images to align)
        anchors (iterable): Integer indices of `observer.images` to use as anchors
        template (tuple): Template histogram (values, quantiles) for histogram matching
        matches (array): Grid of `Matches` objects (see `self.build_matches`)
    """

    def __init__(self, observer, anchors=None, template=None):
        self.observer = observer
        if anchors is None:
            is_anchor = [img.anchor for img in self.observer.images]
            anchors = np.where(is_anchor)[0]
            if len(anchors) == 0:
                warnings.warn('No anchor image found, using first image as anchor')
                anchors = (0, )
        self.anchors = anchors
        if template is None:
            template = self.read_image(self.observer.images[self.anchors[0]])
        if isinstance(template, np.ndarray):
            template = helpers.compute_cdf(template, return_inverse=False)
        self.template = template
        # Placeholders
        self.matches = None
        self.viewdirs = np.vstack([img.cam.viewdir.copy()
            for img in self.observer.images])

    def set_cameras(self, viewdirs):
        for i, img in enumerate(self.observer.images):
            img.cam.viewdir = viewdirs[i]

    def reset_cameras(self):
        self.set_cameras(viewdirs=self.viewdirs.copy())

    def read_image(self, img):
        """
        Read image data preprocessed for keypoint detection and matching.

        Images are converted to grayscale,
        then histogram matched to `self.template`.

        Arguments:
            img (`glimpse.Image`): Image object

        Returns:
            array: Preprocessed grayscale image (uint8)
        """
        I = img.read()
        if I.ndim > 2:
            I = helpers.rgb_to_gray(I, method='average', weights=None)
        if hasattr(self, 'template') and self.template is not None:
            I = helpers.match_histogram(I, template=self.template)
        return I.astype(np.uint8)

    def build_keypoints(self, masks=None, overwrite=False,
        clear_images=True, clear_keypoints=False, **params):
        """
        Build image keypoints and their descriptors.

        The result for each `Image` is stored in `Image.keypoints`
        and written to a binary `pickle` file if `Image.keypoints_path` is set.

        Arguments:
            masks (list or array): Boolean array(s) (uint8) indicating regions in which to detect keypoints
            overwrite (bool): Whether to recompute and overwrite existing keypoint files
            clear_images (bool): Whether to clear cached image data (`self.observer.images[i].I`)
            clear_keypoints (bool): Whether to clear cached keypoints (`Image.keypoints`).
                Ignored if `Image.keypoints_path` keypoints were not written to file.
            **params: Additional arguments to `optimize.detect_keypoints()`
        """
        if masks is None or isinstance(masks, np.ndarray):
            masks = (masks, ) * len(self.observer.images)
        for img, mask in zip(self.observer.images, masks):
            print(img.path)
            if overwrite or img.read_keypoints() is None:
                I = self.read_image(img)
                img.keypoints = detect_keypoints(I, mask=mask, **params)
                if img.keypoints_path:
                    img.write_keypoints()
                    if clear_keypoints:
                        img.keypoints = None
                if clear_images:
                    img.I = None

    def build_matches(self, max_dt=datetime.timedelta(days=1), path=None, overwrite=False,
        clear_keypoints=True, **params):
        """
        Build matches between each image and its nearest neighbors.

        Results are stored in `self.matches` as an (n, n) array of augmented `Matches`,
        and the result for each `Image` pair (i, j) optionally written to a binary `pickle`
        file with name `basenames[i]-basenames[j].pkl`.

        Arguments:
            max_dt (`datetime.timedelta`): Maximum time seperation between
                pairs of images to match
            path (str): Directory for match files.
                If `None`, no files are written.
            overwrite (bool): Whether to recompute and overwrite existing match files
            clear_keypoints (bool): Whether to clear cached keypoints (`Image.keypoints`)
            **params: Additional arguments to `optimize.match_keypoints()`
        """
        n = len(self.observer.images)
        matches = np.full((n, n), None).astype(object)
        if path:
            basenames = [os.path.splitext(os.path.basename(img.path))[0]
                for img in self.observer.images]
            if len(basenames) != len(set(basenames)):
                raise ValueError('Image basenames are not unique')
        for i, imgA in enumerate(self.observer.images[:-1]):
            if i > 0:
                print('') # new line
            print('Matching', i, '->', end=' ')
            for j, imgB in enumerate(self.observer.images[(i + 1):], i + 1):
                if (imgB.datetime - imgA.datetime) > max_dt:
                    continue
                print(j, end=' ')
                if path:
                    outfile = os.path.join(path, basenames[i] + '-' + basenames[j] + '.pkl')
                if path and not overwrite and os.path.exists(outfile):
                    match = helpers.read_pickle(outfile)
                    # Point matches to existing Camera objects
                    match.cams = (imgA.cam, imgB.cam)
                    matches[i, j] = match
                else:
                    uvA, uvB = match_keypoints(imgA.read_keypoints(), imgB.read_keypoints(), **params)
                    match = RotationMatchesXYZ(cams=(imgA.cam, imgB.cam), uvs=(uvA, uvB))
                    matches[i, j] = match
                    if path is not None:
                        helpers.write_pickle(match, outfile)
                if clear_keypoints:
                    imgA.keypoints = None
                    imgB.keypoints = None
        self.matches = matches

    def fit(self, anchor_weight=1e6, method='bfgs', **params):
        """
        Return optimal camera view directions.

        The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is used to find
        the camera view directions that minimize the sum of the absolute differences
        (L1-norm). See `scipy.optimize.minimize(method='bfgs')`.

        Arguments:
            anchor_weight (float): Weight on anchor image view directions being correct
            **params: Additional arguments to `scipy.optimize.minimize()`

        Returns:
            `scipy.optimize.OptimizeResult`: The optimization result.
                Attributes include solution array `x`, boolean `success`, and `message`.
        """
        def fun(viewdirs):
            viewdirs = viewdirs.reshape(-1, 3)
            self.set_cameras(viewdirs=viewdirs)
            objective = 0
            gradients = np.zeros(viewdirs.shape)
            for i in self.anchors:
                objective += (anchor_weight / 2.0) * np.sum((viewdirs[i] - self.viewdirs[i])**2)
                gradients[i] += anchor_weight * (viewdirs[i] - self.viewdirs[i])
            n = len(self.observer.images)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    m = self.matches[i, j]
                    if m:
                        # Project matches
                        dxyz = m.predicted(cam=0) - m.predicted(cam=1)
                        objective += np.sum(np.abs(dxyz))
                        # i -> j
                        xy_hat = np.column_stack((m.xys[0], np.ones(m.size)))
                        dD_dw = np.matmul(m.cams[0].Rprime, xy_hat.T)
                        delta = np.sign(dxyz).reshape(-1, 3, 1)
                        gradient = np.sum(np.matmul(dD_dw.T, delta).T, axis=2).squeeze()
                        gradients[i] += gradient
                        # j -> i
                        gradients[j] -= gradient
            # Update console output
            sys.stdout.write('\r' + str(objective))
            sys.stdout.flush()
            return objective, gradients.ravel()
        viewdirs_0 = [img.cam.viewdir for img in self.observer.images]
        result = scipy.optimize.minimize(
            fun=fun, x0=viewdirs_0, jac=True, method=method, **params)
        self.reset_cameras()
        if not result.success:
            sys.stdout.write('\n') # new line
            print(result.message)
        return result

# ---- RANSAC ----

def ransac(model, sample_size, max_error, min_inliers, iterations=100):
    """
    Fit model parameters to data using the Random Sample Consensus (RANSAC) algorithm.

    Inspired by the pseudocode at https://en.wikipedia.org/wiki/Random_sample_consensus

    Arguments:
        model (object): Model and data object with the following methods:

            - `data_size()`: Returns maximum sample size
            - `fit(index)`: Accepts sample indices and returns model parameters
            - `errors(params, index)`: Accepts sample indices and model parameters and returns an error for each sample

        sample_size (int): Size of sample used to fit the model in each iteration
        max_error (float): Error below which a sample element is considered a model inlier
        min_inliers (int): Number of inliers (in addition to `sample_size`) for a model to be considered valid
        iterations (int): Number of iterations

    Returns:
        array (int): Values of model parameters
        array (int): Indices of model inliers
    """
    i = 0
    params = None
    err = np.inf
    inlier_idx = None
    while i < iterations:
        maybe_idx, test_idx = ransac_sample(sample_size, model.data_size())
        # maybe_inliers = data[maybe_idx]
        maybe_params = model.fit(maybe_idx)
        if maybe_params is None:
            continue
        # test_data = data[test_idx]
        test_errs = model.errors(maybe_params, test_idx)
        also_idx = test_idx[test_errs < max_error]
        if len(also_idx) > min_inliers:
            # also_inliers = data[also_idx]
            better_idx = np.concatenate((maybe_idx, also_idx))
            better_params = model.fit(better_idx)
            if better_params is None:
                continue
            better_errs = model.errors(better_params, better_idx)
            this_err = np.mean(better_errs)
            if this_err < err:
                params = better_params
                err = this_err
                inlier_idx = better_idx
        i += 1
    if params is None:
        raise ValueError('Best fit does not meet acceptance criteria')
    return params, inlier_idx

def ransac_sample(sample_size, data_size):
    """
    Generate index arrays for a random sample and its outliers.

    Arguments:
        sample_size (int): Size of sample
        data_size (int): Size of data

    Returns:
        array (int): Sample indices
        array (int): Outlier indices
    """
    if sample_size >= data_size:
        raise ValueError('`sample_size` is larger or equal to `data_size`')
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    return indices[:sample_size], indices[sample_size:]

# ---- Keypoints ----

def detect_keypoints(I, mask=None, method='sift', root=True, **params):
    """
    Return keypoints and descriptors for an image.

    Arguments:
        I (array): 2 or 3-dimensional image array (uint8)
        mask (array): Regions in which to detect keypoints (uint8)
        root (bool): Whether to return square root L1-normalized descriptors.
            See https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf.
        **params: Additional arguments passed to `cv2.xfeatures2d.SIFT()` or `cv2.xfeatures2d.SURF()`.
            See https://docs.opencv.org/master/d2/dca/group__xfeatures2d__nonfree.html.

    Returns:
        list: Keypoints as cv2.KeyPoint objects
        array: Descriptors as array rows
    """
    if method == 'sift':
        try:
            detector = cv2.xfeatures2d.SIFT_create(**params)
        except AttributeError:
            # OpenCV 2
            detector = cv2.SIFT(**params)
    elif method == 'surf':
        try:
            detector = cv2.xfeatures2d.SURF_create(**params)
        except AttributeError:
            # OpenCV 2
            detector = cv2.SURF(**params)
    keypoints, descriptors = detector.detectAndCompute(I, mask=mask)
    # Empty result: ([], None)
    if root and descriptors is not None:
        descriptors *= 1 / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)
    return keypoints, descriptors

def match_keypoints(ka, kb, mask=None, max_ratio=None, max_distance=None,
    indexParams=dict(algorithm=1, trees=5), searchParams=dict(checks=50)):
    """
    Return the coordinates of matched keypoint pairs.

    Arguments:
        ka (tuple): Keypoints of image A (keypoints, descriptors)
        kb (tuple): Keypoints of image B (keypoints, descriptors)
        mask (array): Region in which to retain keypoints (uint8)
        max_ratio (float): Maximum descriptor-distance ratio between the best
            and second best match. See http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20.
        max_distance (float): Maximum coordinate-distance of matched keypoints
        indexParams (dict): Undocumented argument passed to `cv2.FlannBasedMatcher()`
        searchParams (dict): Undocumented argument passed to `cv2.FlannBasedMatcher()`

    Returns:
        array: Coordinates of matches in image A
        array: Coordinates of matches in image B
    """
    flann = cv2.FlannBasedMatcher(indexParams=indexParams, searchParams=searchParams)
    n_nearest = 2 if max_ratio else 1
    matches = flann.knnMatch(ka[1], kb[1], k=n_nearest, mask=mask)
    uvA = np.array([ka[0][m[0].queryIdx].pt for m in matches])
    uvB = np.array([kb[0][m[0].trainIdx].pt for m in matches])
    if max_ratio:
        is_valid = np.array([m.distance / n.distance for m, n in matches]) < max_ratio
        uvA = uvA[is_valid]
        uvB = uvB[is_valid]
    if max_distance:
        is_valid = np.linalg.norm(uvA - uvB, axis=1) < max_distance
        uvA = uvA[is_valid]
        uvB = uvB[is_valid]
    return uvA, uvB

# ---- Helpers ----

def interpolate_line(vertices, num=None, step=None, distances=None, normalized=False):
    """
    Return points at the specified distances along an N-dimensional line.

    Arguments:
        vertices (array): Coordinates of vertices (NxD)
        num (int): Number of evenly-spaced points to return
        step (float): Target distance between evenly-spaced points (ignored if `num` is not None)
        distances (array): Distance of points along line (ignored if either `num` or `step` are not None)
        normalized (bool): Whether `step` or `distances` represent a fraction of the line's todal length
    """
    # Compute cumulative length at each vertex
    d = np.insert(
            np.cumsum(
                np.sqrt(
                    np.sum(np.diff(vertices, axis=0) ** 2, axis=1))),
            0, 0)
    if normalized:
        d /= d[-1]
    # Prepare distances
    if distances is None:
        if num is None:
            num = np.round(d[-1] / step)
        distances = np.linspace(start=0, stop=d[-1], num=num, endpoint=True)
    # Interpolate each dimension and combine
    return np.column_stack(
        (np.interp(distances, d, vertices[:, i]) for i in range(vertices.shape[1])))

def find_nearest_neighbors(A, B):
    """
    Find the nearest neighbors between two sets of points.

    Arguments:
        A (array): First set of points (NxD)
        B (array): Second set of points (MxD)

    Returns:
        array: Indices of the nearest neighbors of `A` in `B`
    """
    D = scipy.spatial.distance.cdist(A, B, metric='sqeuclidean')
    return np.argmin(D, axis=1)

def prune_controls(cams, controls):
    """
    Return only controls which reference the specified cameras.

    Arguments:
        cams (list): Camera objects
        controls (list): Camera control (Points, Lines, and Matches)

    Returns:
        list: Camera control which reference the cameras in `cams`
    """
    return [control for control in controls
        if len(set(cams) & set((isinstance(control, Matches) and control.cams) or [control.cam])) > 0]

def test_cameras(model):
    """
    Test Cameras model for errors.

    Arguments:
        model (`Cameras`): Cameras object
    """
    # Error: No controls reference the cameras
    if not model.controls:
        raise ValueError('No controls reference the cameras')
    # Error: 'f' or 'c' in `group_params` but image sizes not equal
    if 'f' in model.group_params or 'c' in model.group_params:
        sizes = np.unique([cam.imgsz for cam in model.cams], axis=0)
        if len(sizes) > 1:
            raise ValueError("'f' or 'c' in `group_params` but image sizes not equal: " +
                str(sizes.tolist()))
    # Precompute for remaining tests
    control_cams = [(isinstance(control, Matches) and control.cams) or [control.cam] for control in model.controls]
    is_directions_control = [isinstance(control, (Points, Lines)) and control.directions for control in model.controls]
    is_xyz_cam = ['xyz' in params for params in model.cam_params]
    is_directions_cam = [directions and cam in ctrl_cams
        for cam in model.cams
            for ctrl_cams, directions in zip(control_cams, is_directions_control)]
    # Error: Not all cameras appear in controls
    control_cams_flat = [cam for cams in control_cams for cam in cams]
    if len(set(model.cams) & set(control_cams_flat)) < len(model.cams):
        raise ValueErrors('Not all cameras appear in controls')
    # Error: 'xyz' cannot be in `group_params` if any `control.directions` is True
    if 'xyz' in model.group_params and any(is_directions_control):
        raise ValueError("'xyz' cannot be in `group_params` if any `control.directions` is True")
    # Error: 'xyz' cannot be in `cam_params` if `control.directions` is True for control involving that camera
    is_xyz_directions_cam = [is_xyz and is_directions for is_xyz, is_directions in zip(is_xyz_cam, is_directions_cam)]
    if any(is_xyz_directions_cam):
        raise ValueError("'xyz' cannot be in `cam_params` if `control.directions` is True for control involving that camera")
    return True

def camera_scale_factors(cam, controls=None):
    """
    Return camera variable scale factors.

    These represent the estimated change in each variable in a camera vector needed
    to displace the image coordinates of a feature by one pixel.

    Arguments:
        cam (Camera): Camera object
        controls (list): Camera control (Points, Lines), used to estimate impact of
            camera position (`cam.xyz`).
    """
    # Compute pixels per unit change for each variable
    dpixels = np.ones(20, dtype=float)
    # Compute average distance from image center
    # https://math.stackexchange.com/questions/15580/what-is-average-distance-from-center-of-square-to-some-point
    mean_r_uv = (cam.imgsz.mean() / 6) * (np.sqrt(2) + np.log(1 + np.sqrt(2)))
    mean_r_xy = mean_r_uv / cam.f.mean()
    ## xyz (if f is not descaled)
    # Compute mean distance to world features
    if controls:
        means = []
        weights = []
        for control in controls:
            if isinstance(control, (Points, Lines)) and cam is control.cam and not control.directions:
                weights.append(control.size)
                if isinstance(control, Points):
                    means.append(np.linalg.norm(control.xyz.mean(axis=0) - cam.xyz))
                elif isinstance(control, Lines):
                    means.append(np.linalg.norm(np.vstack(control.xyzs).mean(axis=0) - cam.xyz))
        if means:
            dpixels[0:3] = cam.f.mean() / np.average(means, weights=weights)
    ## viewdir[0, 1]
    # First angle rotates camera left-right
    # Second angle rotates camera up-down
    imgsz_degrees = (2 * np.arctan(cam.imgsz / (2 * cam.f))) * (180 / np.pi)
    dpixels[3:5] = cam.imgsz / imgsz_degrees # pixels per degree
    ## viewdir[2]
    # Third angle rotates camera around image center
    theta = np.pi / 180
    dpixels[5] = 2 * mean_r_uv * np.sin(theta / 2) # pixels per degree
    ## imgsz
    dpixels[6:8] = 0.5
    ## f (if not descaled)
    dpixels[8:10] = mean_r_xy
    ## c
    dpixels[10:12] = 1
    ## k (if f is not descaled)
    # Approximate at mean radius
    # NOTE: Not clear why '2**power' terms are needed
    dpixels[12:18] = [
        mean_r_xy**3 * cam.f.mean() * 2**(1./2),
        mean_r_xy**5 * cam.f.mean() * 2**(3./2),
        mean_r_xy**7 * cam.f.mean() * 2**(5./2),
        mean_r_xy**3 / (1 + cam.k[3] * mean_r_xy**2) * cam.f.mean() * 2**(1./2),
        mean_r_xy**5 / (1 + cam.k[4] * mean_r_xy**4) * cam.f.mean() * 2**(3./2),
        mean_r_xy**7 / (1 + cam.k[5] * mean_r_xy**6) * cam.f.mean() * 2**(5./2)
    ]
    # p (if f is not descaled)
    # Approximate at mean radius at 45 degree angle
    dpixels[18:20] = np.sqrt(5) * mean_r_xy**2 * cam.f.mean()
    # Convert pixels per change to change per pixel (the inverse)
    return 1 / dpixels

def camera_bounds(cam):
    # Distortion bounds based on tested limits of Camera.undistort_oulu()
    k = cam.f.mean() / 4000
    p = cam.f.mean() / 40000
    return np.array([
        # xyz
        [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
        # viewdir
        [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
        # imgsz
        [0, np.inf], [0, np.inf],
        # f
        [0, np.inf], [0, np.inf],
        # c
        [-0.5, 0.5] * cam.imgsz, [-0.5, 0.5] * cam.imgsz,
        # k
        [-k, k], [-k / 2, k / 2], [-k / 2, k / 2], [-k, k], [-k, k], [-k, k],
        # p
        [-p, p], [-p, p]
    ], dtype=float)

def parse_params(params=None, default_bounds=None):
    """
    Return a mask of selected camera parameters and associated bounds.

    Arguments:
        params (dict): Parameters to select by name and indices. For example:

            - {'viewdir': True} : All `viewdir` elements
            - {'viewdir': 0} : First `viewdir` element
            - {'viewdir': [0, 1]} : First and second `viewdir` elements

            Bounds can be specified inside a tuple (indices, min, max).
            Singletons are expanded as needed, and `np.inf` with the
            appropriate sign can be used to indicate no bound:

            - {'viewdir': ([0, 1], -np.inf, 180)}
            - {'viewdir': ([0, 1], -np.inf, [180, 180])}

            `None` or `np.nan` may also be used. These are replaced by the
            values in `default_bounds` (for example, from `camera_bounds()`),
            or (-)`np.inf` if `None`.

    Returns:
        array: Parameter boolean mask (20, )
        array: Parameter min and max bounds (20, 2)
    """
    if params is None:
        params = dict()
    attributes = ('xyz', 'viewdir', 'imgsz', 'f', 'c', 'k', 'p')
    indices = (0, 3, 6, 8, 10, 12, 18, 20)
    mask = np.zeros(20, dtype=bool)
    bounds = np.full((20, 2), np.nan)
    for key, value in params.items():
        if key in attributes:
            if isinstance(value, tuple):
                selection = value[0]
            else:
                selection = value
            if selection or selection == 0:
                i = attributes.index(key)
                if selection is True:
                    positions = range(indices[i], indices[i + 1])
                else:
                    positions = indices[i] + np.atleast_1d(selection)
                mask[positions] = True
            if isinstance(value, tuple):
                min_bounds = np.atleast_1d(value[1])
                if len(min_bounds) == 1:
                    min_bounds = np.repeat(min_bounds, len(positions))
                max_bounds = np.atleast_1d(value[2])
                if len(max_bounds) == 1:
                    max_bounds = np.repeat(max_bounds, len(positions))
                bounds[positions] = np.column_stack((min_bounds, max_bounds))
    if default_bounds:
        missing_min = (bounds[:, 0] == None) | (np.isnan(bounds[:, 0]))
        missing_max = (bounds[:, 1] == None) | (np.isnan(bounds[:, 1]))
        bounds[missing_min, 0] = default_bounds[missing_min, 0]
        bounds[missing_max, 1] = default_bounds[missing_max, 1]
    missing_min = (bounds[:, 0] == None) | (np.isnan(bounds[:, 0]))
    missing_max = (bounds[:, 1] == None) | (np.isnan(bounds[:, 1]))
    bounds[missing_min, 0] = -np.inf
    bounds[missing_max, 1] = np.inf
    return mask, bounds

def build_lmfit_params(cams, cam_params=None, group_params=None):
    """
    Build lmfit.Parameters() object.

    Arguments:
        cams: Camera objects
        cam_params: Camera parameter specifications
        group_params: Group parameter specification
    """
    # Extract parameter masks and bounds
    temp = [parse_params(params=params) for params in cam_params]
    cam_masks = [x[0] for x in temp]
    cam_bounds = [x[1] for x in temp]
    group_mask, group_bounds = parse_params(params=group_params)
    # Labels: (cam<camera_index>_)<attribute><position>
    attributes = ('xyz', 'viewdir', 'imgsz', 'f', 'c', 'k', 'p')
    lengths = [3, 3, 2, 2, 2, 6, 2]
    base_labels = np.array([attribute + str(i) for attribute, length in zip(attributes, lengths) for i in range(length)])
    group_labels = base_labels[group_mask]
    cam_labels = np.hstack(('cam' + str(i) + '_' + label for i, mask in enumerate(cam_masks) for label in base_labels[mask]))
    labels = np.hstack((group_labels, cam_labels))
    # Values
    # NOTE: Group values from first camera
    # values = optimize.sample_cameras(cams, cam_masks, group_mask)
    group_values = cams[0].vector[group_mask]
    cam_values = np.hstack((cam.vector[mask] for cam, mask in zip(cams, cam_masks)))
    values = np.hstack((group_values, cam_values))
    # Bounds
    # NOTE: Default group bounds from first camera
    default_bounds = camera_bounds(cams[0])
    bounds = np.vstack((
        np.where(np.isnan(group_bounds), default_bounds, group_bounds)[group_mask],
        np.vstack(np.where(np.isnan(bounds), default_bounds, bounds)[mask] for bounds, mask in zip(cam_bounds, cam_masks))
    ))
    # Build lmfit.Parameters()
    params = lmfit.Parameters()
    for i, label in enumerate(labels):
        params.add(name=label, value=values[i], vary=True, min=bounds[i, 0], max=bounds[i, 1])
    # Build apply function
    def apply_params(values):
        if isinstance(values, lmfit.parameter.Parameters):
            values = np.array(list(values.valuesdict().values()))
        n_group = group_mask.sum()
        group_values = values[0:n_group]
        n_cams = [mask.sum() for mask in cam_masks]
        cam_breaks = np.cumsum([n_group] + n_cams)
        for i, cam in enumerate(cams):
            cam.vector[cam_masks[i]] = values[cam_breaks[i]:cam_breaks[i + 1]]
            cam.vector[group_mask] = group_values
    return params, apply_params

def model_criteria(fit):
    """
    Return values of common model selection criteria.

    The following criteria are returned:

        - 'aic': Akaike information criterion (https://en.wikipedia.org/wiki/Akaike_information_criterion)
        - 'caic': Consistent Akaike information criterion
        - 'bic': Bayesian information criterion (https://en.wikipedia.org/wiki/Bayesian_information_criterion)
        - 'ssd': Shortest data description (http://www.sciencedirect.com/science/article/pii/0005109878900055)
        - 'mdl': Minimum description length (https://en.wikipedia.org/wiki/Minimum_description_length)

    Orekhov et al. 2007 (http://imaging.utk.edu/publications/papers/2007/ICIP07_vo.pdf)
    compare these for camera model selection.

    Arguments:
        fit (`lmfit.MinimizerResult`): Model fit (e.g. `Cameras.fit()`)
    """
    n = fit.ndata
    k = fit.nvarys
    base = n * np.log(fit.chisqr / n)
    return dict(
        aic = base + 2 * k,
        caic = base + k * (np.log(n) + 1),
        bic = base + 2 * k * np.log(n),
        ssd = base + k * np.log((n + 2) / 24) + np.log(k + 1),
        mdl = base + 1 / (2 * k * np.log(n)))
