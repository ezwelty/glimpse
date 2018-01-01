import numpy as np
import scipy.spatial
import cv2
import lmfit
import matplotlib
import helper
import sys
sys.path.append('./cg-calibrations')
import cgcalib

# ---- Controls ----

# Controls (within Cameras) support RANSAC with the following API:
# .size()
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
    """

    def __init__(self, cam, uv, xyz, directions=False):
        if len(uv) != len(xyz):
            raise ValueError("`uv` and `xyz` have different number of rows")
        self.cam = cam
        self.uv = uv
        self.xyz = xyz
        self.directions = directions
        self.original_cam_xyz = cam.xyz.copy()

    def size(self):
        """
        Return the total number of point pairs.
        """
        return len(self.uv)

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
            raise ValueError("Camera has changed position ('xyz') and `directions=True`")
        return self.cam.project(self.xyz[index], directions=self.directions)

    def is_static(self):
        """
        Test whether the camera is at its original position.
        """
        return (self.cam.xyz == self.original_cam_xyz).all()

    def plot(self, index=None, scale=1, selected="red", unselected=None, **quiver_args):
        """
        Plot reprojection errors as quivers.

        Arrows point from observed to predicted coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            scale (float): Scale of quivers
            selected (color): Matplotlib color for selected points, or `None` to hide
            unselected (color): Matplotlib color for unselected points, or `None` to hide
            **quiver_args: Further arguments to matplotlib.pyplot.quiver
        """
        if index is None:
            index = slice(None)
            other_index = slice(0)
        else:
            other_index = np.delete(np.arange(self.size()), index)
        uv = self.observed()
        puv = self.predicted()
        duv = scale * (puv - uv)
        if selected is not None:
            matplotlib.pyplot.quiver(
                uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1],
                color=selected, scale=1, scale_units='xy', angles='xy', **quiver_args)
        if unselected is not None:
            matplotlib.pyplot.quiver(
                uv[other_index, 0], uv[other_index, 1], duv[other_index, 0], duv[other_index, 1],
                color=unselected, scale=1, scale_units='xy', angles='xy', **quiver_args)

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
        step (float): Along-line distance between image points interpolated from lines `uvs`
        original_cam_xyz (array): Original camera position (`cam.xyz`)
    """

    def __init__(self, cam, uvs, xyzs, directions=False, step=None):
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
        self.original_cam_xyz = cam.xyz.copy()

    def size(self):
        """
        Return the number of image points.
        """
        return len(self.uvi)

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
            raise ValueError("Camera has changed position ('xyz') and `directions=True`")
        xy_step = 1 / self.cam.f.mean()
        uv_edges = self.cam.edges(step = self.cam.imgsz / 2)
        xy_edges = self.cam._image2camera(uv_edges)
        xy_box = np.hstack((np.min(xy_edges, axis=0), np.max(xy_edges, axis=0)))
        puvs = []
        for xyz in self.xyzs:
            # TODO: Instead, clip lines to 3D polar viewbox before projecting
            # Project world lines to camera
            xy = self.cam._world2camera(xyz, directions=self.directions)
            # Discard nan values (behind camera)
            lines = helper.boolean_split(xy, np.isnan(xy[:, 0]), include='false')
            for line in lines:
                # Clip lines in view
                # Resolves coordinate wrap around with large distortion
                for cline in helper.clip_polyline_box(line, xy_box):
                    # Interpolate clipped lines to ~1 pixel density
                    puvs.append(self.cam._camera2image(
                        interpolate_line(np.array(cline), step=xy_step, normalized=False)))
        if puvs:
            return puvs
        else:
            # FIXME: Fails if lines slip out of camera view
            # TODO: Return center of image instead of error?
            raise ValueError("All line vertices are outside camera view")

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

    def plot(self, index=None, scale=1, selected="red", unselected=None,
        observed="green", predicted="yellow", **quiver_args):
        """
        Plot the reprojection errors as quivers.

        Arrows point from observed to predicted image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            scale (float): Scale of quivers
            selected (color): Matplotlib color for selected points, or `None` to hide
            unselected (color): Matplotlib color for unselected points, or `None` to hide
            observed (color): Matplotlib color for image lines, or `None` to hide
            predicted (color): Matplotlib color for world lines, or `None` to hide
            **quiver_args: Further arguments to matplotlib.pyplot.quiver
        """
        # Plot image lines
        if observed:
            for uv in self.uvs:
                matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], color=observed)
        # Plot world lines
        if predicted:
            puvs = self.project()
            for puv in puvs:
                matplotlib.pyplot.plot(puv[:, 0], puv[:, 1], color=predicted)
        # Plot errors
        if selected or unselected:
            if index is None:
                index = slice(None)
                other_index = slice(0)
            else:
                other_index = np.delete(np.arange(self.size()), index)
            uv = self.observed()
            if not predicted:
                puvs = self.project()
            puv = np.row_stack(puvs)
            min_index = find_nearest_neighbors(uv, puv)
            duv = scale * (puv[min_index, :] - uv)
            if selected is not None:
                matplotlib.pyplot.quiver(
                    uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1],
                    color=selected, scale=1, scale_units='xy', angles='xy', **quiver_args)
            if unselected is not None:
                matplotlib.pyplot.quiver(
                    uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1],
                    color=unselected, scale=1, scale_units='xy', angles='xy', **quiver_args)

class Matches(object):
    """
    `Matches` store image-image point correspondences.

    The image coordinates (`uvs[i]`) of one camera (`cams[i]`) are projected into the other camera (`cams[j]`),
    then compared to the expected image coordinates for that camera (`uvs[j]`).

    Attributes:
        cams (list): Pair of Camera objects
        uvs (list): Pair of image coordinate arrays (Nx2)
    """

    def __init__(self, cams, uvs):
        if len(cams) != 2 or len(uvs) != 2:
            raise ValueError("`cams` and `uvs` must each have two elements")
        if cams[0] is cams[1]:
            raise ValueError("Both cameras are the same object")
        if uvs[0].shape != uvs[1].shape:
            raise ValueError("Image coordinate arrays have different shapes")
        self.cams = cams
        self.uvs = uvs

    def size(self):
        """
        Return the number of point pairs.
        """
        return len(self.uvs[0])

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
            raise ValueError("Cameras have different positions ('xyz')")
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
                raise IndexError("Camera index out of range")
            return cam
        else:
            return self.cams.index(cam)

    def plot(self, index=None, cam=0, scale=1, selected="red", unselected=None, **quiver_args):
        """
        Plot the reprojection errors as quivers.

        Arrows point from the observed to the predicted coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            cam (Camera or int): Camera to plot
            scale (float): Scale of quivers
            selected (color): Matplotlib color for selected points, or `None` to hide
            unselected (color): Matplotlib color for unselected points, or `None` to hide
            **quiver_args: Further arguments to matplotlib.pyplot.quiver
        """
        if index is None:
            index = slice(None)
            other_index = slice(0)
        else:
            other_index = np.delete(np.arange(self.size()), index)
        uv = self.observed(cam=cam)
        puv = self.predicted(cam=cam)
        duv = scale * (puv - uv)
        if selected is not None:
            matplotlib.pyplot.quiver(
                uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1],
                color=selected, scale=1, scale_units='xy', angles='xy', **quiver_args)
        if unselected is not None:
            matplotlib.pyplot.quiver(
                uv[other_index, 0], uv[other_index, 1], duv[other_index, 0], duv[other_index, 1],
                color=unselected, scale=1, scale_units='xy', angles='xy', **quiver_args)

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

    def plot(self, params=None, index=slice(None), selected="red", unselected="grey", polynomial="red"):
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
        - image-image point coordinates (Matches)

    If used with RANSAC (see `optimize.ransac`) with multiple control objects,
    results may be unstable since samples are drawn randomly from all observations,
    and computation will be slow since errors are calculated for all points then subset.

    Arguments:
        cam_params (list): Parameters to optimize for each camera seperately (see `parse_params()`)
        group_params (dict): Parameters to optimize for all cameras (see `parse_params()`)
        nan_policy (str): Action taken by `lmfit.Minimizer` for NaN values
            (https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer)

    Attributes:
        cams (list): Camera objects
        controls (list): Camera control (Points, Lines, and Matches objects)
        vectors (list): Original camera vectors (for resetting camera parameters)
        params (`lmfit.Parameters`): Parameter initial values and bounds
        options (dict): Additional arguments passed to `lmfit.Minimizer`
    """

    def __init__(self, cams, controls, cam_params=dict(viewdir=True), group_params=dict(), weights=None, nan_policy="omit", **options):
        self.cams = cams if isinstance(cams, list) else [cams]
        self.vectors = [cam.vector.copy() for cam in self.cams]
        controls = controls if isinstance(controls, list) else [controls]
        self.controls = prune_controls(self.cams, controls)
        self.cam_params = cam_params if isinstance(cam_params, list) else [cam_params]
        self.group_params = group_params
        test_cameras(self)
        temp = [parse_params(params) for params in self.cam_params]
        self.cam_masks = [x[0] for x in temp]
        self.cam_bounds = [x[1] for x in temp]
        self.group_mask, self.group_bounds = parse_params(self.group_params)
        # TODO: Avoid computing masks and bounds twice
        self.params, self.apply_params = build_lmfit_params(self.cams, self.cam_params, group_params)
        self.weights = weights
        self.options = options
        self.options['nan_policy'] = nan_policy
        if not self.options.has_key('diag'):
            # Compute optimal variable scale factors
            scales = [camera_scale_factors(cam, self.controls) for cam in self.cams]
            # TODO: Weigh each camera by number of control points
            group_scale = np.vstack((scale[self.group_mask] for scale in scales)).mean(axis=0)
            cam_scales = np.hstack((scale[mask] for scale, mask in zip(scales, self.cam_masks)))
            self.options['diag'] = np.hstack((group_scale, cam_scales))

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
        return np.array([control.size() for control in self.controls]).sum()

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

    def fit(self, index=None, cam_params=None, group_params=None, full=False):
        """
        Return optimal camera parameter values.

        The Levenberg-Marquardt algorithm is used to find the camera parameter values
        that minimize the reprojection residuals (`.residuals()`) across all control.
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

        Returns:
            array or `lmfit.Parameters` (`full=True`): Parameter values ordered first
                by group or camera (group, cam0, cam1, ...),
                then ordered by position in `Camera.vector`.
        """
        def callback(params, iter, resid, *args, **kwargs):
            err = np.sqrt(np.sum(resid.reshape(-1, 2)**2, axis=1)).mean()
            sys.stdout.write("\r" + str(err))
            sys.stdout.flush()
        iterations = max(
            len(cam_params) if cam_params else 0,
            len(group_params) if group_params else 0)
        if iterations:
            for n in range(iterations):
                iter_cam_params = cam_params[n] if cam_params else self.cam_params
                iter_group_params = group_params[n] if group_params else self.group_params
                options = self.options.copy()
                options.pop('diag', None)
                model = Cameras(self.cams, self.controls, iter_cam_params, iter_group_params, **options)
                values = model.fit(index=index)
                if values is not None:
                    model.set_cameras(params=values)
            self.update_params()
        minimizer = lmfit.Minimizer(self.residuals, self.params, iter_cb=callback, fcn_kws=dict(index=index), **self.options)
        result = minimizer.leastsq()
        sys.stdout.write("\n")
        values = np.array(result.params.valuesdict().values())
        if iterations:
            self.reset_cameras()
            self.update_params()
        if not result.success:
            print result.message
        if full:
            return result
        elif result.success:
            return np.array(result.params.valuesdict().values())

    def plot(self, params=None, cam=0, index=None, scale=1, selected="red", unselected=None,
        lines_observed="green", lines_predicted="yellow", **quiver_args):
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
            selected (color): Matplotlib color for selected points, or `None` to hide
            unselected (color): Matplotlib color for unselected points, or `None` to hide
            lines_observed (color): Matplotlib color for image lines, or `None` to hide
            lines_predicted (color): Matplotlib color for world lines, or `None` to hide
            **quiver_args: Further arguments to matplotlib.pyplot.quiver
        """
        if params is not None:
            vectors = [camera.vector.copy() for camera in self.cams]
            self.set_cameras(params)
        cam = self.cams[cam] if isinstance(cam, int) else cam
        cam_controls = prune_controls([cam], self.controls)
        if index is not None and len(self.control) > 1:
            # TODO: Map index to subindices for each control
            raise ValueError("Plotting with `index` not yet supported with multiple controls")
        for control in cam_controls:
            if isinstance(control, Lines):
                control.plot(index=index, scale=scale, selected=selected, unselected=unselected,
                    observed=lines_observed, predicted=lines_predicted, **quiver_args)
            elif isinstance(control, Points):
                control.plot(index=index, scale=scale, selected=selected, unselected=unselected, **quiver_args)
            elif isinstance(control, Matches):
                control.plot(cam=cam, index=index, scale=scale, selected=selected, unselected=unselected, **quiver_args)
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

class OrthoImage(object):
    def __init__(self, img, dem, ortho, params):
        self.img = img
        self.vector = img.cam.vector.copy()
        self.I = img.read(gray=True)
        self.dem = dem
        self.ortho = ortho
        self.params, self.set_camera = build_lmfit_params([img.cam], [params])
        self.visible = dem.visible(img.cam.xyz)
        self.options = dict(nan_policy='omit')
        self.options['diag'] = camera_scale_factors(img.cam)[parse_params(params)[0]]

    def reset_camera(self):
        self.img.cam.vector = self.vector.copy()

    def correlation_coefficient(self, patch1, patch2):
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            return product / stds

    def observed(self):
        return self.I

    def predicted(self, params=None):
        if params is not None:
            self.set_camera(params)
        return cgcalib.dem_to_image(self.img.cam, self.dem, self.ortho.Z, mask=self.visible)

    def correlations(self, params=None, radius=2, step=10):
        I_obs = self.observed()
        I_sim = self.predicted(params)
        nrows, ncols = I_obs.shape
        correl = np.zeros_like(I_obs, dtype=float)
        for i in range(radius, nrows - (radius + 1), step):
            for j in range(radius, ncols - (radius + 1), step):
                correl[i, j] = self.correlation_coefficient(
                    I_sim[i - radius: i + radius + 1, j - radius: j + radius + 1],
                    I_obs[i - radius: i + radius + 1, j - radius: j + radius + 1])
        correl[correl == 0] = np.nan
        return correl

    def residuals(self, params=None, **kwargs):
        return 1 - self.correlations(params=params, **kwargs)

    def fit(self):
        def callback(params, iter, resid, *args, **kwargs):
            err = resid.mean()
            sys.stdout.write("\r" + str(err))
            sys.stdout.flush()
        minimizer = lmfit.Minimizer(self.residuals, self.params, iter_cb=callback, **self.options)
        result = minimizer.leastsq()
        sys.stdout.write("\n")
        if not result.success:
            print result.message
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
        raise ValueError("Best fit does not meet acceptance criteria")
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
        raise ValueError("`sample_size` is larger or equal to `data_size`")
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    return indices[:sample_size], indices[sample_size:]

# ---- Build matches ----

def sift_matches(images, masks=None, ratio=0.7, nfeatures=0, **params):
    """
    Return `Matches` constructed from SIFT matches between sequential images.

    Arguments:
        images (list): Image objects.
            Matches are computed between each sequential pair.
        masks (list or array): Regions in which to detect features (uint8) -
            either in all images (if array) or in each image (if list)
        ratio (float): Maximum distance ratio of preserved matches (lower ratio ~ better match),
            calculated as the ratio between the distance of the nearest and second nearest match.
            See http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20
        nfeatures (int): The number of best features (ranked by local contrast) to retain
            from each image for matching, or `0` for all
        **params: Additional arguments passed to `cv2.SIFT()`.
            See https://docs.opencv.org/2.4.13/modules/nonfree/doc/feature_detection.html#sift-sift

    Returns:
        list: Matches objects
    """
    if masks is None or isinstance(masks, np.ndarray):
        masks = (masks, ) * len(images)
    sift = cv2.SIFT(nfeatures=nfeatures, **params)
    # Extract keypoints (keypoints, descriptors)
    keypoints = [(sift.detectAndCompute(img.read(), mask=mask)) for img, mask in zip(images, masks)]
    # Match keypoints
    # TODO: Find algorithm definitions for FlannBasedMatcher index parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    controls = list()
    for i in range(len(images) - 1):
        M = flann.knnMatch(keypoints[i][1], keypoints[i + 1][1], k=2)
        is_good = np.array([m.distance / n.distance for m, n in M]) < ratio
        A = np.array([keypoints[i][0][m.queryIdx].pt for m, n in M])[is_good, :]
        B = np.array([keypoints[i + 1][0][m.trainIdx].pt for m, n in M])[is_good, :]
        controls.append(Matches((images[i].cam, images[i + 1].cam), (A, B)))
    return controls

def surf_matches(images, masks=None, ratio=0.7, hessianThreshold=1e3, **params):
    """
    Return `Matches` constructed from SURF matches between sequential images.

    Arguments:
        images (list): Image objects.
            Matches are computed between each sequential pair.
        masks (list or array): Regions in which to detect features (uint8) -
            either in all images (if array) or in each image (if list)
        ratio (float): Maximum distance ratio of preserved matches (lower ratio ~ better match),
            calculated as the ratio between the distance of the nearest and second nearest match.
            See http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20
        hessianThreshold (float): Threshold for the hessian keypoint detector used in SURF
        **params: Additional arguments passed to `cv2.SURF()`.
            See https://docs.opencv.org/2.4.13/modules/nonfree/doc/feature_detection.html#surf-surf

    Returns:
        list: Matches objects
    """
    if masks is None or isinstance(masks, np.ndarray):
        masks = (masks, ) * len(images)
    surf = cv2.SURF(hessianThreshold=hessianThreshold, **params)
    # Extract keypoints (keypoints, descriptors)
    keypoints = [(surf.detectAndCompute(img.read(), mask=mask)) for img, mask in zip(images, masks)]
    # Match keypoints
    # TODO: Find algorithm definitions for FlannBasedMatcher index parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    controls = list()
    for i in range(len(images) - 1):
        M = flann.knnMatch(keypoints[i][1], keypoints[i + 1][1], k=2)
        is_good = np.array([m.distance / n.distance for m, n in M]) < ratio
        A = np.array([keypoints[i][0][m.queryIdx].pt for m, n in M])[is_good, :]
        B = np.array([keypoints[i + 1][0][m.trainIdx].pt for m, n in M])[is_good, :]
        controls.append(Matches((images[i].cam, images[i + 1].cam), (A, B)))
    return controls

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
        raise ValueError("No controls reference the cameras")
    # Error: 'f' or 'c' in `group_params` but image sizes not equal
    if model.group_params.has_key('f') or model.group_params.has_key('c'):
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
    if (set(model.cams) & set(control_cams_flat)) < len(model.cams):
        raise ValueErrors("Not all cameras appear in controls")
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
                weights.append(control.size())
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

def parse_params(params=None):
    """
    Return a mask of selected camera parameters and associated bounds.

    Arguments:
        params (dict): Parameters to select by name and indices. For example:

            - {'viewdir': True} : All `viewdir` elements
            - {'viewdir': 0} : First `viewdir` element
            - {'viewdir': [0, 1]} : First and second `viewdir` elements

            Bounds can be specified inside a tuple (indices, min, max).
            Singletons are expanded as needed, and `None` can be used to
            indicate no bound. The following are equivalent:

            - {'viewdir': ([0, 1], None, 180)}
            - {'viewdir': ([0, 1], None, [180, 180])}
            - {'viewdir': ([0, 1], [None, None], [180, 180])}

    Returns:
        array: Parameter boolean mask (20,)
        array: Parameter min and max bounds (20, 2)
    """
    if params is None:
        params = dict()
    attributes = ['xyz', 'viewdir', 'imgsz', 'f', 'c', 'k', 'p']
    indices = [0, 3, 6, 8, 10, 12, 18, 20]
    mask = np.full(20, False)
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
    attributes = ['xyz', 'viewdir', 'imgsz', 'f', 'c', 'k', 'p']
    lengths = [3, 3, 2, 2, 2, 6, 2]
    base_labels = np.array([attribute + str(i) for attribute, length in zip(attributes, lengths) for i in range(length)])
    group_labels = base_labels[group_mask]
    cam_labels = np.hstack(("cam" + str(i) + "_" + label for i, mask in enumerate(cam_masks) for label in base_labels[mask]))
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
            values = np.array(values.valuesdict().values())
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
        mdl = base + 1 / (2 * k * np.log(n))
    )
