import numpy as np
import scipy.optimize
import scipy.spatial
# HACK: See https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python#comment56913201_21789908
import sys
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot

# ---- Controls ----

# Points and Matches support RANSAC with the following API:
# .size()
# .observed(index)
# .predicted(index)

# Lines do not support RANSAC, providing only:
# .observed()
# .predicted()

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
        Count the number of point pairs.
        """
        return len(self.uv)
    
    def observed(self, index=slice(None)):
        """
        Retrieve the observed image coordinates.
        
        Arguments:
            index (array_like): Integer indices of points to return
        """
        return self.uv[index]
    
    def predicted(self, index=slice(None)):
        """
        Predict image coordinates from world coordinates.
        
        If the camera position (`cam.xyz`) has changed and `xyz` are ray directions (`directions=True`),
        the point correspondences are invalid and an error is raised.
        
        Arguments:
            index (array_like): Integer indices of world points to project
        """
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
        Plot the reprojection errors as quivers.
        
        Arrows point from the observed to the predicted coordinates.
        
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
    
    def observed(self):
        """
        Retrieve the observed image coordinates.
        
        Returns:
            array: Image coordinates (Nx2)
        """
        return self.uvi
    
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
        # TODO: Clip world line to camera view. Needed for lines passing behind camera.
        # Project world lines to camera
        xys = [self.cam._world2camera(xyz, directions=self.directions) for xyz in self.xyzs]
        # Interpolate so that projected vertices are ~1 pixel apart
        xyis = [interpolate_line(xy, step=1/self.cam.f.mean(), normalized=False) for xy in xys]
        # Project camera line onto image
        return [self.cam._camera2image(xyi) for xyi in xyis]
    
    def predicted(self):
        """
        Retrieve the projected world coordinates nearest to the image coordinates.
        
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
    
    def plot(self, scale=1, errors="red", observed="green", predicted="yellow", **quiver_args):
        """
        Plot the reprojection errors as quivers.
        
        Arrows point from the observed to the predicted image coordinates.
        
        Arguments:
            scale (float): Scale of quivers
            errors (color): Matplotlib color for quivers, or `None` to hide
            observed (color): Matplotlib color for image lines, or `None` to hide
            predicted (color): Matplotlib color for world lines, or `None` to hide
            **quiver_args: Further arguments to matplotlib.pyplot.quiver
        """
        # Plot image line
        if observed:
            for uv in self.uvs:
                matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], color=observed)
        # Plot world line
        if predicted:
            puvs = self.project()
            for puv in puvs:
                matplotlib.pyplot.plot(puv[:, 0], puv[:, 1], color=predicted)
        # Plot errors
        if errors:
            uvi = self.observed()
            if not predicted:
                puvs = self.project()
            puv = np.row_stack(puvs)
            min_index = find_nearest_neighbors(uvi, puv)
            duv = scale * (puv[min_index, :] - uvi)
            matplotlib.pyplot.quiver(
                uvi[:, 0], uvi[:, 1], duv[:, 0], duv[:, 1],
                color=errors, scale=1, scale_units='xy', angles='xy', **quiver_args)

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
        Count the number of point pairs.
        """
        return len(self.uvs[0])
    
    def observed(self, index=slice(None), cam=0):
        """
        Retrieve the observed image coordinates for a camera.
        
        Arguments:
            index (array_like): Integer indices of points to return
            cam (Camera or int): Camera object
        """
        cam_idx = self.cam_index(cam)
        return self.uvs[cam_idx][index]
    
    def predicted(self, index=slice(None), cam=0):
        """
        Predict image coordinates for a camera from the coordinates of the other camera.
        
        If the cameras are not at the same position, the point correspondences cannot be
        projected explicitly and an error is raised.
        
        Arguments:
            index (array_like): Integer indices of points to project from other camera
            cam (Camera or int): Camera object to project points into
        """
        if not self.is_static():
            raise ValueError("Cameras have different positions ('xyz')")
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

# Polynomial and Cameras support RANSAC with the following API:
# .data_size()
# .fit(index)
# .errors(params, index)

# However, Cameras only supports RANSAC if 1-2 cameras and 1 Points or Matches control.

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
    
    Can only be used with RANSAC algorithm (see optimize.ransac) with 1-2 cameras and 1 Points or Matches control.
    
    Arguments:
        cam_params (dict or list): Parameters to optimize seperately for each camera. For example:
                
                - {'viewdir': True} : All `viewdir` elements
                - {'viewdir': 0} : First `viewdir` element
                - {'viewdir': [0, 1]} : First and second `viewdir` elements
        
        group_params (dict): Parameters to optimize for all cameras (see `cam_params`)
    
    Attributes:
        cams (list): Camera objects
        controls (list): Camera control (Points, Lines, and Matches objects)
        vectors (list): Original camera vectors (for resetting camera parameters)
        cam_masks (list): Boolean camera masks computed from `cam_params`
        group_mask (array): Boolean camera mask computed from `group_params`
        tol (float): Tolerance for termination (see scipy.optimize.root)
        options (dict): Solver options (see scipy.optimize.root)
    """
    
    def __init__(self, cams, controls, cam_params={'viewdir': True}, group_params={}, tol=None, options=None):
        self.cams = cams if isinstance(cams, list) else [cams]
        self.vectors = [cam.vector.copy() for cam in self.cams]
        # Drop controls which don't include a listed camera
        controls = controls if isinstance(controls, list) else [controls]
        self.controls = prune_controls(self.cams, controls)
        # Check inputs
        cam_params = cam_params if isinstance(cam_params, list) else [cam_params]
        test_camera_options(self.cams, self.controls, cam_params, group_params)
        # Precompute vector masks
        self.cam_masks = [camera_mask(params) for cam, params in zip(self.cams, cam_params)]
        self.group_mask = camera_mask(group_params)
        # Optimization settings
        self.tol = tol
        self.options = options
    
    def soft_update_cameras(self, params):
        """
        Set camera parameters without updating baseline values.
        
        Original camera vectors (`self.vectors`) are unchanged so the operation can be reversed
        with `self.reset_cameras()`.
        
        Arguments:
            params (array): Parameter values [group | cam0 | cam1 | ...]
        """
        update_cameras(self.cams, values=params, cam_masks=self.cam_masks, group_mask=self.group_mask)
        
    def update_cameras(self, params):
        """
        Set camera parameters and update baseline values.
        
        Original camera vectors (`self.vectors`) are updated so that `self.reset_cameras()` resets
        the cameras to these new values.
        
        Arguments:
            params (array): Parameter values [group | cam0 | cam1 | ...]
        """
        self.soft_update_cameras(params)
        self.vectors = [cam.vector.copy() for cam in self.cams]
    
    def reset_cameras(self, vectors=None):
        """
        Set camera parameters to their baseline values.
        
        Arguments:
            vectors (list): Camera vectors. If `None` (default), `self.vectors` is used.
        """
        if vectors is None:
            vectors = self.vectors
        for cam, vector in zip(self.cams, vectors):
            cam.vector = vector.copy()
    
    def test_ransac(self):
        """
        Test whether RANSAC is supported.
        
        `Cameras` only supports RANSAC (`ransac()`) if:
        
            - `self.cams` is length 1 or 2,
            - `self.controls` is length 1, and
            - `self.controls[0]` is Points or Matches
        """
        if len(self.cams) > 2 or len(self.controls) > 1 or isinstance(self.controls[0], Lines):
            raise ValueError("Only supported for 1-2 Camera and 1 Points or Matches control")
    
    def data_size(self):
        """
        Return the number of data points.
        
        Requires `self.test_ransac()` to be True.
        """
        self.test_ransac()
        return self.controls[0].size()
    
    def observed(self, index=None):
        """
        Return the observed image coordinates for all camera control.
        
        Matches return the image coordinates for the first camera (see `Matches.observed()`).
        
        Arguments:
            index (array or slice): Indices of points to return. If `None` (default), all points are returned.
                Other values require `self.test_ransac()` to be True.
        """
        if index is None:
            return np.vstack((control.observed() for control in self.controls))
        else:
            self.test_ransac()
            return self.controls[0].observed(index)
    
    def predicted(self, params=None, index=None):
        """
        Return the predicted image coordinates for all camera control.
        
        Matches return the image coordinates for the first camera (see `Matches.observed()`).
        
        Arguments:
            params (array): Parameter values [group | cam0 | cam1 | ...]
            index (array or slice): Indices of points to return. If `None` (default), all points are returned.
                Other values require `self.test_ransac()` to be True.
        """
        if params is not None:
            vectors = [cam.vector.copy() for cam in self.cams]
            self.soft_update_cameras(params)
        if index is None:
            result = np.vstack((control.predicted() for control in self.controls))
        else:
            result = self.controls[0].predicted(index)
        if params is not None:
            self.reset_cameras(vectors)
        return result
    
    def residuals(self, params=None, index=None):
        """
        Return the reprojection residuals for all camera control.
        
        Residuals are computed as `self.predicted()` - `self.observed()`
        and flattened to a 1-dimensional array.
        
        Arguments:
            params (array): Parameter values [group | cam0 | cam1 | ...]
            index (array or slice): Indices of points to return. If `None` (default), all points are returned.
                Other values require `self.test_ransac()` to be True.
        """
        return (self.predicted(params, index) - self.observed(index)).flatten()
    
    def errors(self, params=None, index=None):
        """
        Return the reprojection errors for all camera control.
        
        Errors are computed as the distance between `self.predicted()` and `self.observed()`.
        
        Arguments:
            params (array): Parameter values [group | cam0 | cam1 | ...]
            index (array or slice): Indices of points to return. If `None` (default), all points are returned.
                Other values require `self.test_ransac()` to be True.
        """
        # TODO: Skip square root for speed
        return np.sqrt(np.sum((self.predicted(params, index) - self.observed(index)) ** 2, axis=1))
    
    def fit(self, index=None):
        """
        Return optimal camera parameter values.
        
        The Levenberg-Marquardt algorithm is used to find the camera parameter values
        that minimize the reprojection residuals (`self.residuals()`) across all control.
        
        Arguments:
            index (array or slice): Indices of points to return. If `None` (default), all points are returned.
                Other values require `self.test_ransac()` to be True.
                
        Returns:
            array: Parameter values [group | cam0 | cam1 | ...]
        """
        initial_params = sample_cameras(self.cams, self.cam_masks, self.group_mask)
        result = scipy.optimize.root(self.residuals, initial_params, args=(index),
            method='lm', tol=self.tol, options=self.options)
        if result['success']:
            return result['x']
        else:
            print result['message']
            return None
    
    def plot(self, params=None, cam=0, index=None, scale=1, selected="red", unselected=None,
        lines_errors=None, lines_observed="green", lines_predicted="yellow", **quiver_args):
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
            lines_errors (color): Matplotlib color for Lines error quivers, or `None` to hide
            lines_observed (color): Matplotlib color for image lines, or `None` to hide
            lines_predicted (color): Matplotlib color for world lines, or `None` to hide
            **quiver_args: Further arguments to matplotlib.pyplot.quiver
        """
        if params is not None:
            vectors = [camera.vector.copy() for camera in self.cams]
            self.soft_update_cameras(params)
        if index is not None:
            self.test_ransac()
        cam = self.cams[cam] if isinstance(cam, int) else cam
        cam_controls = prune_controls([cam], self.controls)
        for control in cam_controls:
            if isinstance(control, Lines):
                control.plot(scale=scale, errors=lines_errors, observed=lines_observed, predicted=lines_predicted, **quiver_args)
            elif isinstance(control, Points):
                control.plot(index=index, scale=scale, selected=selected, unselected=unselected, **quiver_args)
            elif isinstance(control, Matches):
                control.plot(cam=cam, index=index, scale=scale, selected=selected, unselected=unselected, **quiver_args)
        if params is not None:
            self.reset_cameras(vectors)

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

def camera_mask(params={}):
    """
    Return a boolean mask of the selected camera parameters.
    
    The returned boolean array is intended to be used for setting or getting a subset of the
    image.Camera.vector, which is a flat vector of all Camera attributes [xyz, viewdir, imgsz, f, c, k, p].
    
    Arguments:
        params (dict): Parameters to select by name and indices. For example:
                
                - {'viewdir': True} : All `viewdir` elements
                - {'viewdir': 0} : First `viewdir` element
                - {'viewdir': [0, 1]} : First and second `viewdir` elements
    """
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

def update_cameras(cams, values, cam_masks, group_mask=camera_mask()):
    """
    Set camera parameters for multiple cameras.
    
    Arguments:
        cams (list): Camera objects
        values (array): Parameter values [group | cam0 | cam1 | ... ]
        cam_masks (list): Masks of parameters to set for each camera
        group_mask (array): Mask of parameters to set for all cameras
    """
    n_group = group_mask.sum()
    group_values = values[0:n_group]
    n_cams = [mask.sum() for mask in cam_masks]
    cam_breaks = np.cumsum([n_group] + n_cams)
    for i in range(len(cams)):
        cams[i].vector[cam_masks[i]] = values[cam_breaks[i]:cam_breaks[i + 1]]
        cams[i].vector[group_mask] = group_values

def sample_cameras(cams, cam_masks, group_mask=camera_mask()):
    """
    Get camera parameters for multiple cameras.
    
    Group parameters (`group_mask`) are returned as the mean of all cameras.
    
    Arguments:
        cams (list): Camera objects
        cam_masks (list): Masks of parameters to get for each camera
        group_mask (array): Mask of parameters to get for all cameras
    
    Returns:
        array: Parameter values [group | cam0 | cam1 | ... ]
    """
    # Assign group values to mean of starting camera values
    group_values = np.vstack((cam.vector[group_mask] for cam in cams)).mean(axis=0)
    cam_values = np.hstack((cam.vector[mask] for cam, mask in zip(cams, cam_masks)))
    return np.hstack((group_values, cam_values))

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

def test_camera_options(cams, controls, cam_params, group_params):
    """
    Test Cameras model options for errors.
    
    The following checks are performed:
        
        - Error: Not all cameras appear in controls.
        - Error: 'xyz' cannot be in `group_params` if any `control.directions` is True
        - Error: 'xyz' cannot be in `cam_params` if `control.directions` is True for control involving that camera
    
    Arguments:
        cams (list): Camera objects
        controls (list): Camera control (Points, Lines, and Matches objects)
        cam_params (list): Parameters to optimize seperately for each camera
        group_params (dict): Parameters to optimize for all cameras
    """
    control_cams = [(isinstance(control, Matches) and control.cams) or [control.cam] for control in controls]
    is_directions_control = [isinstance(control, (Points, Lines)) and control.directions for control in controls]
    is_xyz_cam = ['xyz' in params for params in cam_params]
    is_directions_cam = [directions and cam in ctrl_cams
        for cam in cams
            for ctrl_cams, directions in zip(control_cams, is_directions_control)]
    # Error: Not all cameras appear in controls
    control_cams_flat = [cam for cams in control_cams for cam in cams]
    if (set(cams) & set(control_cams_flat)) < len(cams):
        raise ValueErrors("Not all cameras appear in controls")
    # Error: 'xyz' cannot be in `group_params` if any `control.directions` is True
    if 'xyz' in group_params and any(is_directions_control):
        raise ValueError("'xyz' cannot be in `group_params` if any `control.directions` is True")
    # Error: 'xyz' cannot be in `cam_params` if `control.directions` is True for control involving that camera
    is_xyz_directions_cam = [is_xyz and is_directions for is_xyz, is_directions in zip(is_xyz_cam, is_directions_cam)]
    if any(is_xyz_directions_cam):
        raise ValueError("'xyz' cannot be in `cam_params` if `control.directions` is True for control involving that camera")
    return None
