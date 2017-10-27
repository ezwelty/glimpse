import numpy as np
import scipy.optimize

def ransac(data, model, sample_size, max_error, min_inliers, iterations=10):
    """
    Fit model parameters to data using the Random Sample Consensus (RANSAC) algorithm.
    
    Inspired by the pseudocode at https://en.wikipedia.org/wiki/Random_sample_consensus
    
    Arguments:
        data (list, array): Input data to filter with RANSAC
        model (object): Object with the following methods:
        
            - `fit(data[i])`: Accepts `data` sample and returns model parameters
            - `errors(params, data[i])`: Accepts `data` sample and model parameters and returns an error for each `data` element
        
        sample_size (int): Size of `data` sample used to fit the model in each iteration
        max_error (float): Error below which a data point is considered a model inlier
        min_inliers (int): Number of inliers (in addition to `sample_size`) for a model to be considered valid
        iterations (int): Number of iterations
    
    Returns:
        array (int): Values of model parameters
        array (int): Index of model inliers
    """
    i = 0
    params = None
    err = np.inf
    inlier_idx = None
    while i < iterations:
        maybe_idx, test_idx = ransac_sample(sample_size, len(data))
        maybe_inliers = data[maybe_idx]
        maybe_params = model.fit(maybe_inliers)
        if maybe_params is None:
            continue
        test_data = data[test_idx]
        test_errs = model.errors(maybe_params, test_data)
        also_idx = test_idx[test_errs < max_error]
        if len(also_idx) > min_inliers:
            also_inliers = data[also_idx]
            better_data = np.concatenate((maybe_inliers, also_inliers))
            better_params = model.fit(better_data)
            if better_params is None:
                continue
            better_errs = model.errors(better_params, better_data)
            this_err = np.mean(better_errs)
            if this_err < err:
                params = better_params
                err = this_err
                inlier_idx = np.concatenate((maybe_idx, also_idx))
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

class Polynomial:
    """
    Least-squares 1-dimensional polynomial model.

    Fits a polynomial of degree `deg` to points (x, y) (rows of `data`) and
    returns the coefficients that minimize the squared error (`params`).
    
    Attributes:
        deg (int): Degree of the polynomial
    """
    
    def __init__(self, deg=1):
        self.deg = deg
    def fit(self, data):
        return np.polyfit(data[:, 0], data[:, 1], deg=self.deg)
    def predict(self, params, data):
        return np.polyval(params, data[:, 0])
    def errors(self, params, data):
        prediction = self.predict(params, data)
        return np.abs(prediction - data[:, 1])

class Camera:
    """
    Camera model optimization from paired image-world coordinates.

    Fits camera model parameters to image-world point correspondences (uv, xyz) (rows of `data`) and
    returns the values that minimize the reprojection error (`params`).
    
    Arguments:
        params (dict): Parameters of `cam` to optimize (see Camera.optimize)
        
    Attributes:
        cam (Camera): Camera to optimize
        mask (array): Binary mask specifying which elements of `cam.vector` to optimize
        directions (bool): Whether `xyz` are absolute coordinates (False) or ray directions (True).
            If True, 'xyz' cannot be in `params`.
        tol (float): Tolerance for termination (see scipy.optimize.root)
        options (dict): Solver options (see scipy.optimize.root)
    """
    
    def __init__(self, cam, params={'viewdir': True}, directions=False, tol=None, options=None):
        if directions and 'xyz' in params:
            raise ValueError("'xyz' cannot be in `params` when `directions` is True")
        self.cam = cam
        self.mask = cam._vector_mask(params)
        self.directions = directions
        self.tol = tol
        self.options = options
    def predict(self, params, data):
        cam = self.cam.copy()
        cam._update_vector(self.mask, params)
        return cam.project(data[:, 2:5], directions=self.directions)
    def residuals(self, params, data):
        prediction = self.predict(params, data)
        return (prediction - data[:, 0:2]).flatten()
    def errors(self, params, data):
        prediction = self.predict(params, data)
        return np.sqrt(np.sum((prediction - data[:, 0:2]) ** 2, axis=1))
    def fit(self, data):
        result = scipy.optimize.root(self.residuals, self.cam.vector[self.mask], args=(data),
            method='lm', tol=self.tol, options=self.options)
        if result['success']:
            return result['x']
        else:
            print result['message']
            return None
