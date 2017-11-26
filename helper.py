import numpy as np
import cPickle

# Save and load commands for efficient pickle objects
def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = cPickle.load(f)
        return loaded_object

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def merge_dicts(*args):
    """
    Merge dictionaries.

    Precedence goes to the latter value for each key.

    Arguments:
        *args: Dictionaries
    """
    merge = dict()
    for d in args:
        merge.update(d)
    return merge

def format_list(obj, length=1, default=None, dtype=float, ltype=np.array):
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
