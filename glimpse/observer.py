from .imports import (np, sklearn, scipy, cv2, matplotlib, datetime)
from . import (helpers)

class Observer(object):
    """
    An `Observer` contains a sequence of `Image` objects and the methods to compute
    the misfit between image subsets.

    Arguments:
        reference_halwidth: Halfwidth of reference subimage
        search_halfwidth: Halfwidth of test subimage

    Attributes:
        images (list): Image objects
        datetimes (array): Image capture times
        ref_template (array): a subimage which acts as a color palette to match
        pca (sklearn.decomposition.PCA): Principal components analyzer
        sigma_pixel (float): Average std between all images for a pixel
        hw_row, hw_col, sd_row, sd_col: Halfwidths for reference and test subimages
        row_diff_0, col_diff_0: Residuals between prediction in pixel coords and viable index
        ref_subimage: Reference subimage
        rows_predicted, cols_predicted:
    """

    def __init__(self, images, sigma_pixel=0.3, reference_halfwidth=15, search_halfwidth=20):
        self.images = images
        self.datetimes = np.array([img.datetime for img in self.images])
        self.pca = sklearn.decomposition.PCA(n_components=1, svd_solver='arpack', whiten=True)
        self.sigma_pixel = sigma_pixel
        self.hw_row = reference_halfwidth
        self.hw_col = reference_halfwidth
        self.sd_row = search_halfwidth
        self.sd_col = search_halfwidth
        self.ref_hist_template = None

    def image_index(self, img, max_seconds=0.1):
        """
        Retrieve the integer index of an image.

        Arguments:
            img: Either an index (int),
                Image to find in `self.images`, or
                Date and time to match against `self.datetimes` (datetime).
            max_seconds (float): If `img` is datetime,
                maximum distance in seconds to be considered a match.
        """
        if isinstance(img, int):
            try:
                self.images[img]
            except IndexError:
                raise IndexError("Index out of range")
            return img
        elif isinstance(img, datetime.datetime):
            delta_seconds = np.abs([dt.total_seconds()
                for dt in (img - self.datetimes)])
            index = np.argmin(delta_seconds)
            if delta_seconds[index] > max_seconds:
                distance = delta_seconds[index] - max_seconds
                raise IndexError("Nearest image out of range by " + str(distance) + " seconds")
            return index
        else:
            return self.images.index(img)

    def link_tracker(self, tracker):
        """
        Assign Tracker object.

        Sets some initial variables associated with the initial point.

        Arguments:
            tracker (Tracker): Tracker object
        """
        col_0, row_0 =  self.images[0].cam.project(tracker.particle_mean[:, [0, 1, 4]]).squeeze()
        self.rc = self.images[0].cam.project(tracker.particles[:, [0, 1, 4]])[:, ::-1]
        self.log_like = np.zeros((tracker.n))
        self.ref_hist_template = self.get_subimage(row_0, col_0, self.sd_row, self.sd_col, 0, get_ref_template=True)
        self.ref_subimage = self.get_subimage(row_0, col_0, self.hw_row, self.hw_col, 0)

    def get_subimage(self, row, col, hw_row, hw_col, image_index, do_histogram_matching=True, do_pca=True, do_median_filtering=True, median_filter_size=(5, 5), get_ref_template=False):
        """
        Extract a subimage given a row and column, and apply some image processing.

        Arguments:
            row, col: row and column index
            hw_row, hw_col: halfwidth of subimage to extract
            image_index: which stored image to extract
            pca: apply PCA transformation to grayscale
            median_filter_size: apply highpass filter with specified bandwidth

        Returns:
            array: Extracted subimage
        """
        row_nearest = int(round(row))
        col_nearest = int(round(col))
        row_hw = np.linspace(row - hw_row, row + hw_row, 2 * hw_row + 1, endpoint=True)
        col_hw = np.linspace(col - hw_col, col + hw_col, 2 * hw_col + 1, endpoint=True)
        chip = self.images[image_index].read()[
            (row_nearest - hw_row - 1):(2 + row_nearest + hw_row),
            (col_nearest - hw_col - 1):(2 + col_nearest + hw_col), :].copy()
        if get_ref_template:
            return chip
        rows = range(row_nearest - hw_row - 1, row_nearest + hw_row + 2)
        cols = range(col_nearest - hw_col - 1, col_nearest + hw_col + 2)
        if do_histogram_matching:
            # If a reference template has been defined, perform histogram matching.  This is
            # very helpful for ameliorating the effects of illumination changes.
            chip[:, :, 0] = helpers.hist_match(chip[:, :, 0], self.ref_hist_template[:, :, 0])
            chip[:, :, 1] = helpers.hist_match(chip[:, :, 1], self.ref_hist_template[:, :, 1])
            chip[:, :, 2] = helpers.hist_match(chip[:, :, 2], self.ref_hist_template[:, :, 2])
        if do_pca:
            # If a pca has been defined, compute an intensity image using it.
            m, n, q = chip.shape
            Q = chip.reshape((m*n, q))
            self.pca.fit(Q)
            self.pca.components_ = np.sign(self.pca.components_[0]) * self.pca.components_
            Qp = self.pca.transform(Q)
            chip = Qp.reshape((m, n))
        Ri = scipy.interpolate.RectBivariateSpline(rows, cols, chip, kx=3, ky=3)
        chip = Ri(row_hw, col_hw, grid=True)
        if do_median_filtering:
            # Do median highpass filtering
            chip_lowpass = scipy.ndimage.filters.median_filter(chip, (median_filter_size[0], median_filter_size[1]))
            chip -= chip_lowpass
        return chip

    def compute_likelihood(self, pmean, particles, t):
        """
        Compute the likelihood of each particle in a population based on SSE between a reference and test subimage.

        Arguments:
            pmean: particle mean values
            particles: particle array
            t (datetime): Target date and time

        Returns:
            log_likelihood: log likelihood for each particle
        """
        # Check for an image at the current time.
        try:
            image_index = self.image_index(t, max_seconds=0.1)
        except IndexError:
            # If none, return a constant likelihood
            return 0.0
        # Extract a subimage around predicted coordinates
        current_image = self.images[image_index]
        col_1, row_1 = current_image.cam.project(pmean[:, [0, 1, 4]]).squeeze()
        self.test_subimage = self.get_subimage(row_1, col_1, self.sd_row, self.sd_col, image_index)
        # Try generating a likelihood interpolant and evaluate for each particle
        #try:
        like_interpolant = self.get_likelihood_interpolant(self.ref_subimage, self.test_subimage, row_1, col_1)
        self.rc = current_image.cam.project(particles[:, [0, 1, 4]])[:, ::-1]
        self.log_like = like_interpolant(self.rc[:, 0], self.rc[:, 1], grid=False) / self.sigma_pixel**2
        # If too close to the image boundary, return a constant log_likelihood
        #except IndexError:
        #    log_like = 1.
        return self.log_like

    def get_likelihood_interpolant(self, ref_subimage, test_subimage, row, col):
        """
        Produce an object for evaluating the log likelihood of particles.

        Arguments:
            ref_subimage, test_subimage:
            row_nearest, col_nearest: the row and column corresponding to the center of test_subimage
            row_diff_*: corrections for rounding to indices
        Returns:
            local_interp: A Bivariate Spline that can be used to assign
                          likelihood to particles
        """
        # Compute normalized correlation coefficients
        rhos = cv2.matchTemplate(test_subimage.astype('float32'), ref_subimage.astype('float32'), method=cv2.TM_SQDIFF)
        rhos /= (ref_subimage.shape[0] * ref_subimage.shape[1])
        rcoords = row - (self.sd_row - self.hw_row) + np.array(range(rhos.shape[0]))
        ccoords = col - (self.sd_col - self.hw_col) + np.array(range(rhos.shape[1]))
        # Get subpixel accuracy by fitting the correlation surface with a cubic spline and maximizing
        local_interp = scipy.interpolate.RectBivariateSpline(rcoords, ccoords, rhos, kx=3, ky=3)
        return local_interp

    # Real time plotting utilities
    def initialize_plot(self, ax):
        self.ax = ax
        self.im_plot = ax.imshow(self.images[0].I, interpolation='none')
        self.le0, self.re0, self.be0, self.te0 = self.im_plot.get_extent()
        #self.spprd = ax.scatter(self.cols_predicted, self.rows_predicted, s=50, c='green', label='Prior Prediction')
        self.sppnts = self.ax.scatter(self.rc[:, 1], self.rc[:, 0], s=25, c=-self.log_like, cmap=matplotlib.pyplot.cm.gnuplot2, linewidths=0, alpha=0.2, vmin=-3., vmax=-1, label='Particle Position/Log-Likelihood')
        self.ax.legend()
        self.cb = matplotlib.pyplot.colorbar(self.sppnts, ax=self.ax, orientation='horizontal', aspect=30, pad=0.07)
        self.cb.set_label('Log-likelihood')
        self.cb.solids.set_edgecolor('face')
        self.cb.solids.set_alpha(1)
        self.row_0 = np.mean(self.rc[:, 0])
        self.col_0 = np.mean(self.rc[:, 1])
        self.re = ax.add_patch(matplotlib.patches.Rectangle((self.col_0 - self.hw_col, self.row_0 - self.hw_row), self.hw_col * 2 + 1, self.hw_row * 2 + 1, fill=False))
        ax.set_xlim(self.col_0 - 50, self.col_0 + 50)
        ax.set_ylim(self.row_0 + 50, self.row_0 - 50)

    def update_plot(self, t):
        try:
            image_index = self.image_index(t, max_seconds=0.1)
        except IndexError:
            return
        self.sppnts.remove()
        self.sppnts = self.ax.scatter(self.rc[:, 1], self.rc[:, 0], s=25, c=-self.log_like, cmap=matplotlib.pyplot.cm.gnuplot2, linewidths=0, alpha=0.2, vmin=-3., vmax=-1)
        self.row_1 = np.mean(self.rc[:, 0])
        self.col_1 = np.mean(self.rc[:, 1])
        self.re.set_bounds(self.col_1 - self.hw_col, self.row_1 - self.hw_row, 2 * self.hw_col + 1, 2 * self.hw_row + 1)
        col_offset = self.col_1 - self.col_0
        row_offset = self.row_1 - self.row_0
        self.im_plot.set_data(self.images[image_index].read())
        self.ax.set_xlim(self.col_1 - 50, self.col_1 + 50)
        self.ax.set_ylim(self.row_1 + 50, self.row_1 - 50)
        #self.im_plot.set_extent((self.le0 + col_offset, self.re0 + col_offset, self.be0 + row_offset, self.te0 + row_offset))
