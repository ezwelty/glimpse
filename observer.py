import numpy as np
import sklearn.decomposition as sde
import scipy.ndimage.filters as filts
import scipy.interpolate as si
import cv2
import helper
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Observer(object):
    """
    The observer class contains a set of images and the methods to compute
    the misfit between subsets thereof.  

    Attributes:
        images (list): list of Image objects
        times (list): list of times associated with images
        ref_template (array): a subimage which acts as a color palette to match
        pca: principal components analyzer
        sigma_pixel: average std between all images for a pixel
        hw_row,hw_col,sd_row,sd_col: halfwidths for reference and test subimages
        row_diff_0,col_diff_0: residual between prediction in pixel coords and viable index
        ref_subimage: reference subimage
        rows_predicted,cols_predicted: 
        
    """

    def __init__(self,images,times,sigma_pixel=0.3,reference_halfwidth=15,search_halfwidth=20): 
        """
        Construct the observer

        Arguments:
            images (list): list of Image objects
            times (list): list of times associated with images
            sigma_pixel: average std between all images for a pixel
            reference_halwidth,search_halfwidth: halfwidths for reference and test subimages
        """
             
        self.images = images
        self.times = times
        self.ref_hist_template=None
        self.pca = sde.PCA(n_components=1,svd_solver='arpack',whiten=True)
 
        self.sigma_pixel = sigma_pixel
        self.hw_row = reference_halfwidth
        self.hw_col = reference_halfwidth
        self.sd_row = search_halfwidth
        self.sd_col = search_halfwidth

    def link_tracker(self,tracker):
        """
        Sets some initial variables associated with the initial point
     
        Arguments:
            xyz (array): initial spatial coordinates to track
        """
        pmean,pcov = tracker._estimate()
  
        col_0,row_0 =  self.images[0].cam.project(pmean[:,[0,1,4]]).squeeze()
        self.rc = self.images[0].cam.project(tracker.particles[:,[0,1,4]])[:,::-1]
        self.log_like = np.zeros((tracker.N))
        
        self.ref_hist_template = self.get_subimage(row_0,col_0,self.sd_row,self.sd_col,0,get_ref_template=True)
        self.ref_subimage = self.get_subimage(row_0,col_0,self.hw_row,self.hw_col,0)

    def get_subimage(self,row,col,hw_row,hw_col,image_index,do_histogram_matching=True,do_pca=True,do_median_filtering=True,median_filter_size=(5,5),get_ref_template=False):
        """
        Extract a subimage given a row and column, and apply some image processing
        Arguments:
            row,col: row and column index
            hw_row,hw_col: halfwidth of subimage to extract
            image_index: which stored image to extract
            pca: apply PCA transformation to grayscale
            median_filter_size: apply highpass filter with specified bandwidth
        Returns: 
            chip: extracted subimage
        """

        row_nearest = int(round(row))
        col_nearest = int(round(col))
        
        row_hw = np.linspace(row-hw_row,row+hw_row,2*hw_row+1,endpoint=True)
        col_hw = np.linspace(col-hw_col,col+hw_col,2*hw_col+1,endpoint=True)

        chip = self.images[image_index].read()[row_nearest-hw_row-1:2+row_nearest+hw_row,col_nearest-hw_col-1:2+col_nearest+hw_col,:].copy()

        if get_ref_template:
            return chip

        rows = range(row_nearest-hw_row-1,row_nearest+hw_row+2)
        cols = range(col_nearest-hw_col-1,col_nearest+hw_col+2)
        if do_histogram_matching:
            # If a reference template has been defined, perform histogram matching.  This is 
            # very helpful for ameliorating the effects of illumination changes.
            chip[:,:,0] = helper.hist_match(chip[:,:,0],self.ref_hist_template[:,:,0])
            chip[:,:,1] = helper.hist_match(chip[:,:,1],self.ref_hist_template[:,:,1])
            chip[:,:,2] = helper.hist_match(chip[:,:,2],self.ref_hist_template[:,:,2])
        if do_pca:
            # If a pca has been defined, compute an intensity image using it.
            m,n,q = chip.shape
            Q = chip.reshape((m*n,q))
            self.pca.fit(Q)
            self.pca.components_ = np.sign(self.pca.components_[0])*self.pca.components_
            Qp = self.pca.transform(Q)
            chip = Qp.reshape((m,n))
        Ri = si.RectBivariateSpline(rows,cols,chip,kx=3,ky=3)
        chip = Ri(row_hw,col_hw,grid=True)

        if do_median_filtering:
            # Do median highpass filtering
            chip_lowpass = filts.median_filter(chip,(median_filter_size[0],median_filter_size[1]))
            chip -= chip_lowpass
        return chip

    
    def compute_likelihood(self,pmean,particles,t):
        """
        Computes the likelihood of each particle in a population based on SSE between a reference and test subimage
        Arguments:
            pmean: particle mean values
            particles: particle array
            t: current time (in Unix time)
        Returns:
            log_likelihood: log likelihood for each particle
        """
        # Check to see if there exists an image at the current time.  
        # If not, return a constant likelihood
        tdist = np.abs(t-self.times)
        image_index = np.argmin(tdist)
        time_mismatch = np.min(tdist)
        if time_mismatch>0.1:
            return 0.0

        # Extract a subimage around predicted coordinates
        current_image = self.images[image_index]
        col_1,row_1 = current_image.cam.project(pmean[:,[0,1,4]]).squeeze()
        self.test_subimage = self.get_subimage(row_1,col_1,self.sd_row,self.sd_col,image_index)                  
        # Try generating a likelihood interpolant and evaluate for each particle

        #try:
        like_interpolant = self.get_likelihood_interpolant(self.ref_subimage,self.test_subimage,row_1,col_1)

        self.rc = current_image.cam.project(particles[:,[0,1,4]])[:,::-1]

        self.log_like = like_interpolant(self.rc[:,0],self.rc[:,1],grid=False)/self.sigma_pixel**2
        # If too close to the image boundary, return a constant log_likelihood
        #except IndexError:
        #    log_like = 1.

        return self.log_like
        
    def get_likelihood_interpolant(self,ref_subimage,test_subimage,row,col):
        """
        Produce an object for evaluating the log likelihood of particles
 
        Arguments:
            ref_subimage,test_subimage:
            row_nearest,col_nearest: the row and column corresponding to the center of test_subimage
            row_diff_*: corrections for rounding to indices
        Returns:
            local_interp: A Bivariate Spline that can be used to assign
                          likelihood to particles
        """ 
        # Compute normalized correlation coefficients
        rhos = cv2.matchTemplate(test_subimage.astype('float32'),ref_subimage.astype('float32'),method=cv2.TM_SQDIFF)
        rhos/=(ref_subimage.shape[0]*ref_subimage.shape[1])

        rcoords = row - (self.sd_row-self.hw_row) + np.array(range(rhos.shape[0]))
        ccoords = col - (self.sd_col-self.hw_col) + np.array(range(rhos.shape[1]))

        # Get subpixel accuracy by fitting the correlation surface with a cubic spline and maximizing
        local_interp = si.RectBivariateSpline(rcoords,ccoords,rhos,kx=3,ky=3)

        return local_interp

    # Real time plotting utilities
    def initialize_plot(self,ax):
        self.ax = ax
        self.im_plot = ax.imshow(self.images[0].I,interpolation='none')
        self.le0,self.re0,self.be0,self.te0 = self.im_plot.get_extent()

        #self.spprd = ax.scatter(self.cols_predicted,self.rows_predicted,s=50,c='green',label='Prior Prediction')
        self.sppnts = self.ax.scatter(self.rc[:,1],self.rc[:,0],s=25,c=-self.log_like,cmap=plt.cm.gnuplot2,linewidths=0,alpha=0.2,vmin=-3.,vmax=-1,label='Particle Position/Log-Likelihood')
        self.ax.legend()

        self.cb = plt.colorbar(self.sppnts,ax=self.ax,orientation='horizontal',aspect=30,pad=0.07)
        self.cb.set_label('Log-likelihood')
        self.cb.solids.set_edgecolor("face")
        self.cb.solids.set_alpha(1)

        self.row_0 = np.mean(self.rc[:,0])
        self.col_0 = np.mean(self.rc[:,1])

        self.re = ax.add_patch(patches.Rectangle((self.col_0-self.hw_col,self.row_0-self.hw_row),self.hw_col*2+1,self.hw_row*2+1,fill=False))
        ax.set_xlim(self.col_0-50,self.col_0+50)
        ax.set_ylim(self.row_0+50,self.row_0-50)

    def update_plot(self,t):
        tdist = np.abs(t-self.times)
        closest_time = np.argmin(tdist)
        time_mismatch = np.min(tdist)
        if time_mismatch<0.01:

            self.sppnts.remove()

            self.sppnts = self.ax.scatter(self.rc[:,1],self.rc[:,0],s=25,c=-self.log_like,cmap=plt.cm.gnuplot2,linewidths=0,alpha=0.2,vmin=-3.,vmax=-1)

            self.row_1 = np.mean(self.rc[:,0])
            self.col_1 = np.mean(self.rc[:,1])
        
        
            self.re.set_bounds(self.col_1-self.hw_col,self.row_1-self.hw_row,2*self.hw_col+1,2*self.hw_row+1)
            col_offset = self.col_1-self.col_0
            row_offset = self.row_1-self.row_0
            self.im_plot.set_data(self.images[closest_time].I)
            self.ax.set_xlim(self.col_1-50,self.col_1+50)
            self.ax.set_ylim(self.row_1+50,self.row_1-50)
            #self.im_plot.set_extent((self.le0+col_offset,self.re0+col_offset,self.be0+row_offset,self.te0+row_offset))



    
            
    
 
 

        
