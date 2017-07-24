import numpy as np
import time
import datetime
import cPickle
import gzip
import os
import gdal
import sys
import cv2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.interpolate as si
import scipy.optimize as so
import scipy.misc as sm
import scipy.spatial as ssp
import scipy.ndimage.filters as filts 
import scipy.ndimage as snd
import skimage.feature as skif
import filterpy.kalman as fpk
import sklearn.decomposition as sde
import matplotlib.patches as patches

import multiprocess as mp
import camera
import sharedmem

from fractions import gcd

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

class ParticleTracker(object):
    """
    Implements a constant velocity model particle filter
    """    
    def __init__(self):
        pass

    def set_times(self,times):
        # Times need to be set such that all of the times at which observations occur are present.
        # See driver for example code of how to do this.
        self.times = times
        self.dt = np.hstack((0,np.diff(self.times)))

    def set_observers(self,observers):
        # Set observer objects, which compute particle likelihoods.  The code can include an arbitrary number of observers.
        self.observers = observers

    def set_elevation_model(self,x,y,Z,filter_crevasses=True,max_filter_kernel_size=5,gauss_filter_kernel_size=5,complex_crevasse_filtering=False):
        # Set the DEM, smooth, and interpolate.
        self.x = x
        self.y = y
        self.X,self.Y = np.meshgrid(self.x,self.y)
        self.Z = Z
        if filter_crevasses:
            if complex_crevasse_filtering:
                # Find local maxima, and fit a surface between them. Slow.
                Z_maximumed = filts.maximum_filter(self.Z,size=max_filter_kernel_size)
                maximum_mask = (Z_maximumed==self.Z).ravel()
                Xmax = self.X.ravel()[maximum_mask]
                Ymax = self.Y.ravel()[maximum_mask]
                Zmax = self.Z.ravel()[maximum_mask]
                maxint = si.LinearNDInterpolator(np.vstack((Xmax,Ymax)).T,Zmax)
                Zf = maxint(self.X.ravel(),self.Y.ravel()).reshape(self.X.shape)
                self.Z_filtered = filts.gaussian_filter(Zf,gauss_filter_kernel_size)
            else:
                # Apply a maximum filter, then perform Gaussian smoothing. Fast.
                self.Z_filtered = filts.gaussian_filter(filts.maximum_filter(self.Z,size=max_filter_kernel_size),gauss_filter_kernel_size)
        else:
            self.Z_filtered = self.Z
        # Interpolate using a 3rd order spline
        self.z_interpolant = si.RectBivariateSpline(self.x,self.y[::-1],np.nan_to_num(self.Z_filtered[::-1,:]).T,kx=3,ky=3)

    def predict(self,particles,dt,wx,wy,wz):
        # Predict the updated particle state.
        # For the horizontal dimensions, we use a constant velocity model subject to Gaussian random accelerations.
        # For the vertical dimension, we include a systematic error term that undergoes small perturbations
        # representing random error.
        N = len(particles)
        ax = wx*np.random.randn(N)  # x acceleration
        ay = wy*np.random.randn(N)  # y acceleration
        particles[:,0] += dt*particles[:,2] + 0.5*ax*dt**2  # 2nd order update
        particles[:,1] += dt*particles[:,3] + 0.5*ay*dt**2 
        particles[:,2] += dt*ax*np.random.randn(N) 
        particles[:,3] += dt*ay*np.random.randn(N)

        # Z is DEM value plus systematic plus random errors
        particles[:,4] = self.z_interpolant(particles[:,0],particles[:,1],grid=False) + particles[:,5]
        #particles[:,5] = dt*np.random.randn(N)*wz*np.sqrt(particles[:,2]**2 + particles[:,3]**2)                               

    @staticmethod
    def update(particles,weights,log_likelihoods):
        # Update the particle filter weights based on log likelihoods from observers
        weights.fill(1.)
        weights *= np.exp(-sum(log_likelihoods))
        weights += 1e-300
        weights /= weights.sum()
     
    @staticmethod
    def estimate(particles,weights):
        # return weighted means and covariances
        mean = np.average(particles,weights=weights,axis=0)
        cov = np.cov(particles.T,aweights=weights)
        return mean,cov

    @staticmethod
    def systematic_resample(weights):
        # implement systematic resampling of particles (kills unlikely particles, and reproduces likely ones)
        N = len(weights)

        # make N subdivisions, choose positions 
        # with a consistent random offset
        positions = (np.arange(N) + np.random.random()) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    @staticmethod    
    def resample_from_index(particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        weights /= np.sum(weights)
    #profile
    def track(self,x,y,N=1000,do_plot=True,sigma_x0=1.0,sigma_y0=1.0,sigma_vx0=10.0,sigma_vy0=10.0,sigma_z0=10.0,wx=5.0,wy=5.0,wz=0.3,n_iterations=1):

        x_index = np.argmin(np.abs(self.x - x ))
        y_index = np.argmin(np.abs(self.y[::-1] - y ))

        self.z_interpolant = si.RectBivariateSpline(self.x[x_index-50:x_index+50],self.y[::-1][y_index-50:y_index+50],(np.nan_to_num(self.Z_filtered[::-1,:]).T)[x_index-50:x_index+50,y_index-50:y_index+50],kx=3,ky=3)

        # Initialize particles according to initial guesses of uncertainty values.  
        self.particles = np.zeros((N,6))

        self.particles[:,0] = x + sigma_x0*np.random.randn(N)
        self.particles[:,1] = y + sigma_y0*np.random.randn(N)
        self.particles[:,2] = sigma_vx0*np.random.randn(N)
        self.particles[:,3] = sigma_vy0*np.random.randn(N)
        self.particles[:,5] = sigma_z0*np.random.randn(N)
        self.particles[:,4] = self.z_interpolant(self.particles[:,0],self.particles[:,1],grid=False) + self.particles[:,5]

        self.weights = np.ones(N)
        self.weights /= N

        # Get mean of estimated state
        p,c = self.estimate(self.particles,self.weights)
 
        tic = time.time()
        for j in range(n_iterations):
        # We run the observation procedure several times, starting with refined velocity estimates as an initial guess.
            for observer in self.observers:
                # Initialize observers based on initial point
                observer.initialize(p[0],p[1],p[4])
            self.particles[:,0] = x + sigma_x0*np.random.randn(N)
            self.particles[:,1] = y + sigma_y0*np.random.randn(N)
            self.particles[:,4] = self.z_interpolant(self.particles[:,0],self.particles[:,1],grid=False) + self.particles[:,5] 
            # Log the mean and covariance at each step
            mean,cov = self.estimate(self.particles,self.weights)
            self.means = [mean]
            self.covs = [cov]
            
            if do_plot:
                self.initialize_plot()

            for t,dt in zip(self.times[1:],self.dt[1:]):
                # Track over each time past the initial image

                # Predict next step
                self.predict(self.particles,dt,wx,wy,wz)
                pmean,cov = self.estimate(self.particles,self.weights)

                # Compute likelihoods from image tracking
                likelihoods = [observer.compute_likelihood(pmean,self.particles,t) for observer in self.observers]

                # Update particle weights 
                self.update(self.particles,self.weights,likelihoods)

                # Resample particle weights
                indexes = self.systematic_resample(self.weights)
                self.resample_from_index(self.particles, self.weights, indexes) 

                new_mean,new_cov = self.estimate(self.particles,self.weights)
                self.means.append(new_mean)
                self.covs.append(new_cov)
                if do_plot:
                    self.update_plot(t)

    def initialize_plot(self):
        # Perform plotting animation.  Don't use this with multiprocessing!
        x = self.particles[:,0].mean()
        y = self.particles[:,1].mean()
        z = self.particles[:,4].mean()
        plt.ion()
        number_of_plots = 1 + len(self.observers)
        fig,self.ax = plt.subplots(nrows=1,ncols=number_of_plots,figsize=(10*number_of_plots,10))
        fig.tight_layout()
        self.ax[0].contourf(self.X,self.Y,self.Z_filtered,np.linspace(z-50,z+50,41),cmap=plt.cm.gray)
        self.meplot = self.ax[0].scatter(self.means[0][0],self.means[0][1],c='red',s=50)
        self.pa_plot = self.ax[0].scatter(self.particles[:,0],self.particles[:,1],s=10,c=np.sqrt(self.particles[:,2]**2+self.particles[:,3]**2),cmap=plt.cm.gnuplot2,vmin=0,vmax=20,alpha=0.5,linewidths=0)
        self.ax[0].axis('equal')
        self.ax[0].set_xlim(x-200,x+200)
        self.ax[0].set_ylim(y-200,y+200)

        for a,o in zip(self.ax[1:],self.observers):
            o.initialize_plot(a)

    def update_plot(self,t):
        self.meplot.remove()
        self.pa_plot.remove()
        self.meplot = self.ax[0].scatter([m[0] for m in self.means],[m[1] for m in self.means],s=50,c='red')
        self.pa_plot = self.ax[0].scatter(self.particles[:,0],self.particles[:,1],s=10,c=np.sqrt(self.particles[:,2]**2+self.particles[:,3]**2),cmap=plt.cm.gnuplot2,vmin=0,vmax=20,alpha=0.5,linewidths=0)
    
        for a,o in zip(self.ax[1:],self.observers):
            o.update_plot(t)
        plt.pause(0.1)


class Observer(object):
    """  
    Observer class acts as a camera model and more!  Has several functions:  First, it computes perturbations in image 
    coordinates from wind/thermal expansion, etc.  Second it finds the normalized cross correlation peak (and stats 
    related to the uniqueness of that peak).  Finally it computes the (log) likelihood function for each particle 
    based on this observation.  
    """
    def __init__(self,cam,times,images,sigma_1=50.,sigma_0=0.3,B=0.3,outlier_tol_pixels=3):
        # Camera model is precomputed (see optimize_camera.py)
        self.cam = cam
        self.images = images
        self.times = times
    
        # This is used for histogram equalization of image chips.
        self.ref_template = None

        # This is used to find the grayscale Z-normalization of each image chip.
        self.pca = sde.PCA(n_components=1,svd_solver='arpack',whiten=True)

        # These are parameters controlling the functional relationship between pixel accuracy and delcorr/std
        self.sigma_1 = sigma_1    # Pixel accuracy (e.g. std deviation) when correlation is perfect
        self.sigma_0 = sigma_0    # Pixel accuracy when delcorr/std is 1.
        self.B = B                # Shape parameter
        self.outlier_tol_pixels = outlier_tol_pixels   # Pixel distance at which peak is disregarded as an outlier

    def initialize(self,x,y,z,hw_row=15,hw_col=15,sd_row=20,sd_col=20):
        # Initialize image-space arrays 
        row_0,col_0 = self.observe(np.array(x),np.array(y),np.array(z))[0]

        self.rows_predicted = [row_0]
        self.cols_predicted = [col_0]
        self.rows_observed = [row_0]
        self.cols_observed = [col_0]
        self.rows_updated = [row_0]
        self.cols_updated = [col_0]
        self.observation_strength = [1.0]

        row_nearest = int(np.rint(row_0))
        col_nearest = int(np.rint(col_0))
        self.row_diff_0 = row_nearest-row_0
        self.col_diff_0 = col_nearest-col_0

        self.hw_row = hw_row
        self.hw_col = hw_col
        self.sd_row = sd_row
        self.sd_col = sd_col

        # Set reference template
        self.ref_template = self.images[0][row_nearest-sd_row:1+row_nearest+sd_row,col_nearest-sd_col:1+col_nearest+sd_col] 

        # Set reference sub-image (chip) based on first image  
        self.ref_chip = self.get_chip(row_nearest,col_nearest,hw_row,hw_col,0)

    def observe(self,x,y,z):
        # Project from spatial to image coordinates
        uv,inframe = self.cam.project(np.vstack((x,y,z)).T)
        return uv[:,::-1]
    #profile
    def compute_likelihood(self,pmean,particles,t):
        # This is where the heavy lifting occurs.
        # First, determine whether the current time step is associated with an image
        tdist = np.abs(t-self.times)
        closest_time = np.argmin(tdist)
        time_mismatch = np.min(tdist)
        rcmean = self.observe(pmean[0],pmean[1],pmean[4]) 
        row_prediction = rcmean[0][0]
        col_prediction = rcmean[0][1] 
        self.rows_predicted.append(row_prediction)
        self.cols_predicted.append(col_prediction)
        if time_mismatch>0.0001:
        # If not, return a constant likelihood (aka doesn't change particle weights)
            return 0.0
        
        # Compute the correction for camera motion
        row_offset = self.row_predict(self.rows_predicted[0],self.cols_predicted[0],self.correction_parameters[closest_time])-self.rows_predicted[0]
        col_offset = self.col_predict(self.rows_predicted[0],self.cols_predicted[0],self.correction_parameters[closest_time])-self.cols_predicted[0]

        # retrieve the current image
        new_img_index = closest_time
        try:
            # Find the correlation peak (might change this to something more probabilistic)
            like_interpolant,row_n0,col_n0,rho_opt = self.get_best_offset(row_prediction,col_prediction,self.row_diff_0,self.col_diff_0,self.ref_chip,new_img_index,row_offset,col_offset)

        #

            # Get the particles in image space
            rc = self.observe(particles[:,0],particles[:,1],particles[:,4])

        # Weighted least squares log likelihood
        #r_dist = row_n - rc[:,0] 
        #c_dist = col_n - rc[:,1]
        #log_like = (r_dist**2 + c_dist**2)/(meas_err**2)
            log_like = like_interpolant(rc[:,0],rc[:,1],grid=False)/0.3**2
            self.rows_observed.append(row_n0)
            self.cols_observed.append(col_n0)
            self.observation_strength.append(rho_opt)
        except IndexError:
            log_like = 1.

        return log_like
        
    @staticmethod
    def generate_ref_chips(corrpoints,hw,ref_image,median_filter_size=(5,5)):
        # Generate the reference subimages for tracking stationary targets.  See get_chip() for img processing steps
        pca = sde.PCA(n_components=1,svd_solver='arpack',whiten=True)
        ref_chips = []
        for corrpoint in corrpoints:
            row,col = corrpoint
            ref_chip = ref_image[row-hw:1+row+hw,col-hw:1+col+hw].copy()
            if ref_chip.ndim==3:
                m,n,q = ref_chip.shape
                Q = ref_chip.reshape((m*n),q)
                pca.fit(Q)
                pca.components_ = np.sign(pca.components_[0])*pca.components_
                Qp = pca.transform(Q)
                ref_chip = Qp.reshape((m,n))

            ref_chip_lp = filts.median_filter(ref_chip,median_filter_size)
            ref_chip -= ref_chip_lp
            ref_chips.append([corrpoint,ref_chip])

        return ref_chips
    #profile
    def get_chip(self,row,col,hw_row,hw_col,image_index,pca=True,median_filter_size=(5,5)):
        # Extract the subimage
        chip = self.images[image_index][row-hw_row:1+row+hw_row,col-hw_col:1+col+hw_col].copy()
        if self.ref_template is not None:
            # If a reference template has been defined, perform histogram matching.  This is 
            # very helpful for ameliorating the effects of illumination changes.
            chip[:,:,0] = hist_match(chip[:,:,0],self.ref_template[:,:,0])
            chip[:,:,1] = hist_match(chip[:,:,1],self.ref_template[:,:,1])
            chip[:,:,2] = hist_match(chip[:,:,2],self.ref_template[:,:,2])
        if pca:
            # If a pca has been defined, compute an intensity image using it.
            m,n,q = chip.shape
            Q = chip.reshape((m*n,q))
            self.pca.fit(Q)
            self.pca.components_ = np.sign(self.pca.components_[0])*self.pca.components_
            Qp = self.pca.transform(Q)
            chip = Qp.reshape((m,n))

        # Do median highpass filtering
        chip_lowpass = filts.median_filter(chip,(median_filter_size[0],median_filter_size[1]))
        chip -= chip_lowpass
        return chip
    
    def get_best_offset(self,row_prediction,col_prediction,row_diff_0,col_diff_0,ref_chip,test_image_index,row_offset,col_offset,median_filter_window=(5,5)):
        # Find the best image offset estimate in pixel space

        # Round the prediction to the nearest integer
        row_nearest = int(np.rint(row_prediction))
        col_nearest = int(np.rint(col_prediction))
                    
        # Extract a sub-image
        test_chip = self.get_chip(row_nearest,col_nearest,self.sd_row,self.sd_col,test_image_index)
       
        # Compute normalized correlation coefficients
        rhos = cv2.matchTemplate(test_chip.astype('float32'),ref_chip.astype('float32'),method=cv2.TM_SQDIFF)
        rhos/=(ref_chip.shape[0]*ref_chip.shape[1])
        #rhos = skif.match_template(test_chip,ref_chip)

        # Find optimal value
        row_n0,col_n0 = np.unravel_index(rhos.argmin(),rhos.shape)
        rho_opt = rhos[row_n0,col_n0]
                
        rcoords = row_nearest - (self.sd_row-self.hw_row) + np.array(range(rhos.shape[0])) - row_offset - row_diff_0
        ccoords = col_nearest - (self.sd_col-self.hw_col) + np.array(range(rhos.shape[1])) - col_offset - col_diff_0
        row_n0 += row_nearest - (self.sd_row-self.hw_row) - row_offset - row_diff_0
        col_n0 += col_nearest - (self.sd_col-self.hw_col) - col_offset - col_diff_0

        # Get subpixel accuracy by fitting the correlation surface with a cubic spline and maximizing
        local_interp = si.RectBivariateSpline(rcoords,ccoords,rhos,kx=3,ky=3)

        return local_interp,row_n0,col_n0,rho_opt

    def track_corrections(self,ref_chips,sd=50,outlier_tol=5,correction_model='rotation_translation',plot=False):
        # Track ostensibly non-mobile points for the purpose of correcting camera motion
        # Takes as an argument ref_chips object, which is a list of tuples, each containing
        # the point and the subimage to correlate (this way so that you can provide not the
        # first image in the sequence if so desired.) 
        pca = sde.PCA(n_components=1,svd_solver='arpack',whiten=True)
        self.correction_model = correction_model
        self.ref_chips = ref_chips
        corrpoint_list = []
        results = []

        #loop over each point
        for corrpoints,ref_chip in ref_chips:
            hw = (ref_chip.shape[0]-1)/2
            row,col = corrpoints
            corrpoint_list.append(corrpoints)
            
            rowlist = []
            collist = []
            rowlist_float = []
            collist_float = []
            corrs = []
            delcorrs = []
            #loop over all the images
            for i in range(0,len(self.images)):
                if self.images[i] is not None:
                    img_1 = self.images[i].copy()
                    test_chip = self.get_chip(row,col,sd,sd,i)
                    
                    # Find the best offset and save relevant statistics
                    rhos = skif.match_template(test_chip,ref_chip)
                    row_n0,col_n0 = np.unravel_index(rhos.argmax(),rhos.shape)

                    local_interp = si.RectBivariateSpline(range(rhos.shape[0]),range(rhos.shape[1]),rhos)
                    xopt = so.fmin_cg(lambda x:-local_interp(x[0],x[1])[0][0],(row_n0,col_n0),fprime=lambda x:-np.array([local_interp(x[0],x[1],dx=1)[0][0],local_interp(x[0],x[1],dy=1)[0][0]]),disp=0)
                    rho_opt = max(local_interp(xopt[0],xopt[1])[0][0],1e-3)

                    rho_maxfilt = filts.maximum_filter(rhos,size=(5,5))
                    peaks = rho_maxfilt==rhos
                    rho_peaks = np.sort(rhos.ravel()[peaks.ravel()])
                    if len(rho_peaks)>1:
                        rho_second = rho_peaks[-2]
                    else:
                        rho_second = 0
                    delta_rho_i = rho_opt - rho_second
                    std_rho_i = rhos.std()

                    row_n = row - (sd-hw) + xopt[0]# - row_avg[i]
                    col_n = col - (sd-hw) + xopt[1]# - col_avg[i]
             
                    row_b = int(np.rint(row_n))
                    col_b = int(np.rint(col_n))

                    corrs.append(rho_opt)
                    delcorrs.append(delta_rho_i/std_rho_i)
                    rowlist.append(row_b)
                    collist.append(col_b)
                    rowlist_float.append(row_n)
                    collist_float.append(col_n)
                else:
                    corrs.append(0)
                    delcorrs.append(0)
                    rowlist.append(0)
                    collist.append(0)
                    rowlist_float.append(0)
                    collist_float.append(0)

            results.append([corrs,rowlist,collist,rowlist_float,collist_float,delcorrs])

        corrpoint_array = np.array(corrpoint_list)

        results_array = np.array(results)
        corrs = results_array[:,0,:].squeeze()
        delcorrs = results_array[:,5,:].squeeze()
        rows = results_array[:,3,:].squeeze()
        cols = results_array[:,4,:].squeeze()
        r0 = corrpoint_array[:,0]
        c0 = corrpoint_array[:,1]
        times = range(len(rows[0]))

        ps = []
        po = []

        # Objective function for the fit (minimize the difference between predicted and observed image coordinates based on
        # an arbitrary model

        #def F(p,rows_obs,cols_obs,rows_known,cols_known,weights):
         #   rows_predicted = np.array([self.row_predict(row_known,col_known,p) for row_known,col_known in zip(rows_known,cols_known)])
         #   cols_predicted = np.array([self.col_predict(row_known,col_known,p) for row_known,col_known in zip(rows_known,cols_known)])
         #   outliers = ((np.abs(rows_known - rows_obs)<outlier_tol) * (np.abs(cols_known-cols_obs)<outlier_tol)).astype(float)
         #   return np.sum(outliers*weights*np.abs(rows_predicted - rows_obs)**2 + outliers*weights*np.abs(cols_predicted - cols_obs)**2)/(outliers*weights).sum()

        def F(p,rows_obs,cols_obs,rows_known,cols_known,weights):
            gamma = 0.0001
            rows_obs = rows_obs.copy()
            cols_obs = cols_obs.copy()
            rows_known = rows_known.copy()
            cols_known = cols_known.copy()
            weights = weights.copy()
            rows_predicted = np.array([self.row_predict(row_known,col_known,p) for row_known,col_known in zip(rows_known,cols_known)])
            cols_predicted = np.array([self.col_predict(row_known,col_known,p) for row_known,col_known in zip(rows_known,cols_known)])
            inliers = ((np.abs(rows_known - rows_obs)<outlier_tol) * (np.abs(cols_known-cols_obs)<outlier_tol))#.astype(float)
            outliers = np.invert(inliers)
            rows_obs[outliers] = rows_known[outliers]
            cols_obs[outliers] = cols_known[outliers]
            weights[outliers] = gamma 
            I = np.sum(weights*np.abs(rows_predicted - rows_obs)**2 + weights*np.abs(cols_predicted - cols_obs)**2)/np.sum(weights)
            return I

        # Correction models include rotation_translation (the best choice), pure translation, and an affine transform
        if self.correction_model=='rotation_translation':
            x0 = np.array([0,0,0])
        elif self.correction_model=='translation':
            x0 = np.array([0,0])
        elif self.correction_model=='affine':
            x0 = np.array([0,0,0,0,0,0])
        else:
            x0 = np.array([])

        # For each time step, fit the model
        for i in range(rows.shape[1]):
            if self.images[i] is not None:
                rows_observed = rows[:,i]
                cols_observed = cols[:,i]
                rows_known = r0
                cols_known = c0
                weights = delcorrs[:,i]

                popt = so.minimize(F,x0,args=(rows_observed,cols_observed,rows_known,cols_known,weights),method='powell',tol=1e-6)
                ps.append(popt['x'])
                po.append(np.sqrt(popt['fun']))
            else:
                ps.append(None)
                po.append(None)    

        if plot:
            for i in range(rows.shape[0]):
                fig,axs = plt.subplots(nrows=2,ncols=1)
                axs[0].plot(times,rows[i] - r0[i])
                axs[0].scatter(times,rows[i] - r0[i],s=50,c=delcorrs[i])
                axs[1].plot(times,cols[i] - c0[i])
                axs[1].scatter(times,cols[i] - c0[i],s=50,c=delcorrs[i])
                rows_predict = np.array([self.row_predict(r0[i],c0[i],p) for p in ps])
                cols_predict = np.array([self.col_predict(r0[i],c0[i],p) for p in ps])
                axs[0].plot(times,rows_predict - r0[i],'--',lw=4)
                axs[1].plot(times,cols_predict - c0[i],'--',lw=4)
                axs[0].set_ylim(-5,5)
                axs[1].set_ylim(-5,5)

            fig,axs = plt.subplots()
            axs.imshow(self.images[0],interpolation='none')
            axs.plot(self.col_predict(2104,3327,ps[0]),self.row_predict(2104,3327,ps[0]),'ro')
                
            fig,axs = plt.subplots()
            axs.imshow(self.images[10],interpolation='none')
            axs.plot(self.col_predict(2104,3327,ps[10]),self.row_predict(2104,3327,ps[10]),'ro')
            
            fig,axs = plt.subplots()
            axs.imshow(self.images[20],interpolation='none')
            axs.plot(self.col_predict(2104,3327,ps[20]),self.row_predict(2104,3327,ps[20]),'ro')

        self.correction_parameters = ps
        self.correction_strength = po

    # Predict rows and columns based on a camera motion model and parameters.
    def row_predict(self,row,col,p):
        if self.correction_model == 'rotation_translation':
            rmid = self.images[0].shape[0]
            cmid = self.images[0].shape[1]
            return np.cos(p[0])*(row-rmid) - np.sin(p[0])*(col-cmid) + rmid + p[1]
        elif self.correction_model == 'translation':
            return row + p[0]
        elif self.correction_model == 'affine':
            return row + p[0] + p[1]*row + p[2]*col
        else:
            return row

    def col_predict(self,row,col,p):
        if self.correction_model == 'rotation_translation':
            rmid = self.images[0].shape[0]
            cmid = self.images[0].shape[1]
            return np.sin(p[0])*(row-rmid) + np.cos(p[0])*(col-cmid) + cmid + p[2]
        elif self.correction_model == 'translation':
            return col + p[1]
        elif self.correction_model == 'affine':
            return col + p[3] + p[4]*row + p[5]*col
        else:
            return col

    # Real time plotting utilities
    def initialize_plot(self,ax):
        self.ax = ax
        self.im_plot = ax.imshow(self.images[0],interpolation='none')
        self.le0,self.re0,self.be0,self.te0 = self.im_plot.get_extent()

        self.spobs = ax.scatter(self.cols_observed,self.rows_observed,s=50,c=self.observation_strength,vmin=0,vmax=1,cmap=plt.cm.inferno,marker='x')
        #self.spupd = ax.scatter(self.cols_updated,self.rows_updated,s=50,c='red')
        self.spprd = ax.scatter(self.cols_predicted,self.rows_predicted,s=50,c='green')

        col_0 = self.cols_predicted[0]
        row_0 = self.rows_predicted[0]

        self.re = ax.add_patch(patches.Rectangle((col_0-self.hw_col,row_0-self.hw_row),self.hw_col*2+1,self.hw_row*2+1,fill=False))
        ax.set_xlim(col_0-50,col_0+50)
        ax.set_ylim(row_0+50,row_0-50)

    def update_plot(self,t):
        tdist = np.abs(t-self.times)
        closest_time = np.argmin(tdist)
        time_mismatch = np.min(tdist)
        if time_mismatch<0.01:
            self.im_plot.set_data(self.images[closest_time])
            row_offset = self.row_predict(self.rows_predicted[-1],self.cols_predicted[-1],self.correction_parameters[closest_time])-self.rows_predicted[-1]
            col_offset = self.col_predict(self.rows_predicted[-1],self.cols_predicted[-1],self.correction_parameters[closest_time])-self.cols_predicted[-1]
            self.im_plot.set_extent((self.le0-col_offset,self.re0-col_offset,self.be0-row_offset,self.te0-row_offset))

        self.spobs.remove()
        self.spprd.remove()
        #self.spupd.remove()

        self.spobs = self.ax.scatter(self.cols_observed,self.rows_observed,s=50,c=self.observation_strength,vmin=0,vmax=2,cmap=plt.cm.inferno,marker='x')
        #self.spupd = self.ax.scatter(self.cols_updated,self.rows_updated,s=50,c='red')
        self.spprd = self.ax.scatter(self.cols_predicted,self.rows_predicted,s=50,c='green')
        
        self.re.set_bounds(self.cols_predicted[-1]-self.hw_col,self.rows_predicted[-1]-self.hw_row,2*self.hw_col+1,2*self.hw_row+1)




if __name__=='__main__':

    year_start = int(sys.argv[1])
    month_start = int(sys.argv[2])
    day_start = int(sys.argv[3])
    year_end = int(sys.argv[4])
    month_end = int(sys.argv[5])
    day_end = int(sys.argv[6])
   
    start_time_dt = datetime.datetime(year_start,month_start,day_start,20,0,0)
    end_time_dt = datetime.datetime(year_end,month_end,day_end,20,0,0)
    start_time = time.mktime(start_time_dt.timetuple())
    end_time = time.mktime(end_time_dt.timetuple())
    #start_time = time.mktime(datetime.datetime(2013,9,16,0,0,0).timetuple())
    #end_time = time.mktime(datetime.datetime(2013,9,19,0,0,0).timetuple())

    simulation_directory = 'dual_camera_2013/'

    from itertools import islice#, izip as zip # if Python 2.x

    def nearest_neighbours(x, lst):
        if x <= lst[0]:
            return 0
        elif x >= lst[-1]:
            return len(lst)-1
        else:
            for i,y in enumerate(lst[:-1]):
                if y <= x <= lst[i+1]:
                    return i,i+1

    def get_cropped_dem(path,xmin,xmax,ymin,ymax):
        dem = gdal.Open(path)
        Z = dem.ReadAsArray()
        Z[Z<-10000] = np.nan
        geotransform = dem.GetGeoTransform()
    
        # Define map coordinates
        originX = geotransform[0]
        originY = geotransform[3]
        pixelX = geotransform[1]
        pixelY = geotransform[5]

        x = originX + pixelX*np.array(range(Z.shape[1]))
        y = originY + pixelY*np.array(range(Z.shape[0]))

        # Downsample DEM with rate n_skip
        n_skip = 1

        x = x[::n_skip]
        y = y[::n_skip]
        Z = Z[::n_skip,::n_skip]
        x_index = (x>xmin)*(x<xmax)
        y_index = (y>ymin)*(y<ymax)
        x = x[x_index]
        y = y[y_index]
        Z = Z[y_index,:][:,x_index]

        return x,y,Z
            
    dem_list = os.listdir(simulation_directory+'dem')
    dem_list = [n for n in dem_list if 'geo' in n]
    dem_list.sort()
    dem_dates = [time.mktime(datetime.datetime(int(n[8:12]),int(n[12:14]),int(n[14:16]),0,0,0).timetuple()) for n in dem_list]

    dem_indices = nearest_neighbours(start_time,dem_dates)

    xmin = 492000
    xmax = 501500
    ymin = 6.777e6
    ymax = 6.781e6+5000

    if np.isscalar(dem_indices):
        x,y,Z = get_cropped_dem(simulation_directory+'dem/'+dem_list[dem_indices],xmin,xmax,ymin,ymax)
    else:
        x0,y0,Z0 = get_cropped_dem(simulation_directory+'dem/'+dem_list[dem_indices[0]],xmin,xmax,ymin,ymax)
        x1,y1,Z1 = get_cropped_dem(simulation_directory+'dem/'+dem_list[dem_indices[1]],xmin,xmax,ymin,ymax)
        t_interval = dem_dates[dem_indices[1]] - dem_dates[dem_indices[0]]
        d0 = start_time - dem_dates[dem_indices[0]]
        d1 = dem_dates[dem_indices[1]] - start_time
        x = x0*d1/t_interval + x1*d0/t_interval
        y = y0*d1/t_interval + y1*d0/t_interval
        Z = Z0*d1/t_interval + Z1*d0/t_interval
    
    def get_image_times(img_dir):
        img_names = np.sort(os.listdir(img_dir))
        img_names.sort()
        datetimes = [datetime.datetime(int(n[6:10]),int(n[10:12]),int(n[12:14]),int(n[15:17]),int(n[17:19]),int(n[19:21])) for n in img_names]
        times = np.array([time.mktime(dd.timetuple()) for dd in datetimes])#/(60**2*24)
        return img_names,times
        
    cam_0 = cPickle.load(open(simulation_directory+'camera_model/cam_AK01.p'))
    cam_1 = cPickle.load(open(simulation_directory+'camera_model/cam_AK10.p'))
    img_dir_0 = simulation_directory+'images/AK01/unprocessed/'
    img_dir_1 = simulation_directory+'images/AK10/unprocessed/'
    #img_dir_0 = simulation_directory+'images/unprocessed/AK01/'
    #img_dir_1 = simulation_directory+'images/unprocessed/AK10/'
    names_0,times_0 = get_image_times(img_dir_0)
    names_1,times_1 = get_image_times(img_dir_1)
    
    indices_0 = np.array([t>=start_time and t<=end_time for t in times_0])
    indices_1 = np.array([t>=start_time and t<=end_time for t in times_1])

    images_0 = [sm.imread(img_dir_0+img_name) for img_name,ii in zip(names_0,indices_0) if ii]
    imgs_0 = []
    for i in images_0:
        arr = sharedmem.empty(i.shape,dtype='uint8')
        arr[:] = i
        imgs_0.append(arr)

    times_0 = times_0[indices_0]/(60**2*24)
    
    images_1 = [sm.imread(img_dir_1+img_name) for img_name,ii in zip(names_1,indices_1) if ii]
    imgs_1 = []
    for i in images_1:
        arr = sharedmem.empty(i.shape,dtype='uint8')
        arr[:] = i
        imgs_1.append(arr)
    times_1 = times_1[indices_1]/(60**2*24)

    times = np.sort(np.hstack((times_0,times_1)))

    xin,yin = cPickle.load(open(simulation_directory+'initial_points/xy_start.p'))
    xys = zip(xin,yin)
    #xys = [np.array([498431,6.78293e6])]
    #xys = [np.array([498766,6.78294e6])]
    #xys = [np.array([498800,6.78186e6])]
    #xys = [np.array([499109,6.78208e6])]

    corrpoints_0 = [[1736,118],[759,374],[797,55],[715,554],[668,674],[663,1441],[841,1782],[647,2384],[860,2814],[848,1804]]
    corrpoints_1 = [[2276,3207],[999,3512],[2490,2343],[1086,3701],[903,2527],[888,1790],[877,977],[844,512],[875,65]]
    ref_image_0 = imgs_0[0]
    ref_image_1 = imgs_1[0]

    tracker = ParticleTracker()
    tracker.set_elevation_model(x,y,Z,filter_crevasses=True,max_filter_kernel_size=5,gauss_filter_kernel_size=5,complex_crevasse_filtering=False)

    observer_0 = Observer(cam_0,times_0,imgs_0,outlier_tol_pixels=2.5)
    observer_1 = Observer(cam_1,times_1,imgs_1,outlier_tol_pixels=5.0)

    tracker.set_observers([observer_0,observer_1])
    tracker.set_times(times)

    ref_chips_0 = observer_0.generate_ref_chips(corrpoints_0,25,observer_0.images[0])
    ref_chips_1 = observer_1.generate_ref_chips(corrpoints_1,25,observer_1.images[0])

    observer_0.track_corrections(ref_chips_0,correction_model='rotation_translation',outlier_tol=10,plot=False)
    observer_1.track_corrections(ref_chips_1,correction_model='rotation_translation',outlier_tol=10,plot=False)
    
    pixel_buffer = 100

    row_min_0 = pixel_buffer
    col_min_0 = pixel_buffer
    row_max_0 = imgs_0[0].shape[0]-pixel_buffer
    col_max_0 = imgs_0[0].shape[1]-pixel_buffer
   
    row_min_1 = pixel_buffer
    col_min_1 = pixel_buffer
    row_max_1 = imgs_1[0].shape[0]-pixel_buffer
    col_max_1 = imgs_1[0].shape[1]-pixel_buffer

    results = []
    xys_good = []

    def find_occluded(x,y,x_cam,y_cam,z_cam,tracker,n_samples):
        z = tracker.z_interpolant(x,y)[0][0]
        xs = np.linspace(x_cam,x,n_samples)[20:]
        ys = np.linspace(y_cam,y,n_samples)[20:]
        zs = np.linspace(z_cam,z,n_samples)[20:]
        zt = tracker.z_interpolant(xs,ys,grid=False) 
        return np.any(zt>zs)

    for x,y in xys:
        uv = observer_0.cam.project(np.array([x,y,tracker.z_interpolant(x,y)]).reshape((1,3)))[0]
        row = uv[0][1]
        col = uv[0][0]
        good_row_0 = row>row_min_0 and row<row_max_0
        good_col_0 = col>col_min_0 and col<col_max_0

        uv = observer_1.cam.project(np.array([x,y,tracker.z_interpolant(x,y)]).reshape((1,3)))[0]
        row = uv[0][1]
        col = uv[0][0]
        good_row_1 = row>row_min_1 and row<row_max_1
        good_col_1 = col>col_min_1 and col<col_max_1
 
        good = good_row_0*good_col_0*good_row_1*good_col_1
        if good:
            xys_good.append((x,y))

    def run_tracker(xy):
        tracker.track(xy[0],xy[1],N=3000,do_plot=False,n_iterations=1,wx=2,wy=2,sigma_x0=1.0,sigma_y0=1.0,sigma_vx0=10,sigma_vy0=10,sigma_z0=0)
        return [tracker.means,tracker.covs,tracker.particles,tracker.weights,[[o.rows_observed,o.cols_observed,o.rows_predicted,o.cols_predicted,o.rows_updated,o.cols_updated,o.observation_strength] for o in tracker.observers]]

    #tic = time.time()
    #run_tracker(xys[0])
    #print time.time()-tic

    tic = time.time()
    with sharedmem.MapReduce() as pool:
        results = pool.map(run_tracker,xys_good)
    print time.time() - tic

    #avtime = datetime.datetime.fromtimestamp(0.5*start_time + 0.5*end_time)
    save_zipped_pickle([start_time,end_time,results],'dual_camera_2013/results/dual_4/results_{0:04d}{1:02d}{2:02d}_{3:04d}{4:02d}{5:02d}.p.gz'.format(start_time_dt.year,start_time_dt.month,start_time_dt.day,end_time_dt.year,end_time_dt.month,end_time_dt.day))

    x = np.array([r[0][-1][0] for r in results])
    y = np.array([r[0][-1][1] for r in results])
    vx = np.array([r[0][-1][2] for r in results])
    vy = np.array([r[0][-1][3] for r in results])

    point_loc = np.vstack((x,y)).T
    tree = ssp.KDTree(point_loc)
    hull = ssp.ConvexHull(point_loc)
    vxx = vx.copy()
    vyy = vy.copy()
    n = len(vxx)
    dist_tol = 250
    n_neighbors = 30
    for i in range(n):
        dists,indices = tree.query([x[i],y[i]],n_neighbors)
        vxx[i] = np.nanmedian(vx[indices][dists<dist_tol])
        vyy[i] = np.nanmedian(vy[indices][dists<dist_tol])

    xg = np.arange(x.min(),x.max(),100)
    yg = np.arange(y.min(),y.max(),100)
    grid = np.meshgrid(xg,yg)
    from scipy.interpolate.interpnd import _ndim_coords_from_arrays
    xi = _ndim_coords_from_arrays(tuple(grid), ndim=point_loc.shape[1])
    D,indexes = tree.query(xi)

    Xg,Yg = grid
    Zg = np.array([tracker.z_interpolant(xx,yy)[0][0] for xx,yy in zip(Xg.ravel(),Yg.ravel())]).reshape(Xg.shape)
    V = np.hypot(vxx,vyy)

    rbi = si.Rbf(x,y,V,smooth=1.0,function='thin_plate')
    Vg = rbi(Xg.ravel(),Yg.ravel())
    Vg = Vg.reshape(Xg.shape)
    Vg[D>250] = np.nan
    Vg[Zg<20] = np.nan

    fig,ax = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=(6,8))
    avtime = datetime.datetime.fromtimestamp(0.5*start_time + 0.5*end_time)
    fig.tight_layout()
    im = ax.imshow(Vg,extent=(Xg.min(),Xg.max(),Yg.min(),Yg.max()),origin='lower',vmin=0,vmax=20,alpha=1.0,cmap=plt.cm.gnuplot2)
    qv_1 = ax.quiver(x,y,vxx,vyy,scale=300,width=0.005,clim=(0,20),alpha=0.5,headaxislength=1,headwidth=1.,minlength=0,cmap=plt.cm.gnuplot2,pivot='middle')
    ax.contour(tracker.X,tracker.Y,tracker.Z,np.linspace(20,500,41),cmap=plt.cm.gray,alpha=0.2)
    plt.colorbar(im,ax=ax,orientation='horizontal',pad=0.03,aspect=30)
    ax.text(0.1,0.9,avtime,transform=ax.transAxes)
    ax.axis('equal')
    ax.set_title('Timelapse 3 day average (m/d)')
    ax.set_xlim(496000,499500)
    ax.set_ylim(6.779e6+500,6.779e6+4500)
    fig.savefig('dual_camera_2013/plots/dual_4/tsx_comparison_{0:04d}{1:02d}{2:02d}_{3:04d}{4:02d}{5:02d}.pdf'.format(start_time_dt.year,start_time_dt.month,start_time_dt.day,end_time_dt.year,end_time_dt.month,end_time_dt.day))
    #from scipy.io import loadmat

    #vg = loadmat('dual_camera_2013/tsx/vgrid_2013_06_08_11.mat')['vgrid']
    #X = vg['x'][0][0]
    #Y = vg['y'][0][0]
    #VX = vg['vx'][0][0]/365
    #VY = vg['vy'][0][0]/365
    #VV = np.hypot(VX,VY)

    #fig,ax = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True,figsize=(15,8))
    #fig,ax = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True,figsize=((8,8))
    #avtime = datetime.datetime.fromtimestamp(0.5*start_time + 0.5*end_time)
    #fig.tight_layout()
    #ax[0].imshow(VV,extent=(X.min(),X.max(),Y.min(),Y.max()),origin='lower',vmin=0,vmax=20,cmap=plt.cm.gnuplot2,alpha=0.8)
    #qv_0 = ax[0].quiver(X,Y,VX/VV*np.sqrt(VV),VY/VV*np.sqrt(VV),VV,scale=50,width=0.01,clim=(0,20),alpha=1.0,headaxislength=1,headwidth=1,minlength=0,cmap=plt.cm.gnuplot2,pivot='middle')
    #ax[0].contour(tracker.X,tracker.Y,tracker.Z,np.linspace(0,500,41),cmap=plt.cm.gray,alpha=0.2)
    #ax[0].axis('equal')
    #ax[0].set_title('TSX 14 day average (m/d)')
    #plt.colorbar(qv_0,ax=ax[0],orientation='horizontal',pad=0.03,aspect=30)
    #ax[1].imshow(Vg,extent=(Xg.min(),Xg.max(),Yg.min(),Yg.max()),origin='lower',vmin=0,vmax=20,alpha=0.8,cmap=plt.cm.gnuplot2)
    #qv_1 = ax[1].quiver(x,y,vxx/V*np.sqrt(V),vyy/V*np.sqrt(V),V,scale=50,width=0.01,clim=(0,20),alpha=1.0,headaxislength=1,headwidth=1,minlength=0,cmap=plt.cm.gnuplot2,pivot='middle')
    #ax[1].contour(tracker.X,tracker.Y,tracker.Z,np.linspace(0,500,41),cmap=plt.cm.gray,alpha=0.2)
    #plt.colorbar(qv_1,ax=ax[1],orientation='horizontal',pad=0.03,aspect=30)
    #ax[1].text(0.1,0.9,avtime,transform=ax[1].transAxes)
    #ax[1].axis('equal')
    #ax[1].set_title('Timelapse 3 day average (m/d)')
    #ax[1].set_xlim(496000,499500)
    #ax[1].set_ylim(6.779e6,6.779e6+5000)
    #fig.savefig('dual_camera_2013/_plots/dual/tsx_comparison_{0:04d}{1:02d}{2:02d}.pdf'.format(start_time_dt.year,start_time_dt.month,start_time_dt.day))

