import numpy as np
import sklearn.decomposition as sde
import cv2

class Observer(object):
    def __init__(self,times,images,sigma_pixel=0.3,reference_halfwidth=15,search_halfwidth=20):
        self.images = images
        self.times = times
        self.ref_template=None
        self.pca = sde.PCA(n_components=1,svd_solver='arpack',whiten=True)
 
        self.sigma_pixel = sigma_pixel
        self.hw_row = reference_halfwidth
        self.hw_col = reference_halfwidth
        self.sd_row = search_halfwidth
        self.sd_col = search_halfwidth

    def initialize_point(self,xyz):
        row_0,col_0 = self.images[0].cam.project(xyz)
        
        row_nearest = int(np.rint(row_0))
        col_nearest = int(np.rint(col_0))
        self.row_diff_0 = row_nearest-row_0
        self.col_diff_0 = col_nearest-col_0

        self.ref_hist_template = self.get_subimage(row_nearest,col_nearest,self.sd_row,self.sd_col,0,pca=False,median_filter_size=None)
        self.ref_subimage = self.get_subimage(row_nearest,col_nearest,self.hw_row,self.hw_col,0)

    def get_subimage(self,row,col,hw_row,hw_col,image_index,pca=True,median_filter_size=(5,5)):
        chip = self.images[image_index].read()[row-hw_row:1+row+hw_row,col-hw_col:1+col+hw_col].copy()
        if self.ref_hist_template is not None:
            chip[:,:,0] = hist_match(chip[:,:,0],self.ref_template[:,:,0])
            chip[:,:,1] = hist_match(chip[:,:,1],self.ref_template[:,:,1])
            chip[:,:,2] = hist_match(chip[:,:,2],self.ref_template[:,:,2])
        if pca:
            m,n,q = chip.shape
            Q = chip.reshape((m*n,q))
            self.pca.fit(Q)
            self.pca.components_ = np.sign(self.pca.components_[0])*self.pca.components_
            Qp = self.pca.transform(Q)
            chip = Qp.reshape((m,n))
        if median_filter_size is not None:
            # Do median highpass filtering
            chip_lowpass = filts.median_filter(chip,(median_filter_size[0],median_filter_size[1]))
            chip -= chip_lowpass
        return chip
    
    def compute_likelihood(self,pmean,particles,t):
        tdist = np.abs(t-self.times)
        image_index = np.argmin(tdist)
        time_mismatch = np.min(tdist)
        if time_mismatch>0.1:
            return 0.0

        current_image = self.images[image_index]
        rcmean = current_image.cam.project(pmean[[0,1,4]]) 
        row_1 = rcmean[0][0]
        col_1 = rcmean[0][1] 
        row_nearest = int(np.rint(row_1))
        col_nearest = int(np.rint(col_1))
        row_diff_1 = row_nearest-row_1
        col_diff_1 = col_nearest-col_1
        test_subimage = self.get_subimage(row_nearest,col_nearest,self.sd_row,self.sd_col,image_index)                  

        rc_cam0 = self.images[0].cam.project(pmean[[0,1,4]])
        self.rows_predicted.append(rc_cam0[0][0])
        self.cols_predicted.append(rc_cam0[0][1])
        
        try:
            like_interpolant = self.get_likelihood_interpolant(self.ref_subimage,test_subimage,row_nearest,col_nearest,row_diff_1,col_diff_1,self.row_diff_0,self.col_diff_0)

            rc = current_image.project(particles[:,[0,1,4]])

            log_like = like_interpolant(rc[:,0],rc[:,1],grid=False)/self.sigma_pixel**2
        except IndexError:
            log_like = 1.

        return log_like
        
    def get_likelihood_interpolant(self,ref_subimage,test_subimage,row_nearest,col_nearest,row_diff_1,col_diff_1,row_diff_0,col_diff_0):
       
        # Compute normalized correlation coefficients
        rhos = cv2.matchTemplate(test_subimage.astype('float32'),ref_subimage.astype('float32'),method=cv2.TM_SQDIFF)
        rhos/=(ref_chip.shape[0]*ref_chip.shape[1])

        rcoords = row_nearest - (self.sd_row-self.hw_row) + np.array(range(rhos.shape[0])) + row_diff_1 - row_diff_0
        ccoords = col_nearest - (self.sd_col-self.hw_col) + np.array(range(rhos.shape[1])) + col_diff_1 - col_diff_0

        # Get subpixel accuracy by fitting the correlation surface with a cubic spline and maximizing
        local_interp = si.RectBivariateSpline(rcoords,ccoords,rhos,kx=3,ky=3)

        return local_interp

    def solve_orientations(self):
        # NEED TO WRITE ME!
        pass

    
            
    
 
 

        
