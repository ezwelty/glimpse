% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly excecuted under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 3750.8 ; 3747.9 ];

%-- Principal point:
cc = [ 2148.1 ; 1417.0 ];

%-- Skew coefficient:
alpha_c = 0.0;

%-- Distortion coefficients:
kc = [ -0.1 ; 0.1 ; 0.0 ; 0.0 ; -0.0 ];

%-- Focal length uncertainty:
fc_error = [ 1.80 ; 1.82 ];

%-- Principal point uncertainty:
cc_error = [ 1.0 ; 1.4 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.0;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.002 ; 0.004 ; 0.00 ; 0.000 ; 0.000 ];

%-- Image size:
nx = 4288;
ny = 2848;
