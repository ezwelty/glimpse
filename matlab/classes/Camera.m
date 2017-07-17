classdef Camera
  % CAMERA Distorted camera model.
  %
  % This class is an implementation of the distorted camera model used by OpenCV:
  % http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
  %
  % Note: Pixel coordinates are defined such that [0, 0] is the upper-left of the
  % upper-left pixel and [nx, ny] is the lower-right of the lower-right pixel,
  % where nx and ny are the width and height of the image.
  %
  % Camera Properties:
  % xyz      - Position in world coordinates [x, y, z]
  % viewdir  - View direction in degrees [yaw, pitch, roll]
  %            yaw: clockwise rotation about z-axis (0 = look north)
  %            pitch: rotation from horizon (+ look up, - look down)
  %            roll: rotation about optical axis (+ down right, - down left, from behind)
  % f        - Focal length in pixels [fx, fy]
  % c        - Principal point coordinates in pixels [cx, cy]
  % k        - Radial distortion coefficients [k1, ..., k6]
  % p        - Tangential distortion coefficients [p1, p2]
  % imgsz    - Image size in pixels [nx|ncols|width, ny|nrows|height]
  % sensorsz - Sensor size in mm [width, height] (optional)
  %
  % Camera Properties (dependent):
  % fullmodel - Vector of all 20 camera parameters
  %             [xyz(1:3), imgsz(1:2), viewdir(1:3), f(1:2), c(1:2), k(1:6), p(1:2)]
  % R         - Rotation matrix corresponding to the view direction (read-only)
  % fmm       - Focal length in mm [fx, fy] (not set unless sensorsz is defined)
  %
  % Camera Methods:
  % Camera      - Construct a new Camera object
  % project     - Project world coordinates to image coordinates (3D -> 2D)
  % invproject  - Project image coordinates to world coordinates (2D -> 3D)
  % optimizecam - Optimize camera parameters to mimimize the distance between
  %               projected world coordinates and their expected image coordinates
  % sensorSize  - Lookup sensor size by camera make and model
  %
  % ImGRAFT - An image georectification and feature tracking toolbox for MATLAB
  % Copyright (C) 2014 Aslak Grinsted (https://glaciology.net/)

  properties
    xyz
    viewdir
    f
    c
    k
    p
    imgsz
    sensorsz
  end

  properties (Dependent)
    fullmodel
    R
    fmm
    framepoly
  end

  methods

    % Camera creation

    function cam = Camera(varargin)
      % CAMERA Construct a new camera object.
      %
      % There are several ways to call this method -
      %
      % 1. Initialize with default parameters, then edit individual parameters:
      %
      %   cam = Camera()
      %   cam.viewdir = [pi 0 0] % look west
      %
      % 2. Specify camera parameters as a list:
      %
      %   cam = Camera(xyz, imgsz, viewdir, f, c, k, p, sensorsz)
      %
      % 3. Specify Camera parameters as named arguments:
      %
      %   cam = Camera('xyz', [0 0 10], 'viewdir', [pi 0 0])
      %
      % 4. Specify Camera parameters in a structure with matching field names:
      %
      %   S.xyz = [0 0 10]; S.viewdir = [pi 0 0];
      %   cam = Camera(S)
      %
      % 5. Specify Camera parameters as a 20-element or shorter (fullmodel) vector:
      %
      %   cam = Camera([1 1 0 1024 768 pi 0 0 1000 1000 512 384 0 0 0 0 0 0 0 0])

      % Single vector argument: build camera from fullmodel vector
      if nargin == 1 && isnumeric(varargin{1})
        cam.fullmodel = varargin{1};
        return
      end
      % All other cases: set and validate camera parameters individually
      p = inputParser;
      p.CaseSensitive = false;
      p.StructExpand = true;
      p.addOptional('xyz', [0 0 0], @(x) isnumeric(x) && length(x) <= 3);
      p.addOptional('imgsz', [], @(x) isnumeric(x) && length(x) <= 2);
      p.addOptional('viewdir', [0 0 0], @(x) isnumeric(x) && length(x) <= 3);
      p.addOptional('f', [], @(x) isnumeric(x) && length(x) <= 2);
      p.addOptional('c', [], @(x) isnumeric(x) && length(x) == 2);
      p.addOptional('k', [0 0 0 0 0 0], @(x) isnumeric(x) && length(x) <= 6);
      p.addOptional('p', [0 0], @(x) isnumeric(x) && length(x) <= 2);
      p.addOptional('sensorsz', [], @(x) isnumeric(x) && length(x) <= 2);
      % HACK: Removes non-matching field names from structure
      if nargin == 1 && isstruct(varargin{1})
        fields = fieldnames(varargin{1});
        values = struct2cell(varargin{1});
        include = ismember(fields, p.Parameters);
        varargin{1} = cell2struct(values(include), fields(include));
      end
      p.parse(varargin{:});
      % Set parameters
      for field = fieldnames(p.Results)'
        cam.(field{1}) = p.Results.(field{1});
      end
    end

    function cam = set.xyz(cam, value)
      % xyz: 3-element vector, default = 0
      value(end + 1:3) = 0;
      cam.xyz = value;
    end

    function cam = set.imgsz(cam, value)
      % imgsz: 2-element vector, no default, expand [#] to [#, #]
      if length(value) == 1
        value(end + 1) = value(end);
      end
      cam.imgsz = value;
      cam = cam.update_c();
    end

    function cam = set.c(cam, value)
      % c: 2-element vector, default = imgsz / 2
      if isempty(value)
        cam = cam.update_c();
      else
        cam.c = value;
      end
    end

    function cam = update_c(cam)
      if isempty(cam.c) && ~isempty(cam.imgsz)
        cam.c = cam.imgsz / 2;
        cam.c(cam.imgsz == 0) = 0;
      end
    end

    function cam = set.viewdir(cam, value)
      % viewdir: 3-element vector, default = 0
      value(end + 1:3) = 0;
      cam.viewdir = value;
    end

    function cam = set.f(cam, value)
      % f: 2-element vector, no default, expand [f] to [f, f]
      if length(value) == 1
        value(end + 1) = value(end);
      end
      cam.f = value;
    end

    function cam = set.k(cam, value)
      % k: 6-element vector, default = 0
      value(end + 1:6) = 0;
      cam.k = value;
    end

    function cam = set.p(cam, value)
      % p: 2-element vector, default = 0
      value(end + 1:2) = 0;
      cam.p = value;
    end

    function cam = set.sensorsz(cam, value)
      % sensorsz: 2-element vector, no default, expand [#] to [#, #]
      if length(value) == 1
        value(end + 1) = value(end);
      end
      cam.sensorsz = value;
    end

    function cam = set.fmm(cam, value)
      if isempty(cam.sensorsz)
        error('Camera sensor size not set.');
      else
        cam.f = value .* cam.imgsz ./ cam.sensorsz;
      end
    end

    function cam = set.R(cam, value)
      % cos(elevation) != 0
      if abs(value(3, 3)) ~= 1
        w = asin(value(3, 3));
        p = atan2(value(3, 1) / cos(w), value(3, 2) / cos(w));
        k = atan2(-value(1, 3) / cos(w), -value(2, 3) / cos(w));
      % cos(elevation) == 0
      else
        k = 0; % (unconstrained)
        if value(3, 3) == 1
          w = pi / 2;
          p = -k + atan2(-value(1, 2), value(1, 1));
        else
          w = -pi / 2;
          p = k + atan2(-value(1, 2), value(1, 1));
        end
      end
      cam.viewdir = rad2deg([p w k]);
    end

    function cam = set.fullmodel(cam, value)
      if length(value) < 20
        error('Camera.fullmodel must have 20 elements.')
      end
      cam.xyz = value(1:3);
      cam.imgsz = value(4:5);
      cam.viewdir = value(6:8);
      cam.f = value(9:10);
      cam.c = value(11:12);
      cam.k = value(13:18);
      cam.p = value(19:20);
    end

    function value = get.R(cam)
      % Initial rotations of camera reference frame
      % (camera +z pointing up, with +x east and +y north)
      % Point camera north: -90 deg counterclockwise rotation about x-axis
      %   ri = [1 0 0; 0 cosd(-90) sind(-90); 0 -sind(-90) cosd(-90)];
      % (camera +z now pointing north, with +x east and +y down)

      % View direction rotations
      C = cosd(cam.viewdir); S = sind(cam.viewdir);
      % syms C1 C2 C3 S1 S2 S3
      % yaw: counterclockwise rotation about y-axis (relative to north, from above: +cw, - ccw)
      %   ry = [C1 0 -S1; 0 1 0; S1 0 C1];
      % pitch: counterclockwise rotation about x-axis (relative to horizon: + up, - down)
      %   rp = [1 0 0; 0 C2 S2; 0 -S2 C2];
      % roll: counterclockwise rotation about z-axis (from behind camera: + ccw, - cw)
      %   rr = [C3 S3 0; -S3 C3 0; 0 0 1];

      % Apply all rotations in order
      %   R = rr * rp * ry * ri;
      value = [ C(1) * C(3) + S(1) * S(2) * S(3),  C(1) * S(2) * S(3) - C(3) * S(1), -C(2) * S(3); ...
                C(3) * S(1) * S(2) - C(1) * S(3),  S(1) * S(3) + C(1) * C(3) * S(2), -C(2) * C(3); ...
                C(2) * S(1)                     ,  C(1) * C(2)                     ,  S(2)       ];
    end

    function value = get.fmm(cam)
      if isempty(cam.sensorsz)
        error('Camera sensor size not set.');
      else
        value = cam.f .* cam.sensorsz ./ cam.imgsz;
      end
    end

    function value = get.fullmodel(cam)
      fullmodel = [cam.xyz, cam.imgsz, cam.viewdir, cam.f, cam.c, cam.k, cam.p];
      if length(fullmodel) == 20
        value = [cam.xyz, cam.imgsz, cam.viewdir, cam.f, cam.c, cam.k, cam.p];
      else
        error('Camera parameters missing or not of the correct length.')
      end
    end

    function value = get.framepoly(cam)
      value = [0 0; 0 cam.imgsz(2); cam.imgsz; cam.imgsz(1) 0; 0 0];
    end

    % Methods

    function cam = idealize(cam)
      % IDEALIZE Idealize a camera.
      %
      %   cam = cam.idealize()
      %
      % Sets principal point (c) and distortion coefficients to their defaults.

      cam.k = [];
      cam.p = [];
      cam.c = [];
    end

    function cam = resize(cam, scale)
      % RESIZE Resize a camera.
      %
      %   cam = cam.resize(scale)
      %   cam = cam.resize(imgsz)
      %
      % Adjusts the focal length (f) and principal point (c) to the new image
      % size (imgsz).
      %
      % Inputs:
      %   scale - Scale factor
      %   imgsz - Target image size [nx|ncols|width, ny|nrows|height]

      if isempty(cam.imgsz) || nargin < 2
        return
      end
      % Calculate scale from target image size
      if length(scale) > 1
        scale = Camera.getScaleFromSize(cam.imgsz, scale);
      end
      target_size = round(scale * cam.imgsz);
      scale = target_size ./ cam.imgsz;
      % Apply scale
      if ~isempty(cam.f)
        cam.f = cam.f .* scale;
      end
      if ~isempty(cam.c)
        cam.c = (target_size / 2) + (cam.c - (cam.imgsz / 2)) .* scale;
      end
      cam.imgsz = target_size;
    end

    function pyramid = viewpyramid(cam, radius, include_origin)
      % VIEWPYRAMID Boundaries of camera view.
      %
      %   value = cam.viewpyramid(radius = 1, include_origin = true)
      %
      % Inputs:
      %   radius         - Length of pyramid edges
      %   include_origin - Whether to include lines to camera position
      %
      % Outputs:
      %   pyramid - Pyramid points [x1 y1 z1; ...]

      if nargin < 2 || isempty(radius)
        radius = 1;
      end
      if nargin < 3 || isempty(include_origin)
        include_origin = true;
      end
      u = (0:cam.imgsz(1))';
      v = (0:cam.imgsz(2))';
      edges = {
        cbind(u, 0),
        cbind(cam.imgsz(1), v),
        cbind(flip(u), cam.imgsz(2)),
        cbind(0, flip(v))
      };
      for i = 1:length(edges)
        dxyz = cam.invproject(edges{i});
        scaled_dxyz = dxyz .* (radius ./ sqrt(sum(dxyz.^2, 2)));
        edges{i} = cam.xyz + scaled_dxyz;
        if include_origin
          edges{i} = [cam.xyz; edges{i}; cam.xyz];
        end
      end
      pyramid = cell2mat(edges);
    end

    function box = viewbox(cam, radius)
      % VIEWBOX Bounding box of camera view.
      %
      %   value = cam.viewpyramid(radius = 1)
      %
      % Inputs:
      %   radius - Maxmimum camera view distance
      %
      % Outputs:
      %   box - Corners of bounding box [min(x y z); max(x y z)]
      %
      % See also viewpyramid

      if nargin < 2
        radius = [];
      end
      pyramid = cam.viewpyramid(radius);
      box = [min(pyramid); max(pyramid)];
    end

    function h = plot(cam, dim, radius, color)
      % PLOT Plot camera.
      %
      %   h = cam.plot(dim = 3, radius = 1, color = rand(1, 3))
      %
      % Inputs:
      %   dim    - Dimension of plot (2 or 3)
      %   radius - Radius of plotted view pyramid
      %   color  - Color of camera
      %
      % Outputs:
      %   h - Plot handle
      %
      % See also viewpyramid

      % TODO: In 2D, plot simple camera cone.

      if nargin < 2 || isempty(dim)
        dim = 3;
      end
      if nargin < 3 || isempty(radius)
        radius = 1;
      end
      if nargin < 4 || isempty(color)
        color = rand(1, 3);
      end
      if dim == 3
        corner_directions = cam.invproject(cam.framepoly);
        corners = cam.xyz + corner_directions .* (radius ./ sqrt(sum(corner_directions.^2, 2)));
        h = plot3(cbind(cam.xyz(1), corners(:, 1))', cbind(cam.xyz(2), corners(:, 2))', cbind(cam.xyz(3), corners(:, 3))', 'color', color);
        hold on
        pyramid = cam.viewpyramid(radius, false);
        fill3(pyramid(:, 1), pyramid(:, 2), pyramid(:, 3), color, 'FaceAlpha', 0.5);
      elseif dim == 2
        pyramid = cam.viewpyramid(radius, true);
        h = plot(pyramid(:, 1), pyramid(:, 2), 'color', color);
      end
      xlabel('X'), ylabel('Y'), zlabel('Z')
      hold off
    end

    function h = plot_distortion(cam, scale, normalize, varargin)
      % PLOT_DISTORTION Plot distortion as displacement vectors.
      %
      %   h = cam.plot_distortion(scale = 1, normalize = false, ...)
      %
      % Vectors point from the current to ideal image positions.
      %
      % Inputs:
      %   scale     - Scale factor for displacement vectors
      %   normalize - Whether to plot image with unit width
      %   ...       - Arguments passed to quiver()
      %
      % Outputs:
      %   h - Plot handle
      %
      % See also quiver

      if nargin < 2 || isempty(scale)
        scale = 1;
      end
      if nargin < 3 || isempty(normalize)
        normalize = false;
      end
      nu = 50;
      duv = cam.imgsz(1) / nu;
      u = 0:duv:cam.imgsz(1);
      v = 0:duv:cam.imgsz(2);
      [U V] = meshgrid(u, v);
      P0 = [U(:) V(:)];
      P1 = cam.idealize().camera2image(cam.image2camera(P0));
      box = cam.framepoly;
      if normalize
        box = box / cam.imgsz(1);
        P0 = P0 / cam.imgsz(1);
        P1 = P1 / cam.imgsz(1);
      end
      h = plot(box(:, 1), box(:, 2), 'k:');
      hold on
      quiver(P0(:, 1), P0(:, 2), scale * (P1(:, 1) - P0(:, 1)), scale * (P1(:, 2) - P0(:, 2)), 0, varargin{:});
      set(gca, 'ydir', 'reverse');
      axis equal
      hold off
    end

    % Transformations

    function in = infront(cam, xyz)
      % INFRONT Check whether points are infront of the camera.
      %
      %   in = cam.infront(xyz)
      %
      % Inputs:
      %   xyz - World coordinates [x1 y1 z1; ...]

      dxyz = xyz - cam.xyz;
      Xc = dxyz * cam.R';
      in = Xc(:, 3) > 0;
    end

    function [in, xy] = inview(cam, xyz, directions)
      % INVIEW Checks whether points are within the camera's view.
      %
      %   in = cam.inview(xyz)
      %
      % Inputs:
      %   xyz - World coordinates [x1 y1 z1; ...]

      if nargin < 3 || isempty(directions)
        directions = false;
      end
      if size(xyz, 2) > 2
        xyz = cam.world2camera(xyz, directions);
      end
      % If ideal, project and check if in frame.
      if all(cam.k == 0) && all(cam.p == 0)
        uv = cam.camera2image(xyz);
        in = cam.inframe(uv);
      % Otherwise, inverse-project edges and test points are inside.
      else
        u = (0:(cam.imgsz(1)/100):cam.imgsz(1))';
        v = (0:(cam.imgsz(2)/100):cam.imgsz(2))';
        edge_uv = [
          cbind(u, 0);
          cbind(cam.imgsz(1), v);
          cbind(flip(u), cam.imgsz(2));
          cbind(0, flip(v))
        ];
        edge_xy = cam.image2camera(edge_uv);
        in = inpolygon(xyz(:, 1), xyz(:, 2), edge_xy(:, 1), edge_xy(:, 2));
      end
      if nargout > 1
        xy = xyz;
      end
    end

    function in = inframe(cam, uv)
      % INFRAME Check whether points are on the image.
      %
      %   in = cam.inframe(uv)
      %
      % Inputs:
      %   uv - Image coordinates [u1 v1; ...]

      in = all(uv >= 0, 2) & uv(:, 1) <= cam.imgsz(1) & uv(:, 2) <= cam.imgsz(2);
    end

    function xy = image2camera(cam, uv)
      % IMAGE2CAMERA Convert image to camera coordinates.
      %
      %   xy = cam.image2camera(uv)
      %
      % Inputs:
      %   uv - Image coordinates [u1 v1; ...]
      %
      % Outputs:
      %   xy - Normalized camera coordinates [x1 y1; ...]
      %
      % See also camera2image

      xy = [(uv(:, 1) - cam.c(1)) / cam.f(1), (uv(:, 2) - cam.c(2)) / cam.f(2)];
      xy = cam.undistort(xy);
    end

    function uv = camera2image(cam, xy)
      % CAMERA2IMAGE Convert camera to image coordinates.
      %
      %   uv = cam.camera2image(xy)
      %
      % Inputs:
      %   xy - Normalized camera coordinates [x1 y1; ...]
      %
      % Outputs:
      %   uv - Image coordinates [u1 v1; ...]
      %
      % See also image2camera

      xy = cam.distort(xy);
      uv = [cam.f(1) * xy(:, 1) + cam.c(1), cam.f(2) * xy(:, 2) + cam.c(2)];
    end

    function dxyz = camera2world(cam, xy)
      % CAMERA2WORLD Convert camera coordinates to world ray directions.
      %
      %   dxyz = cam.camera2world(xy)
      %
      % Inputs:
      %   xy - Normalized camera coordinates [x1 y1; ...]
      %
      % Outputs:
      %   dxyz - World ray directions [dx1 dy1 dz1; ...]
      %
      % See also world2camera

      dxyz = [xy ones(size(xy, 1), 1)] * cam.R;
    end

    function [xy, infront] = world2camera(cam, xyz, directions)
      % WORLD2CAMERA Convert world coordinates (directions) to camera coordinates.
      %
      %   [xy, infront] = cam.camera2world(xyz, directions = false)
      %
      % Inputs:
      %   xyz        - World coordinates [x1 y1 z1; ...]
      %   directions - Whether xyz represents coordinates or ray directions
      %
      % Outputs:
      %   xy      - Camera coordinates [x1 y1; ...], NaN if behind camera
      %   infront - Whether point is in front or behind camera
      %
      % See also world2camera

      if nargin < 3 || isempty(directions)
        directions = false;
      end
      if ~directions
        % Convert coordinates to directions
        xyz = xyz - cam.xyz;
      end
      xyz = xyz * cam.R';
      % Normalize by perspective division
      xy = xyz(:, 1:2) ./ xyz(:, 3);
      % Convert points behind camera to NaN
      infront = xyz(:, 3) > 0;
      xy(~infront, :) = NaN;
    end

    function [uv, infront] = project(cam, xyz, directions)
      % PROJECT Project coordinates (directions) to images coordinates.
      %
      %   [uv, infront] = cam.project(xy)
      %   [uv, infront] = cam.project(xyz, directions = false)
      %
      % Inputs:
      %   xy         - Normalized camera coordinates [x1 y1; ...]
      %   xyz        - World coordinates [x1 y1 z1; ...]
      %   directions - Whether xyz represents coordinates or ray directions
      %
      % Outputs:
      %   uv      - Image coordinates [u1 v1; ...]
      %   infront - Whether point is in front or behind camera
      %
      % See also invproject

      if nargin < 3 || isempty(directions)
        directions = false;
      end
      if size(xyz, 2) == 3
        [xy, infront] = cam.world2camera(xyz, directions);
        uv = cam.camera2image(xy);
      elseif size(xyz, 2) == 2
        uv = cam.camera2image(xyz);
        infront = true(size(xyz, 1), 1);
      else
        error('Unsupported point dimensions')
      end
    end

    function xyz = invproject(cam, uv, S)
      % INVPROJECT Project image coordinates to world coordinates (directions).
      %
      %   dxyz = cam.invproject(uv)
      %   xyz = cam.invproject(uv, S)
      %
      % Image coordinates are projected out of the camera as rays. If a surface
      % (S) is specified, the intersections with the surface are returned
      % (or NaN if none). Otherwise, ray directions are returned.
      %
      % Inputs:
      %   uv - Image coordinates [u1 v1; u2 v2; ...]
      %   S  - Surface, either as a DEM object or an infinite plane
      %        (defined as [a b c d], where ax + by + cz + d = 0)
      %
      % Outputs:
      %   dxyz - World ray directions [dx1 dy1 dz1; ...]
      %   xyz  - World coordinates of intersections [x1 y1 z1; ...]
      %
      % See also project

      xy = cam.image2camera(uv);
      xyz = cam.camera2world(xy);
      is_valid = ~any(isnan(xyz), 2);
      if nargin == 2
        % No surface: Return ray directions
      elseif nargin > 2 && isnumeric(S) && length(S) == 4
        % Plane: Return intersection of rays with plane
        xyz(is_valid, :) = intersectRayPlane(cam.xyz, xyz(is_valid, :), S);
      elseif nargin > 2 && isa(S, 'DEM')
        % DEM: Return intersection of rays with DEM
        for i = find(is_valid)'
          % xyz(i, :) = intersectRayDEM([cam.xyz xyz(i, :)], S);
          xyz(i, :) = S.sample(cam.xyz, xyz(i, :));
        end
      end
    end

    % Lines
    function lines = clip_line_inview(cam, xyz)
      % CLIP_LINE_INFRONT Clips a line into the segments in front of a camera.
      %
      %   lines = cam.clip_line_infront(xyz, intersections = true)
      %
      % Inputs:
      %   xyz           - World coordinates [x1 y1 z1; ...]
      %   intersections - Whether to insert vertices at intersections with camera plane
      %
      % Outputs:
      %   lines - Line segments in front of camera {[x1 y1 z1; ...], ...}
      %
      % See also infront, intersectEdgePlane

      in = cam.inview(xyz);
      lines = splitmat(xyz, in);
    end

    function lines = clip_line_infront(cam, xyz, intersections)
      % CLIP_LINE_INFRONT Clips a line into the segments in front of a camera.
      %
      %   lines = cam.clip_line_infront(xyz, intersections = true)
      %
      % Inputs:
      %   xyz           - World coordinates [x1 y1 z1; ...]
      %   intersections - Whether to insert vertices at intersections with camera plane
      %
      % Outputs:
      %   lines - Line segments in front of camera {[x1 y1 z1; ...], ...}
      %
      % See also infront, intersectEdgePlane

      if nargin < 3 || isempty(intersections)
        intersections = true;
      end
      in = cam.infront(xyz);
      [lines, starts, stops] = splitmat(xyz, in);
      if isempty(find(~in, 1)) || ~intersections
        return
      end
      plane = createPlane(cam.xyz, cam.invproject(cam.c));
      for i = 1:numel(lines)
        if starts(i) > 1 && all(isfinite(xyz(starts(i) - 1, :)))
          edge = horzcat(xyz(starts(i) - 1, :), xyz(starts(i), :));
          pt = intersectEdgePlane(edge, plane);
          lines{i} = vertcat(pt, lines{i});
        end
        if stops(i) < size(xyz, 1) && all(isfinite(xyz(stops(i) + 1, :)))
          edge = horzcat(xyz(stops(i), :), xyz(stops(i) + 1, :));
          pt = intersectEdgePlane(edge, plane);
          lines{i} = vertcat(lines{i}, pt);
        end
      end
    end

    function lines = clip_line_inframe(cam, uv, intersections)
      % CLIP_LINE_INFRAME Clips a line into the segments in image frame.
      %
      %   lines = cam.clip_line_inframe(uv, intersections = true)
      %
      % Inputs:
      %   uv            - Image coordinates [x1 y1; ...]
      %   intersections - Whether to insert vertices at intersections with frame
      %
      % Outputs:
      %   lines - Line segments inside image {[x1 y1; ...], ...}
      %
      % See also inframe

      if nargin < 3 || isempty(intersections)
        intersections = true;
      end
      in = cam.inframe(uv);
      [lines, starts, stops] = splitmat(uv, in);
      if isempty(find(~in, 1)) || ~intersections
        return
      end
      box = [0, 0, cam.imgsz];
      for i = 1:numel(lines)
        if starts(i) > 1
          edge = horzcat(uv(starts(i) - 1, :), uv(starts(i), :));
          pt = intersectEdgeBox(edge, box);
          lines{i} = vertcat(pt, lines{i});
        end
        if stops(i) < size(uv, 1)
          edge = horzcat(uv(stops(i), :), uv(stops(i) + 1, :));
          pt = intersectEdgeBox(edge, box);
          lines{i} = vertcat(lines{i}, pt);
        end
      end
    end

    function [e, d, puv] = projerror_lines(cam, uv, xyz)
      % PROJERROR_LINES Reprojection errors of lines to nearest image points.
      %
      %   [e, d, puv] = cam.projerror_lines(uv, xyz)
      %
      % Inputs:
      %   uv         - Target image coordinates [x1 y1; ...]
      %                An optional third column can contain weights
      %   xyz        - World coordinates {[x1 y1 z1; ...], ...}
      %
      % Outputs:
      %   e    - Pixel errors between projected and target image coordinates [du1 dv1; ...]
      %   d    - Pixel distances between projected and target image coordinates
      %   puv  - Projected coordinates of lines nearest target image coordinates
      %
      % See also projerror

      % TODO: Add resample density argument (currently ~1 point per pixel)

      if ~iscell(xyz)
        xyz = {xyz};
      end
      pts = [];
      for i = 1:length(xyz)
        % Clip segments to camera view
        [in, xy] = cam.inview(xyz{i});
        lines = splitmat(xy, in);
        for j = 1:length(lines)
          % Project to camera
          % xy = cam.world2camera(lines{j});
          % Resample
          % TODO: Use faster interpXn
          % t = cumsum(sqrt(sum(diff(xy).^2, 2)));
          % xy = interpXn(xy, t, 0:(t(end) / 1e3):t(end));
          l = polylineLength(lines{j}, 'open');
          n_pts = ceil(l * max(cam.f));
          resampled_xy = resamplePolyline(lines{j}, n_pts);
          % Project to image
          puv = cam.camera2image(resampled_xy);
          % Merge as points
          pts = vertcat(pts, puv);
        end
      end
      if isempty(pts)
        [e, d, dxyz] = deal([]);
        return
      end
      [d, ind] = min(pdist2(uv, pts, 'euclidean'), [], 2);
      e = pts(ind, :) - uv;
      if size(uv, 2) > 2
        e = e .* (uv(:, 3) / mean(uv(:, 3)));
        d = sqrt(sum(e.^2, 2));
      end
      if nargout > 2
        puv = pts(ind, :);
      end
    end

    % Calibration

    function [e, d] = projerror(cam, uv, xyz, directions)
      % PROJERROR Reprojection errors of coordinates.
      %
      %   e = cam.projerror(uv, xy)
      %   e = cam.projerror(uv, xyz, directions = false)
      %
      % Inputs:
      %   uv         - Target image coordinates [x1 y1; ...]
      %                An optional third column can contain weights
      %   xy         - Normalized camera coordinates [x1 y1; ...]
      %   xyz        - World coordinates [x1 y1 z1; ...]
      %   directions - Whether xyz represents coordinates or ray directions
      %
      % Outputs:
      %   e - Pixel errors between projected and target image coordinates [du1 dv1; ...]
      %   d - Pixel distances between projected and target image coordinates
      %
      % See also project

      if nargin < 4 || isempty(directions)
        directions = false;
      end
      puv = cam.project(xyz, directions);
      e = puv - uv(:, 1:2);
      if size(uv, 2) > 2
        e = e .* (uv(:, 3) / mean(uv(:, 3)));
      end
      if nargout > 1
        d = sqrt(sum(e.^2, 2));
      end
    end

    function [X, edge] = horizon(cam, dem, ddeg)
      if nargin < 3
        ddeg = 1;
      end
      viewedges = cam.viewpyramid();
      dxy = bsxfun(@minus, viewedges(1:(end - 1), 1:2), viewedges(end, 1:2));
      angles = atan2d(dxy(:, 2), dxy(:, 1));
      ray_angles = [min(angles):ddeg:max(angles)]';
      [X, edge] = dem.horizon(cam.xyz, ray_angles);
    end

    % Helper functions

    function pixels = size_in_pixels(cam, size, distance)
      xy = [0, 0; size / distance, 0];
      uv = cam.camera2image(xy);
      pixels = sqrt(sum(diff(uv).^2));
    end

  end % methods

  methods (Static)

    function selected = select_params(varargin)
      % SELECT_PARAMS Generate a boolean fullmodel selector.
      %
      %   selected = Camera.select_params(...)
      %   selected = Camera.select_params({...})
      %   selected = Camera.select_params(params)
      %
      % Inputs:
      %   ...    - Name-value pairs: '<name>', <elements>, ...
      %            e.g. 'f', 'xyz', 3 => All f elements and 3rd xyz element
      %   params - 20-element vector coercible to logical (1 and '1' = true)
      %
      % Outputs:
      %   selected - 20-element logical vector of selected parameters

      % Constants
      param_names = {'xyz', 'imgsz', 'viewdir', 'f', 'c', 'k', 'p'};
      param_indices = {[1:3], [4:5], [6:8], [9:10], [11:12], [13:18], [19:20]};
      selected = false(size(cell2mat(param_indices)));
      % Check inputs
      params = varargin;
      if isempty(params)
        return
      end
      if iscell(params{1})
        params = params{1};
      end
      % If single non-matching object, attempt to coerce to logical indices
      if length(params) == 1 && all(not(strcmpi(params{1}, params)))
        selected = (params{1}(:) == 1 | params{1}(:) == '1')';
        selected(end + 1:20) = false;
        return
      end
      % Parse cell array (given as name-element pairs)
      % e.g. {'viewdir', 'xyz', 3} => All viewdir elements and 3rd xyz element
      is_param = cellfun(@ischar, params);
      is_pos = cellfun(@isnumeric, params);
      for i = find(is_param)
        is_match = strcmpi(params{i}, param_names);
        if any(is_match)
          ind = param_indices{is_match};
          if i < length(params) && is_pos(i + 1)
            ind = ind(params{i + 1});
          end
          selected(ind) = true;
        end
      end
    end

    function [e, d, puv] = projerror_bundle(cams, uv, xyz, luv, lxyz, ldmax)
      n = numel(cams);
      [e, d, puv] = deal(cell(n, 1));
      for i = 1:n
        if ~isempty(uv{i}) && ~isempty(xyz{i})
          [e{i}, d{i}] = cams{i}.projerror(uv{i}, xyz{i});
        end
        if ~isempty(luv{i}) && ~isempty(lxyz{i})
          if nargout < 2
            [e_lines, d_lines] = cams{i}.projerror_lines(luv{i}, lxyz{i});
          else
            [e_lines, d_lines, puv{i}] = cams{i}.projerror_lines(luv{i}, lxyz{i});
          end
          d_lines(d_lines > ldmax) = ldmax;
          e{i} = vertcat(e{i}, e_lines);
          d{i} = vertcat(d{i}, d_lines);
        end
      end
    end

    function cams = update_bundle(params, cams, is_flexible, is_fixed)
      % params: n_fix + sum(n_flex)
      n_fixed = sum(is_fixed);
      n_flexible = sum(is_flexible, 2);
      % Fullmodel template
      temp = zeros(1, 20);
      temp(is_fixed) = params(1:n_fixed);
      params(1:n_fixed) = [];
      % Update each camera
      for i = 1:length(cams)
        model = temp;
        model(is_flexible(i, :)) = params(1:n_flexible(i));
        cams{i}.fullmodel = cams{i}.fullmodel + model;
        % cams{i}.fullmodel = model;
        params(1:n_flexible(i)) = [];
      end
    end

    function [newcams, fit] = optimize_bundle(varargin)
      % OPTIMIZECAM  Calibrate a camera from paired image-world coordinates.
      %
      %   [newcam, rmse, aic] = cam.optimizecam(xyz, uv, freeparams)
      %
      % Uses an optimization routine to minize the root-mean-square reprojection
      % error of image-world point correspondences (xyz, uv) by adjusting the
      % specified camera parameters.
      %
      % If uv has three columns, the third column is interpreted as a weight
      % in the misfit function.
      %
      % Inputs:
      %   xyz        - World coordinates [x1 y1 z1; x2 y2 z2; ...]
      %   uv         - Image coordinates [u1 v1; u2 v2; ...]
      %                (optional 3rd column may specify weights)
      %   freeparams - Either a string, array, or 20-element vector describing
      %                which parameters should be optimized (see Examples).
      %
      % Outputs:
      %   newcam - Optimized camera
      %   rmse   - Root-mean-square reprojection error
      %   aic    - Akaike information criterion for reprojection errors, which
      %            can help determine an appropriate degree of complexity for
      %            the camera model (i.e. avoid overfitting).
      %            NOTE: Only strictly applicable for unweighted fitting.
      %
      % Examples:
      %   % Optimize all elements of viewdir:
      %   cam.optimizecam(xyz, uv, '00000111000000000000')
      %   cam.optimizecam(xyz, uv, 'viewdir')
      %   cam.optimizecam(xyz, uv, {'viewdir'})
      %   % Also optimize the third (z) element of xyz:
      %   cam.optimizecam(xyz, uv, '00100111000000000000')
      %   cam.optimizecam(xyz, uv, {'viewdir', 'xyz', 1})
      %

      % INPUTS: cams, uv, xyz, flexparams, fixparams, luv, lxyz, ldmax
      % Enforce defaults
      if length(varargin) < 4
        error('Not enough input arguments.');
      end
      defaults = {[], [], [], [], [], [], [], Inf};
      for i = 5:8
        if i > length(varargin) || isempty(varargin{i})
          varargin{i} = defaults{i};
        end
      end
      % Convert inputs to cell arrays
      for i = [1:4, 6:7]
        if ~iscell(varargin{i})
          varargin{i} = {varargin{i}};
        end
      end
      % Expand inputs
      n_cams = length(varargin{1});
      for i = [2:4, 6:7]
        if ~rem(n_cams, length(varargin{i}))
          varargin{i} = repmat(varargin{i}, 1, n_cams / length(varargin{i}));
        end
        if n_cams ~= length(varargin{i})
          error('Input arrays cannot be coerced to equal length.');
        end
      end
      % Assign inputs
      [cams, uv, xyz, flexparams, fixparams, luv, lxyz, ldmax] = deal(varargin{1:8});
      % Set free parameters
      temp = cellfun(@Camera.select_params, flexparams, 'uniform', false);
      is_flexible = vertcat(temp{:});
      is_fixed = Camera.select_params(fixparams);
      params_initial = zeros(1, sum(is_fixed) + sum(is_flexible(:)));
      % Optimize (initial)
      function e = ef(params)
        newcams = Camera.update_bundle(params, cams, is_flexible, is_fixed);
        [duv, d] = Camera.projerror_bundle(newcams, uv, xyz, cell(1, length(newcams)), cell(1, length(newcams)), []);
        e = reshape(vertcat(duv{:}), [], 1);
      end
      [params_final, ssq] = LMFnlsq(@ef, params_initial);
      cams = Camera.update_bundle(params_final, cams, is_flexible, is_fixed);
      function d = ef2(params)
        newcams = Camera.update_bundle(params, cams, is_flexible, is_fixed);
        duv = Camera.projerror_bundle(newcams, uv, xyz, cell(1, length(newcams)), cell(1, length(newcams)), []);
        e = vertcat(duv{:});
        d = sqrt(sum(e.^2, 2));
        d(d > ldmax) = ldmax;
      end
      % Optimize (iterate)
      has_lines = find(not(cellfun('isempty', luv)) & not(cellfun('isempty', lxyz)));
      if length(has_lines) > 0
        fprintf('Refining...         ');
        original_uv = uv;
        original_xyz = xyz;
        for i = has_lines
          uv{i} = [uv{i}; luv{i}];
        end
        previous_ssq = Inf;
        for iteration = 1:50
          previous_cams = cams;
          for i = has_lines
            [~, ~, puv] = cams{i}.projerror_lines(luv{i}, lxyz{i});
            xyz{i} = [original_xyz{i}; cams{i}.xyz + cams{1}.invproject(puv)];
          end
          [params_final, ssq] = LMFnlsq(@ef2, params_initial);
          cams = Camera.update_bundle(params_final, cams, is_flexible, is_fixed);
          fprintf(['\b\b\b\b\b\b\b\b' num2str(ssq, '%1.2e')]);
          if ssq >= previous_ssq
            cams = previous_cams;
            break
          end
          previous_ssq = ssq;
        end
        fprintf('\n');
        uv = original_uv;
        xyz = original_xyz;
      else
        cams = newcams;
      end
      % Model statistics
      n = reshape(cellfun(@(x, y) size(x, 1) + size(y, 1), uv, luv), 1, []);
      e = Camera.projerror_bundle(cams, uv, xyz, luv, lxyz, ldmax);
      rss = reshape(cellfun(@(x) sum(sum(x.^2, 2)), e), 1, []);
      fit = struct();
      fit.rmse = sqrt(rss ./ n);
      n = sum(n);
      k = length(params_final);
      rss = sum(rss);
      % AIC: https://en.wikipedia.org/wiki/Akaike_information_criterion
      % AIC small sample correction: http://brianomeara.info/tutorials/aic/
      fit.aic = n .* log(rss ./ n) + 2 * k .* (n ./ (n - k - 1));
      fit.bic = n .* log(rss ./ n) + 2 * k .* log(n);
      % Camera model selection: http://imaging.utk.edu/publications/papers/2007/ICIP07_vo.pdf
      fit.mdl = n .* log(rss ./ n) + 1 ./ (2 * k .* log(n));
    end

    function [images, fit] = optimize_images(images, flexparams, fixparams, ldmax, lxyz)
      % Enforce defaults
      if nargin < 2 || isempty(flexparams)
        flexparams = [];
      end
      if nargin < 3 || isempty(fixparams)
        fixparams = [];
      end
      if nargin < 4 || isempty(ldmax)
        ldmax = Inf;
      end
      if nargin < 5
        lxyz = arrayfun(@(img) img.gcl.xyz, images, 'uniform', false);
      end
      % Collect cameras and control from images
      cams = {images.cam};
      uv = arrayfun(@(img) img.gcp.uv, images, 'uniform', false);
      xyz = arrayfun(@(img) img.gcp.xyz, images, 'uniform', false);
      if isempty(lxyz)
        luv = [];
      else
        luv = arrayfun(@(img) img.gcl.uv, images, 'uniform', false);
        if ~iscell(lxyz{1})
          lxyz = {lxyz};
        end
      end
      % Optimize cameras
      [newcams, fit] = Camera.optimize_bundle(cams, uv, xyz, flexparams, fixparams, luv, lxyz, ldmax);
      % Save new cameras
      for i = 1:length(newcams)
        images(i).cam = newcams{i};
      end
    end

    function scale = getScaleFromSize(original_size, target_size)
      % GETSCALEFROMSIZE Calculate the scale that achieves a target size.
      %
      %   scale = Camera.getScaleFromSize(original_size, target_size)
      %
      % Inputs:
      %   original_size - Starting image size [nx, ny]
      %   target_size   - Target image size [nx, ny]
      %
      % Outputs:
      %   scale - Scalar scale factor consistent with the target size.

      original_size = round(original_size);
      target_size = round(target_size);
      if all(original_size == target_size)
        scale = 1;
        return
      end
      scale_bounds = target_size ./ original_size;
      error_function = @(scale) sum(abs(round(original_size * scale) - target_size));
      [scale, value] = fminbnd(error_function, min(scale_bounds), max(scale_bounds));
      if value > 0
        error('No scale can achieve the target size.');
      end
    end

    function sensorsz = getSensorSize(varargin)
      % SENSORSIZE Get the sensor size of a digital camera model.
      %
      %   sensorsz = Camera.getSensorSize(makemodel)
      %   sensorsz = Camera.getSensorSize(make, model)
      %
      % Returns the CCD sensor width and height in mm for the specified camera.
      % Data is from Digital Photography Review (https://dpreview.com).
      % See also https://www.dpreview.com/articles/8095816568/sensorsizes.
      %
      % Inputs:
      %   makemodel - Camera make and model [make ' ' model]
      %   make      - Camera make
      %   model     - Camera model
      %
      % Outputs:
      %   sensorsz - Sensor size in mm [width, height]

      % Check inputs
      if nargin < 1
        error('Specify make & model of camera.')
      end
      makemodel = deblank(strtrim(varargin{1}));
      if nargin > 1
        makemodel = [makemodel, ' ', deblank(strtrim(varargin{2}))];
      end
      % Load sensor sizes (mm)
      sensor_sizes = {
        'NIKON CORPORATION NIKON D2X', [23.7 15.7]; % https://www.dpreview.com/reviews/nikond2x/2
        'NIKON CORPORATION NIKON D200', [23.6 15.8]; % https://www.dpreview.com/reviews/nikond200/2
      };
      % Check for match
      match = find(strcmpi(makemodel, sensor_sizes(:, 1)));
      % If match found, return sensor size
      if isempty(match)
        warning(['No sensor size found for "' makemodel '".']);
        sensorsz = [];
      else
        sensorsz = sensor_sizes{match, 2};
      end
    end

  end % methods (Static)

  methods (Access = private)

    function xy = distort(cam, xy)
      % DISTORT Apply distortion to normalized camera coordinates.
      %
      %   xy = cam.distort(xy)
      %
      % Inputs:
      %   xy - Normalized camera coordinates [x1 y1; ...]
      %
      % Outputs:
      %   xy - Distorted normalized camera coordinates [x1 y1; ...]
      %
      % See also undistort

      if any([cam.k, cam.p])
        r2 = sum(xy.^2, 2);
        if any(cam.k)
          % Radial lens distortion
          % dr = (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) / (1 + k4 * r^2 + k5 * r^4 + k6 * r^6)
          dr = 1 + cam.k(1) * r2 + cam.k(2) * r2.^2 + cam.k(3) * r2.^3;
          if any(cam.k(4:6))
            dr = dr ./ (1 + cam.k(4) * r2 + cam.k(5) * r2.^2 + cam.k(6) * r2.^3);
          end
        end
        if any(cam.p)
          % Tangential lens distortion
          % dtx = 2xy * p1 + p2 * (r^2 + 2x^2)
          % dty = p1 * (r^2 + 2y^2) + 2xy * p2
          xty = xy(:, 1) .* xy(:, 2);
          dtx = 2 * xty * cam.p(1) + cam.p(2) * (r2 + 2 * xy(:, 1).^2);
          dty = cam.p(1) * (r2 + 2 * xy(:, 2).^2) + 2 * xty * cam.p(2);
        end
        % Compute distorted camera coordinates
        % x' = dr * x + dtx
        % y' = dr * y + dty
        if any(cam.k)
          xy = xy .* dr;
        end
        if any(cam.p)
          xy = xy + [dtx dty];
        end
      end
    end

    function xy = undistort(cam, xy)
      % UNDISTORT Undo distortion on normalized camera coordinates.
      %
      %   xy = cam.undistort(xy)
      %
      % Inputs:
      %   xy - Distorted normalized camera coordinates [x1 y1; ...]
      %
      % Outputs:
      %   xy - normalized camera coordinates [x1 y1; ...]
      %
      % See also distort

      if any([cam.k, cam.p])
        % May fail for large negative k1.
        if cam.k(1) < -0.5
          warning(['Large, negative k1 (', num2str(cam.k(1), 3), '). Undistort may fail.'])
        end
        % If only k1 is nonzero, use closed form solution.
        % Cubic roots solution from Numerical Recipes in C 2nd Edition:
        % http://apps.nrbook.com/c/index.html (pages 183-185)
        if sum([cam.k cam.p] ~= 0) == 1 && cam.k(1)
          phi = atan2(xy(:, 2), xy(:, 1));
          Q = -1 / (3 * cam.k(1));
          R = -xy(:, 1) ./ (2 * cam.k(1) * cos(phi));
          % For negative k1
          if cam.k(1) < 0
            th = acos(R ./ sqrt(Q^3));
            r = -2 * sqrt(Q) * cos((th - 2 * pi) / 3);
          % For positive k1
          else
            A = (sqrt(R.^2 - Q^3) - R).^(1/3);
            B = Q * (1 ./ A);
            r = (A + B);
          end
          xy = [r .* cos(phi), r .* sin(phi)];
          xy = real(xy);
        % Otherwise, use iterative solution.
        else
          xyi = xy; % initial guess
          for n = 1:20
            r2 = sum(xy.^2, 2);
            if any(cam.k)
              % Radial lens distortion
              % dr = (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) / (1 + k4 * r^2 + k5 * r^4 + k6 * r^6)
              dr = 1 + cam.k(1) * r2 + cam.k(2) * r2.^2 + cam.k(3) * r2.^3;
              if any(cam.k(4:6))
                dr = dr ./ (1 + cam.k(4) * r2 + cam.k(5) * r2.^2 + cam.k(6) * r2.^3);
              end
            end
            if any(cam.p)
              % Tangential lens distortion
              % dtx = 2xy * p1 + p2 * (r^2 + 2x^2)
              % dty = p1 * (r^2 + 2y^2) + 2xy * p2
              xty = xy(:, 1) .* xy(:, 2);
              dtx = 2 * xty * cam.p(1) + cam.p(2) * (r2 + 2 * xy(:, 1).^2);
              dty = cam.p(1) * (r2 + 2 * xy(:, 2).^2) + 2 * xty * cam.p(2);
            end
            % Remove distortion
            % x = (x' - dtx) / dr
            % y = (y' - dty) / dr
            if any(cam.p)
              xy = xyi - [dtx, dty];
              if any(cam.k)
                xy = xy ./ dr;
              end
            else
              xy = xyi ./ dr;
            end
          end
        end
      end
    end

  end % methods (private)

end % classdef
