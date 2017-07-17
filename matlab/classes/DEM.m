classdef DEM
  % DEM Digital elevation model.
  %
  % DEM Properties (read-only):
  % Z     - Grid of values on a regular xy grid
  % xlim  - Outer bounds of the grid in x [left, right]
  % ylim  - Outer bounds of the grid in y [top, bottom]
  % zlim   - Limits of Z [min(Z), max(Z)]
  % nx,ny - Dimensions of grid [size(Z, 2)], [size(Z, 1)]
  % dx,dy - Grid cell size [range(x) / nx], [range(y) / ny]
  % x,y    - Cell center coordinates as row vectors [left to right], [top to bottom]
  % X,Y    - Cell center coordinates as grids, equivalent to meshgrid(x, y)
  %
  % DEM Methods:
  % DEM    - Construct a new DEM object
  % crop   - Return a cropped DEM
  % resize - Return a resized DEM
  % plot   - Plot a DEM as a hillshade in 2D or 3D

  properties (SetAccess = private)
    Z
    xlim
    ylim
    zlim
    min
    max
    nx,ny
    dx,dy
    x,y
    X,Y
    zmin
    zmax
  end

  methods

    % Create DEM

    function dem = DEM(varargin)
      % DEM  Construct a new DEM object.
      %
      %   dem = DEM(geotiff)
      %   dem = DEM(Z, xlim = [0 size(Z, 2)], ylim = [0 size(Z, 1)])
      %   dem = DEM(Z, x/X, y/Y)
      %
      % Inputs:
      %   Z         - Grid of elevations
      %   xlim,ylim - Outer x,y limis of the grid [left, right], [top, bottom]
      %   x,y       - Cell center coordinate vectors [left to right], [top to bottom]
      %               Monotonic and evenly spaced
      %   X,Y       - Cell center coordinate grids
      %               Equivalent to meshgrid(x, y)
      %
      % Assumes that Z and x,y coordinates are supplied such that:
      % left   @  xlim(1)  , x(1)  , or X(1, 1)
      % right  @  xlim(end), x(end), or X(1, end)
      % top    @  ylim(1)  , y(1)  , or Y(1, 1)
      % bottom @  ylim(end), y(end), or Y(end, 1)
      %
      % See also meshgrid

      % --- Return empty ---
      if nargin < 1
        return
      end
      % --- Build from geotiff file ---
      if ischar(varargin{1})
        [Z, ~, bbox] = geotiffread(varargin{1});
        dem = DEM(double(Z), bbox(:, 1), flip(bbox(:, 2)));
        return
      end
      % --- Build from numeric arguments ---
      % Z, zlim
      dem.Z = DEM.parse_Z(varargin{1});
      dem.zlim = dem.compute_zlim();
      % nx, ny
      dem.nx = dem.compute_nx();
      dem.ny = dem.compute_ny();
      % xlim, x, X
      if nargin < 2
        xlim = [0 dem.nx];
      else
        xlim = varargin{2};
      end
      [dem.xlim, x, X] = DEM.parse_xlim(xlim);
      dem.dx = dem.compute_dx();
      if length(x) == dem.nx
        dem.x = x;
      else
        dem.x = dem.compute_x();
      end
      if all(size(X) == size(dem.Z))
        dem.X = X;
      end
      % ylim, y, Y
      if nargin < 3
        ylim = [dem.ny 0];
      else
        ylim = varargin{3};
      end
      [dem.ylim, y, Y] = DEM.parse_ylim(ylim);
      dem.dy = dem.compute_dy();
      if length(y) == dem.ny
        dem.y = y;
      else
        dem.y = dem.compute_y();
      end
      if all(size(Y) == size(dem.Z))
        dem.Y = Y;
      end
      % min, max
      dem.min = dem.compute_min();
      dem.max = dem.compute_max();
    end

    function dem = build(dem, properties)
      if nargin < 2
        meta = ?DEM;
        properties = {meta.PropertyList.Name};
      end
      if ischar(properties)
        properties = {properties};
      end
      for p = properties
        if isempty(dem.(p{1})) || nargin > 1
          dem.(p{1}) = dem.(['compute_' (p{1})]);
        end
      end
    end

    % Modify & Display DEM

    function dem = crop(dem, xlim, ylim, zlim)
      % CROP Return a cropped DEM.
      %
      %   dem = dem.crop(xlim = dem.xlim, ylim = dem.ylim, zlim = dem.zlim)
      %
      % Crops a DEM to the specified x, y, and z limits. Includes cells
      % intersected by the boundary (rather than resampling the DEM to conform
      % to the crop boundaries). Values of Z outside zlim are set to NaN.
      %
      % Inputs:
      %   xlim - x crop boundaries
      %   ylim - y crop boundaries
      %   zlim - z crop boundaries

      % --- Crop xlim ---
      if nargin < 2 || isempty(xlim)
        xlim = dem.xlim;
      else
        xlim = range_intersection(sort(dem.xlim), sort(xlim));
        if range(xlim) == 0 || any(size(xlim) == 0)
          error('Crop bounds do not intersect DEM.')
        end
        if sign(diff(xlim)) ~= sign(diff(dem.xlim))
          xlim = flip(xlim);
        end
      end
      % --- Crop ylim ---
      if nargin < 3 || isempty(ylim)
        ylim = dem.ylim;
      else
        ylim = range_intersection(sort(dem.ylim), sort(ylim));
        if range(ylim) == 0 || any(size(ylim) == 0)
          error('Crop bounds do not intersect DEM.')
        end
        if sign(diff(ylim)) ~= sign(diff(dem.ylim))
          ylim = flip(ylim);
        end
      end
      % --- Convert limits to grid indices ---
      mincol = floor(abs(xlim(1) - dem.xlim(1)) / dem.dx) + 1;
      maxcol = ceil(abs(xlim(2) - dem.xlim(1)) / dem.dx);
      minrow = floor(abs(ylim(1) - dem.ylim(1)) / dem.dy) + 1;
      maxrow = ceil(abs(ylim(2) - dem.ylim(1)) / dem.dy);
      % --- Crop xy ---
      z = dem.Z(minrow:maxrow, mincol:maxcol);
      xlim = interp1([0 dem.nx], dem.xlim, [mincol - 1, maxcol]);
      ylim = interp1([0 dem.ny], dem.ylim, [minrow - 1, maxrow]);
      % --- Crop z ---
      if nargin > 3 && ~isempty(zlim)
        z(z > max(zlim) | z < min(zlim)) = NaN;
      end
      % --- Return cropped DEM ---
      dem = DEM(z, xlim, ylim);
    end

    function dem = resize(dem, scale, method)
      % RESIZE Return a resized DEM.
      %
      %   dem = dem.resize(scale, method = 'bicubic')
      %   dem = dem.resize(size, method = 'bicubic')
      %
      % Resamples a DEM grid by the specified scale factor and interpolation
      % method. Cell size is changed, but xy boundaries remain the same.
      %
      % Inputs:
      %   scale  - Scale factor
      %   size   - Target grid size [nx, ny]
      %   method - Interpolation method, passed to imresize
      %
      % See also imresize

      % --- Check inputs ---
      if nargin < 3
        method = 'bicubic';
      end
      % --- Resize grid ---
      z = imresize(dem.Z, flip(scale), method);
      % --- Return resized DEM ---
      dem = DEM(z, dem.xlim, dem.ylim);
    end

    function dem = smooth(dem, width)
      % FILL Fill holes in DEM.
      %
      %   dem = dem.fill(width = 100)
      %
      % Based on the crevasse-filling method of Messerli & Grinsted 2015:
      % http://www.geosci-instrum-method-data-syst.net/4/23/2015/
      %
      % Inputs:
      %   width - Width of smoothing window

      % FIXME: Supports only square pixels. Enforce square pixels globally?

      % --- Set defaults ---
      if nargin < 2
        width = 100;
      end
      % --- Fill holes ---
      w = 2 .* round((width / dem.dx + 1) / 2) - 1; % nearest odd integer
      gauss_filter = fspecial('gaussian', w, 6);
      Zg = nanconv(dem.Z, gauss_filter, 'edge', 'nanout');
      disk = fspecial('disk', w);
      a = nanstd(dem.Z(:) - Zg(:));
      Zd = nanconv(exp((dem.Z - Zg) ./ a), disk, 'edge', 'nanout');
      Zf = a * log(Zd) + Zg;
      dem.Z = Zf;
      affected_props = {'zmin', 'zmax', 'zlim'};
      is_built = cellfun(@(p) ~isempty(dem.(p)), affected_props);
      dem = dem.build(affected_props(is_built));
    end

    function dem = fill_circle(dem, center, radius, value)
      % FILL_CIRCLE Assign value to pixels inside circle.
      %
      %   dem = dem.fill_circle(center, radius = 100, value = NaN)
      %
      % Inputs:
      %   center - Circle center coordinates [x y]
      %   radius - Circle radius
      %   value  - Value to apply inside circle

      % --- Set defaults ---
      if nargin < 3
        radius = 100;
      end
      if nargin < 4
        value = NaN;
      end
      % --- Circle indices ---
      [x0, y0] = dem.xy2ind(center);
      r = round(radius / dem.dx);
      [xc, yc] = getmidpointcircle(x0, y0, r);
      % --- Filled circle indices ---
      ind = [];
      y = unique(yc);
      yin = ~(y < 1 | y > dem.ny);
      for yi = reshape(y(yin), 1, [])
        xb = xc(yc == yi);
        xi = max(min(xb), 1):min(max(xb), dem.nx);
        ind((end + 1):(end + length(xi))) = sub2ind([dem.ny, dem.nx], repmat(yi, 1, length(xi)), xi);
      end
      % --- Apply to DEM ---
      dem.Z(ind) = value;
    end

    function h = plot(dem, dim, shade)
      % PLOT Plot a DEM.
      %
      %   h = dem.plot(dim = 2, shade - hillshade(dem.Z, dem.x, dem.y))
      %
      % Plots a DEM in either 2 or 3 dimensions with a custom overlay.
      %
      % Inputs:
      %   dim   - Dimension of plot (2 or 3)
      %   shade - Custom shading image, same size as dem.Z
      %
      % Outputs:
      %   h - Figure handle
      %
      % See also imagesc, surf, hillshade

      % --- Choose dimension ---
      if nargin < 2
        dim = 2;
      end
      % --- Resize large DEM ---
      if (dem.nx * dem.ny > 2e6)
        scale = sqrt(2e6 / (dem.nx * dem.ny));
        dem = dem.resize(scale);
        if nargin > 2
          shade = imresize(shade, scale);
        end
        warning('DEM automatically downsized for faster plotting');
      end
      % --- Plot DEM ---
      if nargin < 3
        shade = hillshade(dem.Z, dem.x, dem.y);
      end
      if dim == 2
        h = imagesc(dem.x, dem.y, shade);
        axis image
        set(gca, 'ydir', 'normal')
      elseif dim == 3
        h = surf(dem.x, dem.y', dem.Z, double(shade), 'EdgeColor', 'none');
        shading interp
        axis equal
      end
      colormap gray
    end

    % Interact with DEM

    function z = sample_points(dem, xy, method)
      % SAMPLE_POINTS Sample DEM at xy points.
      %
      %   z = dem.sample_points(xy, method = 'linear')
      %
      % Inputs:
      %   xy     - Point coordinates [x1 y1; ...; xn yn]
      %   method - Interpolation method (see interp2). Only 'nearest' and
      %            'spline' return values between center coordinates and
      %            bounds.
      %
      % Outputs:
      %   z - Value at points, or NaN if outside bounds
      %
      % See also interp2, xy2ind, ind2xyz

      % --- Choose interpolation method ---
      if nargin < 3
        method = 'linear';
      end
      % --- Select inbound points ---
      in = dem.inbounds(xy);
      % --- Interpolate Z at points ---
      z = nan(size(xy, 1), 1);
      switch method
        case 'nearest'
          zi = dem.xy2ind(xy(in, :));
          z(in) = dem.ind2xyz(zi);
        otherwise
          method = ['*' method];
          z(in) = interp2(dem.X, dem.Y, dem.Z, xy(in, 1), xy(in, 2), method);
      end
    end

    function z = sample_points_tri(dem, xy)
      % SAMPLE_POINTS_TRI Sample triangles at xy points.
      %
      %   z = dem.sample_points_tri(xy)
      %
      % Inputs:
      %   xy     - Point coordinates [x1 y1; ...; xn yn]
      %
      % Outputs:
      %   z - Value at points, or NaN if outside bounds

      % TODO: Compare to rayTriangleIntersection, intersectLineTriangle3d, intersectRayPolygon3d

      % --- Process only inbound points ---
      in = dem.zmin.inbounds(xy);
      % --- Locate points on reduced grid ---
      [xi, yi] = dem.zmin.xy2ind(xy(in, :));
      % --- Determine whether upper or lower triangle (SLOW) ---
      u = xy(in, 1) - (dem.zmin.xlim(1) + (xi - 1) * sign(diff(dem.zmin.xlim)) * dem.dx);
      v = xy(in, 2) - (dem.zmin.ylim(1) + yi * sign(diff(dem.zmin.ylim)) * dem.dy);
      in_upper = v >= u * (dem.zmin.dy / dem.zmin.dx);
      % --- Build triangle grid indices ---
      xi_tri = repmat(xi, 1, 3);
      yi_tri = repmat(yi, 1, 3);
      % upper-left triangle
      xi_tri(in_upper, :) = xi_tri(in_upper, :) + [0 0 1];
      yi_tri(in_upper, :) = yi_tri(in_upper, :) + [1 0 0];
      % lower-right triangle
      xi_tri(~in_upper, :) = xi_tri(~in_upper, :) + [1 1 0];
      yi_tri(~in_upper, :) = yi_tri(~in_upper, :) + [0 1 1];
      % --- Convert to linear indices ---
      zi_tri = sub2ind(size(dem.Z), yi_tri, xi_tri);
      % --- Find line-triangle intersection (SLOWEST) ---
      z_in = nan(length(xi), 1);
      xy_in = xy(in, :);
      for i = 1:length(xi)
        ind = zi_tri(i, :)';
        tri = [dem.X(ind) dem.Y(ind) dem.Z(ind)];
        u = tri(1, :) - tri(3, :);
        v = tri(2, :) - tri(3, :);
        n = cross(u, v);
        d = n(1) * tri(1, 1) + n(2) * tri(1, 2) + n(3) * tri(1, 3);
        z_in(i) = (d - (n(1) * xy_in(i, 1) + n(2) * xy_in(i, 2))) / n(3);
      end
      % --- Compile results ---
      z = nan(size(xy, 1), 1);
      z(in) = z_in;
    end

    function X = sample_ray_tri(dem, origin, direction, first, xy0, r0)
      % SAMPLE_RAY_TRI Sample triangles along ray.
      %
      %   X = dem.sample_ray_tri(origin, direction, first = true)
      %   X = dem.sample_ray_tri(origin, direction, first = true, dlim)
      %   X = dem.sample_ray_tri(origin, direction, first = true, xy0, r0)
      %
      % Inputs:
      %   origin    - Origin of ray [x, y, z]
      %   direction - Direction of ray [dx, dy, dz]
      %   first     - Whether to return first or all intersections
      %   dlim      - Minimum (and optionally maximum) xy-distance to search
      %   xy0       - Start of search [x, y]
      %   r0        - Radius of search from xy0
      %
      % Outputs:
      %   X - Coordinates of intersections [x1 y1 z1; ... ; xn yn zn]
      %
      % See Amanatides & Woo 1987:
      % http://www.cse.yorku.ca/~amana/research/grid.pdf

      % TODO: Compare to intersectLineMesh3d
      % TODO: Vectorize for shared origin (as when emanating from camera)
      % TODO: Pre-compute triangles
      % TODO: Traverse grid in a quadtree

      X = [];
      % --- Check inputs ---
      if nargin < 4 || isempty(first)
        first = true;
      end
      if any(isnan([origin, direction]))
        if first
          X = nan(1, 3);
        end
        return
      end
      % --- Intersect ray with bounding box ---
      if nargin < 5
        boxmin = dem.zmin.min;
        boxmax = dem.zmin.max;
      else
        if nargin < 6
          % Apply xy-distance limits
          if length(xy0) == 1
            xy0(2) = Inf;
          end
          corners = origin(1:2) + (direction(1:2) ./ sqrt(sum(direction(1:2).^2))) .* reshape(xy0(1:2), 2, []);
          rbox = [min(corners(:, 1)), max(corners(:, 1)), min(corners(:, 2)), max(corners(:, 2))];
        else
          % Apply search radius (if specified)
          rbox = [xy0(1) + [-r0, r0], xy0(2) + [-r0, r0]];
        end
        dembox = [sort(dem.zmin.xlim), sort(dem.zmin.ylim)];
        box = intersectBoxes(dembox, rbox);
        [xi, yi] = dem.zmin.xy2ind(combvec(box(1:2), box(3:4))');
        xbox = sort(dem.zmin.xlim(1) + [min(xi) - 1, max(xi)] * dem.zmin.dx * sign(diff(dem.zmin.xlim)));
        ybox = sort(dem.zmin.ylim(1) + [min(yi) - 1, max(yi)] * dem.zmin.dy * sign(diff(dem.zmin.ylim)));
        boxmin = [xbox(1), ybox(1), dem.zmin.min(3)];
        boxmax = [xbox(2), ybox(2), dem.zmin.max(3)];
      end
      % Intersect with outer grid boundaries
      [tmin, tmax] = intersectRayBox(origin, direction, boxmin, boxmax);
      if isempty(tmin)
        if first
          X = nan(1, 3);
        end
        return
      end
      % Compute endpoints of ray within grid
      start = origin + tmin * direction;
      stop = origin + tmax * direction;
      % Find starting voxel coordinates
      x = ceil((start(1) - dem.zmin.min(1)) / dem.zmin.dx);
      y = ceil((start(2) - dem.zmin.min(2)) / dem.zmin.dy);
      % Find ending voxel coordinates
      mx = ceil((stop(1) - dem.zmin.min(1)) / dem.zmin.dx);
      my = ceil((stop(2) - dem.zmin.min(2)) / dem.zmin.dy);
      % Snap to 1 (from 0)
      x = max(x, 1);
      y = max(y, 1);
      % Snap to nx, ny (from above)
      % (Necessary for rounding errors)
      x = min(x, dem.zmin.nx);
      y = min(y, dem.zmin.ny);
      % x-direction
      if direction(1) > 0
        stepX = 1;
        tDeltaX = dem.zmin.dx / direction(1);
        tMaxX = tmin + (dem.zmin.min(1) + x * dem.zmin.dx - start(1)) / direction(1);
      elseif direction(1) < 0
        stepX = -1;
        tDeltaX = - dem.zmin.dx / direction(1);
        tMaxX = tmin + (dem.zmin.min(1) + (x - 1) * dem.zmin.dx - start(1)) / direction(1);
      else
        stepX = 0;
        tDeltaX = tmax;
        tMaxX = tmax;
      end
      % y-direction
      if direction(2) > 0
        stepY = 1;
        tDeltaY = dem.zmin.dy / direction(2);
        tMaxY = tmin + (dem.zmin.min(2) + y * dem.zmin.dy - start(2)) / direction(2);
      elseif direction(2) < 0
        stepY = -1;
        tDeltaY = - dem.zmin.dy / direction(2);
        tMaxY = tmin + (dem.zmin.min(2) + (y - 1) * dem.zmin.dy - start(2)) / direction(2);
      else
        stepY = 0;
        tDeltaY = tmax;
        tMaxY = tmax;
      end
      % Traverse grid
      % voxels = [];
      z_in = start(3);
      xpos = diff(dem.zmin.xlim) > 0;
      ypos = diff(dem.zmin.ylim) > 0;
      while (x <= dem.zmin.nx) && (x >= 1) && (y <= dem.zmin.ny) && (y >= 1)
        z_out = origin(3) + min(tMaxY, tMaxX) * direction(3);
        % Flip indices as needed
        % TODO: Avoid by fixing origin in Z matrices
        if xpos
          xi = x;
        else
          xi = dem.zmin.nx - x + 1;
        end
        if ypos
          yi = y;
        else
          yi = dem.zmin.ny - y + 1;
        end
        % Test for intersection
        if ~(isnan(dem.zmax.Z(yi, xi)) || isnan(dem.zmin.Z(yi, xi))) && ~((z_in > dem.zmax.Z(yi, xi) && z_out > dem.zmax.Z(yi, xi)) || (z_in < dem.zmin.Z(yi, xi) && z_out < dem.zmin.Z(yi, xi)))
          sqi = sub2ind(size(dem.Z), [yi + 1; yi; yi; yi + 1], [xi; xi; xi + 1; xi + 1]);
          square = [dem.X(sqi), dem.Y(sqi), dem.Z(sqi)];
          tri1 = square([1 2 3], :);
          tri2 = square([3 4 1], :);
          [intersection, ~, ~, t] = rayTriangleIntersection(origin, direction, tri1(1, :), tri1(2, :), tri1(3, :));
          if ~intersection
            [intersection, ~, ~, t] = rayTriangleIntersection(origin, direction, tri2(1, :), tri2(2, :), tri2(3, :));
          end
          if intersection
            X(end + 1, :) = origin + t * direction;
            if first
              break
            end
          end
        end
        if x == mx && y == my
          % voxels(end + 1, :) = [xi, yi];
          break
        end
        % return voxels
        % voxels(end + 1, :) = [xi, yi];
        if tMaxX < tMaxY
          tMaxX = tMaxX + tDeltaX;
          x = x + stepX;
        else
          tMaxY = tMaxY + tDeltaY;
          y = y + stepY;
        end
        z_in = z_out;
      end
      if isempty(X) && first
        X = nan(1, 3);
      end
    end

    function in = inbounds(dem, xy, on)
      % INBOUNDS Test whether points are within DEM bounds.
      %
      %   in = dem.inbounds(xy, on = true)
      %
      % Inputs:
      %   xy - Point coordinates [x1 y1; ... ; xn yn]
      %   on - Whether to include points on bound edges

      box = [sort(dem.xlim), sort(dem.ylim)];
      if nargin < 3 || on
        in = xy(:, 1) >= box(1) & xy(:, 1) <= box(2) & xy(:, 2) >= box(3) & xy(:, 2) <= box(4);
      else
        in = xy(in, 1) > box(1) & xy(in, 1) < box(2) & xy(in, 2) > box(3) & xy(in, 2) < box(4);
      end
    end

    function [xi, yi] = xy2ind(dem, xy)
      % XY2IND Convert XY points to grid indices.
      %
      %   zi = dem.xy2ind(xy)
      %   [xi, yi] = dem.xy2ind(xy)
      %
      % Inputs:
      %   xy - Point coordinates [x1 y1; ... ; xn yn]
      %
      % Outputs:
      %   zi     - Linear indices
      %   xi, yi - Subscript indices in x (cols) and y (rows)

      % --- Filter inbound points ---
      in = dem.inbounds(xy);
      [xi, yi] = deal(nan(size(xy(:, 1))));
      % --- xy indices ---
      xi(in) = ceil((xy(in, 1) - dem.xlim(1)) ./ diff(dem.xlim) * dem.nx);
      yi(in) = ceil((xy(in, 2) - dem.ylim(1)) ./ diff(dem.ylim) * dem.ny);
      % --- Include edges ---
      xi(in & xi == 0) = 1;
      yi(in & yi == 0) = 1;
      % --- Convert to linear indices ---
      if nargout < 2
        xi = sub2ind([dem.ny, dem.nx], yi, xi);
      end
    end

    function [z, xy] = ind2xyz(dem, ind)
      % IND2XYZ Convert indices to XYZ points.
      %
      %   z = dem.ind2xyz(ind)
      %   [z, xy] = dem.ind2xyz(ind)
      %
      % Inputs:
      %   zi     - Linear indices or subscript indices in x (cols) and y (rows)
      %
      % Outputs:
      %   z  - Z coordinates [z1; ...; zn]
      %   xy - XY coordinates [x1 y1; ... ; xn yn]

      % --- Linear indices ---
      if any(size(ind) == 1)
        if nargout > 1
          [yi, xi] = ind2sub([dem.ny dem.nx], ind);
          xy = [dem.x(xi)' dem.y(yi)'];
        end
      % --- Subscript indices ---
      else
        if nargout > 1
          xy = [dem.x(ind(:, 1))' dem.y(ind(:, 2))'];
        end
        ind = sub2ind([dem.ny, dem.nx], ind(:, 2), ind(:, 1));
      end
      ind = reshape(ind, [], 1);
      z = dem.Z(ind);
    end

    function [X, edge] = horizon(dem, xyz, angles)
      if nargin < 3
        angles = (0:1:359)';
      end
      angles = reshape(angles, [], 1);
      n_angles = length(angles);
      directions = [cosd(angles), sind(angles), repmat(0, n_angles, 1)];
      X = nan(n_angles, 3);
      in = nan(n_angles, 1);
      for i = 1:n_angles
        cells = traverseRayDEM([xyz directions(i, :)], dem);
        % Convert to upper-left matrix indices (flip y)
        xi = cells(:, 1);
        yi = dem.ny - (cells(:, 2) - 1);
        % Retrieve true x,y,z based on cell xy
        [z, xy] = dem.ind2xyz([xi yi]);
        elevation = atand((z - xyz(3)) ./ sqrt((xy(:, 1) - xyz(1)).^2 + (xy(:, 2) - xyz(2)).^2));
        [~, i_max] = nanmax(elevation);
        if nargout > 1
          if i_max == length(elevation) || all(isnan(elevation((i_max + 1):end)))
            edge(i) = true;
          else
            edge(i) = false;
          end
        end
        X(i, :) = [xy(i_max, :), z(i_max)];
      end
    end

    function vis = visible(dem, xyz)
      X = dem.X(:) - xyz(1);
      Y = dem.Y(:) - xyz(2);
      Z = dem.Z(:) - xyz(3);
      d = sqrt(X.^2 + Y.^2 + Z.^2);
      x = (atan2(Y, X) + pi) / (pi * 2);
      y = Z ./ d;

      [~, ix] = sortrows([round(sqrt((X / dem.dx).^2 + (Y / dem.dy).^2)) x]); %round

      loopix = find(diff(x(ix)) < 0);
      vis = true(size(X, 1), 1);

      maxd = max(d); % TODO: optimize
      N = ceil(2 * pi / (dem.dx / maxd)); % number of points in voxel horizon

      voxx = (0:N)' / N;
      voxy = zeros(size(voxx)) - Inf;

      for k = 1:length(loopix) - 1
          lp = ix((loopix(k) + 1):loopix(k + 1));
          lp = lp([end 1:end 1]);
          yy = y(lp); xx = x(lp);
          xx(1) = xx(1) - 1; xx(end) = xx(end) + 1; % TODO: why?
          vis(lp(2:end - 1)) = interp1q(voxx, voxy, xx(2:end - 1)) < yy(2:end - 1);
          voxy = max(voxy, interp1q(xx, yy, voxx));
      end
      vis = reshape(vis, [dem.ny dem.nx]);
    end

  end % methods

  methods (Static, Access = private)

    function Z = parse_Z(value)
      if isempty(value) || ~isnumeric(value)
        error('Value must be non-empty and numeric.');
      end
      if size(value, 3) > 1
        warning('Dimensions greater than 2 currently not supported. Using first 2D layer.');
      end
      Z = value(:, :, 1);
    end

    function [xlim, x, X] = parse_xlim(value)
      if isempty(value) || ~isnumeric(value)
        error('Value must be non-empty and numeric.');
      end
      [x, X] = deal([]);
      isgrid = all(size(value) > 1);
      isvector = ~isgrid && any(size(value) > 2);
      if isgrid
        if any(value(1, 1) ~= value(:, 1))
          error('X does not have all equal rows.');
        end
        if nargout > 2
          X = value;
        end
        value = value(1, :);
      end
      if size(value, 1) > 1
        value = reshape(value, 1, length(value));
      end
      if isgrid || isvector
        d = diff(value);
        if any(d(1) ~= d)
          error('x is not equally spaced monotonic.');
        end
        if nargout > 1
          x = value;
        end
        xlim = [value(1) - diff(value(1:2)) / 2, value(end) + diff(value(end-1:end)) / 2];
      else
        xlim = value;
      end
    end

    function [ylim, y, Y] = parse_ylim(value)
      if isempty(value) || ~isnumeric(value)
        error('Value must be non-empty and numeric.');
      end
      [y, Y] = deal([]);
      isgrid = all(size(value) > 1);
      isvector = ~isgrid && any(size(value) > 2);
      if isgrid
        if any(value(1, 1) ~= value(1, :))
          error('Y does not have all equal rows.');
        end
        if nargout > 2
          Y = value;
        end
        value = value(:, 1);
      end
      if size(value, 1) > 1
        value = reshape(value, 1, length(value));
      end
      if isgrid || isvector
        d = diff(value);
        if any(d(1) ~= d)
          error('y is not equally spaced monotonic.');
        end
        if nargout > 1
          y = value;
        end
        ylim = [value(1) - diff(value(1:2)) / 2, value(end) + diff(value(end-1:end)) / 2];
      else
        ylim = value;
      end
    end

  end % static methods

  methods

    function value = compute_zlim(dem)
      value = [min(min(dem.Z)), max(max(dem.Z))];
    end

    function value = compute_min(dem)
      value = [min(dem.xlim), min(dem.ylim), min(dem.zlim)];
    end

    function value = compute_max(dem)
      value = [max(dem.xlim), max(dem.ylim), max(dem.zlim)];
    end

    function value = compute_nx(dem)
      value = size(dem.Z, 2);
    end

    function value = compute_ny(dem)
      value = size(dem.Z, 1);
    end

    function value = compute_dx(dem)
      value = range(dem.xlim) / dem.nx;
    end

    function value = compute_dy(dem)
      value = range(dem.ylim) / dem.ny;
    end

    function value = compute_x(dem)
      if dem.xlim(1) <= dem.xlim(2)
        value = (min(dem.xlim) + dem.dx / 2):dem.dx:(max(dem.xlim) - dem.dx / 2);
      else
        value = (max(dem.xlim) - dem.dx / 2):-dem.dx:(min(dem.xlim) + dem.dx / 2);
      end
    end

    function value = compute_y(dem)
      if dem.ylim(1) <= dem.ylim(2)
        value = (min(dem.ylim) + dem.dy / 2):dem.dy:(max(dem.ylim) - dem.dy / 2);
      else
        value = (max(dem.ylim) - dem.dy / 2):-dem.dy:(min(dem.ylim) + dem.dy / 2);
      end
    end

    function value = compute_X(dem)
      value = repmat(dem.x, dem.ny, 1);
    end

    function value = compute_Y(dem)
      value = repmat(dem.y', 1, dem.nx);
    end

    function value = compute_zmin(dem)
      % --- Compute Z min ---
      Zmin = ordfilt2(dem.Z, 1, ones(2, 2));
      % Strip padded rows and columns
      Zmin = Zmin(1:(end - 1), 1:(end - 1));
      % --- Build offset DEMs ---
      % Trim boundaries to outer center coordinates
      value = DEM(Zmin, dem.x([1, end]), dem.y([1, end]));
    end

    function value = compute_zmax(dem)
      % --- Compute Z max ---
      Zmax = ordfilt2(dem.Z, 4, ones(2, 2));
      % Strip padded rows and columns
      Zmax = Zmax(1:(end - 1), 1:(end - 1));
      % --- Build offset DEMs ---
      % Trim boundaries to outer center coordinates
      value = DEM(Zmax, dem.x([1, end]), dem.y([1, end]));
    end

  end % private methods

end % classdef
