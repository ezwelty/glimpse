function xyz = intersectRayPlane(varargin)
  % INTERSECTRAYPLANE  Find the intersections of rays with an (infinite) plane.
  %
  %   xyz = intersectRayPlane(rays, plane)
  %   xyz = intersectRayPlane(P0, v, plane)
  %
  % Finds the intersections of rays with a plane using the algebraic method.
  %
  % Inputs:
  %   rays  - Origin and direction of rays [x y z dx dy dz; ...]
  %   P0    - Shared point of origin [x y z]
  %   v     - Direction vectors [dx dy dz; ...]
  %   plane - Plane [a b c d], where ax + by + cz + d = 0
  %
  % Outputs:
  %   xyz   - Coordinates of the intersection points (NaN if none) [x y z; ...]

  if (nargin < 2)
    error('Specify at least 2 input arguments.')
  end

  % Compute intersections
  % plane: ax + by + cz + d = 0, normal vector n = [a b c]
  % ray: P0 + t * v
  % intersect at t = - (P0 · n + d) / (v · n), if t >= 0

  if nargin == 2 % unique starting points
    P0 = varargin{1}(:, 1:3);
    v = varargin{1}(:, 4:6);
    n = varargin{2}(1:3);
    d = varargin{2}(4);
    t = - (n(1) * P0(:, 1) + n(2) * P0(:, 2) + n(3) * P0(:, 3) + d) ./ (n(1) * v(:, 1) + n(2) * v(:, 2) + n(3) * v(:, 3));
    xyz = nan(size(v, 1), 3);
    i = t >= 0;
    xyz(i, :) = bsxfun(@plus, bsxfun(@times, t(i), v(i, :)), P0(i, :));
  end

  if nargin > 2 % shared starting point (faster)
    P0 = varargin{1};
    v = varargin{2};
    n = varargin{3}(1:3);
    d = varargin{3}(4);
    t = - (dot(n, P0) + d) ./ (n(1) * v(:, 1) + n(2) * v(:, 2) + n(3) * v(:, 3));
    xyz = nan(size(v, 1), 3);
    i = t >= 0;
    xyz(i, :) = bsxfun(@plus, bsxfun(@times, t(i), v(i, :)), P0);
  end
