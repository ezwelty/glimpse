function [voxels, start, stop] = traverseRayDEM(ray, dem)
  % TRAVERSERAYDEM Find the DEM grid cells traversed by a ray.
  %
  %   cells = traverseRayDEM(ray, dem)
  %
  % Traversal algorithm through a 2D grid, adapted from the 3D algorithm
  % proposed by Amanatides and Woo (1987).
  %
  %
  % Input:  ray     origin and direction of ray [x y z dx dy dz]
  %       dem     DEM structure
  %
  % Output:   cells     DEM cells intersected by the ray
  %       start     entrance point
  %       stop    exit point
  %
  % See also intersectRayBox, intersectRayDEM.
  %
  % Adapted to 2D from code by Jesus Mena (MATLAB Central #26834).

  % Check number of arguments
  if (nargin < 2)
    error('Specify two input arguments.')
  end

  % Test size of arguments
  if (length(ray) ~= 6)
    error('Ray must have length = 6.')
  end

  % Decompose input
  origin = ray(1:3);
  direction = ray(4:6);

  % Test for ray - box intersection
  boxmin = [min(dem.xlim), min(dem.ylim), min(dem.zlim)];
  boxmax = [max(dem.xlim), max(dem.ylim), max(dem.zlim)];
  [tmin, tmax] = intersectRayBox(origin, direction, boxmin, boxmax);

  % Calculate voxels (2D)
  if ~isempty(tmin)

    % Compute endpoints of ray within grid
    start = origin + tmin * direction;
    stop = origin + tmax * direction;

    % Find starting voxel coordinates
    x = ceil((start(1) - min(dem.xlim)) / dem.dx);
    y = ceil((start(2) - min(dem.ylim)) / dem.dy);
    % Snap to 1 (from 0)
    if x == 0
      x = 1;
    end
    if y == 0
      y = 1;
    end
    % Snap to nx, ny (from above)
    % TODO: Necessary for rounding errors?
    if x == dem.nx + 1
      x = dem.nx;
    end
    if y == dem.ny + 1
      y = dem.ny;
    end

    % Set x,y increments based on ray slope
    if direction(1) >= 0
      % Increasing x
      tCellX = x / dem.nx;
      stepX = 1;
    else
      % Decreasing x
      tCellX = (x - 1) / dem.nx;
      stepX = -1;
    end
    if direction(2) >= 0
      % Increasing y
      tCellY = y / dem.ny;
      stepY = 1;
    else
      % Decreasing y
      tCellY = (y-1) / dem.ny;
      stepY = -1;
    end

    % TODO: ?
    boxSize = boxmax - boxmin;
    cellMaxX = min(dem.xlim) + tCellX * boxSize(1);
    cellMaxY = min(dem.ylim) + tCellY * boxSize(2);

    % Compute values of t at which ray crosses vertical, horizontal voxel boundaries
    tMaxX = tmin + (cellMaxX - start(1)) / direction(1);
    tMaxY = tmin + (cellMaxY - start(2)) / direction(2);

    % Width and height of voxel in t
    tDeltaX = dem.dx / abs(direction(1));
    tDeltaY = dem.dy / abs(direction(2));

    % Find ending voxel coordinates
    mx = ceil((stop(1) - min(dem.xlim)) / dem.dx);
    my = ceil((stop(2) - min(dem.ylim)) / dem.dy);

    % Return list of traversed voxels
    voxels = [];
    while (x <= dem.nx) && (x >= 1) && (y <= dem.ny) && (y >= 1)

      if x == mx && y == my
        voxels(end + 1, :) = [x y];
        break
      end

      % return voxels
      voxels(end + 1, :) = [x y];

      if tMaxX < tMaxY
        tMaxX = tMaxX + tDeltaX;
        x = x + stepX;
      else
        tMaxY = tMaxY + tDeltaY;
        y = y + stepY;
      end
    end

  else
    % Return empty
    voxels = [];
    start = [];
    stop = [];
  end
end
