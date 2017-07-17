function points = polygon2grid(polygon, dx, dy, on)
  % POLYGON2GRID Grid of points inside polygon.
  %
  %   points = polygon2grid(polygon, dx, dy, on = false)

  % TODO: Buffer argument to remove points close to edge

  if nargin < 4 || isempty(on)
    on = false;
  end
  x = min(polygon(:, 1)):dx:max(polygon(:, 1));
  y = min(polygon(:, 2)):dy:max(polygon(:, 2));
  [X Y] = meshgrid(x, y);
  [in, on] = inpolygon(X, Y, polygon(:, 1), polygon(:, 2));
  if ~on
    in = in & ~on;
  end
  points = [X(in) Y(in)];
end
