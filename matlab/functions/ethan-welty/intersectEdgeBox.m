function points = intersectEdgeBox(edge, box)
  % INTERSECTEDGEBOX Find intersections of an edge with a box.
  %
  %   points = intersectEdgeBox(edge, box)
  %
  % Inputs:
  %   edge - [x1, y1, (z1), x2, y2, (z2)]
  %   box  - [xmin, ymin, (zmin), xmax, ymax, (zmax))]
  %
  % Outputs:
  %   points - [xi, yi, (zi), ...]
  %
  % See also intersectRayBox

  dim = numel(edge) / 2;
  origin = edge(1:dim);
  direction = edge((dim + 1):end) - origin;
  boxmin = box(1:dim);
  boxmax = box((dim + 1):end);
  [tmin tmax] = intersectRayBox(origin, direction, boxmin, boxmax);
  points = [];
  for t = [tmin, tmax]
    if t > 0 & t < 1
      points = vertcat(points, origin + direction * t);
    end
  end
end
