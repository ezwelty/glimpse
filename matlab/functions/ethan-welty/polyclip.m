function c = polyclip(p, l)
  n_pts = size(l, 1);
  n_edges = n_pts - 1;
  c_in = {};
  c_out = {};
  in = [];
  % [xi, yi, ii] = polyxpoly(p(:, 1), p(:, 2), l(:, 1), l(:, 2)); % BROKEN: tolerance errors
  Xi = intersectPolylines(p, l);
  ii = nan(size(Xi));
  for i_x = 1:size(Xi, 1)
    ii(i_x, 1) = projPointOnPolygon(Xi(i_x, :), p);
    ii(i_x, 2) = projPointOnPolyline(Xi(i_x, :), l);
  end
  xi = Xi(:, 1); yi = Xi(:, 2);
  % Sort intersects by line edge
  [~, si] = sort(ii(:, 2));
  xi = xi(si); yi = yi(si); ii = ii(si, :);
  % Remove duplicate intersects
  [~, ui] = unique([xi yi], 'rows', 'stable');
  xi = xi(ui); yi = yi(ui); ii = ii(ui, :);
  % Convert edge positions to edge indices
  ii_pos = ii;
  ii = floor(ii_pos) + 1;
  % For each cutline...
  n_clines = length(xi) - 1;
  for i_cline = 1:n_clines
    % Skip parallel overlaps (by testing first interior edge of each)
    ci = i_cline:(i_cline + 1);
    p_ends = p([min(ii(ci, 1)), min(ii(ci, 1)) + 1], :);
    l_ends = l([min(ii(ci, 2)), min(ii(ci, 2)) + 1], :);
    if isParallel(diff(p_ends), diff(l_ends))
      continue
    end
    % Collect line vertices between intersects
    i_pts = (floor(ii_pos(ci(1), 2)) + 2):(ceil(ii_pos(ci(2), 2)));
    % if ii(ci(1), 2) == ii(ci(2), 2)
    %   i_pts = [];
    % else
    %   i_pts = (ii(i_cline, 2) + 1):ii(i_cline + 1, 2);
    % end
    % Append to intersects
    cline = [xi(ci(1)) yi(ci(1)) ; l(i_pts, :) ; xi(ci(2)) yi(ci(2))];
    % If starting, determine whether in or out
    if isempty(in)
      in = all(inpolygon(cline(1:2, 1), cline(1:2, 2), p(:, 1), p(:, 2)));
    % Otherwise, alternate
    else
      in = ~in;
    end
    % Cut polygons
    if ~in
      % Form outside polygon adjacent to original polygon
      c = polyclip_single(p, cline, ii(i_cline:(i_cline + 1), 1), in);
      c_out{length(c_out) + 1} = c{1};
    else
      if length(c_in) == 0
        % Inside polygons formed by cutting original polygon
        c = polyclip_single(p, cline, ii(i_cline:(i_cline + 1), 1), in);
        c_in = {c{1} c{2}};
      else
        % Detect intersecting inside polygon, then cut it
        for i_poly = 1:length(c_in)
          if inpolygon(cline(1, 1), cline(1, 2), c_in{i_poly}(:, 1), c_in{i_poly}(:, 2))
            [~, ~, nii] = polyxpoly(c_in{i_poly}(:, 1), c_in{i_poly}(:, 2), cline([1 1 end end], 1), cline([1 1 end end], 2));
            c = polyclip_single(c_in{i_poly}, cline, nii([1 end], 1), in);
            % Replace cut polygon with resulting polygons
            c_in = [c_in([1:(i_poly - 1), (i_poly + 1):length(c_in)]) c{1} c{2}];
            break
          end
        end
      end
    end
  end
  if length(c_in) == 0
    c_in{1} = p;
  end
  c = {c_in, c_out};
  for i = 1:length(c)
    for j = 1:length(c{i})
      [x, y] = poly2cw(c{i}{j}(:, 1), c{i}{j}(:, 2));
      c{i}{j} = [x, y];
    end
  end
end

% l must start and end on polygon boundary
% p must be closed
function c = polyclip_single(p, l, ii, in)
  n_pts = size(l, 1);
  n_edges = n_pts - 1;
  c = {};
  if ii(2) > ii(1)
    ip1 = ii(2):-1:(ii(1) + 1);
  else
    ip1 = (ii(2) + 1):1:ii(1);
  end
  c1 = [l ; p(ip1, :) ; l(1, :)];
  keep = [true ; sum(abs(diff(c1)), 2) > 0];
  c{length(c) + 1} = c1(keep, :);
  if in
    if ii(2) > ii(1)
      ip2 = [(ii(2) + 1):1:(size(p, 1) - 1) 1:1:ii(1)]; % assumes closed polygon
    else
      ip2 = [ii(2):-1:1 (size(p, 1)-1):-1:(ii(1) + 1)]; % assumes closed polygon
    end
    c2 = [l ; p(ip2, :) ; l(1, :)];
    keep = [true ; sum(abs(diff(c2)), 2) > 0];
    c{length(c) + 1} = c2(keep, :);
  end
end

  % # Assign direction to each polygon feature
  %  polygons = arcpy.UpdateCursor(diffshp)
  %  for row in polygons:
  %    # calculate centroid
  %    cpt = arcpy.PointGeometry(row.shape.centroid)
  %    # Find which side centroid is to first terminus
  %    # +1 if right of, -1 if left of
  %    vertices = arcpy.management.SplitLineAtPoint(lines[i], cpt, arcpy.Geometry())[1].getPart(0)
  %    Ax = vertices[0].X
  %    Ay = vertices[0].Y
  %    Bx = vertices[1].X
  %    By = vertices[1].Y
  %    Px = cpt.firstPoint.X
  %    Py = cpt.firstPoint.Y
  %    dot = ((Ay - By) * (Px - Ax)) + ((Bx - Ax) * (Py - Ay))
  %    if (dot < 0):
  %      row.TYPE = 1
  %    if (dot > 0):
  %      row.TYPE = -1
  %    if (dot == 0):
  %      row.TYPE = 0
  %      print "Warning: found coincident centroid!"
  %    polygons.updateRow(row)

  % % Plotting
  % %p = [0 1; 1 1; 1 0; 0 0; 0 1]; % intersect spans polygon closure
  % p = [0 0; 0 1; 0.25 1; 0.25 1.25; 0.5 1.25; 0.5 1; 1 1; 1 0; 0 0];
  % %l = [0 1; 0 2; 1 2; 1 1;]; % out only
  % %l = [-0.5 0.5; 0.5 0.5; 0.5 1.5; 0.75 1.5; 0.75 0.5; 1.5 0.5];
  % %l = [-0.5 0.5; 0.5 0.5; 0.5 1.5; 0.25 1.5; 0.25 0.75; -1.5 0.75; -1.5 0.25; 0.5 0.25; 1.5 0.25];
  % %l = [-1 0.25; 0.5 0.25; 1.5 0.25];
  % %l = [-0.5 0.5; 0.5 0.5; 0.5 1.5; 0.25 1.5; 0.25 0.75; -1 0.75; -1 0.25; 0 0.25];
  % l = [-0.5 1; 0.5 1; 0.5 0];
  % c = polyclip(p, l);
  % plot(p(:, 1), p(:, 2), 'k-', l(:, 1), l(:, 2), 'k-'); hold on
  % % Xi = intersectPolylines(p, l);
  % % plot(Xi(:, 1), Xi(:, 2), 'r*')
  % for i_poly = 1:length(c{1})
  %   patch(c{1}{i_poly}(:, 1), c{1}{i_poly}(:, 2), 1, 'facecolor', 'g')
  %   % patch(c{1}{i_poly}(:, 1), c{1}{i_poly}(:, 2), i_poly)
  % end
  % for i_poly = 1:length(c{2})
  %   patch(c{2}{i_poly}(:, 1), c{2}{i_poly}(:, 2), 1, 'facecolor', 'r')
  %   % patch(c{2}{i_poly}(:, 1), c{2}{i_poly}(:, 2), i_poly + length(c{1}))
  % end
  % plot(p(:, 1), p(:, 2), 'k-', l(:, 1), l(:, 2), 'k:');
