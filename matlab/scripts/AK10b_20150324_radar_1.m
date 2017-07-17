%% Velocity from Time-lapse
% Images from a time-lapse camera are calibrated, aligned, then used to
% estimate glacier motion by projecting feature tracks onto a digital
% elevation model (DEM).

%% Assign constants

%%
% Paths
IMGRAFT_PATH = '.';
PROJECT = 'AK10b_20150324_radar_1';
DATA_DIR = fullfile('demos', 'data');
IMG_PATH = fullfile(DATA_DIR, 'images', PROJECT, '*.JPG');
GLACIER_DEM_PATH = fullfile(DATA_DIR, 'dems', 'SETSM_WV01_20150423_102001003C038400_102001003C64D800_seg1_2m_v1.0_dem_proj.tif');
HORIZON_DEM_PATH = GLACIER_DEM_PATH;
OUT_DIR = fullfile('demos', PROJECT);
addpath(genpath(IMGRAFT_PATH));

%%
% Parameters
HORIZON_DEM_SCALE = 0.25;
HORIZON_DDEG = 0.05; % deg
GLACIER_DEM_SCALE = 0.25;
GLACIER_DEM_SMOOTH_WIDTH = 300; % m
COAST_HAE = 17; % m, FIXME: Replace with timeseries
CAM_ID = lower('AK10b');
SVG_GCP = 'gcp';
SVG_COAST = 'coast';
SVG_HORIZON = 'horizon';
SVG_LAND = 'land';
SVG_GLACIER = 'glacier';
SVG_LINE_DENSITY = 0.05; % points per pixel
LDMAX = 25; % pixels (max distance to assign when calibrating with lines)
% IMG_SCALE = 1;
% HORIZON_DEM_DISTANCE = 20e3; % m
% GLACIER_DEM_DISTANCE = 5e3; % m

%%
% Datasets
CAM = readtable(fullfile(DATA_DIR, 'cameras.csv'));
CAM.Properties.RowNames = lower(CAM.id);
GCP = readtable(fullfile(DATA_DIR, 'points.csv'));
GCP.Properties.RowNames = lower(GCP.name);
COAST = shaperead(fullfile(DATA_DIR, 'lines', 'coastline.shp'));
COAST = arrayfun(@(s) [s.X' s.Y' repmat(COAST_HAE, length(s.X), 1)], COAST, 'uniform', false);

%% Prepare variables

%%
% Initialize the camera with a surveyed position and estimated direction.
xyz = CAM{CAM_ID, {'x_wgs84', 'y_wgs84', 'z_hae'}};
viewdir = CAM{CAM_ID, {'yaw', 'pitch', 'roll'}};
cam = Camera('xyz', xyz, 'viewdir', viewdir);

%%
% Load images and SVG markup from file.
%
%  gcp . <name> : [x y]
%  land . polygon_<i> : [x1 y1; ... ; xn yn]
%  glacier . polygon_<i> : [x1 y1; ... ; xn yn]
%  coast . polyline_<i> : [x1 y1; ... ; xn yn]
%  horizon . polyline_<i> : [x1 y1; ... ; xn yn]
%
%  FIXME: xmlread (for svg) very slow.
%
images = Image(IMG_PATH, cam);

%%
% Format ground control points (GCP).
%
%  gcp.uv : [u1 v1; ... ]
%  gcp.xyz : [x1 y1 z1; ... ]
%
has_gcp = find(arrayfun(@(img) isfield(img.svg, SVG_GCP), images));
for i_img = has_gcp
  gcp_names = fieldnames(images(i_img).svg.gcp);
  images(i_img).gcp.uv = cell2mat(struct2cell(images(i_img).svg.gcp));
  images(i_img).gcp.xyz = GCP{lower(gcp_names), {'x_wgs84', 'y_wgs84', 'z_hae'}};
end

%%
% Load digital elevation model (DEM) from file, crop to camera, and smooth
% to fill crevasses.
%
%  FIXME: Cannot crop by camera view when missing initial estimate
%  TODO: Build and use canonical regional DEM for horizon detection
%
HORIZON_DEM = DEM(HORIZON_DEM_PATH).crop([], [], [0 Inf]).resize(HORIZON_DEM_SCALE).build();
% box = images(1).cam.viewbox(HORIZON_DEM_DISTANCE);
% HORIZON_DEM = DEM(HORIZON_DEM_PATH).crop(box(:, 1), box(:, 2), [0 Inf]).resize(HORIZON_DEM_SCALE).build();
% box = images(1).cam.viewbox(GLACIER_DEM_DISTANCE);
% GLACIER_DEM = DEM(GLACIER_DEM_PATH).crop(box(:, 1), box(:, 2), [0 Inf]).resize(GLACIER_DEM_SCALE).build();
% SMOOTH_GLACIER_DEM = GLACIER_DEM.smooth(GLACIER_DEM_SMOOTH_WIDTH).build();
GLACIER_DEM = DEM(GLACIER_DEM_PATH).crop([], [], [0 Inf]).resize(GLACIER_DEM_SCALE).build();
SMOOTH_GLACIER_DEM = GLACIER_DEM.smooth(GLACIER_DEM_SMOOTH_WIDTH);

%%
% Compute the world coordinates of the horizon as seen from this camera.
%
%  FIXME: Slow
%  TODO: Cache results by CAM_ID
%  TODO: Traverse grid in concentric circles or triangular wedges
%
if isempty(cam.viewdir)
  angles = 0:HORIZON_DDEG:(360 - HORIZON_DDEG);
  HORIZON = HORIZON_DEM.horizon(cam.xyz, angles);
else
  HORIZON = images(1).cam.horizon(HORIZON_DEM, HORIZON_DDEG);
end

%%
% Compile control lines (same for all images).
LXYZ = [COAST; HORIZON];

%%
% Format "ground control lines" (GCL).
%
%  gcl.uv : [u1 v1; ... ] (as point matrix)
%  gcl.xyz : {[x1 y1 z1; ... ], ... } (as polyline array)
%
has_lines = find(arrayfun(@(img) any(isfield(img.svg, {SVG_COAST, SVG_HORIZON})), images));
for i = has_lines

  %%
  % Collect line traces from SVG data.
  [coast, horizon] = deal({});
  if isfield(images(i).svg, SVG_COAST)
    coast = struct2cell(images(i).svg.(SVG_COAST));
  end
  if isfield(images(i).svg, SVG_HORIZON)
    horizon = struct2cell(images(i).svg.(SVG_HORIZON));
  end
  lines = [coast; horizon];

  %%
  % Sample points along lines.
  for j = 1:length(lines)
    line_length = polylineLength(lines{j});
    n_points = round(line_length * SVG_LINE_DENSITY);
    lines{j} = resamplePolyline(lines{j}, n_points);
  end
  images(i).gcl.uv = cell2mat(lines);

  %%
  % Attach world coordinates of lines (now only used by Image.plot).
  images(i).gcl.xyz = LXYZ;
end

%%
% Format "fixed" (land) and "free" (glacier) polygons.
%
%  freepolys = {[x1 y1 z1; ...], ...}
%  fixedpolys = {[x1 y1 z1; ...], ...}
%
for i = 1:length(images)
  if isfield(images(i).svg, SVG_LAND)
    for j = fieldnames(images(i).svg.(SVG_LAND))'
      images(i).fixedpolys{end + 1} = images(i).svg.(SVG_LAND).(j{1});
    end
  end
  if isfield(images(i).svg, SVG_GLACIER)
    for j = fieldnames(images(i).svg.(SVG_GLACIER))'
      images(i).freepolys{end + 1} = images(i).svg.(SVG_GLACIER).(j{1});
    end
  end
end

%% Calibrate anchor images
% Use available ground control points and lines to optimize all anchor
% images at once.
% 
%  FIXME: Very slow when solving many parameters with line control
%
is_anchor = arrayfun(@(img) any(~isempty(img.gcp.xyz) & ~isempty(img.gcl.xyz)), images);
anchor_ind = find(is_anchor);
flexparams = {'viewdir'};
fixparams = {'f', 'k', 1, 'c', 'p', 1};
[anchors, fit] = Camera.optimize_images(images(is_anchor), flexparams, fixparams, LDMAX, LXYZ)

%%
% Plot the results.
for i = 1:length(anchors)
  figure();
  anchors(i).plot(true);
  title(['\fontsize{14} ', 'Image #', num2str(anchor_ind(i)), ': RMSE ', num2str(fit.rmse(i), '%.2f'), 'px']);
  legend('ground control point', 'projection error', 'ground control line', 'projection error', 'projection')
end

%%
% Save the results.
images(is_anchor) = anchors;

%% Orient images relative to anchors
% Assign each image to the nearest (temporal) anchor.
anchor_dates = [images(is_anchor).date_num];
for i = find(~is_anchor)
  [~, i_min] = min(abs(anchor_dates - images(i).date_num));
  images(i).anchor = anchor_ind(i_min);
end

%%
% Match features between image and anchor, filter with RANSAC, and optimize orientation.
%
%  TODO: Support variable image size
%
for i = find(~is_anchor)
  I = images(i).read();
  i0 = images(i).anchor;
  I0 = images(i0).read();

  %%
  % Generate grid of points in land polygons and match between images.
  %
  %  FIXME: Assumes a maximum motion between images
  %  TODO: Move point generation to anchor-level loop
  %
  [gdu, gdv] = deal(10);
  pts = [];
  for j = 1:length(images(i0).fixedpolys)
    pts = [pts; polygon2grid(images(i0).fixedpolys{j}, gdu, gdv)];
  end
  [du, dv, correlation, signal, pu, pv] = templatematch(I0, I, pts(:, 1), pts(:, 2), 'templatewidth', gdu, 'searchwidth', 2 * gdu, 'method', 'NCC', 'super', 2);
  is_strong = correlation > quantile(correlation, 0.8) & signal > quantile(signal, 0.8);
  matches = horzcat(pu, pv, du, dv);
  matches = matches(is_strong, :);

  % Alternatively, load manual matches from SVG
  % [~, base, ~] = fileparts(images(i0).file);
  % pts = cell2mat(struct2cell(images(i).svg.(base)));
  % uv1 = pts(1:2:size(pts, 1), :);
  % uv0 = pts(2:2:size(pts, 1), :);
  % matches = [uv0, uv1 - uv0];

  %%
  % Filter matches with RANSAC.
  %
  %  FIXME: Assumes images are the same size and camera
  %  TODO: Express threshold in pixels
  %
  xy0 = images(i0).cam.image2camera(matches(:, 1:2));
  xy = images(i0).cam.image2camera(matches(:, 1:2) + matches(:, 3:4));
  [F, in] = ransacfitfundmatrix(xy0', xy', 1e-6);
  mean_motion = mean(sqrt(sum(matches(in, 3:4).^2, 2)));

  %%
  % Plot filtered matches.
  figure
  imshow(I0 / 1.5), hold on
  s = 10;
  quiver(matches(:, 1), matches(:, 2), s * matches(:, 3), s * matches(:, 4), 0, 'r');
  quiver(matches(in, 1), matches(in, 2), s * matches(in, 3), s * matches(in, 4), 0, 'y');
  title(['\fontsize{14} ', 'Matches: Image #', num2str(i0), ' -> ', num2str(i)]);
  legend('all matches', 'filtered matches')

  %%
  % Orient camera
  %
  %  FIXME: Assumes images are the same size and camera
  %
  uv = matches(in, 1:2) + matches(in, 3:4);
  xyz = images(i0).cam.xyz + images(i0).cam.invproject(matches(in, 1:2));
  [newcams, fit] = Camera.optimize_bundle(images(i0).cam, uv, xyz, 'viewdir')
  images(i).cam = newcams{1};

  %%
  % Plot results
  figure
  imshow(I0 / 1.5), hold on
  uv0 = matches(in, 1:2);
  duv = matches(in, 3:4);
  s = 10;
  quiver(uv(:, 1), uv(:, 2), -s * duv(:, 1), -s * duv(:, 2), 0, 'r');
  puv = images(i).cam.project(xyz);
  duv = puv - uv;
  quiver(uv(:, 1), uv(:, 2), -s * duv(:, 1), -s * duv(:, 2), 0, 'g');
  title(['\fontsize{14} ', 'Orientation for Image #', num2str(i), ': RMSE ', num2str(fit.rmse(1), '%.2f'), 'px']);
  legend('filtered matches', 'optimized matches')

  %%
  % Transform free polygons.
  for j = 1:length(images(i0).freepolys)
    images(i).freepolys{j} = images(i).cam.project(images(i0).cam.invproject(images(i0).freepolys{j}), true);
  end
end

%%
% Save aligned images to file.
%
%  TODO: Also undistort images
%  TODO: Support variable image sizes
%
mkdir(OUT_DIR, 'aligned');
i0 = 1;

%%
% Copy target image.
[~, filename, ext] = fileparts(images(i0).file);
outfile = fullfile(OUT_DIR, 'aligned', [num2str(i0), '-', num2str(i0), '-', filename, ext]);
copyfile(images(i0).file, outfile);

%%
% Project other images to target and save to file.
dxyz = [];
for i = setdiff(1:length(images), i0)
  if isempty(dxyz)
    [I0, dxyz] = images(i).project(images(i0).cam);
  else
    I0 = images(i).project(images(i0).cam, dxyz);
  end
  [~, filename, ext] = fileparts(images(i).file);
  outfile = fullfile(OUT_DIR, 'aligned', [num2str(i0), '-', num2str(i), '-', filename, ext]);
  imwrite(I0, outfile, 'Quality', 95);
end

%% Track moving features
% Match features between consecutive image pairs using the Farneback
% optical flow method:
% <https://www.mathworks.com/help/vision/ref/opticalflowfarneback-class.html>.
%
%  TODO: Set pyramid levels based on expected motion.
%
obj = opticalFlowFarneback('NumPyramidLevels', 3, 'NeighborhoodSize', 3, 'FilterSize', 25);
flow = cell(length(images));
for i = 1:length(images)
  I = rgb2gray(images(i).read());
  flow{i} = obj.estimateFlow(I);
end
% Plot results
% for i = 2:length(images)
%   images(i).plot();
%   hold on;
%   plot(flow{i}, 'Decimation', [20, 20], 'Scale', 20);
%   pause;
% end

%%
% Generate a regular grid of glacier points.
[gdx, gdy] = deal(10);
gxyz = [];
i0 = 1;
for poly = images(i0).freepolys
  xyz = images(i0).cam.invproject(poly{:});
  for i_pt = 1:size(xyz, 1)
    xyz(i_pt, :) = GLACIER_DEM.sample_ray_tri(images(i0).cam.xyz, xyz(i_pt, :), true, 500);
  end
  xyz(any(isnan(xyz), 2), :) = [];
  gxy = polygon2grid(xyz, gdx, gdy);
  z = SMOOTH_GLACIER_DEM.sample_points_tri(gxy);
  gxyz0 = [gxy, z];
  gxyz0(isnan(z) | z < COAST_HAE, :) = [];
  uv = images(i0).cam.project(gxyz0);
  gxyz1 = images(i0).cam.invproject(uv);
  for i_pt = 1:size(gxyz1, 1)
    gxyz1(i_pt, :) = SMOOTH_GLACIER_DEM.sample_ray_tri(images(i0).cam.xyz, gxyz1(i_pt, :), true, 1);
  end
  visible = sqrt(sum((gxyz1 - gxyz0).^2, 2)) < 1;
  gxyz = [gxyz; [gxyz0(visible, :)]];
end

% Plot glacier points on map.
% figure()
% GLACIER_DEM.plot(2);
% hold on
% plot(xyz(:, 1), xyz(:, 2), 'y-');
% plot(gxyz(:, 1), gxyz(:, 2), 'r.');

%%
% Plot glacier points on image.
figure()
images(i0).plot(false, true);
hold on
guv = images(i0).cam.project(gxyz);
plot(guv(:, 1), guv(:, 2), 'y.');

%%
% Sample motion at glacier points.
motion = struct();
for i0 = 1:(length(images) - 1)
  i = i0 + 1;

  %%
  % Calculate starting positions (i0).
  uv0 = images(i0).cam.project(gxyz);
  [u, v] = meshgrid(0.5:(images(i0).cam.imgsz(1) - 0.5), 0.5:(images(i0).cam.imgsz(2) - 0.5));
  du = interp2(u, v, flow{i}.Vx, uv0(:, 1), uv0(:, 2), '*cubic');
  dv = interp2(u, v, flow{i}.Vy, uv0(:, 1), uv0(:, 2), '*cubic');

  %%
  % Calculate ending positions (i).
  uv = uv0 + [du, dv];
  xyz = images(i).cam.invproject(uv);
  for i_pt = 1:size(xyz, 1)
    xyz(i_pt, :) = SMOOTH_GLACIER_DEM.sample_ray_tri(images(i).cam.xyz, xyz(i_pt, :), true, gxyz(i_pt, :), 10);
  end

  %%
  % Store in data structure.
  motion(i0).t0 = images(i0).date_num;
  motion(i0).t = images(i).date_num;
  motion(i0).uv0 = uv0;
  motion(i0).uv = uv;
  motion(i0).xyz0 = gxyz;
  motion(i0).xyz = xyz;

  % I0 = images(i0).read();
  % I = images(i).read();
  % puv = images(i0).cam.project(gxyz);
  % [du, dv, correlation, signal, pu, pv] = templatematch(I0, I, puv(:, 1), pts(:, 2), 'templatewidth', 9, 'searchwidth', 30, 'method', 'NCC', 'super', 2);
  % is_strong = correlation > quantile(correlation, 0.8) & signal > quantile(signal, 0.8);
  % matches = horzcat(pu, pv, du, dv);
  % matches = matches(is_strong, :);
  % motion(i0).flow.uv0 = uv0;
  % motion(i0).flow.uv = uv;
  % motion(i0).flow.xyz0 = gxyz;
  % motion(i0).flow.xyz = xyz;
  % motion(i0).ncc.uv0 = uv0;
  % motion(i0).ncc.uv = uv;
  % motion(i0).ncc.xyz0 = gxyz;
  % motion(i0).ncc.xyz = xyz;
end

%%
% Visualize results.
for i0 = 1:(length(images) - 1)
  [dx, dy] = deal(50);
  vrange = [0, 5];

  %%
  % Plot glacier motion on map.
  %
  %  TODO: Incorporate into DEM.plot function
  %
  figure();
  showimg(GLACIER_DEM.x, GLACIER_DEM.y, hillshade(GLACIER_DEM.Z, GLACIER_DEM.x, GLACIER_DEM.y));
  hold on
  ddays = motion(i0).t - motion(i0).t0;
  v = sqrt(sum((motion(i0).xyz - motion(i0).xyz0).^2, 2)) / ddays;
  [X, Y, V] = pts2grid(motion(i0).xyz0(:, 1), motion(i0).xyz0(:, 2), v, dx, dy);
  alphawarp(X, Y, V, 1);
  caxis(vrange);
  colormap jet;
  colorbar
  [~, ~, DX] = pts2grid(motion(i0).xyz0(:, 1), motion(i0).xyz0(:, 2), motion(i0).xyz(:, 1) - motion(i0).xyz0(:, 1), dx, dy);
  [~, ~, DY] = pts2grid(motion(i0).xyz0(:, 1), motion(i0).xyz0(:, 2), motion(i0).xyz(:, 2) - motion(i0).xyz0(:, 2), dx, dy);
  s = 5;
  quiver(X, Y, s * DX / ddays, s * DY / ddays, 0, 'r');
  xlim([min(min(X)), max(max(X))]);
  ylim([min(min(Y)), max(max(Y))]);
  
  %%
  % Plot glacier motion on image.
  figure();
  ind = 1:size(motion(i0).uv, 1);
  ind(isnan(v(ind))) = [];
  images(i0).plot();
  hold on
  s = 5;
  quiver(motion(i0).uv0(ind, 1), motion(i0).uv0(ind, 2), s * (motion(i0).uv(ind, 1) - motion(i0).uv0(ind, 1)), s * (motion(i0).uv(ind, 2) - motion(i0).uv0(ind, 2)), 0, 'w');
  colors = jet(100);
  scatter(motion(i0).uv0(ind, 1), motion(i0).uv0(ind, 2), 20, colors(ceil(size(colors, 1) * min(max(vrange), v(ind)) / max(vrange)), :), 'markerFaceColor', 'flat');
  caxis(vrange);
  colormap jet;
  colorbar
end

%% Compare to radar velocities
% Prepare radar velocities.
VEL_PATH = fullfile(DATA_DIR, 'velocities', 'v_201410081703_201410122133_50m.tif');
[V_ref, ~, box] = geotiffread(VEL_PATH);
V_ref(V_ref < -3e38) = NaN;
V_refm = sqrt(V_ref(:, :, 1).^2 + V_ref(:, :, 2).^2);
hshade = hillshade(GLACIER_DEM.Z, GLACIER_DEM.x, GLACIER_DEM.y);

%%
% Plot results on a map.
figure('units','normalized','outerposition',[0 0 1 1]);
% Radar velocities
ax1 = subtightplot(1, 3, 1);
showimg(GLACIER_DEM.x, GLACIER_DEM.y, hshade);
hold on
alphawarp(X, Y, V_refm, 1);
caxis(vrange);
colormap jet;
colorbar
title('Radar velocities (m/day)', 'fontsize', 14);
% Time-lapse velocities
ax2 = subtightplot(1, 3, 2);
showimg(GLACIER_DEM.x, GLACIER_DEM.y, hshade);
hold on
alphawarp(X, Y, V, 1);
caxis(vrange);
colormap jet;
colorbar
title('Time-lapse velocities (m/day)', 'fontsize', 14);
% Normalized difference
ax3 = subtightplot(1, 3, 3);
showimg(GLACIER_DEM.x, GLACIER_DEM.y, hshade);
hold on
alphawarp(X, Y, 100 * abs(V - V_refm) ./ V_refm, 1);
caxis([0, 100]);
colormap jet;
colorbar
s = 20;
quiver(X, Y, s * V_ref(:, :, 1), s * V_ref(:, :, 2), 0, 'g');
quiver(X, Y, s * DX / ddays, s * DY / ddays, 0, 'r');
legend('', 'radar', 'time-lapse');
title(['Percent error (%) | Velocity (', num2str(s), 'x)'], 'fontsize', 14);
linkaxes([ax1, ax2, ax3]);
xlim([min(min(X)) - 1000, max(max(X)) + 1000]);
ylim([min(min(Y)), max(max(Y))]);

%%
% Plot error statistics.
xy = [X(:), Y(:)];
z = SMOOTH_GLACIER_DEM.sample_points(xy);
xyz = [xy, z];
dxy = sqrt(sum((xyz - images(1).cam.xyz).^2, 2));
fdv = abs(V_refm - V) ./ V_refm;
figure();
% Error versus distance from camera
subtightplot(2, 1, 1, 0.1);
x = dxy;
y = fdv;
y(V(:) < 1) = NaN;
plot(x(:), 100 * y(:), '.')
ylim([0, 100])
title('Percent error vs. Distance (m)', 'fontsize', 14);
% Error versus angle from horizontal
subtightplot(2, 1, 2, 0.1);
x = abs(atan2d(images(1).cam.xyz(3) - xyz(:, 3), dxy));
y = fdv;
y(V(:) < 1) = NaN;
plot(x(:), 100 * y(:), '.')
ylim([0, 100])
set(gca, 'xdir', 'reverse')
title('Percent error vs. Angle to horizontal (deg)', 'fontsize', 14);
