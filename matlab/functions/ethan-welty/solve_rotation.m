function [newimg, fit, F] = solve_rotation(img0, img, gridsize, templatesize, searchsize)

  if nargin < 4
    templatesize = gridsize;
  end
  if nargin < 5
    searchsize = gridsize * 2;
  end

  %%
  % Match features between image and anchor, filter with RANSAC, and optimize orientation.
  %
  %  TODO: Support variable image size
  %
  I0 = img0.read();
  I = img.read();

  %%
  % Generate grid of points in land polygons and match between images.
  %
  %  FIXME: Assumes a maximum motion between images
  %  TODO: Move point generation to anchor-level loop
  %
  if any(ismember(fieldnames(img0), 'fixedpolys'))
    pts = [];
    for j = 1:length(img0.fixedpolys)
      pts = [pts; polygon2grid(img0.fixedpolys{j}, gridsize, gridsize)];
    end
  else
    pts = polygon2grid(img0.cam.framepoly, gridsize, gridsize);
  end
  [du, dv, correlation, signal, pu, pv] = templatematch(I0, I, pts(:, 1), pts(:, 2), 'templatewidth', templatesize, 'searchwidth', searchsize, 'method', 'NCC', 'super', 2);
  is_strong = correlation > quantile(correlation, 0.8) & signal > quantile(signal, 0.8);
  matches = horzcat(pu, pv, du, dv);
  matches = matches(is_strong, :);

  %%
  % Filter matches with RANSAC.
  %
  %  FIXME: Assumes images are the same size and camera
  %  TODO: Express threshold in pixels
  %
  xy0 = img0.cam.image2camera(matches(:, 1:2));
  xy = img0.cam.image2camera(matches(:, 1:2) + matches(:, 3:4));
  [F, in] = ransacfitfundmatrix(xy0', xy', 1e-6);

  uv0 = matches(:, 1:2);
  duv = matches(:, 3:4);
  uv = uv0 + duv;
  mean_motion = mean(sqrt(sum(duv(in, :).^2, 2)));

  %%
  % Orient camera
  %
  %  FIXME: Assumes images are the same size and camera
  %
  xyz = img0.cam.xyz + img0.cam.invproject(uv0(in, :));
  [newcams, fit] = Camera.optimize_bundle(img0.cam, uv(in, :), xyz, 'viewdir');
  newimg = img;
  newimg.cam = newcams{1};

  puv = newimg.cam.project(xyz);
  pduv = puv - uv;

  %%
  % Plot results
  s = round(10 / mean_motion);
  figure();
  imshow(I0 / 1.5), hold on;
  quiver(uv0(:, 1), uv0(:, 2), s * duv(:, 1), s * duv(:, 2), 0, 'r');
  quiver(uv0(in, 1), uv0(in, 2), s * duv(in, 1), s * duv(in, 2), 0, 'y');
  quiver(uv(in, 1), uv(in, 2), s * pduv(in, 1), s * pduv(in, 2), 0, 'g');
  title(['\fontsize{14} ', 'Feature matches: 0 -> 1 | Orientation: RMSE ', num2str(fit.rmse(1), '%.2f'), ' px | Magnification: ' num2str(s) ' x']);
  legend('all matches', 'filtered matches', 'orientation errors');
end
