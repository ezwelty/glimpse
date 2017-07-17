function xyz = intersectDEM(cam, dem, uv, xy0, visible)

      uv = double(uv);
      nanix = any(isnan(uv), 2);
      anynans = any(nanix);
      if anynans
        uv(nanix, :) = [];
      end

      if nargin < 5
        visible = voxelviewshed(dem.X, dem.Y, dem.Z, cam.xyz);
      end
      X = dem.X; Y = dem.Y; Z = dem.Z ./ visible;
      xyz = nan(size(uv, 1), 3);
      if nargin < 4 || isempty(xy0)
        uv0 = cam.project([X(visible(:)), Y(visible(:)), Z(visible(:))]);
        inframe = cam.inframe(uv0);
        uv0(:, 3) = dem.X(visible(:));
        uv0(:, 4) = dem.Y(visible(:));
        uv0(:, 5) = dem.Z(visible(:));
        uv0 = double(uv0(inframe, :));
        if exist('scatteredInterpolant','file') > 1
          Xscat = scatteredInterpolant(uv0(:, 3), uv0(:, 4), uv0(:,3));
          Xscat.Points = uv0(:, 1:2);
          Yscat = Xscat; Yscat.Values = uv0(:, 4);
          Zscat = Xscat; Zscat.Values = uv0(:, 5);
        else
          %fallback for older versions of matlab.
          Xscat = TriScatteredInterp(uv0(:, 3), uv0(:, 4), uv0(:, 3));  %#ok<REMFF1>
          Xscat.X = uv0(:, 1:2);
          Yscat = Xscat; Yscat.V = uv0(:,4);
          Zscat = Xscat; Zscat.V = uv0(:,5);
        end
        xy0 = [Xscat(uv(:,1), uv(:,2)), Yscat(uv(:,1), uv(:,2)), Zscat(uv(:, 1), uv(:, 2))];
        xyz = xy0;
        if anynans
          xyz(find(~nanix), :) = xyz; %find necessary because it ensures that xyz can grow.
          xyz(find(nanix), :) = nan;
        end
        return
      end

      if Y(2, 2) < Y(1, 1)
        X = flipud(X); Y = flipud(Y); Z = flipud(Z);
      end
      if X(2, 2) < X(1, 1)
        X = fliplr(X); Y = fliplr(Y); Z = fliplr(Z);
      end

      if exist('griddedInterpolant','file') > 1
        zfun = griddedInterpolant(X', Y', Z'); % TODO: improve robustness.
      else
        %fallback for older versions of matlab. slower
        zfun = @(x, y) interp2(X, Y, Z, x, y);
      end
      for ii = 1:length(uv)
        %misfit=@(xy)sum((cam.project([xy zfun(xy(1),xy(2))])-uv(ii,1:2)).^2);
        misfitlm = @(xy) (cam.project([xy(:)' zfun(xy(1), xy(2))]) - uv(ii, 1:2))'.^2;
        try
          %[xyz(ii,1:2),err]=fminunc(misfit,xy0(ii,1:2),optimset('LargeScale','off','Display','off','TolFun',0.001)); % TODO: remove dependency. can i use LMFnlsq?
          xyz(ii, 1:2) = LMFnlsq(misfitlm, xy0(ii, 1:2));
          xyz(ii, 3) = zfun(xyz(ii, 1), xyz(ii, 2));
          if sum(misfitlm(xyz(ii, 1:2))) > 2^2
            xyz(ii, :) = nan; % do not accept greater than 2 pixel error.
          end
        catch
        end
      end

      if anynans
        xyz(find(~nanix), :) = xyz; %find necessary because it ensures that xyz can grow.
        xyz(find(nanix), :) = nan;
      end
    end
