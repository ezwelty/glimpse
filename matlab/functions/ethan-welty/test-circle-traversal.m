n = 1000;
nx = n;
ny = n;
x0 = nx / 2;
y0 = ny / 2;
radius = 1 : 1/10 : (nx / 2);
I = zeros(ny, nx);

for i = 1 : numel(radius)
  [x y] = getmidpointcircle(x0, y0, radius(i));
  x = round(x); y = round(y);
  out = x < 1 | y < 1 | x > nx | y > ny;
  ind = sub2ind([ny, nx], y(~out), x(~out));
  I(ind) = 1;
  % for j = 1:numel(x)
  %   xp = x(j);
  %   yp = y(j);
  %   if (xp < 1 || yp < 1 || xp > image_size || yp > image_size )
  %     continue
  %   end
  %   I(xp, yp, :) = 1;
  % end
end

imshow(I);
