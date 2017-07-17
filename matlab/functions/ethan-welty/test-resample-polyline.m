X = rand(1e2, 3);
tic
for i = 1:1e2
  l = polylineLength(X);
  % z = resamplePolyline(X, 1e3);
end
toc

tic
for i = 1:1e2
  t = cumsum(sqrt(sum(diff(X).^2, 2)));
  % z = interpXn(X, t, 0:(t(end) / 1e3):t(end));
end
toc
