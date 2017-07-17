function [C starts stops] = splitmat(X, in)
  % SPLITMAT Split matrix into cells by logical index.
  %
  %   C = splitmat(X, in)
  %
  % Inputs:
  %   X  - Matrix
  %   in - Logical index of length matching a dimension of X
  %
  % Outputs:
  %   C - Cell array of matrix subset
  %   I - Cell array of subset indices
  %
  % See also mat2cell

  % TODO: Extend to N-D

  dim = find(length(in) == size(X));
  if numel(dim) > 1 && any(size(X) > 1)
    dim = find(size(in) > 1);
  end
  if isempty(dim)
    eror('Length of in does not match a dimension of X');
  end
  in = horzcat(false, reshape(in, 1, []), false);
  d = diff(in);
  starts = find(d == 1);
  stops = find(d == -1) - 1;
  C = cell(size(starts));
  if dim == 1
    for i = 1:numel(starts)
      C{i} = X(starts(i):stops(i), :);
    end
  else
    for i = 1:numel(starts)
      C{i} = X(:, starts(i):stops(i));
    end
  end
end
