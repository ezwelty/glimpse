function value = rbind(varargin)
  % RBIND Vertcat with automatic expansion.
  %
  %   value = rbind(...)

  cols = cellfun(@(x) size(x, 2), varargin);
  reps = max(cols) ./ cols;
  reps(isinf(reps)) = 1;
  if any(reps ~= round(reps))
    error('Number of cols in input is not a multiple of longest input.');
  end
  expand = reps > 1;
  expand_reps = mat2cell(reps(expand), 1, repmat(1, 1, sum(expand)));
  varargin(expand) = cellfun(@(x, rep) repmat(x, 1, rep), varargin(expand), expand_reps, 'UniformOutput', false);
  value = vertcat(varargin{:});
end
