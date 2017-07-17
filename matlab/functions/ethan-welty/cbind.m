function value = cbind(varargin)
  % CBIND Horzcat with automatic expansion.
  %
  %   value = cbind(...)

  rows = cellfun(@(x) size(x, 1), varargin);
  reps = max(rows) ./ rows;
  reps(isinf(reps)) = 1;
  if any(reps ~= round(reps))
    error('Number of rows in input is not a multiple of longest input.');
  end
  expand = reps > 1;
  expand_reps = mat2cell(reps(expand), 1, repmat(1, 1, sum(expand)));
  varargin(expand) = cellfun(@(x, rep) repmat(x, rep, 1), varargin(expand), expand_reps, 'UniformOutput', false);
  value = horzcat(varargin{:});
end
