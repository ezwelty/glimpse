function files = expand_path(path, relative)
  % EXPAND_PATH Expand path with wildcards to file paths.
  %
  %   files = expand_path(path, relative = true)
  %
  % Inputs:
  %   path - Relative or absolute path with asterisk wildcards
  %   relative - Whether to return relative (or absolute) paths
  %
  % See also dir, relativepath

  if nargin < 2
    relative = true;
  end
  results = dir(path);
  if relative && exist('relativepath', 'file')
    paths = cellfun(@relativepath, {results.folder}, 'uniform', false);
  else
    paths = {results.folder};
  end
  files = fullfile(paths, {results.name})';
end
