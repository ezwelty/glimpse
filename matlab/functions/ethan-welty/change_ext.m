function new_paths = change_ext(paths, new_ext)
  % CHANGE_EXT Change or strip file extensions.
  %
  %   new_paths = change_ext(paths, new_ext = '')
  %
  % Inputs:
  %   paths   - File paths
  %   new_ext - New file extension
  %
  % See also regexprep

  if nargin < 2
    new_ext = '';
  end
  new_paths = regexprep(paths, '\.[^\.]*$', new_ext);
end
