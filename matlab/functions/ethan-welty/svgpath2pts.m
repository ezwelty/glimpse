% SVGPATH2PTS  Convert an SVG path to a list of points.
%
%   X = svgpath2pts(path)
%
% Calculates the absolute coordinates of the vertices in an SVG path.
% Curvature is discarded. See http://www.w3.org/TR/SVG/paths.html.
%
% Input:  path  string corresponding to SVG path "d" attribute
%
% Output:   X     NX2 matrix of x and y coordinates
%
% See also svg2struct.

function X = svgpath2pts(path)

  % Parse letter and number lists
  letters = regexp(path,'[a-zA-Z]','match');
  numbers = regexp(path,'[\.,0-9\-]+','match');
  N = length(letters);

  % Extract coordinates of vertices
  for n = 1:N

    % initialize current path segment
    tag = letters{n};
    temp = textscan(numbers{n}, '%f', 'delimiter', ' ,', 'multipledelimsasone', 1);
    data = temp{1};
    if length(data) > 1
      data = reshape(data, 2, length(data) / 2)';
    end

    switch tag

      % path always begins with M (move to)
      case 'M'
        if n == 1
          X = data;
        else
          warning(['Found M at element: ' num2str(n)]);
        end

      % L/l (line to)
      case 'L'
        X = [X ; data];
      case 'l'
        X = [X ; X(end,:) + data];

      % H/h (horizontal line to)
      case 'H'
        X = [X ; data X(end,2)];
      case 'h'
        X = [X ; X(end,1)+data X(end,2)];

      % V/v (vertical line to)
      case 'V'
        X = [X ; X(end,1) data];
      case 'v'
        X = [X ; X(end,1) X(end,2)+data];

      % C/c (curve to)
      case 'C'
        X = [X ; data(3,:)];
      case 'c'
        X = [X ; X(end,:) + data(3,:)];

      % S/s (simple curve to)
      case 'S'
        X = [X ; data(2,:)];
      case 's'
        X = [X ; X(end,:) + data(2,:)];

      % Z/z (close path)
      case {'Z', 'z'}
        X = [X ; X(1,:)];

      % Not all supported. Warn if others encountered.
      otherwise
        warning(['Unsupported tag encountered: ' tag])
    end
  end

  % Remove repeats
  % % FIXME: Would not work if contained NaNs.
  % % FIXME: Only remove adjacent duplicates!
  % if size(X,1) > 1
  %   diffx = logical(diff(X(:,1)));
  %   diffy = logical(diff(X(:,2)));
  %   uniqueX = diffx | diffy;
  %   X = X(uniqueX,:);
  % end
end
