function value = ifelse(test, yes, no)
  % IFELSE Compact if-else statement.
  %
  %   value = ifelse(test, yes = true, no = false)
  %
  % Inputs:
  %   test - Logical
  %   yes  - Returned if true
  %   no   - Returned if false
  
  if test
    if nargin < 2
      value = true;
    else
      value = yes;
    end
  else
    if nargin < 3
      value = false;
    else
      value = no;
    end
  end
end
