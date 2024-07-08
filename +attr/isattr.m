function tf = isattr(x, str)
import attr.*

% niao
  tf = isfield(x, str) || isprop(x, str);
end
