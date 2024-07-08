function output = rmattrs(input, attrsReq)
% Just a wrap of rmfield. Program created for nameing consistency.
% This program is not supposed to operate on any object as input
import attr.*

output = rmfield(input, attrsReq);
end
