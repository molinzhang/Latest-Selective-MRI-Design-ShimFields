# A Unified Interface of Parsing and Set/Getting for Matlab

Rename this directory from `attr.mat` to `+attr`.

## Doc
`attrParser.m`,\
Similar to `mirt/utils/vararg_pair.m`. Based on MatLab built-in parsing, which
might get better performance with Matlab evolving over time.

`attrs.m`,\
an unified interface for `fieldnames()` and `properties()`. It retrieves fields
regardless of whether its input is a structure or an object.

`chkattrs.m`,\
assert whether a field exists in its input.

`cutattrs.m`,\
cut prescribed fields from its input.

`getattrs.m`,\
retrieve (multiple) fields values from its input, e.g.,
`[fa, fb] = getattrs(input, {'fa','fb'});`

`isattr.m`,\
an unified interface for `isfield()` and `isprop()`.

`mrgattrs.m`,\
merge fields in its src_intpu to its dst_input.

`rmattrs.m`,\
remove prescribed fields from its input.

