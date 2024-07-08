function beff = rfgr2beff(rf, gr, loc, masks, varargin)
%INPUTS:
% - rf (1, nT, (nCoils))
% - gr (xyz, nT)
% - loc (*Nd, xyz)
%OPTIONALS:
% - b0Map (*Nd)
% - b1Map (*Nd, 1, nCoils)
% - gam (1,)
%OUTPUTS:
% - beff (*Nd, xyz, nT)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this matlab file is used for plotting. 
% When we have the optimized rf, gz, (shimming current), 
% we have to do the blocksim again to get the M_xy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import attr.*

shape = size(loc);
nT = size(rf, 2);
shape = shape(1:end-1); % volumes under masks
loc = reshape(loc, [], 3);  % -> (prod(Nd), xyz)

%% parsing
[arg.b0Map, arg.b1Map, arg.gam] = deal([], [], mrphy.utils.envMR('get', 'gam'));
arg = attrParser(arg, varargin);

[b0Map, b1Map, gam] = getattrs(arg, {'b0Map', 'b1Map', 'gam'});

%% form beff
%%% load shim array files
%field = load('/home/molin/shimm_nick/AutoDiffPulses-master/demo/ISMRM/1.6/field_r2.mat');
%field = field.field_;
fie = load("/home/molin/shimm_nick/Dynamic_resemble/demo/ISMRM/compare/field_64_body_4mm.mat");
%fie = load("/home/molin/shimm_nick/decompose_refocus/demo/ISMRM/compare/phantom_3mm_field.mat");
%field_total = fie.fields;
field_total = reshape(fie.fields, [88, 88, 60, size(fie.fields,2)]);
%field_total = reshape(fie.fields, [30,30,20, size(fie.fields,2)]);
%field_total = reshape(fie.fields, [64, 64, 38, size(fie.fields,2)]);
for idx = 1:size(field_total,4)
fieldint = field_total(:,:,:,idx);
field(:,idx) = fieldint(masks == 1);
end
if size(gr,1) == 3
bz = reshape(loc*gr, [shape, 1, nT]); % (prod(Nd), nT) -> (*Nd, 1, nT)
else
bz = reshape(loc*gr(1:3,:), [shape, 1, nT]) + reshape(field*gr(4:end,:), [shape, 1, nT]);    
end
if ~isempty(b0Map), bz = bsxfun(@plus, bz, bsxfun(@rdivide, b0Map, gam)); end

rf = repmat(rf, prod(shape), 1); % -> (prod(Nd), nT, (nCoils))
if ~isempty(b1Map), rf=bsxfun(@times, reshape(b1Map,prod(shape),1,[]), rf); end
rf = reshape(sum(rf, 3), [shape, 1, nT]); % (prod(Nd),nT,(nCoils)) -> (*Nd,1,nT) prod is product of the elements of the array

%% return
beff = cat(numel(shape)+1, real(rf), imag(rf), bz); % (*Nd, xyz, nT)

end
