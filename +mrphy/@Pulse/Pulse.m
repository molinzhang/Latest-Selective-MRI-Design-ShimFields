% object of MRI excitation pulse
%
% Brief intro on methods:
%
%   Tianrui Luo, 2019
%}

classdef Pulse < matlab.mixin.SetGet & matlab.mixin.Copyable
  properties (SetAccess = immutable)
    %dt = mrphy.utils.envMR('get', 'dt');       % s
    %gmax = mrphy.utils.envMR('get', 'gMax');   % G/cm
    %smax = mrphy.utils.envMR('get', 'sMax');   % G/cm/s
    %rfmax = mrphy.utils.envMR('get', 'rfMax'); % G
  end
  properties (SetAccess = public, GetAccess = public)
    rf % (1, nT, (nCoils))
    gr % (xyz, nT) % modified -> (xyz + channel, nT)
    nr
    lr
    dt
    smax
    gmax
    rfmax
    desc = 'generic pulse'
  end
  
  methods (Access = public)
    function obj = Pulse(varargin)
      % The constructor of a class will automatically create a default obj.
      import attr.*

      st = cutattrs(struct(varargin{:}), {}, properties('mrphy.Pulse'));
      for fName = fieldnames(st)', obj.(fName{1}) = st.(fName{1}); end
      assert(size(obj.rf,1)==1 && size(obj.gr,1)==3);
      assert(size(obj.rf,2)==size(obj.gr,2));
      % TODO:
      % - add sanity checks
    end

    function beff = beff(obj, loc, masks, varargin)
      %INPUTS:
      % - loc (*Nd, xyz)
      %OPTIONALS:
      % - b0Map (*Nd)
      % - b1Map (*Nd, 1, nCoils)
      % - gam (1,)
      %OUTPUTS:
      % - beff (*Nd, xyz, nT)
      import attr.*

      [arg.b0Map,arg.b1Map,arg.gam] =deal([],[],mrphy.utils.envMR('get','gam'));
      arg = attrParser(arg, varargin);
      kw = [fields(arg), struct2cell(arg)]';
      beff = mrphy.beffective.rfgr2beff(obj.rf, obj.gr, loc, masks, kw{:});
    end

    function st = asstruct(obj)
      warning('off', 'MATLAB:structOnObject')
      st = struct(obj);
      warning('on', 'MATLAB:structOnObject')
    end

  end
    
  methods % set and get, sealed if the property cannot be redefined
    % TODO:
    % - add sanity checks
  end

end
