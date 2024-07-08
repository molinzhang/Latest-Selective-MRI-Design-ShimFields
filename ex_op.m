function pAD = ex_op(Inivar_name, gpu_id, varargin)
  import attr.*
  % IniVar = matfile('IniVars.mat');
  IniVar = matfile(Inivar_name);
  IniVar.Properties.Writable = true;
  
  
  %% Load path to B0 map; B1+ map or bridcage field map or PTx field map; shim coil field map.
  % If not using any of them, set it to '';
  % FOV in unit cm.
  % Path of mask. Mask should contain Target and Mask.
  % Load optimized RF pulse; shim array current; 
  % Time resolution dt;
  % Points in RF; If optimized RF is loaded, it will be cut with the first n points.
  % smax of shim current and gradient. Even with large smax, it will also be smooth.
    % Be careful with this! Probabily needs to be super high! smax = 10000 doesn't work for slice selective excitation.
  % rf max. 
  % niter: Num for op iterations.
  % niter_gr: iterations for gr optimization.
  % niter_rf: iterations for rf optimization. Note that, when use inference, this is also how many diagonal sub-points are used for evaluation.
  % op_metrics: loss function. ml2xy -> mse on magnitude of Mxy : not consider phase; l2xy -> mse on Mxy : consider phase.
  % train: training or inference
  % save_mag: whether save magnetization after applying pulse. Recommend True for inference and False for training.
  % save_path: Path to save the magentization.
  % shim_path: Path to the shim field map. It should be in the data shape of (*Nd, num_shim). Reshape the fields in python from (3D, num_coil) to (1D, num_coil). The fields should be in the direction of main B0 field. 
  % lr_rf: learning rate for rf optimizer
  % lr_nr: learning rate for nonlinear optimizer.
  % eta: coeff of rf penalization.
  % adaptive_weights: use adaptive weights strategy.
  % weight_freq: frequency of updating weights (epochs).
  % weights_blur: use a gaussian blur kernel in space. This is designed for voxel-wise adaptive weights where edges, spikes and peak could appear.
  % slice_weight: weight is adaptive in terms of slice
  % thick_weight_init: true: use larger weights for 26-35; otherwise, only larger weight for slice of interest.
  % traj_con: whether use trajectory constraints. 
  % traj_type: 'z' or 'total'. 'z' means constraint in terms of deviation from z axis. 'total': total trajectory length in space.
  % lambda_traj: coeffcients for traj_con loss.
  % lambda_loss: coeffient for Md M loss.
  % lambda_pengr: coeff for l2 pen of non linear gradient.
  % lambda_edge: coeff for pen of edge. Double sides
  % op_fat: include fat into optimization for uniform pulse
  % if_so_spatial: stochastic offset in spatial domain
  % For now, we don't use shim fields and linear fields. Defualt is 0.
  
  arg.B0_path = 'data/non_exist.mat'; 
  arg.B1_path = 'data/non_exist.mat';
  arg.shim_path = '/home/molin/shimm_nick/Dynamic_resemble/demo/ISMRM/compare/field_64_body_4mm_renew.mat';
  arg.mask_path = '/home/molin/github_code_upload/MR_excitation_op_update/data/Mask_sagital_pregnancy_small.mat';%'/home/molin/shimm_nick/decompose_refocus/demo/ISMRM/compare/MASK_12mmROI_slice_finer_3pixel.mat';
  arg.rf_path = '';

  arg.FOV = [36, 36, 24];
  arg.num_coil = 64;
  arg.dt = 8e-6;
  arg.rf_points = 500;
  arg.smax = 10000000000;
  arg.rfmax = 1.0;
  
  arg.niter = 100;
  arg.niter_rf = 2;
  arg.niter_gr = 2;
  arg.train_rf = true;
  arg.train_nr = true;
  arg.save_mag = false;
  arg.save_path = '/home/molin/github_code_upload/results/inversion/center_thin2thick';
  % '/home/molin/github_code_upload/results/tmp'

  % niter = 1;
  % niter_rf = 80;  
  % train_rf = false;
  % train_nr = false;
  % save_mag = true;
  % save_path = '/home/molin/github_code_upload/mrf_rf_backup/results_midbrain_noadaptive/pulse_profile2/';
  
  arg.op_metrics = 'l2z'; %'ml2xy';

  
  arg.lr_rf = 5e-4;
  arg.lr_nr = 5e-4;

  arg.adaptive_weights = false;
  arg.weight_freq = 10.0;
  arg.weights_blur = false;
  arg.slice_weight = true;
  arg.thick_weight_init = true;

  arg.traj_con = false;
  arg.traj_type = 'z'; % 'total'
  arg.lambda_traj = 0.1; % 0.01

  arg.eta = 0.10; % 0.10
  arg.lambda_loss = 0.10; % 0,10
  arg.lambda_pengr = 0.10; % 0.0
  arg.lambda_edge = 10.0; % 10.0


  arg.op_fat = false;
  arg.if_so_spatial = true;

  if arg.train_nr == false
    arg.niter_gr = 0;
  end

  arg = attrParser(arg, varargin);

  
  cube = IniVar.cube;
  
  %% Build cude 
  MASK1 = load(arg.mask_path);
  MASK2 = MASK1.Mask;
  MASK = MASK2;

  CUBE = cube;
  
  CUBE.mask = logical(MASK);
  CUBE.nM = sum(sum(sum(MASK)));
  
  
  CUBE.dim = size(MASK);
  CUBE.fov = arg.FOV;  % unit cm. 
  

  if exist(arg.B0_path, 'file')
      B0_field = load(arg.B0_path);
      B0_field = B0_field.B0;
  else
      B0_field = zeros(size(MASK));
  end
  

  if exist(arg.B1_path, 'file')
      B1_field = load(arg.B1_path);
      B1_field = B1_field.B1;
  else
      B1_field = zeros(size(MASK));
  end
  
  
  % Update acquired B0 maps.
  b0Map = nan(CUBE.dim);
  b0Map(MASK == 1) = B0_field(MASK == 1);
  CUBE.b0Map_ = b0Map(MASK == 1); 
  CUBE.b0Map = b0Map;
  
  
  % Update acquired B1 maps. This is not a property of CUBE.
  b1Map = nan(CUBE.dim);
  b1Map(MASK == 1) = B1_field(MASK == 1);
  b1Map_ = b1Map(MASK == 1); 
  
  
  % Be careful with mashgrid. index order of ijk and xyz matters when FOV is not isotropic. 
  [Xv, Yv, Zv] = meshgrid(-arg.FOV(2)/2:CUBE.res(2):arg.FOV(2)/2-0.01, -arg.FOV(1)/2:CUBE.res(1):arg.FOV(1)/2-0.01, -arg.FOV(3)/2:CUBE.res(3):arg.FOV(3)/2-0.01);
  CUBE.loc = nan([CUBE.dim 3]);
  
  
  inter = CUBE.loc(:,:,:,1);
  inter(MASK == 1) = Xv(MASK ==1);
  CUBE.loc(:,:,:,1) = inter;
  inter = CUBE.loc(:,:,:,2);
  inter(MASK == 1) = Yv(MASK ==1);
  CUBE.loc(:,:,:,2) = inter;
  inter = CUBE.loc(:,:,:,3);
  inter(MASK == 1) = Zv(MASK ==1);
  CUBE.loc(:,:,:,3) = inter;
  CUBE.loc_ =  [Xv(MASK ==1), Yv(MASK== 1), Zv(MASK ==1)];
  
  
  % Initial magnitization
  Mag = zeros(([CUBE.dim 3]));
  mag = Mag(:,:,:,3);
  mag(MASK == 1) = 1;
  Mag(:,:,:,3) = mag;
  CUBE.M = Mag;
  mag1= Mag(:,:,:,1);
  mag2 = Mag(:,:,:,2);
  mag3 = Mag(:,:,:,3);
  
  
  CUBE.M_ = [mag1(MASK ==1), mag2(MASK ==1), mag3(MASK == 1)];
  % assume T1, T2, gamma is all the same;
  T1 = CUBE.T1_(1); T2 = CUBE.T2_(1); gamma = CUBE.gam_(1);
  CUBE.T1_ = zeros(CUBE.nM,1);
  CUBE.T1_(:,1) = T1;
  CUBE.T2_ = zeros(CUBE.nM,1);
  CUBE.T2_(:,1) = T2;
  CUBE.gam_ = zeros(CUBE.nM,1);
  CUBE.gam_(:,1) = gamma;
  
  
  % gamma : Hz/Gauss
  cube = CUBE;
  IniVar.cube = CUBE;
  
  % modify target and pulse
  
  Target = IniVar.target_OV90;
  TAR = double(MASK1.Target);
  
  % Target for Mz in the ROI: (0,0,1) -> (1,0,0)
  % Target for Mz in the ROI interferers / supressed region: (0,0,1) -> (0,0,1)
  % Target.d = cat(4, TAR, zeros(size(TAR)), 1-TAR);  
  
  invs = ones(size(TAR));
  invs(TAR == 1) = -1;
  Target.d = cat(4, zeros(size(TAR)), zeros(size(TAR)), invs);  

  % Change mask_weight to assign different weights on different region. Usually slices near ROI.
  Mask_weight = double(MASK);

  if arg.thick_weight_init
    Slice_of_interest = squeeze(Mask_weight(:,:,26:35)); % 19, 44
    Slice_of_interest(Slice_of_interest == 1) = 3.1; %1.1; % 4
    Mask_weight(:,:,26:35) = Slice_of_interest;
  else
    Slice_of_interest = squeeze(Mask_weight(:,:,31)); % 19, 44
    Slice_of_interest(Slice_of_interest == 1) = 1.1; %1.1; % 4
    Mask_weight(:,:,31) = Slice_of_interest;
  end

  % Slice_of_interest = squeeze(Mask_weight(:,:,16:25)); % 19, 44
  % Slice_of_interest(Slice_of_interest == 1) = 1.1; %1.1; % 4
  % Mask_weight(:,:,16:25) = Slice_of_interest;

  Mask_weight(TAR == 1) = 10;
  Mask_weight(Mask_weight == 1) = 0.01;
  Target.weight = Mask_weight;
  
  IniVar.target_OV90 = Target;

  
  if exist(arg.rf_path, 'file')
    data = load(arg.rf_path);
    RF = data.rf;
    RF = RF(1, 1:arg.rf_points);
    if isfield(data,'gr')
      non_linear = data.gr(4:end, 1:arg.rf_points);
    else
      non_linear = zeros(arg.num_coil,1);
      non_linear = repmat(non_linear, 1, size(RF,2));      
    end
  else
    RF = ones(1, arg.rf_points) * 0.1;
    % change the size to zeros(num_coil, 1) if you are using shim array.
    non_linear = zeros(arg.num_coil,1);
    non_linear = repmat(non_linear, 1, size(RF,2));
  end

  kk = IniVar.pIni_OV90;
  
  kk.rf = RF;
  kk.nr = non_linear;
  kk.lr = zeros(3,size(RF,2));
  kk.lr(3,:) = 0.2;
  kk.gr = [kk.lr; kk.nr]; % For non-selective design, we don't use gradient.
  kk.dt = arg.dt;
  kk.smax = arg.smax;
  kk.rfmax = arg.rfmax;
  IniVar.pIni_OV90 = kk;
  
  pAD = adpulses.opt.arctanAD(IniVar.target_OV90, ...
                                  cube, ...
                                  IniVar.pIni_OV90, ...
                                  'err_meth', arg.op_metrics, ...
                                  'doClean', false,  ...
                                  'gpuID', gpu_id, ...
                                  'niter', arg.niter, ... 
                                  'niter_gr', arg.niter_gr, ...
                                  'niter_rf', arg.niter_rf, ...
                                  'b1Map', b1Map, ...
                                  'shim_path', arg.shim_path, ...
                                  'train_rf', arg.train_rf, ...
                                  'train_nr', arg.train_nr, ...
                                  'lr_rf', arg.lr_rf, ...
                                  'lr_nr', arg.lr_nr, ...
                                  'eta', arg.eta, ...
                                  'if_so_spatial', arg.if_so_spatial, ...
                                  'op_fat', arg.op_fat, ...
                                  'save_mag', arg.save_mag, ...
                                  'save_path', arg.save_path, ...
                                  'res', cube.res(1), ...
                                  'adaptive_weights', arg.adaptive_weights, ...
                                  'weight_freq', arg.weight_freq, ...
                                  'weights_blur', arg.weights_blur, ...
                                  'slice_weight', arg.slice_weight, ...
                                  'traj_con', arg.traj_con, ...
                                  'traj_type', arg.traj_type, ...
                                  'lambda_traj', arg.lambda_traj, ...
                                  'lambda_loss', arg.lambda_loss, ...
                                  'lambda_pengr', arg.lambda_pengr, ...
                                  'lambda_edge', arg.lambda_edge);
  end
  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
