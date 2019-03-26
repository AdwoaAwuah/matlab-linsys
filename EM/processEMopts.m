function [opts] = processEMopts(opts,nu,nx,ny)

%Define default values:
defaultOpts.Niter=2e3;%Max number of iters
defaultOpts.Nreps=10;%Number of repetitions for randomStartEM
defaultOpts.robustFlag=false;
defaultOpts.outlierReject=false; %No rejection of outliers in KF/KS
defaultOpts.fastFlag=1;  %statKF/KS auto-select fast samples by default
defaultOpts.convergenceTol=5e-3; % 5e-3 minimum improvement in logL (per dim, but not per sample) every 100  iterations
defaultOpts.targetTol=1e-3;  % .1% minimum improvement towards target every 100 iters. This is possibly TOO tolerant, it never stops an iteration early, no matter what.
defaultOpts.targetLogL=[];
defaultOpts.diagA=false;
defaultOpts.sphericalR=false;
defaultOpts.diagR=false;
defaultOpts.thR=0;
defaultOpts.outlierReject=false;
defaultOpts.indD=1:nu; %Include all
defaultOpts.indB=1:nu; %Include all
defaultOpts.logFlag=false;
defaultOpts.fixA=[];
defaultOpts.fixB=[];
defaultOpts.fixC=[];
defaultOpts.fixD=[];
defaultOpts.fixQ=[];
defaultOpts.fixR=[];
defaultOpts.fixX0=[];
defaultOpts.fixP0=[];
defaultOpts.includeOutputIdx=1:ny; %Include all

%Assign any options that were not provided with default values:
fNames=fieldnames(defaultOpts);
for i=1:length(fNames)
  if ~isfield(opts,fNames{i}) || isempty(opts.(fNames{i}))
    opts.(fNames{i})=defaultOpts.(fNames{i});
  end
end

%Do a sanity check on options that may be conflicting:
if ~isempty(opts.fixB)
  opts.indB=1:nu;
  if any(size(opts.fixB)~=[nx,nu])
    error('EMopts:providedBdimMismatch','Provided B matrix size is inconsistent with number of inputs or states.')
  end
end
if ~isempty(opts.fixD)
  opts.indD=1:nu;
  if any(size(opts.fixD)~=[ny,nu])
    error('EMopts:providedDdimMismatch','Provided D matrix size is inconsistent with number of inputs or outputs.')
  end
end

%Reintepret some special options:
if isnan(opts.fixQ)
  opts.fixQ=zeros(nx);
end
if isnan(opts.fixX0)
  opts.fixX0=zeros(nx,1);
  opts.fixP0=zeros(nx); %No uncertainty if initial state is given
end
if isnan(opts.fixP0)
  opts.fixP0=diag(inf(size(opts.fixX0))); %Max uncertainty
end
if islogical(opts.includeOutputIdx)
  if length(opts.includeOutputIdx)~=ny
    error('EMopts:outputIdxListSizeMismatch','Provided list of output indexes to be included is inconsistent with output size.')
  else
  opts.includeOutputIdx=find(opts.includeOutputIdx); %Logical to index list, not sure this is necessary
  end
end
