%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../aux/'))
addpath(genpath('../sim/'))
addpath(genpath('../../robustCov/'))
%% Create model:
clearvars
D1=2;
D2=180;
N=1000;
A=randn(D1);
A=[.97,0;0,.995];
A=jordan(A); %Using A in its jordan canonical form so we can compare identified systems, WLOG
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0001;
R=eye(D2)*.01;
%% Simulate many times:
for k=1:10
    NN=size(U,2);
    x0{k}=randn(D1,1);
    [Y{k},X{k}]=fwdSim(U,A,B,C,D,x0{k},Q,R);
    Uu{k}=U;
    P0{k}=eye(size(Q));
end
%%
[A,B,C,x0,~,Q,P0] = canonizev2(A,B,C,X,Q,P0);
logLo=dataLogLikelihood(Y,Uu,A,B,C,D,Q,R,x0,P0)

%% Estimate params with a single realization:
fastFlag=0; %Should be empty for regular EM
[Ae,Be,Ce,De,Qe,Re,x0e,P0e,bestLogLe]=EM(Y{1},Uu{1},D1,[],fastFlag);
[Ae,Be,Ce,x0e,~,Qe,P0e] = canonizev2(Ae,Be,Ce,x0e,Qe,P0e);
logL=dataLogLikelihood(Y{1},Uu{1},Ae,Be,Ce,De,Qe,Re,x0e,P0e);
%Add actual states:
fh=gcf;
hold on
plot(X{1}','k')
text(100,max(X{1}(:)),num2str(logLo))

%% Estimate with many:
fastFlag=0; %Should be empty for regular EM
[Ae,Be,Ce,De,Qe,Re,x0e,P0e,bestLogLe]=EM(Y,Uu,D1,[],fastFlag);
[Ae,Be,Ce,x0e,~,Qe,P0e] = canonizev2(Ae,Be,Ce,x0e,Qe,P0e);
logL=dataLogLikelihood(Y,Uu,Ae,Be,Ce,De,Qe,Re,x0e,P0e);
fh=gcf;
for i=1:numel(X)
    subplot(numel(X),1,i)
    hold on
plot(X{i}','k')
end
%% PLot?
