function [fh,model] = legacy_vizSingleModelMLMC_LongAdapt(singleModel,Y,U)
if isa(singleModel,'linsys')
    singleModel=struct(singleModel);
end
if ~isfield(singleModel,'J')
    singleModel.J=singleModel.A;
end
M=size(singleModel.J,1);
fh=figure('Units','Normalized','OuterPosition',[0 0 .5 1],'Color',ones(1,3));
if nargin>1
    Nx=11;
else
    Nx=4;
end
Nu=size(singleModel.B,2);
Ny=M+1+Nu;
model{1}=singleModel;
Nu=size(model{1}.B,2);
Nc=size(Y,1);

%% First: normalize model, compute basic parameters of interest:
if nargin<3
    U=[zeros(Nu,100) ones(Nu,1000)]; %Step response
end
rotationMethod='canonical';
%rotationMethod='orthonormal';
for i=1:length(model)
%     [model{i}.J,model{i}.B,model{i}.C,~,~,model{i}.Q] = canonize(model{i}.J,model{i}.B,model{i}.C,[],model{i}.Q,[],rotationMethod);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,[],[],[]); %Simulating from MLE initial state
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;
    model{i}.logLtest=dataLogLikelihood(Y,U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.Q,model{i}.R,[],[],'approx');
    if nargin>1
        opts.fastFlag=0;
       [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,model{i}.J,model{i}.C,model{i}.Q,model{i}.R,[],[],model{i}.B,model{i}.D,U,opts);
        model{i}.Xs=Xs; %Smoothed data, (aka Posterior, final hidden states after optimization)  
        model{i}.Pp=Pp; %One-step ahead uncertainty from filtered data.
        model{i}.Pf=Pf;
        model{i}.Xf=Xf; %Filtered data
        model{i}.Xp=Xp; %Predicted data
        model{i}.out=model{i}.C*model{i}.Xs+model{i}.D*U; %output
        model{i}.res=Y-model{i}.out;
        model{i}.oneAheadStates=model{i}.J*model{i}.Xs(:,1:end-1)+model{i}.B*U(:,1:end-1); %one step ahead model
        model{i}.oneAheadOut=model{i}.C*(model{i}.oneAheadStates)+model{i}.D*U(:,2:end);
        model{i}.oneAheadRes=Y(:,2:end)-model{i}.oneAheadOut;

    end
end

%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
cc=[0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
ex1=cc(2,:);
ex2=cc(5,:);
mid=ones(1,3);
N=100;
gamma=1.5; %gamma > 1 expands the white (mid) part of the map, 'hiding' low values. Gamma<1 does the opposite
gamma=1;
map=[flipud(mid+ (ex1-mid).*([1:N]'/N).^gamma); mid; (mid+ (ex2-mid).*([1:N]'/N).^gamma)];

%% Plot
ytl={'GLU','TFL','HIP','RF','VL','VM','SEMT','SEMB','BF','MG','LG','SOL','PER','TA'};
% ytl={'TA', 'PER', 'SOL', 'LG', 'MG', 'BF', 'SEMB', 'SEMT', 'VM', 'VL', 'RF', 'TFL','HIP', 'GLU'}
yt=1:14;
fs=7;
% STATES
CD=[model{1}.C model{1}.D];
XU=[model{1}.Xs;U];
% rotMed='orthonormal';
% rotMed='orthomax';
rotMed='none';
%rotMed='promax';
%rotMed='quartimax';
%rotMed='pablo';
%rotMed='none';
[CDrot,XUrot]=rotateFac(CD,XU,rotMed);
if strcmp(rotMed,'none')
    factorName=[strcat('C_',num2str([1:size(model{1}.C,2)]'));strcat('D_',num2str([1:size(model{1}.D,2)]'))];
    latentName=[strcat('State ',' ', num2str([1:size(model{1}.C,2)]'));strcat('Input ',' ', num2str([1:size(model{1}.D,2)]'))];
else
    factorName=strcat('Factor ',num2str([1:size(CD,2)]'));
    latentName=strcat('Latent ',num2str([1:size(CD,2)]'));
end
aC=prctile(abs(Y(:)),98);
CDiR=CDrot'*inv(model{1}.R);
CDiRCD=CDiR*CDrot;
projY=CDiRCD\CDiR*Y;
for i=1:size(CD,2)
   
    hold on
    if i<3%nargin>1
        ph(i)=subplot(Nx,2,i); %TOP row: states temporal evolution and data projection
        scatter(1:size(Y,2),projY(i,:),5,.7*ones(1,3),'filled')
        set(gca,'ColorOrderIndex',1)
        %p(i)=plot(model{1}.smoothStates(i,:),'LineWidth',2,'DisplayName',['Deterministic state, \tau=' num2str(-1./log(model{1}.J(i,i)),3)]);
        %title('(Smoothed) Step-response states')
        if i<=size(CD,2)-2
            title([latentName(i,:), '\tau=' num2str(-1./log(model{1}.J(i,i)),3)]) %DMMO
        else
            title(latentName(i,:))
        end
        p(i)=plot(XUrot(i,:),'LineWidth',2,'Color','k');
        %patch([1:size(model{1}.Xf,2),size(model{1}.Xf,2):-1:1]',[model{1}.Xf(i,:)+sqrt(squeeze(model{1}.Pf(i,i,:)))', fliplr(model{1}.Xf(i,:)-sqrt(squeeze(model{1}.Pf(i,i,:)))')]',p(i).Color,'EdgeColor','none','FaceAlpha',.3)
        ax=gca;
        ax.Position=ax.Position+[0 .045 0 0];
%         axis tight
        grid on
    end


    subplot(Nx,Ny,Ny+i+[0,Ny])% Second row: checkerboards
    try
        imagesc((reshape(CDrot(:,i),12,Nc/12)'))
    catch
        imagesc((CDrot(:,i)))
    end
    set(gca,'XTick',[],'YTick',yt,'YTickLabel',ytl,'FontSize',fs)
    ax=gca;
    ax.YAxis.Label.FontSize=12;
%     colormap(flipud(map))
    colormap(map)
%     caxis([-aC aC])
    caxis([-1 1])
    axis tight
    title(factorName(i,:))
    ax=gca;
    ax.Position=ax.Position+[0 .03 0 0];
    hold on
    aa=axis;
    plot([0.1 1.9]+.5, [16 16],'k','LineWidth',2,'Clipping','off')
    text(.8,17,'DS','FontSize',6)
    plot([2.1 5.9]+.5, [16 16],'k','LineWidth',2,'Clipping','off')
    text(2.8,17,'SINGLE','FontSize',6)
    plot([6.1 7.9]+.5, [16 16],'k','LineWidth',2,'Clipping','off')
    text(6.9,17,'DS','FontSize',6)
    plot([8.1 11.9]+.5, [16 16],'k','LineWidth',2,'Clipping','off')
    text(9,17,'SWING','FontSize',6)
    axis(aa)

end
% linkaxes(ph,'y')
%Covariances
%subplot(Nx,Ny,Ny)
%imagesc(model{1}.Q)
%set(gca,'XTick',[],'YTick',[],'YTickLabel',ytl,'FontSize',8)
%colormap(flipud(map))
%aC=.5*max(abs(model{1}.Q(:)));
%caxis([-aC aC])
%axis tight
%subplot(Nx,Ny,2*Ny+[0,Ny])
%imagesc(model{1}.R)
%set(gca,'XTick',[],'YTick',[],'YTickLabel',ytl,'FontSize',8)
%colormap(flipud(map))
%aC=.5*max(abs(model{1}.R(:)));
%caxis([-aC aC])
%axis tight

if nargin<2
    %Third row: one-ahead step-response


else %IF DATA PRESENT:
N=size(Y,2);
% viewPoints=[1,40,51,151,251,651,940,951,1001,1101,N-11]+5;
viewPoints=[40,51,940,951,1050,1140];
% viewPoints=[35,41,460,490,650];
% viewPoints=[130,140,2130,2140,2250,2550]+3;
%viewPoints=[151,175,1044,1051,1075,1251]+3;
binw=4; %Plus minus 2
viewPoints(viewPoints>N-binw/2)=[];
Ny=length(viewPoints);
M=length(model);
dd=Y(:,1:50);
dd=dd-mean(dd,2); %Baseline residuals under flat model
meanVar=mean(sum(dd.^2,1),2);
for k=1:3
    for i=1:Ny
        switch k
        case 1 % Third row, actual data
            dd=Y(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn='data';
        case 2 %Fourth row: one-ahead data predictions
            dd=model{1}.oneAheadOut(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn={'MLE prediction';'(one-step ahead)'};
        case 3 % Fifth row:  data residuals (checkerboards)
            dd=model{1}.oneAheadRes(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn='residual';
        end

        subplot(Nx,Ny,i+(1+2*k)*Ny+[0,Ny])
        try
            imagesc(reshape(mean(dd,2),12,size(Y,1)/12)')
        catch
            imagesc(mean(dd,2))
        end
        %if i==1
        set(gca,'XTick',[],'YTick',yt,'YTickLabel',ytl,'FontSize',fs)
    %else
%set(gca,'XTick',[],'YTick',yt,'YTickLabel',[],'FontSize',fs)
%    end
        ax=gca;

        colormap(flipud(map))
%         caxis([-aC aC])
        caxis([-1 1])
        axis tight
        if k==1
            %title(['Output at t=' num2str(viewPoints(i))])
%             txt={'early adap (1-4)','late adap (last 4)','early wash (1-5)','early(ish) wash. (26-30)','mid wash. (201-205)'};
%             txt={'base late','early adap','late adap','early wash','mid wash','late wash'};
            txt={'Base Late','Early Adapt','Late Adap','Early Post','Mid Post','Late Post'};
            title(txt{i})
            ax=gca;
            ax.Title.FontSize=10;
        end
        if k==3
            title(['RMSE=' num2str(sqrt(mean(mean(dd.^2,1),2)))]) %/sqrt(meanVar)
        end
        if i==1
            ylabel(nn)
            ax=gca;
            ax.YAxis.Label.FontWeight='normal';
            ax.YAxis.Label.FontSize=12;
        end
    end
end

% Sixth row: residual RMSE, Smoothed, first PC of residual, variance by itself
Ny=1;
subplot(Nx,Ny,1+9*Ny)
hold on
dd=model{1}.oneAheadRes;
%dd=Y-CD*projY;
aux1=sqrt(mean(dd.^2)); %/sqrt(meanVar);
binw=5;
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
p1=plot(aux1,'LineWidth',2,'DisplayName',[num2str(size(model{1}.C,2)) '-state model']);
%title('MLE one-ahead output error (RMSE, mov. avg.)')

%Add previous stride model:
ind=find(diff(U(1,:))~=0);
Y2=Y;
Y(:,ind)=nan;
aux1=(Y(:,2:end)-Y(:,1:end-1));%/sqrt(2);
aux1=sqrt(mean(aux1.^2)); %/sqrt(meanVar);
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'LineWidth',1,'DisplayName','Prev. datapoint','Color',.5*ones(1,3)) ;
ylabel({'residual';' RMSE'})
ax=gca;
ax.YAxis.Label.FontSize=12;
ax.YAxis.Label.FontWeight='normal';
ax.YTick=[1:3];

%Add flat model:
aux1=Y2-(Y2/U)*U;
aux1=sqrt(mean(aux1.^2));%/sqrt(meanVar);
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux1,'LineWidth',1,'DisplayName','Flat','Color','k') ;


%Add data reproduce (05/03/2022)
C=model{1}.C;
Cinv=pinv(C)';
xhat = Y2'*Cinv; %x= y/C
yhat= C * xhat' ; %yhat = C 
% for step= 3:size(Y2,2)
%     RMSE(step,1) = sqrt(mean((Y2(:,step) - yhat(:,step)).^2));
% end

 RMSE = sqrt(mean((Y2(:,4:end) - yhat(:,4:end)).^2));
plot(movmean(RMSE,5),'LineWidth',1,'DisplayName','Regression Model','Color','#EDB120') ;
legend('Location','NorthEastOutside','AutoUpdate','off')
yl=ax.YAxis.Limits;
pp=patch([50 950 950 50],[0 0 .6 .6],.7*ones(1,3),'FaceAlpha',.5,'EdgeColor','none');
uistack(pp,'bottom')
ax.YAxis.Limits=yl;
yticks('auto')
% axis tight
grid on 

% ylim([0 1])
% set(gca,'YScale','log')


subplot(Nx,Ny,2+9*Ny)

hold on 
% LTI - SSM
aux1= 1 - sum(dd.^2)./sum((Y2(:,2:end)- mean(Y2(:,2:end))).^2);
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'LineWidth',2,'DisplayName',[num2str(size(model{1}.C,2)) '-state model']);
aux1=rsqr_uncentered(Y2(:,4:end)',model{1}.out(:,4:end)');
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'LineWidth',1,'DisplayName','LTI-SSM Uncentred','Color','g') ;


% Regression 
aux1 = 1 - sum((Y2- yhat).^2)./sum((Y2- mean(Y2)).^2);
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'LineWidth',1,'DisplayName','Regression Model','Color','#EDB120') ;
ylabel({'R^{2}'})
aux1=rsqr_uncentered(Y2(:,4:end)',yhat(:,4:end)');
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'--','LineWidth',1,'DisplayName','Regression Model Uncentred','Color','r') ;
grid on
ax.YAxis.Label.FontSize=12;
legend('Location','NorthEastOutside','AutoUpdate','off')
pp=patch([50 950 950 50],[0 0 1 1],.7*ones(1,3),'FaceAlpha',.5,'EdgeColor','none');
uistack(pp,'bottom')

%subplot(Nx,Ny,2+9*Ny)
%[pp,cc,aa]=pca((dd'),'Centered','off');

%%pp - data projected in the principal component 
%%cc - Corresponding matrix of eigenvectors
%%aa - vector of eigent values 
%hold on
%aux1=conv(cc(:,1)',ones(1,binw)/binw,'valid');
%plot(aux1,'LineWidth',1) ;
%title('First PC of residual, mov. avg.')
%grid on

%subplot(Nx,Ny,3+9*Ny)
%hold on
%aux1=conv2(Y,[-.5,1,-.5],'valid'); %Y(k)-.5*(y(k+1)+y(k-1));
%aux1=sqrt(sum(aux1.^2))/sqrt(1.5);
%aux1=aux1./(8.23+sqrt(sum(Y(:,2:end-1).^2))); %
%aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
%plot(aux1,'LineWidth',1) ;
%title('Instantaneous normalized std of data')
%grid on
%set(gca,'YScale','log')

end
%% Save fig
fName='OpenSans';
txt=findobj(gcf,'Type','Text');
set(txt,'FontName',fName);
ax=findobj(gcf,'Type','Axes');
set(ax,'FontName',fName);
for i=1:length(ax)
    ax(i).Title.FontWeight='normal';
end
