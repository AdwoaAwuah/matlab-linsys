function [fh,model] = legacy_vizSingleModel_FreeModel_ShortAdaptation(singleModel,Y,U,lower)

%Funtion to plot the time courses of the hidden states, data, fit and the
%residual of the model
%This function was initially created to plot 


%also see legacy_vizSingleModelMLMC

if isa(singleModel,'linsys')
    singleModel=struct(singleModel);
end

M=size(zeros(2,2),1);
fh=figure('Units','Normalized','OuterPosition',[0 0 .5 1],'Color',ones(1,3));
if nargin>1
    Nx=12;
else
    Nx=4;
end
singleModel.B=zeros(2,2);
Nu=size(singleModel.B,2);
% Nu=[2];
Ny=M+1+Nu;
model{1}=singleModel;
Nu=size(model{1}.B,2);
% Nu=size(model{1}.C,2);
Nc=size(Y,1);
model{1}.D=zeros(size(model{1}.C,1),size(U,1));
if nargin<3
    lower=false;
end
%% First: normalize model, compute basic parameters of interest:
if nargin<3
    U=[zeros(Nu,100) ones(Nu,1000)]; %Step response
end
if lower==1
    
    %     C=model{1}.C; %getting C from the struture
    %     Cinv=pinv(C)'; %pseudoinverse of the C (C is not a squared matrix)
    %     X = Y'*Cinv; %x= y/C getting the dynamics of the hidden states (we are using least-sqaures seee: Penrose, Roger (1956))
    C=model{1}.C;
    if size(C,2)==1
         X=1*(ones(size(C,2),size(Y,2)))' ;
    else
        
        X=0.5*(ones(size(C,2),size(Y,2)))' ;
    end
%     X(:,1)=ones(length(Y),1);
    Y2= C * X' ; % Data reconstructed with the perdetermine dynamics
    model{1}.Data=Y;
    model{1}.States=X;
    model{1}.Out=Y2;
    model{1}.Res=Y-Y2;
    

elseif lower==2
    [pp,cc,aa]=pca(Y');%,'Centered','off');
    PC= [(cc(:,1:2)*pp(:,1:2)') + nanmean(Y')]';
    X=cc(:,1:2);
    C=pp(:,1:2);
    % Winv=pinv(W);
    % dynamics= Y'*Winv';
    % PC= W * dynamics' ;
%     ResPC2= Y - PC;
    
    model{1}.Data=Y;
    model{1}.States=cc(:,1:2);
    model{1}.Out=PC;
    model{1}.Res=Y - PC;
    
    
else
    C=model{1}.C; %getting C from the struture
    Cinv=pinv(C)'; %pseudoinverse of the C (C is not a squared matrix)
    X = Y'*Cinv; %x= y/C getting the dynamics of the hidden states (we are using least-sqaures seee: Penrose, Roger (1956))
%     X(:,1)=ones(length(Y),1);
    Y2= C * X' ; % Data reconstructed with the perdetermine dynamics
    model{1}.Data=Y;
    model{1}.States=X;
    model{1}.Out=Y2;
    model{1}.Res=Y-Y2;
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
% ytl={'HIP','GLU','TFL','RF','VL','VM','SEMT','SEMB','BF','MG','LG','SOL','PER','TA'};
ytl={'TA', 'PER', 'SOL', 'LG', 'MG', 'BF', 'SEMB', 'SEMT', 'VM', 'VL', 'RF', 'HIP','TFL', 'GLU'};
ytl(end:-1:1) = ytl(:);
% muscleOrder={'TA', 'PER', 'SOL', 'LG', 'MG', 'BF', 'SEMB', 'SEMT', 'VM', 'VL', 'RF', 'TFL', 'GLU', 'HIP'};
% ytl={'TA', 'PER', 'SOL', 'LG', 'MG', 'BF', 'SEMB', 'SEMT', 'VM', 'VL', 'RF', 'TFL','HIP', 'GLU'}
yt=1:length(ytl);
fs=7;
% STATES
CD=[model{1}.C model{1}.D];
XU=[X';U];

rotMed='none'; % I am not rotating the data 

[CDrot,XUrot]=rotateFac(CD,XU,rotMed);
if strcmp(rotMed,'none')
    factorName=[strcat('C_',num2str([1:size(model{1}.C,2)]'));strcat('D_',num2str([1:size(model{1}.D,2)]'))];
    latentName=[strcat('State ',' ', num2str([1:size(model{1}.C,2)]'));strcat('Input ',' ', num2str([1:size(model{1}.D,2)]'))];
else
    factorName=strcat('Factor ',num2str([1:size(CD,2)]'));
    latentName=strcat('Latent ',num2str([1:size(CD,2)]'));
end
aC=prctile(abs(Y(:)),98);
% CDiR=CDrot'*inv(model{1}.R);
% CDiR=CDrot';
% CDiRCD=CDiR*CDrot;
% projY=CDiRCD\CDiR*Y;
projY=XU;
for i=1:size(CD,2)-1
   
    hold on
    if i<size(CD,2)-1
        ph(i)=subplot(Nx,size(CD,2)-2,i); %TOP row: states temporal evolution and data projection
        scatter(1:size(Y,2),projY(i,:),5,.7*ones(1,3),'filled')
        title(latentName(i,:))
%         p(i)=plot(XUrot(i,:),'LineWidth',2,'Color','k');
        ax=gca;
        ax.Position=ax.Position+[0 .045 0 0];
        axis tight
        grid on
        set(gca,'ColorOrderIndex',1)
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
    colormap(flipud(map))
%     colormap(map)
%     caxis([-aC aC])
    caxis([-1 1])
    axis tight
    title(factorName(i,:))
    ax=gca;
    ax.Position=ax.Position+[0 .03 0 0];
    hold on
    aa=axis;
    plot([0.1 1.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
    text(.8,17,'DS','FontSize',6)
    plot([2.1 5.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
    text(2.8,17,'SINGLE','FontSize',6)
    plot([6.1 7.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
    text(6.9,17,'DS','FontSize',6)
    plot([8.1 11.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
    text(9,17,'SWING','FontSize',6)
    axis(aa)
    

end
subplot(Nx,Ny,Ny+i+[0,Ny])% Second row: checkerboards
caxis([-1 1])
colorbar
colorbar('Ticks',[-1,1],'TickLabels',{'<Dom','<Non-dm'})
ax=gca;
ax.Position=ax.Position+[0 .03 0 0];
hold on
aa=axis;
plot([0.1 1.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
text(.8,17,'DS','FontSize',6)
plot([2.1 5.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
text(2.8,17,'SINGLE','FontSize',6)
plot([6.1 7.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
text(6.9,17,'DS','FontSize',6)
plot([8.1 11.9]+.5, [15 15],'k','LineWidth',2,'Clipping','off')
text(9,17,'SWING','FontSize',6)
axis(aa)
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
viewPoints=[3,35,44,470,483,675]; %PATR - PATS 
% viewPoints=[3,437,443,635]; %PATR - PATS 
binw=4; %Plus minus 2
viewPoints(viewPoints>N-binw/2)=[];
Ny=length(viewPoints);
M=length(model);
dd=Y(:,2:39);
dd=dd-mean(dd,2); %Baseline residuals under flat model
meanVar=1;%mean(sum(dd.^2,1),2);
for k=1:3
    for i=1:Ny
        switch k
        case 1 % Third row, actual data
            dd=Y(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn='data';
        case 2 %Fourth row: one-ahead data predictions
            dd=model{1}.Out(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn={'Data Fit'};
        case 3 % Fifth row:  data residuals (checkerboards)
            dd=model{1}.Res(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn='residual';
        end

        subplot(Nx,Ny,i+(1+2*k)*Ny+[0,Ny])
        try
            imagesc(reshape(median(dd,2),12,size(Y,1)/12)')

        catch
            imagesc(median(dd,2))
        end
        %if i==1
        
        %         if k==2
%         fdr=.063;
%         [pvalc,hc,alphaAdj_c]=checkerstatsV2(reshape(dd,12,14,5),[],1,0,fdr,'benhoch',0);%mindif has to be zero, since signrank cannot reliably do a two-tail test agains another value
% %         [pvalc,hc,alphaAdj_c]=checkerstatsV2(reshape(dd,12,14,5),[],1,0,fdr,'none',0);%mindif has to be zero, since signrank cannot reliably do a two-tail test agains another value
%         hold on
%         dataEcmed=reshape(nanmean(dd,2),12,14)';
%         for ii=1:size(hc,1)
%             for kk=1:size(hc,2)
%                 if hc(ii,kk)==1  && abs(dataEcmed(ii,kk))>0.1 %We use 0.1 to measure 10% changes from baseline behavior as a significant change %since statistical testing was done againts zero, amplitude testing happens here
%                     plot3(kk,ii,1,'.','MarkerSize',6,'Color','k')
%                     
%                 end
%             end
%         end
            
%         end
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
            title(['Output at t=' num2str(viewPoints(i))])%,[' p-th=',num2str(round(100*alphaAdj_c)/100) ])
%             txt={'Base Late','Early Adapt','Late Adap','Early Post','Mid Post','Late Post'};
%             title(txt{i})
            ax=gca;
            ax.Title.FontSize=10;
        end
        if k==2
%             title(['p-th=',num2str(round(1000*alphaAdj_c)/1000) ])
            %             txt={'Base Late','Early Adapt','Late Adap','Early Post','Mid Post','Late Post'};
            %             title(txt{i})
            ax=gca;
            ax.Title.FontSize=10;
        end
        
        
        if k==3
%             title(['RMSE=' num2str(sqrt(mean(sum(dd.^2,1),2)))])
            title(['RMSE=' num2str(sqrt(mean(sum((dd.^2),1),2))/sqrt(meanVar))]);%,['p-th=',num2str(round(1000*alphaAdj_c)/1000)])
%              title(['normalized RMSE=' num2str(sqrt(mean(sum(dd.^2,1),2))/sqrt(meanVar))])
%              title(['RRMSE=' num2str(sqrt(mean((dd.^2),1)))])
%             title(['RMSE=' num2str(sqrt(mean(mean(dd,2).^2)))])
        end
        if i==1
            ylabel(nn)
            ax=gca;
            ax.YAxis.Label.FontWeight='normal';
            ax.YAxis.Label.FontSize=12;
        end
    end
end

% PCA analysis for upper bound 
Y2=Y;
% adapt=Y(:,41:480);
% post=Y(:,481:491);
% [pp,cc,aa]=pca(adapt','Centered','off');
% [pp_2,cc_,aa_]=pca(post','Centered','off');
% [coeff,score,latent,tsquared,explained,mu]=pca(Yasym,'Centered','off');
%%Input has to be row observation  and columns variables 
%%pp - Corresponding matrix of eigenvectors
%%cc - data projected in the principal component 
%%aa - vector of eigent values
% 
% W=[pp(:,1:2) pp_2(:,1)];
% Winv=pinv(W);
% dynamics= Y'*Winv';
% PC= W * dynamics' ; 
% Res= Y - PC;

[pp,cc,aa]=pca(Y');%,'Centered','off');
PC= [(cc(:,1:2)*pp(:,1:2)') + nanmean(Y')]'; 
% [pp,cc,aa]=pca(Y','Centered','off');
% Winv=pinv(W);
% dynamics= Y'*Winv';
% PC= W * dynamics' ;
ResPC2= Y - PC;

% Res=PC- movmean(Y,k);

% 2 states 
C=C(:,1:2); %getting C from the struture
Cinv=pinv(C)'; %pseudoinverse of the C (C is not a squared matrix)
X = Y'*Cinv; %x= y/C getting the dynamics of the hidden states (we are using least-sqaures seee: Penrose, Roger (1956))
Y_2= C* X' ; % Data reconstructed with the perdetermine dynamics
Res_2=Y-Y_2;
% 
% % Instantaneous SD
Res_3= conv2(Y,[0,1,-1],'valid'); %Y(k)-.5*(y(k+1)+y(k-1));
% Res_3= conv2(Yasym,[0,1,-1],'valid'); %Y(k)-.5*(y(k+1)+y(k-1));

% Sixth row: residual RMSE, Smoothed, first PC of residual, variance by itself
% RMES
Ny=1;
subplot(Nx,Ny,1+9*Ny)
hold on
dd=model{1}.Res;
%dd=Y-CD*projY;
aux1=sqrt(sum(dd.^2))/sqrt(meanVar);


binw=5;
% figure
hold on 
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'LineWidth',2,'DisplayName','3 states','Color','r');
 
% aux2=sqrt(sum(Res.^2))/sqrt(meanVar);
% aux2=conv(aux2,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux2,'LineWidth',2,'DisplayName','PCA ','Color','k');

aux2=sqrt(sum(ResPC2.^2))/sqrt(meanVar);
aux2=conv(aux2,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux2,'LineWidth',2,'DisplayName','PCA_{all-data}','Color','k');%,"#A2142F");
% 
aux3=sqrt(sum(Res_2.^2))/sqrt(meanVar);
aux3=conv(aux3,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux3,'-.','LineWidth',2,'DisplayName','2 states','Color','b');
% 
aux3=sqrt(sum(Res_3.^2))/sqrt(2);
aux3=conv(aux3,ones(1,binw)/binw,'valid'); %Smoothing
plot([nan nan aux3],'-','LineWidth',2,'DisplayName','Instantaneous SD','Color',"#0072BD");
ylabel({'residual';' RMSE'})

% aux3=sqrt(sum(Res_4.^2))/sqrt(k);
% aux3=conv(aux3,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux3,'-.','LineWidth',2,'DisplayName',['2 STD'],'Color',"#D95319");
% ylabel({'residual';' RMSE'})
%title('MLE one-ahead output error (RMSE, mov. avg.)')

%Add previous stride model:
% ind=find(diff(U(1,:))~=0);
% % Y(:,ind)=nan;
% aux1=(Y(:,2:end)-Y(:,1:end-1))/sqrt(2);
% aux1=sqrt(mean(aux1.^21sqrt(meanVar);
% % aux1=sqrt(mean(aux1.^2));%/sqrt(meanVar);
% aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
% % plot(aux1,'LineWidth',1,'DisplayName','Te','Color',.5*ones(1,3)) ;

% ax=gca;
% ax.YAxis.Label.FontSize=12;
% ax.YAxis.Label.FontWeight='normal';
% ax.YTick=[1:3];

%Add flat model:
% aux1=Y-(Y/U)*U;
% aux1=sqrt(mean(aux1.^2));%/sqrt(meanVar);
% aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux1,'LineWidth',1,'DisplayName','Flat','Color','k') ;


%Add data reproduce (05/03/2022)
if lower==1
    yhat=model{1}.Out;
else
%     C=model{1}.C;
%     Cinv=pinv(C)';
%     xhat = Y2'*Cinv; %x= y/C
    yhat= model{1}.Out; %C * xhat' ; %yhat = C
    
end

% for step= 1:size(Y2,2)
% 
%     cross(step,1) = corr(Y2(:,step),yhat(:,step));
%     cross2(step,1) = corr(Y2(:,step),PC(:,step));
%     cross3(step,1) = corr(Y2(:,step),Y_2(:,step));
% end

legend('Location','NorthEastOutside','AutoUpdate','off')
yl=ax.YAxis.Limits;
pp=patch([40 480 480 40],[0 0 max(aux1) max(aux1)],.7*ones(1,3),'FaceAlpha',.5,'EdgeColor','none'); %PATR - PATS 
uistack(pp,'bottom')
ax.YAxis.Limits=yl;
axis tight
yticks('auto')
grid on
% ylim([0 1])
% set(gca,'YScale','log')

subplot(Nx,5,51:54)
hold on

aux1 = 1 - sum((Y2- yhat).^2)./sum((Y2- mean(Y2)).^2);
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'LineWidth',2,'DisplayName','3 states','Color','r') ;


aux2= 1 - sum((Y2- PC).^2)./sum((Y2- mean(Y2)).^2);
aux2=conv(aux2,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux2,'LineWidth',2,'DisplayName','PCA','Color','k') ;

aux3= 1 - sum((Y2- Y_2).^2)./sum((Y2- mean(Y2)).^2);
aux3=conv(aux3,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux3,'-.','LineWidth',2,'DisplayName','2 States','Color','b') ;
ylabel({'R^{2}'})
grid on
ax.YAxis.Label.FontSize=12;
legend('Location','NorthEastOutside','AutoUpdate','off')
pp=patch([40 480 480 40],[0 0 1 1],.7*ones(1,3),'FaceAlpha',.5,'EdgeColor','none'); %PATR - PATS 
uistack(pp,'bottom')
axis tight
yticks('auto')

% subplot(Nx,Ny,3+9*Ny)
% hold on 
% 
% % aux1 = 1 - sum((Y2- yhat).^2)./sum((Y2- mean(Y2)).^2);
% aux1=conv(cross,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux1,'LineWidth',2,'DisplayName','3 states','Color','r') ;
% 
% aux2=conv(cross2,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux2,'--','LineWidth',2,'DisplayName','PCA','Color','k') ;
% 
% aux3=conv(cross3,ones(1,binw)/binw,'valid'); %Smoothing
% plot(aux3,'-.','LineWidth',2,'DisplayName','2 states','Color','b') ;
% ylabel({'Pearson'; 'Correlation (r)'})
% 
% grid on
% ax.YAxis.Label.FontSize=12;
% legend('Location','NorthEastOutside','AutoUpdate','off')
% yline(nanmean(aux1(1:40)))
% pp=patch([40 480 480 40],[0 0 1 1],.7*ones(1,3),'FaceAlpha',.5,'EdgeColor','none'); %PATR - PATS 
% uistack(pp,'bottom')
% axis tight
% yticks('auto')

subplot(Nx,5,56:59)
[pp,cc,aa]=pca(dd');%,'Centered','off');
% [pp,cc,aa]=pca((dd(:,481:491)'),'Centered','off');
% [coeff,score,latent,tsquared,explained,mu]=pca(Yasym,'Centered','off');
%%Input has to be row observation  and columns variables 
%%pp -  Corresponding matrix of eigenvectors 
%%cc - data projected in the principal component 
%%aa - vector of eigent values 
hold on
aux1=conv(cc(:,1)',ones(1,binw)/binw,'valid');
plot(aux1,'LineWidth',1) ;
title('First PC of residual, mov. avg.')
grid on
hold on 
subplot(Nx,5,[55 60])
imagesc(reshape(pp(:,1),12,14)')
colormap(flipud(map))
caxis([-1 1])
set(gca,'XTick',[],'YTick',yt,'YTickLabel',ytl,'FontSize',fs)

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
