%% A 2D inversion code for self-potential data/gradient data 
%% based on a hybrid sensitivity and depth weighing scheme

clc; 
close all; 
clear all;

%% mesh initialization

NX = 50;               % the number of grids in horizontal direction；
NY = 25;               % the number of grids in vertical direction；
NE = NX*NY;            % the number of total elements；
ND = (NX+1)*(NY+1);    % the number of total nodes；
noise = 0.05;          % the noise level

%% current sources initialization

sources1=[];
for i=12:15
    index=(NX/2+i*(NX+1)-13):1:(NX/2+i*(NX+1)-6);
    sources1 = [sources1;index];
end

sources2=[];
for i=12:15
    index=(NX/2+i*(NX+1)+7):1:(NX/2+i*(NX+1)+14);
    sources2 = [sources2;index];
end

P0=zeros(ND,1);
P0(sources1(:,:))=-10;
P0(sources2(:,:))=10;
P0_1=reshape(P0,NX+1,NY+1);

%% conductivity initialization

% set the background conductivity
SGM = zeros(NE,1);   
sigma_backg = 0.01;    
SGM(:) = sigma_backg;   

%% set the conductivity anomalies for sources
% for source1
for i=13:15
    for j=NX/2-13:NX/2-7
        SGM((i-1)*NX+j)=1;
    end
end
% for source2
for i=13:15
    for j=NX/2+7:NX/2+13
        SGM((i-1)*NX+j)=1;
    end
end

SGM = reshape(SGM,NX,NY);

%% the given wave-numbers and related weights

kbo=[0.004758,0.0407011,0.1408855,0.393225,1.088038];
gg=[0.0099472,0.031619,0.0980327,0.2511531,0.7260814];

%% set the projection matrix to calculate the data on the ground surface

% for SP data
% C=zeros(NX+1,ND);             
% for i=1:NX+1
%     C(i,i)=1;
% end

% for SP gradient data
C=zeros(NX,ND);             
for i=1:NX
    C(i,i)=-1;
    C(i,i+1)=1;
end

%% call the SP forward modeling based on FEM

[UU,UU1,J]=Forward_main(P0,NX,NY,NE,ND,SGM,kbo,gg,C);
d=UU1;
dclean=d;

%% add noises and obtain the data weighting matrix Wd

d_noise=zeros(length(d),1);
noiseratio=randn(length(d),1);
noiseratio=noiseratio/max(abs(noiseratio));
stdratio=std(noiseratio);
d_noise=d.*(1+noise*noiseratio);

Wd=zeros(length(C(:,1)));        
for i=1:length(d)
    Wd(i,i)=1/(d(i)*noise*stdratio);
end

%% simulate the reference position 
%% add noises before this step, or there will be a 0 element in Wd

% only for SP data inversion 
% d_noise=d_noise-d_noise(1);
% d_noise=d_noise-d_noise(end);

%% initial model to avoid singularity
x0=zeros(ND,1);
x0(:,:)=eps;      

%% inversion

rms_rec=[100];
target=1.0;        % target RMSE
maxit=4;           % the maximum iteration number

lamda=1; 
beta=0.5;  
q=0.4;
x=x0;

Wm=zeros(ND,ND);
for i=1:ND
    Wm(i,i)=1;
end

% depth weighting matrix
Wz=zeros(ND,ND);
for i=1:NY+1
    for j=1:NX+1
        index=(i-1)*(NX+1)+j;
        Wz(index,index)=1/(i.^q);
    end
end

% inversion iteration
for i=1:maxit
    [~,d_pred]=Forward_main(x,NX,NY,NE,ND,SGM,kbo,gg,C);  % 调用正演函数计算先验数据
        misfit_old=sqrt((norm(Wd*(d_noise-d_pred),2)).^2/length(d)); % 求初始拟合差

    
    if misfit_old<target                                 
            disp('the inversion has converged！');
            disp('the number of iteration is：');
            disp(i-1);
            break;
    end

    % sensitivity matrix
    K1=J;

    % choose a weighting scheme from the following methods

    % sensitivity weighting 
    W=diag(sum(K1.^2,1));     
    W=W.^0.5;  

    % hybrid weighting 
    W=W.*Wz;
    
    % depth weighting
    % W=Wz;

    Ohm=zeros(ND,ND);

    % the minimum norm solution
    if i==1
        for ii=1:ND
            Ohm(ii,ii)=sqrt(W(ii,ii)^2);    
        end
    end
    
    % focusing iteration
    if i>1
        for ii=1:ND
            Ohm(ii,ii)=sqrt(W(ii,ii)^2/(x(ii)^2+beta^2));    
        end
    end

    K2=K1/Ohm;
    KK=C*K2;

    % the weighted model parameters
    x_new=(KK'*(Wd'*Wd)*KK+lamda*(Wm'*Wm))\(KK'*(Wd'*Wd)*d_noise);   
    % the real model parameters
    x_new=Ohm\x_new;  

    [~,d_pred]=Forward_main(x_new,NX,NY,NE,ND,SGM,kbo,gg,C);    
    rmse=sqrt((norm(Wd*(d_noise-d_pred),2)).^2/length(d))

    if rmse<target                                 
        disp('the inversion has converged！');
        x=x_new;
        disp('the number of iteration is：');
        disp(i);
        break;
    else
        x=x_new;
        xref=x_new;
    end

    lamda=lamda*0.1;   
    beta=beta/2;   
   
end


[~,d_pred]=Forward_main(x,NX,NY,NE,ND,SGM,kbo,gg,C);   
rmse=sqrt(norm((d_noise-d_pred)/noise,2)/(NX+1)); 
P=x;
P_1=reshape(P,NX+1,NY+1);

%% plot the results
x = -NX/2:1:NX/2;    
y = 0:1:NY;

% data
figure(1) 
plot(x(1:length(d)),d_noise,'-rd','LineWidth',1.5);       
hold on
plot(x(1:length(d)),d_pred,'-bs','LineWidth',1.5);     % 
hold on
plot(x(1:length(d)),dclean,'-k+','LineWidth',1.5);
xlabel('Distance/m','fontsize',16);
ylabel('Voltage/mV','fontsize',16);
o=legend(' Observed data',' Predicted data',' Clean data','location','northeast');
set(o,'box','off');   
set(gca,'fontsize',16); 
set (gca,'position',[0.16,0.17,0.81,0.79] );     
box on
hold off

% sources
figure(2)
subplot(121)
imagesc(P0_1')
colormap("jet")
set(gca,'fontsize',16);
h=colorbar;               
set(get(h,'Title'),'string','mA');
meshgrid(x,y);
pcolor(x,y,P0_1');            
hold on
xlabel('Distance/m','fontsize',16);     
ylabel('Depth/m','fontsize',16);
set(gca,'fontsize',16);
set(gca,'XAxisLocation','top');  
set(gca,'YDir','reverse');       
shading interp;
h=colorbar;               
set(get(h,'Title'),'string','mA');
P_max=max(max(P0_1));
P_min=min(min(P0_1));
caxis([P_min P_max]);     
contour(x,y,P0_1',5,'k-');
axis equal;
hold off
box on

subplot(122)
meshgrid(x,y);
pcolor(x,y,P_1');           
caxis([-10 10])
hold on
xlabel('Distance/m','fontsize',16);     
ylabel('Depth/m','fontsize',16);
set(gca,'fontsize',16);
set(gca,'XAxisLocation','top');  
set(gca,'YDir','reverse');       
shading interp;
h=colorbar;              
set(get(h,'Title'),'string','mA');
P_max=max(max(P_1));
P_min=min(min(P_1));
contour(x,y,P_1');
colormap(jet);
axis equal;
hold off
box on
set(gca,'XAxisLocation','top');  
set(gca,'YDir','reverse');       


%% 2D SP forward modeling main fuction based on FEM
function [UU,UU1,J]=Forward_main(P,NX,NY,NE,ND,SGM,kbo,gg,C)
sources=find(P~=0);

% calculate the SP response with different wave-numbers
U=zeros(length(kbo),ND);
J=zeros((NX+1)*(NY+1));
for nk=1:length(kbo)
    KBO=kbo(nk);
    [XY,I4,~,~]=Forward_XZI4(NX,NY,NE,ND);                % call Forward_XZI4 subfunction to obtain node index
    [K]=Forward_analysis(NX,NY,ND,XY,I4,KBO,SGM,sources); % call Forward_analysis subfunction to obtain stiffness matrix
    u=K\P;                                                % solve linear equation
    J=J+inv(K)*gg(nk);
    for i=1:ND
        U(nk,i)=u(i);                                        
    end
end

%% inverse Fourier Transform
UU=zeros(ND,1);

for i=1:ND
    for j=1:length(kbo)
        UU(i)=UU(i)+U(j,i)*gg(j);  
    end
end

% the data on the ground surface
UU1=C*UU;      
end

%% 
function [XY,I4,X,Y] = Forward_XZI4(NX,NY,NE,ND);
% XY——A two-dimensional array, store the x and y coordinates of each node
% I4——A two-dimensional array, store the node numbers of grids
% X，Y——the x and y coordinates

XY=zeros(ND,2);
I4=zeros(NE,4);
X = 0:1:NX;     % the step represents the length of grid in the horizontal direction
Y = 0:1:NY;     % the step represents the length of grid in the vertical direction

% obtain the coordinate of each node
for i=1:NY+1
    for j=1:NX+1
        N=(i-1)*(NX+1)+j;
        XY(N,1)=X(j);    
        XY(N,2)=Y(i);    
    end
end
k=0;

% indexs of the elements
for i=1:NY
    for j=1:NX
        N=(i-1)*NX+j;           
        I4(N,1)=N+k;            
        I4(N,2)=I4(N,1)+1;      
        I4(N,3)=I4(N,2)+(NX+1); 
        I4(N,4)=I4(N,1)+(NX+1); 
    end
    k=k+1;      
end
end

function [K]=Forward_analysis(NX,NY,ND,XY,I4,KBO,SGM,sources)

% Set the source location to the geometric center of anomalies
% a=0;b=0;
% for i=1:length(sources)
%     a=a+floor(sources(i)/(NX+1));  
%     b=b+mod(sources(i),(NX+1));    
% end
% a=round(a/length(sources));        
% b=round(b/length(sources));        
% source=a*(NX+1)+b;                 

% or set at the center of simulation space
source=round(((NX+1)*(NY+1))/2+(NX/2)+1);

% calculate the stiffness matrix
K=sparse(ND,ND);
for IZ=1:NY
    for IX=1:NX
        L=(IZ-1)*NX+IX; 
        sgm=SGM(L);                
        A=abs(XY(I4(L,2),1)-XY(I4(L,1),1)); 
        B=abs(XY(I4(L,4),2)-XY(I4(L,1),2)); 
        BA=B/A;
        AB=A/B;
        KE1(1,1)=sgm*(AB+BA)/3.0;
        KE1(2,1)=sgm*(BA-2.0*AB)/6.0;
        KE1(2,2)=KE1(1,1);
        KE1(3,1)=sgm*(-AB-BA)/6.0;
        KE1(3,2)=sgm*(-2.0*BA+AB)/6.0;
        KE1(3,3)=KE1(1,1);
        KE1(4,1)=KE1(3,2);
        KE1(4,2)=KE1(3,1);
        KE1(4,3)=KE1(2,1);
        KE1(4,4)=KE1(1,1);
        KE1(1,2)=KE1(2,1);
        KE1(1,3)=KE1(3,1);
        KE1(1,4)=KE1(4,1);
        KE1(2,3)=KE1(3,2);
        KE1(2,4)=KE1(4,2);
        KE1(3,4)=KE1(4,3);
      
        KE2= [4.0,2.0,1.0,2.0;2.0,4.0,2.0,1.0;1.0,2.0,4.0,2.0;2.0,1.0,2.0,4.0];
        KE2(:,:)=KE2(:,:)*A*B*sgm*KBO^2/36.0; 
        
        KK=[I4(L,1),I4(L,2),I4(L,3),I4(L,4)];  
       
        for m = 1:4
            for n = 1:4
                K = K + sparse(KK(m),KK(n),KE1(m,n),size(K,1),size(K,1));
                K = K + sparse(KK(m),KK(n),KE2(m,n),size(K,1),size(K,1));
            end
        end
        
    end
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    
end
% boundary condition

% left boundary
for i=1:NY
    L=1+(i-1)*NX; 
    r=sqrt((XY(source(1),1)-XY(I4(L,1),1))^2+(XY(source(1),2)-(XY(I4(L,1),2)+XY(I4(L,4),2))/2)^2);
    x=KBO*r;
    [K0,K1]=Forward_Bessel(x);
    belta=(abs(XY(I4(L,4),2)-XY(I4(L,1),2))*SGM(L)*K1*KBO*abs((XY(source(1),1)-XY(I4(L,1),1)))/r/(6*K0));
    ke3=[2,1,0,0;1,2,0,0;0,0,0,0;0,0,0,0];
    KE3=belta*ke3;
    KK=[I4(L,1),I4(L,4)];

    for m = 1:2
        for n = 1:2
            K = K + sparse(KK(m),KK(n),KE3(m,n),size(K,1),size(K,1));
        end
    end
end
    
% right boundary
for i=1:NY
    L=NX+(i-1)*NX;
    r=sqrt((XY(source(1),1)-XY(I4(L,2),1))^2+(XY(source(1),2)-(XY(I4(L,2),2)+XY(I4(L,3),2))/2)^2);
    x=KBO*r;
    [K0,K1]=Forward_Bessel(x);
    belta=(abs(XY(I4(L,2),2)-XY(I4(L,3),2))*SGM(L)*K1*KBO*abs((XY(source(1),1)-XY(I4(L,2),1)))/r/(6*K0));
    ke3=[2,1,0,0;1,2,0,0;0,0,0,0;0,0,0,0];
    KE3=belta*ke3;
    KK=[I4(L,2),I4(L,3)];

    for m = 1:2
        for n = 1:2
            K = K + sparse(KK(m),KK(n),KE3(m,n),size(K,1),size(K,1));
        end
    end
end

% bottom boundary
for i=1:NX
    L=NX*(NY-1)+i; 
    r=sqrt((XY(source(1),2)-XY(I4(L,3),2))^2+(XY(source(1),1)-(XY(I4(L,3),1)+XY(I4(L,4),1))/2)^2);
    x=KBO*r;
    [K0,K1]=Forward_Bessel(x);
    belta=(abs(XY(I4(L,4),1)-XY(I4(L,3),1))*SGM(L)*K1*KBO*abs((XY(source(1),2)-XY(I4(L,3),2)))/r/(6*K0));
     ke3=[2,1,0,0;1,2,0,0;0,0,0,0;0,0,0,0];
    KE3=belta*ke3;
    KK=[I4(L,3),I4(L,4)];

    for m = 1:2
        for n = 1:2
            K = K + sparse(KK(m),KK(n),KE3(m,n),size(K,1),size(K,1));
        end
    end
end
end

function [K0,K1]=Forward_Bessel(x);
K0=besselk(0,x);
K1=besselk(1,x);
end