clear all; close all; clc;
set(groot, 'defaultLineLineWidth', 1.5);
set(groot,'defaultLineMarkerSize',10);
set(groot,'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontSize',18);
set(groot,'defaultAxesTitleFontSizeMultiplier',1.1);
set(groot,'defaultLegendFontSize',15);


N = 64; w = 6; N = w * floor(64/w);
pts = dlmread('particles.txt'); [Np,~] = size(pts);
coords = dlmread('coords.txt');
Fe = dlmread('spread.txt'); 
fl = dlmread('forces.txt')
flinterp = dlmread('interp.txt')
Fex = Fe(:,1);
Fey = Fe(:,2);
Fez = Fe(:,3);

xE = coords(:,1);
yE = coords(:,2);
zE = coords(:,3);

figure(1)
plot3(pts(:,1),pts(:,2),pts(:,3),'ro','MarkerFaceColor','r','MarkerSize',10); grid on; hold on;
%plot3(xE,yE,Fex,'ro');
%plot3(xE,yE,zE,'go','MarkerSize',5);

%[xE,yE,zE] = ndgrid(0:(N-1));
cols = distinguishable_colors(N);
figure(1); 
quiver3(xE,yE,zE,Fex,Fey,Fez,5); hold on;% 'color',cols(j,:),'LineWidth',1);  hold on; grid on;
levellist = linspace(min(Fex(:)),max(Fex(:)),100);
Fex = reshape(Fex,N,N,N); xE = reshape(xE, N,N,N); yE = reshape(yE,N,N,N); zE = reshape(zE,N,N,N);
for i = 1:length(levellist)
    level = levellist(i);
    p = patch(isosurface(xE,yE,zE,Fex,level));
    p.FaceVertexCData = level;
    p.FaceColor = 'blue';
    p.EdgeColor = 'none';
    p.FaceAlpha = 0.3;
    axis tight;
    %camlight;
    lighting gouraud;
end
view(3)
xlabel('x'); ylabel('y'); zlabel('z');
up = max(xE(:));
axis([0,up,0,up,0,up]);