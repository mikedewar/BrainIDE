function z = Define2DGaussian(xmu, ymu, var, rho,SpacePoints,SpaceMin,SpaceMax)

% make a contour plot and a surface plot of a 2D Gaussian 
% xmu, ymu - mean of x, y 
% xvar, yvar - variance of x, y 
% rho correlation coefficient between x and y

sd = sqrt(var); % std deviation on x axis 
% ysd = sqrt(yvar); % std deviation on y axis
if (abs(rho) >= 1.0)
    disp('error: rho must lie between -1 and 1');
    return 
end
covxy = rho*sd*sd;                % calculation of the covariance
C = [var covxy; covxy var];   % the covariance matrix 
A = inv(C);                                 % the inverse covariance matrix

x = linspace(SpaceMin,SpaceMax,SpacePoints);    % xmu-PlotWidth*maxsd:0.1:xmu+PlotWidth*maxsd; % location of points at which x is calculated 
[X, Y] = meshgrid(x,x);                                         % matrices used for plotting

% Compute value of Gaussian pdf at each point in the grid 
z =  exp(-(A(1,1)*(X-xmu).^2 + 2*A(1,2)*(X-xmu).*(Y-ymu) + A(2,2)*(Y-ymu).^2));

