function [x_plus P_plus] = GetFilteredStateAndCovariance(P_minus,C,Sigma_y,x_minus,y_t_plus_1)

K = P_minus*C'/(C*P_minus*C'+Sigma_y);
x_plus = x_minus + K*(y_t_plus_1 - C*x_minus);
P_plus = (eye(size(P_minus))-K*C)*P_minus;