function [P_b_t x_b_t] = GetSmoothedStateAndCovariance(M_t_plus_1, P_b_minus_t_plus_1, ...
                x_f_t, x_b_t_plus_1, x_b_minus_t_plus_1, P_b_t_plus_1,P_f_t)
            
S_t = M_t_plus_1/P_b_minus_t_plus_1;
x_b_t = x_f_t + S_t*(x_b_t_plus_1-x_b_minus_t_plus_1);
P_b_t = P_f_t + S_t*(P_b_t_plus_1-P_b_minus_t_plus_1)*S_t';


