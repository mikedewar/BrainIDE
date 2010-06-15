
function M_t_plus_1 = CalcCrossCovariance(Wc,X_b_minus_t, x_f_t, X_b_minus_t_plus_1, x_b_minus_t_plus_1)

temp1 = X_b_minus_t - repmat(x_f_t,1,size(X_b_minus_t,2));
temp2 = X_b_minus_t_plus_1 - repmat(x_b_minus_t_plus_1,1,size(X_b_minus_t_plus_1,2));
temp3 = zeros(length(Wc),size(X_b_minus_t,1),size(X_b_minus_t,1));      % initialise for speed
for n=1:length(Wc)
    temp3(n,:,:) = Wc(n)*temp1(:,n)*temp2(:,n)';
end
M_t_plus_1 = squeeze(sum(temp3,1));

