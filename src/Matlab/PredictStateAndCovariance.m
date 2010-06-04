% calculate predicted state and covariance

function [x_minus P_minus] = PredictStateAndCovariance(Wm,Wc,Q_X,Sigma_e)

x_minus = Q_X*Wm;
temp = Q_X - repmat(x_minus,1,size(Q_X,2));
temp2 = zeros(size(Q_X,2),size(Q_X,1),size(Q_X,1));
for n=1:size(Q_X,2)
    temp2(n,:,:) = Wc(n)*temp(:,n)*temp(:,n)';
end
P_minus  =squeeze(sum(temp2,1)) + Sigma_e;