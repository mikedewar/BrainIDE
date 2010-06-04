function Q_X = Q(X,phi_unwrapped,Ts_invGamma_theta_phi_psi,Delta_squared,f_max,varsigma,v_0,xi)

% compute the approximated field
% phi^T(r)x_t
v_approx = X'*phi_unwrapped;

% for checking - to be removed later
% for n=1:size(X,2)
%     v_approx2(n,:) = X(:,n)'*phi_unwrapped;
% end
% figure
% imagesc(v_approx-v_approx2)

f = f_max./(1+exp(varsigma*(v_0-v_approx)))*Delta_squared;
% Ts_invGamma_theta_phi_psi = *Ts_invGamma_theta_phi_psi;

temp = zeros(size(X));
for n=1:size(f,1)
    temp(:,n) = Ts_invGamma_theta_phi_psi*f(n,:)';
end

% this part is also used for testing
% for n=1:163
%     for nn=1:81
%         temp2(nn,:) = f(n,:).*Psi_with_theta(nn,:);
%     end
%     out(n,:) = sum(temp2,2);
% end
% figure
% imagesc(temp'-out)   

Q_X = temp + xi*X;