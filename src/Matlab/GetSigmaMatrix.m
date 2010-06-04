% generate Sigma matrix

function X = GetSigmaMatrix(x,P,sqrt_L_plus_lambda)

    L = size(x,1);                      % number of states
    X = zeros(L,2*L+1);             % initialise for speed

    sqrt_P = chol(P,'lower');       % take the matrix sqrt of covariance
    sqrt_L_plus_lambda_times_sqrt_P = sqrt_L_plus_lambda*sqrt_P;

    X(:,1) = x;                                       % this is either the initial or filtered state
    x_mat = repmat(X(:,1),1,L);                 % make x bigger for subtraction and addition
    X(:,2:L+1) = x_mat + sqrt_L_plus_lambda_times_sqrt_P;
    X(:,L+2:2*L+1) = x_mat - sqrt_L_plus_lambda_times_sqrt_P;

end