
function [x_1 z_1] = UpdateStateAndObservation(x_0, k , Q, R)

    v = sqrt(Q)*randn(1,1);         % define process disturbance
    n = sqrt(R)*randn(1,1);         % define measure noise
    
    f = x_0/2 + 25*x_0/(1+x_0^2) + 8*cos(1.2*k);    % state evolution equation
    
    x_1 = f + v;                                    % update state
    z_1 = (x_1^2)/20 + n;                           % create measurement
    
end