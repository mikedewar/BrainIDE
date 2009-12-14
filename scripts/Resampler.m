function [new_x new_w] = Resampler(x,w)
    
    N = length(w);              % number of particles
    N_inv = 1/N;                % 
    new_x = zeros(N,1);         % initialise
    
    c = cumsum(w);              % cumulative dist of weights
    c = c/max(c);               % normalise
    u1 = rand(1,1)/N;       % with a uniform distribution there is no need to use the sqrt 

    i = 1;
    for j = 1:N       
        u(j) = u1 + N_inv*(j-1);    % this guy should never be  > 1 but it is with the definition in the tutorial paper
        
        while u(j) > c(i)           % find where the cdf is greater than u
            i = i + 1;
        end       
        
        new_x(j) = x(i);              % use the state corresponding to where u > than cdf
        new_w(j) = 1/N;
        i_j(j) = i;
        
    end
end