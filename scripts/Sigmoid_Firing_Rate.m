% voltage to fiting rate simoidal function

function f = Sigmoid_Firing_Rate(nu, beta, threshold, Field)

f = nu./(1 + exp(beta*(threshold - Field)));

end