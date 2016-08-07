function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
W = zeros(L_out, 1 + L_in);
W=rand(L_out, 1 + L_in);

end
