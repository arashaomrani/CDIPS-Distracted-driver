function [p,a] = predictOneVsAll(all_theta, X)


m = size(X, 1);
num_labels = size(all_theta, 1);


p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


%   

for i=1:num_labels
    Z=X*all_theta(i,:)';
    a(:,i)=sigmoid(Z);


end

[p,In]=max(a, [], 2);
p=In;
% =========================================================================


end
