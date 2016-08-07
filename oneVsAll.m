function [all_theta] = oneVsAll(X, y, num_labels, lambda)


m = size(X, 1);
n = size(X, 2);


all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


%
%     % Set Initial theta
    %load('theta_int_50_0.1_128_96.mat');
    %initial_theta = all_theta_int';
     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
     options = optimset('GradObj', 'on', 'MaxIter',200);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
   % all_theta(1,:)=ones(n+1,1);
    for c=1:num_labels
    [theta] =  ...
         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                 initial_theta, options);
   all_theta(c,:)=theta;
   end













% =========================================================================


end
