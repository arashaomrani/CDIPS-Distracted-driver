

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 100;  % 20x20 Input Images of Digits
hidden_layer_size = 100;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
load('X_0.1_train.mat');
load('y_0.1_train.mat');
load('X_0.1_cv.mat');
load('y_1_cv.mat');
X=X_train;
y=y_train;
y(y==0)=10;
y_test(y_test==0)=10;
X=im2double(X);
X_test=im2double(X_test);


m = size(X, 1);

fprintf('\nInitializing Neural Network Parameters ...\n')
%%initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);%%initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
load('nn_hl100_lambda1_it400_10x10.mat');
initial_Theta1=Theta1;
initial_Theta2=Theta2;
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 8: Training NN ===================

%
fprintf('\nTraining Neural Network... \n')


options = optimset('MaxIter', 200);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

[h2, pred] = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

[h2_test, pred_test] = predict(Theta1, Theta2, X_test);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
