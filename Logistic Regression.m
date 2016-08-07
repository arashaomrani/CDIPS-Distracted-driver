%

%% Initialization
clear ; close all; clc
%% import data
load('X_0.1_train_rotate.mat');
load('y_0.1_train.mat');
load('X_0.1_cv.mat');
load('y_0.1_cv.mat');
X=X_train;
y=y_train;
y(y==0)=10;
y_test(y_test==0)=10;

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 1228;  % 64x48 Input Images of Digits
num_labels = 10;          % 10 labels, from 0 to 9   
                          % (note that we have mapped "0" to label 10)
X=im2double(X);
X_test=im2double(X_test);
m = size(X, 1);




%% ============ Part 2: Vectorize Logistic Regression ============

%

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

pred_test = predictOneVsAll(all_theta, X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);

