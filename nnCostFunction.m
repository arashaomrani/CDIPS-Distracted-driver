function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 
x2 = [ones(m, 1) X];
Z2=x2*Theta1';
a2=sigmoid(Z2);
m = size(a2, 1);
a3 = [ones(m, 1) a2];
Z3=a3*Theta2';
h=sigmoid(Z3);
m = length(y);
Y=zeros(m,num_labels);
for n=1:m
    Y(n,y(n))=1;
end
alpha=0;
for i=1:m
 for k= 1:num_labels
  alpha1(i,k)=(-1/m)*(Y(i,k)*log(h(i,k))+(1-Y(i,k))*log(1-h(i,k)));
  alpha=alpha+alpha1(i,k);       
 end
end
J=alpha;
Reg=0;
for i= 2: size(Theta1,2)
    for j= 1:size(Theta1,1)
         Reg1(i,j)=(lambda/(2*m))*Theta1(j,i)^2;
         Reg=Reg1(i,j)+Reg;
    end
end
for i= 2: size(Theta2,2)
    for j= 1:size(Theta2,1)
         Reg2(i,j)=(lambda/(2*m))*Theta2(j,i)^2;
         Reg=Reg2(i,j)+Reg;
    end
end
J=J+Reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 
delta3=h-Y;
theta2=Theta2(:,2:size(Theta2,2));
theta1=Theta1(:,2:size(Theta1,2));
delta2=(delta3*theta2).*sigmoidGradient(Z2);

Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));

Delta2=delta3'*a3;
Delta1=delta2'*x2;


for i=1:size(Theta1,1)
    for j=1:size(Theta1,2)
        if j == 1
            D1(i,j)=(1/m)*Delta1(i,j);
        else
            D1(i,j)=(1/m)*Delta1(i,j)+lambda/m*Theta1(i,j);
        end
    end
end
for i=1:size(Theta2,1)
    for j=1:size(Theta2,2)
        if j == 1
            D2(i,j)=(1/m)*Delta2(i,j);
        else
            D2(i,j)=(1/m)*Delta2(i,j)+lambda/m*Theta2(i,j);
        end
    end
end
Theta1_grad=D1;
Theta2_grad=D2;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
