function [J, grad] = lrCostFunction(theta, X, y, lambda)


% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

 
J = 0;
grad = zeros(size(theta));
h1=X*theta;
h=1./(1+exp(-1*h1));
h2=(h-y).*X(:,1);
grad(1)=(1/m)*sum(h2);

    for j=2:n

       grad(j)=(1/m)*sum((h-y).*X(:,j))+lambda*(theta(j))/m;

    end
    
    alpha=0;
    for i= 1:m
      
        alpha1=(-1/m)*(y(i)*log(h(i))+(1-y(i))*log(1-h(i)));
        alpha=alpha+alpha1;
        
    end
    J=alpha;
      Reg=0;
        for j= 2:n
        Reg1=(lambda/(2*m))*theta(j)^2;
        Reg=Reg1+Reg;
        end
        J=J+Reg;



% =============================================================

grad = grad(:);

end
