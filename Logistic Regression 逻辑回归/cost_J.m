function [J, grad] = cost_J(theta)

load('data.txt');

y = data(:, 3);
m = length(y);
X = [data(:, 1:2)];
X = [ones(m,1) X];

h = 1./(1+exp(-X*theta));

J = - (1/m) * (y'*log(h) + (1-y)'*log(1-h));
 
grad = (1/m) * X'*(h-y);